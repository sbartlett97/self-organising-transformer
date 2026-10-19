from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

import hub_utils
from plastic_model import PlasticConfig, PlasticTransformer
from compute import Shape, unit_flops
from plasticity import StructuralController
from train import BatchIterator, add_schedule_args, evaluate_loss, load_tokens, make_scheduler, pick_device, TrainConfig


@dataclass
class Phase2Config(TrainConfig):
    birth_mode: str = "heuristic"  # 'heuristic' | 'random' | 'none' (proposal §11's three conditions)
    seed_fraction: float = 0.33
    growth_interval: int = 250
    birth_limit: int = 2
    birth_limit_end: int | None = None  # None = flat birth_limit all run; else linearly tapers birth_limit -> birth_limit_end over training
    death_limit: int = 2
    minimum_age: int = 1000
    death_relative_margin: float = 0.5
    death_patience: int = 3
    death_criterion: str = "importance"
    birth_score_floor: float = 0.0
    birth_ramp_steps: int = 250
    death_ramp_steps: int = 250
    death_freeze_lr_fraction: float = 0.05
    birth_freeze_lr_fraction: float = 0.05
    birth_score_norm: str = "none"  # 'none' | 'flops': divide head/FFN birth scores by FLOPs relative to one FFN group
    # No birth or death whose ramp would end after this step. None = no step freeze.
    # Use it with --lr_schedule wsd and the same value as --lr_decay_start, so the decay trains a fixed structure.
    structure_freeze_step: int | None = None
    output_dir: str = "./outputs/phase2"


def build_model(cfg: Phase2Config, tokenizer) -> PlasticTransformer:
    config = PlasticConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=cfg.seq_len,
        max_layers=cfg.max_layers,
        d_model=cfg.d_model,
        max_heads=cfg.max_heads,
        max_ffn_dim=cfg.max_ffn_dim,
        ffn_groups=cfg.ffn_groups,
        skip_span=cfg.skip_span,
        dropout=cfg.dropout,
        architecture_mode="sot",
        gate_init_logit=cfg.gate_init_logit,
        seed_fraction=cfg.seed_fraction,
        growth_interval=cfg.growth_interval,
        birth_limit=cfg.birth_limit,
        minimum_age=cfg.minimum_age,
        death_relative_margin=cfg.death_relative_margin,
    )
    return PlasticTransformer(config)


def save_checkpoint(model, tokenizer, cfg: Phase2Config, controller: StructuralController, out_dir: Path, step: int, history: list, growth_log: list, is_final=False):
    path = out_dir / ("final" if is_final else f"checkpoint-{step}")
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    with open(path / "train_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(path / "architecture.json", "w") as f:
        json.dump(model.export_architecture(), f, indent=2)
    with open(path / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(path / "events.json", "w") as f:
        json.dump(controller.events_json(), f, indent=2)
    with open(path / "growth_log.json", "w") as f:
        # Every growth_interval check's full plasticity_event dict (plateau,
        # max_dormant_birth_score, birth_score_floor, state_counts),
        # not just the checks that produced a birth/death -- events.json
        # alone can't show *why* growth stalled (e.g. plateau crashing to 0
        # right as births stop), only *that* it did. Previously only
        # printed to stdout, which is lost the moment a pod is torn down --
        # this makes it part of the checkpoint, pushed to the Hub like
        # everything else.
        json.dump(growth_log, f, indent=2)
    return path


# Config fields a resume may change. Any other difference makes the resume stop.
RESUME_FREE_FIELDS = {"resume_from", "device", "use_trackio", "trackio_space_id", "push_to_hub", "hub_private", "hub_model_id"}


def resume_path(resume_from: str) -> Path:
    """--resume_from is a training_state.pt, or the output_dir that holds one."""
    path = Path(resume_from)
    return path / "training_state.pt" if path.is_dir() else path


def save_training_state(out_dir: Path, state: dict) -> None:
    """One rolling training_state.pt at out_dir: all a resume needs to continue
    the run exactly. A crash during the write leaves the previous file intact."""
    tmp = out_dir / "training_state.pt.tmp"
    torch.save(state, tmp)
    os.replace(tmp, out_dir / "training_state.pt")


def rng_states() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_states(states: dict) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if states["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


def train(cfg: Phase2Config):
    set_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_data, val_data = load_tokens(cfg, tokenizer)

    model = build_model(cfg, tokenizer).to(device)
    birth_cost = None
    if cfg.birth_score_norm == "flops":
        f = unit_flops(Shape.from_config(model.config, seq_len=cfg.seq_len))
        birth_cost = {kind: v / f["ffn_group"] for kind, v in f.items()}
        print(f"birth_cost (FLOPs relative to one FFN group): {birth_cost}")
    controller = StructuralController(
        model,
        growth_interval=cfg.growth_interval,
        birth_limit=cfg.birth_limit,
        death_limit=cfg.death_limit,
        minimum_age=cfg.minimum_age,
        death_relative_margin=cfg.death_relative_margin,
        death_patience=cfg.death_patience,
        death_criterion=cfg.death_criterion,
        birth_score_floor=cfg.birth_score_floor,
        birth_ramp_steps=cfg.birth_ramp_steps,
        death_ramp_steps=cfg.death_ramp_steps,
        death_freeze_lr_fraction=cfg.death_freeze_lr_fraction,
        birth_freeze_lr_fraction=cfg.birth_freeze_lr_fraction,
        birth_cost=birth_cost,
        seed=cfg.seed,
    )
    controller.seed_architecture()
    model.train()
    hub_utils.init_trackio(cfg, phase="phase2", condition=cfg.hub_condition or cfg.birth_mode)

    # Probes carry the birth signal (dL/dg of dormant units) and must stay
    # exactly 0: their own group has lr 0 and no weight decay, and they are
    # out of gradient clipping. The main group must stay first:
    # scheduler.get_last_lr()[0] is the peak-relative LR the freezes read.
    probe_ids = {id(p) for p in model.probe_parameters()}
    main_params = [p for p in model.parameters() if id(p) not in probe_ids]
    optimizer = torch.optim.AdamW(
        [{"params": main_params},
         {"params": model.probe_parameters(), "lr": 0.0, "weight_decay": 0.0}],
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    scheduler = make_scheduler(optimizer, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    # lr_budget[s] = sum of LR / peak LR over training steps s+1..max_steps (step n runs at
    # lr_lambda(n-1)): the peak-LR-equivalent steps a unit born at the check of step s can train for.
    lr_budget = [0.0] * (cfg.max_steps + 1)
    for n in reversed(range(cfg.max_steps)):
        lr_budget[n] = lr_budget[n + 1] + scheduler.lr_lambdas[0](n)

    lagrange = torch.tensor(0.0, device=device)
    history = []
    growth_log = []
    global_step = 0
    # The data position: the seed of the current epoch order, and the batches used from it.
    iter_seed, iter_used = cfg.seed, 0
    if cfg.resume_from:
        path = resume_path(cfg.resume_from)
        state = torch.load(path, map_location=device, weights_only=False)
        changed = {k for k, v in asdict(cfg).items() if k not in RESUME_FREE_FIELDS and state["cfg"].get(k) != v}
        if changed:
            raise ValueError(f"--resume_from: config differs from the saved run in {sorted(changed)}")
        model.load_state_dict(state["model"])
        controller.load_state_dict(state["controller"])
        # Load the optimizer after make_scheduler: its param groups carry the LR of the saved step.
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        lagrange.fill_(state["lagrange"])
        history, growth_log = state["history"], state["growth_log"]
        global_step, iter_seed, iter_used = state["global_step"], state["iter_seed"], state["iter_used"]
        set_rng_states(state["rng"])
        del state
        print(f"Resumed from {path} at step {global_step}")
    train_iter = iter(BatchIterator(train_data, cfg.batch_size, seed=iter_seed, start=iter_used))

    pbar = tqdm(total=cfg.max_steps, desc="train", unit="step", initial=global_step)
    while global_step < cfg.max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_task = 0.0
        accum_cost = 0.0
        for _ in range(cfg.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                iter_seed, iter_used = cfg.seed + global_step + 1, 0
                train_iter = iter(BatchIterator(train_data, cfg.batch_size, seed=iter_seed))
                batch = next(train_iter)
            iter_used += 1
            batch = batch.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16, enabled=(device.type in {"cuda", "cpu"})):
                out = model(batch, labels=batch)
                task_loss = out["loss"]
                cost = model.architecture_cost()
                progress = min(1.0, global_step / max(1, cfg.gate_warmup_steps))
                reg = cfg.arch_reg * progress * model.sparsity_loss()
                constrained = lagrange.detach() * (cost - cfg.target_cost)
                loss = (task_loss + reg + constrained) / cfg.grad_accum
            scaler.scale(loss).backward()
            accum_task += float(task_loss.detach())
            accum_cost += float(cost.detach())

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(main_params, cfg.grad_clip)
        # Gradients are unscaled and still populated here -- this is where the
        # controller reads per-unit gradient pressure for its birth signal.
        # train_loss (not the sparse eval_every val_loss) feeds the plateau
        # signal every step, since growth_interval is typically far more
        # frequent than eval_every.
        controller.record_step(train_loss=accum_task / cfg.grad_accum)
        # Inactive units get zero gradient, but weight decay and Adam momentum
        # would still move them. Keep them bit-identical.
        inactive = model.snapshot_inactive()
        scaler.step(optimizer)
        model.restore_inactive(inactive)
        scaler.update()
        scheduler.step()

        mean_cost = accum_cost / cfg.grad_accum
        with torch.no_grad():
            lagrange += cfg.lagrangian_lr * (mean_cost - cfg.target_cost)
            lagrange.clamp_(0.0, 10.0)
        global_step += 1
        train_loss = accum_task / cfg.grad_accum
        pbar.update(1)
        pbar.set_postfix(loss=f"{train_loss:.3f}", cost=f"{mean_cost:.3f}", lgr=f"{float(lagrange.item()):.3f}")

        if (cfg.birth_mode != "none" and global_step % cfg.growth_interval == 0
                and controller.ramps_fit(cfg.max_steps - global_step)):
            if cfg.birth_limit_end is not None:
                # Linear taper birth_limit -> birth_limit_end over training:
                # aggressive early growth while the dormant pool is full of
                # genuinely good candidates, tapering down as it's more
                # likely only marginal signal is left -- birth_score_floor
                # alone doesn't stop "always fill the quota" once the floor
                # itself is cleared, so this bounds how large that quota is
                # as training progresses. See the 2026-10-08 scale-up
                # entries in RESEARCH_LOG.md for why this was added.
                progress = min(1.0, global_step / cfg.max_steps)
                controller.birth_limit = round(cfg.birth_limit + (cfg.birth_limit_end - cfg.birth_limit) * progress)
            frozen = (cfg.structure_freeze_step is not None
                      and global_step + max(cfg.birth_ramp_steps, cfg.death_ramp_steps) > cfg.structure_freeze_step)
            event = controller.update(global_step, mode=cfg.birth_mode,
                                      lr_fraction=scheduler.get_last_lr()[0] / cfg.learning_rate,
                                      lr_budget_remaining=lr_budget[global_step],
                                      structure_frozen=frozen)
            event["birth_limit_used"] = controller.birth_limit
            pbar.write(json.dumps({"step": global_step, "plasticity_event": event}))
            growth_log.append(event)
            sc = event["state_counts"]
            hub_utils.trackio_log({
                "step": global_step,
                "births": len(event["births"]),
                "deaths": len(event["deaths"]),
                "plateau": event["plateau"],
                "max_dormant_birth_score": event["max_dormant_birth_score"],
                "below_floor_count": event["below_floor_count"],
                "death_eligible_pool": event["death_eligible_pool"],
                "max_death_streak": event["max_death_streak"],
                "birth_limit_used": controller.birth_limit,
                "state_mature": sc.get("mature", 0),
                "state_dormant": sc.get("dormant", 0),
                "state_dead": sc.get("dead", 0),
                "state_protected": sc.get("protected", 0),
            })

        if global_step % cfg.log_every == 0 or global_step == 1:
            record = {
                "step": global_step,
                "train_loss": train_loss,
                "cost": mean_cost,
                "lagrange": float(lagrange.item()),
                "lr": scheduler.get_last_lr()[0],
                "state_counts": controller.state_counts(),
                **model.gate_summary(),
            }
            pbar.write(json.dumps(record))
            history.append(record)
            hub_utils.trackio_log(record)

        if global_step % cfg.eval_every == 0 or global_step == cfg.max_steps:
            val_loss = evaluate_loss(model, val_data, cfg.batch_size, device)
            record = {"step": global_step, "val_loss": val_loss, "val_ppl": math.exp(min(20, val_loss))}
            pbar.write(json.dumps(record))
            history.append(record)
            hub_utils.trackio_log(record)
            save_checkpoint(model, tokenizer, cfg, controller, out_dir, global_step, history, growth_log)

        if global_step % cfg.save_every == 0:
            save_checkpoint(model, tokenizer, cfg, controller, out_dir, global_step, history, growth_log)

        if global_step % cfg.eval_every == 0 or global_step % cfg.save_every == 0:
            save_training_state(out_dir, {
                "cfg": asdict(cfg), "global_step": global_step,
                "model": model.state_dict(), "controller": controller.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "lagrange": float(lagrange.item()),
                "history": history, "growth_log": growth_log,
                "iter_seed": iter_seed, "iter_used": iter_used, "rng": rng_states(),
            })

    pbar.close()
    final_path = save_checkpoint(model, tokenizer, cfg, controller, out_dir, global_step, history, growth_log, is_final=True)
    print("\nDISCOVERED ARCHITECTURE")
    print(json.dumps(model.export_architecture(threshold=0.5), indent=2))
    print("\nFINAL STATE COUNTS")
    print(json.dumps(controller.state_counts(), indent=2))
    print(f"\nTotal births: {sum(1 for e in controller.events if e['event'] == 'birth')}")
    print(f"Total deaths: {sum(1 for e in controller.events if e['event'] == 'death')}")

    if cfg.push_to_hub:
        from huggingface_hub import HfApi
        condition = cfg.hub_condition or cfg.birth_mode
        username = HfApi().whoami()["name"]
        dataset = hub_utils.dataset_tag(cfg.dataset_name)
        repo_id = cfg.hub_model_id or hub_utils.repo_id_for(username, "phase2", dataset, condition, cfg.seed, cfg.hub_target_cost_tag)
        hub_utils.push_checkpoint(final_path, repo_id, cfg.hub_private, "phase2", dataset, condition, cfg.seed)
        hub_utils.ensure_collection(username, "phase2", dataset, repo_id)
    hub_utils.trackio_finish()


def parse_args() -> Phase2Config:
    p = argparse.ArgumentParser()
    p.add_argument("--birth_mode", choices=["heuristic", "random", "none"], default="heuristic")
    p.add_argument("--seed_fraction", type=float, default=0.33)
    p.add_argument("--growth_interval", type=int, default=250)
    p.add_argument("--birth_limit", type=int, default=2)
    p.add_argument("--birth_limit_end", type=int, default=None, help="If set, linearly tapers birth_limit -> this value over training, instead of a flat birth_limit the whole run")
    p.add_argument("--death_limit", type=int, default=2)
    p.add_argument("--minimum_age", type=int, default=1000)
    p.add_argument("--death_relative_margin", type=float, default=0.5, help="A mature unit dies only once its gate falls below this fraction of its own kind's (layer/head/ffn_group) current mature-population median")
    p.add_argument("--death_criterion", choices=["importance", "gate"], default="importance", help="Signal compared against the kind median for death: EMA of Taylor importance |g*dL/dg| (default), or the original gate value")
    p.add_argument("--death_patience", type=int, default=3, help="Consecutive growth checks a unit must stay below death_relative_margin before it actually dies")
    p.add_argument("--birth_score_floor", type=float, default=0.0, help="A dormant candidate is birth-eligible only if its raw score exceeds this -- excludes genuinely zero-signal candidates from filling the birth_limit quota, see plasticity.py's StructuralController docstring")
    p.add_argument("--birth_ramp_steps", type=int, default=250, help="Steps over which a born unit's scale climbs linearly 0 -> 1. 0 = full scale in one step.")
    p.add_argument("--death_ramp_steps", type=int, default=250, help="Steps over which a dying unit's scale falls linearly 1 -> 0. 0 = one-step cut.")
    p.add_argument("--death_freeze_lr_fraction", type=float, default=0.05, help="No death once LR / peak LR falls below this. 0 = never freeze.")
    p.add_argument("--birth_freeze_lr_fraction", type=float, default=0.05, help="No birth once LR / peak LR falls below this. 0 = never freeze.")
    p.add_argument("--birth_score_norm", choices=["none", "flops"], default="none", help="'flops' divides each head and FFN-group birth score by the unit's FLOPs relative to one FFN group, so the shared ranking compares score per unit of compute. Layers are not changed. 'none' = raw scores (attempts 1-9).")
    p.add_argument("--structure_freeze_step", type=int, default=None, help="No birth or death whose ramp would end after this step. Use with --lr_schedule wsd and the same value as --lr_decay_start. None = no step freeze.")
    p.add_argument("--output_dir", default="./outputs/phase2")
    p.add_argument("--resume_from", default=None, help="A training_state.pt, or the output_dir that holds one. The run continues from that step, bit-identical to a run with no stop. Every other flag must match the saved run.")
    p.add_argument("--dataset_name", default="roneneldan/TinyStories")
    p.add_argument("--dataset_config", default=None)
    p.add_argument("--max_train_docs", type=int, default=None, help="Cap on train-split documents to tokenize, for fast diagnostic runs")
    p.add_argument("--tokenizer_name", default="gpt2")
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    add_schedule_args(p)
    p.add_argument("--arch_reg", type=float, default=0.015)
    p.add_argument("--target_cost", type=float, default=0.14)
    p.add_argument("--lagrangian_lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--max_layers", type=int, default=12)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--max_heads", type=int, default=6)
    p.add_argument("--max_ffn_dim", type=int, default=1536)
    p.add_argument("--ffn_groups", type=int, default=64)
    p.add_argument("--skip_span", type=int, default=3)
    p.add_argument("--gate_init_logit", type=float, default=-1.0)
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_model_id", default=None, help="Overrides the auto-generated repo id")
    p.add_argument("--hub_private", action="store_true")
    p.add_argument("--hub_condition", default=None, help="Defaults to --birth_mode (heuristic/random/none)")
    p.add_argument("--hub_target_cost_tag", default=None)
    p.add_argument("--no_trackio", dest="use_trackio", action="store_false", help="Disable live Trackio metrics (on by default)")
    p.add_argument("--trackio_space_id", default=None, help="HF Space to sync live metrics to, e.g. username/sot-tracking -- defaults to <your HF username>/sot-tracking")
    a = p.parse_args()
    return Phase2Config(**vars(a), model_type="sot")


if __name__ == "__main__":
    train(parse_args())
