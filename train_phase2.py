from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

import hub_utils
from plastic_model import PlasticConfig, PlasticTransformer
from plasticity import StructuralController
from train import BatchIterator, evaluate_loss, load_tokens, pick_device, TrainConfig


@dataclass
class Phase2Config(TrainConfig):
    birth_mode: str = "heuristic"  # 'heuristic' | 'random' | 'none' (proposal §11's three conditions)
    seed_fraction: float = 0.33
    growth_interval: int = 250
    birth_limit: int = 2
    death_limit: int = 2
    minimum_age: int = 1000
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
    )
    return PlasticTransformer(config)


def save_checkpoint(model, tokenizer, cfg: Phase2Config, controller: StructuralController, out_dir: Path, step: int, history: list, is_final=False):
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
    return path


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
    controller = StructuralController(
        model,
        growth_interval=cfg.growth_interval,
        birth_limit=cfg.birth_limit,
        death_limit=cfg.death_limit,
        minimum_age=cfg.minimum_age,
        seed=cfg.seed,
    )
    controller.seed_architecture()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, cfg.max_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    lagrange = torch.tensor(0.0, device=device)
    history = []
    global_step = 0
    train_iter = iter(BatchIterator(train_data, cfg.batch_size, seed=cfg.seed))

    pbar = tqdm(total=cfg.max_steps, desc="train", unit="step")
    while global_step < cfg.max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_task = 0.0
        accum_cost = 0.0
        for _ in range(cfg.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(BatchIterator(train_data, cfg.batch_size, seed=cfg.seed + global_step + 1))
                batch = next(train_iter)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        # Gradients are unscaled and still populated here -- this is where the
        # controller reads per-unit gradient pressure for its birth signal.
        # train_loss (not the sparse eval_every val_loss) feeds the plateau
        # signal every step, since growth_interval is typically far more
        # frequent than eval_every.
        controller.record_step(train_loss=accum_task / cfg.grad_accum)
        scaler.step(optimizer)
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

        if cfg.birth_mode != "none" and global_step % cfg.growth_interval == 0:
            event = controller.update(global_step, mode=cfg.birth_mode)
            pbar.write(json.dumps({"step": global_step, "plasticity_event": event}))

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

        if global_step % cfg.eval_every == 0 or global_step == cfg.max_steps:
            val_loss = evaluate_loss(model, val_data, cfg.batch_size, device)
            record = {"step": global_step, "val_loss": val_loss, "val_ppl": math.exp(min(20, val_loss))}
            pbar.write(json.dumps(record))
            history.append(record)
            save_checkpoint(model, tokenizer, cfg, controller, out_dir, global_step, history)

        if global_step % cfg.save_every == 0:
            save_checkpoint(model, tokenizer, cfg, controller, out_dir, global_step, history)

    pbar.close()
    final_path = save_checkpoint(model, tokenizer, cfg, controller, out_dir, global_step, history, is_final=True)
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


def parse_args() -> Phase2Config:
    p = argparse.ArgumentParser()
    p.add_argument("--birth_mode", choices=["heuristic", "random", "none"], default="heuristic")
    p.add_argument("--seed_fraction", type=float, default=0.33)
    p.add_argument("--growth_interval", type=int, default=250)
    p.add_argument("--birth_limit", type=int, default=2)
    p.add_argument("--death_limit", type=int, default=2)
    p.add_argument("--minimum_age", type=int, default=1000)
    p.add_argument("--output_dir", default="./outputs/phase2")
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
    a = p.parse_args()
    return Phase2Config(**vars(a), model_type="sot")


if __name__ == "__main__":
    train(parse_args())
