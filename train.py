from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

import hub_utils
from model import SOTConfig, SelfOrganizingTransformer


@dataclass
class TrainConfig:
    model_type: str = "sot"
    tokenizer_name: str = "gpt2"
    dataset_name: str = "roneneldan/TinyStories"
    dataset_config: str | None = None
    text_column: str = "text"
    max_train_docs: int | None = None
    seq_len: int = 256
    vocab_size: int = 50257
    max_layers: int = 12
    d_model: int = 384
    max_heads: int = 6
    max_ffn_dim: int = 1536
    ffn_groups: int = 64
    skip_span: int = 3
    dropout: float = 0.0
    batch_size: int = 16
    grad_accum: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 10000
    eval_every: int = 500
    save_every: int = 1000
    log_every: int = 25
    grad_clip: float = 1.0
    arch_reg: float = 0.015
    target_cost: float = 0.14
    lagrangian_lr: float = 0.01
    gate_warmup_steps: int = 1000
    gate_init_logit: float = -1.0
    device: str = "auto"
    seed: int = 42
    num_workers: int = 2
    output_dir: str = "./outputs/sot"
    push_to_hub: bool = False
    hub_model_id: str | None = None
    hub_private: bool = False
    hub_condition: str | None = None
    hub_target_cost_tag: str | None = None
    use_trackio: bool = True
    trackio_space_id: str | None = None
    resume_from: str | None = None


class PackedTokens:
    def __init__(self, token_ids: List[int], seq_len: int):
        self.tokens = np.asarray(token_ids, dtype=np.int32)
        self.seq_len = seq_len
        self.n = (len(self.tokens) - 1) // seq_len

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.tokens[start : start + self.seq_len]
        return torch.from_numpy(x.astype(np.int64))


class BatchIterator:
    def __init__(self, packed: PackedTokens, batch_size: int, seed: int = 42):
        self.packed = packed
        self.batch_size = batch_size
        self.rng = random.Random(seed)

    def __iter__(self):
        order = list(range(len(self.packed)))
        self.rng.shuffle(order)
        for i in range(0, len(order), self.batch_size):
            ids = order[i : i + self.batch_size]
            if len(ids) < self.batch_size:
                continue
            yield torch.stack([self.packed[j] for j in ids])


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tokenise_split(ds, tokenizer, text_column: str, max_docs: int | None = None) -> List[int]:
    if max_docs is not None and max_docs < len(ds):
        # Tokenizing is a single-threaded Python loop over the whole split; for a short
        # diagnostic run that only needs a few million tokens, capping the document count
        # avoids paying for a full pass over a multi-million-row corpus every invocation.
        ds = ds.select(range(max_docs))
    out: List[int] = []
    eos = tokenizer.eos_token_id
    for row in tqdm(ds, desc="tokenizing", unit="doc"):
        text = row[text_column]
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        out.extend(ids)
        if eos is not None:
            out.append(eos)
    return out


def load_tokens(cfg: TrainConfig, tokenizer):
    kwargs = {}
    if cfg.dataset_config:
        raw = load_dataset(cfg.dataset_name, cfg.dataset_config)
    else:
        raw = load_dataset(cfg.dataset_name)
    train_tokens = tokenise_split(raw["train"], tokenizer, cfg.text_column, cfg.max_train_docs)
    val_split = "validation" if "validation" in raw else "train"
    if val_split == "train":
        # Deterministic small holdout from the tail if no validation split exists.
        cut = int(len(train_tokens) * 0.99)
        val_tokens = train_tokens[cut:]
        train_tokens = train_tokens[:cut]
    else:
        val_tokens = tokenise_split(raw[val_split], tokenizer, cfg.text_column)
    return PackedTokens(train_tokens, cfg.seq_len), PackedTokens(val_tokens, cfg.seq_len)


def build_model(cfg: TrainConfig, tokenizer) -> SelfOrganizingTransformer:
    config = SOTConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=cfg.seq_len,
        max_layers=cfg.max_layers,
        d_model=cfg.d_model,
        max_heads=cfg.max_heads,
        max_ffn_dim=cfg.max_ffn_dim,
        ffn_groups=cfg.ffn_groups,
        skip_span=cfg.skip_span,
        dropout=cfg.dropout,
        architecture_mode=cfg.model_type,
        gate_init_logit=cfg.gate_init_logit,
    )
    model = SelfOrganizingTransformer(config)
    return model


def evaluate_loss(model, packed: PackedTokens, batch_size: int, device: torch.device, max_batches: int = 50) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for bi, batch in enumerate(BatchIterator(packed, batch_size, seed=1234)):
            batch = batch.to(device)
            out = model(batch, labels=batch)
            losses.append(float(out["loss"].item()))
            if bi + 1 >= max_batches:
                break
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


def save_checkpoint(model, tokenizer, cfg: TrainConfig, out_dir: Path, step: int, history: list, is_final=False, optimizer=None, lagrange=None):
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
    if optimizer is not None:
        # Rolling, single copy at out_dir's root (not per-checkpoint) -- a
        # full AdamW state dict roughly doubles the model's own size, and
        # resuming only ever wants the latest one anyway, not a history of
        # them. Lets --resume_from pick up optimizer momentum/variance and
        # the Lagrangian multiplier, not just weights, avoiding the
        # few-dozen-step wobble of restarting Adam cold.
        torch.save(
            {"optimizer_state_dict": optimizer.state_dict(), "lagrange": float(lagrange.item()) if lagrange is not None else 0.0, "global_step": step},
            out_dir / "training_state.pt",
        )
    return path


def train(cfg: TrainConfig):
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

    history = []
    global_step = 0
    resumed_lagrange = 0.0
    training_state = {}
    if cfg.resume_from:
        # --resume_from points at a specific checkpoint-N (or final)
        # directory; training_state.pt (optimizer/lagrange/step) lives one
        # level up, at the run's own out_dir root -- see save_checkpoint's
        # rolling-single-copy note. Model dims (--max_layers/--d_model/etc.)
        # must match what the checkpoint was saved with; from_pretrained
        # fails loudly on a shape mismatch rather than silently truncating.
        resume_path = Path(cfg.resume_from)
        model = SelfOrganizingTransformer.from_pretrained(resume_path).to(device)
        metrics_path = resume_path / "metrics.json"
        if metrics_path.exists():
            history = json.loads(metrics_path.read_text())
        state_path = resume_path.parent / "training_state.pt"
        if state_path.exists():
            training_state = torch.load(state_path, map_location=device, weights_only=True)
            global_step = training_state.get("global_step", 0)
            resumed_lagrange = training_state.get("lagrange", 0.0)
        else:
            print(f"--resume_from set but no training_state.pt found at {state_path} -- resuming weights only, step/optimizer/lagrange reset to 0")
        print(f"Resumed from {resume_path} at step {global_step}")
    else:
        model = build_model(cfg, tokenizer).to(device)
    model.train()
    hub_utils.init_trackio(cfg, phase="phase1", condition=cfg.hub_condition or cfg.model_type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    if "optimizer_state_dict" in training_state:
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, cfg.max_steps)
    for _ in range(global_step):
        scheduler.step()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Learned Lagrange multiplier for the compute budget.
    lagrange = torch.tensor(resumed_lagrange, device=device)
    micro_step = 0
    train_iter = iter(BatchIterator(train_data, cfg.batch_size, seed=cfg.seed + global_step))

    pbar = tqdm(total=cfg.max_steps, desc="train", unit="step", initial=global_step)
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
                # Warm up architectural pressure so useful representations form before pruning pressure.
                progress = min(1.0, global_step / max(1, cfg.gate_warmup_steps))
                reg = cfg.arch_reg * progress * model.sparsity_loss()
                constrained = lagrange.detach() * (cost - cfg.target_cost)
                loss = (task_loss + reg + constrained) / cfg.grad_accum
            scaler.scale(loss).backward()
            accum_task += float(task_loss.detach())
            accum_cost += float(cost.detach())
            micro_step += 1

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        mean_cost = accum_cost / cfg.grad_accum
        # Projected dual ascent. Restricting lambda prevents runaway regularisation.
        with torch.no_grad():
            lagrange += cfg.lagrangian_lr * (mean_cost - cfg.target_cost)
            lagrange.clamp_(0.0, 10.0)
        global_step += 1
        train_loss = accum_task / cfg.grad_accum
        pbar.update(1)
        pbar.set_postfix(loss=f"{train_loss:.3f}", cost=f"{mean_cost:.3f}", lgr=f"{float(lagrange.item()):.3f}")

        if global_step % cfg.log_every == 0 or global_step == 1:
            record = {
                "step": global_step,
                "train_loss": train_loss,
                "cost": mean_cost,
                "lagrange": float(lagrange.item()),
                "lr": scheduler.get_last_lr()[0],
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
            save_checkpoint(model, tokenizer, cfg, out_dir, global_step, history, optimizer=optimizer, lagrange=lagrange)

        if global_step % cfg.save_every == 0:
            save_checkpoint(model, tokenizer, cfg, out_dir, global_step, history, optimizer=optimizer, lagrange=lagrange)

    pbar.close()
    final_path = save_checkpoint(model, tokenizer, cfg, out_dir, global_step, history, is_final=True, optimizer=optimizer, lagrange=lagrange)
    print("\nDISCOVERED ARCHITECTURE")
    print(json.dumps(model.export_architecture(threshold=0.5), indent=2))
    if cfg.push_to_hub:
        if not cfg.hub_condition:
            raise ValueError("--hub_condition is required with --push_to_hub (e.g. dense, dense_8l, dense_half_ffn, sot)")
        username = HfApi().whoami()["name"]
        dataset = hub_utils.dataset_tag(cfg.dataset_name)
        repo_id = cfg.hub_model_id or hub_utils.repo_id_for(username, "phase1", dataset, cfg.hub_condition, cfg.seed, cfg.hub_target_cost_tag)
        hub_utils.push_checkpoint(final_path, repo_id, cfg.hub_private, "phase1", dataset, cfg.hub_condition, cfg.seed)
        hub_utils.ensure_collection(username, "phase1", dataset, repo_id)
    hub_utils.trackio_finish()


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["sot", "dense"], default="sot")
    p.add_argument("--output_dir", default="./outputs/sot")
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
    p.add_argument("--hub_condition", default=None, help="e.g. dense, dense_8l, dense_half_ffn, sot -- required with --push_to_hub")
    p.add_argument("--hub_target_cost_tag", default=None, help="e.g. 017 for a target_cost=0.17 run, included in the auto-generated repo id")
    p.add_argument("--no_trackio", dest="use_trackio", action="store_false", help="Disable live Trackio metrics (on by default)")
    p.add_argument("--trackio_space_id", default=None, help="HF Space to sync live metrics to, e.g. username/sot-tracking -- defaults to <your HF username>/sot-tracking")
    p.add_argument("--resume_from", default=None, help="Path to a checkpoint-N (or final) directory to resume from -- model dims (--max_layers/--d_model/etc.) must match what it was saved with; optimizer/lagrange/step come from training_state.pt in its parent dir, if present")
    a = p.parse_args()
    return TrainConfig(**vars(a))


if __name__ == "__main__":
    train(parse_args())
