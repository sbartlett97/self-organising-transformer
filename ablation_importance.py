"""Does a unit's gate value measure its importance?

Phase 2's death rule compares gate values. The 2026-10-09 diagnostic showed
mature gates settle in a narrow band (~84-93% of the kind median), so death
never fires. Hypothesis: gate magnitude is not importance at all -- the
sparsity/cost terms push on gate size, and the output projection behind each
gate can rescale freely to compensate (g*W is what the forward pass sees).

This script tests that directly on a trained checkpoint. For a sample of
units it measures the true importance (val loss change when the unit's gate
is forced to exactly 0, on fixed held-out batches) and compares it against
(a) the gate value and (b) a first-order Taylor estimate |g * dL/dg| that
costs one backward pass. It also ablates a few dormant units as a control:
if a gate of ~0.018 still carries real loss, the scale compensation is real.

    python ablation_importance.py sbartlett97/sot-phase2-wikitext103-heuristic-seed42-tc768d013-v4 \
        --output results/ablation_phase2_v4.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from evaluate import load_model
from train import BatchIterator, PackedTokens, pick_device, tokenise_split

ZERO_LOGIT = -1e4  # sigmoid(-1e4) == 0.0 exactly in fp32


def resolve_checkpoint(path_or_repo: str) -> Path:
    if Path(path_or_repo).exists():
        return Path(path_or_repo)
    return Path(snapshot_download(path_or_repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]))


def logit_ref(model, uid):
    kind, li, idx = uid
    block = model.blocks[li]
    if kind == "layer":
        return block.layer_logit, None
    if kind == "head":
        return block.attn.head_logits, idx
    return block.ffn.group_logits, idx


def get_logit(model, uid) -> float:
    param, idx = logit_ref(model, uid)
    return float(param.item()) if idx is None else float(param[idx].item())


def set_logit(model, uid, value: float) -> None:
    param, idx = logit_ref(model, uid)
    with torch.no_grad():
        if idx is None:
            param.fill_(value)
        else:
            param[idx] = value


def all_units(model):
    cfg = model.config
    for li in range(len(model.blocks)):
        yield ("layer", li, 0)
        for hi in range(cfg.max_heads):
            yield ("head", li, hi)
        for gi in range(cfg.ffn_groups):
            yield ("ffn_group", li, gi)


def spread_sample(units, n, rng):
    """Take n units at evenly spaced gate quantiles, so the sample covers
    the full gate range instead of clustering at the median."""
    if len(units) <= n:
        return list(units)
    ordered = sorted(units, key=lambda u: u[1])
    picks = np.linspace(0, len(ordered) - 1, n).round().astype(int)
    return [ordered[i] for i in sorted(set(picks.tolist()))]


def spearman(a, b) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def batch_loss(model, batches) -> float:
    with torch.no_grad():
        return float(np.mean([model(b, labels=b)["loss"].item() for b in batches]))


def taylor_scores(model, batches, units) -> dict:
    """First-order estimate of the loss change from removing a unit:
    removing gate g is a step of -g, so dL ~= -g * dL/dg. Averaged over
    batches as |per-batch estimate| (Molchanov et al. 2019 style), since
    signed per-batch estimates cancel."""
    sums = {uid: 0.0 for uid in units}
    for b in batches:
        model.zero_grad(set_to_none=True)
        model(b, labels=b)["loss"].backward()
        for uid in units:
            param, idx = logit_ref(model, uid)
            grad_logit = float(param.grad.item() if idx is None else param.grad[idx].item())
            g = float(torch.sigmoid(torch.tensor(get_logit(model, uid))))
            # dL/dg = dL/dlogit / (g * (1 - g)), so g * dL/dg = dL/dlogit / (1 - g)
            sums[uid] += abs(grad_logit / max(1.0 - g, 1e-6))
    model.zero_grad(set_to_none=True)
    return {uid: s / len(batches) for uid, s in sums.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="Local checkpoint dir or Hub repo id")
    p.add_argument("--per_kind", type=int, default=20, help="Active units sampled per kind (layer kind is capped by layer count)")
    p.add_argument("--dormant_per_kind", type=int, default=5, help="Dormant control units sampled per kind")
    p.add_argument("--active_logit", type=float, default=-2.5, help="Logit above which a unit counts as active (Phase 2: dormant=-4, dead=-8, born=-1)")
    p.add_argument("--n_batches", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_seed", type=int, default=1234, help="Shuffle seed for the held-out batches; change it to re-test on different data")
    p.add_argument("--output", default="results/ablation_importance.json")
    args = p.parse_args()

    rng = random.Random(args.seed)
    ckpt = resolve_checkpoint(args.checkpoint)
    train_cfg = json.loads((ckpt / "train_config.json").read_text())
    device = pick_device(args.device)

    model = load_model(str(ckpt), torch.float32).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    # Validation split only -- the full train split is not needed and is slow to tokenise.
    seq_len = train_cfg["seq_len"]
    ds_args = (train_cfg["dataset_name"], train_cfg["dataset_config"]) if train_cfg.get("dataset_config") else (train_cfg["dataset_name"],)
    val = load_dataset(*ds_args, split="validation")
    needed_docs = min(len(val), 20000)
    tokens = tokenise_split(val, tokenizer, train_cfg.get("text_column", "text"), needed_docs)
    packed = PackedTokens(tokens, seq_len)
    batches = []
    for b in BatchIterator(packed, args.batch_size, seed=args.batch_seed):
        batches.append(b.to(device))
        if len(batches) >= args.n_batches:
            break
    if len(batches) < args.n_batches:
        raise SystemExit(f"Only {len(batches)} batches available; lower --n_batches or --batch_size")

    # Classify and sample units.
    by_kind_active, by_kind_dormant = {}, {}
    for uid in all_units(model):
        logit = get_logit(model, uid)
        gate = float(torch.sigmoid(torch.tensor(logit)))
        bucket = by_kind_active if logit > args.active_logit else by_kind_dormant
        bucket.setdefault(uid[0], []).append((uid, gate))
    sample = []
    for kind, units in by_kind_active.items():
        sample += [(uid, gate, "active") for uid, gate in spread_sample(units, args.per_kind, rng)]
    for kind, units in by_kind_dormant.items():
        picks = rng.sample(units, min(args.dormant_per_kind, len(units)))
        sample += [(uid, gate, "dormant") for uid, gate in picks]

    base = batch_loss(model, batches)
    print(f"baseline loss {base:.4f} on {args.n_batches}x{args.batch_size}x{seq_len} tokens")
    taylor = taylor_scores(model, batches, [uid for uid, _, _ in sample])

    rows = []
    for i, (uid, gate, status) in enumerate(sample):
        original = get_logit(model, uid)
        set_logit(model, uid, ZERO_LOGIT)
        delta = batch_loss(model, batches) - base
        set_logit(model, uid, original)
        rows.append({"unit": list(uid), "kind": uid[0], "status": status, "gate": gate, "logit": original,
                     "delta_loss": delta, "taylor": taylor[uid]})
        print(f"[{i + 1}/{len(sample)}] {status:7s} {uid} gate={gate:.3f} dloss={delta:+.5f} taylor={taylor[uid]:.5f}")

    # Check the restore loop left the model untouched.
    restored = batch_loss(model, batches)
    assert abs(restored - base) < 1e-4, f"model not restored: {restored} vs {base}"

    summary = {}
    for kind in sorted({r["kind"] for r in rows}):
        act = [r for r in rows if r["kind"] == kind and r["status"] == "active"]
        dor = [r for r in rows if r["kind"] == kind and r["status"] == "dormant"]
        d = [r["delta_loss"] for r in act]
        summary[kind] = {
            "n_active": len(act),
            "n_active_total": len(by_kind_active.get(kind, [])),
            "n_dormant_total": len(by_kind_dormant.get(kind, [])),
            "spearman_delta_vs_gate": spearman([r["gate"] for r in act], d),
            "spearman_delta_vs_taylor": spearman([r["taylor"] for r in act], d),
            "gate_min_over_median": (min(r["gate"] for r in act) / float(np.median([r["gate"] for r in act]))) if act else None,
            "delta_min": min(d, default=None),
            "delta_median": float(np.median(d)) if d else None,
            "delta_max": max(d, default=None),
            "dormant_delta_mean": float(np.mean([r["delta_loss"] for r in dor])) if dor else None,
            "dormant_delta_max": max((r["delta_loss"] for r in dor), default=None),
        }

    out = {"checkpoint": args.checkpoint, "batch_seed": args.batch_seed, "baseline_loss": base, "n_batches": args.n_batches,
           "batch_size": args.batch_size, "seq_len": seq_len, "summary": summary, "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))

    print("\nkind       n   rho(dloss,gate)  rho(dloss,taylor)  dloss min/median/max         dormant dloss mean/max")
    for kind, s in summary.items():
        dm = "n/a" if s["dormant_delta_mean"] is None else f"{s['dormant_delta_mean']:+.5f}/{s['dormant_delta_max']:+.5f}"
        print(f"{kind:10s} {s['n_active']:3d}  {s['spearman_delta_vs_gate']:+.3f}           {s['spearman_delta_vs_taylor']:+.3f}"
              f"             {s['delta_min']:+.4f}/{s['delta_median']:+.4f}/{s['delta_max']:+.4f}   {dm}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
