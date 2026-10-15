"""Val ppl of the structure that compute.py counts, on the same model.

The supernet's val ppl includes dormant units (gate 0.014-0.05) and dead
units (gate ~3e-4), which still change the output (Bugs.md: "Dormant units
are not inert"). A physical model of the present structure has neither. This
script forces every unit that is not present fully off (logit -30, gate
~1e-13) and reports val ppl both ways, beside the FLOPs and params of that
structure.

Not build_from_sot: it selects by gate >= 0.5 (every attempt-7 layer gate is
below 0.49) and snaps head counts to divisors of d_model.

Example:
  python structure_eval.py sbartlett97/sot-phase2-wikitext103-heuristic-seed42-tc768d013-v7-lrfreeze \
      --events results/phase2_v7_lrfreeze/events.json --death_ramp_steps 250
"""
from __future__ import annotations

import argparse
import json
import math

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from ablation_importance import batch_loss, resolve_checkpoint, set_logit
from compute import Replay, Shape, flops_per_token, param_count
from evaluate import load_model
from train import BatchIterator, PackedTokens, pick_device, tokenise_split

OFF_LOGIT = -30.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="Local path or Hub repo of a Phase 2 checkpoint")
    p.add_argument("--events", required=True, help="events.json of the run")
    p.add_argument("--death_ramp_steps", type=int, required=True)
    p.add_argument("--step", type=int, default=None, help="Structure after this growth check (default: max_steps)")
    p.add_argument("--batches", type=int, default=48)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out", default=None, help="Write the result as JSON here")
    a = p.parse_args()

    ckpt = resolve_checkpoint(a.checkpoint)
    cfg = json.loads((ckpt / "train_config.json").read_text())
    step = a.step if a.step is not None else cfg["max_steps"]
    dev = pick_device("auto")
    model = load_model(str(ckpt), torch.float32).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(ckpt)

    s = Shape.from_config(model.config, seq_len=cfg["seq_len"])
    replay = Replay(s, cfg["seed"], cfg["seed_fraction"], json.load(open(a.events)), a.death_ramp_steps)
    states = replay.states(step)
    structure = replay.structure(step)

    val = load_dataset(cfg["dataset_name"], cfg["dataset_config"], split="validation")
    packed = PackedTokens(tokenise_split(val, tok, cfg.get("text_column", "text"), len(val)), cfg["seq_len"])
    batches = []
    for b in BatchIterator(packed, a.batch_size, seed=1234):
        batches.append(b.to(dev))
        if len(batches) >= a.batches:
            break

    trained = batch_loss(model, batches)
    present_layers = {("layer", l["index"], 0) for l in structure}
    off = [uid for uid, st in states.items()
           if st not in ("alive", "dying") or (uid[0] != "layer" and ("layer", uid[1], 0) not in present_layers)]
    for uid in off:
        set_logit(model, uid, OFF_LOGIT)
    physical = batch_loss(model, batches)

    f, fd = flops_per_token(structure, s), flops_per_token(s.dense(), s)
    pc, pd = param_count(structure, s), param_count(s.dense(), s)
    result = {
        "checkpoint": a.checkpoint, "step": step, "eval_tokens": a.batches * a.batch_size * cfg["seq_len"],
        "units_forced_off": len(off),
        "trained_supernet": {"loss": trained, "ppl": math.exp(trained)},
        "present_structure_only": {"loss": physical, "ppl": math.exp(physical)},
        "structure": [{k: l[k] for k in ("index", "heads", "ffn_groups")} for l in structure],
        "flops_per_token": f["total"], "flops_fraction_of_dense": f["total"] / fd["total"],
        "params": pc["total"], "params_fraction_of_dense": pc["total"] / pd["total"],
    }
    print(json.dumps(result, indent=2))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
