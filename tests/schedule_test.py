"""Check train.make_scheduler: 'cosine' gives the same LR as
get_cosine_schedule_with_warmup at every step (attempts 1-9 and the dense
controls stay reproducible), and 'wsd' has the right shape.

Run: python tests/schedule_test.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import get_cosine_schedule_with_warmup

from train import TrainConfig, make_scheduler


def lrs(cfg, n):
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=cfg.learning_rate)
    sched = make_scheduler(opt, cfg)
    out = []
    for _ in range(n):
        out.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    return out, sched


# ---- test 1: cosine is exactly the old schedule ----
cfg = TrainConfig(max_steps=10000, warmup_steps=500)
new, _ = lrs(cfg, cfg.max_steps)
opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=cfg.learning_rate)
ref_s = get_cosine_schedule_with_warmup(opt, cfg.warmup_steps, cfg.max_steps)
ref = []
for _ in range(cfg.max_steps):
    ref.append(ref_s.get_last_lr()[0])
    opt.step()
    ref_s.step()
assert new == ref, "cosine must match get_cosine_schedule_with_warmup exactly"
print("test1: PASS (cosine == get_cosine_schedule_with_warmup at all 10000 steps)")

# ---- test 2: wsd shape ----
w = TrainConfig(max_steps=10000, warmup_steps=500, lr_schedule="wsd", lr_decay_start=7000)
got, sched = lrs(w, w.max_steps)
f = sched.lr_lambdas[0]
peak = w.learning_rate
assert got[0] == 0.0 and math.isclose(got[250], peak / 2)
assert all(x == peak for x in got[500:7001]), "peak LR from warmup end to the decay start"
assert math.isclose(f(8500), 0.5) and f(10000) == 0.0 and 0 < f(9999) < 1e-6
assert all(a >= b for a, b in zip(got[7000:], got[7001:])), "the decay never rises"
budget_cos = sum(get_cosine_schedule_with_warmup(torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0), 500, 10000).lr_lambdas[0](n) for n in range(10000))
budget_wsd = sum(f(n) for n in range(10000))
print(f"test2: LR budget (peak-LR-equivalent steps): cosine {budget_cos:.0f}, wsd(7000) {budget_wsd:.0f}")
assert abs(budget_cos - 5000) < 5 and abs(budget_wsd - (250 + 6500 + 1500)) < 5
print("test2: PASS (wsd: warmup, flat peak until 7000, cosine to 0 at 10000)")

# ---- test 3: bad settings raise ----
for bad in [dict(lr_schedule="wsd"), dict(lr_schedule="wsd", lr_decay_start=10000),
            dict(lr_schedule="wsd", lr_decay_start=100), dict(lr_schedule="linear")]:
    try:
        lrs(TrainConfig(max_steps=10000, warmup_steps=500, **bad), 1)
        raise AssertionError(f"{bad} accepted")
    except ValueError:
        pass
print("test3: PASS (missing or out-of-range lr_decay_start and unknown schedules raise)")
print("\nALL TESTS PASSED")
