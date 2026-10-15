"""Check compute.py against real models: the parameter formula against
numel() and the FLOPs formula against torch's FlopCounterMode, for a
CompactTransformer with irregular layer shapes and for the gated supernet
(which always runs its full dense shape).

Run: python tests/compute_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.flop_counter import FlopCounterMode

from compact_model import CompactConfig, CompactTransformer
from compute import Replay, Shape, flops_per_token, param_count, seed_split, training_flops, unit_order
from plastic_model import PlasticConfig, PlasticTransformer
from plasticity import LifecycleState, StructuralController

torch.manual_seed(0)
B, T = 2, 16
cfg = PlasticConfig(vocab_size=97, max_position_embeddings=T, max_layers=4, d_model=32, max_heads=4,
                    max_ffn_dim=48, ffn_groups=6, skip_span=2, dropout=0.0, architecture_mode="sot",
                    seed_fraction=0.4)
s = Shape.from_config(cfg, seq_len=T)


def measured_flops_per_token(model) -> float:
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(ids)
    return fc.get_total_flops() / (B * T)


# ---- test 1: irregular compact structure -- params and FLOPs match the formulas exactly ----
layers = [{"heads": 1, "ffn_groups": 6}, {"heads": 3, "ffn_groups": 1}, {"heads": 4, "ffn_groups": 4}]
specs = [{"original_index": i, "heads": l["heads"], "head_dim": s.head_dim,
          "ffn_dim": l["ffn_groups"] * s.group_size, "skip_from": []} for i, l in enumerate(layers)]
compact = CompactTransformer(CompactConfig(vocab_size=cfg.vocab_size, max_position_embeddings=T,
                                           d_model=cfg.d_model, layers=specs))
got_p = sum(p.numel() for p in compact.parameters())
want_p = param_count(layers, s)["total"]
got_f = measured_flops_per_token(compact)
want_f = flops_per_token(layers, s)["total"]
print(f"test1: params {got_p} vs {want_p}; flops/token {got_f} vs {want_f}")
assert got_p == want_p and got_f == want_f
print("test1: PASS (compact params and FLOPs match the formulas)")

# ---- test 2: the gated supernet runs its full dense shape, whatever its gates ----
sot = PlasticTransformer(cfg)
StructuralController(sot, seed=0).seed_architecture()  # most gates near 0: no FLOP saved
got_f = measured_flops_per_token(sot)
want_f = flops_per_token(s.dense(), s)["total"]
logit_params = sum(p.numel() for n, p in sot.named_parameters() if "logit" in n)
got_p = sum(p.numel() for p in sot.parameters()) - logit_params
want_p = param_count(s.dense(), s)["total"]
print(f"test2: params {got_p} vs {want_p}; flops/token {got_f} vs {want_f}")
assert got_p == want_p and got_f == want_f
print("test2: PASS (executed FLOPs = dense, with gates mostly off)")

# ---- test 3: seed_split reproduces seed_architecture() exactly ----
ctl = StructuralController(PlasticTransformer(cfg), seed=7)
assert list(ctl.units) == unit_order(s), "unit order must match _register_units"
ctl.seed_architecture()
alive = {uid for uid, t in ctl.units.items() if t.state == LifecycleState.MATURE}
assert alive == seed_split(s, 7, cfg.seed_fraction)
print(f"test3: PASS (seed split replays exactly: {len(alive)} of {len(ctl.units)} units)")

# ---- test 4: Replay follows births, death ramps, and drops units in absent layers ----
seed_alive = seed_split(s, 7, cfg.seed_fraction)
dormant_head = next(u for u in unit_order(s) if u[0] == "head" and u not in seed_alive
                    and ("layer", u[1], 0) in seed_alive)
live_layer = next(u for u in unit_order(s) if u[0] == "layer" and u in seed_alive)
events = [{"step": 10, "unit": list(dormant_head), "event": "birth"},
          {"step": 20, "unit": list(live_layer), "event": "death"}]
r = Replay(s, 7, cfg.seed_fraction, events, death_ramp_steps=5)
assert r.states(9)[dormant_head] == "dormant" and r.states(10)[dormant_head] == "alive"
assert r.states(20)[live_layer] == "dying" and r.states(24)[live_layer] == "dying"
assert r.states(25)[live_layer] == "dead"
assert r.counts(20)["dead"] == 1, "a dying unit is already DEAD in the controller's counts"
assert any(l["index"] == live_layer[1] for l in r.structure(24)), "a dying layer still has tensors"
assert all(l["index"] != live_layer[1] for l in r.structure(25)), "a dead layer has none"
print("test4: PASS (births, death ramps and absent layers)")

# ---- test 5: training FLOPs integrate the structure step by step ----
tps = B * T
tf = training_flops(r, max_steps=40, tokens_per_step=tps)
manual = sum(3 * flops_per_token(r.structure(n - 1), s)["total"] * tps for n in range(1, 41))
print(f"test5: integrated {tf['realisable']} vs per-step sum {manual}; fraction {tf['fraction']:.3f}")
assert tf["realisable"] == manual
assert tf["dense"] == 3 * flops_per_token(s.dense(), s)["total"] * tps * 40
print("test5: PASS (piecewise sum equals the per-step sum)")

print("\nALL TESTS PASSED")
