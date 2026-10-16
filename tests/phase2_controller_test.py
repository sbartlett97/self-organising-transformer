"""
Zero-cost local correctness test for Phase 2's StructuralController, before
any cloud spend -- mirrors the gate-fold exactness test pattern used earlier
in this project. Verifies:
  1. seed_architecture() produces roughly the requested DORMANT/MATURE split.
  2. A DORMANT candidate with a high birth score actually gets born
     (heuristic mode): state -> PROTECTED, logit reset, event logged.
  3. A PROTECTED unit is immune to death scoring within minimum_age, even if
     its current gate value would otherwise read as "should die".
  4. [UPDATED for death_relative_margin] A mature unit whose gate sits well
     below its own kind's current mature-population median, sustained for
     death_patience consecutive update() calls, gets hard-clamped to DEAD --
     while a healthier peer of the same kind survives.
  5. Random-mode birth selects from the dormant pool and respects birth_limit.
  6. events_json() is well-formed and JSON-serializable.
  7. [UPDATED] death_limit caps simultaneous deaths even when more units are
     eligible under the new relative-median criterion.
  8. High plateau must NOT indiscriminately push every dormant candidate
     over threshold -- only ones that already show gradient pressure should
     benefit (the Stage 2 churn-loop fix).
  9. [NEW] The "everyone crashes together" edge case: if every mature unit
     of a kind is forced equally bad, nobody reads as relatively worse than
     the pack (median == everyone), so nobody dies -- a known, accepted
     limitation of a purely relative criterion, verified so it's a
     documented trade-off, not a silent surprise.
  10. [NEW, birth_score_floor] A dormant candidate with a modest score --
     too small to have cleared the old absolute birth_score_threshold --
     still gets born when plateau=0 (loss actively improving, no plateau
     boost at all), as long as it's the top-ranked dormant candidate. This
     is the actual fix: growth timing no longer depends on loss having
     happened to go flat.
  11. [NEW] record_step() must not let a single non-finite gradient
     permanently poison a unit's gradient_ema -- an EMA can never recover
     from one NaN/inf update (ema_decay*NaN + ... is NaN forever after).
  12. [NEW] A NaN-poisoned dormant candidate's score must not crowd a
     genuinely good candidate out of the top-birth_limit selection --
     NaN comparisons are always False, so leaving a NaN in the list before
     sort() breaks its total-order assumption. Verified directly: with one
     candidate forced to NaN and birth_limit=1, the real (finite, positive)
     candidate still gets born -- observed directly in a real run where
     growth froze for 1250+ steps once a NaN appeared.
"""
import json
import math
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from plastic_model import PlasticConfig, PlasticTransformer
from plasticity import LifecycleState, StructuralController

torch.manual_seed(0)

cfg = PlasticConfig(
    vocab_size=100,
    max_position_embeddings=16,
    max_layers=4,
    d_model=16,
    max_heads=4,
    max_ffn_dim=16,
    ffn_groups=4,
    skip_span=0,
    dropout=0.0,
    architecture_mode="sot",
    seed_fraction=0.6,  # higher than Stage 2's 0.33 -- need several same-kind mature units for tests 4/7/9
    growth_interval=5,
    birth_limit=2,
    minimum_age=10,
    death_relative_margin=0.5,
    death_patience=2,
)
model = PlasticTransformer(cfg)
model.train()

controller = StructuralController(
    model,
    growth_interval=cfg.growth_interval,
    birth_limit=cfg.birth_limit,
    minimum_age=cfg.minimum_age,
    death_relative_margin=cfg.death_relative_margin,
    death_patience=cfg.death_patience,
    death_criterion="gate",  # tests 1-13 drive death through gate values (the original rule)
    birth_ramp_steps=0,  # tests 1-13 expect the original one-step birth
    death_ramp_steps=0,  # ... and the original one-step death
    seed=0,
)

# ---- test 1: seed split ----
controller.seed_architecture()
counts = controller.state_counts()
n_units = len(controller.units)
print(f"test1: n_units={n_units} counts={counts}")
mature_frac = counts.get("mature", 0) / n_units
assert 0.4 < mature_frac < 0.8, f"seed fraction out of expected range: {mature_frac}"
assert counts.get("dead", 0) == 0 and counts.get("protected", 0) == 0
print("test1: PASS (seed split within expected range, no premature protected/dead)")


def in_live_layer(ctl, uid):
    """Death and birth skip heads/FFN groups inside a DORMANT or DEAD layer
    (their output is exactly zero), so tests that need eligible units pick
    them from live layers."""
    return not ctl._in_inactive_layer(uid)


def all_layers_live(ctl):
    """Make every layer MATURE (scale 1) and every protected unit mature. The
    4-layer test model leaves too few live layers after the seed split for
    tests that need several mature units per kind."""
    for li in range(cfg.max_layers):
        lu = ("layer", li, 0)
        ctl.units[lu].state = LifecycleState.MATURE
        ctl._set_scale(lu, 1.0)
        ctl._set_probe(lu, False)
    for t in ctl.units.values():
        if t.state == LifecycleState.PROTECTED:
            t.state = LifecycleState.MATURE
            t.ramp_remaining = 0

# ---- drive a few real forward/backward passes so gradients exist ----
opt = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randint(0, cfg.vocab_size, (2, 8))


def step_once():
    opt.zero_grad(set_to_none=True)
    out = model(x, labels=x)
    out["loss"].backward()
    controller.record_step(train_loss=float(out["loss"].item()))
    opt.step()


for _ in range(3):
    step_once()

# ---- test 2: force a dormant candidate's score far above its peers, verify birth ----
dormant_uid = next(uid for uid, t in controller.units.items() if t.state == LifecycleState.DORMANT)
controller.units[dormant_uid].gradient_ema = 10.0
controller.units[dormant_uid].contribution_ema = 10.0
result = controller.update(step=5, mode="heuristic")
print(f"test2: update result={result}")
assert dormant_uid in result["births"], f"expected {dormant_uid} to be born, got {result['births']}"
assert controller.units[dormant_uid].state == LifecycleState.PROTECTED
got_logit = controller._get_logit(dormant_uid)
assert abs(got_logit - controller.birth_init_logit) < 1e-6, got_logit
assert any(e["unit"] == dormant_uid and e["event"] == "birth" for e in controller.events)
print("test2: PASS (high-score dormant candidate born, logit reset, event logged)")

# ---- test 3: the newly protected unit must survive a death pass even with a bad gate value ----
controller._set_logit(dormant_uid, -8.0)  # simulate a gate value that would otherwise scream "kill me"
deaths = controller._death_pass(step=6)
print(f"test3: deaths after forcing bad gate on protected unit={deaths}")
assert dormant_uid not in deaths, "protection window was not respected"
assert controller.units[dormant_uid].state == LifecycleState.PROTECTED
print("test3: PASS (protected unit immune to death scoring)")

all_layers_live(controller)  # tests 4-9 need several mature units per kind in live layers

# ---- test 4: a mature unit well below its kind's median, sustained, dies -- a healthy peer survives ----
# pick a kind with >=3 mature units so there's a real median to be below/above
from collections import defaultdict

mature_by_kind = defaultdict(list)
for uid, t in controller.units.items():
    if t.state == LifecycleState.MATURE and in_live_layer(controller, uid):
        mature_by_kind[uid[0]].append(uid)
kind, uids = max(mature_by_kind.items(), key=lambda kv: len(kv[1]))
assert len(uids) >= 3, f"need >=3 mature units of one kind for test4/7/9, got {len(uids)} of kind {kind!r}"

worst_uid, healthy_uid = uids[0], uids[1]
for uid in uids:
    controller.units[uid].steps_since_birth = cfg.minimum_age + 1  # past protection
controller._set_logit(worst_uid, -8.0)  # gate ~0.0003 -- far below any real median
controller._set_logit(healthy_uid, 2.0)  # gate ~0.88 -- clearly healthy
for uid in uids[2:]:
    controller._set_logit(uid, 0.5)  # gate ~0.62 -- unremarkable, establishes a real median

d1 = controller._death_pass(step=10)
assert worst_uid not in d1, "should need death_patience consecutive passes, not just one"
d2 = controller._death_pass(step=11)
print(f"test4: death passes -> d1={d1} d2={d2}")
assert worst_uid in d2, f"expected {worst_uid} dead after {cfg.death_patience} sustained passes below its kind's median"
assert controller.units[worst_uid].state == LifecycleState.DEAD
assert controller._get_scale(worst_uid) == 0.0, "a dead unit's scale is exactly 0"
assert controller.units[healthy_uid].state == LifecycleState.MATURE, "healthy peer should not have died"
print("test4: PASS (unit well below its kind's median dies after death_patience; healthy peer survives)")

# ---- test 5: random-mode birth respects birth_limit and only touches dormant units ----
pre_dormant = {uid for uid, t in controller.units.items() if t.state == LifecycleState.DORMANT}
result_r = controller.update(step=20, mode="random")
print(f"test5: random births={result_r['births']}")
assert len(result_r["births"]) <= cfg.birth_limit
assert all(uid in pre_dormant for uid in result_r["births"])
print("test5: PASS (random mode respects birth_limit, only selects from dormant pool)")

# ---- test 6: events_json is well-formed and serializable ----
events = controller.events_json()
json.dumps(events)  # raises if not serializable
assert all(set(e.keys()) == {"step", "unit", "event", "score"} for e in events)
assert all(isinstance(e["unit"], list) and len(e["unit"]) == 3 for e in events)
print(f"test6: PASS ({len(events)} events, all well-formed and JSON-serializable)")

# ---- test 7: death_limit caps simultaneous deaths even when more units are eligible ----
# A relative-median criterion needs the "low" group to stay a MINORITY of
# the kind's mature population, or the median itself gets dragged down with
# them (exactly what test9 checks deliberately) -- so leave the majority of
# this kind's mature units untouched at their natural value, and force only
# a few (more than death_limit, fewer than half the population) low.
mature_by_kind = defaultdict(list)
for uid, t in controller.units.items():
    if t.state == LifecycleState.MATURE and in_live_layer(controller, uid):
        mature_by_kind[uid[0]].append(uid)
kind2, uids2 = max(mature_by_kind.items(), key=lambda kv: len(kv[1]))
n_low = controller.death_limit + 1
assert len(uids2) >= 2 * n_low, f"need the low group to stay a clear minority for this test, got {len(uids2)} mature units of kind {kind2!r}"
low_uids = uids2[:n_low]
for uid in low_uids:
    controller.units[uid].steps_since_birth = cfg.minimum_age + 1
    controller._set_logit(uid, -8.0)
for uid in uids2[n_low:]:
    controller.units[uid].steps_since_birth = cfg.minimum_age + 1
controller._death_pass(step=30)  # streak=1 for everyone below the floor
deaths7 = controller._death_pass(step=31)  # streak=2 -> eligible, but capped
print(f"test7: {len(low_uids)} units eligible, death_limit={controller.death_limit}, actual deaths={len(deaths7)}")
assert len(deaths7) == controller.death_limit, f"expected exactly {controller.death_limit} deaths, got {len(deaths7)}"
still_mature = [uid for uid in low_uids if controller.units[uid].state == LifecycleState.MATURE]
assert len(still_mature) == len(low_uids) - controller.death_limit
print("test7: PASS (death_limit caps simultaneous deaths; remaining eligible units survive to the next check)")

# ---- test 8: high plateau must NOT indiscriminately push every dormant
# candidate over threshold -- only ones that already show gradient pressure
# should benefit (the Stage 2 churn-loop fix: plateau modulates per-candidate
# signal rather than broadcasting additively to everyone) ----
controller._train_loss_ema = None  # prevent _plateau_score() from appending a fresh (non-flat) sample
controller._loss_history = [1.0] * 10  # perfectly flat -> plateau saturates near 1.0
plateau = controller._plateau_score()
assert plateau > 0.9, f"expected saturated plateau, got {plateau}"

dormant_uids = [uid for uid, t in controller.units.items() if t.state == LifecycleState.DORMANT and in_live_layer(controller, uid)]
uid_with_pressure, uid_without_pressure = dormant_uids[0], dormant_uids[1]
controller.units[uid_with_pressure].gradient_ema = 0.02
controller.units[uid_with_pressure].contribution_ema = 0.0
controller.units[uid_without_pressure].gradient_ema = 0.0
controller.units[uid_without_pressure].contribution_ema = 0.0
for uid in dormant_uids[2:]:
    controller.units[uid].gradient_ema = 0.0
    controller.units[uid].contribution_ema = 0.0

births, _ = controller._birth_pass(step=100, mode="heuristic")
print(f"test8: plateau={plateau:.3f}, births={births}")
assert uid_without_pressure not in births, (
    f"zero-gradient candidate {uid_without_pressure} was born under saturated plateau -- "
    "plateau is still broadcasting indiscriminately, fix did not take effect"
)
print("test8: PASS (saturated plateau does not sweep up a candidate with zero gradient pressure)")

# ---- test 9: "everyone crashes together" -- a known, accepted limitation of a
# purely relative criterion. If every mature unit of a kind is forced equally
# bad, the kind's own median collapses with them, so none reads as relatively
# worse than the pack -- confirmed here, not just assumed. ----
mature_by_kind = defaultdict(list)
for uid, t in controller.units.items():
    if t.state == LifecycleState.MATURE:
        mature_by_kind[uid[0]].append(uid)
kind3, uids3 = max(mature_by_kind.items(), key=lambda kv: len(kv[1]))
assert len(uids3) >= 2
for uid in uids3:
    controller.units[uid].steps_since_birth = cfg.minimum_age + 1
    controller._set_logit(uid, -8.0)
controller._death_pass(step=40)
deaths9 = controller._death_pass(step=41)
print(f"test9: deaths when the whole '{kind3}' mature population crashes together: {deaths9}")
assert deaths9 == [], "expected nobody to die when the entire comparison population is equally bad"
print("test9: PASS (confirmed: a uniform population-wide crash is invisible to a purely relative criterion)")

# ---- test 10: the actual fix -- a modest-score candidate (below the OLD
# absolute threshold of 0.01) still gets born when plateau=0, as long as
# it's the top-ranked dormant candidate. This is the exact scenario that
# was broken before: growth used to require a plateau boost to clear a
# fixed bar; now it only needs to be comparatively best. ----
controller._loss_history = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05]  # steeply declining -> plateau=0
controller._train_loss_ema = None
plateau10 = controller._plateau_score()
assert plateau10 == 0.0, f"expected plateau=0 for steeply-declining loss, got {plateau10}"

dormant_uids_10 = [uid for uid, t in controller.units.items() if t.state == LifecycleState.DORMANT]
assert len(dormant_uids_10) >= 1, "need at least one dormant unit left for test10"
best_uid = dormant_uids_10[0]
controller.units[best_uid].gradient_ema = 0.01  # modest -- w_g=0.5 -> b=0.005, below the OLD 0.01 threshold
controller.units[best_uid].contribution_ema = 0.0
for uid in dormant_uids_10[1:]:
    controller.units[uid].gradient_ema = 0.0
    controller.units[uid].contribution_ema = 0.0

births10, plateau_used = controller._birth_pass(step=200, mode="heuristic")
print(f"test10: plateau={plateau_used}, best_uid score~0.005 (below old 0.01 threshold), births={births10}")
assert plateau_used == 0.0
assert best_uid in births10, (
    f"expected {best_uid} (top-ranked, modest score, plateau=0) to be born -- "
    "growth is still gated by an absolute bar, the fix did not take effect"
)
print("test10: PASS (modest top-ranked candidate born under plateau=0 -- growth no longer depends on loss-plateau timing)")

# ---- test 11: record_step() must not let a non-finite gradient permanently
# poison a unit's gradient_ema ----
probe_uid = next(uid for uid, t in controller.units.items())
controller.units[probe_uid].gradient_ema = 0.5  # a known-good prior value
param, idx = controller._logit_param(probe_uid)
if idx is None:
    param.grad = torch.tensor(float("nan"))
else:
    param.grad = torch.full_like(param, 0.0)
    param.grad[idx] = float("nan")
controller.record_step(train_loss=1.0)
print(f"test11: gradient_ema after NaN grad injected = {controller.units[probe_uid].gradient_ema}")
assert controller.units[probe_uid].gradient_ema == 0.5, (
    f"expected gradient_ema to stay at its prior value (0.5), got {controller.units[probe_uid].gradient_ema} -- "
    "a non-finite gradient was allowed to poison the EMA"
)
print("test11: PASS (non-finite gradient did not update gradient_ema, prior value preserved)")

# ---- test 12: a NaN-poisoned dormant candidate must not crowd out a
# genuinely good one from the top-birth_limit selection ----
dormant_uids_12 = [uid for uid, t in controller.units.items() if t.state == LifecycleState.DORMANT]
assert len(dormant_uids_12) >= 2, "need at least 2 dormant units for test12"
nan_uid, good_uid = dormant_uids_12[0], dormant_uids_12[1]
controller.units[nan_uid].gradient_ema = float("nan")
controller.units[nan_uid].contribution_ema = 0.0
controller.units[good_uid].gradient_ema = 0.02
controller.units[good_uid].contribution_ema = 0.0
for uid in dormant_uids_12[2:]:
    controller.units[uid].gradient_ema = 0.0
    controller.units[uid].contribution_ema = 0.0

births12, _ = controller._birth_pass(step=300, mode="heuristic")
print(f"test12: births={births12}, nan_candidates={controller._last_nan_candidates}")
assert nan_uid not in births12, "the NaN-poisoned candidate should never be born"
assert good_uid in births12, (
    f"expected the genuinely good candidate {good_uid} to be born despite a NaN-poisoned "
    f"peer in the dormant pool -- sort() instability may be crowding it out"
)
assert controller._last_nan_candidates >= 1
print("test12: PASS (NaN-poisoned candidate excluded, genuinely good candidate still born)")

# ---- test 13: [NEW, death-streak-accumulation diagnostic] below_floor_count
# and max_death_streak must reflect LIVE state this check, not a historical
# peak -- otherwise a unit that dipped below its kind's relative floor once
# and recovered would look identical in growth_log.json to one still
# actively accumulating toward death_patience, defeating the whole point of
# adding these counters (telling apart "eligibility never fires" from
# "streak keeps resetting before patience completes"). ----
mature_by_kind = defaultdict(list)
for uid, t in controller.units.items():
    if t.state == LifecycleState.MATURE:
        mature_by_kind[uid[0]].append(uid)
kind13, uids13 = max(mature_by_kind.items(), key=lambda kv: len(kv[1]))
assert len(uids13) >= 2, f"need >=2 mature units of one kind for test13, got {len(uids13)} of kind {kind13!r}"
dip_uid, steady_uid = uids13[0], uids13[1]
for uid in uids13:
    controller.units[uid].steps_since_birth = cfg.minimum_age + 1
    controller._death_streak.pop(uid, None)  # deterministic start, independent of earlier tests' leftovers
controller._set_logit(dip_uid, -8.0)  # forces it below its kind's relative floor
for uid in uids13[1:]:
    controller._set_logit(uid, 0.5)  # unremarkable, establishes a real median

result13a = controller.update(step=400, mode="heuristic")
print(f"test13a: below_floor_count={result13a['below_floor_count']} max_death_streak={result13a['max_death_streak']} pool={result13a['death_eligible_pool']}")
assert result13a["below_floor_count"] >= 1, "dip_uid should register as below its kind's relative floor"
assert result13a["max_death_streak"] == 1, f"expected streak=1 after one check below floor, got {result13a['max_death_streak']}"
assert controller.units[dip_uid].state == LifecycleState.MATURE, "death_patience=2 -- should not die after just 1 check"

controller._set_logit(dip_uid, 0.5)  # recovers above the floor
result13b = controller.update(step=401, mode="heuristic")
print(f"test13b: below_floor_count={result13b['below_floor_count']} max_death_streak={result13b['max_death_streak']}")
assert controller._death_streak[dip_uid] == 0, "recovered unit's streak should reset to 0, not keep climbing"
assert result13b["max_death_streak"] == 0, (
    f"expected max_death_streak to reflect current state (0, nobody below floor after recovery), got "
    f"{result13b['max_death_streak']} -- looks like it's reporting a historical peak instead of live state"
)
print("test13: PASS (below_floor_count/max_death_streak track live streak state, not a running historical peak)")

# ===== death_criterion="importance" (tests 14-18) =====
# A fresh model + controller, so nothing above leaks in. Death now ranks units
# by an EMA of the Taylor importance |g * dL/dg| against their kind's median.
torch.manual_seed(1)
imp_model = PlasticTransformer(cfg)
imp_model.train()
imp = StructuralController(
    imp_model,
    growth_interval=cfg.growth_interval,
    birth_limit=cfg.birth_limit,
    minimum_age=cfg.minimum_age,
    death_relative_margin=cfg.death_relative_margin,
    death_patience=cfg.death_patience,
    death_ramp_steps=0,
    seed=0,
)
assert imp.death_criterion == "importance", "importance must be the default criterion"
imp.seed_architecture()
all_layers_live(imp)

# ---- test 14: importance_ema follows |dL/dlogit / (1 - g)| from one real step ----
imp_opt = torch.optim.SGD(imp_model.parameters(), lr=0.01)
imp_opt.zero_grad(set_to_none=True)
imp_model(x, labels=x)["loss"].backward()
probe = ("head", 0, 0)
param, idx = imp._logit_param(probe)
grad = float(param.grad[idx].item())
gate = float(torch.sigmoid(param[idx]).item())
imp.record_step(train_loss=1.0)
expected = (1 - imp.ema_decay) * abs(grad / (1 - gate))
got = imp.units[probe].importance_ema
print(f"test14: grad={grad:.3e} gate={gate:.3f} importance_ema={got:.3e} expected={expected:.3e}")
assert abs(got - expected) <= 1e-9 + 1e-6 * abs(expected), "importance_ema does not match the Taylor formula"
print("test14: PASS (importance_ema = EMA of |dL/dlogit / (1-g)|)")


def imp_mature_kind(min_units):
    by_kind = defaultdict(list)
    for uid, t in imp.units.items():
        if t.state == LifecycleState.MATURE:
            by_kind[uid[0]].append(uid)
    kind, uids = max(by_kind.items(), key=lambda kv: len(kv[1]))
    assert len(uids) >= min_units, f"need >={min_units} mature units of one kind, got {len(uids)} of {kind!r}"
    for uid in uids:
        imp.units[uid].steps_since_birth = cfg.minimum_age + 1
        imp._death_streak.pop(uid, None)
    return uids


# ---- test 15: low importance dies even with a HIGH gate; a LOW-gate but important peer survives ----
uids = imp_mature_kind(4)
weak, low_gate_important = uids[0], uids[1]
for uid in uids:
    imp.units[uid].importance_ema = 1.0
    imp._set_logit(uid, 0.0)
imp.units[weak].importance_ema = 0.01  # far below 0.5 x median(1.0)
imp._set_logit(weak, 2.0)  # gate ~0.88 -- would look healthiest under the gate rule
imp.units[low_gate_important].importance_ema = 5.0
imp._set_logit(low_gate_important, -3.0)  # gate ~0.05 -- would look weakest under the gate rule
d1 = imp._death_pass(step=500)
assert weak not in d1, "should need death_patience passes, not one"
d2 = imp._death_pass(step=501)
print(f"test15: d1={d1} d2={d2}")
assert weak in d2, "unimportant unit should die regardless of its high gate"
assert imp.units[weak].state == LifecycleState.DEAD
assert imp.units[low_gate_important].state == LifecycleState.MATURE, "important unit must survive despite its low gate"
print("test15: PASS (death follows importance, not gate value)")

# ---- test 16: a gate-only dip does NOT trigger death under the importance criterion ----
uids = imp_mature_kind(3)
for uid in uids:
    imp.units[uid].importance_ema = 1.0
    imp._set_logit(uid, 0.5)
dip = uids[0]
imp._set_logit(dip, -8.0)  # the test-4 trigger under the old rule
for step in range(600, 600 + cfg.death_patience + 2):
    assert dip not in imp._death_pass(step=step)
print(f"test16: dip streak={imp._death_streak[dip]}")
assert imp._death_streak[dip] == 0 and imp.units[dip].state == LifecycleState.MATURE
print("test16: PASS (a low gate alone no longer makes a unit eligible)")

# ---- test 17: update() reports the criterion and per-kind importance spread ----
res17 = imp.update(step=700, mode="heuristic")
print(f"test17: criterion={res17['death_criterion']} importance_spread kinds={sorted(res17['importance_spread'])}")
assert res17["death_criterion"] == "importance"
assert res17["importance_spread"] and all({"min", "median", "max"} <= set(v) for v in res17["importance_spread"].values())
json.dumps(res17, default=list)
print("test17: PASS (diagnostics report criterion and importance spread)")

# ---- test 18: an unknown criterion is rejected ----
try:
    StructuralController(imp_model, death_criterion="magnitude")
    raise AssertionError("expected ValueError for unknown death_criterion")
except ValueError:
    print("test18: PASS (unknown death_criterion rejected)")


# ===== birth_ramp_steps (tests 19-21) =====
# A fresh model with no gradients: record_step then only advances the ramp.
torch.manual_seed(2)
ramp_model = PlasticTransformer(cfg)
ramp = StructuralController(ramp_model, birth_limit=1, minimum_age=cfg.minimum_age, birth_ramp_steps=10, seed=0)
assert ramp.birth_ramp_steps == 10
ramp.seed_architecture()
all_layers_live(ramp)
born = next(uid for uid, t in ramp.units.items() if t.state == LifecycleState.DORMANT and uid[0] != "layer")
ramp.units[born].gradient_ema = 10.0

# ---- test 19: birth sets the logit to birth_init_logit; the scale climbs 0 -> 1 linearly ----
res19 = ramp.update(step=5, mode="heuristic")
assert res19["births"] == [born], res19["births"]
assert ramp.units[born].state == LifecycleState.PROTECTED
assert ramp._get_scale(born) == 0.0, "birth must start at scale 0 (no jump in the output)"
assert ramp._get_logit(born) == ramp.birth_init_logit
assert res19["ramping"] == 1
for _ in range(5):
    ramp.record_step()
mid = ramp._get_scale(born)
for _ in range(5):
    ramp.record_step()
end = ramp._get_scale(born)
for _ in range(3):
    ramp.record_step()
print(f"test19: mid={mid:.3f} end={end:.3f} after={ramp._get_scale(born):.3f}")
assert abs(mid - 0.5) < 1e-6 and end == 1.0 and ramp.units[born].ramp_remaining == 0
assert ramp._get_scale(born) == 1.0, "scale must stay at 1 once the ramp ends"
print("test19: PASS (born scale ramps linearly 0 -> 1, then stops)")

# ---- test 20: the ramp does not touch the logit, so optimizer updates are kept ----
ramp.units[born].gradient_ema = 0.0
born2 = next(uid for uid, t in ramp.units.items() if t.state == LifecycleState.DORMANT and uid[0] != "layer")
ramp.units[born2].gradient_ema = 10.0
res20 = ramp.update(step=10, mode="heuristic")
assert res20["births"] == [born2], res20["births"]
for i in range(10):
    ramp.record_step()
    if i == 3:
        ramp._set_logit(born2, ramp._get_logit(born2) + 0.5)  # stand-in for an optimizer step
got20 = ramp._get_logit(born2)
print(f"test20: end logit={got20:.3f} expected={ramp.birth_init_logit + 0.5:.3f}")
assert abs(got20 - (ramp.birth_init_logit + 0.5)) < 1e-6
print("test20: PASS (optimizer movement during the ramp is kept)")

# ---- test 21: a negative ramp length is rejected ----
try:
    StructuralController(ramp_model, birth_ramp_steps=-1)
    raise AssertionError("expected ValueError for negative birth_ramp_steps")
except ValueError:
    print("test21: PASS (negative birth_ramp_steps rejected)")


# ===== units inside a dead layer (tests 22-25) =====
# Death is final, so a head/FFN group whose layer is DEAD can never matter
# again: birth and death must skip it, and it must stay out of the medians.
torch.manual_seed(3)
dl_model = PlasticTransformer(cfg)
dl = StructuralController(dl_model, birth_limit=1, death_limit=2, minimum_age=cfg.minimum_age,
                          death_relative_margin=0.5, death_patience=1, birth_ramp_steps=0, death_ramp_steps=0, seed=0)
dl.seed_architecture()
for t in dl.units.values():  # every unit MATURE, old enough, equal importance
    t.state = LifecycleState.MATURE
    t.steps_since_birth = cfg.minimum_age + 1
    t.importance_ema = 1.0
dead_li = 2
dl.units[("layer", dead_li, 0)].state = LifecycleState.DEAD

# ---- test 22: a dormant unit in a dead layer is never born ----
inside = ("head", dead_li, 0)
outside = ("head", 0, 0)
for uid in (inside, outside):
    dl.units[uid].state = LifecycleState.DORMANT
dl.units[inside].gradient_ema = 100.0  # by far the best raw score
dl.units[outside].gradient_ema = 1.0
births22, _ = dl._birth_pass(step=10, mode="heuristic")
print(f"test22: births={births22}")
assert births22 == [outside], births22
assert dl.units[inside].state == LifecycleState.DORMANT
births22r, _ = dl._birth_pass(step=11, mode="random")
assert inside not in births22r, "random mode must skip dead-layer units too"
print("test22: PASS (dead-layer candidate skipped by heuristic and random birth)")

# ---- test 23: a mature unit in a dead layer is never killed, and is out of the median ----
orphan = ("ffn_group", dead_li, 0)
dl.units[orphan].importance_ema = 0.0  # would be the first to die if it were counted
for uid, t in dl.units.items():
    if uid[1] == dead_li and uid[0] != "layer" and t.state == LifecycleState.MATURE:
        t.importance_ema = 0.0  # all orphans look dead: would drag the FFN median to ~0
deaths23 = dl._death_pass(step=12)
print(f"test23: deaths={deaths23} ffn median={dl._last_importance_spread['ffn_group']['median']}")
assert not any(dl._in_inactive_layer(u) for u in deaths23), deaths23
assert dl._last_importance_spread["ffn_group"]["min"] == 1.0, "orphans must not enter the median pool"
assert deaths23 == [], "no live unit is below 0.5 x median"
print("test23: PASS (dead-layer units neither killed nor counted in the median)")

# ---- test 24: a layer and its own units die in the same pass -> its units do not take the slots ----
li = 1
dl.units[("layer", li, 0)].importance_ema = 0.0
for hi in range(cfg.max_heads):
    dl.units[("head", li, hi)].importance_ema = 0.0  # in the same pass, these would fill the other slot
victim = ("head", 3, 1)
dl.units[victim].importance_ema = 0.1
deaths24 = dl._death_pass(step=13)
print(f"test24: deaths={deaths24}")
assert ("layer", li, 0) in deaths24
assert victim in deaths24, "the second slot must go to a live unit, not to a head inside the layer that just died"
assert not any(u[0] == "head" and u[1] == li for u in deaths24)
print("test24: PASS (same-pass layer death does not leak slots to its own units)")

# ---- test 25: update() reports the units stranded in dead layers ----
res25 = dl.update(step=14, mode="heuristic")
expected = sum(1 for uid, t in dl.units.items() if t.state != LifecycleState.DEAD and uid[0] != "layer" and uid[1] in (dead_li, li))
print(f"test25: in_inactive_layers={res25['in_inactive_layers']} expected={expected}")
assert res25["in_inactive_layers"] == expected and expected > 0
print("test25: PASS (in_inactive_layers reported)")

# ===== death_ramp_steps (tests 26-30) =====
# A fresh model with no gradients: record_step then only advances the ramp.
torch.manual_seed(4)
dr_model = PlasticTransformer(cfg)
dr = StructuralController(dr_model, birth_limit=0, death_limit=1, minimum_age=cfg.minimum_age,
                          death_relative_margin=0.5, death_patience=1, death_ramp_steps=10, seed=0)
assert dr.death_ramp_steps == 10
dr.seed_architecture()
for uid, t in dr.units.items():  # every unit MATURE at scale 1, old enough, equal importance
    t.state = LifecycleState.MATURE
    t.steps_since_birth = cfg.minimum_age + 1
    t.importance_ema = 1.0
    dr._set_scale(uid, 1.0)
    dr._set_probe(uid, False)
dying = ("layer", 1, 0)
dr.units[dying].importance_ema = 0.0
dr._set_logit(dying, 1.0)

# ---- test 26: death sets DEAD at once but does not cut the scale ----
res26 = dr.update(step=5, mode="heuristic")
print(f"test26: deaths={res26['deaths']} dying={res26['dying']} ramping={res26['ramping']}")
assert res26["deaths"] == [dying]
assert dr.units[dying].state == LifecycleState.DEAD
assert dr._get_scale(dying) == 1.0, "death must not cut the scale when ramping"
assert res26["dying"] == 1 and res26["ramping"] == 0
print("test26: PASS (death decision is immediate, the cut is not)")

# ---- test 27: the scale falls linearly to exactly 0, then stays there ----
for _ in range(5):
    dr.record_step()
mid_scale = dr._get_scale(dying)
for _ in range(5):
    dr.record_step()
end_scale = dr._get_scale(dying)
for _ in range(3):
    dr.record_step()
print(f"test27: mid_scale={mid_scale:.4f} end_scale={end_scale}")
assert abs(mid_scale - 0.5) < 1e-6
assert end_scale == 0.0 and dr.units[dying].ramp_remaining == 0
assert dr._get_scale(dying) == 0.0, "scale must stay at 0 once the ramp ends"
print("test27: PASS (dying scale ramps linearly to exactly 0, then stops)")

# ---- test 28: the optimizer cannot hold up a dying unit ----
dying2 = ("head", 0, 1)
dr.units[dying2].importance_ema = 0.0
res28 = dr.update(step=10, mode="heuristic")
assert res28["deaths"] == [dying2], res28["deaths"]
for i in range(10):
    dr.record_step()
    if i == 3:
        dr._set_logit(dying2, dr._get_logit(dying2) + 2.0)  # stand-in for an optimizer step against the ramp
    if i == 4:
        assert abs(dr._get_scale(dying2) - 0.5) < 1e-6, "ramp must follow its schedule, not the optimizer"
gates28 = dr_model._gates()[1]
print(f"test28: end scale={dr._get_scale(dying2)} effective gate={float(gates28[0, 1])}")
assert float(gates28[0, 1]) == 0.0, "the effective gate must be exactly 0 whatever the logit"
print("test28: PASS (optimizer movement against the death ramp cannot hold the unit up)")

# ---- test 29: units in a dying layer are out of birth, death and the medians at once ----
orphan29 = ("head", 1, 0)
dr.units[orphan29].importance_ema = 0.0  # would die next if its layer were still live
res29 = dr.update(step=15, mode="heuristic")
print(f"test29: deaths={res29['deaths']} in_inactive_layers={res29['in_inactive_layers']}")
assert orphan29 not in res29["deaths"] and dr.units[orphan29].state == LifecycleState.MATURE
assert res29["in_inactive_layers"] == cfg.max_heads + cfg.ffn_groups
print("test29: PASS (a dying layer's units count as in a dead layer)")

# ---- test 30: a negative death ramp length is rejected ----
try:
    StructuralController(dr_model, death_ramp_steps=-1)
    raise AssertionError("expected ValueError for negative death_ramp_steps")
except ValueError:
    print("test30: PASS (negative death_ramp_steps rejected)")

# ---- test 31: structure changes stop when a ramp cannot finish before training ends ----
fit = StructuralController(dr_model, birth_ramp_steps=10, death_ramp_steps=20)
assert fit.ramps_fit(20) and fit.ramps_fit(100)
assert not fit.ramps_fit(19) and not fit.ramps_fit(0)
no_ramp = StructuralController(dr_model, birth_ramp_steps=0, death_ramp_steps=0)
assert no_ramp.ramps_fit(0), "without ramps, a check on the final step is still allowed"
print("test31: PASS (ramps_fit uses the longer ramp; no ramps -> no freeze)")

# ===== death_freeze_lr_fraction (tests 32-35) =====
torch.manual_seed(5)
fz_model = PlasticTransformer(cfg)
fz = StructuralController(fz_model, birth_limit=0, death_limit=1, minimum_age=cfg.minimum_age,
                          death_relative_margin=0.5, death_patience=1, death_ramp_steps=0, seed=0)
assert fz.death_freeze_lr_fraction == 0.05, "0.05 must be the default"
fz.seed_architecture()
for t in fz.units.values():  # every unit MATURE, old enough, equal importance
    t.state = LifecycleState.MATURE
    t.steps_since_birth = cfg.minimum_age + 1
    t.importance_ema = 1.0
weak_fz = ("head", 1, 2)
fz.units[weak_fz].importance_ema = 0.0

# ---- test 32: below the LR fraction, an eligible unit does not die, but its streak still counts ----
res32 = fz.update(step=5, mode="heuristic", lr_fraction=0.01)
print(f"test32: deaths={res32['deaths']} frozen={res32['deaths_frozen']} streak={fz._death_streak[weak_fz]}")
assert res32["deaths"] == [] and res32["deaths_frozen"] is True
assert fz.units[weak_fz].state == LifecycleState.MATURE
assert fz._death_streak[weak_fz] >= 1 and res32["below_floor_count"] >= 1, "diagnostics must still update while frozen"
print("test32: PASS (no death in the LR tail; streaks and diagnostics still update)")

# ---- test 33: at or above the LR fraction, the same unit dies ----
res33 = fz.update(step=10, mode="heuristic", lr_fraction=0.05)
print(f"test33: deaths={res33['deaths']} frozen={res33['deaths_frozen']}")
assert res33["deaths"] == [weak_fz] and res33["deaths_frozen"] is False
print("test33: PASS (deaths resume at the threshold)")

# ---- test 34: no LR given, or a threshold of 0, never freezes ----
weak_fz2 = ("head", 2, 1)
fz.units[weak_fz2].importance_ema = 0.0
res34 = fz.update(step=15, mode="heuristic")
assert res34["deaths"] == [weak_fz2] and res34["deaths_frozen"] is False
fz.death_freeze_lr_fraction = 0.0
weak_fz3 = ("head", 3, 0)
fz.units[weak_fz3].importance_ema = 0.0
res34b = fz.update(step=20, mode="heuristic", lr_fraction=0.0)
print(f"test34: deaths={res34['deaths']} / {res34b['deaths']}")
assert res34b["deaths"] == [weak_fz3] and res34b["deaths_frozen"] is False
print("test34: PASS (lr_fraction=None or threshold 0 -> no freeze)")

# ---- test 35: a threshold outside [0, 1) is rejected ----
for bad in (-0.1, 1.0):
    try:
        StructuralController(fz_model, death_freeze_lr_fraction=bad)
        raise AssertionError(f"expected ValueError for death_freeze_lr_fraction={bad}")
    except ValueError:
        pass
print("test35: PASS (death_freeze_lr_fraction outside [0, 1) rejected)")

# ===== birth_freeze_lr_fraction (tests 36-39) =====
torch.manual_seed(6)
bf_model = PlasticTransformer(cfg)
bf = StructuralController(bf_model, birth_limit=1, death_limit=0, birth_ramp_steps=0, seed=0)
assert bf.birth_freeze_lr_fraction == 0.05, "0.05 must be the default"
bf.seed_architecture()
strong_bf = next(uid for uid, t in bf.units.items() if t.state == LifecycleState.DORMANT)
bf.units[strong_bf].gradient_ema = 10.0
bf.units[strong_bf].contribution_ema = 10.0
logit_before = bf._get_logit(strong_bf)

# ---- test 36: below the LR fraction, a strong candidate is not born, but its score still counts ----
res36 = bf.update(step=5, mode="heuristic", lr_fraction=0.01)
print(f"test36: births={res36['births']} frozen={res36['births_frozen']} max_score={res36['max_dormant_birth_score']:.3f}")
assert res36["births"] == [] and res36["births_frozen"] is True
assert bf.units[strong_bf].state == LifecycleState.DORMANT
assert bf._get_logit(strong_bf) == logit_before, "a frozen birth must not touch the logit"
assert res36["max_dormant_birth_score"] > 1.0, "birth diagnostics must still update while frozen"
assert not any(e["event"] == "birth" for e in bf.events)
print("test36: PASS (no birth in the LR tail; scores and diagnostics still update)")

# ---- test 37: at or above the LR fraction, the same unit is born ----
res37 = bf.update(step=10, mode="heuristic", lr_fraction=0.05)
print(f"test37: births={res37['births']} frozen={res37['births_frozen']}")
assert res37["births"] == [strong_bf] and res37["births_frozen"] is False
print("test37: PASS (births resume at the threshold)")

# ---- test 38: random mode is frozen too; no LR, or a threshold of 0, never freezes ----
assert bf.update(step=15, mode="random", lr_fraction=0.01)["births"] == []
res38 = bf.update(step=20, mode="random")
assert len(res38["births"]) == 1 and res38["births_frozen"] is False
bf.birth_freeze_lr_fraction = 0.0
res38b = bf.update(step=25, mode="random", lr_fraction=0.0)
print(f"test38: births={res38['births']} / {res38b['births']} deaths_frozen={res38b['deaths_frozen']}")
assert len(res38b["births"]) == 1 and res38b["births_frozen"] is False
assert res38b["deaths_frozen"] is True, "the two freezes must be independent"
print("test38: PASS (random mode frozen; lr_fraction=None or threshold 0 -> no freeze; independent of the death freeze)")

# ---- test 39: a threshold outside [0, 1) is rejected ----
for bad in (-0.1, 1.0):
    try:
        StructuralController(bf_model, birth_freeze_lr_fraction=bad)
        raise AssertionError(f"expected ValueError for birth_freeze_lr_fraction={bad}")
    except ValueError:
        pass
print("test39: PASS (birth_freeze_lr_fraction outside [0, 1) rejected)")

# ===== units in a DORMANT layer (tests 40-42) =====
# A layer with scale 0 makes all its units' outputs exactly 0, so they are out
# of death, birth and the medians, as in a DEAD layer. This controller keeps
# its real seed split (no all_layers_live).
torch.manual_seed(7)
il_model = PlasticTransformer(cfg)
il = StructuralController(il_model, birth_limit=1, death_limit=2, minimum_age=cfg.minimum_age,
                          death_relative_margin=0.5, death_patience=1, birth_ramp_steps=0, death_ramp_steps=0, seed=0)
il.seed_architecture()
dormant_layers = [li for li in range(cfg.max_layers) if il.units[("layer", li, 0)].state == LifecycleState.DORMANT]
live_layers = [li for li in range(cfg.max_layers) if il.units[("layer", li, 0)].state == LifecycleState.MATURE]
assert dormant_layers and live_layers, (dormant_layers, live_layers)
dl_i = next(li for li in dormant_layers
            if any(il.units[("head", li, h)].state == LifecycleState.MATURE for h in range(cfg.max_heads))
            and any(il.units[("head", li, h)].state == LifecycleState.DORMANT for h in range(cfg.max_heads)))
mature_in = next(("head", dl_i, h) for h in range(cfg.max_heads) if il.units[("head", dl_i, h)].state == LifecycleState.MATURE)
dormant_in = next(("head", dl_i, h) for h in range(cfg.max_heads) if il.units[("head", dl_i, h)].state == LifecycleState.DORMANT)
for t in il.units.values():
    t.steps_since_birth = cfg.minimum_age + 1
    t.importance_ema = 1.0
assert any(il.units[("head", li, h)].state == LifecycleState.MATURE for li in live_layers for h in range(cfg.max_heads))

# ---- test 40: a mature head in a dormant layer never dies and is out of the median ----
il.units[mature_in].importance_ema = 0.0  # exactly what its zero gradient gives it
deaths40 = il._death_pass(step=5) + il._death_pass(step=6)
print(f"test40: dormant layer {dl_i}; deaths={deaths40} head median min={il._last_importance_spread['head']['min']}")
assert mature_in not in deaths40 and il.units[mature_in].state == LifecycleState.MATURE
assert il._last_importance_spread["head"]["min"] == 1.0, "units in a dormant layer must not enter the median"
print("test40: PASS (mature unit in a dormant layer: no death, not in the median)")

# ---- test 41: a dormant head in a dormant layer is never born ----
for t in il.units.values():
    t.gradient_ema = 0.0
il.units[dormant_in].gradient_ema = 100.0
births41, _ = il._birth_pass(step=7, mode="heuristic")
il.birth_limit = 1000  # random mode then takes every eligible candidate
births41r, _ = il._birth_pass(step=8, mode="random")
il.birth_limit = 1
print(f"test41: heuristic births={births41}; random births include it: {dormant_in in births41r}")
assert dormant_in not in births41 and dormant_in not in births41r
assert il.units[dormant_in].state == LifecycleState.DORMANT
print("test41: PASS (dormant-layer candidate skipped by heuristic and random birth)")

# ---- test 42: a layer birth restarts its live units and leaves its dormant units dormant ----
torch.manual_seed(8)
lb_model = PlasticTransformer(cfg)
lb = StructuralController(lb_model, birth_limit=1, minimum_age=cfg.minimum_age, birth_ramp_steps=0, seed=0)
lb.seed_architecture()
lb_layer = ("layer", dl_i, 0)
assert lb.units[lb_layer].state == LifecycleState.DORMANT
for uid, t in lb.units.items():
    t.gradient_ema = 0.0
    t.steps_since_birth = cfg.minimum_age + 1
    lb._death_streak[uid] = 1
lb.units[lb_layer].gradient_ema = 100.0
births42, _ = lb._birth_pass(step=9, mode="heuristic")
assert births42 == [lb_layer], births42
children = [u for u in lb.units if u[0] != "layer" and u[1] == dl_i]
restarted = [u for u in children if lb.units[u].state == LifecycleState.PROTECTED]
still_dormant = [u for u in children if lb.units[u].state == LifecycleState.DORMANT]
print(f"test42: layer {dl_i} born; {len(restarted)} live units restarted, {len(still_dormant)} stay dormant")
assert restarted and still_dormant and len(restarted) + len(still_dormant) == len(children)
assert all(lb.units[u].steps_since_birth == 0 and lb._death_streak[u] == 0 for u in restarted)
assert all(lb._get_scale(u) == 1.0 for u in restarted), "live units keep scale 1"
assert all(lb._get_scale(u) == 0.0 and float(lb._buffer(u, "probe_mask")[0][lb._buffer(u, "probe_mask")[1]]) == 1.0 for u in still_dormant)
assert lb._get_scale(lb_layer) == 1.0
print("test42: PASS (layer birth: live units restart as PROTECTED at age 0; dormant units keep their probe)")

# ---- test 43: birth_cost ranks dormant units by score per unit of compute ----
def cost_ctl(birth_cost):
    """Layers 0-2 live, layer 3 dormant. One dormant head and one dormant FFN
    group in layer 0. Raw scores: head 3, layer 3 2, FFN group 1, rest 0."""
    torch.manual_seed(9)
    c = StructuralController(PlasticTransformer(cfg), birth_limit=1, minimum_age=cfg.minimum_age,
                             birth_ramp_steps=0, birth_cost=birth_cost, seed=0)
    c.seed_architecture()
    all_layers_live(c)
    for uid in [("layer", 3, 0), ("head", 0, 0), ("ffn_group", 0, 0)]:
        c.units[uid].state = LifecycleState.DORMANT
        c._set_scale(uid, 0.0)
        c._set_probe(uid, True)
    for t in c.units.values():
        t.gradient_ema = t.contribution_ema = 0.0
    for uid, g in [(("head", 0, 0), 3.0), (("layer", 3, 0), 2.0), (("ffn_group", 0, 0), 1.0)]:
        c.units[uid].gradient_ema = g / c.w_g  # score = w_g * gradient_ema (plateau 0, contribution 0)
    return c

cost = {"head": 4.0, "ffn_group": 1.0}
raw43, _ = cost_ctl(None)._birth_pass(step=10, mode="heuristic")
c43 = cost_ctl(cost)
cost43, _ = c43._birth_pass(step=10, mode="heuristic")
print(f"test43: raw births={raw43}; with birth_cost={cost43}; by kind={c43._last_max_birth_score_by_kind}")
assert raw43 == [("head", 0, 0)], "raw scores: the head (3) wins"
assert cost43 == [("layer", 3, 0)], "with cost: layer (2, divisor 1) > FFN group (1) > head (3/4)"
assert c43._last_max_birth_score_by_kind == {"layer": 2.0, "ffn_group": 1.0, "head": 0.75}, c43._last_max_birth_score_by_kind
c43b = cost_ctl(cost)
c43b.units[("layer", 3, 0)].gradient_ema = 0.0
assert c43b._birth_pass(step=10, mode="heuristic")[0] == [("ffn_group", 0, 0)], "with cost and no layer: the FFN group wins"
c43c = cost_ctl(cost)
log43 = c43c.update(step=10, mode="heuristic")
assert log43["birth_cost"] == cost and log43["max_birth_score_by_kind"]["head"] == 0.75, log43
assert c43c.events[-1]["score"] == 2.0, "the event logs the cost-adjusted score"
for bad in [{"head": 0.0}, {"head": -1.0}, {"head": float("nan")}]:
    try:
        StructuralController(PlasticTransformer(cfg), birth_cost=bad)
        raise AssertionError(f"birth_cost={bad} accepted")
    except ValueError:
        pass
print("test43: PASS (birth_cost divides head/FFN scores, layers keep divisor 1, growth log has per-kind scores)")

print("\nALL TESTS PASSED")
