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

# ---- test 4: a mature unit well below its kind's median, sustained, dies -- a healthy peer survives ----
# pick a kind with >=3 mature units so there's a real median to be below/above
from collections import defaultdict

mature_by_kind = defaultdict(list)
for uid, t in controller.units.items():
    if t.state == LifecycleState.MATURE:
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
assert abs(controller._get_logit(worst_uid) - controller.dead_logit) < 1e-6
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
    if t.state == LifecycleState.MATURE:
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

dormant_uids = [uid for uid, t in controller.units.items() if t.state == LifecycleState.DORMANT]
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
born = next(uid for uid, t in ramp.units.items() if t.state == LifecycleState.DORMANT)
ramp.units[born].gradient_ema = 10.0
start = ramp._get_logit(born)

# ---- test 19: birth does not move the logit; it climbs linearly to birth_init_logit ----
res19 = ramp.update(step=5, mode="heuristic")
assert res19["births"] == [born], res19["births"]
assert ramp.units[born].state == LifecycleState.PROTECTED
assert abs(ramp._get_logit(born) - start) < 1e-6, "birth must not jump the logit when ramping"
assert res19["ramping"] == 1
for _ in range(5):
    ramp.record_step()
mid = ramp._get_logit(born)
assert abs(mid - (start + ramp.birth_init_logit) / 2) < 1e-4, (start, mid)
for _ in range(5):
    ramp.record_step()
end = ramp._get_logit(born)
for _ in range(3):
    ramp.record_step()
print(f"test19: start={start:.3f} mid={mid:.3f} end={end:.3f} after={ramp._get_logit(born):.3f}")
assert abs(end - ramp.birth_init_logit) < 1e-4 and ramp.units[born].ramp_remaining == 0
assert abs(ramp._get_logit(born) - end) < 1e-6, "logit must stop moving once the ramp ends"
print("test19: PASS (born logit ramps linearly to birth_init_logit, then stops)")

# ---- test 20: the ramp adds to the optimizer's updates instead of overwriting them ----
ramp.units[born].gradient_ema = 0.0
born2 = next(uid for uid, t in ramp.units.items() if t.state == LifecycleState.DORMANT)
ramp.units[born2].gradient_ema = 10.0
res20 = ramp.update(step=10, mode="heuristic")
assert res20["births"] == [born2], res20["births"]
start2 = ramp._get_logit(born2)
for i in range(10):
    ramp.record_step()
    if i == 3:
        ramp._set_logit(born2, ramp._get_logit(born2) + 0.5)  # stand-in for an optimizer step
got20 = ramp._get_logit(born2)
print(f"test20: start={start2:.3f} end={got20:.3f} expected={ramp.birth_init_logit + 0.5:.3f}")
assert abs(got20 - (ramp.birth_init_logit + 0.5)) < 1e-4
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
assert not any(dl._in_dead_layer(u) for u in deaths23), deaths23
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
print(f"test25: in_dead_layers={res25['in_dead_layers']} expected={expected}")
assert res25["in_dead_layers"] == expected and expected > 0
print("test25: PASS (in_dead_layers reported)")

# ===== death_ramp_steps (tests 26-30) =====
# A fresh model with no gradients: record_step then only advances the ramp.
torch.manual_seed(4)
dr_model = PlasticTransformer(cfg)
dr = StructuralController(dr_model, birth_limit=0, death_limit=1, minimum_age=cfg.minimum_age,
                          death_relative_margin=0.5, death_patience=1, death_ramp_steps=10, seed=0)
assert dr.death_ramp_steps == 10
dr.seed_architecture()
for uid, t in dr.units.items():  # every unit MATURE, old enough, equal importance
    t.state = LifecycleState.MATURE
    t.steps_since_birth = cfg.minimum_age + 1
    t.importance_ema = 1.0
dying = ("layer", 1, 0)
dr.units[dying].importance_ema = 0.0
dr._set_logit(dying, 1.0)


def gate_of(uid):
    return 1.0 / (1.0 + math.exp(-dr._get_logit(uid)))


start_gate = gate_of(dying)
dead_gate = 1.0 / (1.0 + math.exp(-dr.dead_logit))

# ---- test 26: death sets DEAD at once but does not move the logit ----
res26 = dr.update(step=5, mode="heuristic")
print(f"test26: deaths={res26['deaths']} dying={res26['dying']} ramping={res26['ramping']}")
assert res26["deaths"] == [dying]
assert dr.units[dying].state == LifecycleState.DEAD
assert abs(dr._get_logit(dying) - 1.0) < 1e-6, "death must not cut the logit when ramping"
assert res26["dying"] == 1 and res26["ramping"] == 0
print("test26: PASS (death decision is immediate, the cut is not)")

# ---- test 27: the gate falls linearly to sigmoid(dead_logit), then the logit stays at dead_logit ----
for _ in range(5):
    dr.record_step()
mid_gate = gate_of(dying)
assert abs(mid_gate - (start_gate + dead_gate) / 2) < 1e-5, (start_gate, mid_gate)
for _ in range(5):
    dr.record_step()
end_logit = dr._get_logit(dying)
for _ in range(3):
    dr.record_step()
print(f"test27: start_gate={start_gate:.4f} mid_gate={mid_gate:.4f} end_logit={end_logit:.3f}")
assert end_logit == dr.dead_logit and dr.units[dying].ramp_remaining == 0
assert dr._get_logit(dying) == dr.dead_logit, "logit must stop moving once the ramp ends"
print("test27: PASS (dying gate ramps linearly to the dead gate, then stops)")

# ---- test 28: the optimizer cannot hold up a dying unit ----
dying2 = ("head", 0, 1)
dr.units[dying2].importance_ema = 0.0
dr._set_logit(dying2, 0.0)
res28 = dr.update(step=10, mode="heuristic")
assert res28["deaths"] == [dying2], res28["deaths"]
for i in range(10):
    dr.record_step()
    if i == 3:
        dr._set_logit(dying2, dr._get_logit(dying2) + 2.0)  # stand-in for an optimizer step against the ramp
    if i == 4:
        on_schedule = 0.5 + (dead_gate - 0.5) * 0.5
        assert abs(gate_of(dying2) - on_schedule) < 1e-5, "ramp must follow its schedule, not the optimizer"
print(f"test28: end_logit={dr._get_logit(dying2):.3f}")
assert dr._get_logit(dying2) == dr.dead_logit
print("test28: PASS (optimizer movement against the death ramp is overridden)")

# ---- test 29: units in a dying layer are out of birth, death and the medians at once ----
orphan29 = ("head", 1, 0)
dr.units[orphan29].importance_ema = 0.0  # would die next if its layer were still live
res29 = dr.update(step=15, mode="heuristic")
print(f"test29: deaths={res29['deaths']} in_dead_layers={res29['in_dead_layers']}")
assert orphan29 not in res29["deaths"] and dr.units[orphan29].state == LifecycleState.MATURE
assert res29["in_dead_layers"] == cfg.max_heads + cfg.ffn_groups
print("test29: PASS (a dying layer's units count as in a dead layer)")

# ---- test 30: a negative death ramp length is rejected ----
try:
    StructuralController(dr_model, death_ramp_steps=-1)
    raise AssertionError("expected ValueError for negative death_ramp_steps")
except ValueError:
    print("test30: PASS (negative death_ramp_steps rejected)")

print("\nALL TESTS PASSED")
