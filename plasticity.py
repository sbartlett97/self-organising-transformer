from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch


class LifecycleState(Enum):
    DORMANT = "dormant"
    PROTECTED = "protected"
    MATURE = "mature"
    DEAD = "dead"


@dataclass
class UnitTelemetry:
    unit_id: tuple
    state: LifecycleState = LifecycleState.DORMANT
    gradient_ema: float = 0.0
    contribution_ema: float = 0.0
    importance_ema: float = 0.0
    age: int = 0
    steps_since_birth: int = 0
    ramp_remaining: int = 0


class StructuralController:
    """Implements proposal §11's discrete birth/death lifecycle
    (DORMANT -> PROTECTED -> MATURE -> DEAD) on top of a PlasticTransformer's
    continuous gate substrate. It sets each unit's scale in the model
    (gate = scale * sigmoid(logit)): DORMANT and DEAD units have scale 0, an
    exact zero with zero gradient, so only the active structure affects the
    loss. Birth ramps the scale 0 -> 1; death ramps it 1 -> 0. A dormant
    unit's birth signal is the gradient on its zero-valued probe (dL/dg),
    which never updates any weight. Units inside a DORMANT or DEAD layer
    have no effect either, so birth, death and the kind medians skip them.

    Scope: manages layer/head/ffn_group units only. Skip-edge lifecycle is
    left continuous (Phase-1-style), same as every other phase up to the
    proposal's Phase 6 (§15), which is the first to treat skip edges as a
    distinct mutation type.
    """

    def __init__(
        self,
        model,
        growth_interval: int = 250,
        birth_limit: int = 2,
        death_limit: int = 2,
        minimum_age: int = 1000,
        # An absolute threshold here assumed "good" gates reach ~0.5+, but
        # this architecture's gates settle around 0.2-0.35 even for
        # genuinely kept components in Phase 1 (same miscalibration
        # materialize.py's old absolute threshold=0.5 default had, already
        # fixed there via a relative, auto-median threshold -- see
        # hub_utils.py's _architecture_section). Confirmed in Stage 2's
        # first real-scale run (2026-10-06): death_threshold=0.5 flagged
        # nearly the entire mature population every check regardless of
        # merit, keeping births and deaths in lockstep and flatlining net
        # population growth despite 2.5x unused architecture-cost budget.
        # death_relative_margin replaces it: a unit is death-eligible only
        # when its gate falls below `margin` x its own kind's (layer/head/
        # ffn_group) current mature-population median -- "meaningfully
        # behind its surviving peers," recalibrated every check, rather
        # than a fixed magnitude this architecture may never reach.
        death_relative_margin: float = 0.5,
        death_patience: int = 3,
        # What death compares against the kind median. "importance" (default)
        # is an EMA of the first-order Taylor estimate |g * dL/dg| of the loss
        # change from removing the unit. "gate" is the original gate-value
        # rule, kept for comparison. Gate value does not track importance in
        # this architecture: an ablation test on the Phase 2 attempt-4
        # checkpoint (ablation_importance.py, RESEARCH_LOG.md 2026-10-09)
        # found rank correlation with true loss change of -0.72 for layers
        # and ~0 for FFN groups, and gates so uniform that the gate rule
        # never fired in four runs. The Taylor estimate tracked true
        # importance (0.93/0.53/0.56 on unseen batches) and is spread widely.
        death_criterion: str = "importance",
        ema_decay: float = 0.98,
        birth_weight_gradient: float = 0.5,
        birth_weight_contribution: float = 0.3,
        birth_weight_plateau: float = 0.2,
        # A dormant unit's gate sits at dormant_floor_logit, deep in the
        # sigmoid's saturated tail -- gradient flowing back to its own gate
        # logit is structurally small *regardless of whether activating it
        # would actually help*, since sigmoid saturation suppresses the
        # gradient, not just the forward contribution. An absolute
        # birth_score_threshold (the old design) could therefore only ever
        # be cleared when the plateau term's multiplicative boost pushed
        # scores up -- meaning growth timing tracked "has global loss been
        # flat for the last 2500 steps" rather than "does this candidate
        # show genuine demand" (confirmed in two 2026-10 real-scale runs:
        # births occurred in tight bursts exactly when plateau was
        # elevated, and stopped dead the moment it crashed to 0, on both a
        # win and a loss). Same class of miscalibration as the old absolute
        # death_threshold=0.5, fixed there via a relative, auto-median
        # comparison. Fixed here the mirror-image way: birth eligibility is
        # now the top birth_limit dormant candidates by raw score each
        # check (ranking, not an absolute floor), so "is there a
        # comparatively-best candidate right now" decides growth, not
        # "has anyone happened to clear a fixed bar this cycle".
        # birth_score_floor only excludes genuinely zero-signal candidates
        # (a unit that has received no gradient and shows no plateau boost
        # at all) -- it is a sanity guard, not the primary gate.
        birth_score_floor: float = 0.0,
        dormant_floor_logit: float = -4.0,
        birth_init_logit: float = -1.0,
        # Steps over which a born unit's scale climbs linearly 0 -> 1 (its
        # logit is set to birth_init_logit at birth and trains freely). 0 =
        # full scale in one step. A one-step birth adds untrained weights at
        # full gate: attempt 5 (RESEARCH_LOG.md 2026-10-13) saw val ppl go
        # 30.7 -> 143 right after two head births in layers 0-1.
        birth_ramp_steps: int = 250,
        # Steps over which a dying unit's scale falls linearly 1 -> 0. 0 =
        # cut in one step. Attempt 5 (RESEARCH_LOG.md 2026-10-13): layer 7's
        # one-step death at step 4500 cost about 1000 steps of val recovery.
        # The scale ends at exactly 0 whatever the optimizer does to the
        # logit, so a unit chosen to die cannot be held up.
        death_ramp_steps: int = 250,
        # No death once the LR falls below this fraction of its peak (pass
        # the fraction to update()). 0 = never freeze. Attempt 6
        # (RESEARCH_LOG.md 2026-10-14) killed 5 units, layer-0 heads among
        # them, at LR 0.2-1.5% of peak: val ppl went 27.29 -> 29.28, since the
        # rest of the network could not adapt. Deaths at 16-23% of peak did
        # no harm. 0.05 stops deaths from step 8750 of 10000 on that schedule.
        # Streaks and diagnostics still update while frozen.
        death_freeze_lr_fraction: float = 0.05,
        # No birth once the LR falls below this fraction of its peak. 0 =
        # never freeze. Attempt 7 (RESEARCH_LOG.md 2026-10-14) froze deaths
        # but still fell 27.29 -> 30.66 val ppl: one head was born at each
        # of steps 9000-9750, at LR <= 2.7% of peak. Too little LR remains
        # to train a born unit, and setting those 4 heads back to dormant
        # gave 27.16. Birth scores and diagnostics still update while frozen.
        birth_freeze_lr_fraction: float = 0.05,
        seed: int = 0,
    ):
        self.model = model
        self.growth_interval = growth_interval
        self.birth_limit = birth_limit
        self.death_limit = death_limit
        self.minimum_age = minimum_age
        self.death_relative_margin = death_relative_margin
        self.death_patience = death_patience
        if death_criterion not in ("importance", "gate"):
            raise ValueError(f"unknown death_criterion {death_criterion!r} (expected 'importance' or 'gate')")
        self.death_criterion = death_criterion
        self.ema_decay = ema_decay
        self.w_g = birth_weight_gradient
        self.w_c = birth_weight_contribution
        self.w_p = birth_weight_plateau
        self.birth_score_floor = birth_score_floor
        self.dormant_floor_logit = dormant_floor_logit
        self.birth_init_logit = birth_init_logit
        if birth_ramp_steps < 0:
            raise ValueError(f"birth_ramp_steps must be >= 0, got {birth_ramp_steps}")
        self.birth_ramp_steps = birth_ramp_steps
        if death_ramp_steps < 0:
            raise ValueError(f"death_ramp_steps must be >= 0, got {death_ramp_steps}")
        self.death_ramp_steps = death_ramp_steps
        if not 0.0 <= death_freeze_lr_fraction < 1.0:
            raise ValueError(f"death_freeze_lr_fraction must be in [0, 1), got {death_freeze_lr_fraction}")
        self.death_freeze_lr_fraction = death_freeze_lr_fraction
        if not 0.0 <= birth_freeze_lr_fraction < 1.0:
            raise ValueError(f"birth_freeze_lr_fraction must be in [0, 1), got {birth_freeze_lr_fraction}")
        self.birth_freeze_lr_fraction = birth_freeze_lr_fraction
        self._rng = random.Random(seed)

        self.units: dict[tuple, UnitTelemetry] = {}
        self._loss_history: list[float] = []
        self._train_loss_ema: Optional[float] = None
        self._last_max_birth_score: float = 0.0
        self._last_nan_candidates: int = 0
        self._last_below_floor_count: int = 0
        self._last_death_eligible_pool: int = 0
        self._last_max_death_streak: int = 0
        self._last_gate_spread: dict = {}
        self._last_importance_spread: dict = {}
        self.events: list[dict] = []
        self._death_streak: dict[tuple, int] = {}
        self._register_units()

    def _register_units(self) -> None:
        cfg = self.model.config
        for li in range(len(self.model.blocks)):
            self.units[("layer", li, 0)] = UnitTelemetry(unit_id=("layer", li, 0))
            for hi in range(cfg.max_heads):
                self.units[("head", li, hi)] = UnitTelemetry(unit_id=("head", li, hi))
            for gi in range(cfg.ffn_groups):
                self.units[("ffn_group", li, gi)] = UnitTelemetry(unit_id=("ffn_group", li, gi))

    def _logit_param(self, uid: tuple):
        kind, li, idx = uid
        block = self.model.blocks[li]
        if kind == "layer":
            return block.layer_logit, None
        if kind == "head":
            return block.attn.head_logits, idx
        if kind == "ffn_group":
            return block.ffn.group_logits, idx
        raise ValueError(f"unknown unit kind {kind!r}")

    def _get_logit(self, uid: tuple) -> float:
        param, idx = self._logit_param(uid)
        return float(param.item()) if idx is None else float(param[idx].item())

    def _set_logit(self, uid: tuple, value: float) -> None:
        param, idx = self._logit_param(uid)
        with torch.no_grad():
            if idx is None:
                param.fill_(value)
            else:
                param[idx] = value

    def _buffer(self, uid: tuple, what: str):
        kind, li, idx = uid
        name = {"layer": "layer", "head": "head", "ffn_group": "group"}[kind]
        buf = getattr(self.model, f"{name}_{what}")
        return buf, (li if kind == "layer" else (li, idx))

    def _get_scale(self, uid: tuple) -> float:
        buf, i = self._buffer(uid, "scale")
        return float(buf[i])

    def _set_scale(self, uid: tuple, value: float) -> None:
        buf, i = self._buffer(uid, "scale")
        buf[i] = value

    def _set_probe(self, uid: tuple, on: bool) -> None:
        buf, i = self._buffer(uid, "probe_mask")
        buf[i] = 1.0 if on else 0.0

    def _probe_grad(self, uid: tuple) -> Optional[float]:
        kind, li, idx = uid
        name = {"layer": "layer", "head": "head", "ffn_group": "group"}[kind]
        grad = getattr(self.model, f"{name}_probe").grad
        if grad is None:
            return None
        return float(grad[li] if kind == "layer" else grad[li, idx])

    def _in_inactive_layer(self, uid: tuple) -> bool:
        """True for a head or FFN group whose layer is DORMANT or DEAD (a
        dying layer counts as DEAD). Its output is zero, so it cannot affect
        the loss: birth and death skip it, and it stays out of the kind
        medians (its importance is exactly 0). In attempt 5
        (RESEARCH_LOG.md 2026-10-13), 58 of 71 head/FFN deaths hit units in
        dead layers."""
        return uid[0] != "layer" and self.units[("layer", uid[1], 0)].state in (LifecycleState.DORMANT, LifecycleState.DEAD)

    def seed_architecture(self, active_fraction: Optional[float] = None) -> None:
        """Apply Phase 2's small-seed split: a minority of units start MATURE
        (active init), the rest DORMANT (suppressed floor). Call once, right
        after constructing the model and before training starts."""
        fraction = active_fraction if active_fraction is not None else self.model.config.seed_fraction
        for uid, t in self.units.items():
            if self._rng.random() < fraction:
                t.state = LifecycleState.MATURE
                self._set_logit(uid, self.birth_init_logit)
                self._set_scale(uid, 1.0)
                self._set_probe(uid, False)
            else:
                t.state = LifecycleState.DORMANT
                self._set_logit(uid, self.dormant_floor_logit)
                self._set_scale(uid, 0.0)
                self._set_probe(uid, True)

    def record_step(self, train_loss: Optional[float] = None) -> None:
        """Call every training step, after optimizer.zero_grad's companion
        backward pass has populated (and been unscaled into) param.grad, and
        before the next zero_grad call. Cheap: no model forward pass.

        `train_loss` should be this step's training loss, not a sparse
        held-out eval -- it needs to update every step (growth_interval is
        typically much more frequent than eval_every) to give the plateau
        signal anything to work with."""
        for uid, t in self.units.items():
            if t.state == LifecycleState.DORMANT:
                # Birth signal: dL/dg from the zero-valued probe. Dormant
                # weights stay at their random init, so this is far smaller
                # than the pre-2026-10-15 signal (dL/dlogit of a unit that
                # was secretly live and trained): birth_score_floor values
                # from attempts 1-8 (0.005) would block every birth.
                g = self._probe_grad(uid)
                if g is not None and math.isfinite(g):
                    t.gradient_ema = self.ema_decay * t.gradient_ema + (1 - self.ema_decay) * abs(g)
            elif t.state != LifecycleState.DEAD:
                param, idx = self._logit_param(uid)
                grad = param.grad
                if grad is not None:
                    g = float(grad.item()) if idx is None else float(grad[idx].item())
                    # An EMA can never recover from a single non-finite
                    # update (NaN forever after; observed: growth froze for
                    # 1250+ steps with max_dormant_birth_score stuck at NaN).
                    # Skip the update rather than let one bad gradient take
                    # out a unit for the rest of training.
                    if math.isfinite(g):
                        t.gradient_ema = self.ema_decay * t.gradient_ema + (1 - self.ema_decay) * abs(g)
                        # Taylor importance |gate * dL/dgate|. gate = s * sigmoid(logit), so
                        # dL/dlogit = dL/dgate * s * sig * (1 - sig) and gate * dL/dgate =
                        # dL/dlogit / (1 - sig): no extra backward pass.
                        sig = 1.0 / (1.0 + math.exp(-self._get_logit(uid)))
                        taylor = abs(g / max(1.0 - sig, 1e-6))
                        t.importance_ema = self.ema_decay * t.importance_ema + (1 - self.ema_decay) * taylor
            if t.ramp_remaining > 0:
                t.ramp_remaining -= 1
                if t.state == LifecycleState.DEAD:
                    self._set_scale(uid, t.ramp_remaining / self.death_ramp_steps)
                else:
                    self._set_scale(uid, 1.0 - t.ramp_remaining / self.birth_ramp_steps)
            gate_val = self._get_scale(uid) / (1.0 + math.exp(-self._get_logit(uid)))
            t.contribution_ema = self.ema_decay * t.contribution_ema + (1 - self.ema_decay) * gate_val
            t.age += 1
            t.steps_since_birth += 1
        if train_loss is not None:
            self._train_loss_ema = (
                train_loss if self._train_loss_ema is None
                else self.ema_decay * self._train_loss_ema + (1 - self.ema_decay) * train_loss
            )

    def _plateau_score(self) -> float:
        """Loss-plateau demand P_i (shared across candidates this update):
        how flat the (EMA-smoothed) training loss has been across recent
        growth events. Higher = more plateaued = more demand for new
        capacity. Snapshots the EMA once per call -- call this only from
        update(), not every step."""
        if self._train_loss_ema is not None:
            self._loss_history.append(self._train_loss_ema)
        hist = self._loss_history[-10:]
        if len(hist) < 4:
            return 0.0
        slope = (hist[-1] - hist[0]) / max(len(hist), 1)
        return float(max(0.0, 1.0 - min(abs(slope) * 50.0, 1.0)))

    def _death_pass(self, step: int, frozen: bool = False) -> list:
        """Rate-limited to death_limit per call (symmetric to birth_limit) --
        an uncapped death pass can kill every eligible unit in one event if
        they share similar init/regularizer pressure (observed in local
        testing: all seed-mature units crossed the threshold in the same
        growth tick), which violates §11's "stable topology evolution"
        success criterion. Eligible-but-not-selected units keep their streak
        and remain eligible next call, so a real die-off still happens, just
        gradually.

        Eligibility is relative, not absolute: a unit's death signal
        (importance_ema, or gate value under death_criterion="gate") is
        compared against its own kind's (layer/head/ffn_group) current
        mature-population median, recomputed fresh every call -- see
        death_relative_margin's and death_criterion's notes in __init__."""
        mature_gates_by_kind: dict = defaultdict(list)
        mature_importance_by_kind: dict = defaultdict(list)
        for uid, t in self.units.items():
            if t.state == LifecycleState.MATURE and not self._in_inactive_layer(uid):
                mature_gates_by_kind[uid[0]].append(torch.sigmoid(torch.tensor(self._get_logit(uid))).item())
                mature_importance_by_kind[uid[0]].append(t.importance_ema)
        signal_by_kind = mature_importance_by_kind if self.death_criterion == "importance" else mature_gates_by_kind
        kind_median = {kind: statistics.median(vals) for kind, vals in signal_by_kind.items()}

        # Diagnostic-only bookkeeping (2026-10-09): death_streak itself was
        # never surfaced anywhere, so after four real-scale runs with zero
        # deaths there was no way to tell "eligibility never fires" (gates
        # never spread below their kind's relative floor) apart from
        # "eligibility fires but the streak keeps resetting before
        # death_patience completes" (recovery churn). These counters answer
        # that directly from growth_log.json without changing any behavior.
        below_floor_count = 0
        current_streaks = []  # streak values for units actually in play this check -- a dead unit's
        # frozen streak (last value before it died) must not leak into "current" state once it leaves
        # the mature pool, or max_death_streak would stay pinned >= death_patience forever after a
        # single death even though nothing is actively accumulating any more.

        eligible = []
        for uid, t in self.units.items():
            if t.state != LifecycleState.MATURE or t.steps_since_birth < self.minimum_age or self._in_inactive_layer(uid):
                continue
            if self.death_criterion == "importance":
                signal = t.importance_ema
            else:
                signal = torch.sigmoid(torch.tensor(self._get_logit(uid))).item()
            relative_floor = kind_median[uid[0]] * self.death_relative_margin
            death_score = relative_floor - signal  # positive = below its kind's relative floor
            if signal < relative_floor:
                self._death_streak[uid] = self._death_streak.get(uid, 0) + 1
                below_floor_count += 1
            else:
                self._death_streak[uid] = 0
            current_streaks.append(self._death_streak[uid])
            if self._death_streak.get(uid, 0) >= self.death_patience:
                eligible.append((death_score, uid))
        eligible.sort(key=lambda x: -x[0])

        self._last_below_floor_count = below_floor_count
        self._last_death_eligible_pool = len(current_streaks)
        self._last_max_death_streak = max(current_streaks, default=0)
        self._last_gate_spread = {
            kind: {"min": min(vals), "median": statistics.median(vals), "max": max(vals)}
            for kind, vals in mature_gates_by_kind.items()
        }
        self._last_importance_spread = {
            kind: {"min": min(vals), "median": statistics.median(vals), "max": max(vals)}
            for kind, vals in mature_importance_by_kind.items()
        }

        deaths = []
        for death_score, uid in eligible:
            if frozen or len(deaths) >= self.death_limit:
                break
            if self._in_inactive_layer(uid):  # its layer died earlier in this same pass
                continue
            t = self.units[uid]
            t.state = LifecycleState.DEAD
            t.ramp_remaining = self.death_ramp_steps
            if self.death_ramp_steps == 0:
                self._set_scale(uid, 0.0)
            deaths.append(uid)
            self.events.append({"step": step, "unit": uid, "event": "death", "score": death_score})
        return deaths

    def _promote_pass(self) -> None:
        for t in self.units.values():
            if t.state == LifecycleState.PROTECTED and t.steps_since_birth >= self.minimum_age:
                t.state = LifecycleState.MATURE

    def _birth_pass(self, step: int, mode: str, frozen: bool = False) -> tuple[list, float]:
        plateau = self._plateau_score()
        dormant = [uid for uid, t in self.units.items() if t.state == LifecycleState.DORMANT and not self._in_inactive_layer(uid)]

        if mode == "heuristic":
            # plateau (P_i in proposal §11) must be per-candidate, not a flat
            # value broadcast to every dormant unit -- an additive
            # `w_p * plateau` term pushes ALL candidates above threshold
            # simultaneously once plateau is large (loss naturally flattens
            # in any normal run eventually), collapsing "selective, demand-
            # gated growth" into "always max out birth_limit, ranked by
            # noise" (observed in Stage 2: a steady ~1750-step churn loop
            # concentrated on whichever candidate held a marginal gradient_ema
            # edge). Instead, plateau *modulates* each candidate's own
            # gradient pressure -- it amplifies units that already show
            # demand, rather than lowering the bar for every dormant unit
            # regardless of its own signal.
            finite_grads = [self.units[uid].gradient_ema for uid in dormant if math.isfinite(self.units[uid].gradient_ema)]
            max_grad = max(finite_grads, default=0.0)
            all_scores = []
            nan_candidates = 0
            for uid in dormant:
                t = self.units[uid]
                g_norm = (t.gradient_ema / max_grad) if max_grad > 0 and math.isfinite(t.gradient_ema) else 0.0
                b = self.w_g * t.gradient_ema + self.w_c * t.contribution_ema + self.w_p * plateau * g_norm
                # Filter non-finite scores out *before* sorting, not after --
                # NaN comparisons are always False, so a NaN entry left in
                # the list breaks sort()'s total-order assumption and can
                # scatter a poisoned candidate into the top-K slots, crowding
                # out a genuinely good one even though the NaN entry itself
                # would correctly fail the floor check on its own. record_step
                # now guards gradient_ema against this at the source (see its
                # docstring), so this is belt-and-suspenders for anything
                # already-poisoned (e.g. a resumed older run).
                if math.isfinite(b):
                    all_scores.append((b, uid))
                else:
                    nan_candidates += 1
            self._last_max_birth_score = max((b for b, _ in all_scores), default=0.0)
            self._last_nan_candidates = nan_candidates
            all_scores.sort(key=lambda x: -x[0])
            # Relative, not absolute: the top birth_limit candidates by
            # score, excluding only genuinely zero-signal ones -- see
            # birth_score_floor's docstring in __init__ for why this
            # replaced a fixed threshold.
            chosen = [(b, uid) for b, uid in all_scores[: self.birth_limit] if b > self.birth_score_floor]
        elif mode == "random":
            picks = self._rng.sample(dormant, min(self.birth_limit, len(dormant)))
            chosen = [(None, uid) for uid in picks]
        else:
            raise ValueError(f"unknown birth mode {mode!r} (expected 'heuristic' or 'random')")
        if frozen:
            chosen = []

        births = []
        for score, uid in chosen:
            t = self.units[uid]
            t.state = LifecycleState.PROTECTED
            t.steps_since_birth = 0
            self._set_probe(uid, False)
            self._set_logit(uid, self.birth_init_logit)
            t.ramp_remaining = self.birth_ramp_steps
            self._set_scale(uid, 0.0 if self.birth_ramp_steps > 0 else 1.0)
            if uid[0] == "layer":
                # The layer's live heads/FFN groups had no effect until now,
                # so their age restarts: no death on an importance EMA that
                # has only ever seen zero gradient.
                for child, ct in self.units.items():
                    if child[0] != "layer" and child[1] == uid[1] and ct.state == LifecycleState.MATURE:
                        ct.state = LifecycleState.PROTECTED
                        ct.steps_since_birth = 0
                        self._death_streak[child] = 0
            births.append(uid)
            self.events.append({"step": step, "unit": uid, "event": "birth", "score": score})
        return births, plateau

    def ramps_fit(self, steps_remaining: int) -> bool:
        """True when a birth or death started now finishes its ramp before
        training ends. A ramp starts at the next record_step, so it needs
        steps_remaining >= its length. Attempt 5 (RESEARCH_LOG.md
        2026-10-13) ran its last check on the final step: births and deaths
        there kept their old gates, so architecture.json (gates) disagreed
        with state_counts (states). The training loop skips update() when
        this is False."""
        return steps_remaining >= max(self.birth_ramp_steps, self.death_ramp_steps)

    def update(self, step: int, mode: str = "heuristic", lr_fraction: Optional[float] = None) -> dict:
        """Call every growth_interval steps from the training loop. `mode` is
        'heuristic' (the real B_i-ranked controller) or 'random' (same
        timing/budget, uniform-random candidate selection) -- proposal §11's
        two growth conditions. The third required condition, "no growth", is
        simply never calling update() at all. `lr_fraction` is the current
        LR / peak LR; None means unknown, and nothing is frozen."""
        deaths_frozen = lr_fraction is not None and lr_fraction < self.death_freeze_lr_fraction
        births_frozen = lr_fraction is not None and lr_fraction < self.birth_freeze_lr_fraction
        deaths = self._death_pass(step, frozen=deaths_frozen)
        self._promote_pass()
        births, plateau = self._birth_pass(step, mode, frozen=births_frozen)
        return {
            "step": step,
            "births": births,
            "deaths": deaths,
            "deaths_frozen": deaths_frozen,
            "births_frozen": births_frozen,
            "plateau": plateau,
            "max_dormant_birth_score": self._last_max_birth_score,
            "birth_score_floor": self.birth_score_floor,
            "nan_candidates": self._last_nan_candidates,
            "below_floor_count": self._last_below_floor_count,
            "death_eligible_pool": self._last_death_eligible_pool,
            "max_death_streak": self._last_max_death_streak,
            "death_criterion": self.death_criterion,
            "gate_spread": self._last_gate_spread,
            "importance_spread": self._last_importance_spread,
            "in_inactive_layers": sum(1 for uid, t in self.units.items() if t.state != LifecycleState.DEAD and self._in_inactive_layer(uid)),
            "ramping": sum(1 for t in self.units.values() if t.ramp_remaining > 0 and t.state != LifecycleState.DEAD),
            "dying": sum(1 for t in self.units.values() if t.ramp_remaining > 0 and t.state == LifecycleState.DEAD),
            "state_counts": self.state_counts(),
        }

    def state_counts(self) -> dict:
        counts: dict[str, int] = {}
        for t in self.units.values():
            counts[t.state.value] = counts.get(t.state.value, 0) + 1
        return counts

    def events_json(self) -> list:
        return [{**e, "unit": list(e["unit"])} for e in self.events]


__all__ = ["LifecycleState", "UnitTelemetry", "StructuralController"]
