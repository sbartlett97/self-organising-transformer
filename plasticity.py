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


class StructuralController:
    """Implements proposal §11's discrete birth/death lifecycle
    (DORMANT -> PROTECTED -> MATURE -> DEAD) on top of a PlasticTransformer's
    continuous gate substrate. Mutates gate logits directly -- birth resets a
    logit to a normal active init, death hard-clamps it to a strongly negative
    value -- but never touches the forward pass itself; see the Phase 2 plan's
    "hybrid continuous substrate, discrete lifecycle overlay" design note for
    why (early phases may use cost proxies rather than real sparse compute,
    proposal §9).

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
        dead_logit: float = -8.0,
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
        self.dead_logit = dead_logit
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

    def seed_architecture(self, active_fraction: Optional[float] = None) -> None:
        """Apply Phase 2's small-seed split: a minority of units start MATURE
        (active init), the rest DORMANT (suppressed floor). Call once, right
        after constructing the model and before training starts."""
        fraction = active_fraction if active_fraction is not None else self.model.config.seed_fraction
        for uid, t in self.units.items():
            if self._rng.random() < fraction:
                t.state = LifecycleState.MATURE
                self._set_logit(uid, self.birth_init_logit)
            else:
                t.state = LifecycleState.DORMANT
                self._set_logit(uid, self.dormant_floor_logit)

    def record_step(self, train_loss: Optional[float] = None) -> None:
        """Call every training step, after optimizer.zero_grad's companion
        backward pass has populated (and been unscaled into) param.grad, and
        before the next zero_grad call. Cheap: no model forward pass.

        `train_loss` should be this step's training loss, not a sparse
        held-out eval -- it needs to update every step (growth_interval is
        typically much more frequent than eval_every) to give the plateau
        signal anything to work with."""
        for uid, t in self.units.items():
            param, idx = self._logit_param(uid)
            grad = param.grad
            if grad is not None:
                g = float(grad.item()) if idx is None else float(grad[idx].item())
                # An EMA can never recover from a single non-finite update --
                # ema_decay*NaN + ... is NaN forever after, permanently
                # poisoning that unit's birth score (and, worse, Python's
                # sort() becomes unreliable on a list containing NaN, which
                # can crowd a genuinely-good candidate out of the top-K
                # selection even though the NaN one correctly fails the
                # floor check itself -- observed directly: growth froze for
                # 1250+ steps with max_dormant_birth_score stuck at NaN).
                # Skip the update rather than let one bad gradient (fp16
                # overflow slipping through before the scaler catches it,
                # a rare spike, whatever) take out a unit for the rest of
                # training.
                gate_val = torch.sigmoid(torch.tensor(self._get_logit(uid))).item()
                if math.isfinite(g):
                    t.gradient_ema = self.ema_decay * t.gradient_ema + (1 - self.ema_decay) * abs(g)
                    # Taylor importance |g * dL/dg|. dL/dlogit = dL/dg * g(1-g),
                    # so g * dL/dg = dL/dlogit / (1-g): no extra backward pass.
                    taylor = abs(g / max(1.0 - gate_val, 1e-6))
                    t.importance_ema = self.ema_decay * t.importance_ema + (1 - self.ema_decay) * taylor
            gate_val = torch.sigmoid(torch.tensor(self._get_logit(uid))).item()
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

    def _death_pass(self, step: int) -> list:
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
            if t.state == LifecycleState.MATURE:
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
            if t.state != LifecycleState.MATURE or t.steps_since_birth < self.minimum_age:
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
        for death_score, uid in eligible[: self.death_limit]:
            t = self.units[uid]
            t.state = LifecycleState.DEAD
            self._set_logit(uid, self.dead_logit)
            deaths.append(uid)
            self.events.append({"step": step, "unit": uid, "event": "death", "score": death_score})
        return deaths

    def _promote_pass(self) -> None:
        for t in self.units.values():
            if t.state == LifecycleState.PROTECTED and t.steps_since_birth >= self.minimum_age:
                t.state = LifecycleState.MATURE

    def _birth_pass(self, step: int, mode: str) -> tuple[list, float]:
        plateau = self._plateau_score()
        dormant = [uid for uid, t in self.units.items() if t.state == LifecycleState.DORMANT]

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

        births = []
        for score, uid in chosen:
            t = self.units[uid]
            t.state = LifecycleState.PROTECTED
            t.steps_since_birth = 0
            self._set_logit(uid, self.birth_init_logit)
            births.append(uid)
            self.events.append({"step": step, "unit": uid, "event": "birth", "score": score})
        return births, plateau

    def update(self, step: int, mode: str = "heuristic") -> dict:
        """Call every growth_interval steps from the training loop. `mode` is
        'heuristic' (the real B_i-ranked controller) or 'random' (same
        timing/budget, uniform-random candidate selection) -- proposal §11's
        two growth conditions. The third required condition, "no growth", is
        simply never calling update() at all."""
        deaths = self._death_pass(step)
        self._promote_pass()
        births, plateau = self._birth_pass(step, mode)
        return {
            "step": step,
            "births": births,
            "deaths": deaths,
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
