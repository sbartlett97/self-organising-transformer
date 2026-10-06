from __future__ import annotations

import random
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
        death_threshold: float = 0.5,
        death_patience: int = 3,
        ema_decay: float = 0.98,
        birth_weight_gradient: float = 0.5,
        birth_weight_contribution: float = 0.3,
        birth_weight_plateau: float = 0.2,
        # contribution_ema's ceiling is sigmoid(dormant_floor_logit)=~0.018
        # regardless of model/dataset scale (it's a function of these
        # hyperparameters, not of training progress), so birth_score_threshold
        # has to be picked relative to that ceiling, not just "scale will fix
        # it". 0.01 was confirmed empirically (not guessed) against a local
        # smoke test where achieved scores topped out around 0.003-0.014 --
        # expect this to need re-tuning once real-scale dynamics are observed
        # (staged rollout step 2 exists specifically to check this).
        birth_score_threshold: float = 0.01,
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
        self.death_threshold = death_threshold
        self.death_patience = death_patience
        self.ema_decay = ema_decay
        self.w_g = birth_weight_gradient
        self.w_c = birth_weight_contribution
        self.w_p = birth_weight_plateau
        self.birth_score_threshold = birth_score_threshold
        self.dormant_floor_logit = dormant_floor_logit
        self.birth_init_logit = birth_init_logit
        self.dead_logit = dead_logit
        self._rng = random.Random(seed)

        self.units: dict[tuple, UnitTelemetry] = {}
        self._loss_history: list[float] = []
        self._train_loss_ema: Optional[float] = None
        self._last_max_birth_score: float = 0.0
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
                t.gradient_ema = self.ema_decay * t.gradient_ema + (1 - self.ema_decay) * abs(g)
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
        gradually."""
        eligible = []
        for uid, t in self.units.items():
            if t.state != LifecycleState.MATURE or t.steps_since_birth < self.minimum_age:
                continue
            gate_val = torch.sigmoid(torch.tensor(self._get_logit(uid))).item()
            death_score = 1.0 - gate_val
            if death_score > self.death_threshold:
                self._death_streak[uid] = self._death_streak.get(uid, 0) + 1
            else:
                self._death_streak[uid] = 0
            if self._death_streak.get(uid, 0) >= self.death_patience:
                eligible.append((death_score, uid))
        eligible.sort(key=lambda x: -x[0])

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
            scored = []
            all_scores = []
            for uid in dormant:
                t = self.units[uid]
                b = self.w_g * t.gradient_ema + self.w_c * t.contribution_ema + self.w_p * plateau
                all_scores.append(b)
                if b >= self.birth_score_threshold:
                    scored.append((b, uid))
            self._last_max_birth_score = max(all_scores) if all_scores else 0.0
            scored.sort(key=lambda x: -x[0])
            chosen = scored[: self.birth_limit]
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
            "birth_score_threshold": self.birth_score_threshold,
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
