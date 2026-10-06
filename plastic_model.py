from __future__ import annotations

from typing import Any

from model import SOTConfig, SelfOrganizingTransformer


class PlasticConfig(SOTConfig):
    model_type = "self_organizing_transformer_plastic"

    def __init__(
        self,
        seed_fraction: float = 0.33,
        growth_interval: int = 250,
        birth_limit: int = 2,
        minimum_age: int = 1000,
        death_relative_margin: float = 0.5,
        death_patience: int = 3,
        ema_decay: float = 0.98,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.seed_fraction = seed_fraction
        self.growth_interval = growth_interval
        self.birth_limit = birth_limit
        self.minimum_age = minimum_age
        self.death_relative_margin = death_relative_margin
        self.death_patience = death_patience
        self.ema_decay = ema_decay


class PlasticTransformer(SelfOrganizingTransformer):
    """Phase 2 model: identical forward pass and gate substrate to
    SelfOrganizingTransformer (Phase 1) -- discrete birth/death lifecycle state
    lives entirely in plasticity.StructuralController, which mutates this
    model's gate logits directly (birth = reset to active init, death = hard
    clamp). See the Phase 2 plan for the hybrid continuous-substrate /
    discrete-lifecycle-overlay design: this keeps the tensor math unchanged so
    a dormant unit still produces a real (if small) gradient on its own gate
    logit, which the controller needs for its birth-pressure signal.

    Construction leaves every gate at config.gate_init_logit, same as Phase 1
    -- call StructuralController.seed_architecture() right after building the
    model to apply Phase 2's small-seed split (most units DORMANT at a low
    floor, a minority already MATURE)."""

    config_class = PlasticConfig


__all__ = ["PlasticConfig", "PlasticTransformer"]
