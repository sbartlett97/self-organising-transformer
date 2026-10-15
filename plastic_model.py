from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

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
    """Phase 2 model: the Phase 1 supernet plus a per-unit scale that the
    StructuralController sets. A unit's gate is scale * sigmoid(logit).

    Scale 0 (DORMANT, DEAD, and anything inside a layer with scale 0) is an
    exact zero: the unit adds nothing to the output, and its weights and
    logit get exactly zero gradient. Only the active structure affects the
    loss. Attempts 1-8 used small gates (~0.04) in place of zero, and the
    model came to depend on them: forcing them off took attempt 8 from 26.67
    to 132 val ppl (RESEARCH_LOG.md 2026-10-14).

    Birth signal: a dormant unit's gate also gets a zero-valued probe. The
    probe adds nothing to the output, but its gradient is dL/dg for that
    unit. The probe is never applied to any weight: the training loop puts
    probes in an optimizer group with lr 0, out of gradient clipping.

    Skip edge i <- i-s is scaled by both layers' scales, so a skip to or
    from an absent layer is off, as in CompactTransformer.

    Weight decay and Adam momentum still move parameters with zero gradient.
    snapshot_inactive() / restore_inactive() around optimizer.step() keep
    inactive weights bit-identical.

    The scale and probe-mask buffers are saved with the checkpoint. They
    default to 1 and 0, so a checkpoint saved before they existed loads with
    its original behaviour (gate = sigmoid(logit))."""

    config_class = PlasticConfig

    def __init__(self, config: PlasticConfig):
        super().__init__(config)
        L, H, G = config.max_layers, config.max_heads, config.ffn_groups
        self.register_buffer("layer_scale", torch.ones(L))
        self.register_buffer("head_scale", torch.ones(L, H))
        self.register_buffer("group_scale", torch.ones(L, G))
        self.register_buffer("layer_probe_mask", torch.zeros(L))
        self.register_buffer("head_probe_mask", torch.zeros(L, H))
        self.register_buffer("group_probe_mask", torch.zeros(L, G))
        self.layer_probe = nn.Parameter(torch.zeros(L))
        self.head_probe = nn.Parameter(torch.zeros(L, H))
        self.group_probe = nn.Parameter(torch.zeros(L, G))

    def _init_weights(self, module):
        # post_init() must not touch the probes (they must stay exactly 0).
        if module is self:
            return
        super()._init_weights(module)

    def probe_parameters(self) -> list[nn.Parameter]:
        return [self.layer_probe, self.head_probe, self.group_probe]

    def _skip_scale(self) -> torch.Tensor:
        """[L, skip_span]: layer_scale[i] * layer_scale[i-s], 0 where i-s < 0."""
        L, span = len(self.blocks), self.config.skip_span
        out = torch.zeros(L, span, device=self.layer_scale.device, dtype=self.layer_scale.dtype)
        for s in range(1, span + 1):
            out[s:, s - 1] = self.layer_scale[s:] * self.layer_scale[:-s]
        return out

    def _gates(self):
        """Effective gates (scale * sigmoid(logit)), no probes. architecture_cost,
        sparsity_loss and export_architecture use these, so the cost terms
        add nothing to a probe's gradient: the birth signal is the task loss only."""
        layer, heads, groups, skips = super()._gates()
        if self.config.architecture_mode == "dense":
            return layer, heads, groups, skips
        if skips.numel():
            skips = skips * self._skip_scale()
        return layer * self.layer_scale, heads * self.head_scale, groups * self.group_scale, skips

    def _forward_gates(self):
        layer, heads, groups, skips = self._gates()
        if self.config.architecture_mode == "dense":
            return layer, heads, groups, skips
        return (layer + self.layer_probe_mask * self.layer_probe,
                heads + self.head_probe_mask * self.head_probe,
                groups + self.group_probe_mask * self.group_probe,
                skips)

    def inactive_masks(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """(param, bool mask) for every element that belongs to a unit with
        scale 0, or to a skip edge with scale 0. These elements get exactly
        zero gradient."""
        cfg = self.config
        H, dh, d = cfg.max_heads, cfg.d_model // cfg.max_heads, cfg.d_model
        G, gs, F = cfg.ffn_groups, cfg.max_ffn_dim // cfg.ffn_groups, cfg.max_ffn_dim
        skip_off = self._skip_scale() == 0 if cfg.skip_span > 0 else None
        out = []
        for i, b in enumerate(self.blocks):
            if self.layer_scale[i] == 0:
                for p in b.parameters():
                    out.append((p, torch.ones_like(p, dtype=torch.bool)))
                continue
            head_off = self.head_scale[i] == 0
            if head_off.any():
                out.append((b.attn.qkv.weight, head_off.view(1, H, 1, 1).expand(3, H, dh, d).reshape(3 * d, d)))
                out.append((b.attn.out.weight, head_off.view(1, H, 1).expand(d, H, dh).reshape(d, d)))
                out.append((b.attn.head_logits, head_off))
            group_off = self.group_scale[i] == 0
            if group_off.any():
                channel_off = group_off.repeat_interleave(gs)
                out.append((b.ffn.up.weight, channel_off.view(1, F, 1).expand(2, F, d).reshape(2 * F, d)))
                out.append((b.ffn.down.weight, channel_off.view(1, F).expand(d, F)))
                out.append((b.ffn.group_logits, group_off))
            if skip_off is not None and skip_off[i].any():
                out.append((b.skip_logits, skip_off[i]))
        return out

    @torch.no_grad()
    def snapshot_inactive(self) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return [(p, m, p.detach()[m].clone()) for p, m in self.inactive_masks()]

    @torch.no_grad()
    def restore_inactive(self, snapshot) -> None:
        for p, m, values in snapshot:
            p[m] = values


__all__ = ["PlasticConfig", "PlasticTransformer"]
