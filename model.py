from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel


class SOTConfig(PretrainedConfig):
    model_type = "self_organizing_transformer"

    def __init__(
        self,
        vocab_size: int = 50257,
        max_position_embeddings: int = 512,
        max_layers: int = 12,
        d_model: int = 384,
        max_heads: int = 6,
        max_ffn_dim: int = 1536,
        ffn_groups: int = 64,
        skip_span: int = 3,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        architecture_mode: str = "sot",
        gate_init_logit: float = -1.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.tie_word_embeddings = True
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.max_layers = max_layers
        self.d_model = d_model
        self.max_heads = max_heads
        self.max_ffn_dim = max_ffn_dim
        self.ffn_groups = ffn_groups
        self.skip_span = skip_span
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.architecture_mode = architecture_mode
        self.gate_init_logit = gate_init_logit


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        x_f = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x_f).to(dtype=x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: SOTConfig):
        super().__init__()
        assert config.d_model % config.max_heads == 0
        self.d_model = config.d_model
        self.max_heads = config.max_heads
        self.head_dim = config.d_model // config.max_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.head_logits = nn.Parameter(torch.ones(config.max_heads) * config.gate_init_logit)
        self.register_buffer(
            "causal",
            torch.tril(torch.ones(config.max_position_embeddings, config.max_position_embeddings, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, head_gate: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.max_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal[:t, :t]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        probs = self.attn_dropout(probs)
        y = probs @ v
        y = y * head_gate.view(1, -1, 1, 1)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.out(y))


class GroupedFFN(nn.Module):
    def __init__(self, config: SOTConfig):
        super().__init__()
        assert config.max_ffn_dim % config.ffn_groups == 0
        self.max_ffn_dim = config.max_ffn_dim
        self.ffn_groups = config.ffn_groups
        self.group_size = config.max_ffn_dim // config.ffn_groups
        self.up = nn.Linear(config.d_model, config.max_ffn_dim * 2, bias=False)
        self.down = nn.Linear(config.max_ffn_dim, config.d_model, bias=False)
        self.group_logits = nn.Parameter(torch.ones(config.ffn_groups) * 1.5)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, group_gate: torch.Tensor) -> torch.Tensor:
        u, v = self.up(x).chunk(2, dim=-1)
        h = F.silu(u) * v
        g = group_gate.repeat_interleave(self.group_size)
        h = h * g.view(1, 1, -1)
        return self.dropout(self.down(h))


class SOTBlock(nn.Module):
    def __init__(self, config: SOTConfig, index: int):
        super().__init__()
        self.index = index
        self.ln1 = RMSNorm(config.d_model, config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.d_model, config.layer_norm_eps)
        self.ffn = GroupedFFN(config)
        self.layer_logit = nn.Parameter(torch.tensor(config.gate_init_logit))
        if config.skip_span > 0:
            self.skip_logits = nn.Parameter(torch.ones(config.skip_span) * config.gate_init_logit)
        else:
            self.skip_logits = None

    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.layer_logit)


class SelfOrganizingTransformer(PreTrainedModel):
    config_class = SOTConfig
    base_model_prefix = "sot"
    supports_gradient_checkpointing = True
    _tied_weights_keys = {"lm_head.weight": "embed.weight"}

    def __init__(self, config: SOTConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([SOTBlock(config, i) for i in range(config.max_layers)])
        self.ln_f = RMSNorm(config.d_model, config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def _gates(self):
        if self.config.architecture_mode == "dense":
            device = self.embed.weight.device
            dtype = self.embed.weight.dtype
            layer = torch.ones(len(self.blocks), device=device, dtype=dtype)
            heads = torch.ones((len(self.blocks), self.config.max_heads), device=device, dtype=dtype)
            groups = torch.ones((len(self.blocks), self.config.ffn_groups), device=device, dtype=dtype)
            skips = torch.zeros((len(self.blocks), self.config.skip_span), device=device, dtype=dtype) if self.config.skip_span > 0 else torch.empty((len(self.blocks), 0), device=device, dtype=dtype)
            return layer, heads, groups, skips
        layer = torch.sigmoid(torch.stack([b.layer_logit for b in self.blocks]))
        heads = torch.stack([torch.sigmoid(b.attn.head_logits) for b in self.blocks])
        groups = torch.stack([torch.sigmoid(b.ffn.group_logits) for b in self.blocks])
        if self.config.skip_span > 0:
            skips = torch.stack([torch.sigmoid(b.skip_logits) for b in self.blocks])
        else:
            skips = torch.empty((len(self.blocks), 0), device=layer.device)
        return layer, heads, groups, skips

    def _forward_gates(self):
        """Gates the forward pass uses. Subclasses may add terms that change
        gradients but not values (Phase 2's birth probes)."""
        return self._gates()

    def architecture_cost(self) -> torch.Tensor:
        layer, heads, groups, skips = self._gates()
        # Differentiable proxy. Normalised so 1.0 is approximately the dense supernet.
        layer_factor = layer.sum() / self.config.max_layers
        head_factor = (heads.mean(dim=-1) * layer).sum() / self.config.max_layers
        ffn_factor = (groups.mean(dim=-1) * layer).sum() / self.config.max_layers
        skip_factor = skips.mean() if skips.numel() else torch.tensor(0.0, device=layer.device)
        # Attention and FFN dominate, with small topology penalty.
        return 0.45 * head_factor + 0.50 * ffn_factor + 0.05 * layer_factor + 0.02 * skip_factor

    def sparsity_loss(self) -> torch.Tensor:
        layer, heads, groups, skips = self._gates()
        return (
            0.5 * layer.mean()
            + 0.35 * heads.mean()
            + 0.15 * groups.mean()
            + (0.05 * skips.mean() if skips.numel() else 0.0)
        )

    def gate_summary(self) -> dict:
        """Aggregate and per-layer gate means, for logging architecture drift during training."""
        layer, heads, groups, skips = self._gates()
        by_layer = []
        for i in range(len(self.blocks)):
            by_layer.append(
                {
                    "layer": float(layer[i].item()),
                    "head": float(heads[i].mean().item()),
                    "ffn": float(groups[i].mean().item()),
                    "skip": float(skips[i].mean().item()) if skips.numel() else 0.0,
                }
            )
        return {
            "layer_gate_mean": float(layer.mean().item()),
            "head_gate_mean": float(heads.mean().item()),
            "ffn_gate_mean": float(groups.mean().item()),
            "skip_gate_mean": float(skips.mean().item()) if skips.numel() else 0.0,
            "gates_by_layer": by_layer,
        }

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        **_: Any,
    ):
        del use_cache
        b, t = input_ids.shape
        if t > self.config.max_position_embeddings:
            raise ValueError(f"Sequence length {t} exceeds max_position_embeddings={self.config.max_position_embeddings}")
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.embed(input_ids) + self.pos_embed(pos))
        previous = []
        layer_gates, head_gates, group_gates, skip_gates = self._forward_gates()

        for i, block in enumerate(self.blocks):
            residual = x
            y = block.ln1(x)
            y = block.attn(y, head_gates[i])
            x = residual + layer_gates[i].view(1, 1, 1) * y
            y = block.ln2(x)
            y = block.ffn(y, group_gates[i])
            x = x + layer_gates[i].view(1, 1, 1) * y

            if i > 0 and self.config.skip_span > 0:
                for s in range(1, min(self.config.skip_span, i) + 1):
                    src = previous[-s]
                    x = x + skip_gates[i, s - 1].view(1, 1, 1) * src / math.sqrt(self.config.skip_span)
            previous.append(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {"loss": loss, "logits": logits}

    def export_architecture(self, threshold: float = 0.5) -> dict:
        layer, heads, groups, skips = self._gates()
        arch = {
            "threshold": threshold,
            "layers": [],
            "d_model": self.config.d_model,
            "max_heads": self.config.max_heads,
            "max_ffn_dim": self.config.max_ffn_dim,
            "ffn_group_size": self.config.max_ffn_dim // self.config.ffn_groups,
        }
        for i in range(self.config.max_layers):
            arch["layers"].append(
                {
                    "index": i,
                    "layer_gate": float(layer[i].item()),
                    "active": bool(layer[i].item() >= threshold),
                    "heads": [int(j) for j, v in enumerate(heads[i].tolist()) if v >= threshold],
                    "ffn_groups": [int(j) for j, v in enumerate(groups[i].tolist()) if v >= threshold],
                    "skip_from": [i - s for s, v in enumerate(skips[i].tolist(), start=1) if v >= threshold and s <= i],
                    "head_gate": heads[i].detach().cpu().tolist(),
                    "ffn_gate": groups[i].detach().cpu().tolist(),
                    "skip_gate": skips[i].detach().cpu().tolist(),
                }
            )
        return arch


__all__ = ["SOTConfig", "SelfOrganizingTransformer"]
