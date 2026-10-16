"""Real compute accounting: FLOPs and parameters of the structure that is present.

`model.architecture_cost()` is a gate-weighted proxy, not compute (RESEARCH_LOG.md,
2026-10-14 compute-accounting entry).
This module counts the matmul FLOPs and the parameters a structure would have
as physical tensors ("realisable"). The gated supernet always runs its full
dense shape ("executed"), whatever its gates are.

A structure is a list with one dict per present layer: {"heads": h,
"ffn_groups": g}. Layers that are not present are not in the list.

Forward FLOPs per token (2 per multiply-add), per present layer:
  attention  QKV 2*d*3*h*dh + scores 2*T*h*dh + mix 2*T*h*dh + out 2*h*dh*d
  FFN        up 2*d*2*g*s (SwiGLU) + down 2*g*s*d
plus lm_head 2*d*V once. T is the full seq_len: the code builds the whole
T x T score matrix and masks it. Norms, softmax, skips and embedding lookups
are elementwise and not counted. Training FLOPs = 3 x forward.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Shape:
    d_model: int
    head_dim: int
    group_size: int
    vocab_size: int
    seq_len: int
    max_position_embeddings: int
    max_layers: int
    max_heads: int
    ffn_groups: int

    @classmethod
    def from_config(cls, config, seq_len: int) -> "Shape":
        return cls(
            d_model=config.d_model,
            head_dim=config.d_model // config.max_heads,
            group_size=config.max_ffn_dim // config.ffn_groups,
            vocab_size=config.vocab_size,
            seq_len=seq_len,
            max_position_embeddings=config.max_position_embeddings,
            max_layers=config.max_layers,
            max_heads=config.max_heads,
            ffn_groups=config.ffn_groups,
        )

    def dense(self) -> list[dict]:
        return [{"heads": self.max_heads, "ffn_groups": self.ffn_groups} for _ in range(self.max_layers)]


def flops_per_token(layers: list[dict], s: Shape) -> dict:
    """Forward matmul FLOPs per token, split into attention, FFN and lm_head."""
    d, dh, T = s.d_model, s.head_dim, s.seq_len
    attn = ffn = 0
    for layer in layers:
        hd = layer["heads"] * dh
        attn += 2 * d * 3 * hd + 2 * T * hd + 2 * T * hd + 2 * hd * d
        width = layer["ffn_groups"] * s.group_size
        ffn += 2 * d * 2 * width + 2 * width * d
    head = 2 * d * s.vocab_size
    return {"attention": attn, "ffn": ffn, "lm_head": head, "total": attn + ffn + head}


def param_count(layers: list[dict], s: Shape) -> dict:
    """Parameters of the physical model (tied lm_head counted once, no gate logits)."""
    d, dh = s.d_model, s.head_dim
    attn = sum(4 * d * layer["heads"] * dh for layer in layers)
    ffn = sum(3 * d * layer["ffn_groups"] * s.group_size for layer in layers)
    norms = 2 * d * len(layers) + d
    embed = s.vocab_size * d + s.max_position_embeddings * d
    return {"attention": attn, "ffn": ffn, "norms": norms, "embedding": embed,
            "non_embedding": attn + ffn + norms, "total": attn + ffn + norms + embed}


def unit_order(s: Shape) -> list[tuple]:
    """Unit ids in StructuralController._register_units order."""
    order = []
    for li in range(s.max_layers):
        order.append(("layer", li, 0))
        order += [("head", li, hi) for hi in range(s.max_heads)]
        order += [("ffn_group", li, gi) for gi in range(s.ffn_groups)]
    return order


def seed_split(s: Shape, seed: int, seed_fraction: float) -> set[tuple]:
    """The units seed_architecture() makes MATURE: one Random(seed) draw per
    unit, in registration order. The controller's _rng makes no other draw
    before seed_architecture()."""
    rng = random.Random(seed)
    return {uid for uid in unit_order(s) if rng.random() < seed_fraction}


class Replay:
    """Rebuild a Phase 2 run's structure at any step from its seed split and
    events.json. A unit is "alive" (PROTECTED or MATURE) from its birth to its
    death. It is "present" (has real tensors) while alive and also while its
    death ramp runs: deaths decided at step k ramp over steps k+1..k+ramp."""

    def __init__(self, s: Shape, seed: int, seed_fraction: float, events: list[dict], death_ramp_steps: int):
        self.s = s
        self.death_ramp_steps = death_ramp_steps
        self.seed_alive = seed_split(s, seed, seed_fraction)
        self.events = sorted(({**e, "unit": tuple(e["unit"])} for e in events), key=lambda e: e["step"])

    def states(self, step: int) -> dict[tuple, str]:
        """'alive', 'dying', 'dead' or 'dormant' for each unit, after the
        growth check at `step` (events at `step` included)."""
        st = {uid: ("alive" if uid in self.seed_alive else "dormant") for uid in unit_order(self.s)}
        for e in self.events:
            if e["step"] > step:
                break
            if e["event"] == "birth":
                st[e["unit"]] = "alive"
            elif e["event"] == "death":
                ramping = step < e["step"] + self.death_ramp_steps
                st[e["unit"]] = "dying" if ramping else "dead"
        return st

    def counts(self, step: int) -> dict:
        """Counts in growth_log state_counts form. 'alive' = protected + mature;
        a dying unit is already DEAD in the controller."""
        out = {"alive": 0, "dead": 0, "dormant": 0}
        for v in self.states(step).values():
            out["dead" if v == "dying" else v] += 1
        return out

    def structure(self, step: int) -> list[dict]:
        st = self.states(step)
        present = {"alive", "dying"}
        layers = []
        for li in range(self.s.max_layers):
            if st[("layer", li, 0)] not in present:
                continue
            layers.append({
                "index": li,
                "heads": sum(st[("head", li, h)] in present for h in range(self.s.max_heads)),
                "ffn_groups": sum(st[("ffn_group", li, g)] in present for g in range(self.s.ffn_groups)),
            })
        return layers


def training_flops(replay: Replay, max_steps: int, tokens_per_step: int) -> dict:
    """Cumulative training FLOPs (3 x forward) of the realisable structure,
    and of the dense shape, over max_steps. The structure changes only at
    growth checks and when a death ramp ends, so a per-step sum is exact.

    This is a lower bound for the present controller: birth scores read the
    gradient on dormant gates, so dormant units must run. It is the cost a
    birth signal that needs no dormant compute would give."""
    s = replay.s
    dense = 3 * flops_per_token(s.dense(), s)["total"] * tokens_per_step * max_steps
    change_steps = sorted({0, *[e["step"] for e in replay.events],
                           *[e["step"] + replay.death_ramp_steps for e in replay.events if e["event"] == "death"]})
    real = 0
    for i, start in enumerate(change_steps):
        end = change_steps[i + 1] if i + 1 < len(change_steps) else max_steps
        end = min(end, max_steps)
        if end <= start:
            continue
        # Training step n (1-based) runs with the structure after the check at step n-1.
        f = flops_per_token(replay.structure(start), s)["total"]
        real += 3 * f * tokens_per_step * (end - start)
    return {"realisable": real, "dense": dense, "fraction": real / dense}


__all__ = ["Shape", "flops_per_token", "param_count", "unit_order", "seed_split", "Replay", "training_flops"]
