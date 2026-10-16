"""Dormant and dead units must have exactly zero effect and zero gradient.

Only the active structure may affect the loss (RESEARCH_LOG.md 2026-10-15).
This builds a PlasticTransformer with a dormant layer, a dead layer, dormant
heads and FFN groups in live layers, and skips on, then checks:

1. the output does not change when inactive weights change;
2. inactive weights, logits and skip edges get exactly zero gradient;
3. a dormant unit's probe gradient is dL/dg (finite difference);
4. the cost terms put no gradient on the probes;
5. a real AdamW step (weight decay, momentum) leaves inactive weights and
   probes bit-identical;
6. a physical CompactTransformer of the present structure gives the same logits;
7. a controller-driven run keeps scales and probes in step with states;
8. a checkpoint saved before scales existed loads with gate = sigmoid(logit);
9. a new checkpoint keeps its scales and probe masks.

Run: python tests/zero_effect_test.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from safetensors.torch import load_file, save_file

from compact_model import build_from_plastic
from plastic_model import PlasticConfig, PlasticTransformer
from plasticity import LifecycleState, StructuralController

torch.manual_seed(0)
cfg = PlasticConfig(vocab_size=101, max_position_embeddings=16, max_layers=5, d_model=32, max_heads=4,
                    max_ffn_dim=48, ffn_groups=6, skip_span=2, dropout=0.0, architecture_mode="sot",
                    seed_fraction=0.5, minimum_age=10)
x = torch.randint(0, cfg.vocab_size, (2, 16))


def make_model():
    """Layer 1 dormant, layer 3 dead; in live layers 0, 2, 4: some heads and
    FFN groups dormant or dead. Gate logits random so gates are not uniform."""
    m = PlasticTransformer(cfg).eval()
    with torch.no_grad():
        for b in m.blocks:
            b.layer_logit.normal_(0, 1); b.attn.head_logits.normal_(0, 1)
            b.ffn.group_logits.normal_(0, 1); b.skip_logits.normal_(0, 1)
        m.layer_scale[1] = 0; m.layer_probe_mask[1] = 1      # dormant layer
        m.layer_scale[3] = 0                                 # dead layer
        m.head_scale[0, 1] = 0; m.head_probe_mask[0, 1] = 1  # dormant head
        m.head_scale[2, 3] = 0                               # dead head
        m.group_scale[0, [0, 4]] = 0; m.group_probe_mask[0, [0, 4]] = 1
        m.group_scale[4, 2] = 0
        m.head_scale[4, 0] = 0.5                             # mid-ramp: present
    return m


model = make_model()
ref = model(x)["logits"].detach().clone()

# ---- test 1: changing inactive weights changes nothing ----
masks = model.inactive_masks()
assert masks, "the test structure must have inactive elements"
with torch.no_grad():
    for p, m in masks:
        p[m] = torch.randn_like(p)[m] * 100.0
out1 = model(x)["logits"]
print(f"test1: {sum(int(m.sum()) for _, m in masks)} inactive elements randomised; max |diff| = {(out1 - ref).abs().max():.3e}")
assert torch.equal(out1, ref)
print("test1: PASS (output bit-identical after randomising every inactive weight)")

# ---- test 2: inactive elements get exactly zero gradient ----
model.zero_grad(set_to_none=True)
model(x, labels=x)["loss"].backward()
for p, m in masks:
    assert p.grad is None or torch.count_nonzero(p.grad[m]) == 0, "inactive element got a gradient"
live_grad = model.blocks[0].attn.qkv.weight.grad
assert live_grad is not None and torch.count_nonzero(live_grad) > 0, "live weights must still train"
print("test2: PASS (zero gradient on every inactive weight, logit and skip edge; live weights train)")

# ---- test 3: the probe gradient is dL/dg of the dormant unit ----
# A fresh model: test 1 scaled the inactive weights by 100, which makes the
# loss far from linear in the probe even at eps = 1e-3.
# float64, eps 1e-2: RMSNorm and softmax compute in float32 even here, so a small eps is rounding noise (checked: eps 3e-3..3e-2 agree with the probe to ~1e-4).
model = make_model().double()
model.zero_grad(set_to_none=True)
model(x, labels=x)["loss"].backward()
for name, idx in [("head_probe", (0, 1)), ("group_probe", (0, 4)), ("layer_probe", (1,))]:
    probe = getattr(model, name)
    g = float(probe.grad[idx])
    eps = 1e-2
    with torch.no_grad():
        probe[idx] = eps; lp = float(model(x, labels=x)["loss"])
        probe[idx] = -eps; lm = float(model(x, labels=x)["loss"])
        probe[idx] = 0.0
    fd = (lp - lm) / (2 * eps)
    print(f"test3: {name}{idx} grad={g:.5e} finite-diff={fd:.5e}")
    assert g != 0.0 and abs(g - fd) <= 5e-3 * abs(fd)
for name, idx in [("head_probe", (2, 3)), ("layer_probe", (3,))]:
    assert float(getattr(model, name).grad[idx]) == 0.0, "dead units have no probe"
print("test3: PASS (probe gradient = dL/dg for dormant units; dead units have none)")

# ---- test 4: cost terms put no gradient on the probes ----
model.zero_grad(set_to_none=True)
(model.architecture_cost() + model.sparsity_loss()).backward()
assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.probe_parameters())
print("test4: PASS (birth signal is the task loss only)")

# ---- test 5: a real AdamW step leaves inactive weights and probes bit-identical ----
model = make_model().train()
before = {n: p.detach().clone() for n, p in model.named_parameters()}
probe_ids = {id(p) for p in model.probe_parameters()}
main = [p for p in model.parameters() if id(p) not in probe_ids]
opt = torch.optim.AdamW([{"params": main}, {"params": model.probe_parameters(), "lr": 0.0, "weight_decay": 0.0}],
                        lr=1e-2, weight_decay=0.1, betas=(0.9, 0.95))
masks = model.inactive_masks()
for _ in range(3):
    opt.zero_grad(set_to_none=True)
    model(x, labels=x)["loss"].backward()
    torch.nn.utils.clip_grad_norm_(main, 1.0)
    snap = model.snapshot_inactive()
    opt.step()
    model.restore_inactive(snap)
named = dict(model.named_parameters())
for p, m in masks:
    n = next(k for k, v in named.items() if v is p)
    assert torch.equal(p.detach()[m], before[n][m]), f"inactive part of {n} moved"
assert all(torch.count_nonzero(p) == 0 for p in model.probe_parameters()), "probes must stay exactly 0"
assert not torch.equal(named["blocks.0.attn.qkv.weight"], before["blocks.0.attn.qkv.weight"]), "live weights must move"
print("test5: PASS (inactive weights and probes bit-identical after 3 AdamW steps with weight decay)")

# ---- test 6: the physical model of the present structure gives the same logits ----
model.eval()
compact = build_from_plastic(model).eval()
n_present = sum(1 for i in range(cfg.max_layers) if model.layer_scale[i] > 0)
assert len(compact.blocks) == n_present == 3
with torch.no_grad():
    a, b = model(x)["logits"], compact(x)["logits"]
diff = float((a - b).abs().max())
print(f"test6: {n_present} present layers; max |logit diff| supernet vs physical = {diff:.3e}")
assert diff < 1e-4
print("test6: PASS (supernet output = physical model of the present structure)")

# ---- test 7: a controller-driven run keeps scales and probes in step with states ----
torch.manual_seed(1)
m7 = PlasticTransformer(cfg).train()
ctl = StructuralController(m7, birth_limit=2, death_limit=1, minimum_age=5, death_relative_margin=0.9,
                           death_patience=1, birth_ramp_steps=3, death_ramp_steps=3, seed=0)
ctl.seed_architecture()
opt7 = torch.optim.SGD([p for p in m7.parameters() if id(p) not in {id(q) for q in m7.probe_parameters()}], lr=0.01)
for step in range(1, 61):
    opt7.zero_grad(set_to_none=True); m7.zero_grad(set_to_none=True)
    m7(x, labels=x)["loss"].backward()
    ctl.record_step(train_loss=1.0)
    opt7.step()
    if step % 5 == 0:
        ctl.update(step)
    for uid, t in ctl.units.items():
        scale = ctl._get_scale(uid)
        probe_on = float(ctl._buffer(uid, "probe_mask")[0][ctl._buffer(uid, "probe_mask")[1]]) == 1.0
        if t.state == LifecycleState.DORMANT:
            assert scale == 0.0 and probe_on, (uid, scale)
        elif t.state == LifecycleState.DEAD:
            assert not probe_on and (scale == 0.0 or t.ramp_remaining > 0), (uid, scale)
        else:
            assert not probe_on and (scale == 1.0 or t.ramp_remaining > 0), (uid, scale)
n_birth = sum(e["event"] == "birth" for e in ctl.events); n_death = sum(e["event"] == "death" for e in ctl.events)
print(f"test7: {n_birth} births, {n_death} deaths over 60 steps; states and scales agree at every step")
assert n_birth > 0 and n_death > 0
print("test7: PASS (controller keeps scale/probe in step with lifecycle state)")

# ---- test 8: an old checkpoint (no scale buffers or probes) loads with gate = sigmoid(logit) ----
old = PlasticTransformer(cfg).eval()
with tempfile.TemporaryDirectory() as d:
    old.save_pretrained(d, safe_serialization=True)
    f = Path(d) / "model.safetensors"
    sd = load_file(f)
    sd = {k: v for k, v in sd.items() if "scale" not in k and "probe" not in k}
    save_file(sd, f, metadata={"format": "pt"})
    loaded = PlasticTransformer.from_pretrained(d).eval()
assert torch.all(loaded.layer_scale == 1) and torch.all(loaded.head_scale == 1) and torch.all(loaded.group_scale == 1)
assert all(torch.count_nonzero(p) == 0 for p in loaded.probe_parameters())
with torch.no_grad():
    assert torch.allclose(loaded(x)["logits"], old(x)["logits"], atol=1e-6)
print("test8: PASS (old checkpoint loads unchanged: scales 1, probes 0)")

# ---- test 9: a new checkpoint keeps its structure on reload ----
m9 = make_model().eval()
with tempfile.TemporaryDirectory() as d:
    m9.save_pretrained(d, safe_serialization=True)
    r9 = PlasticTransformer.from_pretrained(d).eval()
for name in ("layer_scale", "head_scale", "group_scale", "layer_probe_mask", "head_probe_mask", "group_probe_mask"):
    assert torch.equal(getattr(r9, name), getattr(m9, name)), name
with torch.no_grad():
    assert torch.equal(r9(x)["logits"], m9(x)["logits"])
print("test9: PASS (scales and probe masks survive save/load; output bit-identical)")

print("\nALL TESTS PASSED")
