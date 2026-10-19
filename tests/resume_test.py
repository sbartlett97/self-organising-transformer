"""Check train_phase2 --resume_from: a run that stops and resumes from its
training_state.pt ends bit-identical to a run with no stop. The check
covers weights, buffers, events, growth log and metrics. The stop comes
after an epoch wrap, and a second wrap comes after the resume. A resume
with a changed config must stop.

Run: python tests/resume_test.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from safetensors.torch import load_file

import train_phase2
from train import PackedTokens
from train_phase2 import Phase2Config

SEQ = 32
rng = np.random.default_rng(0)
# 120 sequences / batch 4 = 30 batches per epoch, 2 per step: the epoch wraps at steps 15, 30 and 45.
TRAIN = PackedTokens(rng.integers(0, 50257, 120 * SEQ + 1).tolist(), SEQ)
VAL = PackedTokens(rng.integers(0, 50257, 20 * SEQ + 1).tolist(), SEQ)
train_phase2.load_tokens = lambda cfg, tok: (TRAIN, VAL)


def config(out_dir, **kw):
    base = dict(
        output_dir=str(out_dir), device="cpu", use_trackio=False, seq_len=SEQ,
        d_model=32, max_heads=4, max_ffn_dim=64, ffn_groups=4, max_layers=3,
        batch_size=4, grad_accum=2, max_steps=60, eval_every=20, save_every=20, log_every=5,
        warmup_steps=5, gate_warmup_steps=10, growth_interval=5, birth_limit=2,
        minimum_age=5, death_patience=1, death_relative_margin=0.9,
        birth_ramp_steps=3, death_ramp_steps=3, lr_schedule="wsd", lr_decay_start=30,
        structure_freeze_step=50, birth_score_norm="flops", model_type="sot",
    )
    base.update(kw)
    return Phase2Config(**base)


class Stop(Exception):
    pass


def run_with_stop(cfg, stop_after):
    """Run until `stop_after` optimizer steps are done, then raise, as a crash would."""
    real = torch.nn.utils.clip_grad_norm_
    calls = {"n": 0}

    def clip(*a, **k):
        if calls["n"] == stop_after:
            raise Stop
        calls["n"] += 1
        return real(*a, **k)

    torch.nn.utils.clip_grad_norm_ = clip
    try:
        train_phase2.train(cfg)
    except Stop:
        return
    finally:
        torch.nn.utils.clip_grad_norm_ = real
    raise AssertionError("the run did not stop")


def final(out_dir):
    d = Path(out_dir) / "final"
    return (load_file(d / "model.safetensors"),
            *(json.loads((d / f).read_text()) for f in ("events.json", "growth_log.json", "metrics.json")))


with tempfile.TemporaryDirectory() as tmp:
    full, part = Path(tmp) / "full", Path(tmp) / "part"
    train_phase2.train(config(full))
    run_with_stop(config(part), stop_after=27)
    saved = torch.load(part / "training_state.pt", weights_only=False)
    assert saved["global_step"] == 20, saved["global_step"]
    assert saved["iter_seed"] != 42, "the saved position must be after the epoch wrap at step 15"
    train_phase2.train(config(part, resume_from=str(part)))

    w_full, ev_full, gl_full, m_full = final(full)
    w_part, ev_part, gl_part, m_part = final(part)

    # ---- test 1: weights and buffers are bit-identical ----
    assert w_full.keys() == w_part.keys()
    for k in w_full:
        assert torch.equal(w_full[k], w_part[k]), k
    print(f"test 1: {len(w_full)} tensors bit-identical")

    # ---- test 2: events, growth log and metrics are identical ----
    births = sum(e["event"] == "birth" for e in ev_full)
    deaths = sum(e["event"] == "death" for e in ev_full)
    assert births > 0 and deaths > 0, (births, deaths)
    assert any(e["step"] > 20 for e in ev_full), "the test needs events after the resume step"
    assert ev_full == ev_part
    assert gl_full == gl_part
    assert m_full == m_part
    print(f"test 2: {births} births, {deaths} deaths, {len(gl_full)} growth checks, {len(m_full)} metric records identical")

    # ---- test 3: a resume with a changed config stops ----
    try:
        train_phase2.train(config(part, resume_from=str(part), learning_rate=1e-3))
    except ValueError as e:
        assert "learning_rate" in str(e), e
        print("test 3: changed config refused")
    else:
        raise AssertionError("a changed config must be refused")

print("all resume tests pass")
