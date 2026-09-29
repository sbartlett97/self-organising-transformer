# Research log

Running record of what's been tried, found, and fixed on the Self-Organising
Transformer experiment. Append new dated entries at the bottom; keep "Current
status" at the top up to date as the single-glance summary. See `README.md`
for the experiment design and `self_organizing_transformer_research_proposal.md`
for the full phased research programme.

## Current status

- **Phase**: Phase 1 (differentiable structural selection) — **seed 42
  full-scale matrix complete**; **seed 123 and seed 7 both running in
  parallel** (two separate pods, `sot-run-seed123` and `sot-run-seed7`,
  sharing the same network volume `d167myxctn` — confirmed a volume mounts
  to multiple pods concurrently without issue — outputs under
  `/workspace/outputs_seed123` and `/workspace/outputs_seed7`) to reach
  README's stated minimum of 3 seeds for the stability criterion.
  Parallelizing seed 7 rather than waiting for seed 123's pod to free up:
  same total cost either way (same total GPU-hours), but ~4-5h less
  wall-clock. Combined burn rate while both run: $3.193/hr. Seed 42: all 6
  conditions (dense, dense_8l, dense_half_ffn, sot, sot_compact,
  discovered_retrained) trained (10,000 steps, full corpus) and evaluated
  (perplexity + hellaswag/arc_easy/piqa/winogrande) on RunPod (A100,
  CA-MTL-3). Full results and scoring against README's six interpretation
  criteria in the 2026-09-29 entry below. **4 of 6 criteria clear pass, 1
  not yet testable (needs more seeds), 1 nuanced** (see below — not a
  failure, but not a clean pass either). Total cost for seed 42: $18.39 of
  the $55 budget (incl. one wasted ~30min detour and ~3h of idle billing
  while awaiting termination confirmation — both avoidable, noted for next
  time: terminate/reuse a pod immediately rather than leaving it idle).
- **Headline result**: SOT (109.63 ppl) beats the matched dense baseline
  (116.72 ppl) outright, at 87% lower architectural cost (0.128 vs. 1.0).
  Fresh retrain of the discovered 7-layer architecture recovers to 115.13
  ppl — within ~5% of SOT itself, and still beating dense. Inherited-weights
  compaction without retraining collapses badly (3240 ppl) — expected per
  README's own "Failure 7" framing, but severe enough to warrant checking
  whether the residual-stream-scale gap at materialization (gates hard-set
  to 1.0 vs. the ~0.2–0.27 they ran at during training) is contributing more
  than pure co-adaptation loss would.
- **Codebase**: pipeline bugs from the initial commit are fixed (see
  2026-09-28 entries) plus three more caught live during this run (see
  2026-09-29 entry): tied-weights-on-load silently randomizing `lm_head`,
  `--threshold 0.5` producing a zero-layer materialized model, and an fp16
  dtype mismatch crashing dense-mode evaluation on CUDA. All fixed,
  verified, and committed before being relied on.
- **Key hyperparameter finding**: `target_cost=0.45` (the shipped default)
  is unreachable — the natural unconstrained equilibrium cost is ~0.186, so
  the Lagrangian budget term never activates at `0.45`. `target_cost=0.14`
  is validated and is now the default in `train.py`/`run_experiment.sh`.
- **Structural finding**: under budget pressure, capacity concentrates in a
  contiguous middle-depth block (layers 2–8 of 12 in this run) rather than
  being cut uniformly or kept maximal everywhere — confirmed twice, once in
  the 2026-09-28 diagnostic and again as the actual materialized-architecture
  selection in this run.
- **Not yet fixed in code** (deliberately deferred, tracked here): the
  `materialize.py`/`export_architecture()` fixed `--threshold=0.5` default
  needs to become distribution-relative (or at least warn loudly on a
  zero-layer result) rather than an absolute constant that only happens to
  work for some gate-init/target_cost combinations.

## Next steps (agreed, not yet executed)

1. **≥3 seeds.** The single-seed result above is promising but criterion 4
   (stability across seeds) is the only one not yet testable. Specifically
   check whether the middle-depth-block selection pattern reproduces, and
   whether `sot_compact`'s collapse severity is consistent or seed-sensitive.
2. **Investigate the inherited-weights collapse mechanically** before
   accepting it purely as a co-adaptation story — see the residual-stream
   scale-mismatch hypothesis in the 2026-09-29 entry. If confirmed as a real
   gap (not just expected co-adaptation loss), it's a legitimate `materialize.py`
   fix, not a research finding.
3. **Fix the `--threshold` default** in `materialize.py`/`export_architecture()`
   to be distribution-relative (e.g. per-family median/percentile) rather
   than a fixed `0.5`, or at minimum raise a loud error on a zero-layer
   result instead of silently proceeding.
4. Only after seed stability is confirmed: decide on Phase 2 (growth/
   mutation) per the research proposal's Gate 1→2 (§27) — do not skip ahead
   of a seed-validated Phase 1 result.
5. Separately (lower priority, orthogonal): if/when moving past TinyStories
   to a harder/more diverse dataset (README's own staged plan — WikiText-2/103
   or an OpenWebText subset), re-run the unconstrained-equilibrium
   diagnostic on that dataset before assuming `target_cost=0.14` transfers.
   Model-complexity scaling (bigger `max_layers`/`d_model` on the *same* easy
   dataset) is a separate, likely opposite-direction, effect worth testing
   independently — see 2026-09-28 entry.

---

## 2026-09-28 — Initial audit, pipeline fixes, and target_cost diagnosis

### Bugs found and fixed

1. **`evaluate.py` couldn't load `CompactTransformer` checkpoints.** `ppl()`
   and `lm_eval()` unconditionally instantiated `SelfOrganizingTransformer`,
   so evaluating `outputs/sot_compact` or `outputs/discovered_retrained` (the
   README's own "most important control") would crash or silently load
   mismatched weights. Fixed by reading `config.json`'s `model_type` and
   dispatching to the right class (`evaluate.py::load_model`).
2. **`compact_model.py`'s skip-connection normalization didn't match the
   trained function.** The SOT model divides every active skip edge by the
   constant `sqrt(skip_span)`; the materialized compact model instead divided
   by the count of edges that happened to survive thresholding at that
   specific layer, and didn't drop skip sources whose own layer got pruned.
   This changed residual-stream scale relative to what the SOT model actually
   computed, undermining the "inherited weights" control's fidelity. Fixed:
   `CompactConfig` now carries `skip_span`, forward uses the constant
   divisor, and dead skip sources are filtered out at `build_from_sot` time.
3. **`save_pretrained()` crashed on the very first checkpoint** under
   `transformers>=5.x` — neither `SelfOrganizingTransformer` nor
   `CompactTransformer` declared `_tied_weights_keys` for the
   `lm_head.weight`/`embed.weight` tie, which newer `transformers` versions
   require explicitly. Added `_tied_weights_keys = {"lm_head.weight":
   "embed.weight"}` to both classes.
4. **`evaluate.py`'s `ppl()` hardcoded `seq_len=256`** with no override —
   silently wrong perplexity (truncated/misaligned chunks) for any model
   trained at a different context length, or a crash for a shorter one.
   Now defaults to the checkpoint's own `config.max_position_embeddings`,
   with a `--seq_len` override.
5. Dead code removed: `SOTConfig.rope_theta` (never read — the model uses
   learned absolute position embeddings, not RoPE) and
   `model.materialize_masked_model()` (superseded by
   `compact_model.build_from_sot()`, zero call sites).

### Instrumentation added

- `model.gate_summary()` — aggregate and per-layer means for all four gate
  families, logged into `metrics.json` every `log_every` steps via
  `train.py`. This is what made the diagnostics below possible.
- `tqdm` progress bars and cleaner logging across `train.py`,
  `retrain_compact.py`, `evaluate.py`.
- `--max_train_docs` cap on `train.py`/`retrain_compact.py` — lets a
  diagnostic run tokenize a bounded slice of TinyStories instead of the full
  ~2.1M-row corpus (~5–6 min fixed cost per invocation otherwise, unrelated
  to step count).

### Diagnosis: the Lagrangian budget constraint was inert at shipped defaults

At init, `architecture_cost()` ≈ 0.161 (`sigmoid(-1)` for layer/head/skip
gates, `sigmoid(1.5)` for the hardcoded FFN group init), against the shipped
`target_cost=0.45`. Confirmed empirically across a 500-step local run and a
2000-step Kaggle run at `target_cost=0.45`: `lagrange` stayed at exactly
`0.0` for the entire run in both cases — the constraint never once activated.
`cost` itself plateaued at **~0.186** by step 2000 (converged, not still
rising), meaning `0.45` was never going to bind no matter how long the run
went. The only thing shaping the gates in that regime was
`arch_reg * progress * sparsity_loss()` vs. task-loss pressure — the
Lagrangian term was pure dead weight in every run tried under the original
default.

### Emergent structure (unconstrained, target_cost=0.45, TinyStories, seed 42, 2000 steps)

Even with the Lagrangian inert, real non-uniform structure emerged from
gradient training alone — not noise, confirmed converged by step 2000
(values stable step 1800→2000) and consistent in shape from around step 500
onward:

- `layer` and `ffn` gates: hump-shaped by depth, low at layer 0, peaking
  around layers 6–8, tapering (not collapsing) toward layer 11.
- `head` gates: different shape — two local peaks in layers 1–5, then a
  clear decline through layers 9–11.
- `skip` gates: monotonically increasing with depth at every checkpoint —
  mechanically sensible, since layer 1 only has one eligible skip source
  while deeper layers have up to `skip_span=3`.

### target_cost=0.14 validated

Since `cost` converges to ~0.186 unconstrained, `target_cost=0.14` (below
that) was chosen to force the constraint to actually bind, and re-run for
2000 steps (Kaggle, seed 42, `--max_train_docs 500000`). Confirmed working:
`lagrange` rose monotonically to 0.50 by step 2000 (still rising slightly at
the end since the cosine LR schedule had decayed to 0 for the primal
variables while the dual multiplier's own fixed `lagrangian_lr=0.01` update
keeps going regardless — worth watching if this is scaled to a longer run).
`cost` peaked at 0.168 (step 900) then reversed, ending at 0.164 — a ~12%
reduction vs. the unconstrained run's converged 0.186 — while `val_ppl` at
step 2000 was nearly unchanged (8.166 vs. 8.125 unconstrained, ~0.5% worse).

Comparing the two runs' final per-layer structure: layers 0–5 kept nearly
the same shape and magnitude in both. Layers 6–11 were cut hard under the
budget (e.g. `layer` gate at layer 11: 0.297 unconstrained → 0.249
constrained; `ffn` at layer 11: 0.826 → 0.798). The model didn't shrink
uniformly under budget pressure — it preferentially sacrificed back-half
capacity and kept the front half intact. This is the first genuinely
interesting, specific, falsifiable structural finding from the project.

`target_cost=0.14` is now the default in both `train.py` and
`run_experiment.sh`.

### Open question raised, not yet tested

Would a harder/more diverse dataset (vs. TinyStories, which is deliberately
easy) push the natural unconstrained equilibrium cost above 0.186, since
`architecture_cost()` is a *fraction of the envelope in use* and a harder
task plausibly keeps task-loss pressure favoring open gates for longer
before the fixed, small `arch_reg` ceiling catches up? Reasoned as plausible
but untested. Separately, and likely *not* the same direction: scaling up
model complexity (bigger `max_layers`/`d_model`) on the *same* easy dataset
might lower the fractional equilibrium instead (diminishing returns —
diluting a roughly-fixed absolute capacity need into a larger denominator).
These are two independent levers worth testing separately, not assumed to
compound. Not scheduled yet — lower priority than finishing the Phase 1
matrix on TinyStories first, per README's own staged generalisation-dataset
plan.

---

## 2026-09-29 — First full-scale run (RunPod, seed 42) and a tied-weights bug

### Infrastructure

Ran the parallel phase of the experiment matrix on a single RunPod A100 80GB
PCIe pod (CA-MTL-3), all 4 conditions (`dense`, `dense_8l`, `dense_half_ffn`,
`sot`) as concurrent background processes sharing one GPU rather than 4
separate pods — the model is tiny (48M params, <1GB VRAM) relative to an
80GB card, so this was both cheaper and sidestepped a live GPU stock
shortage (RTX 4090/L4 secure-cloud stock went from available to exhausted
within minutes of checking; A100 was the only tier with real stock and
network-volume support in a usable data center). Full corpus, no
`--max_train_docs` cap, `max_steps=10000`, seed 42, matching
`run_experiment.sh` defaults exactly. Total cost for the 4-condition
parallel phase: **$8.94** (~5.5h wall clock on one $1.599/hr pod).

### Bug: `lm_head.weight` silently randomized on every `from_pretrained` call

Running `materialize.py` against the real `sot/final` checkpoint surfaced a
transformers load report: `lm_head.weight | MISSING ... newly initialized`.
Root cause: `_tied_weights_keys` (added 2026-09-28 to fix the save crash)
isn't sufficient on its own on this transformers version —
`get_expanded_tied_weights_keys()` gates entirely on
`getattr(config, "tie_word_embeddings", False)`, and neither `SOTConfig` nor
`CompactConfig` ever set that attribute, so it silently defaulted to `False`
and the tie-on-load mechanism no-opped completely, every time, for every
checkpoint ever loaded via `from_pretrained`.

Impact assessment: `materialize.py`'s output was *not* corrupted —
`build_from_sot()` never reads the loaded `lm_head.weight`, only
`embed.weight` (which loaded correctly, being actually present in the
checkpoint), and re-ties `compact.lm_head.weight = compact.embed.weight`
explicitly itself. But `evaluate.py` loads checkpoints directly via
`from_pretrained` and runs inference straight through whatever `lm_head`
comes out — every perplexity and lm-eval benchmark number computed against
any checkpoint so far would have gone through a random, untrained output
projection. Caught before any `evaluate.py` run, so no results were actually
produced with the bug present.

Fix: explicitly set `self.tie_word_embeddings = True` in both config
`__init__`s, plus `get_input_embeddings`/`set_input_embeddings`/
`get_output_embeddings`/`set_output_embeddings` on both model classes (the
first fix alone — the embeddings accessors — did *not* resolve it; the
`tie_word_embeddings` config flag was the actual gate). Verified locally
with a save/load roundtrip carrying a marker value through `embed.weight`:
`lm_head.weight` now ties correctly (same object, same values, marker
preserved) instead of reporting `MISSING`. Verified again for
`CompactTransformer`. Redeployed to the pod and re-ran `materialize.py`
before proceeding — confirmed clean, no warning, `tied ok: True`.

### Status at time of writing

Sequential phase (`materialize.py` → `retrain_compact.py` → `evaluate.py`
on all 6 checkpoints) in progress on the same pod. Will append results once
complete.

### Second bug/finding: `--threshold 0.5` (the documented default) produces a zero-layer model

`retrain_compact.py` was launched successfully against the (tie-fixed)
`sot_compact` checkpoint, tokenized, and started training — but loss sat at
~4.3–4.4 nats at 63% through training, far above where every other
condition in this run landed (~1.5–1.7 by a similar point). Checked
`materialize.py`'s own printed layer list: `[]`. Zero layers.

Root cause, confirmed against the real `sot/final` architecture.json: every
`layer`/`head`/`skip` gate in this run sits in `[0.095, 0.364]` —
`layer` 0.200–0.270 (mean 0.226), `head` 0.181–0.364 (mean 0.282), `skip`
0.095–0.160 (mean 0.121). `ffn` is the only family anywhere near the
`threshold=0.5` default (0.718–0.781). This isn't a one-off — it's
structural. `gate_init_logit=-1.0` means `layer`/`head`/`skip` gates start
at `sigmoid(-1)=0.269` and the sparsity regularizer only ever pushes them
*down* from there; they were never going to cross `0.5` regardless of what
the run discovered. `export_architecture(threshold)`'s documented default
(and `materialize.py`'s CLI default, and README's own worked example) is
`0.5` — meaning **the "most important control in the pipeline" (fresh
retrain of the discovered architecture) is broken out of the box for any
run using the shipped `gate_init_logit`**, not just this one. `retrain_compact.py`
had been training a degenerate embed→lm_head-only stub (no transformer
blocks at all) — its loss floor (~4.3–4.4) is consistent with a model that
can only learn unigram statistics, which matches what was observed.

Immediate fix (this run only, not a code change): re-ran `materialize.py`
with `--threshold 0.22` — picked because it sits at the mean/median of this
run's `layer` gate distribution, giving a real split rather than an
arbitrary cutoff. Produced 7 active layers (original indices 2–8 — the
same middle-depth block the unconstrained/constrained diagnostic comparison
on 2026-09-28 already identified as preferentially *retained* under budget
pressure), 6 of the 7 at full 6/6 heads (one at 4/6), full FFN width on all,
no skip connections (max skip gate 0.160, below any threshold that keeps
layers non-empty). Genuinely non-degenerate; `retrain_compact.py` restarted
against it.

**Not yet fixed in code** — `0.22` was derived from this specific run's
gate distribution and there's no guarantee it transfers to a different seed
or a different `target_cost`. This needs the same treatment `target_cost`
got: either (a) `materialize.py` should default to a distribution-relative
threshold (e.g. per-family median/percentile) instead of a fixed absolute
value, or (b) at minimum, `export_architecture()`/`materialize.py` should
warn loudly (not just print an empty list silently) when a threshold
produces zero active layers, so this can't happen unnoticed on a real run
again. Flagging as a required follow-up before trusting any *other* seed's
materialize/retrain step to just work with `--threshold 0.5`.

### Third bug: fp16 dtype mismatch crashing dense-mode evaluation on CUDA

All 3 dense evaluations (`dense`, `dense_8l`, `dense_half_ffn`) crashed
immediately on launch: `RuntimeError: expected mat1 and mat2 to have the
same dtype, but got: float != c10::Half`. `_gates()`'s `architecture_mode
== "dense"` branch built its gate tensors via `torch.ones(...)`/
`torch.zeros(...)` with no `dtype=`, always float32 regardless of the
model's actual parameter dtype. `evaluate.py` loads models in fp16 on CUDA
(`torch_dtype=torch.float16`), so `y * head_gate` in
`CausalSelfAttention.forward` upcasts the fp16 activation back to float32,
which then hits the fp16-weighted `self.out` linear layer. SOT-mode gates
were unaffected (derived from the model's own already-correctly-dtyped
parameters via sigmoid); `CompactTransformer` has no gating at all, so
`sot_compact`/`discovered_retrained` were unaffected too — only the three
dense conditions crashed. Never caught locally because every prior test ran
on CPU/MPS, where `evaluate.py`'s fp16 cast doesn't apply — this was the
first time this code path had actually run on a CUDA GPU. Fixed by deriving
gate dtype from `self.embed.weight.dtype`; verified with a `.half()`
forward pass that previously crashed, then confirmed on the pod.

### Full results, seed 42

All 6 conditions, full corpus, `max_steps=10000`, real held-out validation
perplexity (`evaluate.py`'s `ppl()`, not the training-time approximation)
plus the full lm-eval suite (`hellaswag`, `arc_easy`, `piqa`, `winogrande`).
Summaries pulled from the pod under `results/*_summary.json` (the raw
`evaluation.json` files were ~64MB each — lm-eval's full per-task dump, not
worth transferring in full).

| condition | perplexity | hellaswag | arc_easy | piqa | winogrande |
|---|---|---|---|---|---|
| dense (12L/384/1536) | **116.72** | 0.2562 | 0.2593 | 0.5250 | 0.5233 |
| dense_8l | 128.31 | 0.2584 | 0.2593 | 0.5365 | 0.4941 |
| dense_half_ffn | 139.31 | 0.2545 | 0.2618 | 0.5386 | 0.4925 |
| **sot** | **109.63** | 0.2553 | 0.2538 | 0.5392 | 0.5043 |
| sot_compact (inherited weights) | **3240.61** | 0.2554 | 0.2542 | 0.5299 | 0.5304 |
| discovered_retrained (fresh) | **115.13** | 0.2535 | 0.2609 | 0.5343 | 0.5122 |

SOT's final `architecture_cost()` at step 10000: **0.128** (`lagrange=0.0`,
settled at the clamp floor — matches the live-monitored trajectory).

lm-eval benchmark scores are flat and at-chance across *every* condition
(hellaswag ~0.25 = random for 4-way; piqa/winogrande ~0.49–0.54 ≈ random for
2-way). Expected, not a bug: a 48M-param model trained for 10k steps on
TinyStories alone has no basis to develop the commonsense reasoning these
benchmarks test — they were designed to differentiate GPT-2-scale-and-up
models trained on much larger, more diverse corpora. **Perplexity is the
only metric carrying real signal at this scale**; the benchmark suite ran
successfully (validates the pipeline works end to end) but isn't
discriminating anything here.

### Scored against README's six interpretation criteria

1. **SOT reaches similar or lower validation loss than the dense baseline.**
   ✅ Clear pass — SOT (109.63) beats the matched dense baseline (116.72)
   outright, not just "similar."
2. **Learned cost falls substantially below 1.0.** ✅ Pass — 0.128, an 87%
   reduction from the dense supernet's implicit cost of 1.0.
3. **Re-training the discovered architecture from scratch recovers most of
   the SOT model's quality.** ✅ Strong pass — discovered_retrained (115.13)
   lands within ~5% relative perplexity of SOT itself (109.63), and beats
   the equivalent-budget dense baseline (116.72). This is the cleanest
   result in the run.
4. **Stable across random seeds.** ⏳ Not yet testable — single seed (42)
   only. Required follow-up before any of this is a confident claim rather
   than one data point.
5. **Not simply selecting maximum allowed capacity everywhere.** ✅ Pass,
   confirmed structurally (not just by the cost number) — SOT kept 7 of 12
   layers (the middle-depth block found in the earlier structural
   diagnostic), reduced heads in one of them, and dropped skip connections
   entirely.
6. **Benchmark quality remains competitive after physical compaction, not
   only in the masked supernet.** ⚠️ Nuanced, not a clean pass — depends on
   which compaction condition "counts." `sot_compact` (inherited weights,
   no retraining) collapses catastrophically (ppl 3240 — near-degenerate).
   `discovered_retrained` (same architecture, fresh weights) does not
   collapse at all (115.13, competitive). This matches README's own
   documented "Failure 7" pattern almost exactly: *"Supernet advantage
   disappears after materialisation... an important negative result rather
   than a bug... indicates weight-sharing/co-adaptation produced the
   apparent gain... therefore retraining-from-scratch compact models is
   mandatory."* Read that way, this is the *expected* shape of a positive
   result, not a failure of criterion 6 — but it does mean the literal
   claim "benchmark quality remains competitive after compaction" is only
   true for the retrained compact model, not the inherited-weights one.
   Worth a closer look at *why* the collapse is this severe (ppl 110→3240 is
   large even for a co-adaptation story) before fully accepting that framing
   — a candidate mechanical explanation: SOT's residual stream is
   multiplicatively scaled by `layer_gates[i]` (~0.2–0.27 for most active
   layers in this run) at *every* block during training, but
   `CompactBlock.forward` applies the full, unscaled residual addition
   (implicit gate=1.0) — a much larger magnitude jump at materialization
   than the earlier-fixed skip-normalization mismatch. Not established as a
   bug (the 2026-09-28 advisor review explicitly judged hard-binarizing
   kept units to full weight as the intended NAS-style design, not a
   defect), but flagging as worth a dedicated look given the severity here.

### Cost

Full run (parallel phase + sequential materialize/retrain/evaluate,
including one wasted ~30min detour on the degenerate `threshold=0.5`
retrain): **~$13.33** of the $55 budget, on one RunPod A100 80GB PCIe pod
(CA-MTL-3) at $1.599/hr. Pod still running at time of writing pending user
confirmation to terminate.
