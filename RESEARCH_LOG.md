# Research log

Running record of what's been tried, found, and fixed on the Self-Organising
Transformer experiment. Append new dated entries at the bottom; keep "Current
status" at the top up to date as the single-glance summary. See `README.md`
for the experiment design and `self_organizing_transformer_research_proposal.md`
for the full phased research programme.

## Current status

- **Phase**: Phase 1 (differentiable structural selection) — **all 3 seeds
  complete** (42, 123, 7), each run as the full 6-condition matrix (dense,
  dense_8l, dense_half_ffn, sot, sot_compact, discovered_retrained;
  10,000 steps, full corpus, perplexity + hellaswag/arc_easy/piqa/
  winogrande) on RunPod (A100, CA-MTL-3). README's stated minimum of 3
  seeds for the stability criterion is now satisfied. Full 3-seed
  comparison, scored against README's six interpretation criteria, in the
  "Seed 7 complete: full 3-seed comparison" 2026-09-29 entry below. **5 of
  6 criteria now clear pass across all 3 seeds** (including stability,
  previously untestable), **1 nuanced** — same as before, but with
  stronger evidence (see below).
- **Headline result**: SOT beats the matched dense baseline in all 3 seeds
  (109.63 vs 116.72; 117.67 vs 129.01; 108.33 vs 114.35 ppl), at ~87%
  lower architectural cost throughout (cost ≈0.125–0.129 vs. dense's
  implicit 1.0). Fresh retrain of each seed's discovered architecture
  recovers to within 5.0–7.4% of SOT itself and still beats dense, in all
  3 seeds — the single most robust result in the whole run. Inherited-
  weights compaction without retraining collapses in all 3 seeds, but
  severity is highly seed-sensitive (3240.61 / 23004.32 / 20551.79 ppl for
  seeds 42/123/7) — seed 42 turns out to be the mild outlier, not the
  typical case, which raises the priority of investigating whether the
  residual-stream-scale gap at materialization (gates hard-set to 1.0 vs.
  the ~0.2–0.27 they ran at during training) is contributing more than
  pure co-adaptation loss would.
- **Layer-selection pattern**: not identical across seeds, but a real
  coarser pattern holds — all 3 seeds exclude layer 0 and layers 9–11, and
  keep 6–7 layers drawn from the 1–8 range (seed 42: `[2..8]` exactly
  contiguous; seed 123: `[1,3,4,5,6,7]`, gap at 2; seed 7: `[1,2,3,4,5,7]`,
  gap at 6). Boundary layers reliably pruned, middle depths reliably kept,
  which single middle layer (if any) drops is noise.
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
- **Materialization bug found, fixed, and validated on real checkpoints**
  (2026-09-29/30): `compact_model.py`'s `build_from_sot` was discarding
  gate *magnitude* when copying weights — it kept the right heads/layers/
  groups but restored them to full (unscaled) weight instead of the
  ~0.2–0.3 magnitude they actually trained at. Verified exact via a
  zero-cost local test (folding `gate*layer_gate` into `out.weight`/
  `down.weight`'s input columns: 19.2% relative error unfixed → 5e-8
  fixed), then re-validated on all 3 seeds' real checkpoints: the bug
  explained 76–99% of the original sot→sot_compact perplexity collapse
  (seed 42: 3240.61→794.77 ppl; seed 123: 23004.32→445.13; seed 7:
  20551.79→280.03). Real pruning cost remains (2.6–7.3x sot's ppl even
  fixed) — criterion 6's structural verdict is unchanged but now much
  better characterized. Full detail in the two 2026-09-29 entries below.
- **`materialize.py --threshold` default** — still not fixed (deferred,
  tracked below); auto-median works fine in practice so far.
- **Dataset scale-up to WikiText-103 (Stage A, 2026-09-30): first
  negative result of the project.** Model config held fixed (same as
  Phase 1) — only the dataset changed. Equilibrium diagnostic found
  WikiText-103's natural cost ≈0.190 (vs. TinyStories' 0.186), giving the
  same `target_cost=0.14` via the same 75%-of-equilibrium ratio. Single
  seed (42), full 6-condition matrix: **`sot` did not beat dense (29.28 vs
  28.23 ppl) and `discovered_retrained` did not either (31.88 ppl, worse
  than `sot` itself)** — both reversals of Phase 1's headline result,
  where SOT and its retrained architecture beat dense in 3/3 TinyStories
  seeds. Structural finding: head-level pruning appeared for the first
  time (layers 8/9 of the kept `[4..9]` block reduced to 4/6 and 2/6
  heads — every TinyStories seed kept full heads everywhere). Leading
  hypothesis, not confirmed: `target_cost=0.14`'s ~86% cost reduction may
  simply be too aggressive on harder data, independent of seed — not
  something more seeds alone would resolve. This needs a decision (more
  seeds at current budget vs. a less aggressive `target_cost` re-run)
  before Phase 1's finding can be called generalised. Full detail and
  reasoning in the 2026-09-30 entry below; plan at
  `/Users/samuelbartlett/.claude/plans/swirling-sleeping-hickey.md`.
  Stage A cost: ~$8.27, under the ~$12-18 estimate.
- **Two more real bugs found while setting up the WikiText-103 run**:
  `retrain_compact.py` and `evaluate.py` had no `--dataset_name`/
  `--dataset_config` CLI exposure at all and silently hardcoded TinyStories
  internally — would have corrupted the WikiText-103 `discovered_retrained`
  and perplexity numbers by quietly training/evaluating against the wrong
  corpus, with no error. Fixed (commit `05a828f`) before any WikiText-103
  data was generated.

## Next steps (agreed, not yet executed)

1. **WikiText-103 Stage A** — in progress (training). Once done: run
   `materialize.py`/`retrain_compact.py`/`evaluate.py` (with
   `--dataset_name wikitext --dataset_config wikitext-103-raw-v1` on the
   dataset-aware ones) to complete the 6-condition matrix, write up the
   comparison against Phase 1's TinyStories numbers, then report back
   before deciding whether to commit to seeds 2/3 (per the plan's staged
   decision point).
2. **Fix the `--threshold` default** in `materialize.py`/`export_architecture()`
   to be distribution-relative (e.g. per-family median/percentile) rather
   than a fixed `0.5`, or at minimum raise a loud error on a zero-layer
   result instead of silently proceeding. Still open, low priority (current
   auto-median default hasn't caused a problem yet).
3. **Seed 42's worse-than-123/7 post-fix recovery ratio is unexplained**
   (Phase 1's gate-fold re-eval, 7.25x vs. 3.78x/2.59x residual ppl ratio)
   — not blocking, worth a look if there's a natural moment to revisit
   architecture selection.
4. Only after WikiText-103 (and any further generalisation datasets) confirm
   the finding holds past an easy dataset: decide on Phase 2 proper —
   growth/mutation per the research proposal's Gate 1→2 (§27) — do not
   skip ahead of a generalisation-validated result. (Note: "Phase 2" here
   in the log has been used loosely for "dataset scale-up"; the research
   proposal's own Phase 2 is specifically growth/mutation — keep these
   distinct going forward.)
5. Model-complexity scaling (bigger `max_layers`/`d_model`) is explicitly
   sequenced *after* the dataset scale-up, not concurrent with it, so each
   experiment isolates one variable — agreed with the user.

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
(CA-MTL-3) at $1.599/hr. Pod terminated immediately on pipeline completion
(see idle-pod lesson below) — the "pending confirmation" note above is
stale; corrected here rather than rewritten in place, log stays append-only.

### Lesson: never leave a pod idle

Waited ~3h for explicit user confirmation before terminating this pod after
its work was done, burning ~$4.80 on idle compute. Corrected by the user:
"Never leave a pod idle. Either use it, or terminate it." Saved as a
standing rule and applied without confirmation from seed 123 onward — pod
`g0u74yigack4gc` (seed 123) was terminated immediately on its pipeline
completion, no idle time billed.

## 2026-09-29 — Seed 123 complete, seed 7 parallel run, storage cleanup

Seed 123 ran the full 6-condition matrix (same hyperparameters, `--seed
123`) on a second pod (`g0u74yigack4gc`), in parallel with seed 42 for part
of its run, then alone. Completed, evaluated, pod terminated immediately.
Summaries pulled locally (`results/seed123_*_summary.json`,
`results/seed123_sot_compact_config.json`) — full 3-seed comparison and
write-up deferred until seed 7 (below) also completes, per the agreed
"≥3 seeds" next step.

Seed 123's discovered architecture selected a **non-contiguous** layer set
(`[1,3,4,5,6,7]`), unlike seed 42's contiguous middle block (`[2..8]`) — an
early signal that the "contiguous middle-depth block" structural finding
may not be a reliable pattern across seeds; needs seed 7 to disambiguate a
trend from noise. Inherited-weights collapse was also far more severe for
seed 123 (3240.61 ppl for seed 42 vs. **23004.32 ppl** for seed 123, ~7x) —
strengthens the case for treating the residual-stream-scale mismatch
hypothesis (see seed 42 entry above) as worth a dedicated mechanical
investigation rather than dismissing the severity as ordinary co-adaptation
loss.

A third pod (`dbk7560mc6t0q4`) was launched in parallel for seed 7 rather
than waiting for a pod to free up — same total GPU-hours either way, but
less wall-clock. Both seed-123 and seed-7 pods shared the same network
volume (`d167myxctn`) concurrently without issue, confirming multi-pod
sharing of one volume works.

### Storage pressure and cleanup

With seed 42 and seed 123 both fully trained (`/workspace/outputs`,
`/workspace/outputs_seed123`) and seed 7 still training
(`/workspace/outputs_seed7`), combined usage reached ~40GB against the
50GB network volume — `train.py`'s `save_every`/`eval_every` schedule
(every 500/1000 steps) leaves many redundant intermediate `checkpoint-N`
directories per condition (160 total across the two completed seeds, ~189MB
each, ~30GB) that are fully superseded by each condition's `final`
directory once training finishes. `df -h /workspace` is not useful for
judging this — it reports the underlying shared MooseFS cluster's total
capacity (755T), not this volume's 50GB quota.

Deleted the 160 intermediate `checkpoint-*` directories under `outputs` and
`outputs_seed123` only (excluded `final`, `sot_compact`,
`discovered_retrained`, and everything under `outputs_seed7`, which was
still being actively written). Freed ~28GB: usage dropped from ~15GB+15GB
to ~1.4GB+1.4GB for the two completed seeds. Total volume usage after
cleanup: ~13.8GB, leaving comfortable headroom for seed 7 to finish training
and run its own materialize/retrain/evaluate stages.

Seed 7 progress at time of cleanup: dense 82%, dense_8l 95%, dense_half_ffn
90%, sot 66% (cost=0.125, tracking target_cost=0.14 as in prior seeds).

## 2026-09-29 — Seed 7 complete: full 3-seed comparison

Seed 7's full 6-condition matrix finished (pod `dbk7560mc6t0q4`, terminated
immediately on completion, no idle time). Materialized architecture:
layers `[1,2,3,4,5,7]` (6 of 12, threshold=0.229 median layer_gate), full
heads/FFN retained in each active layer, no skip connections kept — same
qualitative shape as seeds 42 and 123. All results pulled and committed.

### Full results, all 3 seeds (perplexity)

| Condition | Seed 42 | Seed 123 | Seed 7 |
|---|---:|---:|---:|
| dense | 116.72 | 129.01 | 114.35 |
| dense_8l | 128.31 | 121.65 | 131.46 |
| dense_half_ffn | 139.31 | 123.24 | 119.39 |
| **sot** | **109.63** | **117.67** | **108.33** |
| sot_compact (inherited weights) | 3240.61 | 23004.32 | 20551.79 |
| discovered_retrained | 115.13 | 126.34 | 114.01 |

SOT's architectural cost tracked `target_cost=0.14` in all three runs (seed
42: 0.128, seed 7: 0.129; seed 123's exact final value wasn't captured
before its pod was terminated, but it followed the same convergence
pattern in the training logs at the time).

### Scored against README's six interpretation criteria, now with 3 seeds

1. **SOT reaches similar or lower validation loss than the dense
   baseline.** ✅ Clear pass, 3/3 seeds — SOT beats matched dense in every
   seed (109.63 vs 116.72; 117.67 vs 129.01; 108.33 vs 114.35). This is now
   a stable finding, not a one-off.
2. **Learned cost falls substantially below 1.0.** ✅ Pass, 3/3 — cost
   converges to ~0.125–0.129 in every seed checked, an ~87% reduction from
   the dense supernet's implicit cost of 1.0, consistently.
3. **Re-training the discovered architecture from scratch recovers most of
   the SOT model's quality.** ✅ Strong pass, 3/3 — discovered_retrained
   lands within 5.0–7.4% relative perplexity of SOT itself in every seed
   (115.13 vs 109.63; 126.34 vs 117.67; 114.01 vs 108.33), and beats the
   matched dense baseline in every seed too (by a thin but consistent
   margin: 1.4%, 2.1%, 0.3%). The recovery margin band (~5–7%) is itself
   stable across seeds — this is the most robust result in the whole run.
4. **Stable across random seeds.** ✅ Now testable, and passes on the
   criteria that matter most (SOT-beats-dense, retrain-recovers-quality).
   The layer-*selection* pattern is stable in shape but not in exact
   membership — see below. The inherited-weights collapse severity is
   *not* stable (see below) — that's the one place 3 seeds changed the
   picture from seed 42 alone.
5. **Not simply selecting maximum allowed capacity everywhere.** ✅ Pass,
   3/3 — 6–7 of 12 layers kept in every seed, never all 12.
6. **Benchmark quality remains competitive after physical compaction, not
   only in the masked supernet.** ⚠️ Same nuanced verdict as seed 42, now
   with stronger evidence. True only for `discovered_retrained` (passes,
   3/3, per criterion 3). `sot_compact` (inherited weights, no retraining)
   fails in all 3 seeds, and seed 42 turns out to be the *mild* outlier,
   not representative: 3240.61 (seed 42) vs. 23004.32 (seed 123) vs.
   20551.79 (seed 7) ppl — seeds 123 and 7 agree closely with each other
   (~7x worse than dense) while seed 42 is ~7x milder than either. This
   strengthens rather than weakens the README's "Failure 7" framing
   (retraining-from-scratch is mandatory, not optional) and makes the
   residual-stream-scale-mismatch hypothesis (gates run ~0.2–0.27 during
   training but are hard-set to 1.0 at materialization) more worth a
   dedicated mechanical investigation, not less — if seed 42 undershot the
   "typical" collapse severity by chance, that changes how any future fix
   should be validated (check it against seed 123/7-like severity, not
   seed 42's milder case).

### Layer-selection pattern across seeds

- Seed 42: `[2,3,4,5,6,7,8]` — 7 layers, perfectly contiguous.
- Seed 123: `[1,3,4,5,6,7]` — 6 layers, one gap at layer 2.
- Seed 7: `[1,2,3,4,5,7]` — 6 layers, one gap at layer 6.

Not a stable *exact* pattern — the specific single layer dropped inside the
kept block differs each time (none dropped / layer 2 / layer 6). But there
is a stable *coarser* pattern underneath: all three seeds independently
exclude layer 0 (the block right after embedding) and layers 9–11 (the
last three, right before the output), and all three keep 6–7 layers drawn
from the 1–8 range. Read together: **the boundary layers are reliably
pruned, the middle depths are reliably kept, and which one middle layer
(if any) gets dropped is noise** — a real structural finding, but a softer
one than "the model finds one specific contiguous block."

### Cost

Seed 7's full run (parallel training phase + sequential
materialize/retrain/evaluate) cost is not separately itemized here (billed
jointly with seed 123 while both pods ran in parallel, then alone) — see
combined budget tracking; all 3 seeds' full pipeline stayed well inside the
$55 budget.

### Next steps, updated

`≥3 seeds` (README's stability requirement) is now satisfied and scored
above. Remaining open items, carried forward: (1) the residual-stream-scale
mismatch hypothesis for the inherited-weights collapse, now higher
priority given seeds 123/7 show it's the typical case, not a seed-42
anomaly; (2) the `materialize.py --threshold` default fix (still using
auto-median, which works, but a distribution-relative choice was deferred,
not properly fixed); (3) re-running the equilibrium diagnostic on a harder
dataset before assuming `target_cost=0.14` transfers off TinyStories; (4)
only after (1)–(3), consider Phase 2 (growth/mutation) per the research
proposal's Gate 1→2.

## 2026-09-29 — Residual-stream-scale mismatch confirmed and fixed (bug, not co-adaptation)

Investigated item (1) above. **This was a real bug, not the intended
NAS-style "hard-binarize the kept units" design** the 2026-09-28 advisor
review had assumed — that read doesn't survive closer inspection.
"Binarize which units are structurally present" and "discard the learned
*magnitude* of the units you kept" are different claims; only the first is
required by NAS-style materialization, and `build_from_sot` was doing both.

### The mechanism

During SOT training, every active unit's contribution to the residual
stream is scaled down before being added: attention's per-head gate and
FFN's per-group gate multiply their respective outputs *before* the
`out`/`down` projection, and the whole block's output (attention's and
FFN's) is *again* scaled by that block's `layer_gate` before the residual
add (`model.py`'s `forward`, `x = residual + layer_gates[i] * y`). Real
seed gate values for kept layers ran ~0.2–0.3 (never near 1.0, even for
units well above the selection threshold). `build_from_sot` selected the
right heads/groups/layers to keep, but copied their `out.weight`/
`down.weight` slices at raw, unscaled magnitude — i.e. every kept unit's
contribution was silently restored to ~3-5x its trained magnitude at
materialization. `CompactBlock.forward` has no gate mechanism at all, so
there's nothing downstream that could compensate.

### Verifying it mechanically (zero-cost, no cloud spend)

Built a local exactness test (small random `SOTConfig` model, hand-set
gate logits to realistic values, no training needed): compared the
*current* `build_from_sot`'s compact-model forward pass against the
correct reference (the same SOT model, forward-run with the exact gates
hard-zeroed wherever the compact model drops a unit — "masked-SOT", not
raw SOT, since raw SOT also includes every below-threshold
layer/head/group and the skip terms the compact model has none of).

- Current `build_from_sot` (no gate-magnitude folding): **19.2% relative
  L2 error** in output logits vs. masked-SOT reference. A real, large
  divergence — confirms the hypothesis mechanically, not just by inspection.
- Fix (fold `head_gate * layer_gate` into `out.weight`'s input columns,
  `group_gate * layer_gate` into `down.weight`'s input columns — both
  gates land on a bias-free `Linear`'s input, so this is an exact,
  provable transform, not an approximation): **5e-8 relative error**
  (floating-point noise) against the same reference.

Applied the fix to `compact_model.py`'s `build_from_sot` (11 lines
changed) and re-verified against the actual production code, not just the
scratch duplicate — same 5e-8 result.

### What this does and doesn't establish yet

The exactness test proves the *mechanism* is real and the *fix* is
correct. It does **not** yet tell us how much of the 108→20,551 ppl gap
(seed 7's sot vs. sot_compact) this explains, because masked-SOT (the
correct reference) is not the same thing as "no collapse" — masked-SOT
still drops 5–6 of 12 layers entirely, and that pruning costs something
real regardless of the magnitude bug (this is exactly why
`discovered_retrained` needs retraining to recover, not just correct
copying). Re-running `materialize.py`+`evaluate.py` (fix applied, no
retraining needed) against the surviving `sot/final` checkpoints for all
3 seeds is the next step to answer: does the corrected `sot_compact`
perplexity land close to `sot` (bug was ~all of it), close to the old
20,551-ish number (bug was negligible), or somewhere between (bug and
pruning both contribute, now separable)? That's genuinely new information
either way — not proposed yet, needs a cheap pod re-provisioned against
the network volume (checkpoints for all 3 seeds' `sot/final` survived the
earlier storage cleanup).

`discovered_retrained`'s existing numbers (all 3 seeds) are **unaffected**
by this fix — fresh weights trained from scratch on the same architecture
don't depend on what `sot_compact`'s inherited copy looked like. The
3-seed headline results above stand as written.

### Re-evaluation with the fix: the bug explains most, not all, of the collapse

Re-provisioned a pod against the network volume (`d167myxctn`; `sot/final`
checkpoints for all 3 seeds survived the earlier storage cleanup — it only
removed intermediate `checkpoint-*` dirs), deployed the fixed
`compact_model.py`, re-ran `materialize.py` (no threshold override, same
auto-median as before) and `evaluate.py` against each seed's `sot/final`.
No retraining involved. Cost: ~$0.80 (A100 80GB PCIe, CA-MTL-3, $1.59/hr, ~30 min wall-clock
including one dependency-fix detour — see below).

| Seed | sot_compact (broken) | sot_compact (fixed) | sot (reference) | % of gap explained by the bug | fixed/sot ratio |
|---|---:|---:|---:|---:|---:|
| 42 | 3240.61 | **794.77** | 109.63 | 78.1% | 7.25x |
| 123 | 23004.32 | **445.13** | 117.67 | 98.6% | 3.78x |
| 7 | 20551.79 | **280.03** | 108.33 | 99.2% | 2.59x |

The magnitude bug was the dominant driver of the catastrophic collapse in
every seed — a 76–99% reduction in the sot→sot_compact perplexity gap,
just from copying weights at the magnitude they were actually trained at.
But it wasn't the *whole* story: even fixed, `sot_compact` still lands
2.6–7.3x worse than `sot` itself, in every seed — real, substantial
recovery is still needed, consistent with genuine co-adaptation/pruning
cost (which is exactly why `discovered_retrained` exists as a condition
and why it, not `sot_compact`, is the one that matters for criterion 6).

**A striking reversal in which seed looked "worst."** Before the fix,
seed 42 looked like the mild outlier (3240 vs. ~20–23k for 123/7) and 123/7
looked consistent with each other. After the fix, that pattern flips: 123
and 7 are still consistent with each other (98.6%/99.2% explained, 3.78x/
2.59x residual ratio) but seed 42 is now the *worst*-recovering seed by a
clear margin (78.1% explained, 7.25x residual ratio, both worse than
either other seed). This means seed 42's original "mild" collapse wasn't
evidence of an intrinsically more forgiving architecture — it was largely
an artifact of the same bug happening to do comparatively less damage
there (plausibly because seed 42's kept-layer gate values or head/group
selection happened to sit closer to the bug's implicit gate=1 assumption
than 123/7's did). What's left unexplained is now the interesting
question: why does seed 42's discovered architecture (7 contiguous layers,
`[2..8]`) recover *less* well from correct-magnitude pruning than seed
123's or seed 7's (6 layers each, with one gap) — worth a closer look, but
not chased further in this pass; noted as an open item below.

### Criterion 6, re-scored

Same structural verdict as before (`discovered_retrained` passes, 3/3;
`sot_compact` fails, 3/3 — retraining-from-scratch is still mandatory, not
optional) but the story underneath is now cleaner: the failure is smaller
and better-characterized than the original broken numbers suggested, most
of it was a materialization bug (now fixed) rather than an inherent
property of weight-sharing, and what genuinely remains (2.6–7.3x) is a
believable magnitude for real co-adaptation loss rather than the
near-total-collapse the broken numbers implied.

### A second bug found and fixed along the way

The re-eval pod's fresh `pip install -r requirements.txt` pulled
unpinned-version `transformers>=5.0` (incompatible with the pod's
preinstalled `torch==2.4.1`) and separately `lm-eval==0.4.13` (needs
`typing_extensions>=4.13`; pod had `4.9.0`), each crashing evaluation
outright — the second one only after a full pod's worth of setup, wasting
some of the estimated budget on a false start. Pinned `transformers==4.49.0`
and upgraded `typing_extensions`; not fixed in `requirements.txt` itself
yet (worth doing — the floors `torch>=2.4`/`transformers>=4.49.0` are too
loose for what those minimums actually need at install time).

### Next steps, updated again

(1) done — gate-magnitude fold fix confirmed and re-validated on real
checkpoints. (2) `materialize.py --threshold` distribution-relative fix —
still open. (3) equilibrium diagnostic on a harder dataset — still open.
(4) done — `requirements.txt` now pins `transformers<5.0` (was crashing
against the pod's preinstalled `torch==2.4.1`) and adds an explicit
`typing_extensions>=4.13` floor (`lm-eval==0.4.13` needs it; the pod had
`4.9.0` pulled in transitively). (5) new, lower priority: seed 42's
worse-than-123/7 post-fix recovery ratio is unexplained — not blocking,
but worth a look if there's a natural moment to revisit architecture
selection. Only after these, Phase 2 (growth/mutation) per the research
proposal's Gate 1→2.

## 2026-09-30 — WikiText-103 scale-up kicked off (dataset before model size)

Phase 1 (TinyStories, small model, 3 seeds) is complete and validated —
5 of 6 README criteria clear pass, 1 nuanced but well-characterized after
the gate-fold fix. User asked "next step is to scale this up, yes?".
Agreed: scale the *dataset* before the *model*, since TinyStories is
deliberately easy (simple vocabulary, short repetitive sentences) and
dropping 5-6 of 12 layers may look "fine" there simply because the task
doesn't need the capacity — a falsely encouraging result. The research
proposal (`self_organizing_transformer_research_proposal.md` §5.2)
explicitly names a harder "generalisation dataset" as the required next
step before trusting Phase 1 generalises.

### Dataset choice: WikiText-103-raw-v1, not WikiText-2

The proposal lists WikiText-2 before WikiText-103. Picked WikiText-103
anyway, for a token-budget reason specific to this pipeline's existing
hyperparameters: at `--max_steps 10000 --batch_size 16 --seq_len 256
--grad_accum 8`, one run consumes `16*256*8*10000` ≈ 327M tokens.
WikiText-2 is only ~2M tokens — ~160 epochs over the same tiny corpus,
which would mostly test memorization, not generalisation. WikiText-103
(~103M tokens) gives ~3 epochs at the same step count, a believable
repeat factor comparable to TinyStories' own full-corpus scale.

### Staged approach: 1 seed first, not 3 upfront

User asked directly: "do we need to prove out the 3 seed validation again
if we have already confirmed it?" — good question, answered no. The point
of ≥3 seeds in Phase 1 was to confirm SOT's advantage is *stable*, not a
fluke of one seed; that's answered and doesn't need re-proving. What's
still open is whether the *headline result itself* holds on harder data
at all, and whether WikiText-103 has different seed-to-seed variance than
TinyStories did. Staged instead of committing 3 seeds blind: Stage A is 1
seed (42), full 6-condition matrix; a decision point after Stage A (clean
pass → propose 1-2 more seeds as a separate authorization; borderline/
contradicts Phase 1 → seeds 2/3 become necessary before concluding
anything). Full plan at `/Users/samuelbartlett/.claude/plans/swirling-sleeping-hickey.md`.
Budget for Stage A estimated at ~$12-18 (vs. Phase 1's per-seed cost,
which averaged ~$9.44/seed once the operational pattern was proven,
though the very first seed paid ~$18.39 for one-off inefficiencies).

### Bug found while wiring this up: dataset_name/dataset_config not actually plumbed through

`train.py`'s `TrainConfig.dataset_config` field existed but had no
`--dataset_config` CLI flag — a real gap, since HF's `wikitext` dataset
*requires* a config name (`wikitext-103-raw-v1`) and can't load bare.
Worse: `retrain_compact.py` and `evaluate.py` had **no** `--dataset_name`
exposure at all — both hardcoded `TrainConfig`/`ppl()` defaults to
`"roneneldan/TinyStories"` internally (`retrain_compact.py:36`,
`evaluate.py:44,53`). This means every prior `discovered_retrained`
condition really was retrained on the correct dataset only because it
happened to be TinyStories every time — the moment the dataset changes,
`retrain_compact.py` would have silently retrained on TinyStories while
`evaluate.py` silently perplexity-scored every checkpoint (including
`sot`/`dense`/etc, not just `discovered_retrained`) against TinyStories's
validation split regardless of what the model was actually trained on.
No error, no warning — just quietly wrong numbers. Caught before any
WikiText-103 data was generated. Fixed by adding `--dataset_name`/
`--dataset_config` CLI flags to all three scripts and threading them
through to the relevant `TrainConfig`/`load_dataset` calls (commit
`05a828f`), verified with `--help` locally before redeploying.

### Equilibrium diagnostic: WikiText-103 ≈ TinyStories

Re-ran the same diagnostic methodology as the original 2026-09-28
TinyStories diagnostic: SOT, single seed (42), `--target_cost 0.9`
(deliberately unreachable so the Lagrangian term stays pinned at its
floor and `sparsity_loss`'s natural pull is what's observed), 3000 steps,
same model config as Phase 1. Cost climbed from 0.162 (early) to a clear
plateau at **0.190** by step 3000 (climb rate had slowed to ~0.001/100
steps by the end, essentially flat). This is strikingly close to
TinyStories' own equilibrium (0.186) — plausibly more a property of the
gate-init (`-1.0`)/regularizer (`arch_reg=0.015`) dynamics at this model
scale than of the dataset's difficulty; worth keeping in mind when this
question comes up again at a different model scale. Applying the same
75%-of-equilibrium ratio that validated `target_cost=0.14` for
TinyStories (`0.14/0.186 ≈ 0.75`) gives `0.190 * 0.75 ≈ 0.14` — the exact
same value. Used `target_cost=0.14` for Stage A's `sot` run.

### Operational note: don't chain multiple background jobs in one SSH command

First attempt to launch all 4 Stage A conditions used one SSH call with
4 parenthesized `(setsid bash -c '...' & disown)` groups chained with
`&`. 3 of 4 failed silently — `python3: can't open file '/root/train.py'`
— apparently some race in how `cd`'s effect on cwd propagated to later
subshells in the chain when the SSH connection closed quickly (unlike
prior single-job launches, which reliably hit the expected ~8s client
timeout rather than returning fast). Relaunched the 3 failed jobs
individually (one SSH call per job — the pattern proven reliable
throughout Phase 1) and all 4 came up correctly. Lesson for future
multi-job launches on this infra: always one SSH call per job, never
chain multiple backgrounded jobs in a single call, even though it
appears to work for the first job in the chain.

### Stage A complete: on WikiText-103, SOT does not beat dense

All 6 conditions trained, materialized/retrained where applicable, and
evaluated. Pod `edbsvu7rlrx60p` terminated immediately on completion, no
idle billing. Actual Stage A cost: **~$8.27** (checked via `list-billing`),
comfortably under the ~$12-18 estimate.

| Condition | Perplexity |
|---|---:|
| dense | **28.23** |
| sot | 29.28 |
| dense_8l | 30.14 |
| dense_half_ffn | 30.50 |
| discovered_retrained | 31.88 |
| sot_compact | 8495.94 |

**This is a materially different result from Phase 1.** `sot` is 3.7%
*worse* than dense — the first time in this project SOT hasn't beaten the
matched dense baseline. More significantly, `discovered_retrained` — the
condition that was the most robust, cleanest pass in all 3 TinyStories
seeds (beating dense by 0.3–2.1% and landing within 5–7.4% of `sot`) — is
now *worse than `sot` itself* (8.9% worse) and 12.9% worse than dense.
Fresh weights on the discovered 6-layer architecture did not recover here;
on TinyStories, they reliably did.

Read together, this looks less like "the architecture is fine but the
copied weights are bad" (Phase 1's `sot_compact` story) and more like
"the discovered architecture itself doesn't have enough capacity for this
task" — retraining from scratch can't fix a capacity shortfall the way it
can fix a co-adaptation/weight-magnitude problem. That's a materially
different failure mode than anything seen in Phase 1, and it's exactly
the risk flagged when this phase was proposed: an aggressive
capacity cut (`target_cost=0.14`, ~86% cost reduction) that cost nothing
on an easy dataset may cost real capability on a harder one.

`sot_compact`'s 8495.94 ppl is not comparable to Phase 1's collapse
numbers in one respect worth flagging: this run used the already-fixed
`materialize.py` (gate-magnitude folding) from the start, so there's no
"broken vs. fixed" comparison here — 8495.94 is the *fixed* number, and
it's still a severe collapse, consistent with a harder task making
inherited-weight compaction less forgiving, not just target_cost being
more aggressive.

### Structural finding: head-level pruning appears for the first time

Materialized architecture (auto-threshold=0.2371): layers `[4,5,6,7,8,9]`
kept (6 of 12, perfectly contiguous — unlike any TinyStories seed's exact
membership, though similarly sized). All 3 TinyStories seeds kept **full
heads (6/6)** in every active layer, with only layer/skip-level pruning.
Here, the two deepest kept layers had heads *reduced*: layer 8 to 4/6,
layer 9 to 2/6 (raw threshold-selected counts were 5 and 2 respectively;
5 isn't a legal divisor of `d_model=384` among `{1..6}`, so
`build_from_sot`'s existing legal-divisor snap took over, landing on 4).
This is the first time head-level pruning has actually fired in a
materialized architecture across this whole project — plausible reading:
on a harder task, the model reaches for a second lever (which heads
within a layer matter) once layer-level pruning alone can't hit the cost
budget as cheaply, rather than only ever operating on the coarser lever.
Worth watching whether this becomes a more general pattern as task
difficulty increases further.

### What this does and doesn't mean — my honest read, not a decision

Per the plan's decision point (`swirling-sleeping-hickey.md`): this is
squarely the "borderline/contradicts Phase 1" case, which the plan
explicitly said should not be resolved unilaterally. Recording my own
read here, not as a conclusion:

- **Against pure seed noise**: two independent conditions (`sot` and
  `discovered_retrained`) both lost to dense, in a mutually consistent
  ordering (dense < sot < dense_8l < dense_half_ffn < discovered_retrained
  < sot_compact) rather than one outlier condition. That's a coherent
  pattern, not the kind of single-number fluke a bad batch or an unlucky
  initialization typically produces.
- **For seed noise still being possible**: it's one seed. Phase 1 itself
  showed real seed-to-seed variance (layer-selection pattern, collapse
  severity) even when the headline direction stayed stable — a different
  seed's WikiText-103 architecture could conceivably land in a less
  capacity-starved region and beat dense.
- **My leading hypothesis**: `target_cost=0.14` — validated for
  TinyStories, and re-derived here from a WikiText-103-specific
  equilibrium diagnostic using the *same ratio* — may simply be too
  aggressive a cost budget for a harder dataset, independent of seed.
  The equilibrium ratio transferring (both ≈0.75× natural equilibrium)
  doesn't guarantee the *resulting* cost level is still "free" the way it
  was on TinyStories; easy vs. hard datasets could need different ratios,
  not just different equilibrium points.

Not recommending seeds 2/3 as the next move by default — if the leading
hypothesis is right, more seeds at the same `target_cost=0.14` would just
confirm the same capacity shortfall more times, not resolve it. Worth
discussing with the user before spending further: either more seeds at
the current budget (to separate signal from noise on *this* setting), or
a second diagnostic at a less aggressive `target_cost` (e.g. re-run the
matrix at ~0.25–0.3) to test the "budget too aggressive" hypothesis
directly, before committing to either.

### Updated 3-way dataset comparison

TinyStories (Phase 1, 3 seeds): SOT beat dense in 3/3, discovered_retrained
recovered and beat dense in 3/3, layer selection always kept full heads.
WikiText-103 (Stage A, 1 seed): SOT lost to dense, discovered_retrained
lost to both dense and SOT, and head-level pruning appeared for the first
time. The generalisation test this phase set out to run has produced its
first genuinely negative signal — not fatal to the overall hypothesis,
but a real result that needs resolving before claiming Phase 1 generalises
past an easy dataset.
