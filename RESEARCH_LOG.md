# Research log

Running record of what's been tried, found, and fixed on the Self-Organising
Transformer experiment. Append new dated entries at the bottom; keep "Current
status" at the top up to date as the single-glance summary. See `README.md`
for the experiment design and `self_organizing_transformer_research_proposal.md`
for the full phased research programme.

## Current status

- **⚠️ CORRECTION (2026-10-06): Phase 1 TinyStories's headline result was an
  artifact of a broken `transformers` install, not a real finding.** Every
  perplexity number below marked (original, now known incorrect) was
  computed through a `transformers==5.17.0` environment that violates this
  project's own `requirements.txt` pin (`>=4.49.0,<5.0`) and silently
  corrupts save/reload round-tripping for this custom architecture. See
  the "Transformers-version reload bug" 2026-10-06 entry for the full
  investigation and the corrected numbers; this "Current status" block now
  states the corrected picture directly rather than the original one.
- **Phase**: Phase 1 (differentiable structural selection) — **all 3 seeds
  complete** (42, 123, 7), each run as the full 6-condition matrix (dense,
  dense_8l, dense_half_ffn, sot, sot_compact, discovered_retrained;
  10,000 steps, full corpus, perplexity + hellaswag/arc_easy/piqa/
  winogrande) on RunPod (A100, CA-MTL-3). README's stated minimum of 3
  seeds for the stability criterion is now satisfied. Corrected 3-seed
  comparison, scored against README's six interpretation criteria, in the
  2026-10-06 correction entry below. **3 of 6 criteria clear pass across
  all 3 seeds, 1 fails (SOT no longer beats dense), 2 unaffected/nuanced**
  — a materially different scorecard than originally reported (see below).
- **Corrected headline result**: SOT does **not** beat the matched dense
  baseline on TinyStories — dense is consistently *better* (lower ppl) in
  all 3 seeds (dense ≈4.61–4.64 vs. sot ≈4.74–4.75, a 2.4–2.8% relative
  perplexity gap in dense's favour), at ~87% lower architectural cost
  throughout (cost ≈0.125–0.129 vs. dense's implicit 1.0). Reframed
  positive finding: **SOT trades a small, consistent quality cost for a
  large capacity reduction**, not "beats dense outright" — a materially
  more modest claim than originally reported. Fresh retrain of each seed's
  discovered architecture lands within ~3–5% of SOT itself (consistent
  with the original "recovers most of SOT's quality" framing) but, like
  SOT, now also loses to dense (discovered_retrained ≈4.89–4.97 vs. dense
  ≈4.61–4.64). Inherited-weights compaction without retraining still
  collapses in all 3 seeds even on the gate-fold-fixed checkpoints
  (ppl 349–881, clean numbers unaffected by the transformers bug) — this
  part of the original finding stands, just at a less catastrophic
  magnitude than the original (bug-compounded) 3240–23004 ppl figures.
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
  heads — every TinyStories seed kept full heads everywhere).
- **Follow-up (2026-10-01): target_cost=0.17 re-run, 2 seeds — confirmed
  not a budget-tuning problem.** Tested whether a less aggressive cost
  budget (between the too-tight 0.14 and the natural ~0.19 equilibrium)
  would let SOT match dense. It didn't: `sot` still lost to dense in both
  seeds (29.01/29.05 vs. 28.23/28.11 ppl, 2.8–3.3% gap, barely narrower
  than 0.14's 3.7%), and `discovered_retrained` still failed to recover in
  either (32.41/33.15 ppl — worse than `sot` itself in both, and no better
  than Stage A's 31.88 at the tighter budget). Both seeds independently
  converged on the *exact same* 6 layers (`[5,6,7,8,9,10]`) with a
  monotonic head-count decline toward the output — much tighter cross-seed
  structural agreement than Phase 1 ever showed, arguing the mechanism is
  finding a consistent answer that's genuinely insufficient, not an
  unstable search. Conclusion: this looks like a real capacity shortfall
  at this model scale for this harder dataset, not a `target_cost`-tuning
  artifact — per the research proposal's own Gate 1→2 instruction
  ("investigate the mechanism before adding complexity"), this
  investigation is concluded, with three options on the table (push
  target_cost further — unlikely to help much more; scale up model size
  before further dataset work; or accept this as a real, bounded negative
  result). Full detail in the 2026-10-01 entry below; plan at
  `/Users/samuelbartlett/.claude/plans/swirling-sleeping-hickey.md`.
  Combined WikiText-103 investigation cost: Stage A ~$8.27 + this 2-seed
  re-run ~$19-20 ≈ ~$27-28.
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

**⚠️ CORRECTED 2026-10-06: the perplexity numbers in this table were computed
through a broken `transformers==5.17.0` install and are wrong — see the
"Transformers-version reload bug" entry near the bottom of this log for the
corrected numbers and a materially different verdict. Left in place
unmodified for the historical record; do not cite this table.**

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

**⚠️ CORRECTED 2026-10-06: the table below is wrong (broken `transformers`
install) — see the "Transformers-version reload bug" entry near the bottom
of this log for corrected numbers and verdict. Left unmodified for the
historical record; do not cite this table.**

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

## 2026-10-01 — target_cost=0.17 re-run (2 seeds): the regression is not a budget-tuning problem

User asked to test the hypothesis directly: re-run the WikiText-103 matrix
at `target_cost=0.17` (between the too-aggressive 0.14 and the natural
unconstrained equilibrium ~0.19), 2 seeds (42, 123) this time rather than
1, to separate seed noise from a real effect. User topped up RunPod with
a fresh $30 for this.

**Infrastructure note**: CA-MTL-3 ran out of GPU capacity entirely (tried
A100 80GB PCIe, A100 SXM4, H100 PCIe, A40 — all rejected as unavailable
despite catalog stock elsewhere). Moved to **EU-RO-1** — new network
volume (`7m6hd7xsf7`), two pods, one per seed. This meant *not* reusing
Stage A's dense-family baselines (different region/volume) — both seeds
ran the full fresh 4-condition matrix, which conveniently produced a free
determinism check: seed 42's dense/dense_8l/dense_half_ffn perplexity
came back **bit-identical** to Stage A's numbers (28.23070333054666 to
14 decimal places) — confirms these really are deterministic given a
fixed seed, dataset, and hyperparameters, with `target_cost` genuinely
irrelevant to dense-mode training as expected.

### Full results, both seeds, target_cost=0.17

| Condition | Seed 42 | Seed 123 |
|---|---:|---:|
| dense | 28.23 | 28.11 |
| dense_8l | 30.14 | 29.98 |
| dense_half_ffn | 30.50 | 30.45 |
| **sot** | **29.01** | **29.05** |
| sot_compact | 5480.25 | 7521.70 |
| discovered_retrained | 32.41 | 33.15 |

For reference, Stage A (seed 42, target_cost=0.14): dense 28.23, sot 29.28,
sot_compact 8495.94, discovered_retrained 31.88.

### The headline answer: raising target_cost does not resolve the regression

- **`sot` vs. `dense` gap**: 3.72% (seed 42 @ 0.14) → 2.76% (seed 42 @ 0.17)
  → 3.34% (seed 123 @ 0.17). A ~21% relative increase in allowed
  architectural cost (0.14→0.17) narrowed the gap by at most a quarter in
  the one same-seed comparison available, and the second seed sits right
  back at the original magnitude. **SOT still loses to dense in both
  seeds, at both target_cost settings tested** — this isn't close enough
  to call "matching" under any reasonable tolerance; it's a small but
  consistent, reproducible miss, not noise.
- **`discovered_retrained` does not recover, anywhere.** This is the more
  important number (Phase 1's most robust condition) and the pattern is
  now unambiguous across 3 data points (1 seed @ 0.14, 2 seeds @ 0.17):
  31.88, 32.41, 33.15 — all worse than `sot` itself, all worse than
  `dense`, and if anything trending slightly worse, not better, at the
  higher target_cost. On TinyStories, fresh retraining reliably recovered
  and beat dense in 3/3 seeds. On WikiText-103, across 3 independent
  runs now, it has never once recovered. That's not a budget-tuning
  artifact — a looser budget should make recovery *easier* if the
  problem were simply "too little capacity was allowed," and it didn't.

### Structural finding: tight cross-seed convergence, shifted and more aggressive

Both seeds at `target_cost=0.17` independently selected the **exact same
6 layers**, `[5,6,7,8,9,10]` — shifted one position deeper than Stage A's
`[4..9]` — with within-layer head counts that both show a clear
**monotonic decline toward the output**:

- Seed 42: heads `[4,4,3,4,1,1]`, layer_gates `[0.340,0.319,0.315,0.330,0.313,0.295]`
- Seed 123: heads `[6,4,3,2,1,1]`, layer_gates `[0.339,0.343,0.323,0.312,0.307,0.302]`

This is a *much* tighter cross-seed structural agreement than anything in
Phase 1, where layer *membership* itself varied seed-to-seed (contiguous
vs. gapped, different offsets). Here the exact same 6 layers, and a
qualitatively identical "collapse heads toward 1 near the output" pattern,
emerged twice independently. Read together with the ppl results: the
model isn't failing to find a *consistent* structure — it's finding the
same structure reliably, and that structure just isn't good enough to
match dense on this task. That argues for a genuine capacity ceiling at
this model scale on this dataset, not an optimization/search-stability
problem.

### What this means for Gate 1→2 on WikiText-103

Per the research proposal's own instruction ("if a gate fails, investigate
the mechanism before adding the next layer of complexity"), this was that
investigation. Conclusion: **the WikiText-103 Gate 1→2 failure is not a
`target_cost`-tuning artifact.** Both the headline `sot`-vs-`dense` gap
and `discovered_retrained`'s failure-to-recover are stable across two
independent `target_cost` settings and three seeds. The more likely
explanation is a genuine capacity shortfall: this is the same ~48M-param,
12-layer-max model that was validated on TinyStories, asked to do a
harder, more diverse task — and consistently discovering that the task
needs most or all of that capacity, not that a ~6-layer subset of it
suffices. This is, in a sense, a *sensible* result for the mechanism to
produce (it isn't inventing a bad architecture; dense_8l and
dense_half_ffn — the naive reduced-capacity baselines — also lose to full
dense, by a wider margin than `sot` does) — it just means this model scale
doesn't have slack to cut on this harder dataset the way it did on
TinyStories.

### Recommendation, not a decision

Three honest options, not resolved here:
1. Push `target_cost` higher still (toward or above the ~0.19 natural
   equilibrium) — unlikely to help much further given the trend between
   0.14 and 0.17 was already small, and anything at/above ~0.19 is
   functionally unconstrained (reproduces the diagnostic's own result).
2. Treat this as evidence that **model-size scale-up should come before
   further dataset-difficulty tuning** — a larger supernet (more layers/
   wider `d_model`) might have the slack on WikiText-103 that this one
   doesn't, which would reframe the user's original sequencing (dataset
   before model size) as needing to interleave rather than strictly
   stage.
3. Accept this as a real, informative negative result as-is: Phase 1's
   finding does not generalise unconditionally — it holds on easy tasks
   at this model scale, and the mechanism is honest about it (finds a
   consistent, explainable structure that simply isn't sufficient, rather
   than quietly looking fine) — and move to writing this up as a genuine,
   bounded finding rather than continuing to chase a fix.

### Cost

This 2-seed run: ~$17.80 confirmed via `list-billing` plus a small
unsynced tail (pods terminated ~02:00-02:15 UTC, final hour not yet
reflected) — estimated final total ~$19-20, comfortably within the user's
$30 top-up. Combined with Stage A's $8.27, total WikiText-103 investigation
cost to date: ~$27-28 of $55 (original) + $30 (top-up) = $85 total
authorized across both phases.

## 2026-10-06 — Phase 2 implementation: discrete birth/death structural plasticity

Gate 1→2 formally closed (bounded pass, see 2026-10-01 entry and the
research proposal's own Gate 1→2 annotation). Started Phase 2 per the
proposal's §11 ("Discrete birth/death"): start from a small seed and grow
via discrete lifecycle events, rather than starting over-complete and
pruning down. Plan at `/Users/samuelbartlett/.claude/plans/swirling-sleeping-hickey.md`.

**Design**: hybrid continuous substrate + discrete lifecycle overlay.
Phase 1's `SelfOrganizingTransformer`/`_gates()` mechanism is reused
unchanged (via thin subclass `PlasticTransformer` in the new
`plastic_model.py` — adds lifecycle hyperparameters only, no new
parameters, no forward-pass changes). A new `StructuralController`
(`plasticity.py`) layers a discrete state machine (`DORMANT → PROTECTED
→ MATURE → DEAD`) on top, mutating gate logits directly (birth = reset to
active init, death = hard-clamp) without touching the tensor math — chosen
over true sparse/conditional computation because the proposal explicitly
permits cost proxies in early phases (§9) and real sparse compute is a
much bigger systems lift for a 48M-param validation experiment. New
`train_phase2.py` reuses `train.py`'s `PackedTokens`/`BatchIterator`/
`load_tokens`/`evaluate_loss` (same pattern `retrain_compact.py` already
uses), calling the controller every `growth_interval` steps.

### Two real bugs found and fixed via local testing, before any cloud spend

1. **Plateau signal never built up.** `record_step` was only fed the
   sparse `eval_every`-cadence `val_loss`, but `growth_interval` fires far
   more often — `_loss_history` stayed essentially empty the whole run,
   so the loss-plateau term of the birth score (`w_p · P_i`) was always 0.
   Fixed by feeding every-step `train_loss` into an EMA instead, snapshotted
   once per growth event.
2. **Unbounded mass death.** Only births were rate-limited (`birth_limit`);
   a first end-to-end smoke test showed all 13 seed-mature units crossing
   the death threshold in the *same* growth tick and dying simultaneously
   — they shared the same init and uniform regularizer pressure, so
   nothing differentiated their timing. Added `death_limit`, symmetric to
   `birth_limit`: eligible-but-not-selected units keep their streak and
   remain eligible next check, so death still happens, just gradually
   (confirmed: 12 deaths spread across 6 growth events in the re-test,
   not 13 at once).

### One tuning discovery worth flagging explicitly

`birth_score_threshold`'s default (`0.3`, picked without empirical
grounding) was ~20–100x too high — achieved scores topped out around
0.003–0.014 throughout a 400-step local smoke test. This isn't a scale
artifact to wait out: `contribution_ema`'s ceiling is `sigmoid(dormant_floor_logit)`
≈ 0.018, a fixed function of hyperparameters, not of training progress —
more steps alone wouldn't have fixed it. Lowered the default to `0.01`,
confirmed empirically against the same local test (now gets real births).
Flagging this as **likely still needing re-tuning once real-scale dynamics
are observed** — exactly why the staged rollout's step 2 (single seed,
TinyStories, heuristic-only) exists before committing to the full matrix.

### Verification performed (staged rollout step 1, zero cloud cost)

- 7-case local correctness test on `StructuralController` directly (seed
  split ratio, heuristic birth on a forced high score, protection window
  immunity to death scoring, sustained-low-gate death after
  `death_patience`, random-mode respects `birth_limit`/dormant-only
  selection, well-formed JSON event log, `death_limit` caps simultaneous
  deaths).
- Full `train_phase2.py` smoke run (tiny config, CPU, 400 steps): real
  birth → protection → maturity → death cycling observed end-to-end, not
  just unit-tested in isolation.
- Confirmed `evaluate.py`'s new dispatch branch loads a Phase 2 checkpoint
  correctly.
- Confirmed `materialize.py` needs **zero changes** for Phase 2
  checkpoints — verified empirically, not just assumed from the plan:
  the thin-subclass design means `PlasticTransformer` has no parameters
  `SelfOrganizingTransformer.from_pretrained` wouldn't already know about.

### Status / next step

Committed (`8213f73`). Staged rollout step 2 (single seed, TinyStories,
heuristic-only, real scale) not yet run — next step, and the real test of
whether `birth_score_threshold` and the other hyperparameters need further
tuning before committing to the full 3-condition × 2-dataset matrix.

## 2026-10-06 (continued) — Staged rollout step 2: the plateau term causes a late-run churn loop

Ran `train_phase2.py` at real scale: single seed (42), TinyStories,
`birth_mode=heuristic`, same architecture/optimizer hyperparameters as
Phase 1's TinyStories runs (10000 steps, `target_cost=0.14`), pushed to
`sbartlett97/sot-phase2-tinystories-heuristic-seed42`.

### Steps 250–4250: healthy, selective plasticity

Births and deaths were sparse, bounded, and varied in location —
consistent with "demand-gated growth," the behaviour the design intends.

### Steps ~4250–10000: a steady churn loop

From step ~4250 onward, nearly every `growth_interval=250` event hit
`birth_limit=2`/`death_limit=2` exactly, concentrated heavily on layer 8's
(later also 7/9/10's) FFN groups, with units cycling birth→death on a
~1750-step lifespan. `state_counts` stopped moving — the population was
churning, not converging — and this continued unresolved to step 10000,
while the plateau score climbed from 0 to ~0.9–1.0 over the same window.

**Root cause** (diagnosed, then fixed — commit `285f8ce`): the birth
score's plateau term (`P_i` in proposal §11) was implemented as a flat
global scalar, added identically to every dormant candidate
(`w_p · plateau`). Training loss naturally flattens late in any normal
run, so once `plateau` grew large it alone pushed *every* dormant
candidate over `birth_score_threshold` simultaneously — collapsing
"selective, demand-gated growth" into "always max out `birth_limit`,
ranked by whichever candidate held a marginal `gradient_ema` edge that
tick." That's exactly the observed churn signature: same budget spent
every event, same handful of units cycling.

**The fix**: plateau now *modulates* each candidate's own gradient
pressure rather than being added as a flat bonus —
`b_i = w_g·G_i + w_c·C_i + w_p·plateau·G_i_normalized` — so a saturated
plateau amplifies units that already show demand instead of lowering the
bar for every dormant unit regardless of its own signal. This is also the
formulation that actually matches the proposal's own per-candidate `P_i`
subscript, which the flat-scalar version didn't. Verified locally: the
existing 7-case `StructuralController` test suite still passes, plus a
new 8th case that asserts a saturated plateau does *not* sweep up a
zero-gradient-pressure candidate.

**Important caveat**: this Stage 2 run used the *old* (buggy) formula —
the fix was written and verified afterward, not re-run yet. Its results
below are informative context, not a clean validation of the fixed
controller.

### Final state and evaluation

Final lifecycle state: `mature=271, dormant=503, dead=70, protected=8`;
totals: `66` births, `70` deaths across the run. Final train/val loss
1.7355/1.7737 (val ppl 5.89 — not directly comparable to the numbers
below, since training's `evaluate_loss` and `evaluate.py`'s harness-driven
`ppl()` are different code paths).

`evaluate.py` (held-out perplexity + lm-evaluation-harness) against the
final checkpoint: **perplexity 4.82**, hellaswag acc 0.2557/acc_norm
0.2607, arc_easy acc 0.2588/acc_norm 0.2483, piqa acc 0.5452/acc_norm
0.5212, winogrande acc 0.5099. Pushed successfully; the Hub card
(`sbartlett97/sot-phase2-tinystories-heuristic-seed42`) now carries this
performance section plus a comparison against the Phase 1 TinyStories
seed42 dense baseline (116.72 ppl), rendering as "beats dense by 95.9%."

**That comparison needs an explicit caveat, not a clean headline.**
Phase 1's TinyStories seed42 `evaluate.py`-reported perplexities (dense
116.72, sot 109.63, discovered_retrained 115.13) are themselves under
active, unresolved investigation: the same checkpoints' own
training-time validation loss (`metrics.json`, pulled from each repo) is
dramatically lower (e.g. sot: val_loss 1.546 → ppl ≈ 4.69 at step 10000,
vs. `evaluate.py`'s reported 109.63) — a ~20–25x gap that has been
reproduced locally against the real downloaded Phase 1 checkpoint, in
fp32, independent of the original CUDA/fp16 training run. Weight tying
and `architecture_mode` have been checked and ruled out as the cause;
investigation is ongoing (see below). Phase 2's own number, by contrast,
*is* internally consistent (evaluate.py ppl 4.82 vs. training val_ppl
5.89 — same ballpark, no gap), which narrows the Phase 1 issue toward
something specific to how those earlier checkpoints evaluate on reload,
not a general flaw in `evaluate.py` or in this reload path. Until that's
resolved, read "beats dense by 95.9%" as "Phase 2's real, trustworthy
number is ~4.8, compared against a Phase 1 baseline number that may
itself be significantly overstated" — directionally informative, not a
validated comparison.

### Recommendation

The plateau-churn bug is fixed in code but not yet validated by an actual
run. Before committing cloud spend to the full 3-condition × 2-dataset
matrix, the natural next step is re-running Stage 2 (same single seed,
TinyStories, heuristic mode) with the fixed formula, to confirm the churn
loop is actually resolved and not just locally plausible. **Not launched
yet — pending the user's decision.**

## 2026-10-06 (continued) — Transformers-version reload bug: Phase 1 TinyStories's headline result was an artifact

While checking on Phase 2 Stage 2's `evaluate.py` run, its perplexity (4.82)
looked suspiciously ~24x better than Phase 1 TinyStories's reported numbers
(109–139 ppl). Investigated why.

### Investigation

Ruled out, in order, with direct evidence against the real downloaded
Phase 1 seed42 `sot` checkpoint: weight tying (confirmed intact on reload),
`architecture_mode` (confirmed `"sot"`), the checkpoint's own saved
tokenizer (byte-identical to a fresh `"gpt2"` tokenizer on sample text),
`low_cpu_mem_usage` loading path (state dicts identical with it on/off),
raw tensor dtype (all `float32` on disk, no silent fp16 storage), gate
logit values (bit-exact to 7 significant figures against the checkpoint's
own `metrics.json` training log), and `train.py`'s own data-loading code
having changed since the run (it hadn't — last relevant commit predates
the training run). The decisive test: evaluating the reloaded checkpoint
on its own *training* data (which it was directly fit to, train_loss=1.52
at step 10000) *also* gave a high loss (4.61) — ruling out any
validation-split/data-pipeline explanation entirely, since the model
should show strong memorization on its own training data regardless of
any packing/chunking difference.

That left the environment itself. `config.json`'s `transformers_version`
field records which `transformers` build was active when a checkpoint's
`save_pretrained()` ran. Checking it across checkpoints: Phase 1 TinyStories
(all 18 checkpoints: 3 seeds × `dense`/`dense_8l`/`dense_half_ffn`/`sot`/
`sot_compact`/`discovered_retrained`) all show `5.17.0`. Phase 1
WikiText-103 (both target_cost sweeps, all conditions) and Phase 2 Stage 2
both show `4.57.6`/`4.49.0`. This project's own `requirements.txt` pins
`transformers>=4.49.0,<5.0` — **`5.17.0` is explicitly out of spec.** The
local `.venv` used throughout this session had also silently drifted to
`5.17.0`, matching the Phase 1 TinyStories pod's state and explaining why
every local reproduction attempt faithfully reproduced the same inflated
numbers.

**Confirmed root cause directly**: downgraded the local `.venv` to the
pinned-compliant `transformers==4.57.6` and reloaded the exact same Phase 1
seed42 `sot` checkpoint, byte-for-byte, no other change. `evaluate_loss`
came back `1.545914` — matching the original training-time value
(`1.545951`) to 5 decimal places (the tiny residual is ordinary
floating-point batch-order noise). Under `5.17.0` the same checkpoint,
same code, same data, gave `4.765` (ppl≈117). `transformers==5.17.0`
silently corrupts weight save/reload round-tripping for this custom gated
architecture — gate logits and tied embeddings happened to survive the
round trip intact (hence passing every earlier check), but something else
in the larger weight tensors did not.

**How the bad version got installed**: `pyproject.toml` declares
`dependencies = []` — empty. If any setup step ran `pip install -e .`
rather than (or before) `pip install -r requirements.txt`, nothing would
enforce the `<5.0` pin, and a later `pip install` of `lm-eval[hf]` or
`accelerate` (neither of which caps `transformers`) could silently resolve
the newest available release. This is a plausible, fixable root cause, not
just a one-off fluke — `pyproject.toml`'s dependencies are being synced to
match `requirements.txt` as part of this fix (see commit).

### Scope: only Phase 1 TinyStories is affected

| Run | `transformers_version` | Status |
|---|---|---|
| Phase 1 TinyStories, 3 seeds × 6 conditions (Sept 29) | `5.17.0` | **Corrupted — re-evaluated below** |
| Phase 1 TinyStories `sot_compact_gatefix`, 3 seeds (Sept 30) | `4.49.0` | Clean — gate-fold bug fix validation stands as originally reported |
| Phase 1 WikiText-103, both target_cost sweeps | `4.57.6` | Clean — the "bounded negative result" conclusion is unaffected, no correction needed |
| Phase 2 Stage 2 (2026-10-06) | `4.57.6` | Clean |

### Corrected results, all 3 seeds (perplexity)

Re-evaluated all 18 affected checkpoints locally (CPU, no retraining, no
GPU) under the now-pinned-compliant `transformers==4.57.6`, reusing
`evaluate.py`'s exact existing `ppl()` methodology (sequential chunking,
`seq_len=256`, `batch_size=8`, `max_batches=100`) so the *only* variable
that changed relative to the original numbers is the library version —
isolating the fix to the actual bug, not a methodology change. Did not
re-run the full `lm-eval` harness suite (slow on CPU, and already
established — correctly, independent of this bug — as flat/at-chance for
a 48M-param TinyStories-only model regardless of underlying quality, so
low expected value for the cost).

| Condition | Seed 42 | Seed 123 | Seed 7 |
|---|---:|---:|---:|
| dense | 4.63 | 4.61 | 4.64 |
| dense_8l | 4.83 | 4.80 | 4.81 |
| dense_half_ffn | 4.84 | 4.83 | 4.83 |
| **sot** | **4.75** | **4.74** | **4.75** |
| sot_compact (gate-fold fixed, already-clean numbers) | 881.24 | 525.03 | 349.24 |
| discovered_retrained | 4.89 | 4.96 | 4.97 |

(`sot_compact` here uses the already-clean `sot_compact_gatefix` repos, not
the original bug-compounded `sot_compact` repos, which remain corrupted by
*both* the gate-fold bug and this transformers bug and are not worth
re-evaluating separately.)

### Re-scored against README's six interpretation criteria

1. **SOT reaches similar or lower validation loss than the dense
   baseline.** ❌ **Reversed from the original pass.** SOT is consistently
   *worse* than dense in all 3 seeds (4.750 vs 4.625, +2.71%; 4.740 vs
   4.613, +2.75%; 4.748 vs 4.636, +2.41%). The gap is small (~2.4–2.8%)
   and consistent — arguably "similar" by a generous reading, but not a
   pass under the criterion as originally scored (which required SOT to
   *beat* dense, and it did, on the corrupted numbers).
2. **Learned cost falls substantially below 1.0.** ✅ Unaffected by this
   bug — cost still converges to ~0.125–0.129, an ~87% reduction.
3. **Re-training the discovered architecture from scratch recovers most of
   the SOT model's quality.** ✅ Still a pass in relative terms —
   discovered_retrained lands within ~2.9–4.7% of SOT itself (consistent
   with the original ~5–7% band), but it no longer beats dense (it now
   loses to dense by 5.7–7.6%, since SOT itself loses to dense).
4. **Stable across random seeds.** ✅ Still passes, and arguably *more*
   convincingly than before — the corrected dense-beats-sot gap is tightly
   consistent across all 3 seeds (2.4–2.8%), as is retrain's gap to SOT
   (2.9–4.7%).
5. **Not simply selecting maximum allowed capacity everywhere.** ✅
   Unaffected — still 6–7 of 12 layers kept, never all 12.
6. **Benchmark quality remains competitive after physical compaction, not
   only in the masked supernet.** ⚠️ Same qualitative verdict as before,
   less extreme magnitude: `sot_compact` (gate-fold-fixed, clean numbers)
   still collapses in all 3 seeds (349–881 ppl vs. ~4.7–4.9 for
   `discovered_retrained`) — retraining-from-scratch remains mandatory,
   not optional, just not by as dramatic a margin as the bug-compounded
   original 3240–23004 figures suggested.

**Net**: 3 of 6 criteria clear pass, 1 reversed to a fail (criterion 1), 2
unaffected. The project's actual, defensible Phase 1 TinyStories finding is
**"SOT reaches within ~2.6% of dense perplexity at ~13% of dense's
architectural cost, and this holds stably across 3 seeds"** — a real,
still-interesting compute-efficiency result, but not "SOT beats dense,"
which was always a surprising claim on its face (a gated subset of a dense
model's own capacity outperforming the ungated full model) that should
have been treated with more suspicion at the time.

### What this does and doesn't change

- **WikiText-103's "bounded negative result" is unaffected and stands as
  originally reported** — those checkpoints were evaluated under the
  clean `4.57.6`. If anything, this correction makes the TinyStories and
  WikiText-103 findings more consistent with each other: both now show
  SOT modestly *trailing* dense on raw perplexity rather than TinyStories
  showing an outright win and WikiText-103 showing a loss.
- **The gate-fold bug and its fix (2026-09-29/30) are unaffected and
  stand** — that investigation used `sot_compact_gatefix` checkpoints,
  saved under the clean `4.49.0`.
- **The Gate 1→2 annotation in the research proposal and README's "Phase 1
  result" section both cite the original, now-corrected numbers** — being
  updated in this same pass (see commits).
- **Phase 2 Stage 2's Hub card compared its real ppl (4.82) against the
  corrupted Phase 1 dense baseline (116.72)**, making the "beats dense by
  95.9%" line on that card misleading — corrected by re-pushing with the
  real dense baseline (4.63) as part of this fix.
- Every TinyStories Phase 1 Hub model card is being re-pushed with the
  corrected perplexity in this same pass; weights are untouched (only the
  evaluation summary in each card changes).

### Fix applied

- Local `.venv` downgraded to `transformers==4.57.6`.
- `pyproject.toml`'s `dependencies` synced to `requirements.txt` so an
  editable install can no longer bypass the pin.
- All 18 affected Hub model cards re-pushed with corrected perplexity.
- This log, `README.md`'s "Phase 1 result" section, and the research
  proposal's Gate 1→2 annotation corrected to reflect the real result.

## 2026-10-06 (continued) — Phase 2 Stage 2 re-run: churn fixed, but the clean result is a loss to dense

Re-ran Stage 2 (single seed 42, TinyStories, heuristic birth mode, same
hyperparameters as the original Stage 2 run) with the fixed per-candidate
plateau formula (`285f8ce`), on a fresh pod explicitly verified to be
running the pinned-compliant `transformers==4.57.6` before launch (RTX
4090, $0.74/hr — A100 was out of stock everywhere at launch time; total
cost for the ~2h run ≈$1.50).

### Churn loop: confirmed fixed

Checked every lifecycle event against the same signature that flagged the
original bug (a unit crossing birth→death repeatedly instead of the
population settling): **zero units had ≥4 lifecycle events** across the
full 10000-step run (65 births, 70 deaths total, spread across distinct
units throughout training, not concentrated on a repeat-cycling handful
the way the original run's layer-8 FFN groups were). The fix holds at real
scale, not just in the local unit tests.

### Clean result: Phase 2 loses to both Phase 1 conditions on perplexity

Final state: `mature=270, dormant=504, dead=70, protected=8`. Final train
loss 1.939, val loss 2.152 (val ppl 8.61 — training-time approximation,
not directly comparable, see below). `evaluate.py` perplexity (clean,
transformers-compliant, same methodology as the corrected Phase 1 numbers
above): **7.18**. Phase 1 TinyStories seed42, same corrected methodology:
dense 4.63, sot 4.75. **Phase 2's discrete-growth mechanism underperforms
both Phase 1 conditions on this single seed** — not a marginal gap (7.18
vs. 4.63–4.75 is ~51–55% worse), and the Hub card's own auto-generated
comparison line states this plainly ("loses to the matched dense baseline
(4.63) by 55.2%").

This is a genuine, clean negative result for Stage 2 as currently
configured — not an artifact of the churn bug (which is fixed) or the
transformers bug (this run verified compliant throughout). Plausible,
not-yet-investigated explanations, roughly in order of suspicion:
- `seed_fraction=0.33`'s initial MATURE population may simply be a worse
  starting point than Phase 1's fully-present (always-on, continuously
  gated) architecture — the discrete lifecycle model spends part of the
  10000-step budget growing into capacity Phase 1 has from step 0.
- `minimum_age=1000` is a long protection window relative to
  `growth_interval=250` — a newly birthed unit gets ~4 growth cycles of
  immunity before it can contribute to death-pass eligibility, which may
  be too conservative or too permissive; not yet checked which.
- Single seed — this could be seed noise rather than a structural
  disadvantage of discrete growth; the original (buggy) Stage 2 run and
  this clean one discovered different active-layer sets (originally
  `[2,5,6,7,8,9]`; now `[2,5,7,8,9,10]` — layer 6 swapped for layer 10),
  so the search itself isn't fully stable run-to-run even with the same
  seed and fixed hyperparameters, which the churn fix doesn't address.

### Recommendation

Do not commit to the full 3-condition (`heuristic`/`random`/`none`) ×
2-dataset matrix yet. The churn bug is fixed and confirmed, but this
single clean data point shows discrete growth losing to Phase 1's simpler
continuous gating, which is itself a result worth understanding before
scaling up — either investigate the three candidates above on this same
single seed first, or treat "discrete growth underperforms at this scale"
as a real (if provisional) finding and reconsider whether the full matrix
is the right next step versus adjusting Stage 2's hyperparameters first.
**Flagging for the user's decision, not proceeding unilaterally.**

**Superseded by the next entry** — the user correctly pushed back on
treating this one run as a verdict on the mechanism rather than on this
specific configuration, named `death_threshold` and `birth_limit` as the
two things to check, and was right on both counts.

## 2026-10-06 (continued) — Stage 2 re-run 2: fixing death eligibility turns the loss into a win

Investigated the user's hypothesis directly against this run's own data
before changing anything. `target_cost` turned out not to be the
constraint — `lagrange` stayed at `0.0000` for all 10000 steps and cost
topped out at `0.055`, nowhere near the `0.14` budget; the Lagrangian
penalty never activated once. What the data *did* show: mature population
grew 281→283 early, then **shrank to 270 by step ~4000 and sat flat there
for the remaining 60% of training**, with births and deaths locked in
near-lockstep every quartile (4/8, 19/20, 20/20, 22/22). Root cause:
`death_threshold=0.5` (absolute) assumed "good" gates reach ~0.5+, but
this architecture's gates settle around 0.2–0.35 even for genuinely kept
components (confirmed in both Phase 1's best checkpoint and this run's
own final gates: head_gate 0.14–0.19, ffn_gate 0.13–0.17) — nearly the
entire mature population read as "should die" every check, regardless of
merit, and growth could never outpace that.

**Fix** (commit `d421ff3`): `death_relative_margin` replaces
`death_threshold` — a unit is death-eligible only once its gate falls
below `margin` (default 0.5) × its own kind's (layer/head/ffn_group)
current mature-population median, recomputed every check. Same relative-
threshold fix pattern as `materialize.py`'s earlier `--threshold 0.5`
bug. Verified against 9 local test cases (including a new one
confirming a known, accepted limitation: if an entire kind's mature
population crashes together, the criterion can't see it, since the
median collapses with them) before any cloud spend. Also doubled
`birth_limit` 2→4 per the user's request, since the original run's births
were close to saturated (65 of a possible 80 used).

### Result: a clean win, and the death mechanism wasn't even exercised

Re-ran on a fresh pod, transformers compliance re-verified before launch.
Mature population actually grew this time — 283→342 by step 7000, then
leveled off — and **zero deaths occurred in the entire 10000-step run**
(59 births, 0 deaths, confirmed in `events.json`). Final architecture
cost 0.072 (still well under the 0.14 budget — `lagrange` stayed 0.0 the
whole run again, so cost still isn't the active constraint; the Stage 1
run's growth just plateaued around step 7500 on its own).

`evaluate.py` perplexity: **4.40** — beats Phase 1's corrected dense
(4.63) by 4.9% and sot (4.75) by 7.4%, on the same seed, same clean
methodology. The Hub card's own comparison line confirms it: "beats the
matched dense baseline (4.63) by 4.9%." This is the first result in
either phase where the gating/plasticity mechanism actually beats dense
outright under corrected numbers, not just "trails by a small margin at
lower cost."

**Honest caveat**: zero deaths happened, so this run only tested the
*birth* half of the lifecycle — the relative-threshold fix was effective
enough (combined with the higher birth_limit letting the population reach
a comfortable size before growth naturally plateaued) that death was
never actually triggered. This is a genuinely good result, but it's not
yet evidence that the DEATH side of discrete birth/death lifecycle works
correctly at scale — only that removing its false-positive bug stopped it
from being actively harmful. A future run that pushes the architecture
closer to its cost budget (e.g. a higher `target_cost` or `seed_fraction`,
so growth genuinely contends for capacity) would be a better test of
death actually pruning something real.

### Updated recommendation

This single clean win is enough to justify the full 3-condition ×
2-dataset matrix being back on the table, but it's still one seed. Given
death was never exercised, worth at minimum running the `random` birth-
mode control (same fixed controller, uniform-random candidate selection)
on this same seed before committing to the full matrix, to confirm the
heuristic scoring is adding value over random growth and not just
benefiting from more capacity regardless of *which* units get it. Flagging
for the user's decision on whether to proceed to the full matrix now or
run that control first.
