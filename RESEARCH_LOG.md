# Research log

This file is the full record of the Self-Organising Transformer
experiment. It has two parts:

1. **Summary** (this part, rewritten on 2026-10-16). It states what we
   know now, which numbers are valid, and what comes next. Read it first.
2. **Dated entries**, in time order. They are the record as it was
   written at the time. Later entries correct earlier ones. Where an early
   entry and the summary disagree, the summary is correct.

See `README.md` for the design of the experiment and a quickstart.

## How to read the numbers

Four rules apply to every number in this log:

- **Two perplexity methods.** `evaluate.py` ppl uses sequential chunks
  of the validation split (100 batches of 8 x 256 tokens). Training-time
  val ppl uses random validation batches during training. The two give
  different values for the same model (dense 768: 21.98 and 24.30). Do
  not compare a number from one method with a number from the other.
- **"Cost" is not compute.** Before 2026-10-14, every "cost" figure
  (for example "~87% lower cost") is `architecture_cost()`, a
  gate-weighted proxy. It is about 9x too low for the structures it
  describes. The gated supernet always runs 1.0 of dense FLOPs. Real
  FLOPs come from `compute.py` (2026-10-14).
- **Phase 2 val ppl before 2026-10-15 is supernet val ppl.** Dormant
  units (gate about 0.04) and dead units (gate about 0.001) were still in
  the forward pass and still trained. With them forced off, attempt 8
  goes from 26.67 to 132 val ppl. These numbers do not show the quality of
  the discovered structure.
- **TinyStories Phase 1 numbers before 2026-10-06 are wrong.** A
  `transformers==5.17.0` install corrupted the reload of these models.
  The corrected values are in the 2026-10-06 entry and below. The
  TinyStories `results/*_summary.json` files still hold the faulty
  values (see `results/README.md`).

## Summary

### Phase 1: differentiable gates

An over-complete Transformer (12 layers) has a sigmoid gate on each
layer, attention head, FFN channel group and short skip edge. A
Lagrangian budget (`target_cost`) pushes the gate-weighted cost down. The
controls are dense baselines (full, 8 layers, half FFN), the physical
compact model with inherited weights, and the compact structure
retrained from scratch.

**TinyStories, 12L / d 384, 3 seeds** (`evaluate.py` ppl, corrected
2026-10-06):

| condition | seed 42 | seed 123 | seed 7 |
|---|---:|---:|---:|
| dense | 4.63 | 4.61 | 4.64 |
| dense 8 layers | 4.83 | 4.80 | 4.81 |
| dense half FFN | 4.84 | 4.83 | 4.83 |
| SOT (gated supernet) | 4.75 | 4.74 | 4.75 |
| SOT compact, inherited weights | 881.24 | 525.03 | 349.24 |
| discovered structure, retrained | 4.89 | 4.96 | 4.97 |

**WikiText-103, 12L / d 384** (`evaluate.py` ppl):

| condition | tc 0.14, seed 42 | tc 0.17, seed 42 | tc 0.17, seed 123 |
|---|---:|---:|---:|
| dense | 28.23 | 28.23 | 28.11 |
| dense 8 layers | 30.14 | 30.14 | 29.98 |
| dense half FFN | 30.50 | 30.50 | 30.45 |
| SOT (gated supernet) | 29.28 | 29.01 | 29.05 |
| discovered structure, retrained | 31.88 | 32.41 | 33.15 |

**WikiText-103, 12L / d 768, seed 42** (2026-10-08): dense 21.98, SOT
22.00 (`evaluate.py` ppl); training-time val ppl 24.30 and 24.44.

Findings:

- **SOT trails dense by a small, stable margin.** It is 2.4-2.8% behind
  on TinyStories and 2.8-3.7% behind on WikiText-103 at d 384. At d 768
  the gap is 0.1%. It never beats dense.
- **The cost saving is not proven.** The gate proxy reads 0.12-0.13, but
  the supernet runs all of its tensors. Only the physical compact model
  is cheaper, and with inherited weights it collapses (ppl 349-881 on
  TinyStories).
- **A discovered structure retrained from scratch loses to the dense
  model of the same envelope.** On WikiText-103 it also loses to the
  dense 8-layer and half-FFN baselines.
- **A gate value is not importance** (2026-10-09). Ablation on real
  checkpoints: rank correlation of loss change with gate is -0.72 for
  layers, +0.2 for heads and about 0 for FFN groups. Gate-based
  selection drops layer 0, the most important layer. A Taylor estimate
  `|g * dL/dg|` tracks the ablation (rho 0.93 / 0.53 / 0.56). With layers
  0-1 kept, inherited-weight ppl on TinyStories seed 42 falls from 881
  to 15.8. A 300-step fine-tune recovers most of the rest (5.24-6.47).
- **The selected depth is consistent.** On TinyStories, all 3 seeds keep
  6-7 middle layers and drop layer 0 and layers 9-11. On WikiText-103
  (tc 0.17), both seeds keep layers 5-10 with fewer heads toward the
  output.

### Phase 2: discrete birth and death

Training starts from a seed structure (33% of units MATURE, the rest
DORMANT). Every 250 steps a controller gives birth to the best dormant
units and kills mature units that fall behind their kind. Lifecycle:
DORMANT -> PROTECTED -> MATURE -> DEAD. All runs below are WikiText-103,
12L / d 768, seed 42, 10,000 steps. The val ppl is training-time val ppl
of the **supernet** (see "How to read the numbers"). The Phase 1 dense
768 reference on the same method is 24.30.

| attempt | Hub repo suffix | main change | births | deaths | val ppl, step 10000 |
|---|---|---|---:|---:|---:|
| 1 | (none), deleted | first run at d 768 | 68 | 0 | 27.27 |
| 2 | `-v2`, deleted | relative birth ranking; a NaN stopped growth at step 6500 | 100 | 0 | 25.77 |
| 3 | `-v3`, deleted | NaN fix, no limit on growth | 160 | 0 | 31.89 |
| 4 | `-v4` | birth taper and score floor, looser death | 98 | 0 | 28.08 |
| 5 | `-v5-imp` | death by importance, not gate | 98 | 76 | 35.71 (29.95 at 9500) |
| 6 | `-v6-allfixes` | birth and death ramps, dead-layer rule | 97 | 28 | 29.28 (27.29 at 9000) |
| 7 | `-v7-lrfreeze` | no deaths below 5% of peak LR | 97 | 23 | 30.66 |
| 8 | `-v8-tailfreeze` | no births either below 5% of peak LR | 92 | 23 | 26.98 |
| 9 | `-v9-zeroeffect` | inactive units have zero effect; **present-structure val ppl** | 92 | 60 | **30.61** |

(Attempt 1 births: its `events.json` has 68. The 2026-10-08 entry says
59, which is the TinyStories figure.)

On 2026-10-16 we deleted 7 superseded Hub repos: attempts 1-3, the
Phase 2 TinyStories run, and the three TinyStories `sot_compact` repos
from before the gate-magnitude fix (replaced by `sot_compact_gatefix`).
The logs of the four Phase 2 runs are in `results/phase2_v1_base/`,
`phase2_v2_relrank/`, `phase2_v3_unblocked/` and `phase2_tinystories/`.

An earlier Phase 2 run on TinyStories (d 384, 59 births, 0 deaths) gave
`evaluate.py` ppl 4.40 against dense 4.63. It is also a supernet number,
with every dormant unit still active. It does not show a cheaper model.

Findings:

- **A death rule on gate values never fires.** In attempts 1-4 the
  worst unit of each kind sat at 84-93% of its kind's median gate, above
  every margin we tried. This is the same fault as in Phase 1: the gate
  does not measure importance.
- **Death by importance works.** Attempt 5 killed 76 units, among them
  5 layers. Layer 0 survived.
- **Abrupt structure changes cause val ppl spikes.** Gradual birth and
  death ramps (250 steps) and a rule that skips units inside dead layers
  removed every spike (attempt 6).
- **Structure changes in the LR tail hurt.** With LR below 5% of peak,
  the network cannot adapt to a new or lost unit. Freezing births and
  deaths there gave the best supernet result (attempt 8: val ppl falls
  at every eval to 26.98).
- **The present structure of attempt 8 is 0.466 of dense FLOPs/token**
  (0.472 of params). The lm_head alone is 25% of dense FLOPs.
- **But the supernet depends on its inactive units.** With every
  non-present unit forced off, attempt 8 goes from 26.67 to 132.42 val
  ppl (attempt 7: 30.25 to 244.29). The 706 dormant FFN groups cause most
  of the gap. Dead units also drifted back up after their death ramps.
- **Fix (2026-10-15).** A dormant or dead unit now has scale 0: exact
  zero output and exact zero gradient. A zero-valued probe reads the
  birth signal dL/dg without changing the output. `build_from_plastic()`
  builds the physical model of the present structure, and it matches the
  supernet to 2.4e-7.
- **Attempt 9 (2026-10-16) is the first run on this code.** Its present
  structure alone gets 30.61 val ppl at 0.425 of dense FLOPs/token, and it
  writes fluent text ([samples](docs/samples.md)). Attempt 8's present
  structure got 132 at 0.466. It is 26% behind dense 768 (24.30).
- **Dense models at matched compute beat attempt 9 (2026-10-16).** Dense
  d 768 / 4 layers gets 30.18 and dense d 512 / 9 layers gets 28.98, both at
  0.42 of dense FLOPs. Learned structure does not help yet.

### Bugs that changed results

| date | bug | effect |
|---|---|---|
| 2026-09-29 | tied `lm_head` re-initialised on every `from_pretrained` | evaluation of a random head |
| 2026-09-29 | compaction dropped gate magnitude | compact ppl 3240-23004 instead of 280-795 |
| 2026-09-30 | `retrain_compact.py` and `evaluate.py` ignored `--dataset_name` | found before use; would have used TinyStories |
| 2026-10-06 | `transformers==5.17.0` corrupts reload | all TinyStories Phase 1 ppl wrong (corrected) |
| 2026-10-08 | birth score depended on plateau timing | growth stopped at step 7500 (attempt 1) |
| 2026-10-08 | one NaN gradient poisoned the birth EMA | growth stopped at step 6500 (attempt 2) |
| 2026-10-09 | compaction dropped every skip edge | compact models lost all skips |
| 2026-10-14 | cost proxy is gate-weighted | every "cost" figure about 9x too low |
| 2026-10-14 | dormant and dead units not off | all Phase 2 ppl measure the supernet |

## Where next

In order:

1. **Attempt 9 on the zero-effect code.** Done (2026-10-16): 30.61 val
   ppl at 0.425 of dense FLOPs, present structure only.
2. **A dense control at matched compute.** Done (2026-10-16): dense
   d 768 / 4 layers 30.18, dense d 512 / 9 layers 28.98, both at 0.42 of
   dense FLOPs. Both beat attempt 9 (30.61).
3. **Attempt 10: attempt 9 plus `--birth_score_norm flops`.** It tests if
   the head-over-FFN birth bias causes part of the gap. A Phase 2 run must
   beat 28.98 to show that learned structure helps.
4. **Fresh retrain of the attempt-9 structure.** This separates a good
   structure from good inherited weights, as in Phase 1.
5. **Physical removal during training.** Dead units now have no effect,
   so they can leave the tensors. Dormant units still run because the
   probes need their outputs. The next step is a cheaper birth signal
   (probe a sample of dormant units, or rebuild tensors at each growth
   check). Only then does training compute fall below 1.0 of dense.
6. **Seeds.** Repeat the best Phase 2 configuration on seeds 123 and 7.
7. **Fix the ablation helpers.** `ablation_importance.set_logit` turns a
   unit off through its logit. On new checkpoints it must set the scale
   to 0.
7. **Benchmarks.** Run lm-eval on the Phase 2 models of the present
   structure. The d 384 models are at chance on every task, so these
   benchmarks tell us nothing at that scale.

## Contents of the dated entries

- 2026-09-28 — Initial audit, pipeline fixes, and target_cost diagnosis
- 2026-09-29 — First full-scale run (RunPod, seed 42) and a tied-weights bug
- 2026-09-29 — Seed 123 complete, seed 7 parallel run, storage cleanup
- 2026-09-29 — Seed 7 complete: full 3-seed comparison
- 2026-09-29 — Residual-stream-scale mismatch confirmed and fixed (bug, not co-adaptation)
- 2026-09-30 — WikiText-103 scale-up kicked off (dataset before model size)
- 2026-10-01 — target_cost=0.17 re-run (2 seeds): the regression is not a budget-tuning problem
- 2026-10-06 — Phase 2 implementation: discrete birth/death structural plasticity
- 2026-10-06 (continued) — Staged rollout step 2: the plateau term causes a late-run churn loop
- 2026-10-06 (continued) — Transformers-version reload bug: Phase 1 TinyStories's headline result was an artifact
- 2026-10-06 (continued) — Phase 2 Stage 2 re-run: churn fixed, but the clean result is a loss to dense
- 2026-10-06 (continued) — Stage 2 re-run 2: fixing death eligibility turns the loss into a win
- 2026-10-08 — Width scale-up: resolving the WikiText-103 capacity shortfall
- 2026-10-08 (continued) — Phase 2 at scale: zero deaths, and why
- 2026-10-08 (continued) — birth_score_floor: relative ranking replaces the absolute threshold
- 2026-10-08 (continued) — Tuning attempt: tapered growth + loosened death, still zero deaths
- 2026-10-09 — Death-streak-accumulation diagnostic: eligibility never fires, not a patience/streak problem
- 2026-10-09 (continued) — Gate value is not importance: ablation test on the attempt-4 checkpoint
- 2026-10-09 (continued) — Re-materialising Phase 1 seed 42 with layers 0-1 kept
- 2026-10-09 (continued) — Short fine-tune of the re-materialised compact models
- 2026-10-09 (continued) — Skip-edge compaction bug fixed
- 2026-10-09 (continued) — Fine-tune re-run with the skip fix
- 2026-10-09 (continued) — Importance-based death criterion
- 2026-10-13 — Attempt 5: importance-based death at full scale
- 2026-10-13 (continued) — Birth and death skip units in dead layers
- 2026-10-14 — Gradual death: the gate ramps out
- 2026-10-14 (continued) — No growth check whose ramps cannot finish
- 2026-10-14 (continued) — Phase 2 attempt 6: all four fixes at full scale
- 2026-10-14 (continued) — No death in the LR tail
- 2026-10-14 (continued) — Attempt 7: the LR-tail fault is births, not deaths
- 2026-10-14 (continued) — No birth in the LR tail
- 2026-10-14 (continued) — Attempt 8: no structure change in the LR tail
- 2026-10-14 (continued) — Compute accounting: the cost proxy is wrong, and dormant units are not off
- 2026-10-15 — Dormant and dead units: exact zero effect, zero gradient

---

# Dated entries

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
proposal (§5.2; the proposal is not in this repo)
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
anything).
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

Per the plan's decision point: this is
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
pruning down.

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

## 2026-10-08 — Width scale-up: resolving the WikiText-103 capacity shortfall

Stage 2's win had zero deaths across the whole run — population grew and
plateaued on its own, never under real capacity pressure. Decision (user):
rather than run the previously-proposed `random`-mode control at the
current (apparently too-easy) scale, scale up model capacity and dataset
difficulty together to find a regime where growth genuinely contends for
budget.

**Design**: width-only 2x scale-up (`d_model` 384→768, `max_heads` 6→12,
`max_ffn_dim` 1536→3072, `ffn_groups` 64→128, `max_layers` unchanged at
12 — holds `head_dim`=64 and FFN `group_size`=24 constant, so it's more
of the same granularity, not coarser), on WikiText-103 (already
integrated, already has baseline numbers at the old scale to compare
against). 47.7M → 152.1M params (3.19x), computed exactly via
`model.py`, not estimated. Verified the real target shapes through both
the Phase 1 and Phase 2 code paths locally (CPU, a few steps) before any
cloud spend — no shape errors, dense-mode cost correctly reads 1.0,
Phase 2 registers exactly 1692 units (12 × (1+12+128)).

### Equilibrium + baselines: the shortfall is resolved

Equilibrium diagnostic (same method as the original TinyStories/WikiText-103
calibrations: `target_cost=0.45`, inert, observe where cost naturally
settles): **0.177** — remarkably close to both TinyStories' 0.186 and the
old-scale WikiText-103's 0.190, suggesting this equilibrium is more a
property of the gate/regularizer dynamics than of model scale.
`target_cost=0.13` (75% of equilibrium, same convention as before).

Dense + sot baselines, single seed, scaled model, WikiText-103, `evaluate.py`:

| condition | ppl (evaluate.py) | ppl (training-time val) |
|---|---:|---:|
| dense | **21.98** | 24.30 |
| sot | 22.00 (loses to dense by 0.1%) | 24.44 |

Old scale (12L/384d) dense was **28.23** ppl (2026-09-30 entry) — the
width scale-up alone dropped dense ppl by ~22%, confirming the original
"capacity limit at this model scale" diagnosis and giving real headroom
for a gating mechanism to prove itself against, rather than everyone
losing to an undersized ceiling. Sot essentially matches dense at ~12%
of the cost — the strongest compute-efficiency result in either phase.

### Process note: an idle-pod mistake, owned

After the first scale-up pod (dense+sot baselines) finished, it sat idle
for ~3 hours (finished 17:45 UTC, something eventually stopped it around
20:50 UTC) before I got back to it, costing ~$2.20 for nothing — I got
pulled into the resume-feature conversation and the pod review and never
circled back to terminate it. The pod was gone entirely by the time I
checked (not just stopped), taking the hung second attempt's local-only
files with it (not a real loss — final dense/sot results were already
pushed to the Hub before this happened). Exactly the "never leave a pod
idle" mistake this project already learned once; recommitted to
terminating promptly, and did so consistently for the rest of today's
pods.

### A recurring, unresolved issue: Trackio's remote sync hangs mid-training

Wired in 2026-10-06 (commits around `b5d4b0d`) for live metrics. Twice
today, a training process froze completely (GPU idle at ~15W despite
VRAM still held, process sleeping on `futex_wait_queue`) — once during
the dense+sot concurrent run (both processes stuck simultaneously,
originally misdiagnosed as GPU/VRAM contention), once during a solo sot
retry at 94% complete. Direct investigation (`ss -tnp` on the second
hang) found multiple `CLOSE-WAIT` TCP connections to HuggingFace's
backend IPs with unread data sitting in the receive queue, plus one
`ESTABLISHED` connection matching the hang — a genuine network-level
hang in Trackio's HTTP sync, called synchronously from inside the main
training loop, so a hung call there blocks training entirely. Not yet
root-caused further; `--no_trackio` used for every run for the rest of
today pending a proper fix (likely: move the sync off the training
thread, or wrap it in a hard timeout).

**Mitigation built in response**: `train.py` now supports `--resume_from
<checkpoint_dir>` (commit `e96ee71`) — loads weights via
`from_pretrained`, recovers step/optimizer-state/Lagrange-multiplier from
a rolling `training_state.pt` written alongside each checkpoint, and
continues `metrics.json` rather than overwriting it. Verified locally: a
10-step run resumed to 20 continues the Lagrange multiplier and loss
smoothly, no cold-restart wobble. Would have saved both of today's lost
runs had it existed sooner. `train_phase2.py` doesn't have this yet —
worth adding if Trackio's hang isn't fixed first.

## 2026-10-08 (continued) — Phase 2 at scale: zero deaths, and why

Phase 2 (heuristic), single seed, scaled model, WikiText-103,
`target_cost=0.13`, `birth_limit=4` (doubled per the Stage 2 pattern).

**Result**: final ppl 27.27 (training-time val) — worse than both scaled
Phase 1 baselines, unlike Stage 2's TinyStories win. Zero deaths, 59
births, population flat at 613 mature from step ~7500 onward, final cost
0.063 — well under the 0.13 budget, meaning there was still headroom the
mechanism simply never used.

### Diagnosing why growth stalls: plateau timing, not capacity

Investigated directly against this run's data (and cross-checked against
Stage 2's TinyStories win) before changing anything, per the user's
question about decoupling capacity growth from loss-plateau timing:

- This run: `plateau` was 0.243/0.205/0.237 at steps 6750/7000/7250 (28
  births occurred in that window), then crashed to exactly `0.0` at step
  7500 and stayed there — zero births for the rest of training.
- Stage 2's TinyStories win: birth scores climbed steadily (0.0104 →
  0.0132 → ... → 0.047) right up to step 6000, then stopped dead.

Both tell the same story: `plateau = max(0, 1 - |slope|×50)` over a
rolling 2500-step window swings to exactly 0 whenever loss is still
improving measurably, and only provides its amplification boost during
genuine flat spells. Growth was riding a plateau-driven wave of amplified
scores in both runs, then the model re-entered a phase of real
improvement, the boost vanished, and the *raw* `w_g·gradient_ema +
w_c·contribution_ema` signal alone wasn't enough to clear the fixed
`birth_score_threshold=0.01` for any candidate still dormant — because a
dormant unit's gate sits deep in the sigmoid's saturated tail
(`dormant_floor_logit=-4.0`), so its own gradient is structurally small
*regardless of whether activating it would help*. Same class of bug as
the already-fixed absolute `death_threshold=0.5`: an absolute bar this
architecture can only clear under specific, plateau-driven timing
conditions, not a genuine measure of demand.

## 2026-10-08 (continued) — birth_score_floor: relative ranking replaces the absolute threshold

Fix (commit `fb0a790`, mirrors `death_relative_margin`'s fix): birth
eligibility is now the top `birth_limit` dormant candidates by raw score
each check, excluding only genuinely zero-signal ones
(`birth_score_floor=0.0`), rather than requiring everyone to clear a
fixed bar only reachable during plateau windows. Also added
`growth_log.json` to every checkpoint: every growth check's full
diagnostic (plateau, max score, state counts), not just checks that
produced a birth/death — previously only printed to stdout and lost the
moment a pod was torn down (exactly what made the plateau-crash
correlation above need reconstruction from a check I'd luckily captured
before terminating the first scaled Phase 2 pod). Verified against 10
local test cases (9 prior + a new one confirming the actual fix: a
modest-score candidate that would have failed the old absolute threshold
now gets born under `plateau=0`, purely by being top-ranked) before any
cloud spend.

### Re-run: excellent for 62% of training, then a second, different bug

Births fired at **every single check** from step 250 to 6250 (26
consecutive growth events, 4 births each, `plateau=0.0` almost the
entire time) — unambiguous confirmation the relative-ranking fix works.
Mature population climbed steadily 545→629, fully decoupled from
plateau.

Then at step 6500, `max_dormant_birth_score` went `NaN` and stayed `NaN`
for the remaining 3500 steps (35% of training) — zero births, population
frozen at 645, even as `plateau` later climbed to 0.85-0.91 near the end
(which, pre-freeze, would very likely have triggered more growth). Final
ppl 25.77 — worse than Phase 1's baselines, but a real improvement over
the first scaled attempt's 27.27, purely from the extra growth that
happened before the freeze.

**Root cause**: an EMA can never recover from one non-finite update
(`ema_decay*NaN + ... ` is `NaN` forever after) — a single gradient
overflow slipping through before the scaler catches it permanently
poisons that unit's `gradient_ema`. Worse, Python's `sort()` is
unreliable on a list containing `NaN` (`NaN` comparisons are always
`False`, breaking the total-order assumption sort relies on), so a
poisoned candidate left in the list before sorting can scatter into the
top-`birth_limit` slots — where it then correctly fails the floor check
and gets rejected, but in doing so can crowd out a genuinely good
candidate sorted below it purely due to the broken comparison.

**Fix** (commit `fbdb44d`): `record_step()` now skips the EMA update
entirely for a non-finite gradient rather than propagating it (prevents
poisoning at the source); `_birth_pass()` filters non-finite scores out
*before* `sort()`, not after (defense in depth for anything
already-poisoned, e.g. a resumed older run). Added a `nan_candidates`
diagnostic to the growth-check record so this is visible in
`growth_log.json` if it ever recurs. Verified against 2 new local test
cases (12 total) before any further cloud spend.

### Re-run 2: both bugs fully confirmed fixed — and a new tension surfaces

**Fully clean**: 0 of 40 growth checks had any `nan_candidates`, 0 of 40
had zero births — every single check across the entire run produced
growth, including passing cleanly through step 6500 (the exact point the
previous run died: birth event fired normally there,
`max_dormant_birth_score=0.059`, `nan_candidates=0`). Total births:
**160** = `4 × 40`, the theoretical maximum under `birth_limit=4`. Both
fixes verified at real scale, not just locally.

But: final ppl **31.89** — worse than *every* prior scaled Phase 2
attempt (27.27, 25.77) and much worse than Phase 1's baselines
(21.98/22.00). Mature population reached 689 (more than either previous
attempt), cost only reached **0.085** (still well under the 0.13 budget
even running at full capacity every single check the entire run — the
growth *rate* is simply too slow relative to a 10000-step budget to ever
fill it). Still zero deaths, anywhere, in any of the three attempts.

**Reading**: `birth_score_floor=0.0` ("any positive signal at all") is
such a low bar that it essentially never excludes anyone, so the
controller now *always* fills `birth_limit=4` every check regardless of
whether the remaining dormant pool has anything genuinely worth growing.
Once the clearly-useful candidates are used up, later growth is drawn
from increasingly marginal signal — and with zero deaths anywhere to
prune anything back, those additions just accumulate permanently, likely
diluting the architecture rather than improving it. The relative-ranking
fix correctly solved the bug it targeted (plateau-timing dependence) but
surfaced a different, previously-masked problem: "top-K every time" is
not the same as "only grow when it's actually worth it."

### Status / open question

Not yet decided. Candidates worth investigating, not yet tried: raise
`birth_score_floor` to a modest-but-real positive value (not back to an
unreachable absolute bar, just enough to exclude truly marginal
candidates); taper `birth_limit` down as training progresses so early
growth stays aggressive but late-stage growth tapers off naturally; or
accept that 10000 steps / this growth rate isn't enough budget for the
population to reach a natural capacity ceiling where death would
actually trigger, and extend training length instead. Flagging for the
user's decision before further cloud spend.

## 2026-10-08 (continued) — Tuning attempt: tapered growth + loosened death, still zero deaths

User's call: implement all three candidates from the status above at
once (raise `birth_score_floor`, taper `birth_limit`, loosen death
eligibility), then run once rather than sweep them individually.

**Implementation** (commit `db9f6d7`): `--birth_limit_end` linearly
tapers `birth_limit` from its starting value to this one over training
(`controller.birth_limit` mutated directly in `train_phase2.py`'s loop
before each `controller.update()` call, logged as `birth_limit_used` in
`growth_log.json` for visibility). `--birth_score_floor` and
`--death_patience` were already real `StructuralController` parameters
but had never been exposed via CLI — wired through now. Verified locally
first (40-step/8-check smoke test): `birth_limit_used` tapered correctly
4→3→3→2→2→2→1→1, the floor correctly blocked low-score checks,
`nan_candidates` stayed 0 throughout.

**Run config**: `birth_score_floor=0.005` (≈`contribution_ema`'s ceiling
`sigmoid(dormant_floor_logit)`×`w_c` — a real structural quantity in this
scoring scheme, not an arbitrary guess), `birth_limit` 4→1 tapered,
`minimum_age=500` (halved from 1000), `death_relative_margin=0.6`
(loosened from 0.5), `death_patience=2` (loosened from 3). Same scale as
the prior three attempts (768d width, WikiText-103, `target_cost=0.13`).

### Result: the taper worked exactly as designed, death still never fired

`birth_limit_used` tapered cleanly across the run: 4, 4, 3, 3, 3, 2, 2,
2, 2, 1 (sampled every 4th check) — mechanically verified at real scale,
not just locally. `nan_candidates=0` and at least one birth occurred on
every single one of the 40 checks (the floor thinned growth, never fully
blocked it). Total births: **98** — well below the unconstrained
attempt's 160, confirming the taper+floor combination measurably reduces
indiscriminate late-stage growth.

**But still zero deaths** — the fourth consecutive attempt, now with
every death-eligibility knob loosened at once. Final ppl **28.08**.

Full comparison across all four real-scale attempts, same seed/scale/dataset:

| Attempt | Total births | Deaths | Final ppl |
|---|---:|---:|---:|
| 1 — plateau-gated (bug), stopped early by accident | 59 | 0 | **27.27** |
| 2 — relative ranking fixed, NaN froze it at step 6500 | 100 | 0 | 25.77 |
| 3 — both bugs fixed, fully unblocked (no taper/floor) | 160 | 0 | 31.89 |
| 4 — tapered + floored + loosened death | 98 | 0 | 28.08 |

Taper+floor helped *relative to the unconstrained case* (31.89→28.08,
consistent with the "indiscriminate growth dilutes quality" reading from
the prior entry) but still underperforms attempt 1 — which, despite
having the *least* total growth of any attempt, got there by an
accidental bug-induced early stop, not by design. Loosening every death
knob simultaneously (halved `minimum_age`, raised `death_relative_margin`
0.5→0.6, lowered `death_patience` 3→2) produced no deaths at all, same as
every previous configuration.

### Reading: this probably isn't a parameter-tuning problem anymore

Four attempts, four different configurations spanning a wide range of
birth/death looseness, zero deaths in all of them. Nudging
`birth_score_floor`/`birth_limit_end`/`death_relative_margin`/
`death_patience` further seems unlikely to change this on its own —
these knobs have now been pushed a meaningful distance in the
permissive direction without effect. More likely explanation: within a
10000-step budget, this architecture appears to stay in "still growing
into capacity" mode for the *entire* run rather than transitioning to
"has enough, start pruning" — mature population climbed every single
quartile in every attempt, never plateauing on its own the way it would
need to for death's relative-median comparison to find a stable,
sustained "behind its peers" signal (the median itself is a moving
target while the population is still actively growing). Candidates for
next time, not yet tried: a genuinely longer training budget (more steps
for the population to reach a natural ceiling); or a different
diagnostic pass over `growth_log.json`/`events.json` specifically
checking whether individual units' death-streak counters are even
starting to accumulate (vs. resetting every check) before assuming more
loosening would help. Not decided — flagging for the user.

## 2026-10-09 — Death-streak-accumulation diagnostic: eligibility never fires, not a patience/streak problem

The user picked the second candidate from the entry above: check directly
whether death's `death_patience`-based streak mechanism is the actual
blocker, rather than guessing further.

### Instrumentation

`_death_streak` (`plasticity.py`) was already tracked per-unit but never
surfaced anywhere — `events.json`/`growth_log.json`/Trackio only ever saw
*births* and *deaths*, not the eligibility checks that produced neither.
Added four fields to `_death_pass()`/`update()`'s returned dict, purely
additive (no behavior change):

- `below_floor_count` — how many old-enough mature units fail
  `gate_val < kind_median * death_relative_margin` *this check*.
- `death_eligible_pool` — the denominator (old-enough mature units
  checked at all).
- `max_death_streak` — the highest *current* streak across those units
  this check (explicitly NOT a running historical peak — a unit that
  dips below the floor then recovers must show this reset to 0, verified
  by a dedicated local test).
- `gate_spread` — per kind (`layer`/`head`/`ffn_group`), `{min, median,
  max}` of mature gate values this check.

Verified locally first (test13 added to the local `StructuralController`
test suite, 13/13 passing, zero cost): a unit forced below its kind's
floor shows `max_death_streak=1` after one check and `below_floor_count
>= 1`; once it recovers, both report 0 on the very next check, confirming
the fields track live state rather than a historical peak.

### Run

Reused attempt 4's exact config (768-width WikiText-103 scale,
`birth_limit` 4→1 tapered, `birth_score_floor=0.005`, `minimum_age=500`,
`death_relative_margin=0.6`, `death_patience=2`) so the diagnostic reads
the same run shape that produced the zero-death result under
investigation — just shortened to `--max_steps 3000` (~10 growth checks,
enough to see the pattern settle) instead of a full 10000, since this
only needed to observe a mechanism, not reproduce a full training
trajectory. `--no_trackio`, no `--push_to_hub` (throwaway diagnostic).

**Infrastructure note**: this run hit two unrelated container/host
issues before it could start, both on RunPod's community cloud tier —
worth recording since they'll recur. First pod: the account's previously-
registered SSH key had no matching local private key (generated on a
different machine/session), so `startSsh` access failed outright;
regenerated a dedicated local keypair and registered it via `runpodctl
ssh add-key`. Second, same pod, after fixing SSH: `torch.cuda.is_available()`
returned `False` ("CUDA unknown error") despite `nvidia-smi` working
fine — caused by the container exposing the physical GPU as
`/dev/nvidia6` instead of `/dev/nvidia0`, which CUDA's runtime API (unlike
NVML-based `nvidia-smi`) requires for sequential enumeration; a
`ln -sf /dev/nvidia6 /dev/nvidia0` symlink didn't fix it on that host, so
the pod was torn down and recreated (new host, same fix needed —
`/dev/nvidia1 -> /dev/nvidia0` worked there). Third, same new pod: direct
(non-proxy) SSH rejected the correct key with "Permission denied" because
the container image ships `/root` as world-writable (`777`), which trips
sshd's `StrictModes` pubkey-auth guard regardless of correct
`authorized_keys` permissions — fixed with `chmod 755 /root` via the
(separately-authenticating) `ssh.runpod.io` proxy shell, which uses
account-level auth rather than the container's sshd and so wasn't
affected by the same bug. None of these reflect a problem with this
project's own setup — all three are quirks of the specific community-
cloud host/image combination landed on. Total infra cost from two
false-start pods before the real one: $0.21 (8¢ + 12¢), real diagnostic
run: 2.1 hours, $0.46. Total: **$0.67**.

### Result: eligibility never fires — not once, in 2750 steps of eligibility window

| step | births | deaths | below_floor_count | eligible pool | max_death_streak |
|---:|---:|---:|---:|---:|---:|
| 250  | 4 | 0 | 0 | 0 (not yet aged in) | 0 |
| 500  | 4 | 0 | 0 | 545 | 0 |
| 750  | 3 | 0 | 0 | 545 | 0 |
| 1000 | 3 | 0 | 0 | 549 | 0 |
| 1250 | 3 | 0 | 0 | 553 | 0 |
| 1500 | 2 | 0 | 0 | 556 | 0 |
| 1750 | 2 | 0 | 0 | 559 | 0 |
| 2000 | 2 | 0 | 0 | 562 | 0 |
| 2250 | 2 | 0 | 0 | 564 | 0 |
| 2500 | 2 | 0 | 0 | 566 | 0 |
| 2750 | 1 | 0 | 0 | 568 | 0 |
| 3000 | 1 | 0 | 0 | 570 | 0 |

`below_floor_count` is **0 at every single check**, from the moment the
eligible pool first populates (step 500, 545 units) through the end (step
3000, 570 units). `max_death_streak` never leaves 0 — not one unit, in
2750 steps of eligibility window across a growing population, has ever
had its gate fall below `death_relative_margin(0.6) * its kind's median`,
even for a single check.

`gate_spread` at the final check explains why directly:

```
layer:     min=0.265  median=0.315  max=0.336   (min/median = 84%)
head:      min=0.264  median=0.282  max=0.323   (min/median = 93%)
ffn_group: min=0.262  median=0.299  max=0.310   (min/median = 88%)
```

The *worst* unit of any kind still sits at 84-93% of its kind's median —
nowhere near the 60% floor `death_relative_margin=0.6` sets. The mature
population's gate values are simply too homogeneous for a
40%-below-median criterion to ever separate a "clear loser" from the
pack, regardless of how lenient `death_patience` is.

### Conclusion: this is failure mode (a), not (b) — a margin/homogeneity problem, not a streak-reset problem

The two candidate explanations named when this diagnostic was proposed:
(a) gates never spread enough for eligibility to fire at all, vs. (b)
eligibility fires but streaks keep resetting before `death_patience`
completes. The data is unambiguous: **(a)**. `death_patience`'s
streak-accumulation mechanism was never even exercised in this run —
eligibility itself (`below_floor_count > 0`) never happened once, so
there was nothing for a streak to accumulate or reset. All four prior
attempts' loosening of `death_patience` (3→2) and `minimum_age` (1000→
500) could not possibly have helped, because neither knob is downstream
of the actual blocker.

This reframes the open question from "how should births/death timing be
tuned" to a different, more basic one: **is a population-relative
median threshold the right death criterion at all**, given that this
architecture's gate values appear to converge to a narrow band (roughly
±15% of the kind median, based on this run) rather than spreading out
enough to expose clear underperformers. Candidates worth considering,
not yet tried: a much less conservative `death_relative_margin` (e.g.
0.85-0.9, tuned to what this run's actual spread shows, rather than the
previous 0.5→0.6 progression which was never close to the real range);
an absolute-ish criterion scaled to the observed spread rather than a
flat fraction of the median; or accepting that this architecture may
simply not produce the degree of inter-unit differentiation needed for
relative pruning to work within a 10000-step budget, which would argue
for the "longer training budget" option from the previous entry instead.
Not decided — flagging for the user.

## 2026-10-09 (continued) — Gate value is not importance: ablation test on the attempt-4 checkpoint

The previous entry found death never fires because mature gates sit in a
narrow band. This entry tests a stronger hypothesis: **gate magnitude does
not measure a unit's importance at all**, so no `death_relative_margin`
value can fix death.

### Why this was expected

`architecture_cost()` and `sparsity_loss()` (`model.py:178-195`) push on
sigmoid gate *magnitude*. Each gate multiplies an output that the next
linear layer can rescale freely (`g·W` is what the forward pass sees), so
the model can shrink every gate and grow the weights behind it at no loss
cost. Nothing in the objective makes gates bimodal, and the gate
temperature annealing from proposal §10 is not implemented. A uniform
gate band around 0.3 is the expected equilibrium. A second consequence:
every "cost" figure so far (Phase 1 ≈0.12, Phase 2 0.06-0.085) is a
gate-magnitude proxy. Every unit runs in every forward pass, so real
training and inference FLOPs of the gated `sot` and Phase 2 models equal
the full supernet's. (Phase 1's `sot_compact` and `discovered_retrained`
models are physically smaller and do save real compute.)

### Method

New script `ablation_importance.py`. For a sample of units on a trained
checkpoint it:

1. measures the true importance: the change in held-out loss when the
   unit's gate is forced to exactly 0, on fixed validation batches;
2. computes a first-order Taylor estimate `|g·∂L/∂g|` (Molchanov-style,
   one backward pass);
3. reports Spearman rank correlation of true Δloss against gate value and
   against the Taylor estimate, per kind.

Active units are sampled at evenly spaced gate quantiles (30 per kind, all
12 layers). 10 dormant units per kind are ablated as a control. Ran
locally on MPS (no pod, ~5 and ~10 minutes), twice on disjoint batch
samples: `--batch_seed 1234` (8×8×256 tokens) and `--batch_seed 99`
(16×8×256 tokens). Results in `results/ablation_phase2_v4*.json`.

### Result

| kind | ρ(Δloss, gate) run A / B | ρ(Δloss, Taylor) run A / B | Δloss test-retest ρ | Taylor(A) vs Δloss(B) ρ |
|---|---:|---:|---:|---:|
| layer (12) | **−0.74 / −0.72** | +0.91 / +0.91 | 0.99 | 0.93 |
| head (30) | +0.28 / +0.20 | +0.46 / +0.64 | 0.95 | 0.53 |
| ffn_group (30) | +0.02 / −0.22 | +0.56 / +0.68 | 0.60 | 0.56 |

1. **Gate value does not predict importance.** For FFN groups the
   correlation is zero. For heads it is weak. For layers it is strongly
   *negative*: the early layers have the lowest gates (0.26) and are by far
   the most important (zeroing layer 0 adds +5.0 to loss; layers 9-11 add
   +0.05-0.07 with gates of 0.39-0.48). A gate-based death rule that
   worked would kill layer 0 first.
2. **Importance itself is strongly spread out.** The gates are uniform, but
   true Δloss is not: layers span 0.046 to 5.0, heads span −0.007 to
   +0.030 (some heads *hurt* — removing them lowers loss), FFN groups span
   0.0002 to 0.0025. A death rule on real importance has plenty of signal
   to rank with.
3. **The Taylor estimate tracks true importance at about the noise
   ceiling.** It predicts Δloss on *unseen* batches with ρ 0.93 / 0.53 /
   0.56 (layer/head/FFN). For FFN groups that is close to the 0.60
   test-retest ceiling of the Δloss measurement itself; single 24-channel
   groups are too small to measure precisely on ~16-33k tokens.
4. **Dormant units are not inert.** Dormant gates drifted from the 0.018
   floor to 0.03-0.05 during training, and their output weights scale up
   behind them. A dormant head's mean Δloss (0.0007-0.0008) is about a
   third of an active head's median (0.0024-0.0028); 53% of sampled active
   heads matter less than the most important dormant head. So "dormant"
   units do part of the computation, and the population-level DORMANT /
   MATURE split is partly fictional.

### Conclusion

The zero-deaths result is a design error, not a tuning problem: the death
criterion reads a quantity (gate magnitude) that does not carry the
information it needs. It also explains why "dormant" capacity still
helps Phase 2's loss.

### Cross-check on Phase 1: gate selection drops the most important block

Same script on `sbartlett97/sot-phase1-tinystories-sot-seed42`
(`results/ablation_phase1_tinystories_seed42.json`, `--dormant_per_kind 0`):

| kind | ρ(Δloss, gate) | ρ(Δloss, Taylor) |
|---|---:|---:|
| layer (12) | +0.40 | +0.73 |
| head (30) | +0.80 | +0.83 |
| ffn_group (30) | +0.38 | +0.41 |

Phase 1's gates carry more signal than Phase 2's (heads especially), so
the effect is weaker there, not absent. The layer result is the important
one. Layer 0 has one of the lowest gates (0.206) and is by far the most
important block: zeroing it adds **+3.0** to loss (baseline 1.44), layer 1
adds +0.58, every layer from 6 up adds under +0.12. Seed 42's
materialisation kept `[2..8]`, so it dropped layers 0 and 1 — the two
most important blocks in the model. Layer 0 alone is worth a factor of
about e^3 ≈ 20 in perplexity, which accounts for most of seed 42's
inherited-weight collapse (4.75 → 881.24 ppl after the gate-fold fix,
corrected clean figure).
All three TinyStories seeds excluded layer 0, so the same mechanism very
probably drives the collapse in all of them (only seed 42 tested here).
The fresh-retrain condition is not affected by this: it trains the kept
shape from scratch, so it never depends on the dropped weights.

### Recommended next steps

1. If the gate substrate is kept at all, base death and materialisation
   selection on an EMA of the Taylor importance `|g·∂L/∂g|`, not on gate
   value. A cheap test: re-materialise Phase 1 seed 42 keeping layers 0-1
   and see how much of the inherited-weight collapse disappears. It costs nothing extra: the gradient is already computed.
2. Add real FLOP/parameter accounting for the active structure, and
   compare growth runs against dense at matched *training FLOPs*.
3. For the actual project goal (start from one block, no dormant slots),
   replace the gated supernet with physical growth: resize tensors (with
   AdamW state), zero-init new output projections so births are
   function-preserving, and score candidate units by the gradient on
   their zero-init output weights (GradMax-style). Prior work to check
   first: GradMax (Evci et al. 2022), Net2Net (Chen et al. 2016),
   progressive stacking (Gong et al. 2019), masked structural growth
   (Yao et al. 2023), self-expanding networks (Mitchell et al. 2023).

Caveat: two checkpoints (Phase 2 attempt 4, Phase 1 TinyStories seed 42).
The Phase 1 768-wide `sot` checkpoint is not yet tested.

## 2026-10-09 (continued) — Re-materialising Phase 1 seed 42 with layers 0-1 kept

Direct test of the previous entry's claim: gate-based materialisation
dropped the most important blocks, and that caused most of the
inherited-weight collapse.

**Change**: `materialize.py --layers=<i,j,...>` (and a `layer_indices`
argument on `compact_model.build_from_sot`) replaces the gate-threshold
layer choice with an explicit list. Head and FFN-group choice inside each
layer still uses the threshold. Default behaviour is unchanged.

**Correction to an earlier figure**: the published gate-fix compact model
(`sot-phase1-tinystories-sot_compact_gatefix-seed42`, 881.24 ppl clean)
kept layers `[2..7]` (6 layers). `[2..8]` was the earlier, pre-fix compact
model. Rebuilding with the default median threshold reproduces `[2..7]`
exactly.

**Result** (`evaluate.ppl`, TinyStories validation, 204k tokens, inherited
weights, no retraining; `results/remat_phase1_tinystories_seed42.json`):

| variant | layers kept | ppl |
|---|---|---:|
| sot (full gated model) | all 12 | 4.75 |
| gate selection (previous method) | `[2..7]` (6) | 881.25 |
| gate selection + layers 0-1 | `[0..7]` (8) | **15.83** |
| top-6 by ablation Δloss (same size as gate) | `[0..5]` (6) | 85.59 |

(The gate variant's 881.25 matches the corrected clean figure for this
checkpoint, 881.24, in the 2026-10-06 transformers-correction table. The
794.77 cited in the entries above is the older, pre-correction number.
Corrected 2026-10-09 — an earlier version of this note wrongly put the
difference down to token count.)

1. **Keeping layers 0-1 cuts inherited-weight perplexity by 56x** (881 →
   15.8). The claim holds: dropping the low-gate, high-importance early
   layers caused most of the collapse.
2. **At equal size, importance-based selection beats gate-based by 10x**
   (85.6 vs 881). This is the controlled comparison: same 6-layer budget,
   only the selection signal differs.
3. **Single-unit ablation scores are not additive.** The six layers
   dropped from `[0..5]` each cost at most +0.11 loss alone (sum ≈0.40),
   but dropping them together costs +2.89. Layers depend on each other, so
   ranking by one-at-a-time importance is a good first signal, not a full
   answer. Growth and pruning decisions should re-measure after each
   change, not rank once.
4. A gap to the full model remains (15.8 vs 4.75 at 8 of 12 layers). Some
   is genuine capacity loss; some is that weights co-adapted with the
   dropped layers 8-11. A short fine-tune of the compact model would
   separate the two. Not run.

The fresh-retrain condition (`discovered_retrained`) is unaffected by any
of this; it never uses inherited weights. But the *architecture* it
retrains was also chosen by gate, so it also excluded layer 0's position
in the stack — whether that matters for a from-scratch retrain is a
separate, open question (block position means less when weights are new).

## 2026-10-09 (continued) — Short fine-tune of the re-materialised compact models

Question: of the gap between each compact model (inherited weights) and the
full model, how much is lost capacity and how much is weights that relied
on the removed parts (co-adaptation)? A short fine-tune recovers
co-adaptation loss quickly; lost capacity it cannot recover.

**Change**: `retrain_compact.py --keep_weights` fine-tunes the inherited
weights instead of reinitialising (Hub condition name `compact_finetuned`).

**Run**: RunPod, Secure RTX 4090 ($0.89/hr, EU-RO-1, ~30 min, ≈$0.45,
pod deleted on completion). Same seed-42 checkpoint revision as the
previous entry. 300 steps each, lr 1e-4 (30 warmup, cosine), batch 16 ×
grad_accum 8 × 256 tokens, 100k TinyStories train docs. Before/after ppl
via `evaluate.ppl` (204k tokens). Script `results/finetune_logs/run_ft.sh`;
results `results/finetune_phase1_tinystories_seed42.json`.

**Control added**: all 12 layers with `--threshold 0.17` (keeps every head
and FFN group, drops every skip edge, like the other variants). This
isolates a second bug found while building it — see below.

| variant | layers | ppl before | ppl after 300 steps | reference (10k steps from scratch) |
|---|---|---:|---:|---|
| all 12 layers, no skips | 12 | 10.55 | **4.92** | sot (with skips) 4.75 |
| gate + layers 0-1 | `[0..7]` (8) | 15.83 | **5.24** | dense_8l 4.83 |
| top-6 by ablation Δloss | `[0..5]` (6) | 85.61 | **5.80** | discovered_retrained `[2..7]` 4.89 |
| gate selection | `[2..7]` (6) | 881.17 | **6.47** | discovered_retrained `[2..7]` 4.89 |

(References are the corrected seed-42 numbers from the 2026-10-06 table.
Validation loss was still falling at step 300 in every run, so "after"
values are upper bounds.)

1. **Most of the collapse was co-adaptation, not lost capacity.** 300
   steps (3% of the original training budget) take gate selection from
   881 to 6.5 and `[0..7]` from 15.8 to 5.2.
2. **Selection still matters after the fine-tune.** At equal size (6
   layers), importance selection reaches 5.80 against gate selection's
   6.47 — about 10% lower ppl, down from 10x before fine-tuning. Choosing
   by importance keeps more recoverable function.
3. **Inherited weights + a short fine-tune do not yet beat a fresh
   retrain.** The 6-layer variants (5.80 / 6.47) still trail the 6-layer
   `discovered_retrained` (4.89 after 10k steps), and `[0..7]` (5.24)
   trails `dense_8l` (4.83). With the curves still falling, a longer
   fine-tune may close this; not tested.
4. **The cost of removing 4 layers is small once co-adaptation is
   recovered**: `[0..7]` 5.24 vs the 12-layer control 4.92 (+6.5% ppl)
   under the same fine-tune.

### Second compaction bug: skip edges are restored at full strength or dropped

`CompactTransformer.forward` (`compact_model.py:91-93`) adds each kept
skip edge as `previous[src]/sqrt(skip_span)`, without the trained skip
gate. The SOT model scales each skip by its gate (`model.py`, ≈0.12 mean,
0.16 max for this checkpoint). So:

- at the usual median threshold (≈0.22), every skip gate falls below the
  threshold and **every compact model built so far has silently lost all
  skip paths**;
- at threshold 0, every skip comes back at gate 1.0, about 8x too strong
  (ppl 156,336).

Dropping the skips alone, with all 12 layers kept, raises ppl from 4.75
to 10.55. After 300 fine-tune steps it recovers to 4.92, so this bug is
mostly co-adaptation too, but it inflates every inherited-weight compact
number to date. Same class as the earlier gate-fold bug. Fixed in the
next entry.

## 2026-10-09 (continued) — Skip-edge compaction bug fixed

**Fix** (`compact_model.py`):

- `build_from_sot` now keeps every skip edge whose source layer is also
  kept, and stores each edge's trained gate in a new per-layer `skip_gate`
  list aligned with `skip_from`. It reads the raw `skip_gate` values from
  `export_architecture()`'s output, so that JSON contract is unchanged.
  Skips are no longer filtered by the layer threshold (skip gates train
  at ~0.12, far below the ~0.22 layer threshold). A skip edge costs one
  add, so keeping weak ones costs nothing.
- `CompactTransformer.forward` multiplies each skip by its `skip_gate`.
  Specs without `skip_gate` (every checkpoint built before this fix) keep
  their old behaviour.

**Verification** (Phase 1 TinyStories seed 42, revision `73d297f`):

- Compact model keeping every layer, head and FFN group vs the SOT model,
  same random batch: max |Δlogit| = 4.3e-5 (largest logit ≈15), both in
  memory and after `save_pretrained`/`from_pretrained`. Materialisation
  is now exact when nothing is pruned.
- Published pre-fix checkpoint `sot_compact_gatefix-seed42` still
  evaluates to 881.25 (backward compatible).

**Inherited-weight ppl with skips restored** (no fine-tune;
`results/skipfix_phase1_tinystories_seed42.json`):

| variant | skips dropped (before fix) | skips kept with gates |
|---|---:|---:|
| all 12 layers | 10.55 | **4.75** (= sot) |
| gate + layers 0-1 `[0..7]` | 15.83 | 13.20 |
| top-6 by ablation `[0..5]` | 85.61 | 44.87 |
| gate selection `[2..7]` | 881.17 | 1727.39 |

Restoring skips helps every variant that keeps layers 0-1, and makes the
gate-selected `[2..7]` worse. Likely reason: with layers 0-1 removed, the
residual stream feeding layers 2+ is already wrong, and the restored
skips add more of that wrong signal. Untested. For pruned models,
inherited-weight ppl without a fine-tune measures mismatch, not quality.
The fine-tuned numbers in the previous entry are the better comparison.
Those runs used the pre-fix code (no skips); re-run with the fix in the
next entry — results unchanged within 0.02 ppl.

**Side effect on future runs**: `retrain_compact.py` builds fresh models
from the compact config, so `discovered_retrained` architectures now
include skip edges with fixed (non-trainable) gate scales. Every earlier
`discovered_retrained` run had no skips. New and old retrain numbers are
not directly comparable.

## 2026-10-09 (continued) — Fine-tune re-run with the skip fix

Same protocol as the earlier fine-tune entry (300 steps, lr 1e-4, same
checkpoint revision), re-run with the skip-gate fix (`9e199cc`). RunPod
Secure RTX 4090, US-NC-1, ~35 min, ≈$0.52, pod deleted on completion.
Results `results/finetune_skipfix_phase1_tinystories_seed42.json`.

| variant | before (skips fixed) | after 300 steps (skips fixed) | after 300 steps (pre-fix run) |
|---|---:|---:|---:|
| all 12 layers | 4.75 | 4.87 | 4.92 |
| gate + layers 0-1 `[0..7]` | 13.19 | **5.23** | 5.24 |
| top-6 by ablation `[0..5]` | 44.88 | **5.82** | 5.80 |
| gate selection `[2..7]` | 1726.66 | **6.46** | 6.47 |

1. **After a fine-tune, the skip bug makes no measurable difference** for
   the pruned variants (all within 0.02 ppl of the pre-fix run). The skip
   paths are weak (gates ≈0.12) and 300 steps compensate either way. The
   bug mattered for inherited-weight numbers without fine-tuning and for
   exact reproduction, not for these conclusions. The earlier fine-tune
   entry's findings stand.
2. **The fine-tune protocol itself costs about 2.5%.** The 12-layer
   control now starts exactly at sot's 4.75 and ends at 4.87. Re-warming
   the learning rate on an already-converged model, on a 100k-doc subset,
   perturbs it slightly. So the right baseline for the pruned variants is
   4.87, not 4.75: removing 4 layers (`[0..7]`) costs +7.4% ppl
   (5.23 vs 4.87); removing 6 costs +19.5% (`[0..5]`, importance) or
   +32.6% (`[2..7]`, gate).
3. Importance selection still beats gate selection at equal size after
   fine-tuning (5.82 vs 6.46), and all fine-tuned variants still trail
   fresh 10k-step retrains of similar size (`dense_8l` 4.83,
   `discovered_retrained` 4.89).

## 2026-10-09 (continued) — Importance-based death criterion

Follows from the ablation finding that gate value does not track
importance. Death now ranks units by measured importance instead of gate
value.

**Change** (`plasticity.py`, `train_phase2.py`):

- `UnitTelemetry.importance_ema`: an EMA of the first-order Taylor
  estimate `|g·∂L/∂g|`, the estimated loss change from removing the unit.
  `record_step` computes it from the gradient it already reads:
  `g·∂L/∂g = ∂L/∂logit / (1−g)`. No extra backward pass. Non-finite
  gradients skip the update, like `gradient_ema`.
- `StructuralController(death_criterion=...)`: `"importance"` (new
  default) compares `importance_ema` against `death_relative_margin` × the
  kind's mature-population median. `"gate"` keeps the original gate-value
  rule for comparison runs. `death_patience`, `death_limit` and
  `minimum_age` work as before.
- `update()` now also reports `death_criterion` and `importance_spread`
  (per-kind min/median/max of `importance_ema`) in `growth_log.json`.
- CLI: `train_phase2.py --death_criterion {importance,gate}`.

**Tests**: the controller suite now lives in the repo at
`tests/phase2_controller_test.py` (it was previously only in a session
scratchpad). Tests 1-13 pin `death_criterion="gate"` and still pass. New
tests 14-18 cover the importance path: the EMA matches the Taylor formula
on a real backward pass; a low-importance unit dies even with a high
gate while a low-gate, high-importance peer survives; a gate-only dip no
longer makes a unit eligible; diagnostics report the criterion and
spread; an unknown criterion is rejected. 18/18 pass
(`.venv/bin/python tests/phase2_controller_test.py`).

**Smoke run** (local, tiny 4-layer/128-wide model, 400 steps, TinyStories
3k docs, `growth_interval=25`, `minimum_age=50`, margin 0.5, patience 2):
32 births, **4 deaths** — the first deaths in a real training loop since
the relative-margin change. Importance min/median ratios ran 0.42-0.78
by kind. Mechanism check only, not a quality result.

**Expected behaviour at full scale**: the Taylor scores in the attempt-4
ablation (`results/ablation_phase2_v4_batchseed99.json`) have min/median
of 0.55 (layers), 0.12 (heads), 0.60 (FFN groups). At margin 0.5, about a
third of mature heads would be eligible and no layers or FFN groups.
That sample was chosen by gate quantile and estimated from 16 batches, so
the training-time EMA spread may differ. `death_limit` caps the rate.

**Next**: a real-scale Phase 2 run (attempt-4 config, 768-wide,
WikiText-103) with `--death_criterion importance`, compared against
attempt 4 (28.08 ppl, 0 deaths). Not run yet.

## 2026-10-13 — Attempt 5: importance-based death at full scale

**Run**: attempt 4's exact config (768-wide, WikiText-103, seed 42,
`birth_limit` 4→1, `birth_score_floor=0.005`, `minimum_age=500`,
`death_relative_margin=0.6`, `death_patience=2`, `death_limit=2`) with
one change: `--death_criterion importance` (commit `4d3149d`). Hub tag
`768d013-v5-imp`. RunPod community cloud, A100 SXM 80GB ($1.39/h), 1.27
steps/s, so about 2.2 h for 10k steps. (No community 4090 stock. The one
RTX 5080 host in stock failed `torch.cuda.is_available()` on two pods;
both were deleted.) Checkpoint JSONs are copied to
`results/phase2_v5_importance/` every 500 steps, because Phase 2 pushes
to the Hub only at the end and the host has maintenance at 18:00 BST.

### Deaths fire from the first eligible check

At step 500, 232 of 545 old-enough mature units were below the floor.
Under the gate rule this count was 0 at every check of the diagnostic
run. The first deaths came at step 750, the earliest step that
`death_patience=2` allows. Every check since then has used its full
`death_limit` of 2.

| step | births | deaths | dead layers | val ppl v5 | val ppl attempt 4 | gap |
|---:|---:|---:|---|---:|---:|---:|
| 1000 | 16 | 4 | — | 127.8 | 115.7 | +10.4% |
| 2000 | 30 | 12 | 10, 9, 11 | 54.5 | 54.0 | +0.9% |
| 3000 | 42 | 20 | 10, 9, 11 | 40.4 | 39.7 | +1.8% |
| 4000 | 54 | 28 | 10, 9, 11 | 34.9 | 34.2 | +1.9% |
| 4500 | 60 | 32 | + 7 | 35.9 | 32.4 | +10.8% |
| 5000 | 65 | 36 | 10, 9, 11, 7 | 32.2 | 30.9 | +4.3% |
| 5500 | 69 | 40 | 10, 9, 11, 7 | 30.7 | 29.7 | +3.3% |
| 6000 | 73 | 44 | 10, 9, 11, 7 | **143.1** | 28.6 | — |
| 6500 | 77 | 48 | 10, 9, 11, 7 | 34.4 | 90.5 (spike) | — |

What died (by step 4000: 22 heads, 3 layers, 3 FFN groups):

- **Late layers first**: layer 10 (step 1250), 9 (1500), 11 (1750), 7
  (4500). This matches the ablation: late layers carried about +0.05
  loss each, and layer 0 carried +5.0. Layer 0 is still alive.
- **Heads**: most deaths. This matches the calibration estimate (head
  importance min/median 0.12 vs 0.55-0.60 for the other kinds).
- Births went mostly to FFN groups (39 of 54 by step 4000). The model
  moves capacity from depth into width.

### Three design faults found

1. **Births are not function-preserving** (the step-6000 spike). Birth
   sets the logit from about -4 to -1 in one step, so the gate jumps from
   about 0.02 to 0.27. The unit's weights were never trained to
   contribute at that scale. At step 6000 the controller gave birth to
   heads 0.3 and 1.10, the first head births in the two most important
   layers. Train loss stayed flat at 3.40 (logged before the update), but
   val ppl measured just after the update was 143.1. It recovered to 34.4
   within 500 steps. Attempt 4's unexplained step-6500 spike (90.5) is
   probably the same effect.
2. **Death is a one-step cut.** Layer 7's death at step 4500 matches a
   val ppl rise from 34.9 to 35.9 while attempt 4 fell to 32.4. Recovery
   took about 1000 steps.
3. **The lifecycle ignores the parent layer.** After layer 11 died
   (step 1750), the controller spent death slots on FFN groups inside it
   (11.79, 11.97, 11.66, 11.93, 11.20). These deaths change nothing.
   It also gave birth to 6 FFN groups in layer 7 at steps 4250-4500
   (layer 7 died at step 4500), 2 more at step 5750, and killed 4 FFN
   groups inside dead layer 7 at steps 5750-6000. Units inside a dead
   layer must be neither born nor killed.

### Next

- Fix 1 (this entry's follow-up): ramp a born unit's gate in over
  `birth_ramp_steps` instead of one step.
- Fixes 2 and 3 are not started.

### Final result (step 10000)

Run time 2.2 h. Pod cost about $4.70 (A100, 3.4 h including setup),
plus $0.05 for the two failed RTX 5080 pods. Model:
`sbartlett97/sot-phase2-wikitext103-heuristic-seed42-tc768d013-v5-imp`.

| | attempt 4 (gate death) | attempt 5 (importance death) |
|---|---:|---:|
| births / deaths | 98 / 0 | 98 / 76 |
| deaths by kind | — | 46 FFN groups, 25 heads, 5 layers |
| dead layers | none | 4, 7, 9, 10, 11 |
| mature units at end | 641 | 565 |
| cost proxy at end | 0.0777 | 0.0523 |
| val ppl, step 9500 | 28.6 | 29.95 (+4.7%) |
| val ppl, step 10000 | **28.08** | **35.71** |

- **The final number is not a fair reading.** Layer 4 died at the
  step-10000 check, just before the final eval. Val ppl went 29.95 →
  35.71 with no steps left to recover. Step 9500 is the fair comparison:
  +4.7% ppl for a 33% lower cost proxy and 5 of 12 layers removed. The
  cost proxy is gate-weighted, not real compute (see
  the 2026-10-09 ablation entry).
- **Fault 3 is the largest waste.** 58 of the 71 head/FFN deaths hit
  units inside a layer that was already dead. Only 13 deaths removed
  live capacity. The `death_limit` budget went mostly to no-ops.
- Every structure change at a check right before an eval shows up as a
  spike (steps 4500, 6000, 8500, 10000). A death or birth with no
  recovery time is the common cause.

### Next

1. Fix fault 3: skip units in a dead layer for both birth and death.
2. Fix fault 2: ramp death down like birth, or stop structural changes
   in the last part of training (a freeze window before the end).
3. Re-run attempt 5 with fixes 1-3 (the ramp-up is already in the code).

## 2026-10-13 (continued) — Birth and death skip units in dead layers

Fixes fault 3 from attempt 5: 58 of its 71 head/FFN deaths hit units in
layers that were already dead.

**Change** (`plasticity.py`):

- `_in_dead_layer(uid)`: true for a head or FFN group whose layer is
  DEAD. DEAD is final and the layer's output is gated off, so such a unit
  can never affect the loss again.
- Death skips these units. They also leave the kind-median pools: their
  importance is about 0, so they pulled the median down and made live
  units harder to kill.
- If a layer dies in a pass, its own units cannot take the remaining
  death slots of that pass. The slots go to the next live eligible units.
- Birth (heuristic and random) skips dormant units in dead layers.
- `update()` reports `in_dead_layers`: non-dead units stranded in dead
  layers.

Units in a *dormant* layer are not changed. A dormant layer can still be
born, so its units can matter later.

**Tests**: 25/25 pass. New tests 22-25 cover: no birth in a dead layer
(heuristic and random); no death in a dead layer and no orphan in the
median; a same-pass layer death does not give its slots to its own
units; the diagnostic count. Each new test fails on the old controller.
Test 24 also fails with only the same-pass guard removed.

**Smoke run** (local, tiny model, 400 steps): layer 3 died at step 150.
After that, the controller made no births or deaths in layer 3. All 7
later deaths hit live layers. `in_dead_layers` = 18.

**Next**: gradual death (or a structure freeze before the end), then
re-run attempt 5 with all three fixes.

## 2026-10-14 — Gradual death: the gate ramps out

Fixes fault 2 from attempt 5: death cut the logit to -8 in one step.
Layer 7's death at step 4500 cost about 1000 steps of val recovery.

**Change** (`plasticity.py`, `train_phase2.py`):

- `death_ramp_steps` (default 250, one growth interval; CLI
  `--death_ramp_steps`). 0 restores the one-step cut.
- The unit becomes DEAD at the decision. So it leaves the medians, birth
  and death at once, and a dying layer's heads and FFN groups count as in
  a dead layer at once.
- Over the next `death_ramp_steps` calls to `record_step`, the gate falls
  linearly from its value at the decision to sigmoid(`dead_logit`). The
  last step sets the logit to exactly `dead_logit`.
- The ramp is linear in gate, not in logit. The unit's output scales with
  its gate. A logit-linear ramp from -1 makes about 85% of the gate drop
  in the first quarter of the ramp, which is close to a cut.
- The controller sets the logit exactly at each step. The optimizer can
  not hold up a unit that is chosen to die. This differs from the birth
  ramp, which adds to the optimizer's update.
- `update()` reports `dying`. `ramping` now counts births only.

**Tests**: 30/30 pass. Tests 1-25 pin `death_ramp_steps=0`. New tests
26-30 cover: the decision is immediate and the logit does not move; the
gate falls linearly and ends at exactly `dead_logit`; an optimizer push
against the ramp is overridden; a dying layer's units are excluded at
once; a negative ramp length is rejected. Tests 26, 27, 28 and 30 fail on
the old controller. Test 29 passes there too: it guards the dead-layer
rule during the ramp.

**Smoke run** (local, tiny model, 400 steps, ramp 20): 8 deaths. Each
dying gate fell evenly, for example 0.256 → 0.202 → 0.135 → 0.068 at
ramp steps 0, 5, 10, 15. Each logit ended at exactly -8.000. Layer 3
died at step 150, and its units then counted as `in_dead_layers` (18).

**Open — final-check interaction**: the last growth check fires on the
final step (attempt 5: deaths at step 10000 = `max_steps`). A birth or a
death that starts at that check never ramps. The final `architecture.json`
reads gates, so it disagrees with `state_counts`, which reads states. The
birth ramp has the same problem. A possible guard: skip `update()` when
fewer than the ramp length of steps remain.

**Next**: re-run attempt 5 with all three fixes.

## 2026-10-14 (continued) — No growth check whose ramps cannot finish

Fixes the final-check interaction from the entry above.

**Change**: `StructuralController.ramps_fit(steps_remaining)` is true
when `steps_remaining >= max(birth_ramp_steps, death_ramp_steps)`. A ramp
starts at the next `record_step`, so a check at step s finishes its ramps
at step s + N. `train_phase2.py` skips `update()` when `ramps_fit` is
false. With ramps of 0, a check on the final step is still allowed (the
old behavior).

For the attempt-5 config (10000 steps, interval 250, ramps 250), the last
check is now step 9750, and its ramps end on step 10000.

**Tests**: 31/31 pass. Test 31 covers the boundary, the longer of the two
ramps, and ramps of 0.

**Smoke run** (100 steps, interval 25, ramps 30): checks ran at steps 25
and 50 only. In the final `architecture.json`, all 4 dead units have gate
0.0003 and all 4 born units have gate about 0.27, so gates agree with
states.

**Next**: re-run attempt 5 with all four fixes.

## 2026-10-14 (continued) — Phase 2 attempt 6: all four fixes at full scale

Attempt-5 config plus all four controller fixes: birth ramp, dead-layer
rule, gate-linear death ramp, and no check whose ramps cannot finish
(ramps 250, so the last check is step 9750). Secure Cloud A100 SXM 80GB,
Trackio on. Run time about 2.2 h at 1.29 steps/s; Trackio did not slow
the loop. Pod cost about $4.40. Model:
`sbartlett97/sot-phase2-wikitext103-heuristic-seed42-tc768d013-v6-allfixes`.
Logs: `results/phase2_v6_allfixes/`.

| | attempt 4 | attempt 5 | attempt 6 |
|---|---:|---:|---:|
| births / deaths | 98 / 0 | 98 / 76 | 97 / 28 |
| deaths by kind | — | 46 FFN, 25 heads, 5 layers | 2 FFN, 23 heads, 3 layers |
| dead layers | none | 4, 7, 9, 10, 11 | 9, 10, 11 |
| deaths inside dead layers | — | 58 | 0 |
| mature units at end | 641 | 565 | 612 |
| cost proxy at end | 0.0777 | 0.0523 | 0.0536 |
| val ppl, step 6000 | 28.6 | 143.1 (spike) | 29.66 |
| val ppl, step 9000 | — | — | **27.29** |
| val ppl, step 9500 | 28.6 | 29.95 | 27.90 |
| val ppl, step 10000 | **28.08** | 35.71 | 29.28 |

**What the fixes did:**

- **No spikes.** Val ppl falls every eval from step 500 to step 9000.
  Attempt 5 had spikes at steps 4500, 6000, 8500 and 10000. The layer
  deaths (10 at step 1250, 9 and 11 at step 1750) left no mark.
- **No wasted deaths.** 0 deaths and 0 births inside dead layers
  (attempt 5: 58 deaths). With the orphans out of the medians, the death
  rule fired less: 28 deaths, against 76.
- **Best quality per cost so far.** At step 9000, val ppl 27.29 at cost
  proxy 0.053. Attempt 4 never went below 28.08, at cost 0.078.
- Capacity moved into layer 7: 57 of 66 FFN births went there. Layer 7
  died at step 4500 in attempt 5; here it became the widest layer.

**New fault — deaths in the LR tail.** Val ppl rose from 27.29 (step
9000) to 27.90 (9500) and 29.28 (10000). Train loss rose too, from 3.20
to 3.29 in the last 250 steps. Five deaths came at steps 9250-9750:
heads 8.10, 0.6 and 0.7, and FFN groups 7.71 and 7.122. Heads 0.6 and
0.7 are in layer 0, the most important layer. At these steps the cosine
LR was 4.6e-6 to 5e-7 (peak 3e-4), so the rest of the network could not
adapt. The death ramp spreads the cut but cannot replace the learning
the network needs to recover. Head 0.7 was born at step 6250 and killed
at step 9750: it never got enough training at a useful LR.

**Next**: stop deaths (and possibly births) once LR falls below a set
fraction of its peak, or in the last N% of steps. In this run, a stop
after step 9000 would have given about 27.3. Then re-run.

## 2026-10-14 (continued) — No death in the LR tail

Fixes the attempt-6 fault: 5 deaths at LR 0.2-1.5% of peak cost about 2
ppl at the end of training.

**Change** (`plasticity.py`, `train_phase2.py`):

- `death_freeze_lr_fraction` (default 0.05; CLI
  `--death_freeze_lr_fraction`, 0 = never freeze).
- `update(step, mode, lr_fraction)`: the training loop passes the
  current LR / peak LR. Below the threshold, the death pass kills
  nothing. It still updates streaks and diagnostics, so `growth_log.json`
  shows what would have died. `update()` reports `deaths_frozen`.
- `lr_fraction=None` (the default) never freezes. Births are not frozen.

**Why 5%**: attempt 6's LR fraction at each death check:

| step | 7000 | 7500 | 8000 | 8500 | 8750 | 9000 | 9250 | 9500 | 9750 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LR / peak | 0.227 | 0.161 | 0.105 | 0.060 | 0.042 | 0.027 | 0.015 | 0.007 | 0.002 |

Deaths at steps 7000-7500 did no harm: val ppl kept falling, 28.30 to
27.51. The harmful deaths came at 0.015 or lower. With 0.05, the last
death check is step 8500. That unit's ramp ends at step 8750, with about
1250 steps of training left.

**Tests**: 35/35 pass. New tests 32-35 cover: no death below the
threshold, with streaks and diagnostics still updated; deaths resume at
the threshold; no freeze for `lr_fraction=None` or a threshold of 0;
values outside [0, 1) are rejected. Tests 32, 33 and 35 fail on the old
controller. Test 34 passes there too: it guards the old behavior.

**Smoke run** (800 steps, warmup 500): checks up to step 750 (LR 6.7%
of peak) ran normally. The step-775 check (1.7%) reported
`deaths_frozen`. That check had no eligible unit, so the unit tests, not
the smoke run, show the blocking.

**Open**: births still run in the LR tail. Attempt 6 had 1 head birth
at each of steps 9250, 9500 and 9750. A unit born at near-zero LR keeps
weights that never trained at a useful scale. The data cannot yet
separate their effect from the deaths.

**Next**: re-run attempt 6 with the freeze (attempt 7).

## 2026-10-14 (continued) — Attempt 7: the LR-tail fault is births, not deaths

Attempt 6 plus `--death_freeze_lr_fraction 0.05`. Same Secure Cloud A100,
about 2.3 h, about $4.40. Model:
`sbartlett97/sot-phase2-wikitext103-heuristic-seed42-tc768d013-v7-lrfreeze`.
Logs: `results/phase2_v7_lrfreeze/`.

The run matches attempt 6 exactly up to step 9000 (same seed, and the
freeze changes nothing before step 8750). The freeze worked: no death
after step 7500, and checks 8750-9750 reported `deaths_frozen`. The
would-be death streaks reached 4 by step 9750.

| | attempt 6 | attempt 7 |
|---|---:|---:|
| births / deaths | 97 / 28 | 97 / 23 |
| deaths after step 9000 | 5 | 0 |
| dead layers | 9, 10, 11 | 9, 10, 11 |
| cost proxy at end | 0.0536 | 0.0545 |
| val ppl, step 9000 | 27.29 | 27.29 |
| val ppl, step 9500 | 27.90 | 27.77 |
| val ppl, step 10000 | 29.28 | **30.66** |

So the deaths were not the main cause. Both runs lose quality in the
same window, and both have one head birth at each of steps 9000, 9250,
9500 and 9750.

**Ablation on the attempt-7 final model** (48 val batches of 8 x 256;
`results/phase2_v7_lrfreeze/tail_birth_ablation.py`). Each unit's logit
is set to -4, its dormant value, and the val loss is measured:

| unit | born | dloss | ppl |
|---|---:|---:|---:|
| none (baseline) | — | — | 30.25 |
| head 5.5 | 9000 | -0.004 | 30.14 |
| head 6.10 | 9250 | -0.009 | 29.97 |
| head 6.6 | 9500 | -0.013 | 29.85 |
| head 3.9 | 9750 | -0.091 | 27.63 |
| all 4 tail births | | **-0.108** | **27.16** |
| control: head 6.7, born early | | +0.006 | 30.44 |
| control: head 3.0, born early | | +0.010 | 30.56 |
| control: head 0.7, born early | | +0.004 | 30.35 |

(A fourth control, head 3.4, is not valid: it died at step 5000, so the
test brought it back to life. Its +0.126 is the cost of reviving a dead
head, not of removing a live one.)

- Removing the 4 tail-born heads gives 27.16 ppl, better than step 9000's
  27.29. Removing heads born early makes the model worse. So the tail
  births are harmful, and the effect is specific to them.
- All 4 tail-born heads sit at gate 0.269 = sigmoid(-1): the ramp moved
  them to `birth_init_logit`, and the optimizer, at LR 2.7% of peak or
  lower, did not move them after that. Their weights were trained only
  at the dormant gate scale (about 0.02).
- The last birth does the most harm: head 3.9 ramps in over steps
  9751-10000 and reaches full gate on the final step. No step at any LR
  is left to adapt to it.
- In attempt 6, the same births (with 8.11 in place of 3.9) and the 5
  late deaths together gave 29.28.

**Next**: freeze births in the LR tail with the same rule, so no
structure change happens below 5% of peak LR. Then re-run (attempt 8).
The ablation predicts a final val ppl close to 27.2.

## 2026-10-14 (continued) — No birth in the LR tail

Fixes the attempt-7 fault: 4 heads born at LR 2.7% of peak or lower cost
about 3.1 ppl at the end of training.

**Change** (`plasticity.py`, `train_phase2.py`):

- `birth_freeze_lr_fraction` (default 0.05; CLI
  `--birth_freeze_lr_fraction`, 0 = never freeze). It is separate from
  `death_freeze_lr_fraction`, so each freeze can be set or disabled alone.
- Below the threshold, the birth pass still scores every dormant
  candidate. It then gives no birth, so `growth_log.json` still shows
  `max_dormant_birth_score`. `update()` reports `births_frozen`.
- The freeze applies in heuristic and random mode, so the random-growth
  control keeps the same timing. `lr_fraction=None` never freezes.

**Effect on the attempt-7 schedule**: the last birth check is step 8500
(LR 0.060). The freeze blocks the births at steps 8750-9750: the 4
ablated heads (5.5, 6.10, 6.6, 3.9) and head 0.5 (step 8750, LR 0.042).
Head 0.5 was not in the ablation, so its effect is not known.

**Tests**: 39/39 pass. New tests 36-39 cover: no birth below the
threshold, with the logit unchanged and the birth score still reported;
births resume at the threshold; random mode is frozen too; None or a
threshold of 0 never freezes; the two freezes are independent; values
outside [0, 1) are rejected. Test 36 fails on the old controller (the
strong candidate is born).

## 2026-10-14 (continued) — Attempt 8: no structure change in the LR tail

Attempt 7 plus `--birth_freeze_lr_fraction 0.05`. Same Secure Cloud A100,
seed and config. Commit 281b962. Model:
`sbartlett97/sot-phase2-wikitext103-heuristic-seed42-tc768d013-v8-tailfreeze`.
Logs: `results/phase2_v8_tailfreeze/`.

- Val ppl matched attempt 7 exactly at every eval to step 8500, as
  expected: the freeze cannot act before step 8750.
- Last birth step 8500 (head 2.3), last death step 7500. Checks
  8750-9750 report `births_frozen` and `deaths_frozen`. 92 births, 23
  deaths. Dead layers 9, 10, 11.
- The ablation predicted about 27.2. The result is better: val ppl falls
  at every eval to the end.

| val ppl | 8500 | 9000 | 9500 | 10000 |
|---|---:|---:|---:|---:|
| attempt 6 (no freeze) | 27.37 | 27.29 | | 29.28 |
| attempt 7 (deaths frozen) | 27.37 | 27.29 | 27.77 | 30.66 |
| **attempt 8 (both frozen)** | 27.37 | 27.14 | 27.00 | **26.98** |

Attempt 4 ended at 28.08. The Phase 1 dense 768 baseline is 21.98.

## 2026-10-14 (continued) — Compute accounting: the cost proxy is wrong, and dormant units are not off

`architecture_cost()` is a gate-weighted proxy (see the 2026-10-09 ablation entry). New code counts
real compute:

- `compute.py`: matmul FLOPs/token and parameters of a structure (one
  entry per present layer: heads, FFN groups). `Replay` rebuilds a Phase 2
  run's structure at any step from its seed split (`Random(seed)` in
  `_register_units` order) and `events.json`. A unit is present while
  PROTECTED/MATURE, or DEAD with its death ramp still running.
- `tests/compute_test.py` (5/5): the formulas equal torch
  `FlopCounterMode` and `numel()` exactly, on an irregular
  `CompactTransformer` and on the gated supernet. The supernet runs its
  full dense FLOPs whatever its gates are.
- Replay matches `state_counts` at all 157 growth checks of attempts 5-8,
  and each final gate band (alive >= 0.237, dormant 0.014-0.052, dead
  <= 0.0011).
- `structure_eval.py`: val ppl of the supernet as trained, and with every
  non-present unit forced off (logit -30). 48 val batches of 8 x 256.

Dense 768 x 12: 313.1 MFLOP/token forward, 152.1M params. lm_head is
77.2 MFLOP of that (25%), a floor no structure change removes.

| final structure | layers | FLOPs/token vs dense | params vs dense | training FLOPs (lower bound) | cost proxy |
|---|---:|---:|---:|---:|---:|
| attempt 5 | 7 | 0.428 | 0.432 | 0.444 | |
| attempt 6 | 9 | 0.468 | 0.474 | 0.454 | 0.054 |
| attempt 7 | 9 | 0.473 | 0.478 | 0.454 | 0.055 |
| attempt 8 | 9 | 0.466 | 0.472 | 0.454 | |

- The cost proxy is about 9x too low: it multiplies cost by gates of
  0.25-0.48.
- The training-FLOPs figure is a lower bound only. Birth scores read the
  gradient on dormant gates, so dormant units must run. The gated
  supernet really runs 1.0 of dense.

**Quality of the counted structure** (`structure_eval.py`):

| | as trained | present structure only |
|---|---:|---:|
| attempt 7 | 30.25 | 244.29 |
| attempt 8 | 26.67 | 132.42 |

Split for attempt 7 (`results/compute/dormant_split.py`, `dead_split.py`):

| units forced off | n | mean gate | val ppl |
|---|---:|---:|---:|
| none | | | 30.25 |
| dead units in live layers | 14 | 0.0007 | 30.42 |
| dead layers 9, 10, 11 | 3 | 0.0009 | 33.21 |
| dormant heads in live layers | 47 | 0.035 | 33.53 |
| dormant FFN groups in live layers | 706 | 0.043 | 211.70 |

- **Dormant FFN groups do most of the work behind the gap.** They have
  small gates, but there are 706 of them, and their weights train for the
  whole run. "Dormant" is a low-gate live state, not an off state.
- **Dead units drift back up.** Their gates sit at 0.0007-0.0009, above
  sigmoid(-8) = 0.0003. Nothing holds a DEAD logit after the death ramp,
  so the optimizer moves it. The dead layers still carry about 3 ppl.
- So no Phase 2 result yet shows quality at lower compute. Attempt 8's
  26.98 comes from the full supernet.

**Next** (needs a decision):
1. Make DEAD a hard zero: hold the logit (or zero its gradient) at every
   step after the ramp. Needed before dead units can leave the tensors.
2. Make DORMANT truly off: gate exactly 0 in the forward pass. The birth
   score can read dL/dg directly, which is not zero at g = 0.
3. Re-run, and judge on `structure_eval.py`, not supernet val ppl.

## 2026-10-15 — Dormant and dead units: exact zero effect, zero gradient

Requirement of the premise, missed in attempts 1-8: only the active
structure may affect the loss. The 2026-10-14 compute entry showed the
opposite (attempt 8: 26.67 val ppl as trained, 132 with inactive units
forced off).

**Change**:

- `PlasticTransformer` has a scale per unit (`layer_scale`, `head_scale`,
  `group_scale`, saved with the checkpoint). Gate = scale x
  sigmoid(logit). DORMANT and DEAD units have scale 0: an exact zero
  output, and exactly zero gradient on their weights and logit.
- Skip edge i <- i-s is scaled by both layers' scales, so a skip to or
  from an absent layer is off (as in `CompactTransformer`).
- Birth signal: each dormant unit's gate has a zero-valued probe. It adds
  nothing to the output; its gradient is dL/dg. Probes are in an optimizer
  group with lr 0, out of gradient clipping. The cost terms use gates
  without probes, so the signal is the task loss only. This is a read-only
  gradient on dormant units. Dormant weights stay at their random init, so
  births rank random-init units (GradMax style).
- `snapshot_inactive()` / `restore_inactive()` around `optimizer.step()`:
  weight decay and Adam momentum no longer move inactive weights.
- Ramps act on the scale: birth 0 -> 1 (logit set to `birth_init_logit`,
  trains freely); death 1 -> 0, ending at exactly 0. `dead_logit` is gone.
- Units inside a DORMANT layer are now excluded like those in a DEAD layer
  (`_in_inactive_layer`): their importance is exactly 0. When a layer is
  born, its live units restart as PROTECTED (age 0).
- `compact_model.build_from_plastic()`: physical model of the present
  structure (scale > 0), gates folded into the output weights, no
  threshold, no head-count snapping.
- Birth score is the raw |dL/dg| EMA: about 1e-6 in the smoke run, against
  0.006-0.01 before. `--birth_score_floor 0.005` would block every birth,
  so runs from now on use 0. Within a check every dormant score is G x a
  constant (the contribution term is 0), so the ranking is by G alone.

**Tests**: `tests/zero_effect_test.py` 9/9, the controller suite 42/42
(rewritten for scale ramps; new tests 40-42 cover the dormant-layer rule
and fail with the old dead-layer-only rule), and `tests/compute_test.py` 5/5:

- randomising every inactive weight (x100) leaves the output bit-identical;
- every inactive weight, logit and skip edge gets exactly zero gradient;
- the probe gradient equals a float64 finite difference (to <= 0.13%);
- 3 AdamW steps with weight decay leave inactive weights and probes
  bit-identical;
- `build_from_plastic` matches the supernet's logits to 2.4e-7, with a
  dormant layer, a dead layer, mid-ramp units and skips on;
- in a 60-step controller run, scale and probe agree with the state at
  every step (17 births, 12 deaths);
- old checkpoints load with scale 1 (unchanged behaviour); new ones keep
  their scales.

**Smoke run** (4 layers, d 128, 800 steps): 49 births, 34 deaths (30 FFN
groups, 3 heads, 1 layer). Val loss on 20 batches: supernet 4.592691,
physical model 4.592691 (largest batch difference 4.8e-7), at 0.918 of the
supernet's parameters.

Executed compute is still 1.0 of dense: the tensors are full size, and
the probes need dormant units' outputs. DEAD units need neither, so
removing them physically during training is no longer blocked.

`ablation_importance.set_logit` (and the tail-birth ablation that used it)
turns a unit "off" through its logit. On new checkpoints a logit of -4
leaves gate = scale x 0.018. Ablations of new checkpoints must set the
scale to 0.

From now on, val ppl during training is the val ppl of the present
structure. Results are not comparable with attempts 1-8, which measured
the supernet. Expect worse val ppl than attempt 8's 26.98, at a true cost
of about 0.47 dense or lower.

## 2026-10-16 — Attempt 9 launched; probes on old checkpoints

Attempt 9: attempt-8 settings on the zero-effect code, with
`--birth_score_floor 0`. Secure Cloud A100 SXM 80GB ($1.79/hr). Hub tag
`768d013-v9-zeroeffect`. The first checkpoint (step 1000) gets a check:
supernet loss must equal the loss of `build_from_plastic`, and every
probe must be exactly 0.

Bug found on the pod: `tests/zero_effect_test.py` test 8 failed there
and passed locally. A checkpoint saved before the probes existed loads
them with `torch.empty`. `PlasticTransformer._init_weights` skipped the
top-level module, so the probes kept that memory (zeros on the Mac, not
on the pod). It now sets the probes to 0. All 9 tests pass on the pod
with the fix. Attempt 9 is not affected: it builds the model with the
constructor, which uses `torch.zeros`. The fault affected only old
Phase 2 checkpoints loaded with this code, where non-zero probes times a
zero probe mask added nothing unless the memory held inf or NaN.

## 2026-10-16 — Attempt 9 result: the first valid present-structure number

Attempt 9 finished (exit code 0) and pushed to the Hub. Logs are in
`results/phase2_v9_zeroeffect/`. The pod is deleted.

Check at step 1000: supernet loss = physical loss = 4.714066 (largest
batch difference 4.8e-7). Every probe was exactly 0.

Training-time val ppl. For attempt 9 this is the present structure, because
inactive units have exactly zero effect. For attempt 8 it is the supernet.

| step | 1000 | 2000 | 4000 | 6000 | 8000 | 10000 |
|---|---:|---:|---:|---:|---:|---:|
| attempt 9 (present structure) | 106.72 | 60.54 | 39.63 | 33.63 | 31.15 | **30.61** |
| attempt 8 (supernet) | 107.00 | 56.17 | 35.53 | 29.66 | 27.51 | 26.98 |

Val ppl falls at every eval. The last structure event is at step 8500.

Events: 92 births (52 heads, 31 FFN groups, 9 layers) and 60 deaths (52
heads, 4 FFN groups, 4 layers).

Final present structure: 8 layers (0-5, 8, 10), 36 of 144 heads, 357 of
1536 FFN groups. Per-layer heads: 8, 4, 5, 5, 2, 4, 6, 2. `compute.py`
(seq 256, the settings that give 0.4656 for attempt 8): **0.425 of dense
FLOPs/token**, 0.432 of params (0.237 of non-embedding params).

Findings:

- **The present structure of attempt 9 is far better than that of
  attempt 8.** At 0.425 of dense FLOPs it gets 30.61 val ppl. Attempt 8's
  present structure (0.466 FLOPs) got 132.42 with its dormant units forced
  off. The zero-effect fix works: the model no longer depends on units it
  does not have.
- **It is 26% behind dense 768 (24.30) at 0.425 of the FLOPs.** This does
  not show that learned structure helps. The dense control at matched
  compute ("Where next" item 2) decides that. (Result: the dense controls
  win. See the last 2026-10-16 entry.)
- **Gates of present units are low (0.2-0.45).** No unit passes the 0.5
  threshold of `export_architecture`, so `architecture.json` lists 0
  active layers. The present structure is the set of units with a
  non-zero gate (scale 1). Do not use the 0.5 threshold on Phase 2
  checkpoints.
- **Few FFN births** (31, against 52 head births). This can be a scale
  effect: a head's probe gradient sums over 64 dimensions, an FFN group's
  over 24 channels. Not checked yet.

## 2026-10-16 — Why attempt 9 grew few FFN groups; cost-normalised birth score

Attempt 9 had 52 head births and only 31 FFN-group births, though dormant
FFN groups far outnumber dormant heads. Cause: unit size, not the probe
signal.

- The controller ranks dormant layers, heads and FFN groups in one list,
  by raw score (`|dL/dg|` EMA). There is no adjustment by kind
  (`plasticity.py`, `_birth_pass`).
- A head is 3.6x the params and 4.1x the FLOPs of an FFN group. On the
  final checkpoint (8 val batches), the median raw `|dL/dg|` of present
  heads is 3.4x that of present FFN groups (0.0095 against 0.0028). Per
  parameter the two kinds are equal (1.4e-8 and 1.6e-8).
- So heads won every check while dormant heads remained. In steps
  1000-4000: 36 head births and 1 FFN birth, as dormant heads in live
  layers fell from 72 to 23. In steps 4000-10000, FFN groups took 30 of
  the 37 head/FFN births.
- The ranking bought value per unit, not value per FLOP. At step 10000,
  the top 100 dormant candidates per FLOP are all FFN groups.

Fix: `--birth_score_norm flops` divides each head and FFN-group birth score
by the unit's FLOPs relative to one FFN group (`compute.unit_flops`; a head
is 4.15 at d 768). Layers keep divisor 1, so the layer-against-FFN ranking
does not change. The default `none` keeps attempts 1-9 reproducible. The
growth log now records the best birth score of each kind. Tests:
`phase2_controller_test.py` test 43, `compute_test.py` test 6, and a
120-step smoke run. Attempt 10 uses the attempt-9 settings plus this flag.

## 2026-10-16 — Dense controls at matched compute: both beat attempt 9

Two dense models at the FLOPs of attempt 9 (0.425 of dense 768), trained with
`train.py --model_type dense` on the attempt-9 settings: WikiText-103, seed
42, 10,000 steps, batch 16 x 8 accumulation, seq 256, LR 3e-4. Val ppl uses
the same fixed val batches as attempt 9 (`train.evaluate_loss`). Both ran on
one pod at the same time (about 94% of the solo throughput). Logs are in
`results/dense_c425_d768l4/` and `results/dense_c425_d512l9/`. The checkpoints
are in private Hub repos `sot-phase1-wikitext103-dense_c425_{d768l4,d512l9}-seed42-tc768d013`.
The pod is deleted.

| model | shape | FLOPs/token (of dense 768) | params (of dense 768) |
|---|---|---:|---:|
| attempt 9, present structure | d 768, 8 layers, 36 heads, 357 FFN groups | 0.425 | 0.432 |
| dense A | d 768, 4 layers, 12 heads, FFN 1824 | 0.424 | 0.428 |
| dense B | d 512, 9 layers, 8 heads, FFN 2048 | 0.421 | 0.418 |

Training-time val ppl:

| step | 1000 | 2000 | 4000 | 6000 | 8000 | 10000 |
|---|---:|---:|---:|---:|---:|---:|
| attempt 9 (present structure) | 106.72 | 60.54 | 39.63 | 33.63 | 31.15 | 30.61 |
| dense A (d 768, 4 layers) | 102.24 | 57.60 | 39.33 | 33.43 | 30.77 | 30.18 |
| dense B (d 512, 9 layers) | 106.79 | 57.63 | 38.06 | 32.08 | 29.56 | **28.98** |

Findings:

- **Attempt 9 does not beat a dense model at matched compute.** Dense A is
  1.4% better (30.18 against 30.61). Dense B is 5.3% better (28.98).
  Phase 2 has not yet shown that learned structure helps.
- **Dense A is ahead at every eval.** The gap stays small (0.2-0.4 ppl from
  step 4000). Dense B is ahead from step 2500, and the gap grows to about
  1.6 ppl.
- **Depth helps at this budget.** At the same FLOPs, 9 narrow layers beat 4
  wide layers by 1.2 ppl. Attempt 9 kept 8 layers but at d 768, so the
  lm_head and the embedding take a larger part of its budget than in dense B.
- **Possible causes of the gap, not tested:** the structure changed until
  step 8500, so late units had little time to train; and births favoured
  heads over FFN groups (previous entry). Attempt 10 (`--birth_score_norm
  flops`) tests the second cause.

## 2026-10-19 — Attempts 10 and 11a: cost-normalised births, WSD schedule, signed birth telemetry

An external review of the attempt-9 result made three points. We act on two now.

1. **Late births get almost no LR.** Under cosine, LR / peak at step 8500 is
   0.061. The LR left from step 8500 to 10000 is about 30 peak-LR-equivalent
   steps, of about 5000 in the whole run. Attempt 9 had births until step 8500.
2. **The birth score ignores the sign of dL/dg.** `record_step` keeps an EMA
   of |dL/dg|. A unit whose gradient says "opening me raises the loss" can
   still rank high. This selects for sensitivity, not for expected benefit.
3. **AdamW state of a newborn slice is cold** (zero moments, but the shared
   tensor's step count is high), and a gradient multiplier does not scale an
   Adam update. This matters for a per-unit LR boost (attempt 11b). Not done
   yet.

Code (defaults keep attempts 1-9 and the dense controls reproducible):

- `train.make_scheduler`, `--lr_schedule {cosine,wsd}`, `--lr_decay_start`,
  `--warmup_steps`. 'wsd' = warmup, peak LR until the decay start, then
  cosine to 0 at max_steps. `train.py` and `train_phase2.py` share it, so a
  dense control can use the same schedule. `tests/schedule_test.py`: cosine
  is identical to `get_cosine_schedule_with_warmup` at all 10000 steps. The
  LR budget is 5000 peak-LR-equivalent steps for cosine and 8250 for WSD with
  decay from step 7000.
- `--structure_freeze_step`: no birth or death whose ramp would end after
  this step. The 5% LR freezes do not fire under WSD before the decay.
- Telemetry only, no change to the ranking: `signed_gradient_ema` per dormant
  unit; each birth event logs `abs_gradient_ema`, `signed_gradient_ema`,
  `sign_consistency` (signed / abs, -1 = always "open me"), `lr_fraction` and
  `lr_budget_remaining` (peak-LR-equivalent steps left). Each growth-log
  record has `birth_sign_summary`: per kind and for the top 20 candidates,
  the share with a negative sign and the median consistency. Tests 44 and 45
  in `phase2_controller_test.py`.

Runs (all three on one RTX PRO 6000; A100s were out of stock):

| run | change | compare with |
|---|---|---|
| attempt 10 | attempt-9 settings + `--birth_score_norm flops` | attempt 9 (30.61) |
| attempt 11a | attempt 10 + WSD, decay from 7000 + structure freeze at 7000 | attempt 10, and the WSD dense control |
| dense d512 / 9 layers, WSD | the best dense control, same WSD schedule | attempt 11a |

Decay start 7000: 30% of the run decays. A unit born at the last allowed
check (6750) gets 250 steps of ramp at peak LR, then the full decay
(about 1500 peak-LR-equivalent steps). The search phase is 1500 steps
shorter than attempt 9's (births until 8500). One value only, not tuned.

Check before a result is claimed: cost-normalised births change the mix of
heads and FFN groups, so the final FLOPs of attempts 10 and 11a can move away
from 0.425. Run `structure_eval.py` on both. If either ends far from 0.42 of
dense FLOPs, the dense controls no longer match it.

Pod restart and resume. At about step 2300 (21:50 BST) the host restarted
the pod's container. All three runs stopped with no traceback. The disk was
intact.

- The dense run resumed from its step-9000 checkpoint (`train.py
  --resume_from`, with the optimizer state). After the resume, the batch
  order differs from an unbroken run, so its last 1000 steps are not an
  exact copy.
- `train_phase2.py` had no resume, so attempts 10 and 11a started again from
  step 0. At step 1500 the new runs have the same val loss and the same
  events as the runs before the restart, so the runs are deterministic.
- An error in my relaunch script truncated the stdout log of attempt 10
  near step 1600. The checkpoints keep all its records (metrics, events,
  growth log), so no result is lost.
- New: `train_phase2.py --resume_from <output_dir>`. Each checkpoint now
  writes a rolling `training_state.pt` with an atomic replace. It holds the
  model, optimizer, scheduler, grad scaler, Lagrange multiplier, controller
  state (`StructuralController.state_dict`: unit telemetry, events, death
  streaks, RNG), data position (`BatchIterator(start=...)`), RNG states and
  logs. A resume with a changed config stops. `tests/resume_test.py`: a run
  that stops at step 27 and resumes from step 20 ends bit-identical to an
  unbroken run (weights, buffers, events, growth log, metrics). The test
  fails if the controller state or the data position is not restored.
- The relaunched attempts 10 and 11a run the old code, so they cannot
  resume. A restart to get the new code would cost 26 minutes of work, so
  they continue as they are.
