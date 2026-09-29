# Research log

Running record of what's been tried, found, and fixed on the Self-Organising
Transformer experiment. Append new dated entries at the bottom; keep "Current
status" at the top up to date as the single-glance summary. See `README.md`
for the experiment design and `self_organizing_transformer_research_proposal.md`
for the full phased research programme.

## Current status

- **Phase**: Phase 1 (differentiable structural selection) — full-scale matrix
  in progress. All 4 primary conditions (dense, dense_8l, dense_half_ffn, sot)
  finished a full 10,000-step / full-corpus run, seed 42, on RunPod (A100,
  CA-MTL-3). Sequential phase (materialize → retrain_compact → evaluate) now
  running on the same pod.
- **Codebase**: pipeline bugs from the initial commit are fixed (see 2026-09-28
  entries). `evaluate.py`, `compact_model.py`, `model.py`, `train.py` all
  patched and verified against real runs, including a tied-weights-on-load
  bug caught mid-run on real infrastructure (see below) — fix committed and
  redeployed before any evaluation was run against it.
- **Key hyperparameter finding**: `target_cost=0.45` (the shipped default) is
  unreachable — the natural unconstrained equilibrium cost for the default
  config on TinyStories is **~0.186**, so the Lagrangian budget term never
  activates at `0.45`. `target_cost=0.14` (below that equilibrium) is
  validated to work — confirmed by a full 2000-step run where `lagrange` rose
  to 0.50 and `cost` was pulled down from a peak of 0.168 to 0.164. This is
  now the default in `train.py` and `run_experiment.sh`.
- **Structural finding**: under `target_cost=0.14`, capacity is cut
  disproportionately from later layers (roughly layers 6–11) across the
  `layer`/`head`/`ffn` gate families, while earlier/middle layers keep close
  to their unconstrained shape. Task quality (`val_ppl`) barely changed
  (8.166 vs. 8.125 unconstrained) despite ~12% lower architectural cost.
  Not yet confirmed across seeds or at full corpus scale.
- **Next step**: full-scale `run_experiment.sh` run (all 6 conditions, full
  corpus, `max_steps=10000`), 1 seed first to validate the pipeline end to
  end, then ≥3 seeds. See "Next steps" section below for hardware/parameter
  guidance already agreed.

## Next steps (agreed, not yet executed)

1. Run `./run_experiment.sh` at full scale (no `--max_train_docs` cap) —
   parallelize the 3 dense baselines + SOT across separate instances since
   they're independent; `materialize.py` → `retrain_compact.py` →
   `evaluate.py` is a sequential chain off the SOT checkpoint and runs after.
   No checkpoint-resume exists in `train.py`, so prefer on-demand (not spot)
   instances for anything longer than a session's safe uptime.
2. Repeat for ≥3 seeds. Specifically check whether the "back layers cut
   preferentially" pattern found at seed 42 reproduces, or was seed-specific.
3. Run the full `evaluate.py` benchmark suite (`hellaswag`, `arc_easy`,
   `piqa`, `winogrande`) on every condition — only perplexity has been
   exercised so far.
4. Score the real results against README's six interpretation criteria.
5. Only after that: decide on Phase 2 (growth/mutation) per the research
   proposal's Gate 1→2 (§27) — do not skip ahead of a validated Phase 1
   result.
6. Separately (lower priority, orthogonal): if/when moving past TinyStories
   to a harder/more diverse dataset (README's own staged plan — WikiText-2/103
   or an OpenWebText subset), re-run the same unconstrained-equilibrium
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
