# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository tests

This is a research experiment, not a library. It tests one hypothesis: an
over-complete Transformer can jointly learn its weights and its own effective
topology (which layers/heads/FFN-channel-groups/cross-layer skips matter),
using differentiable gates, without a separate NAS search stage.

Read `README.md` first. It defines the full experiment design: the primary
hypothesis, the required controls (dense baseline, SOT, retrained-discovered
architecture, seed replication), and how to interpret a result as
positive/negative. Do not change the experiment matrix or control structure
without re-reading that design.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python >=3.12. No lockfile; `requirements.txt` is the source of
truth for dependencies (`pyproject.toml` currently declares none).

There are no tests, no linter config, and no CI in this repository. Verify
changes by running a short training loop (see below) and inspecting the
printed loss/cost records and the exported architecture JSON.

## Common commands

Run the full experiment (dense baselines + SOT + materialize + retrain +
evaluate, in one script):

```bash
./run_experiment.sh
```

Environment variables override its defaults: `STEPS`, `SEQ`, `BATCH`,
`ACCUM`, `SEED`, `RUN_BASELINES=0` (skip the three dense baselines).

Individual pipeline stages, in dependency order:

```bash
# 1. Train (model_type: sot | dense)
python train.py --model_type sot --output_dir outputs/sot --max_steps 10000

# 2. Physically compact the discovered architecture (SOT checkpoint -> smaller dense tensors)
python materialize.py outputs/sot/final --threshold 0.5 --output_dir outputs/sot_compact

# 3. Retrain the discovered architecture from scratch (fresh weights, same shape)
python retrain_compact.py outputs/sot_compact --output_dir outputs/discovered_retrained --max_steps 10000

# 4. Evaluate (perplexity + lm-evaluation-harness tasks: hellaswag, arc_easy, piqa, winogrande)
python evaluate.py outputs/sot/final --results outputs/sot/evaluation.json
```

For a quick smoke test of any stage, pass a small `--max_steps` (e.g. 50) and
`--eval_every`/`--save_every` small enough to trigger at least one
checkpoint before that step count is reached.

## Architecture

Three distinct model classes exist, one per pipeline stage. Do not conflate
them:

- **`model.py` — `SelfOrganizingTransformer` / `SOTConfig`**: the
  over-complete supernet. Fixed tensor shapes (`max_layers`, `max_heads`,
  `ffn_groups` all present at all times). Structure is controlled entirely
  by sigmoid gates over learnable logits, applied multiplicatively inside
  `forward`:
  - `layer_logit` per block — whether the block is active.
  - `head_logits` per block (`CausalSelfAttention`) — per-head gate, applied
    after the attention output, before the output projection.
  - `group_logits` per block (`GroupedFFN`) — gates groups of FFN channels
    (`ffn_groups` groups of `max_ffn_dim / ffn_groups` channels each), not
    individual channels.
  - `skip_logits` per block — gates learnable residual connections from up
    to `skip_span` blocks back.
  - `architecture_mode="dense"` forces all gates to 1 (skips to 0) — this is
    how the dense baselines reuse the same model class.
  - `architecture_cost()` and `sparsity_loss()` are differentiable proxies
    over the gate values, weighted head/FFN/layer/skip — used respectively
    for the Lagrangian budget constraint and the warm-up regularizer.
  - `export_architecture(threshold)` binarizes gates at a threshold into the
    architecture JSON consumed downstream by `materialize.py`.

- **`compact_model.py` — `CompactTransformer` / `CompactConfig` +
  `build_from_sot`**: a physically smaller model with real (not masked)
  tensor shapes per layer, built by `build_from_sot` selecting the
  highest-gated heads/FFN-groups from a trained SOT checkpoint and copying
  those weight slices. This is the "inherited weights" condition. Per-layer
  head count is snapped to a legal divisor of `d_model`.

- **`train.py` — `TrainConfig`, training loop**: trains either a
  `SelfOrganizingTransformer` (`--model_type sot`, gated dense loss +
  Lagrangian-constrained architecture cost + warmup-scheduled sparsity
  regularizer, dual ascent on `lagrange` toward `--target_cost`) or a dense
  baseline (`--model_type dense`, plain LM loss, gates forced to 1). Also
  owns tokenization/packing (`PackedTokens`, `BatchIterator` — fixed-length
  contiguous chunks, not padded/attention-masked sequences) and checkpoint
  saving (writes model, tokenizer, `train_config.json`,
  `architecture.json`, `metrics.json` every `eval_every`/`save_every`
  steps and at `final`).

`retrain_compact.py` reuses `TrainConfig`/`BatchIterator`/`evaluate_loss`/
`load_tokens` from `train.py` but trains a fresh (reinitialized)
`CompactTransformer` from an existing one's config — this is the "fresh
retrain" control that isolates whether the *architecture* helps, as opposed
to the supernet's already-trained weights.

`evaluate.py` computes held-out perplexity directly, and wraps the model in
an `lm_eval.api.model.LM` adapter to run `lm-evaluation-harness` tasks
(`hellaswag`, `arc_easy`, `piqa`, `winogrande` by default). The adapter
implements `loglikelihood`, `loglikelihood_rolling`, and `generate_until`
against the raw `forward()` logits — there is no HF `generate` pipeline
integration beyond that.

### Data flow through the pipeline

```
train.py (sot)  --final checkpoint + architecture.json-->
materialize.py  --build_from_sot: selects+copies highest-gated slices-->
  outputs/sot_compact (CompactTransformer, real shapes) + source_architecture.json
retrain_compact.py  --same architecture, fresh random weights-->
  outputs/discovered_retrained
evaluate.py  --run against any checkpoint dir (sot, dense, sot_compact, discovered_retrained)-->
  perplexity + lm-eval results JSON
```

Both `SelfOrganizingTransformer` and `CompactTransformer` are
`transformers.PreTrainedModel` subclasses, so `save_pretrained` /
`from_pretrained` work as normal, and `push_to_hub` uploads the full
checkpoint directory (weights, tokenizer, architecture JSON, train config,
metrics).

## Working in this codebase

- When changing gate mechanics in `model.py`, update `architecture_cost()`
  and `sparsity_loss()` together — they are the two places gate values feed
  back into the training objective, and their relative weights (0.45/0.50/
  0.05/0.02 and 0.5/0.35/0.15/0.05) encode the assumption that attention and
  FFN dominate compute cost.
- `export_architecture()`'s output shape (per-layer `active`, `heads`,
  `ffn_groups`, `skip_from`, plus raw gate values) is a contract consumed by
  both `materialize.py`/`compact_model.py` and anything plotting "active
  heads by depth" from the JSON — keep field names stable or update both
  consumers.
- `compact_model.py`'s `build_from_sot` assumes exactly the gate/weight
  layout in `model.py` (`qkv` interleaved `[3, heads, head_dim]`,
  `up` interleaved `[2, ffn_dim]`) — a change to either module's weight
  layout must be mirrored in the other.
