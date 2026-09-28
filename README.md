# Self-Organising Transformer experiment

This repository implements a controlled test of the hypothesis:

> A deliberately over-complete Transformer can jointly learn weights and its own effective topology, discovering a cheaper heterogeneous architecture during language-model training without a separately defined NAS search stage.

The experiment is deliberately conservative. The first implementation keeps tensor shapes fixed during training and makes topology differentiable through gates. It learns:

- whether each Transformer block should be strongly active;
- which attention heads are useful in each block;
- which groups of MLP channels are useful in each block;
- which short-range cross-layer residual connections are useful.

The discovered architecture is exported to `architecture.json`. A future physical-packing pass can turn that mask into smaller tensors/kernels; keeping this separate prevents kernel implementation details from confounding the scientific result.

## Research design

### Primary hypothesis

At equal training compute, an over-complete Transformer with learned structural gates will discover a heterogeneous topology whose held-out loss and benchmark scores are at least competitive with a dense Transformer of the same maximum envelope, while using materially less effective compute.

### Critical controls

1. **Dense baseline**: same depth, hidden size, heads and MLP envelope, all gates effectively fixed on.
2. **SOT**: same envelope, trainable gates + resource-constrained objective.
3. **Retrained-discovered architecture**: after SOT converges, materialise the architecture and train a fresh model from scratch using the discovered layer/head/MLP/skip structure. This separates “good architecture” from “weight-sharing/supernet” effects.
4. **Seed replication**: run at least 3 seeds for each condition.
5. **Compute-matched comparison**: report total training FLOPs and final active-cost proxy, not just parameter count.

## Recommended first experiment

The default config is deliberately small enough to be practical on a single modern GPU:

- 12 maximum blocks
- `d_model=384`
- 6 attention heads
- 1536 maximum FFN width
- 64 FFN groups
- 3-hop learnable cross-layer residual topology
- context 256
- TinyStories corpus

Use a longer run after the mechanism is validated. TinyStories is a controlled language-modeling environment rather than a claim that this is already an LLM-scale result.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a CUDA machine, install the appropriate PyTorch wheel first if your environment needs a vendor-specific build.

## Run

```bash
./run_experiment.sh
```

Or run just the SOT model:

```bash
python train.py --output_dir outputs/sot --max_steps 10000
```

Evaluation:

```bash
python evaluate.py outputs/sot/final
```

The evaluation reports:

- validation perplexity on TinyStories;
- `hellaswag`;
- `arc_easy`;
- `piqa`;
- `winogrande`.

The latter are run through EleutherAI's `lm-evaluation-harness`, which gives a standardized evaluation interface. Keep the harness version and task versions in the saved results for reproducibility.

## Hugging Face Hub

Authenticate first:

```bash
huggingface-cli login
```

Then:

```bash
python train.py \
  --output_dir outputs/sot \
  --push_to_hub \
  --hub_model_id YOUR_ORG/self-organizing-transformer-tinystories
```

The final checkpoint, tokenizer, architecture JSON, training config and metrics are uploaded.

## Interpreting the result

A convincing positive result requires all of the following:

1. The SOT model reaches similar or lower validation loss than the dense baseline.
2. Its learned cost falls substantially below 1.0.
3. Re-training the discovered architecture from scratch recovers most of the SOT model's quality.
4. The result is stable across random seeds.
5. The same architecture is not simply selecting the maximum allowed capacity everywhere.
6. Benchmark quality remains competitive after physical compaction, rather than only in the masked supernet.

A negative result is also informative. In particular, if gates collapse early, the likely issue is optimisation pressure rather than evidence that self-organisation is impossible; the README's experiment matrix should then be expanded by varying gate warm-up, resource pressure, and growth rules.

## Important limitation

This first implementation learns *effective* topology using differentiable masks. It does not yet repack every selected head/FFN group into physically smaller dense tensors during training. That is intentional: the research question can be tested before introducing hardware-kernel complexity. The exported architecture is the interface for the next phase.

## Suggested second phase

Add a periodic `growth/mutation` operator that can revive dormant layers/heads/groups or create a new skip edge when validation loss plateaus and the measured active cost is below budget. Compare continuous-only SOT against growth+pruning SOT. That experiment directly tests whether the network benefits from genuine architectural innovation, rather than merely selecting a sparse subnetwork.

### Discovered-architecture retraining

After SOT training:

```bash
python materialize.py outputs/sot/final --threshold 0.5 --output_dir outputs/sot_compact
python retrain_compact.py outputs/sot_compact --output_dir outputs/discovered_retrained --max_steps 10000
```

This creates a new model with physically smaller QKV/attention projections and FFNs based on the discovered masks, then trains it from scratch. It is the most important control in the pipeline.

## Recommended experiment matrix

For each seed, run:

| Condition | Purpose |
|---|---|
| Dense 12L/384/1536 | Full standard Transformer reference |
| Dense 8L/384/1536 | Conventional depth reduction |
| Dense 12L/384/768 | Conventional width reduction |
| SOT 12L/384/1536 | Joint topology + weight learning |
| SOT compact, inherited weights | Test whether the learned supernet is useful immediately |
| SOT compact, fresh retrain | Test whether the *architecture itself* is good |

Use the same optimizer, token budget, tokenizer, context length and seed for matched rows. A paper-quality run should repeat every row with at least three independent seeds.

## What to log

In addition to benchmark accuracy, collect:

- training tokens seen;
- wall-clock training time;
- peak GPU memory;
- parameter count before/after compaction;
- estimated training FLOPs;
- estimated inference FLOPs at sequence lengths 128/256/512;
- active layer/head/FFN-group counts by depth;
- number and span of learned skip connections;
- validation loss trajectory;
- benchmark scores with lm-eval task versions.

The architecture JSON is intentionally verbose enough to reconstruct plots such as “active heads by depth” and “effective width by depth”.

## Suggested second experimental phase: genuine structural plasticity

The current SOT tests continuous structural plasticity using dormant capacity. To test your stronger original hypothesis—adding structure that did not initially exist—add a periodic mutation operator:

- initialise only a subset of layers/heads/groups active;
- every `growth_interval` steps, inspect validation loss and current cost;
- if cost is comfortably below target and loss has plateaued, revive one or more dormant units/edges;
- when cost exceeds target, increase pruning pressure;
- record every birth/death event.

This gives a direct dynamic analogue of biological structural plasticity and makes “growth” an observable event rather than merely a gate moving continuously from zero to one.
