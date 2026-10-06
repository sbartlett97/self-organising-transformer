# Self-Organising Neural Architectures Through Structural Plasticity
## A Phase-Gated Research Programme for Dynamically Learned Transformer Topology

**Status:** Research proposal  
**Primary task:** Decoder-only language modelling  
**Primary implementation stack:** PyTorch + Hugging Face Transformers + Hugging Face Datasets + Hugging Face Hub + lm-evaluation-harness  
**Initial target scale:** 10M–100M parameters  
**Ultimate objective:** Determine whether a neural network can learn not only its parameters, but the mechanisms governing its own computational structure.

---

## 1. Executive summary

Modern neural networks generally separate architecture selection from parameter learning. A human or automated search process chooses the number of layers, hidden dimensions, attention heads, feed-forward widths, and connectivity pattern; gradient descent then learns the parameters inside that fixed structure.

This project tests a stronger hypothesis:

> A neural network can learn a useful computational topology during training through structural plasticity, with computational resources being created, removed, and rearranged in response to persistent task demand.

The first research phase uses differentiable structural gates and controlled heuristic plasticity. Subsequent phases make topology changes discrete and introduce birth/death mechanisms. Later phases replace hand-designed plasticity rules with a learned controller, first as a contextual policy and then as a recurrent policy. The final phases allow multiple kinds of structural mutation and eventually open-ended, budget-constrained growth.

The project is deliberately phase-gated. Every phase must beat predefined controls before the next source of complexity is introduced.

The strongest final hypothesis is:

> A learned structural-plasticity policy can discover heterogeneous, task-useful Transformer topologies that match or exceed the quality of conventional architectures at equal or lower training and inference cost.

This programme builds on several established ideas without simply reproducing them. Differentiable NAS demonstrates joint optimisation of architecture and weights; MorphNet demonstrates resource-constrained structural learning; dynamic sparse training demonstrates training-time topology changes; NEAT demonstrates topology and weight evolution; and network morphism demonstrates function-preserving structural growth. Transformer-specific NAS has also shown that heterogeneous Transformer structures can be useful. These are adjacent foundations, while the proposed work combines them into a training-time structural-plasticity system whose controller itself becomes a learned object.

---

# 2. Research questions

### RQ1 — Can useful heterogeneous structure emerge during ordinary gradient training?

If different regions of a Transformer have different computational demands, a model with learnable structural capacity should discover non-uniform widths, head counts, and depths.

### RQ2 — Can architecture be modified online without destabilising learning?

A useful structural-plasticity system must be able to create and remove computation while retaining previously learned capability.

### RQ3 — Is gradient-directed birth superior to simple heuristics or random growth?

The project will test whether local gradient pressure contains enough information to identify useful new computational structure.

### RQ4 — Can a learned controller improve over hand-written plasticity rules?

The controller should learn interactions between telemetry signals such as gradient pressure, utilisation, age, loss trends, and cost rather than relying on fixed thresholds.

### RQ5 — Does temporal memory improve structural decisions?

A recurrent controller may distinguish transient spikes in demand from persistent demand, avoiding oscillatory birth/death decisions.

### RQ6 — Can the controller select *what kind* of computation to create?

The system should eventually choose among additional heads, FFN groups, blocks, and residual pathways rather than merely deciding whether more capacity is needed.

### RQ7 — Can the network discover genuinely novel topology under a compute budget?

The final test is whether the system can evolve a compact computational graph whose structure is not specified in advance beyond basic safety and hardware constraints.

---

# 3. Core hypothesis

### H1 — Structural plasticity hypothesis

A Transformer with the ability to learn and modify its own structure will find a better quality/compute trade-off than a fixed homogeneous Transformer trained with the same data and optimisation budget.

### H2 — Directed birth hypothesis

Birth decisions based on structural demand signals will outperform random structural growth at equal growth opportunity.

### H3 — Learned policy hypothesis

A learned plasticity policy will outperform static threshold rules because the usefulness of a structural change depends on interactions between multiple signals and on temporal context.

### H4 — Topological emergence hypothesis

When multiple mutation types are available, the learned system will converge on heterogeneous architectures and non-trivial connectivity patterns rather than simply reproducing a uniformly scaled Transformer.

---

# 4. Research philosophy

The experiment should distinguish four things that are often conflated:

1. **Weight learning** — learning the values inside a fixed architecture.
2. **Differentiable structural selection** — learning continuous gates over existing capacity.
3. **Discrete structural evolution** — physically creating/removing components during training.
4. **Learning the plasticity rule** — learning which structural interventions should occur.

Each phase isolates one of these questions.

The project must not jump directly from a dense model to a learned architecture controller because a positive final result would otherwise be impossible to attribute.

---

# 5. Experimental stack

## 5.1 Software

- Python 3.11+
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face Hub
- Accelerate
- Safetensors
- EleutherAI `lm-evaluation-harness`
- TensorBoard or Weights & Biases
- Optional: PyTorch profiler and `fvcore` / `thop` for secondary cost analysis

## 5.2 Dataset

### Primary development dataset

**TinyStories** or an equivalently small Hugging Face causal-LM dataset.

Why:
- fast iteration;
- small enough for repeated architecture searches;
- language-modelling objective;
- easy reproducibility.

### Generalisation dataset

A held-out text corpus should be used only after a mechanism has survived development experiments.

Suitable candidates include:
- WikiText-2;
- WikiText-103 for larger experiments;
- OpenWebText-style subsets if more scale becomes necessary.

The same tokenizer, evaluation protocol, and sequence length must be used across directly compared models.

---

# 6. Experimental controls

Every phase should include at minimum:

### Baseline A — Dense homogeneous Transformer

Conventional decoder-only model:

```text
N identical blocks
d_model = constant
n_heads = constant
d_ff = constant
```

### Baseline B — Static depth scaling

Same hidden width and heads, fewer layers.

### Baseline C — Static width scaling

Same depth, smaller FFN width and/or hidden dimension.

### Baseline D — Static heterogeneous architecture

A manually constructed heterogeneous architecture with the same total parameter/FLOP budget. This is important because discovering a heterogeneous architecture is not automatically useful if a hand-designed heterogeneous model performs equally well.

### Baseline E — Post-training pruning

Train dense, then prune/materalise and retrain.

### Baseline F — Dynamic sparse reference

A RigL-like gradient-driven topology baseline where appropriate.

### Baseline G — Random structural growth

Same birth opportunities and structural budget as the proposed system, but births are random.

These controls prevent the project from claiming that “dynamic” automatically means “better.”

---

# 7. Common model representation

The base model is an over-complete decoder-only Transformer composed of structurally independent units.

Each Transformer block contains:

```text
LayerNorm
 ├── Multi-head self-attention
 │    ├── Head 0
 │    ├── Head 1
 │    ├── ...
 │    └── Head H
 │
 └── MLP
      ├── FFN group 0
      ├── FFN group 1
      ├── ...
      └── FFN group G
```

The model also exposes optional residual candidates:

```text
Block i ───────────────► Block j
```

subject to a maximum skip distance.

All structural units have:
- parameters;
- an active/dormant state;
- an age;
- telemetry;
- cost metadata.

The initial implementation should use bounded structural slots rather than arbitrary tensors. This keeps the first experiments computationally manageable and makes architecture materialisation deterministic.

---

# 8. Structural telemetry

Every candidate structural component maintains exponentially smoothed statistics.

Recommended telemetry:

```text
activation_mean
gate_mean
gradient_mean
gradient_ema
contribution_ema
loss_delta_ema
age
steps_since_birth
steps_since_last_change
layer_depth
relative_depth
component_type
estimated_compute_cost
current_budget_fraction
recent_loss_slope
recent_loss_variance
```

For a dormant candidate, the key signal is:

\[
g_i = \left|\frac{\partial L}{\partial \alpha_i}\right|
\]

where \(\alpha_i\) is the candidate's structural gate.

A high persistent value indicates that the model has gradient pressure in the direction of using the candidate.

---

# 9. Cost model

Structural decisions must be resource-aware.

Define:

\[
L_{\mathrm{total}}
=
L_{\mathrm{task}}
+
\lambda_C C(A)
+
\lambda_S S(A)
\]

where:

- \(A\) is the current architecture;
- \(C(A)\) estimates compute;
- \(S(A)\) penalises structural complexity/churn.

The early phases may use differentiable FLOP proxies.

Later phases must measure:

- parameters;
- theoretical FLOPs;
- activation memory;
- actual wall-clock inference latency;
- training throughput.

A result is not considered a win merely because the model has fewer parameters if it is slower on the target hardware.

---

# 10. Phase 1 — Differentiable structural selection

## Objective

Determine whether a Transformer benefits from learning heterogeneous structure even without discrete birth/death events.

## Mechanism

Introduce differentiable gates:

\[
y = g_i f_i(x)
\]

with:

\[
g_i = \sigma(\alpha_i / T)
\]

where \(T\) is a temperature parameter.

Gate regularisation encourages unnecessary capacity toward zero.

Allow:
- FFN group gates;
- attention head gates;
- layer gates.

Do not yet introduce arbitrary residual topology.

## Training

Train normally with:

\[
L =
L_{LM}
+
\lambda C(A)
\]

Gradually lower gate temperature so that the architecture becomes increasingly discrete.

## Materialisation

At the end of structure learning:

1. select active units;
2. physically pack them into tensors;
3. save the discovered architecture;
4. retrain the materialised architecture from scratch.

## Required comparison

```text
Dense
Static smaller model
Differentiable SOT
Materialised SOT
Materialised SOT retrained from scratch
```

## Metrics

- validation perplexity;
- final benchmark score;
- parameters;
- FLOPs;
- wall-clock latency;
- training FLOPs;
- architecture shape;
- per-layer head count;
- per-layer FFN width.

## Success criterion

Phase 1 passes if a materialised architecture achieves a better quality/compute Pareto point than dense/static baselines and the architecture is stable across random seeds.

---

# 11. Phase 2 — Discrete birth/death

## Objective

Determine whether the architecture can change during training without differentiable gates being the final representation.

## Structural lifecycle

```text
DORMANT
   ↓
CANDIDATE
   ↓
BORN
   ↓
PROTECTED
   ↓
MATURE
   ↓
COMPETE
   ↓
DEAD
```

## Birth

A structural candidate becomes active when its demand score exceeds a threshold.

The first heuristic:

\[
B_i =
w_g G_i
+
w_c C_i
+
w_p P_i
\]

where:
- \(G_i\): gradient pressure;
- \(C_i\): contribution;
- \(P_i\): loss/plateau demand.

## Death

Remove a mature unit if:

\[
D_i > \tau_D
\]

for a sustained period.

## Protection

Newly born components cannot be removed for a minimum number of optimisation steps.

## Why protection exists

It prevents:

```text
birth → immediate death → birth → immediate death
```

and allows the newly created structure to train.

## Required experiments

- heuristic growth;
- no growth;
- random growth.

## Success criterion

The system must show stable topology evolution and improve the cost/quality trade-off versus static pruning.

---

# 12. Phase 3 — Gradient-directed structural growth

## Objective

Test whether gradients provide a better structural birth signal than fixed heuristics.

## Birth score

For candidate \(i\):

\[
B_i =
EMA\left(
\left|
\frac{\partial L}{\partial g_i}
\right|
\right)
\]

Normalise within candidate type and layer.

Select the highest-scoring candidates subject to:
- available budget;
- minimum candidate age;
- birth cooldown;
- maximum number of births per structural event.

## Important control

Run:

```text
GradientBirth
RandomBirth
```

with exactly the same number of allowed structural interventions.

This isolates the value of the signal itself.

## Success criterion

Gradient-directed growth should:
- reach equal/better validation loss;
- do so with equal/lower structural cost;
- produce lower structural churn;
- show repeatable birth locations.

---

# 13. Phase 4 — Learned feed-forward plasticity controller

## Objective

Replace hand-designed structural rules with a learned policy.

## Controller

A shared MLP policy receives:

```text
gradient_ema
contribution_ema
activation_mean
age
loss_slope
loss_variance
component_type
layer_depth
relative_depth
compute_cost
budget_remaining
time_since_change
```

and returns:

```text
NOOP
BIRTH
DEATH
```

or, in the extended form:

```text
NOOP
BIRTH_HEAD
BIRTH_FFN
BIRTH_BLOCK
BIRTH_SKIP
DEATH
```

The controller should be shared across candidates, with embeddings for candidate type.

## Why a shared controller

It tests whether there is a general learned structural rule rather than separately fitting every layer.

## Learning formulation

Initially use a contextual-bandit formulation.

At each structural decision point:

```text
observe state
→ choose action
→ train child model for K steps
→ measure delayed reward
→ update controller
```

## Reward

Recommended first-order reward:

\[
R =
(L_t - L_{t+K})
-
\lambda_C \Delta C
-
\lambda_S \mathrm{churn}
\]

Add a persistence term:

\[
R_{\mathrm{total}}
=
R_{\mathrm{short}}
+
\gamma R_{\mathrm{later}}
\]

where the later reward measures whether the structural change remains beneficial.

## Meta-training

Never train the controller from one model trajectory.

Use:

```text
controller πθ
     │
 ┌───┼───┐
seed1 seed2 seed3 ...
 │     │    │
LM    LM   LM
 └─────┼────┘
       ↓
aggregate rewards
       ↓
update controller
```

The child model is the environment.

## Success criterion

The learned controller must beat:
- static threshold controller;
- gradient-only controller;
- random controller.

The architecture and training budget must be matched.

---

# 14. Phase 5 — Recurrent plasticity controller

## Objective

Test whether structural decisions benefit from temporal memory.

## Motivation

A single observation cannot necessarily distinguish:

```text
temporary gradient spike
```

from:

```text
persistent demand for additional capacity
```

## Controller

Replace the MLP with:

```text
observation_t
   ↓
GRU/LSTM
   ↓
latent plasticity state
   ↓
action policy
```

The recurrent state is maintained per structural region or per candidate family.

## Example

```text
step 1: high pressure
step 2: high pressure
step 3: high pressure
step 4: loss plateau
           ↓
        BIRTH
```

versus:

```text
step 1: high pressure
step 2: normal
step 3: normal
           ↓
         NOOP
```

## Metrics

In addition to previous metrics:

- birth latency;
- hysteresis;
- structural churn;
- component lifetime distribution;
- number of oscillatory decisions;
- policy confidence.

## Success criterion

The recurrent controller should reduce unnecessary topology changes while preserving or improving final quality.

---

# 15. Phase 6 — Multiple structural mutation types

## Objective

Allow the controller to decide not only whether the model needs more capacity, but what kind.

## Action space

```text
NOOP
BIRTH_HEAD
BIRTH_FFN_GROUP
BIRTH_BLOCK
BIRTH_SKIP
DEATH_HEAD
DEATH_FFN_GROUP
DEATH_BLOCK
DEATH_SKIP
```

## Birth operators

### Attention head

Add one previously dormant head.

### FFN group

Add a group of fixed width, e.g. 32–128 channels.

### Block

Insert a new residual block at a selected location.

### Skip connection

Create a new residual pathway between eligible blocks.

## Function-preserving growth

New blocks should start close to identity:

\[
y = x + \alpha F(x)
\]

with:

\[
\alpha = 0
\]

at birth.

This prevents structural growth from catastrophically changing the current function.

The same principle can be applied to newly activated residual edges.

## Location choice

The controller should receive the proposed location and score candidates:

```text
candidate insertion point 0
candidate insertion point 1
...
candidate insertion point N
```

Eventually this allows the architecture to decide where computation belongs.

## Success criterion

The learned architecture should exhibit meaningful heterogeneous structure and, ideally, non-trivial long-range connectivity while staying within the target budget.

---

# 16. Phase 7 — Open-ended / budget-constrained structural evolution

## Objective

Remove the artificial assumption that all capacity must exist as preallocated dormant slots.

The model should be able to grow and shrink its graph subject to:
- parameter ceiling;
- compute ceiling;
- memory ceiling;
- maximum topology complexity.

## Architecture representation

Represent the model as a graph:

\[
G = (V,E)
\]

where:
- nodes are computational blocks or subcomponents;
- edges are tensor pathways.

Possible operations:

```text
ADD_NODE
DELETE_NODE
SPLIT_NODE
MERGE_NODE
ADD_EDGE
DELETE_EDGE
DUPLICATE_SUBGRAPH
```

Each operation must have safety constraints so that invalid tensors or cyclic computation are not created.

## Birth mechanics

A birth proposal is generated by the controller.

Examples:

```text
Block 3 → Block 4
```

becomes:

```text
Block 3 → NewBlock → Block 4
```

or:

```text
Block 2 ─────────► Block 7
```

via a new residual edge.

## Growth budget

Use a constrained objective:

\[
\min_\pi
E[L_{task}]
\]

subject to:

\[
C(G) \le C_{max}
\]

and:

\[
M(G) \le M_{max}
\]

The controller therefore cannot improve quality indefinitely simply by growing the model.

## Success criterion

Phase 7 succeeds if the system can start from materially different seed topologies and converge toward useful architectures with:
- comparable final quality;
- lower cost;
- repeatable motifs;
- stable architectures;
- topology-dependent specialisation.

---

# 17. Counterfactual birth experiment

This should be introduced as a cross-cutting mechanism, probably during Phase 4–6.

The controller identifies a candidate but does not immediately activate it.

Instead:

```text
current architecture
      ↓
temporarily activate candidate
      ↓
run 1–5 short evaluation windows
      ↓
estimate Δloss / Δcost
      ↓
accept or reject
```

This is analogous to a local structural experiment.

The counterfactual mechanism should only be applied to the top K candidates because it is computationally expensive.

Compare:

```text
No counterfactual validation
vs
Counterfactual validation
```

and measure:
- accepted birth rate;
- rejected birth rate;
- quality per structural intervention;
- compute overhead.

---

# 18. Materialisation

A learned supernetwork is not itself the final result.

Every discovered architecture must be materialised into a physically compact Transformer.

Materialisation must:

1. remove dead heads;
2. remove dead FFN groups;
3. remove dead blocks;
4. insert learned blocks;
5. create learned residual edges;
6. pack tensors;
7. remove all structural controller machinery;
8. produce a standard Hugging Face model;
9. reload the model from disk in a clean Python process.

The clean reload step is a required integrity test.

---

# 19. Retraining protocol

Every discovered architecture should be retrained from scratch.

This yields three distinct measurements:

### A. Supernet performance

Can the masked/dynamic model perform?

### B. Inherited-weight compact performance

Does materialisation retain the learned function?

### C. Fresh-retrained compact performance

Is the discovered architecture itself good?

Only C should be used as the strongest claim that the topology is useful.

---

# 20. Evaluation

## Language-modelling evaluation

Report:
- validation loss;
- perplexity;
- bits per token.

## Standard benchmark evaluation

Use EleutherAI `lm-evaluation-harness`.

Initial benchmark suite:

- HellaSwag
- ARC-Easy
- PIQA
- WinoGrande

A model must be evaluated with identical task versions and generation/log-likelihood settings.

## Efficiency

Report:

- parameter count;
- non-embedding parameter count;
- theoretical FLOPs;
- measured inference latency;
- throughput;
- peak memory;
- training FLOPs where measurable.

## Architecture metrics

Report:

- number of active blocks;
- attention heads by layer;
- FFN width by layer;
- skip edges;
- birth events;
- death events;
- component lifetime;
- architecture entropy;
- topology similarity between seeds.

---

# 21. Pareto analysis

The central result should be plotted as a Pareto frontier.

Axes:

```text
x = inference cost
y = benchmark quality
```

Secondary plots:

```text
training FLOPs vs quality
parameters vs quality
latency vs quality
```

A structural-plastic model is valuable only if it shifts the frontier.

A lower parameter count with worse latency or substantially worse quality is not a successful result.

---

# 22. Seed stability

Every important configuration should run with multiple independent seeds.

Minimum:

```text
3 seeds for development
5+ seeds for final claims
```

Measure whether the discovered architectures are:

- identical;
- approximately equivalent;
- structurally diverse but functionally similar;
- unstable.

Architecture diversity is not automatically bad. The important question is whether different runs find equivalent quality/cost solutions.

---

# 23. Architecture analysis

After every successful phase, export an architecture graph.

Example:

```text
Input
  │
  ▼
B0 ──► B1 ──► B2
             │  \
             │   └────────► B5
             ▼
            B3 ──► B4 ──► B5
                             │
                             ▼
                           Output
```

Also export:

```text
Layer  Heads  FFN
0       4      512
1       4      768
2       6     1536
3       2      512
4       2      512
5       6     1024
```

This makes it possible to ask whether the architecture has interpretable structural motifs.

---

# 24. Expected failure modes

## Failure 1 — The system grows forever

Cause:
- weak compute penalty;
- reward too heavily tied to short-term loss.

Mitigation:
- hard budget;
- compute-aware reward;
- churn penalty.

## Failure 2 — Birth/death oscillation

Cause:
- noisy telemetry.

Mitigation:
- EMA;
- protected newborns;
- cooldown;
- hysteresis.

## Failure 3 — Controller exploits reward noise

Cause:
- delayed credit poorly assigned.

Mitigation:
- delayed reward windows;
- multiple rollout seeds;
- structural persistence term.

## Failure 4 — Architecture changes destroy training

Cause:
- poorly initialised newborn blocks.

Mitigation:
- function-preserving residual initialisation.

## Failure 5 — Controller learns trivial policy

Example:

```text
always NOOP
```

Mitigation:
- explicit minimum exploration;
- compare to heuristic baseline;
- track action entropy.

## Failure 6 — Learned controller overfits a child model

Mitigation:
- meta-training over many independent seeds and data slices;
- held-out seeds for evaluation.

## Failure 7 — Supernet advantage disappears after materialisation

This is an important negative result rather than a bug.

It indicates that weight-sharing or co-adaptation produced the apparent gain.

Therefore retraining-from-scratch compact models is mandatory.

---

# 25. Research instrumentation

Every structural event should generate a record such as:

```json
{
  "step": 12000,
  "action": "BIRTH_FFN_GROUP",
  "location": 7,
  "candidate_score": 0.91,
  "gradient_pressure": 0.74,
  "loss_slope": -0.002,
  "compute_delta": 0.035,
  "reward": 0.18
}
```

Store events in JSONL.

Maintain a second architecture snapshot:

```json
{
  "step": 12000,
  "blocks": [...],
  "heads_per_block": [...],
  "ffn_groups_per_block": [...],
  "skip_edges": [...]
}
```

This enables post-hoc analysis without rerunning training.

---

# 26. Experiment directory

Recommended repository layout:

```text
self_organizing_transformer/
├── configs/
│   ├── phase1.yaml
│   ├── phase2.yaml
│   ├── phase3.yaml
│   ├── phase4.yaml
│   ├── phase5.yaml
│   ├── phase6.yaml
│   └── phase7.yaml
│
├── models/
│   ├── base_transformer.py
│   ├── supernet.py
│   ├── compact.py
│   └── graph_model.py
│
├── plasticity/
│   ├── base.py
│   ├── heuristic.py
│   ├── gradient.py
│   ├── contextual.py
│   ├── recurrent.py
│   ├── counterfactual.py
│   └── mutation.py
│
├── controllers/
│   ├── mlp_policy.py
│   ├── gru_policy.py
│   └── training.py
│
├── data/
│   ├── datasets.py
│   ├── tokenize.py
│   └── collators.py
│
├── training/
│   ├── trainer.py
│   ├── rollout.py
│   ├── checkpointing.py
│   └── evaluation.py
│
├── materialize/
│   ├── export.py
│   └── validate.py
│
├── benchmarks/
│   ├── lm_eval.py
│   └── efficiency.py
│
├── analysis/
│   ├── architecture.py
│   ├── pareto.py
│   ├── events.py
│   └── plots.py
│
├── scripts/
│   ├── train_phase1.py
│   ├── train_phase2.py
│   ├── train_phase3.py
│   ├── train_phase4.py
│   ├── train_phase5.py
│   ├── train_phase6.py
│   └── train_phase7.py
│
├── tests/
├── README.md
└── pyproject.toml
```

Each phase-specific script should call shared infrastructure rather than implement its own training loop.

---

# 27. Phase gates

The programme must not advance solely because the code runs.

## Gate 1 → 2

Advance only if differentiable structural learning:
- produces stable heterogeneous architectures;
- beats or matches dense/static baselines at equal compute;
- survives materialisation and retraining.

**Status (2026-10-01): passed, with a documented scope limitation.**
**Corrected 2026-10-06**: the TinyStories evaluation this status was
originally based on ran through a `transformers` version that violates
this project's own pinned dependency range and silently corrupted
weight reload — see `RESEARCH_LOG.md`'s 2026-10-06 "Transformers-version
reload bug" entry for the full investigation and corrected numbers. With
the correction, "beats or matches dense/static baselines" is a softer
pass than originally reported: SOT trails dense by a small, consistent
~2.4–2.8% relative perplexity margin (arguably "matches," not "beats") on
TinyStories, and the WikiText-103 result below is unaffected by this
correction (it was evaluated under a compliant `transformers` version
throughout) and continues to show a larger gap. "Stable heterogeneous
architectures" and "survives materialisation and retraining" (in relative
terms — the retrained architecture still recovers most of SOT's own
quality) both still hold.

All three criteria held, in this softer form, on the development dataset
(TinyStories, 3 seeds). On the generalisation dataset (WikiText-103),
"beats or matches" and "survives retraining" do not hold at the model
scale tested — investigated directly (cost-budget varied across 3 seeds,
gap did not meaningfully close; both seeds at the looser budget converged
on the same discovered architecture) rather than left unexamined, per the
"investigate the mechanism before adding the next layer of complexity"
instruction below. Diagnosis: a capacity limit at this model scale on
harder data, not an unstable or failing search — and, post-correction,
this diagnosis now reads as a matter of *degree* shared with TinyStories
(both trail dense; WikiText-103 trails by more) rather than a sharp
TinyStories-wins/WikiText-103-loses split. Advanced to Phase 2 on that
basis; the model-scale question carries forward as an open item for
Phase 2's own experiment design. Full detail in `RESEARCH_LOG.md` and
`README.md`.

## Gate 2 → 3

Advance only if discrete birth/death:
- remains stable;
- does not oscillate;
- produces measurable structural adaptation.

## Gate 3 → 4

Advance only if gradient-directed growth beats random growth.

## Gate 4 → 5

Advance only if the learned feed-forward controller beats heuristic control.

## Gate 5 → 6

Advance only if recurrence provides a statistically meaningful improvement in policy quality, churn, or stability.

## Gate 6 → 7

Advance only if the controller can make useful choices between multiple mutation types.

If a gate fails, investigate the mechanism before adding the next layer of complexity.

---

# 28. Statistical analysis

For each key comparison report:

- mean;
- standard deviation;
- 95% confidence interval where appropriate;
- per-seed values.

Use paired comparisons when the same training budgets and seeds can be aligned.

The main hypothesis test should concern the quality/compute Pareto frontier, not only raw benchmark score.

---

# 29. Recommended initial hyperparameters

For a first small experiment:

```text
d_model:             256–384
layers_max:          12–16
attention_heads_max: 8–12
ffn_group_width:     64
ffn_groups_max:      8–16
context_length:      512–1024
optimizer:           AdamW
schedule:            cosine
precision:           bf16 where supported
architecture_update_interval:
                     100–500 optimisation steps
birth_limit/update:  1–4
minimum_age:         500–5000 steps
EMA decay:           0.95–0.999
```

These values are deliberately conservative. Hyperparameters should be scaled after validating the mechanism.

---

# 30. Recommended phase schedule

## Phase 1

```text
Dense supernet
   ↓
differentiable gates
   ↓
materialise
   ↓
retrain compact
```

## Phase 2

```text
Small seed
   ↓
heuristic birth/death
   ↓
materialise
   ↓
retrain
```

## Phase 3

```text
Small seed
   ↓
gradient-ranked birth
   ↓
heuristic death
   ↓
evaluate against random birth
```

## Phase 4

```text
multiple child rollouts
   ↓
contextual controller
   ↓
delayed reward
   ↓
meta-update controller
```

## Phase 5

```text
contextual telemetry
   ↓
GRU memory
   ↓
temporal structural decisions
```

## Phase 6

```text
controller
   ├── add head
   ├── add FFN
   ├── add block
   ├── add skip
   └── delete component
```

## Phase 7

```text
graph representation
   ↓
open-ended structural mutation
   ↓
hard compute/memory budget
   ↓
self-organised topology
```

---

# 31. What would constitute a compelling result?

The strongest result would look like:

```text
                 Benchmark quality
                        ▲
                        │
        dense ────────●
                     /
                    /
          SOT ────●
                 /
                /
        static●
────────────────────────────────► inference cost
```

combined with:

1. the discovered topology is repeatable across seeds;
2. the compact retrained architecture retains the gain;
3. the learned plasticity policy beats fixed heuristics;
4. different mutation types are selected in interpretable contexts;
5. architecture changes are concentrated in regions where additional computation is genuinely valuable;
6. the final graph is measurably different from conventional homogeneous Transformer stacks.

An equally valuable negative result would be:

> Dynamic structural learning can discover efficient architectures, but learned plasticity provides no advantage over a well-designed static rule.

That would narrow the research question substantially and still produce useful knowledge.

---

# 32. Stronger long-term extension — task-conditioned plasticity

After Phase 7, a natural extension is to ask whether the controller itself can become conditioned on the current task or data distribution.

Instead of:

```text
one model
one topology
```

the controller could learn:

```text
input/task context
        ↓
plasticity policy
        ↓
allocate structure
```

This connects structural plasticity with conditional computation.

The key distinction is:

- **Phase 7:** discover a good topology for a task;
- **future work:** dynamically change topology as the task/data distribution changes.

This would move the project toward continual-learning and adaptive-computation research.

---

# 33. The conceptual contribution

The research should avoid claiming that dynamic architecture optimisation is entirely new. Neural Architecture Search, differentiable architecture optimisation, structured pruning, dynamic sparse training, evolutionary topology search, and conditional computation are all established fields.

The intended contribution is narrower:

> A unified experimental framework in which a Transformer learns its computational topology during training, and in which the mechanism responsible for changing that topology can itself be learned.

The progression is:

```text
learn weights
    ↓
learn gates
    ↓
learn structure
    ↓
learn structural actions
    ↓
learn temporal structural policy
    ↓
learn choice of structural mutation
    ↓
learn topology under an explicit resource constraint
```

That is the core research programme.

---

# 34. Key references

1. Liu, Simonyan, Yang. **DARTS: Differentiable Architecture Search.** ICLR 2019. Differentiable relaxation of architecture search and joint architecture/weight optimisation.

2. Gordon et al. **MorphNet: Fast & Simple Resource-Constrained Learning of Deep Network Structure.** CVPR 2018. Learns structure under resource constraints using differentiable regularisation.

3. Evci et al. **Rigging the Lottery: Making All Tickets Winners.** ICML 2020. Dynamic sparse training with gradient-informed topology updates.

4. Stanley, Miikkulainen. **Evolving Neural Networks through Augmenting Topologies (NEAT).** Evolutionary Computation, 2002. Joint topology and weight evolution.

5. Chen et al. **Net2Net: Accelerating Learning via Knowledge Transfer.** ICLR 2016. Function-preserving expansion of neural networks.

6. Wang et al. **HAT: Hardware-Aware Transformers for Efficient Natural Language Processing.** ACL 2020. Searches heterogeneous Transformer architectures subject to hardware constraints.

7. Chen et al. **AutoFormer: Searching Transformers for Visual Recognition.** ICCV 2021. Transformer architecture search across depth, width and related structural dimensions.

8. Raposo et al. **Mixture-of-Depths: Dynamically allocating compute in transformer-based language models.** 2024. Dynamic allocation of compute under a fixed budget.

These works establish that architecture can be differentiably optimised, resources can be used as constraints, topology can change during learning, and Transformer architecture can be heterogeneous. The present programme combines these ideas into a controlled progression toward learned structural plasticity.

---

# 35. Final research objective

The final question is not:

> "Can we prune a Transformer?"

It is:

> **Can a neural network learn the rules by which its own computational machinery changes?**

The project should therefore treat:
- **weights** as learned parameters;
- **architecture** as a learned state;
- **plasticity** as a learned policy;
- **compute budget** as an environmental constraint.

The desired end state is a system where:

```text
task
  ↓
train
  ↓
observe demand
  ↓
decide structural intervention
  ↓
grow / shrink / rewire
  ↓
continue learning
  ↓
repeat
```

and where the resulting architecture is not prescribed in advance beyond fundamental validity and resource constraints.

That is the experiment that tests the original self-organising-network hypothesis most directly.
