---
library_name: transformers
pipeline_tag: text-generation
tags:
- self-organizing-transformer
- research
- phase2
- tinystories
- heuristic
---

# sot-phase2-tinystories-heuristic-seed42

Part of the [Self-Organising Transformer](https://github.com/sbartlett97/self-organising-transformer) research project — **phase2**, dataset **tinystories**, condition **`heuristic`**, seed **42**.

## Hyperparameters

| Hyperparameter | Value |
|---|---|
| model_type | sot |
| dataset | roneneldan/TinyStories |
| seed | 42 |
| max_steps | 10000 |
| seq_len | 256 |
| batch_size | 16 |
| grad_accum | 8 |
| max_layers | 12 |
| d_model | 384 |
| max_heads | 6 |
| max_ffn_dim | 1536 |
| ffn_groups | 64 |
| target_cost | 0.14 |
| birth_mode | heuristic |
| seed_fraction | 0.33 |
| growth_interval | 250 |
| birth_limit | 4 |
| death_limit | 2 |
| minimum_age | 1000 |

## Discovered architecture

Threshold (auto-median of this checkpoint's layer gates): `0.3179`

| Layer | Active | Heads | FFN groups | Layer gate |
|---:|:---:|---:|---:|---:|
| 0 | — | 0 | 0 | 0.2918 |
| 1 | — | 0 | 0 | 0.2596 |
| 2 | ✅ | 2 | 2 | 0.3788 |
| 3 | — | 0 | 18 | 0.2933 |
| 4 | — | 0 | 18 | 0.2789 |
| 5 | ✅ | 3 | 21 | 0.4556 |
| 6 | ✅ | 0 | 17 | 0.3474 |
| 7 | ✅ | 2 | 22 | 0.4685 |
| 8 | ✅ | 0 | 25 | 0.3255 |
| 9 | ✅ | 1 | 22 | 0.4364 |
| 10 | — | 0 | 18 | 0.3079 |
| 11 | — | 0 | 12 | 0.3104 |

## Training trajectory

- Final train loss: `1.6657` at step 10000
- Final architectural cost: `0.0716`
- Final val loss: `1.6845` (ppl `5.39`) at step 10000

## Structural plasticity events

- Total births: `59`
- Total deaths: `0`
- Total structural events: `59`

## Performance

**Perplexity**: `4.40`
- beats the matched dense baseline (`4.63`) by `4.9%`

| Task | Metric | Value | Stderr |
|---|---|---:|---:|
| hellaswag | acc | 0.2605 | 0.0044 |
| hellaswag | acc_norm | 0.2589 | 0.0044 |
| arc_easy | acc | 0.2588 | 0.0090 |
| arc_easy | acc_norm | 0.2437 | 0.0088 |
| piqa | acc | 0.5397 | 0.0116 |
| piqa | acc_norm | 0.5212 | 0.0117 |
| winogrande | acc | 0.5012 | 0.0141 |