---
library_name: transformers
pipeline_tag: text-generation
tags:
- self-organizing-transformer
- research
- phase2
- wikitext103
- heuristic
---

# sot-phase2-wikitext103-heuristic-seed42-tc768d013-v3

Part of the [Self-Organising Transformer](https://github.com/sbartlett97/self-organising-transformer) research project — **phase2**, dataset **wikitext103**, condition **`heuristic`**, seed **42**.

## Hyperparameters

| Hyperparameter | Value |
|---|---|
| model_type | sot |
| dataset | wikitext |
| dataset_config | wikitext-103-raw-v1 |
| seed | 42 |
| max_steps | 10000 |
| seq_len | 256 |
| batch_size | 16 |
| grad_accum | 8 |
| max_layers | 12 |
| d_model | 768 |
| max_heads | 12 |
| max_ffn_dim | 3072 |
| ffn_groups | 128 |
| target_cost | 0.13 |
| birth_mode | heuristic |
| seed_fraction | 0.33 |
| growth_interval | 250 |
| birth_limit | 4 |
| death_limit | 2 |
| minimum_age | 1000 |

## Discovered architecture

Threshold (auto-median of this checkpoint's layer gates): `0.3992`

| Layer | Active | Heads | FFN groups | Layer gate |
|---:|:---:|---:|---:|---:|
| 0 | — | 0 | 0 | 0.2621 |
| 1 | — | 0 | 0 | 0.2594 |
| 2 | — | 0 | 0 | 0.2854 |
| 3 | — | 0 | 0 | 0.3305 |
| 4 | ✅ | 1 | 0 | 0.4083 |
| 5 | ✅ | 1 | 0 | 0.4603 |
| 6 | ✅ | 1 | 0 | 0.4735 |
| 7 | — | 0 | 0 | 0.3785 |
| 8 | ✅ | 0 | 0 | 0.4545 |
| 9 | ✅ | 1 | 0 | 0.4801 |
| 10 | ✅ | 0 | 0 | 0.4370 |
| 11 | — | 0 | 0 | 0.3901 |

## Training trajectory

- Final train loss: `3.3170` at step 10000
- Final architectural cost: `0.0850`
- Final val loss: `3.4624` (ppl `31.89`) at step 10000

## Structural plasticity events

- Total births: `160`
- Total deaths: `0`
- Total structural events: `160`

## Performance

_Not yet evaluated._