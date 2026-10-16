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

# sot-phase2-wikitext103-heuristic-seed42-tc768d013

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

Threshold (auto-median of this checkpoint's layer gates): `0.3140`

| Layer | Active | Heads | FFN groups | Layer gate |
|---:|:---:|---:|---:|---:|
| 0 | — | 1 | 0 | 0.2908 |
| 1 | — | 1 | 0 | 0.2523 |
| 2 | — | 2 | 0 | 0.3119 |
| 3 | — | 0 | 0 | 0.2620 |
| 4 | — | 0 | 4 | 0.2483 |
| 5 | ✅ | 3 | 11 | 0.4017 |
| 6 | ✅ | 1 | 37 | 0.3288 |
| 7 | ✅ | 0 | 44 | 0.3360 |
| 8 | ✅ | 4 | 46 | 0.4626 |
| 9 | ✅ | 1 | 40 | 0.3162 |
| 10 | ✅ | 3 | 31 | 0.4599 |
| 11 | — | 0 | 38 | 0.2546 |

## Training trajectory

- Final train loss: `3.2363` at step 10000
- Final architectural cost: `0.0630`
- Final val loss: `3.3060` (ppl `27.27`) at step 10000

## Structural plasticity events

- Total births: `68`
- Total deaths: `0`
- Total structural events: `68`

## Performance

_Not yet evaluated._