# Results files

Raw outputs of the runs in `RESEARCH_LOG.md`. The checkpoints are on the
Hugging Face Hub (see the main `README.md`).

## Phase 1 summaries (`*_summary.json`)

Each file holds `evaluate.py` perplexity and lm-eval results for one
condition and seed.

| files | dataset | valid? |
|---|---|---|
| `{,seed123_,seed7_}{dense,dense_8l,dense_half_ffn,sot,sot_compact,discovered_retrained}_summary.json` | TinyStories | **No.** Made with `transformers==5.17.0`, which corrupts the reload. Use the corrected table in `RESEARCH_LOG.md` (2026-10-06). |
| `gatefix_sot_compact_seed*_summary.json` | TinyStories | Gate fold fixed, but the same faulty `transformers` install. Corrected values: 881.24 / 525.03 / 349.24. |
| `wikitext103_seed42_*_summary.json` | WikiText-103, `target_cost` 0.14 | Yes. |
| `wikitext103_seed{42,123}_tc017_*_summary.json` | WikiText-103, `target_cost` 0.17 | Yes. |
| `*_sot_compact_config.json` | both | Config of the compact model. |

The lm-eval scores are at chance level for every d 384 model. They tell
us nothing at this scale.

## Phase 1 ablation and fine-tune

- `ablation_phase1_tinystories_seed42.json`: loss change when each unit is
  zeroed (2026-10-09).
- `remat_phase1_tinystories_seed42.json`, `finetune*_phase1_*.json`,
  `finetune*_logs/`: re-materialisation with layers 0-1 kept, and the
  300-step fine-tunes (2026-10-09).

## Phase 2 (WikiText-103, d 768, seed 42)

- `ablation_phase2_v4*.json`, `.log`: gate against importance on attempt 4.
- `phase2_v1_base/`, `phase2_v2_relrank/`, `phase2_v3_unblocked/`: attempts 1-3.
  `phase2_tinystories/`: the Phase 2 TinyStories run (d 384). Their Hub
  repos were deleted on 2026-10-16. `MODEL_CARD.md` is the last card.
- `phase2_v5_importance/` ... `phase2_v8_tailfreeze/`: attempts 5-8.
  - `metrics.json`: training records. The val ppl is supernet val ppl.
  - `events.json`: every birth and death.
  - `growth_log.json`: one record for each growth check.
  - `architecture.json`: final gates.
  - `SOURCE_CHECKPOINT`: the Hub repo of the run.
- `phase2_v9_zeroeffect/`: attempt 9, the first run on the zero-effect code.
  Its val ppl is the val ppl of the present structure. The present units
  have gates below 0.5, so `architecture.json` lists no active units at
  that threshold. Present means a non-zero gate.
- `phase2_v7_lrfreeze/tail_birth_ablation.py`: the ablation of the heads
  born in the LR tail.

## Compute accounting (`compute/`)

- `v7_structure_eval.json`, `v8_structure_eval.json`, `v9_structure_eval.json`:
  output of `structure_eval.py`. Val ppl as trained and with the present
  structure only, plus FLOPs and parameters. For attempt 9 the two values
  are the same (30.34), because its inactive units have zero effect.
- `dormant_split.py`, `dead_split.py`: which inactive units carry the gap.
- `present_only_sample.py`: text from attempt 8 with only its present
  structure (see `docs/samples.md`).
