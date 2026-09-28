#!/usr/bin/env bash
set -euo pipefail

STEPS=${STEPS:-10000}
SEQ=${SEQ:-256}
BATCH=${BATCH:-16}
ACCUM=${ACCUM:-8}
SEED=${SEED:-42}
RUN_BASELINES=${RUN_BASELINES:-1}

mkdir -p outputs

if [[ "$RUN_BASELINES" == "1" ]]; then
  # Baseline A: standard dense decoder-only Transformer.
  python train.py --model_type dense --output_dir outputs/dense \
    --seq_len "$SEQ" --batch_size "$BATCH" --grad_accum "$ACCUM" \
    --max_steps "$STEPS" --max_layers 12 --d_model 384 --max_heads 6 --max_ffn_dim 1536 --ffn_groups 64 --seed "$SEED"

  # Baseline B: shallower standard Transformer. Same d_model and FFN width.
  python train.py --model_type dense --output_dir outputs/dense_8l \
    --seq_len "$SEQ" --batch_size "$BATCH" --grad_accum "$ACCUM" \
    --max_steps "$STEPS" --max_layers 8 --d_model 384 --max_heads 6 --max_ffn_dim 1536 --ffn_groups 64 --seed "$SEED"

  # Baseline C: same depth but narrower MLP. This gives a conventional compute-reduced point.
  python train.py --model_type dense --output_dir outputs/dense_half_ffn \
    --seq_len "$SEQ" --batch_size "$BATCH" --grad_accum "$ACCUM" \
    --max_steps "$STEPS" --max_layers 12 --d_model 384 --max_heads 6 --max_ffn_dim 768 --ffn_groups 32 --seed "$SEED"
fi

# SOT: over-complete supernet, sparse at initialization, learns architecture with backprop.
python train.py --model_type sot --output_dir outputs/sot \
  --seq_len "$SEQ" --batch_size "$BATCH" --grad_accum "$ACCUM" \
  --max_steps "$STEPS" --max_layers 12 --d_model 384 --max_heads 6 --max_ffn_dim 1536 --ffn_groups 64 \
  --gate_init_logit -1.0 --arch_reg 0.015 --target_cost 0.45 --lagrangian_lr 0.01 --seed "$SEED"

# Physically compact the discovered architecture.
python materialize.py outputs/sot/final --threshold 0.5 --output_dir outputs/sot_compact

# Fresh retraining of exactly that discovered architecture.
python retrain_compact.py outputs/sot_compact --output_dir outputs/discovered_retrained \
  --seq_len "$SEQ" --batch_size "$BATCH" --grad_accum "$ACCUM" --max_steps "$STEPS" --seed "$SEED"

# Identical evaluation protocol.
for m in dense dense_8l dense_half_ffn sot sot_compact discovered_retrained; do
  if [[ -d "outputs/$m/final" ]]; then
    python evaluate.py "outputs/$m/final" --results "outputs/$m/evaluation.json"
  elif [[ -d "outputs/$m" ]]; then
    python evaluate.py "outputs/$m" --results "outputs/$m/evaluation.json"
  fi
done
