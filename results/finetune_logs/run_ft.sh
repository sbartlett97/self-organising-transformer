set -e
cd /workspace/sot
SRC=$(python -c "from huggingface_hub import snapshot_download;print(snapshot_download(\"sbartlett97/sot-phase1-tinystories-sot-seed42\", revision=\"73d297f57d382654badf354d72cff859f7ab9777\"))")
python materialize.py $SRC --output_dir remat_gate > /dev/null
python materialize.py $SRC --output_dir remat_gate_plus01 --layers=0,1,2,3,4,5,6,7 > /dev/null
python materialize.py $SRC --output_dir remat_importance_top6 --layers=0,1,2,3,4,5 > /dev/null
python materialize.py $SRC --output_dir remat_all12 --threshold 0.17 --layers=0,1,2,3,4,5,6,7,8,9,10,11 > /dev/null
echo MATERIALIZED
for n in all12 gate_plus01 importance_top6 gate; do
  python retrain_compact.py remat_$n --keep_weights --no_trackio --max_steps 300 --eval_every 100 --warmup_steps 30 --learning_rate 1e-4 --max_train_docs 100000 --output_dir ft_$n > ft_$n.log 2>&1
  echo "FINETUNED $n"; grep val_loss ft_$n.log | tail -3
done
python - <<PY
import json, evaluate
rows={}
for n,label in [("all12","all 12 layers, no skips"),("gate_plus01","gate + layers 0-1 [0..7]"),("importance_top6","importance top-6 [0..5]"),("gate","gate [2..7]")]:
    before=evaluate.ppl(f"remat_{n}")["perplexity"]; after=evaluate.ppl(f"ft_{n}")["perplexity"]
    rows[label]={"ppl_before":before,"ppl_after_300_steps":after}
    print(f"RESULT {label:28s} before={before:9.2f} after={after:7.2f}", flush=True)
json.dump(rows,open("finetune_results.json","w"),indent=2)
PY
echo ALL_DONE
