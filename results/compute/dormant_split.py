import json, math, sys
sys.path.insert(0, ".")
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from ablation_importance import resolve_checkpoint, get_logit, set_logit, batch_loss
from compute import Replay, Shape
from evaluate import load_model
from train import BatchIterator, PackedTokens, pick_device, tokenise_split
ckpt = resolve_checkpoint("sbartlett97/sot-phase2-wikitext103-heuristic-seed42-tc768d013-v7-lrfreeze")
cfg = json.loads((ckpt / "train_config.json").read_text())
dev = pick_device("auto")
model = load_model(str(ckpt), torch.float32).to(dev).eval()
tok = AutoTokenizer.from_pretrained(ckpt)
val = load_dataset(cfg["dataset_name"], cfg["dataset_config"], split="validation")
packed = PackedTokens(tokenise_split(val, tok, cfg.get("text_column", "text"), len(val)), cfg["seq_len"])
batches = []
for b in BatchIterator(packed, 8, seed=1234):
    batches.append(b.to(dev))
    if len(batches) >= 48: break
s = Shape.from_config(model.config, seq_len=cfg["seq_len"])
st = Replay(s, 42, 0.33, json.load(open("results/phase2_v7_lrfreeze/events.json")), 250).states(10000)
dead_layers = {li for (k, li, _), v in st.items() if k == "layer" and v == "dead"}
base = batch_loss(model, batches); print(f"trained {math.exp(base):.2f}  dead layers {sorted(dead_layers)}")
groups = {
  "dead units + units in dead layers": [u for u, v in st.items() if v == "dead" or (u[0] != "layer" and u[1] in dead_layers)],
  "dormant layers": [u for u, v in st.items() if v == "dormant" and u[0] == "layer"],
  "dormant heads (live layers)": [u for u, v in st.items() if v == "dormant" and u[0] == "head" and u[1] not in dead_layers],
  "dormant ffn groups (live layers)": [u for u, v in st.items() if v == "dormant" and u[0] == "ffn_group" and u[1] not in dead_layers],
}
for name, units in groups.items():
    old = {u: get_logit(model, u) for u in units}
    for u in units: set_logit(model, u, -30.0)
    l = batch_loss(model, batches)
    for u, v in old.items(): set_logit(model, u, v)
    gates = [1/(1+math.exp(-v)) for v in old.values()]
    print(f"{name:<36} n={len(units):<4} gate mean {sum(gates)/max(1,len(gates)):.4f}  ppl {math.exp(l):.2f}")
