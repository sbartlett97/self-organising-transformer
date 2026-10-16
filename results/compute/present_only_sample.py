"""Sample attempt 8 with only its present structure (every other unit at scale 0).

Attempt 8 predates the scale buffers, so it loads with every scale at 1 (the
full supernet). This sets scale 0 on each unit that compute.Replay does not
count as present at step 10000, which is the model structure_eval.py scores
at 132 val ppl. Run from the repo root:
  python results/compute/present_only_sample.py
"""
import json, sys
sys.path.insert(0, ".")
import torch
from transformers import AutoTokenizer
from ablation_importance import resolve_checkpoint
from compute import Replay, Shape
from evaluate import device_from, load_model
from generate import sample

ckpt = resolve_checkpoint("sbartlett97/sot-phase2-wikitext103-heuristic-seed42-tc768d013-v8-tailfreeze")
cfg = json.loads((ckpt / "train_config.json").read_text())
dev = device_from("auto")
model = load_model(str(ckpt), None).to(dev).eval()
tok = AutoTokenizer.from_pretrained(ckpt)
s = Shape.from_config(model.config, seq_len=cfg["seq_len"])
st = Replay(s, cfg["seed"], cfg["seed_fraction"], json.load(open("results/phase2_v8_tailfreeze/events.json")), 250).states(10000)
present = {u for u, v in st.items() if v in ("alive", "dying")}
with torch.no_grad():
    for (kind, li, i), v in st.items():
        on = (kind, li, i) in present and ("layer", li, 0) in present
        buf = {"layer": model.layer_scale, "head": model.head_scale, "ffn_group": model.group_scale}[kind]
        if kind == "layer":
            buf[li] = float(on)
        else:
            buf[li, i] = float(on)
print(f"present: {int(model.layer_scale.sum())} layers, {int(model.head_scale.sum())} heads, {int(model.group_scale.sum())} FFN groups")
for prompt in ["The history of the city begins in", "The species is a medium-sized bird found in"]:
    torch.manual_seed(0)
    out = sample(model, torch.tensor([tok.encode(prompt)], device=dev), 120, cfg["seq_len"], 0.8, 50, tok.eos_token_id)
    print(f"--- {prompt!r} ---")
    print(tok.decode(out[0], skip_special_tokens=True))
