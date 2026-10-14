import json, math, sys, random
sys.path.insert(0, ".")
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from ablation_importance import resolve_checkpoint, get_logit, set_logit, batch_loss
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
ev = json.load(open("results/phase2_v7_lrfreeze/events.json"))
births = [(e["step"], tuple(e["unit"])) for e in ev if e["event"] == "birth"]
tail = [u for s, u in births if s >= 9000]
early_heads = [u for s, u in births if u[0] == "head" and s <= 7500]
random.Random(0).shuffle(early_heads)
base = batch_loss(model, batches)
print(f"baseline loss {base:.4f} ppl {math.exp(base):.2f}")
def ablate(units, label):
    old = {u: get_logit(model, u) for u in units}
    for u in units: set_logit(model, u, -4.0)
    l = batch_loss(model, batches)
    for u, v in old.items(): set_logit(model, u, v)
    print(f"{label:<34} dloss {l-base:+.4f} ppl {math.exp(l):.2f}  gates {[round(1/(1+math.exp(-v)),3) for v in old.values()]}")
for s, u in births:
    if s >= 9000: ablate([u], f"tail birth {u} @ {s}")
ablate(tail, "all tail births (>=9000)")
for u in early_heads[:4]: ablate([u], f"control early head {u}")
ablate(early_heads[:len(tail)], f"control {len(tail)} early heads")
