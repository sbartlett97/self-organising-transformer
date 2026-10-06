from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
import torch
from model import SelfOrganizingTransformer
from compact_model import build_from_sot
from transformers import AutoTokenizer

import hub_utils

p=argparse.ArgumentParser(); p.add_argument('sot_path'); p.add_argument('--threshold',type=float,default=None,help='Layer/head/ffn/skip gate cutoff for export_architecture(). Defaults to the median layer_gate of this checkpoint (a fixed 0.5 is not reachable under gate_init_logit=-1.0 -- it produces a zero-layer model)'); p.add_argument('--output_dir',default='./outputs/sot_compact')
p.add_argument('--push_to_hub',action='store_true'); p.add_argument('--hub_model_id',default=None,help='Overrides the auto-generated repo id'); p.add_argument('--hub_private',action='store_true'); p.add_argument('--hub_phase',default='phase1'); p.add_argument('--hub_target_cost_tag',default=None)
a=p.parse_args()
model=SelfOrganizingTransformer.from_pretrained(a.sot_path)
threshold=a.threshold
if threshold is None:
    layer_gates,_,_,_=model._gates()
    threshold=statistics.median(layer_gates.detach().cpu().tolist())
    print(f'--threshold not given; using median layer_gate={threshold:.4f} of this checkpoint')
compact=build_from_sot(model,threshold)
if len(compact.config.layers)==0:
    raise RuntimeError(
        f'threshold={threshold:.4f} produced a zero-layer compact model (no layer_gate in this '
        f'checkpoint reached it). Pass a lower --threshold explicitly, or inspect '
        f'{a.sot_path}/architecture.json to pick one from the real layer_gate distribution.'
    )
out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); compact.save_pretrained(out,safe_serialization=True); AutoTokenizer.from_pretrained(a.sot_path).save_pretrained(out)
arch=model.export_architecture(threshold); (out/'source_architecture.json').write_text(json.dumps(arch,indent=2))
print(json.dumps(compact.config.layers,indent=2))

source_train_config_path = Path(a.sot_path) / 'train_config.json'
source_train_config = json.loads(source_train_config_path.read_text()) if source_train_config_path.exists() else {}
if source_train_config:
    (out / 'train_config.json').write_text(json.dumps(source_train_config, indent=2))

if a.push_to_hub:
    from huggingface_hub import HfApi
    username = HfApi().whoami()['name']
    dataset = hub_utils.dataset_tag(source_train_config.get('dataset_name', 'unknown'))
    seed = source_train_config.get('seed', 0)
    repo_id = a.hub_model_id or hub_utils.repo_id_for(username, a.hub_phase, dataset, 'sot_compact', seed, a.hub_target_cost_tag)
    hub_utils.push_checkpoint(out, repo_id, a.hub_private, a.hub_phase, dataset, 'sot_compact', seed)
    hub_utils.ensure_collection(username, a.hub_phase, dataset, repo_id)
