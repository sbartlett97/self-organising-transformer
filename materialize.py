from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from model import SelfOrganizingTransformer
from compact_model import build_from_sot
from transformers import AutoTokenizer

p=argparse.ArgumentParser(); p.add_argument('sot_path'); p.add_argument('--threshold',type=float,default=.5); p.add_argument('--output_dir',default='./outputs/sot_compact'); p.add_argument('--push_to_hub',action='store_true'); p.add_argument('--hub_model_id'); p.add_argument('--private',action='store_true'); a=p.parse_args()
model=SelfOrganizingTransformer.from_pretrained(a.sot_path)
compact=build_from_sot(model,a.threshold)
out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); compact.save_pretrained(out,safe_serialization=True); AutoTokenizer.from_pretrained(a.sot_path).save_pretrained(out)
arch=model.export_architecture(a.threshold); (out/'source_architecture.json').write_text(json.dumps(arch,indent=2))
print(json.dumps(compact.config.layers,indent=2))
if a.push_to_hub:
    from huggingface_hub import HfApi,create_repo
    create_repo(a.hub_model_id,repo_type='model',private=a.private,exist_ok=True)
    HfApi().upload_folder(folder_path=str(out),repo_id=a.hub_model_id,repo_type='model')
