from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

import hub_utils
from compact_model import CompactTransformer
from train import TrainConfig, BatchIterator, evaluate_loss, load_tokens, pick_device

p=argparse.ArgumentParser()
p.add_argument('architecture_model')
p.add_argument('--output_dir',default='./outputs/discovered_retrained')
p.add_argument('--dataset_name',default='roneneldan/TinyStories')
p.add_argument('--dataset_config',default=None)
p.add_argument('--seq_len',type=int,default=None)
p.add_argument('--max_train_docs',type=int,default=None,help='Cap on train-split documents to tokenize, for fast diagnostic runs')
p.add_argument('--max_steps',type=int,default=10000)
p.add_argument('--batch_size',type=int,default=16)
p.add_argument('--grad_accum',type=int,default=8)
p.add_argument('--learning_rate',type=float,default=3e-4)
p.add_argument('--weight_decay',type=float,default=.1)
p.add_argument('--warmup_steps',type=int,default=500)
p.add_argument('--eval_every',type=int,default=500)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--device',default='auto')
p.add_argument('--push_to_hub',action='store_true')
p.add_argument('--hub_model_id',default=None,help='Overrides the auto-generated repo id')
p.add_argument('--hub_private',action='store_true')
p.add_argument('--hub_phase',default='phase1')
p.add_argument('--hub_target_cost_tag',default=None)
p.add_argument('--no_trackio',dest='use_trackio',action='store_false',help='Disable live Trackio metrics (on by default)')
p.add_argument('--trackio_space_id',default=None,help='HF Space to sync live metrics to, e.g. username/sot-tracking -- defaults to <your HF username>/sot-tracking')
a=p.parse_args()

set_seed(a.seed)
tok=AutoTokenizer.from_pretrained(a.architecture_model)
if tok.pad_token is None: tok.pad_token=tok.eos_token
base=CompactTransformer.from_pretrained(a.architecture_model)
# Fresh weights, identical architecture.
model=CompactTransformer(base.config)
device=pick_device(a.device); model.to(device).train()
hub_utils.init_trackio(a, phase=a.hub_phase, condition='discovered_retrained')
train_data,val_data=load_tokens(TrainConfig(seq_len=base.config.max_position_embeddings,dataset_name=a.dataset_name,dataset_config=a.dataset_config,max_train_docs=a.max_train_docs),tok)
opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay,betas=(.9,.95))
sched=get_cosine_schedule_with_warmup(opt,a.warmup_steps,a.max_steps)
scaler=torch.cuda.amp.GradScaler(enabled=device.type=='cuda')
iterator=iter(BatchIterator(train_data,a.batch_size,a.seed))
out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
history=[]
pbar=tqdm(range(1,a.max_steps+1),desc='retrain',unit='step')
for step in pbar:
    opt.zero_grad(set_to_none=True)
    for _ in range(a.grad_accum):
        try: batch=next(iterator)
        except StopIteration:
            iterator=iter(BatchIterator(train_data,a.batch_size,a.seed+step)); batch=next(iterator)
        batch=batch.to(device)
        with torch.autocast(device_type=device.type,dtype=torch.float16 if device.type=='cuda' else torch.bfloat16,enabled=device.type in {'cuda','cpu'}):
            loss=model(batch,labels=batch)['loss']/a.grad_accum
        scaler.scale(loss).backward()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update(); sched.step()
    train_loss=loss.item()*a.grad_accum
    pbar.set_postfix(loss=f'{train_loss:.4f}')
    if step%100==0:
        pbar.write(f'step={step} train_loss={train_loss:.4f}')
        record={'step':step,'train_loss':train_loss}
        history.append(record)
        hub_utils.trackio_log(record)
    if step%a.eval_every==0 or step==a.max_steps:
        v=evaluate_loss(model,val_data,a.batch_size,device)
        val_ppl=torch.exp(torch.tensor(min(v,20))).item()
        pbar.write(f'step={step} val_loss={v:.4f} val_ppl={val_ppl:.3f}')
        record={'step':step,'val_loss':v,'val_ppl':val_ppl}
        history.append(record)
        hub_utils.trackio_log(record)
model.save_pretrained(out,safe_serialization=True); tok.save_pretrained(out)
(out/'train_config.json').write_text(json.dumps(vars(a),indent=2))
(out/'metrics.json').write_text(json.dumps(history,indent=2))

if a.push_to_hub:
    from huggingface_hub import HfApi
    username = HfApi().whoami()['name']
    dataset = hub_utils.dataset_tag(a.dataset_name)
    repo_id = a.hub_model_id or hub_utils.repo_id_for(username, a.hub_phase, dataset, 'discovered_retrained', a.seed, a.hub_target_cost_tag)
    hub_utils.push_checkpoint(out, repo_id, a.hub_private, a.hub_phase, dataset, 'discovered_retrained', a.seed)
    hub_utils.ensure_collection(username, a.hub_phase, dataset, repo_id)
hub_utils.trackio_finish()
