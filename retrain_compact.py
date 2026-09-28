from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

from compact_model import CompactTransformer
from train import TrainConfig, BatchIterator, evaluate_loss, load_tokens, pick_device

p=argparse.ArgumentParser()
p.add_argument('architecture_model')
p.add_argument('--output_dir',default='./outputs/discovered_retrained')
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
a=p.parse_args()

set_seed(a.seed)
tok=AutoTokenizer.from_pretrained(a.architecture_model)
if tok.pad_token is None: tok.pad_token=tok.eos_token
base=CompactTransformer.from_pretrained(a.architecture_model)
# Fresh weights, identical architecture.
model=CompactTransformer(base.config)
device=pick_device(a.device); model.to(device).train()
train_data,val_data=load_tokens(TrainConfig(seq_len=base.config.max_position_embeddings,max_train_docs=a.max_train_docs),tok)
opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay,betas=(.9,.95))
sched=get_cosine_schedule_with_warmup(opt,a.warmup_steps,a.max_steps)
scaler=torch.cuda.amp.GradScaler(enabled=device.type=='cuda')
iterator=iter(BatchIterator(train_data,a.batch_size,a.seed))
out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
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
    if step%100==0: pbar.write(f'step={step} train_loss={train_loss:.4f}')
    if step%a.eval_every==0 or step==a.max_steps:
        v=evaluate_loss(model,val_data,a.batch_size,device)
        pbar.write(f'step={step} val_loss={v:.4f} val_ppl={torch.exp(torch.tensor(min(v,20))).item():.3f}')
model.save_pretrained(out,safe_serialization=True); tok.save_pretrained(out)
