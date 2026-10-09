from __future__ import annotations

from typing import Any, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel


class CompactConfig(PretrainedConfig):
    model_type = "self_organizing_transformer_compact"
    def __init__(self, vocab_size=50257, max_position_embeddings=256, d_model=384, layers=None, dropout=0.0, skip_span=0, **kwargs):
        super().__init__(**kwargs)
        self.tie_word_embeddings=True
        self.vocab_size=vocab_size
        self.max_position_embeddings=max_position_embeddings
        self.d_model=d_model
        self.layers=layers or []
        self.dropout=dropout
        self.skip_span=skip_span

class RMSNorm(nn.Module):
    def __init__(self,d,eps=1e-5):
        super().__init__(); self.weight=nn.Parameter(torch.ones(d)); self.eps=eps
    def forward(self,x):
        y=x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+self.eps)
        return (self.weight*y).to(x.dtype)

class CompactAttn(nn.Module):
    def __init__(self,d_model,num_heads,head_dim=64,dropout=0.0,max_seq=256):
        super().__init__()
        self.d_model=d_model; self.num_heads=num_heads; self.head_dim=head_dim
        self.qkv=nn.Linear(d_model,3*num_heads*head_dim,bias=False)
        self.out=nn.Linear(num_heads*head_dim,d_model,bias=False)
        self.drop=nn.Dropout(dropout)
        self.register_buffer('causal',torch.tril(torch.ones(max_seq,max_seq,dtype=torch.bool)),persistent=False)
    def forward(self,x):
        b,t,c=x.shape
        qkv=self.qkv(x).view(b,t,3,self.num_heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v=qkv.unbind(0)
        a=(q@k.transpose(-2,-1))/math.sqrt(self.head_dim)
        a=a.masked_fill(~self.causal[:t,:t],torch.finfo(a.dtype).min)
        a=F.softmax(a.float(),dim=-1).to(a.dtype)
        y=(a@v).transpose(1,2).contiguous().view(b,t,self.num_heads*self.head_dim)
        return self.drop(self.out(y))

class CompactFFN(nn.Module):
    def __init__(self,d_model,ffn_dim,dropout=0.0):
        super().__init__(); self.up=nn.Linear(d_model,2*ffn_dim,bias=False); self.down=nn.Linear(ffn_dim,d_model,bias=False); self.drop=nn.Dropout(dropout)
    def forward(self,x):
        a,b=self.up(x).chunk(2,-1)
        return self.drop(self.down(F.silu(a)*b))

class CompactBlock(nn.Module):
    def __init__(self,d_model,heads,head_dim,ffn_dim,dropout=0.0,max_seq=256):
        super().__init__(); self.ln1=RMSNorm(d_model); self.attn=CompactAttn(d_model,heads,head_dim,dropout,max_seq); self.ln2=RMSNorm(d_model); self.ffn=CompactFFN(d_model,ffn_dim,dropout)
    def forward(self,x):
        x=x+self.attn(self.ln1(x)); x=x+self.ffn(self.ln2(x)); return x

class CompactTransformer(PreTrainedModel):
    config_class=CompactConfig
    _tied_weights_keys = {"lm_head.weight": "embed.weight"}
    def __init__(self,config):
        super().__init__(config)
        self.embed=nn.Embedding(config.vocab_size,config.d_model)
        self.pos_embed=nn.Embedding(config.max_position_embeddings,config.d_model)
        self.drop=nn.Dropout(config.dropout)
        self.layer_specs=config.layers
        self.blocks=nn.ModuleList([CompactBlock(config.d_model,l['heads'],l.get('head_dim', config.d_model//l['heads']),l['ffn_dim'],config.dropout,config.max_position_embeddings) for l in config.layers])
        self.ln_f=RMSNorm(config.d_model)
        self.lm_head=nn.Linear(config.d_model,config.vocab_size,bias=False)
        self.lm_head.weight=self.embed.weight
        self.post_init()
    def get_input_embeddings(self):
        return self.embed
    def set_input_embeddings(self, value):
        self.embed = value
    def get_output_embeddings(self):
        return self.lm_head
    def set_output_embeddings(self, value):
        self.lm_head = value
    def forward(self,input_ids,labels=None,**kwargs):
        t=input_ids.shape[1]
        pos=torch.arange(t,device=input_ids.device).unsqueeze(0)
        x=self.drop(self.embed(input_ids)+self.pos_embed(pos))
        previous={}
        skip_norm=math.sqrt(self.config.skip_span) if self.config.skip_span else 1.0
        for spec, block in zip(self.layer_specs, self.blocks):
            x=block(x)
            # skip_gate holds each edge's trained SOT gate (SOT scales every skip by its gate). Specs
            # built before skip_gate existed have none, and keep their original unscaled behaviour.
            skip_gates=spec.get('skip_gate', [1.0]*len(spec.get('skip_from', [])))
            for src_idx, g in zip(spec.get('skip_from', []), skip_gates):
                if src_idx in previous:
                    x=x+g*previous[src_idx]/skip_norm
            previous[spec['original_index']]=x
        logits=self.lm_head(self.ln_f(x))
        loss=None
        if labels is not None:
            loss=F.cross_entropy(logits[:,:-1].contiguous().view(-1,logits.size(-1)),labels[:,1:].contiguous().view(-1))
        return {'loss':loss,'logits':logits}


def build_from_sot(sot_model,threshold=0.5,layer_indices=None):
    # layer_indices, if given, replaces the gate-threshold layer choice with an explicit list.
    # Gate value is a poor importance signal for layers (layer 0 has a low gate but is the most
    # important block -- see RESEARCH_LOG.md 2026-10-09), so this allows selection by measured
    # importance. Head/FFN-group selection inside each layer still uses the threshold.
    arch=sot_model.export_architecture(threshold)
    layers=[]
    for item in arch['layers']:
        if (item['index'] in layer_indices) if layer_indices is not None else item['active']:
            h=max(1,len(item['heads']))
            # Enforce a legal divisor of d_model. This matters if a threshold leaves an irregular head count.
            legal=[d for d in range(1,sot_model.config.max_heads+1) if sot_model.config.d_model%d==0]
            h=min(legal,key=lambda d: abs(d-h))
            g=max(1,len(item['ffn_groups']))
            # Every skip edge, not the threshold-filtered item['skip_from']: skip gates train far below
            # the layer threshold (~0.12 vs ~0.22), so filtering silently dropped all of them. The gate
            # value travels with the edge so forward() reproduces SOT's scaled contribution.
            edges=[(item['index']-s,v) for s,v in enumerate(item['skip_gate'],start=1) if s<=item['index']]
            layers.append({'original_index':item['index'],'heads':h,'head_dim':sot_model.config.d_model//sot_model.config.max_heads,'ffn_dim':g*arch['ffn_group_size'],'skip_from':[e[0] for e in edges],'skip_gate':[e[1] for e in edges],'layer_gate':item['layer_gate']})
    # A skip source that was itself pruned inactive can never appear in `previous` at forward time;
    # drop those edges now so the stored spec matches what forward() will actually find.
    active_indices={l['original_index'] for l in layers}
    for l in layers:
        kept=[(src,g) for src,g in zip(l['skip_from'],l['skip_gate']) if src in active_indices]
        l['skip_from']=[src for src,_ in kept]; l['skip_gate']=[g for _,g in kept]
    cfg=CompactConfig(vocab_size=sot_model.config.vocab_size,max_position_embeddings=sot_model.config.max_position_embeddings,d_model=sot_model.config.d_model,layers=layers,dropout=sot_model.config.dropout,skip_span=sot_model.config.skip_span)
    compact=CompactTransformer(cfg)
    with torch.no_grad():
        compact.embed.weight.copy_(sot_model.embed.weight); compact.pos_embed.weight.copy_(sot_model.pos_embed.weight)
        for new_i, spec in enumerate(layers):
            src=sot_model.blocks[spec['original_index']]
            dst=compact.blocks[new_i]
            dst.ln1.weight.copy_(src.ln1.weight); dst.ln2.weight.copy_(src.ln2.weight)
            old_h=sot_model.config.max_heads; hd=src.attn.head_dim; keep_heads=spec['heads']; layer_gate=spec['layer_gate']
            # Use the highest-gated selected heads from the source block.
            gate=torch.sigmoid(src.attn.head_logits)
            chosen=torch.argsort(gate,descending=True)[:keep_heads].sort().values.tolist()
            qkv=src.attn.qkv.weight.view(3,old_h,hd,sot_model.config.d_model)
            dst.attn.qkv.weight.copy_(qkv[:,chosen].reshape(3*keep_heads*hd,sot_model.config.d_model))
            # head_gate and layer_gate scale y before the out-projection during SOT's forward
            # (attn's per-head gate, then the block-level layer_gate on attn's whole output) --
            # fold both into out.weight's input columns so the compact model reproduces that
            # scaled contribution exactly, instead of silently reinstating it to full (gate=1) weight.
            outw=src.attn.out.weight.view(sot_model.config.d_model,old_h,hd)
            chosen_head_gate=gate[chosen]*layer_gate
            dst.attn.out.weight.copy_((outw[:,chosen]*chosen_head_gate.view(1,-1,1)).reshape(sot_model.config.d_model,keep_heads*hd))
            groups=spec['ffn_dim']//arch['ffn_group_size']
            gsig=torch.sigmoid(src.ffn.group_logits)
            chosen_g=torch.argsort(gsig,descending=True)[:groups].sort().values.tolist()
            gs=arch['ffn_group_size']; rows=[]
            for gi in chosen_g: rows.extend(range(gi*gs,(gi+1)*gs))
            up=src.ffn.up.weight.view(2,arch['max_ffn_dim'] if 'max_ffn_dim' in arch else sot_model.config.max_ffn_dim,sot_model.config.d_model)
            dst.ffn.up.weight.copy_(up[:,rows].reshape(2*len(rows),sot_model.config.d_model))
            # same fold for the ffn group_gate * layer_gate scaling on down-projection's input.
            down=src.ffn.down.weight[:,rows]
            per_row_group_gate=gsig[chosen_g].repeat_interleave(gs)*layer_gate
            dst.ffn.down.weight.copy_(down*per_row_group_gate.view(1,-1))
            dst.attn.drop.p=src.attn.resid_dropout.p; dst.ffn.drop.p=src.ffn.dropout.p
        compact.ln_f.weight.copy_(sot_model.ln_f.weight); compact.lm_head.weight=compact.embed.weight
    return compact
