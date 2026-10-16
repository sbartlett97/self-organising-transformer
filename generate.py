"""Sample text from any checkpoint of this project (supernet, plastic or compact).

The model classes are not registered with AutoModel, so this loads them with
evaluate.load_model. No KV cache: each step runs the full context again,
which is fast enough at these sizes. The context is cut to the training
seq_len (positions after it were never trained), or to
max_position_embeddings if the checkpoint has no train_config.json.

Example:
  python generate.py sbartlett97/sot-phase1-tinystories-dense-seed42 \
      --prompt "Once upon a time, there was a little dog named" --max_new_tokens 120
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoTokenizer

from ablation_importance import resolve_checkpoint
from evaluate import device_from, load_model


@torch.no_grad()
def sample(model, ids: torch.Tensor, max_new_tokens: int, context: int, temperature: float, top_k: int,
           eos_id: int | None) -> torch.Tensor:
    for _ in range(max_new_tokens):
        logits = model(ids[:, -context:])["logits"][:, -1, :].float()
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(torch.softmax(logits, -1), 1)
        ids = torch.cat([ids, nxt], dim=1)
        if eos_id is not None and int(nxt) == eos_id:
            break
    return ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="Local path or Hub repo id")
    p.add_argument("--prompt", default="Once upon a time,")
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    p.add_argument("--top_k", type=int, default=50, help="0 = no top-k filter")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    ckpt = resolve_checkpoint(a.checkpoint)
    dev = device_from(a.device)
    model = load_model(str(ckpt), None).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(ckpt)
    context = model.config.max_position_embeddings
    if (ckpt / "train_config.json").exists():
        context = min(context, json.loads((ckpt / "train_config.json").read_text()).get("seq_len", context))

    prompt_ids = torch.tensor([tok.encode(a.prompt)], device=dev)
    for i in range(a.num_samples):
        torch.manual_seed(a.seed + i)
        out = sample(model, prompt_ids, a.max_new_tokens, context, a.temperature, a.top_k, tok.eos_token_id)
        print(f"--- sample {i} ---")
        print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
