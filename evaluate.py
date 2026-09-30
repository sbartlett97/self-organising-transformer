from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from compact_model import CompactTransformer
from model import SOTConfig, SelfOrganizingTransformer


def load_model(path: str, dtype: torch.dtype | None):
    """Load whichever model class this checkpoint was saved from.

    Neither SOTConfig nor CompactConfig is registered with AutoConfig, so we
    read config.json directly and dispatch on model_type rather than trying
    to guess the class from a single hardcoded from_pretrained call.
    """
    with open(Path(path) / "config.json") as f:
        model_type = json.load(f).get("model_type")
    if model_type == "self_organizing_transformer_compact":
        return CompactTransformer.from_pretrained(path, torch_dtype=dtype)
    if model_type == "self_organizing_transformer":
        return SelfOrganizingTransformer.from_pretrained(path, torch_dtype=dtype)
    raise ValueError(f"Unrecognized model_type={model_type!r} in {path}/config.json")


def device_from(s: str) -> torch.device:
    if s != "auto":
        return torch.device(s)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ppl(path: str, dataset_name: str = "roneneldan/TinyStories", dataset_config: Optional[str] = None, split: str = "validation", seq_len: Optional[int] = None, batch_size: int = 8, max_batches: int = 100, device: str = "auto"):
    dev = device_from(device)
    model = load_model(path, torch.float16 if dev.type == "cuda" else None).to(dev)
    if seq_len is None:
        # Default to the length the checkpoint was actually trained/packed at, not an arbitrary
        # constant: a mismatch either crashes (seq_len > max_position_embeddings) or silently
        # evaluates on truncated/misaligned chunks (seq_len < training length) with no error.
        seq_len = model.config.max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(path)
    raw = load_dataset(dataset_name, dataset_config) if dataset_config else load_dataset(dataset_name)
    if split not in raw:
        raise ValueError(f"No split {split}; available={list(raw.keys())}")
    ids: List[int] = []
    for row in tqdm(raw[split], desc="tokenizing", unit="doc"):
        ids.extend(tokenizer(row["text"], add_special_tokens=False)["input_ids"])
        if tokenizer.eos_token_id is not None:
            ids.append(tokenizer.eos_token_id)
    usable = len(ids) // seq_len * seq_len
    x = torch.tensor(ids[:usable], dtype=torch.long).view(-1, seq_len)
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    n_batches = min(max_batches, math.ceil(len(x) / batch_size))
    with torch.no_grad():
        for i in tqdm(range(0, len(x), batch_size), total=n_batches, desc="ppl", unit="batch"):
            batch = x[i : i + batch_size].to(dev)
            out = model(batch, labels=batch)
            n = batch[:, 1:].numel()
            total_nll += float(out["loss"].item()) * n
            total_tokens += n
            if (i // batch_size) + 1 >= max_batches:
                break
    avg_nll = total_nll / total_tokens
    return {"loss": avg_nll, "perplexity": math.exp(min(avg_nll, 20.0)), "tokens": total_tokens}


def lm_eval(path: str, tasks: str, batch_size: int = 4, device: str = "auto", output_path: str | None = None):
    # Use the public lm-eval Python API with a small adapter around our PreTrainedModel.
    import lm_eval
    from lm_eval.api.model import LM
    from lm_eval.api.instance import Instance

    dev = device_from(device)
    model = load_model(path, torch.float16 if dev.type == "cuda" else None).to(dev)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model.eval()

    class Adapter(LM):
        def __init__(self):
            super().__init__()
            self.model = model
            self.tokenizer = tokenizer
            self.batch_size = batch_size
            self.device = dev

        def tok_encode(self, string, add_special_tokens=False):
            return self.tokenizer(string, add_special_tokens=add_special_tokens)["input_ids"]

        def tok_decode(self, tokens):
            return self.tokenizer.decode(tokens)

        def loglikelihood(self, requests: List[Instance]):
            results = []
            for req in tqdm(requests, desc="loglikelihood", unit="req"):
                context, continuation = req.args
                ctx_ids = self.tok_encode(context, add_special_tokens=False)
                cont_ids = self.tok_encode(continuation, add_special_tokens=False)
                if not cont_ids:
                    results.append((0.0, True))
                    continue
                input_ids = (ctx_ids + cont_ids)[-self.model.config.max_position_embeddings :]
                cont_start = len(input_ids) - len(cont_ids)
                x = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    logits = self.model(x)["logits"][0]
                    log_probs = torch.log_softmax(logits.float(), dim=-1)
                total = 0.0
                for j, tok in enumerate(cont_ids):
                    pos = cont_start + j - 1
                    if pos < 0:
                        continue
                    total += float(log_probs[pos, tok].item())
                greedy = True
                for j, tok in enumerate(cont_ids):
                    pos = cont_start + j - 1
                    if pos >= 0 and int(logits[pos].argmax().item()) != tok:
                        greedy = False
                        break
                results.append((total, greedy))
            return results

        def loglikelihood_rolling(self, requests: List[Instance]):
            out = []
            for req in tqdm(requests, desc="loglikelihood_rolling", unit="req"):
                text = req.args[0]
                ids = self.tok_encode(text, add_special_tokens=False)
                total = 0.0
                if len(ids) < 2:
                    out.append(0.0)
                    continue
                for start in range(0, len(ids), self.model.config.max_position_embeddings - 1):
                    chunk = ids[start : start + self.model.config.max_position_embeddings]
                    if len(chunk) < 2:
                        break
                    x = torch.tensor(chunk, device=self.device, dtype=torch.long).unsqueeze(0)
                    with torch.no_grad():
                        logits = self.model(x)["logits"][0]
                        lp = torch.log_softmax(logits.float(), dim=-1)
                    target = torch.tensor(chunk[1:], device=self.device)
                    total += float(lp[:-1].gather(1, target[:, None]).sum().item())
                out.append(total)
            return out

        def generate_until(self, requests: List[Instance]):
            out = []
            for req in tqdm(requests, desc="generate_until", unit="req"):
                context, gen_kwargs = req.args
                ids = self.tok_encode(context, add_special_tokens=False)
                max_gen = int(gen_kwargs.get("max_gen_toks", 64))
                x = torch.tensor(ids[-self.model.config.max_position_embeddings :], dtype=torch.long, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    y = self.model.generate(x, max_new_tokens=max_gen, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
                text = self.tok_decode(y[0, x.shape[1] :].tolist())
                until = gen_kwargs.get("until", [])
                for stop in until:
                    if stop in text:
                        text = text.split(stop)[0]
                out.append(text)
            return out

        def eot_token_id(self):
            return self.tokenizer.eos_token_id

        def max_length(self):
            return self.model.config.max_position_embeddings

        def max_gen_toks(self):
            return 64

        def batch_size(self):
            return self.batch_size

        def device(self):
            return self.device

    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    results = lm_eval.simple_evaluate(model=Adapter(), tasks=task_list, batch_size=batch_size, limit=None)
    if output_path:
        Path(output_path).write_text(json.dumps(results, indent=2))
    print(json.dumps(results["results"], indent=2))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--dataset_name", default="roneneldan/TinyStories")
    p.add_argument("--dataset_config", default=None)
    p.add_argument("--tasks", default="hellaswag,arc_easy,piqa,winogrande")
    p.add_argument("--skip_lm_eval", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--ppl_batches", type=int, default=100)
    p.add_argument("--seq_len", type=int, default=None, help="Defaults to the checkpoint's own max_position_embeddings")
    p.add_argument("--results", default="evaluation.json")
    a = p.parse_args()
    out = {"perplexity": ppl(a.path, dataset_name=a.dataset_name, dataset_config=a.dataset_config, seq_len=a.seq_len, max_batches=a.ppl_batches, batch_size=a.batch_size, device=a.device)}
    if not a.skip_lm_eval:
        out["lm_eval"] = lm_eval(a.path, a.tasks, batch_size=a.batch_size, device=a.device)
    Path(a.results).write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out["perplexity"], indent=2))


if __name__ == "__main__":
    main()
