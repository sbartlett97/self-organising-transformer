"""Rebuild the model card (README.md only) of every project repo on the Hub.

Each card is built with hub_utils.build_model_card from the repo's own files
(train_config.json, config.json, architecture.json, metrics.json,
events.json). The Performance section of the earlier card is kept, because
most repos have no local evaluation summary. NOTES and FIXES hold what the
earlier cards did not say: known faults, corrected values and context from
RESEARCH_LOG.md. Weights are never uploaded again.

  python scripts/refresh_hub_cards.py --out /tmp/cards          # write cards locally
  python scripts/refresh_hub_cards.py --out /tmp/cards --push   # and push them
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import HfApi, ModelCard, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from hub_utils import build_model_card

USER = "sbartlett97"
FILES = ["train_config.json", "config.json", "architecture.json", "source_architecture.json", "metrics.json", "events.json"]
NAME = re.compile(r"sot-(phase\d)-(tinystories|wikitext103)-(.+)-seed(\d+)(?:-tc(\S+))?$")

TS_CORRECTED = ("Perplexity corrected on 2026-10-06: the first values came from a `transformers==5.17.0` "
                "install that corrupts the reload of this model. The lm-eval scores were not re-run after "
                "that fix. At this model size they are at chance level in any case.")
GATE_PROXY = ("The gate-weighted cost is a proxy, not compute. This gated model runs every tensor "
              "(1.0 of dense FLOPs). Only a physically smaller model saves compute.")
COMPACT_TS = ("With inherited weights and no further training, this model collapses. Selection by gate "
              "value dropped layer 0, the most important layer (RESEARCH_LOG.md, 2026-10-09). With "
              "layers 0-1 kept and a 300-step fine-tune, most of the loss is recovered.")
COMPACT_WT = "With inherited weights and no further training, this model collapses (ppl in the thousands)."
RETRAINED_WT = "On WikiText-103 the retrained structure loses to the 8-layer and half-FFN dense baselines."
D768 = ("Training-time val ppl (random val batches): dense 24.30, SOT 24.44. Compare Phase 2 runs with "
        "these values, not with the `evaluate.py` values below, because Phase 2 reports training-time val ppl.")
SUPERNET = ("**Every number on this card is a supernet number.** This run predates the zero-effect fix "
            "(2026-10-15): dormant units (gate about 0.04) and dead units (gate about 0.001) were still in "
            "the forward pass and still trained. The checkpoint loads with every scale at 1, so it runs as trained.")

PHASE2 = {
    "v4": ("attempt 4", "the 2026-10-08 tuning entry and the 2026-10-09 ablation entry", None,
           ["Death by gate value: no unit died. The worst unit of each kind sat at 84-93% of its kind's median gate.",
            "Used for the gate-against-importance ablation (RESEARCH_LOG.md, 2026-10-09)."]),
    "v5-imp": ("attempt 5", "the 2026-10-13 attempt-5 entry", None,
               ["First run with death by importance: 76 deaths, among them 5 layers.",
                "Layer 4 died at the final check, just before the final eval. Val ppl was 29.95 at step 9500 and 35.71 at step 10000."]),
    "v6-allfixes": ("attempt 6", "the 2026-10-14 attempt-6 entry", None,
                    ["Birth and death ramps and the dead-layer rule removed every val ppl spike.",
                     "Deaths at LR below 5e-6 (steps 9250-9750) raised val ppl from 27.29 (step 9000) to 29.28."]),
    "v7-lrfreeze": ("attempt 7", "the 2026-10-14 attempt-7 and compute-accounting entries", (30.25, 244.29),
                    ["Deaths frozen below 5% of peak LR. Heads born in the LR tail raised val ppl to 30.66."]),
    "v8-tailfreeze": ("attempt 8", "the 2026-10-14 attempt-8 and compute-accounting entries", (26.67, 132.42),
                      ["Births and deaths frozen below 5% of peak LR: val ppl falls at every eval to 26.98.",
                       "The present structure alone writes degenerate text ([samples](https://github.com/sbartlett97/self-organising-transformer/blob/main/docs/samples.md))."]),
}
# Superseded repos, deleted from the Hub on 2026-10-16. Their results are in RESEARCH_LOG.md.
SUPERSEDED = {
    "sot-phase1-tinystories-sot_compact-seed42",  # before the gate-magnitude fix; see *_gatefix
    "sot-phase1-tinystories-sot_compact-seed123",
    "sot-phase1-tinystories-sot_compact-seed7",
    "sot-phase2-tinystories-heuristic-seed42",  # supernet result, 0 deaths
    "sot-phase2-wikitext103-heuristic-seed42-tc768d013",  # attempt 1: plateau-timing bug
    "sot-phase2-wikitext103-heuristic-seed42-tc768d013-v2",  # attempt 2: NaN bug
    "sot-phase2-wikitext103-heuristic-seed42-tc768d013-v3",  # attempt 3: no limit on growth
}
TS_DENSE = {"42": 4.63, "123": 4.61, "7": 4.64}
GATEFIX_PPL = {"42": 881.24, "123": 525.03, "7": 349.24}


def parse(repo: str):
    m = NAME.match(repo.split("/")[1])
    if not m:
        return None
    phase, dataset, condition, seed, tc = m.groups()
    return phase, dataset, condition, seed, tc


def fetch(repo: str, d: Path) -> None:
    for f in FILES:
        try:
            shutil.copy(hf_hub_download(repo, f), d / f)
        except EntryNotFoundError:
            pass


def old_performance(repo: str) -> str | None:
    content = ModelCard.load(repo).content
    if "## Performance" not in content:
        return None
    return content.split("## Performance", 1)[1].strip()


def plan(repo: str):
    """(phase, dataset, condition, seed, notes, log_entry, performance_md)"""
    phase, dataset, condition, seed, tc = parse(repo)
    notes, perf = [], old_performance(repo)
    if phase == "phase2":
        suffix = tc.split("-", 1)[1]
        label, entry, present_only, extra = PHASE2[suffix]
        notes = [f"Phase 2 {label}.", SUPERNET] + extra
        perf = ("_No `evaluate.py` run._ The final training-time val ppl of the supernet is in "
                "\"Training trajectory\" below. Phase 1 dense 768 on the same method: 24.30.")
        if present_only:
            perf += ("\n\nVal ppl on 48 val batches of 8 x 256 (`structure_eval.py`):\n\n"
                     "| model | val ppl |\n|---|---:|\n"
                     f"| supernet as trained | {present_only[0]:.2f} |\n"
                     f"| present structure only (every other unit forced off) | {present_only[1]:.2f} |")
        return phase, dataset, condition, int(seed), notes, entry, perf
    if dataset == "tinystories":
        entry = "the 2026-09-29 entries (runs) and the 2026-10-06 entry (corrected perplexity)"
        notes.append(TS_CORRECTED)
        if condition == "sot_compact_gatefix":
            notes.append(COMPACT_TS)
            ppl = GATEFIX_PPL[seed]
            perf = re.sub(r"\*\*Perplexity\*\*: `[\d.]+`", f"**Perplexity**: `{ppl:.2f}` (corrected 2026-10-06)", perf)
            perf = re.sub(r"- loses to the matched dense baseline \(`[\d.]+`\) by `[\d.]+%`",
                          f"- loses to the matched dense baseline (`{TS_DENSE[seed]:.2f}`) by "
                          f"`{(ppl - TS_DENSE[seed]) / TS_DENSE[seed] * 100:.1f}%`", perf)
    else:
        entry = {"014": "the 2026-09-30 WikiText-103 entry", "017": "the 2026-10-01 entry",
                 "768d013": "the 2026-10-08 width scale-up entry"}[tc]
        if tc == "768d013":
            notes.append(D768)
        if condition == "sot_compact":
            notes.append(COMPACT_WT)
        if condition == "discovered_retrained":
            notes.append(RETRAINED_WT)
    if condition == "sot":
        notes.append(GATE_PROXY)
    if perf:
        perf = "`evaluate.py` perplexity: 100 val batches of 8 x 256 tokens.\n\n" + perf
    return phase, dataset, condition, int(seed), notes, entry, perf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Directory for the rendered cards")
    p.add_argument("--push", action="store_true", help="Push each card (README.md only)")
    p.add_argument("--only", nargs="*", help="Repo names to process (default: all)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    repos = sorted(m.id for m in HfApi().list_models(author=USER)
                   if parse(m.id) and m.id.split("/")[1] not in SUPERSEDED)
    if a.only:
        repos = [r for r in repos if r.split("/")[1] in a.only]
    for repo in repos:
        phase, dataset, condition, seed, notes, entry, perf = plan(repo)
        with tempfile.TemporaryDirectory() as d:
            fetch(repo, Path(d))
            card = build_model_card(Path(d), repo, phase, dataset, condition, seed,
                                    research_log_entry=entry, notes=notes, performance_md=perf)
        (out / f"{repo.split('/')[1]}.md").write_text(card.content)
        if a.push:
            card.push_to_hub(repo, commit_message="Update model card")
        print(("pushed " if a.push else "wrote ") + repo)


if __name__ == "__main__":
    main()
