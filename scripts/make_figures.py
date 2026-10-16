"""Make the README figures from the local files in results/.

Writes PNGs to docs/figures/. Numbers that are not in a results/ file come
from RESEARCH_LOG.md; each one has a comment that gives the log entry.

Run: python scripts/make_figures.py   (needs requirements-dev.txt)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compute import Replay, Shape, unit_order  # noqa: E402

RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "figures"

# Reference palette (light mode), from the dataviz skill.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]  # blue, orange, aqua, violet
NEUTRAL = "#8a8984"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2,
})

DPI = 150
SIZE = (8, 4.5)

# Phase 2 v8 config (Hub train_config.json / config.json of
# sbartlett97/sot-phase2-wikitext103-heuristic-seed42-tc768d013-v8-tailfreeze).
V8_SHAPE = Shape.from_config(SimpleNamespace(
    d_model=768, max_heads=12, max_ffn_dim=3072, ffn_groups=128, vocab_size=50257,
    max_position_embeddings=256, max_layers=12), seq_len=256)
V8_SEED, V8_SEED_FRACTION, V8_DEATH_RAMP, V8_MAX_STEPS = 42, 0.33, 250, 10000
V8_LAST_ALLOWED_CHECK = 8500  # RESEARCH_LOG 2026-10-14 "Attempt 8": checks 8750-9750 are frozen


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}")


def val_curve(run: str) -> tuple[list[int], list[float]]:
    rows = [r for r in json.loads((RESULTS / run / "metrics.json").read_text()) if "val_ppl" in r]
    return [r["step"] for r in rows], [r["val_ppl"] for r in rows]


def dense768_curve() -> tuple[list[int], list[float]]:
    """Val curve of the Phase 1 dense 768 model, from its Hub repo. Empty if
    the download fails (the chart then draws the final value as a line)."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("sbartlett97/sot-phase1-wikitext103-dense-seed42-tc768d013", "metrics.json")
    except Exception as exc:  # offline, or no access
        print(f"dense 768 curve not available ({type(exc).__name__}); drawing the final value only")
        return [], []
    rows = [r for r in json.loads(Path(path).read_text()) if "val_ppl" in r]
    return [r["step"] for r in rows], [r["val_ppl"] for r in rows]


def fig_phase2_val_ppl() -> None:
    runs = [("phase2_v5_importance", "5: importance death"), ("phase2_v6_allfixes", "6: + ramps, dead-layer skip"),
            ("phase2_v7_lrfreeze", "7: + death freeze in LR tail"), ("phase2_v8_tailfreeze", "8: + birth freeze in LR tail")]
    lo, hi = 20, 45
    fig, ax = plt.subplots(figsize=SIZE)
    for (run, label), color in zip(runs, SERIES):
        x, y = val_curve(run)
        ax.plot(x, [min(v, hi) for v in y], color=color, label=f"Attempt {label}", marker="o", markersize=3)
        ax.annotate(f"{y[-1]:.2f}", (x[-1], min(y[-1], hi)), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)
    # Reference: Phase 1 dense 768, training-time val ppl (the same method as
    # these curves). RESEARCH_LOG 2026-10-08 "Equilibrium + baselines": dense
    # 24.30, sot 24.44 at step 10000. evaluate.py gives 21.98 / 22.00 with a
    # different method, so do not put those numbers on this chart.
    x, y = dense768_curve()
    if x:
        ax.plot(x, [min(v, hi) for v in y], color=NEUTRAL, linestyle="--", linewidth=1.5,
                label="Phase 1 dense 768 (training-time val)")
        ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(6, -6), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)
    else:
        ax.axhline(24.30, color=NEUTRAL, linestyle="--", linewidth=1.5, label="Phase 1 dense 768 (training-time val): 24.30")
    ax.text(200, 21.0, "Phase 1 SOT 768 (gated, training-time val) ends at 24.44", fontsize=8, color=INK_2)
    ax.axvline(V8_LAST_ALLOWED_CHECK, color=GRID, linewidth=1.2, zorder=0)
    ax.text(V8_LAST_ALLOWED_CHECK + 100, hi - 0.5, "LR < 5% of peak\nafter step 8500", ha="left", va="top", fontsize=8, color=INK_2)
    ax.set_ylim(lo, hi)
    ax.set_xlim(0, V8_MAX_STEPS + 900)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Supernet val perplexity")
    ax.set_title("Phase 2, WikiText-103, 768 wide: supernet val ppl")
    fig.text(0.01, -0.04, "Supernet: dormant units still affect the output (RESEARCH_LOG 2026-10-14). "
             f"Attempts 6-8 are identical to step 8500.\n"
             f"Y axis clipped to {lo}-{hi}; early steps and the attempt-5 spike at step 6000 (143) are above it.",
             fontsize=8, color=INK_2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    save(fig, "phase2_val_ppl.png")


def v8_replay() -> Replay:
    events = json.loads((RESULTS / "phase2_v8_tailfreeze" / "events.json").read_text())
    return Replay(V8_SHAPE, V8_SEED, V8_SEED_FRACTION, events, V8_DEATH_RAMP)


def fig_phase2_structure_v8() -> None:
    replay = v8_replay()
    growth = json.loads((RESULTS / "phase2_v8_tailfreeze" / "growth_log.json").read_text())
    steps = [0] + [g["step"] for g in growth]
    kinds = [("layer", "Layers (of 12)"), ("head", "Heads (of 144)"), ("ffn_group", "FFN groups (of 1536)")]
    states = [("alive", "Alive", SERIES[0]), ("dormant", "Dormant", NEUTRAL), ("dead", "Dead", SERIES[1])]
    series = {k: {s: [] for s, _, _ in states} for k, _ in kinds}
    for step in steps:
        st = replay.states(step)
        for k, _ in kinds:
            for s, _, _ in states:
                series[k][s].append(sum(1 for uid, v in st.items() if uid[0] == k and ("dead" if v == "dying" else v) == s))
    # Check: the replay matches the controller's own totals at the last check.
    last = growth[-1]["state_counts"]
    tot = {s: sum(series[k][s][-1] for k, _ in kinds) for s, _, _ in states}
    assert tot["alive"] == last.get("protected", 0) + last.get("mature", 0), (tot, last)
    assert tot["dead"] == last.get("dead", 0) and tot["dormant"] == last.get("dormant", 0), (tot, last)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharex=True)
    for ax, (k, title) in zip(axes, kinds):
        for s, label, color in states:
            ax.plot(steps, series[k][s], color=color, label=label, drawstyle="steps-post")
        ax.axvspan(V8_LAST_ALLOWED_CHECK, V8_MAX_STEPS, color=GRID, alpha=0.6, linewidth=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Training step")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Units")
    axes[0].legend(loc="center right", fontsize=9)
    fig.suptitle("Attempt 8: unit states during training", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.text(0.01, -0.06, "Grey band: no births or deaths after the check at step 8500 (LR below 5% of peak). "
             "Alive = protected + mature. Rebuilt from events.json with compute.Replay.", fontsize=8, color=INK_2)
    fig.tight_layout()
    save(fig, "phase2_structure_v8.png")


def fig_phase2_final_structure_v8() -> None:
    structure = {l["index"]: l for l in v8_replay().structure(V8_MAX_STEPS)}
    # Check against the saved structure_eval result.
    saved = json.loads((RESULTS / "compute" / "v8_structure_eval.json").read_text())["structure"]
    assert [structure[l["index"]] for l in saved] == saved
    layers = list(range(V8_SHAPE.max_layers))
    heads = [structure[i]["heads"] / V8_SHAPE.max_heads if i in structure else 0 for i in layers]
    groups = [structure[i]["ffn_groups"] / V8_SHAPE.ffn_groups if i in structure else 0 for i in layers]
    fig, ax = plt.subplots(figsize=SIZE)
    w = 0.38
    ax.bar([i - w / 2 - 0.01 for i in layers], heads, w, color=SERIES[0], label="Heads present (of 12)")
    ax.bar([i + w / 2 + 0.01 for i in layers], groups, w, color=SERIES[2], label="FFN groups present (of 128)")
    for i in layers:
        if i in structure:
            ax.text(i - w / 2 - 0.01, heads[i] + 0.015, str(structure[i]["heads"]), ha="center", fontsize=8, color=INK_2)
            ax.text(i + w / 2 + 0.01, groups[i] + 0.015, str(structure[i]["ffn_groups"]), ha="center", fontsize=8, color=INK_2)
        else:
            ax.text(i, 0.03, "dead", ha="center", fontsize=9, color=INK_2)
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of the layer present")
    ax.set_ylim(0, 1)
    ax.set_title("Attempt 8: final present structure (step 10000)")
    fig.text(0.01, -0.04, f"9 of 12 layers present; 0.466 of dense FLOPs/token, 0.472 of dense params "
             "(results/compute/v8_structure_eval.json). Bar labels are counts.", fontsize=8, color=INK_2)
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "phase2_final_structure_v8.png")


def fig_phase1_ppl() -> None:
    conditions = [("dense", "Dense"), ("dense_8l", "Dense 8 layers"), ("dense_half_ffn", "Dense half FFN"),
                  ("sot", "SOT (gated)"), ("discovered_retrained", "Discovered, retrained")]
    # TinyStories: corrected numbers, RESEARCH_LOG 2026-10-06 "Transformers-version
    # reload bug" table. The results/*_summary.json TinyStories files hold the
    # original numbers from the broken transformers 5.17.0 install. Do not use them.
    tiny = {"dense": [4.63, 4.61, 4.64], "dense_8l": [4.83, 4.80, 4.81], "dense_half_ffn": [4.84, 4.83, 4.83],
            "sot": [4.75, 4.74, 4.75], "discovered_retrained": [4.89, 4.96, 4.97]}
    # WikiText-103, target_cost 0.17, seeds 42 and 123: these files match the
    # RESEARCH_LOG 2026-10-01 table.
    wiki = {c: [json.loads((RESULTS / f"wikitext103_seed{s}_tc017_{c}_summary.json").read_text())["perplexity"]
                for s in (42, 123)] for c, _ in conditions}
    wiki = {c: [v["perplexity"] if isinstance(v, dict) else v for v in vals] for c, vals in wiki.items()}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, title, seeds in [(axes[0], tiny, "TinyStories, 384 wide (3 seeds)", 3),
                                   (axes[1], wiki, "WikiText-103, 384 wide (2 seeds)", 2)]:
        means = [sum(data[c]) / len(data[c]) for c, _ in conditions]
        colors = [SERIES[0] if c == "sot" else (SERIES[2] if c == "discovered_retrained" else NEUTRAL) for c, _ in conditions]
        ys = range(len(conditions))
        # Dot plot, not bars: the axis does not start at 0, so bar length would mislead.
        for y, (c, _), color in zip(ys, conditions, colors):
            ax.scatter(data[c], [y] * len(data[c]), color=color, s=40, zorder=3, edgecolor=SURFACE, linewidth=1.5)
            ax.plot([means[y]] * 2, [y - 0.25, y + 0.25], color=INK, linewidth=1.5, zorder=4)
            ax.text(max(data[c]) + 0.015 * max(means), y, f"{means[y]:.2f}", va="center", fontsize=9, color=INK_2)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([l for _, l in conditions])
        lo = min(min(v) for v in data.values())
        ax.set_xlim(lo * 0.97, max(max(v) for v in data.values()) * 1.04)
        ax.set_xlabel("Val perplexity (lower is better)")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(len(conditions) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Phase 1: gated SOT against dense controls", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.text(0.01, -0.05, "evaluate.py val ppl (100 batches of 8 x 256). Dots: each seed. Black tick and label: mean over seeds.\n"
             "Inherited-weight compact models (ppl 349-7522) are not shown. "
             "TinyStories numbers: corrected values, RESEARCH_LOG 2026-10-06.", fontsize=8, color=INK_2)
    fig.tight_layout()
    save(fig, "phase1_ppl.png")


def fig_compute() -> None:
    runs = [("v7", "Attempt 7"), ("v8", "Attempt 8")]
    data = [json.loads((RESULTS / "compute" / f"{r}_structure_eval.json").read_text()) for r, _ in runs]
    fig, ax = plt.subplots(figsize=SIZE)
    w = 0.36
    for i, d in enumerate(data):
        a, b = d["trained_supernet"]["ppl"], d["present_structure_only"]["ppl"]
        ax.bar(i - w / 2 - 0.01, a, w, color=SERIES[0], label="Supernet as trained" if i == 0 else None)
        ax.bar(i + w / 2 + 0.01, b, w, color=SERIES[1], label="Present structure only" if i == 0 else None)
        ax.text(i - w / 2 - 0.01, a + 3, f"{a:.1f}", ha="center", fontsize=9, color=INK_2)
        ax.text(i + w / 2 + 0.01, b + 3, f"{b:.1f}", ha="center", fontsize=9, color=INK_2)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([f"{l}\n{d['flops_fraction_of_dense']:.3f} of dense FLOPs/token" for (_, l), d in zip(runs, data)])
    ax.set_ylabel("Val perplexity (48 batches of 8 x 256)")
    ax.set_title("Inactive units still carried the model in attempts 7 and 8")
    fig.text(0.01, -0.04, "Present structure only: every dormant or dead unit forced off (structure_eval.py). "
             "The fix (2026-10-15) gives inactive units exactly zero effect.", fontsize=8, color=INK_2)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="x", visible=False)
    save(fig, "compute_present_vs_supernet.png")


if __name__ == "__main__":
    assert len(unit_order(V8_SHAPE)) == 12 + 144 + 1536
    fig_phase2_val_ppl()
    fig_phase2_structure_v8()
    fig_phase2_final_structure_v8()
    fig_phase1_ppl()
    fig_compute()
