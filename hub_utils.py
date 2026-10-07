from __future__ import annotations

import json
import statistics
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

from huggingface_hub import HfApi, ModelCard, ModelCardData, create_repo

GITHUB_URL = "https://github.com/sbartlett97/self-organising-transformer"


_DATASET_TAGS = {
    "roneneldan/tinystories": "tinystories",
    "wikitext": "wikitext103",  # only wikitext-103-raw-v1 is used in this project
}


def dataset_tag(dataset_name: str) -> str:
    """Short slug for a dataset_name, for use in repo ids. Falls back to a
    lowercased, slash/space-stripped version of the raw name for anything
    not in the explicit map above."""
    key = dataset_name.lower()
    if key in _DATASET_TAGS:
        return _DATASET_TAGS[key]
    return key.replace("/", "-").replace(" ", "-")


def repo_id_for(
    username: str,
    phase: str,
    dataset: str,
    condition: str,
    seed: int,
    target_cost_tag: Optional[str] = None,
) -> str:
    """e.g. sbartlett97/sot-phase1-tinystories-sot-seed42, or with a
    target_cost_tag: sbartlett97/sot-phase1-wikitext103-sot-seed42-tc017."""
    parts = [f"sot-{phase}", dataset, condition, f"seed{seed}"]
    name = "-".join(parts)
    if target_cost_tag:
        name = f"{name}-tc{target_cost_tag}"
    return f"{username}/{name}"


def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _hyperparameters_section(train_config: dict) -> str:
    rows = [
        ("model_type", train_config.get("model_type")),
        ("dataset", train_config.get("dataset_name")),
        ("dataset_config", train_config.get("dataset_config")),
        ("seed", train_config.get("seed")),
        ("max_steps", train_config.get("max_steps")),
        ("seq_len", train_config.get("seq_len")),
        ("batch_size", train_config.get("batch_size")),
        ("grad_accum", train_config.get("grad_accum")),
        ("max_layers", train_config.get("max_layers")),
        ("d_model", train_config.get("d_model")),
        ("max_heads", train_config.get("max_heads")),
        ("max_ffn_dim", train_config.get("max_ffn_dim")),
        ("ffn_groups", train_config.get("ffn_groups")),
        ("target_cost", train_config.get("target_cost")),
    ]
    # Phase 2 fields, when present.
    for key in ("birth_mode", "seed_fraction", "growth_interval", "birth_limit", "death_limit", "minimum_age"):
        if key in train_config:
            rows.append((key, train_config[key]))
    lines = ["| Hyperparameter | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows if v is not None]
    return "\n".join(lines)


def _architecture_section(arch: dict) -> str:
    layers = arch.get("layers", [])
    # architecture.json is saved with whatever threshold export_architecture()
    # happened to be called with -- train.py's periodic/final saves always use
    # its default (0.5), which this project's SOT gates never reach (typically
    # ~0.2-0.34), so the file's own "active"/"heads"/"ffn_groups" fields can be
    # stale/misleading (everything reads as inactive). Recompute using the
    # same auto-median convention materialize.py uses, from the raw per-layer
    # gate values the file always carries regardless of that threshold.
    layer_gates = [l["layer_gate"] for l in layers if isinstance(l.get("layer_gate"), (int, float))]
    threshold = statistics.median(layer_gates) if layer_gates else arch.get("threshold", 0.5)

    lines = ["| Layer | Active | Heads | FFN groups | Layer gate |", "|---:|:---:|---:|---:|---:|"]
    for layer in layers:
        gate = layer.get("layer_gate")
        active = isinstance(gate, (int, float)) and gate >= threshold
        head_gate = layer.get("head_gate") or []
        ffn_gate = layer.get("ffn_gate") or []
        heads = sum(1 for v in head_gate if v >= threshold) if head_gate else len(layer.get("heads", []))
        ffn = sum(1 for v in ffn_gate if v >= threshold) if ffn_gate else len(layer.get("ffn_groups", []))
        gate_s = f"{gate:.4f}" if isinstance(gate, (int, float)) else "—"
        lines.append(f"| {layer.get('index')} | {'✅' if active else '—'} | {heads} | {ffn} | {gate_s} |")
    header = f"Threshold (auto-median of this checkpoint's layer gates): `{threshold:.4f}`\n\n"
    return header + "\n".join(lines)


def _trajectory_section(metrics: list) -> str:
    last_train = next((r for r in reversed(metrics) if "train_loss" in r), None)
    last_val = next((r for r in reversed(metrics) if "val_loss" in r), None)
    lines = []
    if last_train:
        lines.append(f"- Final train loss: `{last_train['train_loss']:.4f}` at step {last_train['step']}")
        if "cost" in last_train:
            lines.append(f"- Final architectural cost: `{last_train['cost']:.4f}`")
    if last_val:
        lines.append(f"- Final val loss: `{last_val['val_loss']:.4f}` (ppl `{last_val['val_ppl']:.2f}`) at step {last_val['step']}")
    return "\n".join(lines) if lines else "_No training metrics recorded._"


def _events_section(events: list) -> str:
    births = sum(1 for e in events if e.get("event") == "birth")
    deaths = sum(1 for e in events if e.get("event") == "death")
    return f"- Total births: `{births}`\n- Total deaths: `{deaths}`\n- Total structural events: `{len(events)}`"


def _performance_section(summary: dict, dense_ppl: Optional[float] = None) -> str:
    ppl = summary.get("perplexity", {})
    ppl_val = ppl.get("perplexity") if isinstance(ppl, dict) else ppl
    lines = []
    if ppl_val is not None:
        lines.append(f"**Perplexity**: `{ppl_val:.2f}`")
        if dense_ppl is not None and dense_ppl > 0:
            delta = (ppl_val - dense_ppl) / dense_ppl * 100
            verdict = "beats" if delta < 0 else "loses to"
            lines.append(f"- {verdict} the matched dense baseline (`{dense_ppl:.2f}`) by `{abs(delta):.1f}%`")
    lm_eval = summary.get("lm_eval_results", {})
    if lm_eval:
        lines.append("\n| Task | Metric | Value | Stderr |\n|---|---|---:|---:|")
        for task, metrics in lm_eval.items():
            if not isinstance(metrics, dict):
                continue
            for key, value in metrics.items():
                if "_stderr" in key or not key.endswith(",none") or not isinstance(value, (int, float)):
                    continue
                metric_name = key.split(",")[0]
                stderr = metrics.get(f"{metric_name}_stderr,none")
                stderr_s = f"{stderr:.4f}" if isinstance(stderr, (int, float)) else "—"
                lines.append(f"| {task} | {metric_name} | {value:.4f} | {stderr_s} |")
    return "\n".join(lines) if lines else "_No evaluation results yet._"


def build_model_card(
    checkpoint_dir: Path,
    repo_id: str,
    phase: str,
    dataset: str,
    condition: str,
    seed: int,
    evaluation_summary: Optional[dict] = None,
    dense_ppl: Optional[float] = None,
    research_log_entry: Optional[str] = None,
) -> ModelCard:
    checkpoint_dir = Path(checkpoint_dir)
    train_config = _read_json(checkpoint_dir / "train_config.json") or {}
    arch = _read_json(checkpoint_dir / "architecture.json") or _read_json(checkpoint_dir / "source_architecture.json")
    metrics = _read_json(checkpoint_dir / "metrics.json") or []
    events = _read_json(checkpoint_dir / "events.json")

    card_data = ModelCardData(
        library_name="transformers",
        pipeline_tag="text-generation",
        tags=["self-organizing-transformer", "research", phase, dataset, condition],
    )

    sections = [
        f"# {repo_id.split('/')[-1]}",
        "",
        f"Part of the [Self-Organising Transformer]({GITHUB_URL}) research project — "
        f"**{phase}**, dataset **{dataset}**, condition **`{condition}`**, seed **{seed}**.",
    ]
    if research_log_entry:
        sections.append(f"Full context: `RESEARCH_LOG.md`, \"{research_log_entry}\".")
    sections += ["", "## Hyperparameters", "", _hyperparameters_section(train_config)]
    if arch:
        sections += ["", "## Discovered architecture", "", _architecture_section(arch)]
    sections += ["", "## Training trajectory", "", _trajectory_section(metrics)]
    if events:
        sections += ["", "## Structural plasticity events", "", _events_section(events)]
    sections += ["", "## Performance", ""]
    sections.append(_performance_section(evaluation_summary, dense_ppl) if evaluation_summary else "_Not yet evaluated._")

    content = f"---\n{card_data.to_yaml()}\n---\n\n" + "\n".join(sections)
    return ModelCard(content)


def push_checkpoint(
    checkpoint_dir: Path,
    repo_id: str,
    private: bool,
    phase: str,
    dataset: str,
    condition: str,
    seed: int,
    evaluation_summary: Optional[dict] = None,
    dense_ppl: Optional[float] = None,
    research_log_entry: Optional[str] = None,
) -> str:
    api = HfApi()
    create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(checkpoint_dir), repo_id=repo_id, repo_type="model")
    card = build_model_card(
        checkpoint_dir, repo_id, phase, dataset, condition, seed,
        evaluation_summary=evaluation_summary, dense_ppl=dense_ppl, research_log_entry=research_log_entry,
    )
    card.push_to_hub(repo_id)
    url = f"https://huggingface.co/{repo_id}"
    print(f"Pushed {checkpoint_dir} -> {url}")
    return url


# Hand-maintained, not derived from RESEARCH_LOG.md -- update the text here
# whenever a phase/dataset's state or headline finding changes materially.
# ensure_collection() re-applies the current text on every call, so a
# collection's description tracks whichever entry here is live, with no
# separate "did I remember to update the Hub" step. Hub API hard-caps
# collection descriptions at 150 chars -- full intent/state/findings live
# in each model card and RESEARCH_LOG.md; this is just a pointer+headline.
_COLLECTION_DESCRIPTIONS = {
    ("phase1", "tinystories"): (
        "Phase 1 gating, 3 seeds x 6 conditions. Corrected 2026-10-06: SOT "
        "trails dense ~2.6% ppl at ~13% cost (bad transformers ver). "
        "See RESEARCH_LOG.md."
    ),
    ("phase1", "wikitext103"): (
        "Phase 1 generalisation test. SOT doesn't beat/match dense here; "
        "a capacity limit at this model scale, stable across seeds. "
        "See RESEARCH_LOG.md."
    ),
    ("phase2", "tinystories"): (
        "Phase 2: discrete birth/death plasticity. After 2 bugfixes: beats "
        "Phase 1 dense by 4.9% ppl (1 seed, 0 deaths -- caveated). "
        "See RESEARCH_LOG.md."
    ),
}


def ensure_collection(username: str, phase: str, dataset: str, repo_id: str) -> str:
    api = HfApi()
    title = f"SOT {phase} — {dataset}"
    collection = api.create_collection(title=title, namespace=username, exists_ok=True, private=False)
    api.add_collection_item(collection.slug, item_id=repo_id, item_type="model", exists_ok=True)
    description = _COLLECTION_DESCRIPTIONS.get((phase, dataset))
    if description:
        api.update_collection_metadata(collection.slug, description=description)
    return collection.slug


# Shared live-metrics tracking (Trackio: https://huggingface.co/blog/trackio).
# A module-level flag rather than a passed-around handle -- each training
# script runs one training loop per process, so there's never more than one
# active run to track, and this keeps call sites (trackio_log/trackio_finish)
# trivial no-ops when tracking is off or unavailable.
_trackio_active = False


def init_trackio(cfg, phase: str, condition: str) -> bool:
    """Best-effort trackio init for a live, optionally remote (HF Space)
    dashboard. Returns True if logging is active. Every failure mode here
    (not installed, no HF auth available, Space creation failing) degrades
    to "skip tracking, keep training" rather than raising -- a dashboard
    being unavailable should never be the reason a real run fails, same
    spirit as Trackio's own internal defensiveness."""
    global _trackio_active
    _trackio_active = False
    if not getattr(cfg, "use_trackio", True):
        return False
    try:
        import trackio
    except ImportError:
        print("trackio not installed -- skipping live tracking (pip install trackio to enable)")
        return False

    dataset = dataset_tag(getattr(cfg, "dataset_name", "unknown"))
    seed = getattr(cfg, "seed", 0)
    name_parts = [dataset, condition, f"seed{seed}"]
    tc_tag = getattr(cfg, "hub_target_cost_tag", None)
    if tc_tag:
        name_parts.append(f"tc{tc_tag}")
    run_name = "-".join(name_parts)

    space_id = getattr(cfg, "trackio_space_id", None)
    if space_id is None:
        try:
            username = HfApi().whoami()["name"]
            space_id = f"{username}/sot-tracking"
        except Exception as exc:
            print(f"trackio: could not resolve HF username for remote sync ({exc}) -- logging locally only")
            space_id = None

    config = asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))
    try:
        trackio.init(project=f"sot-{phase}", name=run_name, group=dataset, config=config, space_id=space_id)
    except Exception as exc:
        print(f"trackio.init failed ({exc}) -- continuing without live tracking")
        return False
    _trackio_active = True
    return True


def trackio_log(record: dict) -> None:
    """Log one record if tracking is active. Silently drops non-scalar
    fields (e.g. gate_summary()'s nested gates_by_layer list) -- Trackio
    logs scalar metrics, and the full record already goes to the local
    metrics.json regardless of this. `step`, if present, is passed as
    Trackio's own step index rather than left in the metrics dict -- else
    Trackio treats it as a reserved key and renames it to `__step`,
    breaking the dashboard's x-axis."""
    if not _trackio_active:
        return
    import trackio

    safe = {k: v for k, v in record.items() if isinstance(v, (int, float, str, bool))}
    step = safe.pop("step", None)
    try:
        trackio.log(safe, step=step)
    except Exception as exc:
        print(f"trackio.log failed ({exc}) -- continuing without live tracking for this step")


def trackio_finish() -> None:
    if not _trackio_active:
        return
    import trackio

    try:
        trackio.finish()
    except Exception as exc:
        print(f"trackio.finish failed ({exc})")


__all__ = [
    "repo_id_for", "build_model_card", "push_checkpoint", "ensure_collection",
    "init_trackio", "trackio_log", "trackio_finish",
]
