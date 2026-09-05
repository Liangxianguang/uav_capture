"""Aggregate the three-seed V21 settled counterfactual ranking audits.

The input is offline-only settled data.  This command never advances the
simulator and never changes a source trace.  It keeps scene manifests
separate by training seed, while requiring a common protocol and environment
contract.  The output is a diagnostic ranking report, not a closed-loop
performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260911, 20260912, 20260913)
VARIANTS = ("m3", "a1", "a2")
MINIMUM_CREDIT = 0.65
HIGH_CREDIT = 0.80
MINIMUM_SEPARATION_M = 0.002
RANKING_MAX_AGGREGATE_SELECTED_NOT_BEST = 0.25
RANKING_MAX_SEED_SELECTED_NOT_BEST = 0.40


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def _sample_std(values: Sequence[float]) -> float | None:
    return float(np.std(np.asarray(values, dtype=float), ddof=1)) if len(values) > 1 else 0.0 if values else None


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Row {line_number} is not an object: {path}")
        rows.append(value)
    if not rows:
        raise ValueError(f"No decision rows: {path}")
    return rows


def _validate_run_entries(report: Mapping[str, Any], seed: int) -> tuple[str, str, str]:
    inputs = report.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"Missing inputs in settled report for seed {seed}")
    protocol = str(inputs.get("protocol_sha256", ""))
    environment = str(inputs.get("environment_config_sha256", ""))
    scene = str(inputs.get("scene_manifest_sha256", ""))
    if not protocol or not environment or not scene:
        raise ValueError(f"Missing protocol/environment/scene hash for seed {seed}")
    seeds = inputs.get("training_seeds")
    if seeds != [seed]:
        raise ValueError(f"Training seed provenance mismatch for seed {seed}: {seeds!r}")
    entries = report.get("runs")
    if not isinstance(entries, list):
        raise ValueError(f"Missing runs in settled report for seed {seed}")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid run entry for seed {seed}")
        variant = str(entry.get("variant", ""))
        if variant not in VARIANTS:
            continue
        if variant in seen:
            raise ValueError(f"Duplicate run variant {variant} for seed {seed}")
        seen.add(variant)
        if int(entry.get("training_seed", -1)) != seed:
            raise ValueError(f"Run seed mismatch for {variant}, seed {seed}")
        if str(entry.get("protocol_sha256", "")) != protocol:
            raise ValueError(f"Run protocol mismatch for {variant}, seed {seed}")
        if str(entry.get("scene_manifest_sha256", "")) != scene:
            raise ValueError(f"Run scene manifest mismatch for {variant}, seed {seed}")
    if seen != set(VARIANTS):
        raise ValueError(f"Incomplete variant set for seed {seed}: {sorted(seen)}")
    return protocol, environment, scene


def load_settled_report(path: Path, seed: int) -> dict[str, Any]:
    path = path.resolve()
    report_path = path / "settled_counterfactual.json"
    rows_path = path / "decision_rows.jsonl"
    if not report_path.is_file() or not rows_path.is_file():
        raise FileNotFoundError(f"Settled report requires JSON and JSONL: {path}")
    report = _json(report_path)
    if report.get("audit_type") != "jepa_safe_capture_v5_settled_counterfactual":
        raise ValueError(f"Unexpected audit type: {path}")
    if report.get("development_only") is not True or report.get("locked_test_opened") is not False:
        raise ValueError(f"Development-only boundary failed: {path}")
    gates = report.get("gates")
    if not isinstance(gates, dict) or not all(bool(value) for value in gates.values()):
        raise ValueError(f"Settled source gates did not pass: {path}")
    protocol, environment, scene = _validate_run_entries(report, seed)
    rows = _read_rows(rows_path)
    keys: set[tuple[str, int, int]] = set()
    for row in rows:
        if int(row.get("training_seed", -1)) != seed:
            raise ValueError(f"Decision row seed mismatch: {path}")
        variant = str(row.get("variant", ""))
        if variant not in VARIANTS:
            raise ValueError(f"Unexpected variant {variant!r}: {path}")
        key = (variant, int(row.get("episode_index", -1)), int(row.get("step", -1)))
        if key in keys:
            raise ValueError(f"Duplicate decision key {key}: {path}")
        keys.add(key)
    declared_count = int(report.get("decision_count", -1))
    if declared_count != len(rows):
        raise ValueError(f"Decision count mismatch in {path}: {declared_count} != {len(rows)}")
    return {
        "path": str(path),
        "seed": seed,
        "report": report,
        "rows": rows,
        "protocol_sha256": protocol,
        "environment_config_sha256": environment,
        "scene_manifest_sha256": scene,
        "report_sha256": sha256(report_path),
        "rows_sha256": sha256(rows_path),
    }


def _selected_credit(row: Mapping[str, Any]) -> float | None:
    credits = row.get("ledger_credits")
    index = row.get("selected_index")
    if not isinstance(credits, list):
        return None
    try:
        index = int(index)
    except (TypeError, ValueError):
        return None
    if index < 0 or index >= len(credits):
        return None
    return _finite(credits[index])


def _selected_state(row: Mapping[str, Any]) -> str:
    states = row.get("ledger_states")
    index = row.get("selected_index")
    if isinstance(states, list):
        try:
            index = int(index)
            if 0 <= index < len(states):
                return str(states[index])
        except (TypeError, ValueError):
            pass
    return "unknown"


def _bucket(value: float | None, boundaries: Sequence[float], names: Sequence[str]) -> str:
    if value is None:
        return "unavailable"
    for boundary, name in zip(boundaries, names):
        if value < boundary:
            return name
    return names[-1]


def _bucket_stats(rows: Sequence[Mapping[str, Any]], field: str, boundaries: Sequence[float], names: Sequence[str]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {name: [] for name in names}
    groups["unavailable"] = []
    for row in rows:
        value = _finite(row.get(field))
        groups[_bucket(value, boundaries, names)].append(row)
    result: dict[str, Any] = {}
    for name, members in groups.items():
        settled = [row for row in members if str(row.get("selected_settled_termination_reason", "ineligible")) != "ineligible"]
        unsafe = [row for row in settled if not _bool(row.get("selected_settled_safety_ok", False))]
        result[name] = {
            "decisions": len(members),
            "settled_decisions": len(settled),
            "unsafe_settled_count": len(unsafe),
            "unsafe_settled_rate": float(len(unsafe) / len(settled)) if settled else None,
        }
    return result


def summarize_rows(rows: Sequence[Mapping[str, Any]], variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}")
    selected_not_best = [_bool(row.get("selected_not_best", False)) for row in rows]
    selected_safe = [_bool(row.get("selected_settled_safe_capture", False)) for row in rows]
    best_safe = [_bool(row.get("best_settled_safe_capture", False)) for row in rows]
    selected_safety = [_bool(row.get("selected_settled_safety_ok", False)) for row in rows]
    settled = [row for row in rows if str(row.get("selected_settled_termination_reason", "ineligible")) != "ineligible"]
    rank_spearman = [value for row in rows if (value := _finite(row.get("predicted_rank_spearman"))) is not None]
    rank_kendall = [value for row in rows if (value := _finite(row.get("predicted_rank_kendall"))) is not None]
    margins = [value for row in rows if (value := _finite(row.get("top_two_margin_m"))) is not None]
    progresses = [value for row in rows if (value := _finite(row.get("selected_settled_progress_m"))) is not None]
    best_progresses = [value for row in rows if (value := _finite(row.get("best_settled_progress_m"))) is not None]
    corrections = [value for row in rows if (value := _finite(row.get("selected_cbf_correction_norm_mps"))) is not None]
    selected_credits = [value for row in rows if (value := _selected_credit(row)) is not None]
    ledger_states = Counter(_selected_state(row) for row in rows)
    pair_labels = Counter(str(row.get("pair_label", "unknown")) for row in rows)
    terminations = Counter(str(row.get("selected_settled_termination_reason", "unknown")) for row in rows)
    eligible_counts = [sum(bool(item) for item in row.get("eligible_mask", [])) for row in rows]
    separation_pass = sum(value >= MINIMUM_SEPARATION_M for value in margins)
    return {
        "variant": variant,
        "decisions": len(rows),
        "settled_decisions": len(settled),
        "selected_not_best_count": int(sum(selected_not_best)),
        "selected_not_best_rate": float(np.mean(selected_not_best)) if rows else None,
        "selected_settled_safe_capture_count": int(sum(selected_safe)),
        "selected_settled_safe_capture_rate": float(np.mean(selected_safe)) if rows else None,
        "best_settled_safe_capture_count": int(sum(best_safe)),
        "best_settled_safe_capture_rate": float(np.mean(best_safe)) if rows else None,
        "selected_settled_safety_count": int(sum(selected_safety)),
        "selected_settled_safety_rate": float(np.mean(selected_safety)) if rows else None,
        "mean_selected_progress_m": _mean(progresses),
        "mean_best_progress_m": _mean(best_progresses),
        "mean_selected_cbf_correction_norm_mps": _mean(corrections),
        "mean_selected_credit": _mean(selected_credits),
        "rank_spearman_mean": _mean(rank_spearman),
        "rank_spearman_count": len(rank_spearman),
        "rank_kendall_mean": _mean(rank_kendall),
        "rank_kendall_count": len(rank_kendall),
        "candidate_separation_count": len(margins),
        "candidate_separation_pass_count": separation_pass,
        "candidate_separation_pass_rate": float(separation_pass / len(margins)) if margins else None,
        "candidate_separation_q10_m": float(np.percentile(margins, 10)) if margins else None,
        "candidate_separation_median_m": float(np.median(margins)) if margins else None,
        "mean_eligible_candidates": _mean(eligible_counts),
        "ledger_state_counts": dict(sorted(ledger_states.items())),
        "pair_label_counts": dict(sorted(pair_labels.items())),
        "termination_counts": dict(sorted(terminations.items())),
        "credit_buckets": _bucket_stats(
            [{**row, "selected_credit_value": _selected_credit(row)} for row in rows],
            "selected_credit_value",
            (MINIMUM_CREDIT, HIGH_CREDIT),
            ("low", "trusted", "high"),
        ),
        "visibility_buckets": _bucket_stats(rows, "selected_visibility_value", (0.30, 0.60), ("low", "medium", "high")),
        "ttc_buckets": _bucket_stats(rows, "selected_ttc_value", (0.30, 1.00), ("risk", "near", "far")),
        "cbf_correction_buckets": _bucket_stats(rows, "selected_correction_value", (0.001, 0.10), ("zero", "low", "high")),
    }


def _enrich_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        index = row.get("selected_index")
        for source, target in (
            ("predicted_visibility", "selected_visibility_value"),
            ("predicted_min_ttc_s", "selected_ttc_value"),
            ("selected_cbf_correction_norm_mps", "selected_correction_value"),
        ):
            if target in row:
                continue
            values = row.get(source)
            value = None
            if source == "selected_cbf_correction_norm_mps" and not isinstance(values, list):
                value = _finite(values)
            elif isinstance(values, list):
                try:
                    index_int = int(index)
                    if 0 <= index_int < len(values):
                        value = _finite(values[index_int])
                except (TypeError, ValueError):
                    value = None
            row[target] = value
        enriched.append(row)
    return enriched


def _combine_summaries(summaries: Sequence[Mapping[str, Any]], variant: str) -> dict[str, Any]:
    selected_not_best = sum(int(item["selected_not_best_count"]) for item in summaries)
    decisions = sum(int(item["decisions"]) for item in summaries)
    selected_safe = sum(int(item["selected_settled_safe_capture_count"]) for item in summaries)
    best_safe = sum(int(item["best_settled_safe_capture_count"]) for item in summaries)
    selected_safety = sum(int(item["selected_settled_safety_count"]) for item in summaries)
    settled = sum(int(item["settled_decisions"]) for item in summaries)
    def weighted(field: str, count_field: str) -> float | None:
        values = [(float(item[field]), int(item[count_field])) for item in summaries if item.get(field) is not None and int(item[count_field]) > 0]
        return float(sum(value * count for value, count in values) / sum(count for _, count in values)) if values else None
    rank_count = sum(int(item["rank_spearman_count"]) for item in summaries)
    kendall_count = sum(int(item["rank_kendall_count"]) for item in summaries)
    separation_count = sum(int(item["candidate_separation_count"]) for item in summaries)
    separation_pass = sum(int(item["candidate_separation_pass_count"]) for item in summaries)
    states = Counter()
    labels = Counter()
    terminations = Counter()
    for item in summaries:
        states.update(item["ledger_state_counts"])
        labels.update(item["pair_label_counts"])
        terminations.update(item["termination_counts"])
    def merge_buckets(field: str) -> dict[str, Any]:
        merged: dict[str, dict[str, int]] = {}
        for item in summaries:
            for name, bucket in item[field].items():
                target = merged.setdefault(name, {"decisions": 0, "settled_decisions": 0, "unsafe_settled_count": 0})
                for key in target:
                    target[key] += int(bucket.get(key, 0))
        for bucket in merged.values():
            settled_count = bucket["settled_decisions"]
            bucket["unsafe_settled_rate"] = float(bucket["unsafe_settled_count"] / settled_count) if settled_count else None
        return dict(sorted(merged.items()))
    return {
        "variant": variant,
        "decisions": decisions,
        "settled_decisions": settled,
        "selected_not_best_count": selected_not_best,
        "selected_not_best_rate": float(selected_not_best / decisions) if decisions else None,
        "selected_settled_safe_capture_count": selected_safe,
        "selected_settled_safe_capture_rate": float(selected_safe / decisions) if decisions else None,
        "best_settled_safe_capture_count": best_safe,
        "best_settled_safe_capture_rate": float(best_safe / decisions) if decisions else None,
        "selected_settled_safety_count": selected_safety,
        "selected_settled_safety_rate": float(selected_safety / decisions) if decisions else None,
        "settled_safety_unsafe_rate": float((settled - selected_safety) / settled) if settled else None,
        "mean_selected_progress_m": weighted("mean_selected_progress_m", "settled_decisions"),
        "mean_best_progress_m": weighted("mean_best_progress_m", "settled_decisions"),
        "mean_selected_cbf_correction_norm_mps": weighted("mean_selected_cbf_correction_norm_mps", "settled_decisions"),
        "mean_selected_credit": weighted("mean_selected_credit", "decisions"),
        "rank_spearman_mean": weighted("rank_spearman_mean", "rank_spearman_count"),
        "rank_spearman_count": rank_count,
        "rank_kendall_mean": weighted("rank_kendall_mean", "rank_kendall_count"),
        "rank_kendall_count": kendall_count,
        "candidate_separation_count": separation_count,
        "candidate_separation_pass_count": separation_pass,
        "candidate_separation_pass_rate": float(separation_pass / separation_count) if separation_count else None,
        "candidate_separation_q10_m": None,
        "candidate_separation_median_m": None,
        "ledger_state_counts": dict(sorted(states.items())),
        "pair_label_counts": dict(sorted(labels.items())),
        "termination_counts": dict(sorted(terminations.items())),
        "credit_buckets": merge_buckets("credit_buckets"),
        "visibility_buckets": merge_buckets("visibility_buckets"),
        "ttc_buckets": merge_buckets("ttc_buckets"),
        "cbf_correction_buckets": merge_buckets("cbf_correction_buckets"),
    }


def aggregate_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(reports) != len(SEEDS):
        raise ValueError(f"Expected exactly {len(SEEDS)} settled seed reports")
    by_seed = {int(item["seed"]): item for item in reports}
    if set(by_seed) != set(SEEDS):
        raise ValueError(f"Seed matrix mismatch: {sorted(by_seed)}")
    protocols = {str(item["protocol_sha256"]) for item in reports}
    environments = {str(item["environment_config_sha256"]) for item in reports}
    if len(protocols) != 1 or len(environments) != 1:
        raise ValueError("Settled reports must share protocol and environment hashes")
    seed_summaries: dict[str, dict[str, Any]] = {}
    aggregate_by_variant: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        summaries = []
        for seed in SEEDS:
            item = by_seed[seed]
            rows = _enrich_rows([row for row in item["rows"] if str(row.get("variant")) == variant])
            if not rows:
                raise ValueError(f"No rows for {variant}, seed {seed}")
            summary = summarize_rows(rows, variant)
            summaries.append(summary)
            seed_summaries.setdefault(str(seed), {})[variant] = summary
        aggregate_by_variant[variant] = _combine_summaries(summaries, variant)
    m3_seed_rates = [float(seed_summaries[str(seed)]["m3"]["selected_not_best_rate"]) for seed in SEEDS]
    ranking_gate = (
        aggregate_by_variant["m3"]["selected_not_best_rate"] is not None
        and aggregate_by_variant["m3"]["selected_not_best_rate"] <= RANKING_MAX_AGGREGATE_SELECTED_NOT_BEST
        and max(m3_seed_rates) <= RANKING_MAX_SEED_SELECTED_NOT_BEST
        and all((seed_summaries[str(seed)]["m3"]["rank_spearman_mean"] or 0.0) >= 0.0 for seed in SEEDS)
    )
    source_gates = all(bool(item["report"].get("all_gates_pass")) for item in reports)
    decision = {
        "source_gates_pass": source_gates,
        "ranking_gate": bool(ranking_gate),
        "selected_not_best_threshold_aggregate": RANKING_MAX_AGGREGATE_SELECTED_NOT_BEST,
        "selected_not_best_threshold_seed": RANKING_MAX_SEED_SELECTED_NOT_BEST,
        "m3_selected_not_best_seed_rates": {str(seed): seed_summaries[str(seed)]["m3"]["selected_not_best_rate"] for seed in SEEDS},
        "m3_rank_spearman_seed_means": {str(seed): seed_summaries[str(seed)]["m3"]["rank_spearman_mean"] for seed in SEEDS},
        "classification": "ranking_ready_for_closed_loop_smoke" if source_gates and ranking_gate else "ranking_unresolved",
        "locked_test_opened": False,
        "next_action": "proceed_to_failure_index_and_hard_replay" if source_gates and not ranking_gate else "stop_and_locate_score_label_horizon_or_action_scale_mismatch",
    }
    return {
        "aggregation_type": "jepa_safe_capture_v21_three_seed_settled_ranking",
        "stage": "S1_settled_ranking",
        "development_only": True,
        "locked_test_opened": False,
        "training_seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "protocol_sha256": next(iter(protocols)),
        "environment_config_sha256": next(iter(environments)),
        "scene_manifest_sha256_by_seed": {str(seed): by_seed[seed]["scene_manifest_sha256"] for seed in SEEDS},
        "per_seed": seed_summaries,
        "by_variant": aggregate_by_variant,
        "decision": decision,
        "inputs": [
            {
                "path": item["path"],
                "seed": item["seed"],
                "scene_manifest_sha256": item["scene_manifest_sha256"],
                "settled_report_sha256": item["report_sha256"],
                "decision_rows_sha256": item["rows_sha256"],
            }
            for item in sorted(reports, key=lambda item: int(item["seed"]))
        ],
    }


def write_tensorboard(report: Mapping[str, Any], logdir: Path) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard directory: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/aggregate", json.dumps({"aggregation_type": report["aggregation_type"], "stage": report["stage"]}, indent=2), 0)
        writer.add_text("Provenance/inputs", json.dumps(report["inputs"], indent=2), 0)
        writer.add_text("Provenance/decision", json.dumps(report["decision"], indent=2), 0)
        for index, variant in enumerate(VARIANTS):
            item = report["by_variant"][variant]
            writer.add_scalar(f"Ranking/{variant}/selected_not_best_rate", float(item["selected_not_best_rate"] or 0.0), index)
            writer.add_scalar(f"Ranking/{variant}/settled_safety_unsafe_rate", float(item["settled_safety_unsafe_rate"] or 0.0), index)
            writer.add_scalar(f"Ranking/{variant}/candidate_separation_pass_rate", float(item["candidate_separation_pass_rate"] or 0.0), index)
            writer.add_scalar(f"Ranking/{variant}/spearman_mean", float(item["rank_spearman_mean"] or 0.0), index)
            writer.add_scalar(f"Ranking/{variant}/kendall_mean", float(item["rank_kendall_mean"] or 0.0), index)
            writer.add_scalar(f"Ranking/{variant}/decisions", int(item["decisions"]), index)
        writer.add_scalar("Gates/source_gates_pass", int(bool(report["decision"]["source_gates_pass"])), 0)
        writer.add_scalar("Gates/ranking_gate", int(bool(report["decision"]["ranking_gate"])), 0)
    return {
        "logdir": str(logdir),
        "event_files": sorted(item.name for item in logdir.glob("events.out.tfevents.*")),
        "scalar_tag_count": 3 * 6 + 2,
        "text_tag_count": 3,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# V21 三 Seed Settled Ranking Aggregate",
        "",
        "> Offline-only local-chunk settlement. Target truth is not available to the online evaluator; this report does not open a locked test.",
        "",
        f"Protocol SHA-256: `{report['protocol_sha256']}`  ",
        f"Environment SHA-256: `{report['environment_config_sha256']}`",
        "",
        "| Variant | Decisions | Selected-not-best | Selected settled safe | Best settled safe | Settled unsafe | Separation pass | Spearman | Kendall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = report["by_variant"][variant]
        def pct(value: Any) -> str:
            return "n/a" if value is None else f"{float(value):.3f}"
        lines.append(
            f"| {variant.upper()} | {item['decisions']} | {pct(item['selected_not_best_rate'])} | "
            f"{pct(item['selected_settled_safe_capture_rate'])} | {pct(item['best_settled_safe_capture_rate'])} | "
            f"{pct(item['settled_safety_unsafe_rate'])} | {pct(item['candidate_separation_pass_rate'])} | "
            f"{pct(item['rank_spearman_mean'])} | {pct(item['rank_kendall_mean'])} |"
        )
    lines += [
        "",
        "## Per-seed M3",
        "",
        "| Seed | Decisions | Selected-not-best | Spearman | Candidate separation pass |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        item = report["per_seed"][str(seed)]["m3"]
        lines.append(f"| {seed} | {item['decisions']} | {float(item['selected_not_best_rate']):.3f} | {item['rank_spearman_mean'] if item['rank_spearman_mean'] is not None else 'n/a'} | {item['candidate_separation_pass_rate'] if item['candidate_separation_pass_rate'] is not None else 'n/a'} |")
    lines += [
        "",
        "## Gates",
        "",
        f"- Source settled gates: `{decision['source_gates_pass']}`.",
        f"- Ranking promotion gate: `{decision['ranking_gate']}`.",
        f"- Classification: `{decision['classification']}`.",
        f"- Next action: `{decision['next_action']}`.",
        "- Local settled outcomes are diagnostic labels, not full-episode policy outcomes.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("results"))
    parser.add_argument("--run-template", default="jepa_safe_capture_v21_settled_seed{seed}")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.development_only:
        raise ValueError("Settled ranking aggregation requires --development-only")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    reports = [
        load_settled_report(args.input_root.resolve() / args.run_template.format(seed=seed), seed)
        for seed in SEEDS
    ]
    report = aggregate_reports(reports)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    (output_dir / "input_manifest.json").write_text(json.dumps(_jsonable({"inputs": report["inputs"], "protocol_sha256": report["protocol_sha256"], "scene_manifest_sha256_by_seed": report["scene_manifest_sha256_by_seed"]}), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    tensorboard = write_tensorboard(report, args.tensorboard_dir)
    report["tensorboard"] = tensorboard
    report["provenance"] = {
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "development_only": True,
        "locked_test_opened": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable({"decision": report["decision"], "tensorboard": tensorboard}), indent=2))


if __name__ == "__main__":
    main()
