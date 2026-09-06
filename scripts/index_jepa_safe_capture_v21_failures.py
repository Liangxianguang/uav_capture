"""Build a read-only V21 failure index from paired smoke traces.

The index joins episode summaries, online step traces, and offline settled
counterfactual rows.  It is intentionally diagnostic: labels identify the
earliest available evidence in a trace and are not causal claims.  No target
ground truth is read by the online path and no source result is modified.
"""

from __future__ import annotations

import argparse
import csv
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
VARIANTS = ("m0", "m3", "a1", "a2")
SETTLED_VARIANTS = ("m3", "a1", "a2")
EXPECTED_EPISODES = 20
OBSERVATION_STALE_LIMIT = 45.0
MESSAGE_AGE_SATURATION_LIMIT = 60.0
VISIBILITY_DEGRADED_LIMIT = 0.50
OSCILLATION_RATE_LIMIT = 0.25
CLEARANCE_GAP_LIMIT_M = 0.05
HIGH_CREDIT_LIMIT = 0.80

LABEL_ORDER = (
    "collision",
    "boundary_violation",
    "pairwise_violation",
    "cbf_controlled_abort",
    "cbf_infeasible_or_unverified",
    "timeout",
    "candidate_capture_regression",
    "high_credit_failure",
    "low_credit_or_nominal_fallback",
    "stale_observation",
    "communication_age_saturated",
    "visibility_degraded",
    "candidate_oscillation",
    "clearance_prediction_gap",
    "unresolved_non_capture",
)


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
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _numbers(values: Iterable[Any]) -> list[float]:
    return [number for value in values if (number := _finite(value)) is not None]


def _mean(values: Iterable[Any]) -> float | None:
    numbers = _numbers(values)
    return float(np.mean(numbers)) if numbers else None


def _percentile(values: Iterable[Any], quantile: float) -> float | None:
    numbers = _numbers(values)
    return float(np.quantile(numbers, quantile)) if numbers else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"Empty JSONL: {path}")
    return rows


def _read_episode_table(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            index = int(raw["episode_index"])
            if index in result:
                raise ValueError(f"Duplicate episode index in {path}: {index}")
            result[index] = dict(raw)
    if sorted(result) != list(range(EXPECTED_EPISODES)):
        raise ValueError(f"Episode indices are not contiguous: {path}")
    return result


def load_smoke_run(
    input_root: Path,
    seed: int,
    variant: str,
    run_template: str = "jepa_safe_capture_v21_smoke_{variant}_seed{seed}",
) -> dict[str, Any]:
    """Load one paired run using an explicit, auditable directory template."""

    path = (input_root / run_template.format(seed=seed, variant=variant)).resolve()
    required = ("summary.json", "provenance.json", "episodes.csv", "scene_manifest.jsonl")
    for name in required:
        if not (path / name).is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    summary = _json(path / "summary.json")
    provenance = _json(path / "provenance.json")
    if provenance.get("development_only") is not True or provenance.get("locked_test_opened") is not False:
        raise ValueError(f"Development boundary failed: {path}")
    declared = provenance.get("variant", {})
    if not isinstance(declared, Mapping) or declared.get("variant") != variant:
        raise ValueError(f"Variant provenance mismatch: {path}")
    if int(provenance.get("training_seed", -1)) != seed:
        raise ValueError(f"Seed provenance mismatch: {path}")
    if int(provenance.get("episodes", -1)) != EXPECTED_EPISODES:
        raise ValueError(f"Episode count mismatch: {path}")
    inputs = provenance.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise ValueError(f"Missing input provenance: {path}")
    manifest = path / "scene_manifest.jsonl"
    manifest_hash = sha256(manifest)
    if str(inputs.get("scene_manifest_sha256", "")) != manifest_hash:
        raise ValueError(f"Scene manifest hash mismatch: {path}")
    traces = sorted((path / "step_traces").glob("episode_*.jsonl"))
    if len(traces) != EXPECTED_EPISODES:
        raise ValueError(f"Expected {EXPECTED_EPISODES} step traces, found {len(traces)}: {path}")
    return {
        "path": path,
        "seed": seed,
        "variant": variant,
        "summary": summary,
        "provenance": provenance,
        "episodes": _read_episode_table(path / "episodes.csv"),
        "manifest_sha256": manifest_hash,
        "summary_sha256": sha256(path / "summary.json"),
        "provenance_sha256": sha256(path / "provenance.json"),
    }


def load_settled_rows(
    settled_root: Path,
    seed: int,
    settled_template: str = "jepa_safe_capture_v21_settled_seed{seed}",
) -> dict[tuple[str, int, int], dict[str, Any]]:
    path = (settled_root / settled_template.format(seed=seed) / "decision_rows.jsonl").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing settled rows: {path}")
    rows = _read_jsonl(path)
    result: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        variant = str(row.get("variant", ""))
        if variant not in SETTLED_VARIANTS:
            continue
        key = (variant, int(row.get("episode_index", -1)), int(row.get("step", -1)))
        if key in result:
            raise ValueError(f"Duplicate settled key {key}: {path}")
        result[key] = row
    if not result:
        raise ValueError(f"No settled rows for seed {seed}: {path}")
    return result


def _trace(path: Path, episode_index: int) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    for row in rows:
        if int(row.get("episode_index", -1)) != episode_index:
            raise ValueError(f"Trace episode mismatch: {path}")
    return rows


def _selected_value(values: Any, index: Any) -> float | None:
    if not isinstance(values, list):
        return _finite(values)
    try:
        index = int(index)
    except (TypeError, ValueError):
        return None
    if 0 <= index < len(values):
        return _finite(values[index])
    return None


def summarize_trace(trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected: list[int] = []
    ranking_modes: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    ledger_states: Counter[str] = Counter()
    ledger_credits: list[float] = []
    cbf_modes: Counter[str] = Counter()
    cbf_unverified = cbf_infeasible = cbf_timeout = cbf_abort = 0
    corrections: list[float] = []
    solve_latencies: list[float] = []
    predicted_clearance: list[float] = []
    observed_clearance: list[float] = []
    observed_visibility: list[float] = []
    observation_ages: list[float] = []
    message_ages: list[float] = []
    for row in trace:
        ranking = row.get("candidate_ranking")
        if isinstance(ranking, Mapping):
            if ranking.get("selected_index") is not None:
                selected.append(int(ranking["selected_index"]))
            if ranking.get("execution_mode") is not None:
                ranking_modes[str(ranking["execution_mode"])] += 1
            if ranking.get("fallback_reason") is not None:
                fallback_reasons[str(ranking["fallback_reason"])] += 1
            states = ranking.get("ledger_states")
            if isinstance(states, list):
                ledger_states.update(str(value) for value in states)
            credits = ranking.get("ledger_credits")
            if isinstance(credits, list):
                ledger_credits.extend(_numbers(credits))
            values = _numbers(ranking.get("predicted_min_clearance_m", []))
            if values:
                predicted_clearance.append(min(values))
        cbf = row.get("cbf")
        if isinstance(cbf, Mapping):
            mode = cbf.get("fallback_mode")
            if mode is not None:
                cbf_modes[str(mode)] += 1
            if _bool(cbf.get("unverified", False)) or not _bool(cbf.get("verified_feasible", True)):
                cbf_unverified += 1
            if _bool(cbf.get("infeasible", False)):
                cbf_infeasible += 1
            if _bool(cbf.get("timed_out", False)):
                cbf_timeout += 1
            if str(cbf.get("fallback_mode", "")) == "controlled_abort":
                cbf_abort += 1
            correction = _finite(cbf.get("action_correction_norm"))
            if correction is not None:
                corrections.append(correction)
            latency = _finite(cbf.get("solve_latency_ms"))
            if latency is not None:
                solve_latencies.append(latency)
        safety = row.get("safety_observables")
        if isinstance(safety, Mapping):
            value = _finite(safety.get("minimum_obstacle_clearance_m"))
            if value is not None:
                observed_clearance.append(value)
        observation = row.get("observation")
        if isinstance(observation, Mapping):
            visible = observation.get("target_visible")
            if isinstance(visible, list) and visible:
                observed_visibility.append(float(np.mean([_bool(value) for value in visible])))
            for source, destination in (
                ("target_observation_age_steps", observation_ages),
                ("message_age_steps", message_ages),
            ):
                values = observation.get(source)
                if isinstance(values, list):
                    destination.extend(_numbers(values))
    switches = sum(left != right for left, right in zip(selected, selected[1:]))
    non_nominal = sum(index != 0 for index in selected)
    gap_count = min(len(predicted_clearance), len(observed_clearance))
    gaps = [predicted_clearance[index] - observed_clearance[index] for index in range(gap_count)]
    return {
        "trace_steps": len(trace),
        "candidate_switch_count": switches,
        "candidate_switch_rate": float(switches / max(len(selected) - 1, 1)) if selected else 0.0,
        "non_nominal_selection_rate": float(non_nominal / max(len(selected), 1)),
        "selected_candidate_indices": selected,
        "ranking_mode_counts": dict(sorted(ranking_modes.items())),
        "fallback_reason_counts": dict(sorted(fallback_reasons.items())),
        "ledger_state_counts": dict(sorted(ledger_states.items())),
        "ledger_credit_mean": _mean(ledger_credits),
        "ledger_credit_min": min(ledger_credits) if ledger_credits else None,
        "cbf_fallback_mode_counts": dict(sorted(cbf_modes.items())),
        "cbf_unverified_steps_trace": cbf_unverified,
        "cbf_infeasible_steps_trace": cbf_infeasible,
        "cbf_timeout_steps_trace": cbf_timeout,
        "cbf_controlled_abort_steps_trace": cbf_abort,
        "cbf_correction_mean_mps": _mean(corrections),
        "cbf_correction_p95_mps": _percentile(corrections, 0.95),
        "cbf_correction_max_mps": max(corrections) if corrections else None,
        "cbf_latency_p95_ms": _percentile(solve_latencies, 0.95),
        "predicted_clearance_mean_m": _mean(predicted_clearance),
        "observed_clearance_mean_m": _mean(observed_clearance),
        "clearance_prediction_gap_mean_m": _mean(gaps),
        "clearance_overoptimism_max_m": max((gap for gap in gaps if gap > 0.0), default=None),
        "observed_visibility_mean": _mean(observed_visibility),
        "observation_age_max_steps": max(observation_ages) if observation_ages else None,
        "message_age_max_steps": max(message_ages) if message_ages else None,
    }


def _settled_episode_diagnostics(
    settled: Mapping[tuple[str, int, int], Mapping[str, Any]],
    variant: str,
    episode_index: int,
) -> dict[str, Any]:
    rows = [row for (row_variant, row_episode, _), row in settled.items() if row_variant == variant and row_episode == episode_index]
    selected_not_best = sum(_bool(row.get("selected_not_best", False)) for row in rows)
    settled_rows = [row for row in rows if str(row.get("selected_settled_termination_reason", "ineligible")) != "ineligible"]
    high_credit_failures = 0
    selected_credit_values: list[float] = []
    for row in settled_rows:
        index = row.get("selected_index")
        credits = row.get("ledger_credits")
        credit = _selected_value(credits, index)
        if credit is not None:
            selected_credit_values.append(credit)
            if credit >= HIGH_CREDIT_LIMIT and not _bool(row.get("selected_settled_safety_ok", False)):
                high_credit_failures += 1
    return {
        "settled_row_count": len(rows),
        "settled_count": len(settled_rows),
        "settled_selected_not_best_count": selected_not_best,
        "settled_selected_not_best_rate": float(selected_not_best / len(rows)) if rows else None,
        "settled_high_credit_failure_count": high_credit_failures,
        "settled_selected_credit_mean": _mean(selected_credit_values),
    }


def classify_failure(
    episode: Mapping[str, Any],
    trace_summary: Mapping[str, Any],
    settled_summary: Mapping[str, Any] | None,
) -> tuple[str, list[str]]:
    labels: list[str] = []
    if _bool(episode.get("collision")):
        labels.append("collision")
    if _bool(episode.get("defender_boundary_violation", episode.get("boundary_violation"))):
        labels.append("boundary_violation")
    if _bool(episode.get("pairwise_violation")):
        labels.append("pairwise_violation")
    controlled_abort = int(episode.get("cbf_controlled_abort_steps", 0) or 0) > 0 or trace_summary["cbf_controlled_abort_steps_trace"] > 0 or str(episode.get("termination_reason", "")) == "cbf_controlled_abort"
    if controlled_abort:
        labels.append("cbf_controlled_abort")
    if int(episode.get("cbf_infeasible_steps", 0) or 0) > 0 or int(episode.get("cbf_timeout_steps", 0) or 0) > 0 or int(episode.get("cbf_unverified_steps", 0) or 0) > 0 or trace_summary["cbf_unverified_steps_trace"] > 0:
        labels.append("cbf_infeasible_or_unverified")
    if str(episode.get("termination_reason", "")) in {"timeout", "truncated"}:
        labels.append("timeout")
    if settled_summary:
        if settled_summary["settled_selected_not_best_count"] > 0:
            labels.append("candidate_capture_regression")
        if settled_summary["settled_high_credit_failure_count"] > 0:
            labels.append("high_credit_failure")
    modes = trace_summary.get("ranking_mode_counts", {})
    if any("fallback" in str(mode) or str(mode) == "safe_hold" for mode in modes):
        labels.append("low_credit_or_nominal_fallback")
    if (trace_summary.get("observation_age_max_steps") or 0.0) > OBSERVATION_STALE_LIMIT:
        labels.append("stale_observation")
    if (trace_summary.get("message_age_max_steps") or 0.0) >= MESSAGE_AGE_SATURATION_LIMIT:
        labels.append("communication_age_saturated")
    if trace_summary.get("observed_visibility_mean") is not None and trace_summary["observed_visibility_mean"] < VISIBILITY_DEGRADED_LIMIT:
        labels.append("visibility_degraded")
    if float(trace_summary.get("candidate_switch_rate") or 0.0) > OSCILLATION_RATE_LIMIT:
        labels.append("candidate_oscillation")
    if (trace_summary.get("clearance_overoptimism_max_m") or 0.0) > CLEARANCE_GAP_LIMIT_M:
        labels.append("clearance_prediction_gap")
    if not labels:
        labels.append("unresolved_non_capture")
    rank = {label: index for index, label in enumerate(LABEL_ORDER)}
    return min(labels, key=lambda label: rank.get(label, len(rank))), labels


def _episode_row(run: Mapping[str, Any], episode_index: int, settled: Mapping[tuple[str, int, int], Mapping[str, Any]] | None) -> dict[str, Any]:
    episode = run["episodes"][episode_index]
    trace_path = run["path"] / "step_traces" / f"episode_{episode_index:04d}.jsonl"
    trace_summary = summarize_trace(_trace(trace_path, episode_index))
    settled_summary = None
    if settled is not None:
        settled_summary = _settled_episode_diagnostics(settled, str(run["variant"]), episode_index)
        if settled_summary["settled_row_count"] == 0:
            raise ValueError(f"Missing settled rows for {run['variant']} seed {run['seed']} episode {episode_index}")
    safe = _bool(episode.get("cooperative_safe_capture", episode.get("safe_capture_success")))
    primary, labels = classify_failure(episode, trace_summary, settled_summary)
    row: dict[str, Any] = {
        "training_seed": run["seed"],
        "variant": run["variant"],
        "episode_index": episode_index,
        "episode_seed": int(episode.get("episode_seed", -1)),
        "layout_seed": int(episode.get("layout_seed", -1)),
        "layout_signature": str(episode.get("layout_signature", "")),
        "scenario": str(episode.get("scenario", "")),
        "observation_condition": str(episode.get("observation_condition", "")),
        "target_motion_mode": str(episode.get("target_motion_mode", "")),
        "obstacle_count": int(episode.get("obstacle_count", -1)),
        "safe_capture": safe,
        "termination_reason": str(episode.get("termination_reason", episode.get("task_termination_reason", ""))),
        "primary_cause": primary,
        "diagnostic_labels": labels,
        "source_manifest_sha256": run["manifest_sha256"],
        "source_summary_sha256": run["summary_sha256"],
        "source_provenance_sha256": run["provenance_sha256"],
    }
    row.update(trace_summary)
    if settled_summary:
        row.update(settled_summary)
    return row


def build_index(
    input_root: Path,
    settled_root: Path,
    run_template: str = "jepa_safe_capture_v21_smoke_{variant}_seed{seed}",
    settled_template: str = "jepa_safe_capture_v21_settled_seed{seed}",
) -> dict[str, Any]:
    runs = [load_smoke_run(input_root, seed, variant, run_template) for seed in SEEDS for variant in VARIANTS]
    manifests_by_seed: dict[int, set[str]] = {seed: set() for seed in SEEDS}
    protocol_hashes: set[str] = set()
    environment_hashes: set[str] = set()
    for run in runs:
        manifests_by_seed[run["seed"]].add(run["manifest_sha256"])
        inputs = run["provenance"].get("inputs", {})
        protocol_hashes.add(str(inputs.get("protocol_sha256", "")))
        environment_hashes.add(str(inputs.get("environment_config_sha256", "")))
    if any(len(values) != 1 for values in manifests_by_seed.values()):
        raise ValueError(f"V21 variants are not paired within seed: {manifests_by_seed}")
    if len(protocol_hashes) != 1 or "" in protocol_hashes or len(environment_hashes) != 1 or "" in environment_hashes:
        raise ValueError("V21 protocol/environment hashes are inconsistent")
    settled_by_seed = {
        seed: load_settled_rows(settled_root, seed, settled_template)
        for seed in SEEDS
    }
    rows: list[dict[str, Any]] = []
    for run in runs:
        settled = settled_by_seed[run["seed"]].copy() if run["variant"] in SETTLED_VARIANTS else None
        rows.extend(_episode_row(run, index, settled) for index in range(EXPECTED_EPISODES))
    failures = [row for row in rows if not row["safe_capture"]]
    safe = [row for row in rows if row["safe_capture"]]
    by_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        subset = [row for row in rows if row["variant"] == variant]
        failed = [row for row in subset if not row["safe_capture"]]
        by_variant[variant] = {
            "episodes": len(subset),
            "safe_capture_count": sum(row["safe_capture"] for row in subset),
            "safe_capture_rate": float(np.mean([row["safe_capture"] for row in subset])),
            "failure_count": len(failed),
            "primary_cause_counts": dict(Counter(row["primary_cause"] for row in failed)),
            "diagnostic_label_counts": dict(Counter(label for row in failed for label in row["diagnostic_labels"])),
            "high_credit_failure_count": sum("high_credit_failure" in row["diagnostic_labels"] for row in failed),
            "settled_rank_mismatch_episode_count": sum("candidate_capture_regression" in row["diagnostic_labels"] for row in failed),
            "fallback_episode_count": sum("low_credit_or_nominal_fallback" in row["diagnostic_labels"] for row in failed),
            "cbf_abort_episode_count": sum("cbf_controlled_abort" in row["diagnostic_labels"] for row in failed),
            "mean_candidate_switch_rate": _mean(row["candidate_switch_rate"] for row in subset),
            "mean_cbf_correction_p95_mps": _mean(row["cbf_correction_p95_mps"] for row in subset),
        }
    by_condition: dict[str, Any] = {}
    for condition in sorted({str(row["observation_condition"]) for row in rows}):
        subset = [row for row in rows if row["observation_condition"] == condition]
        by_condition[condition] = {
            "episodes": len(subset),
            "safe_capture_rate": float(np.mean([row["safe_capture"] for row in subset])),
            "failure_count": sum(not row["safe_capture"] for row in subset),
            "primary_cause_counts": dict(Counter(row["primary_cause"] for row in subset if not row["safe_capture"])),
        }
    safety_gate = all(
        "collision" not in row["diagnostic_labels"]
        and "boundary_violation" not in row["diagnostic_labels"]
        and "pairwise_violation" not in row["diagnostic_labels"]
        for row in rows
    )
    raw_unverified = sum(int(run["episodes"][index].get("raw_unverified_executed_steps", 0) or 0) for run in runs for index in range(EXPECTED_EPISODES))
    return {
        "index_type": "jepa_safe_capture_v21_paired_smoke_failure_index",
        "input_format": "v21",
        "development_only": True,
        "locked_test_opened": False,
        "input_root": str(input_root.resolve()),
        "settled_root": str(settled_root.resolve()),
        "run_template": run_template,
        "settled_template": settled_template,
        "run_count": len(runs),
        "episode_count": len(rows),
        "safe_capture_count": len(safe),
        "failure_count": len(failures),
        "safe_capture_rate": float(len(safe) / len(rows)),
        "primary_cause_counts": dict(Counter(row["primary_cause"] for row in failures)),
        "diagnostic_label_counts": dict(Counter(label for row in failures for label in row["diagnostic_labels"])),
        "by_variant": by_variant,
        "by_observation_condition": by_condition,
        "manifest_sha256_by_seed": {str(seed): next(iter(values)) for seed, values in manifests_by_seed.items()},
        "protocol_sha256": next(iter(protocol_hashes)),
        "environment_config_sha256": next(iter(environment_hashes)),
        "safety_hard_gate": bool(safety_gate and raw_unverified == 0),
        "raw_unverified_executed_steps": raw_unverified,
        "settled_row_coverage": {
            str(seed): {variant: sum(1 for key in settled_by_seed[seed] if key[0] == variant) for variant in SETTLED_VARIANTS}
            for seed in SEEDS
        },
        "target_drift_observable": False,
        "target_drift_note": "Online traces contain no future target labels; target drift is not inferred from proxy fields.",
        "runs": [
            {
                "training_seed": run["seed"],
                "variant": run["variant"],
                "path": str(run["path"]),
                "manifest_sha256": run["manifest_sha256"],
                "summary_sha256": run["summary_sha256"],
                "provenance_sha256": run["provenance_sha256"],
            }
            for run in runs
        ],
        "rows": rows,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "training_seed", "variant", "episode_index", "episode_seed", "layout_seed", "layout_signature",
        "scenario", "observation_condition", "target_motion_mode", "obstacle_count", "safe_capture",
        "termination_reason", "primary_cause", "diagnostic_labels", "trace_steps", "candidate_switch_count",
        "candidate_switch_rate", "non_nominal_selection_rate", "ledger_state_counts", "ledger_credit_mean",
        "ledger_credit_min", "cbf_fallback_mode_counts", "cbf_unverified_steps_trace", "cbf_infeasible_steps_trace",
        "cbf_timeout_steps_trace", "cbf_controlled_abort_steps_trace", "cbf_correction_mean_mps", "cbf_correction_p95_mps",
        "cbf_latency_p95_ms", "predicted_clearance_mean_m", "observed_clearance_mean_m", "clearance_prediction_gap_mean_m",
        "clearance_overoptimism_max_m", "observed_visibility_mean", "observation_age_max_steps", "message_age_max_steps",
        "settled_row_count", "settled_count", "settled_selected_not_best_count", "settled_high_credit_failure_count",
        "source_manifest_sha256", "source_summary_sha256", "source_provenance_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            values = dict(row)
            for key in ("diagnostic_labels", "ledger_state_counts", "cbf_fallback_mode_counts"):
                if key in values:
                    values[key] = json.dumps(values[key], ensure_ascii=True, separators=(",", ":"))
            writer.writerow(values)


def write_tensorboard(logdir: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard directory: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/v21_failure_index", json.dumps({"index_type": report["index_type"], "development_only": True, "locked_test_opened": False}, indent=2), 0)
        writer.add_text("Provenance/runs", json.dumps(report["runs"], indent=2), 0)
        writer.add_text("Provenance/limits", report["target_drift_note"], 0)
        writer.add_scalar("Episodes/total", report["episode_count"], 0)
        writer.add_scalar("Episodes/safe_capture", report["safe_capture_count"], 0)
        writer.add_scalar("Episodes/failure", report["failure_count"], 0)
        writer.add_scalar("Episodes/safe_capture_rate", report["safe_capture_rate"], 0)
        for cause, count in sorted(report["primary_cause_counts"].items()):
            writer.add_scalar(f"Failure/primary/{cause}", count, 0)
        for label, count in sorted(report["diagnostic_label_counts"].items()):
            writer.add_scalar(f"Failure/diagnostic/{label}", count, 0)
        for variant, values in report["by_variant"].items():
            writer.add_scalar(f"Variant/{variant}/safe_capture_rate", values["safe_capture_rate"], 0)
            writer.add_scalar(f"Variant/{variant}/failure_count", values["failure_count"], 0)
            writer.add_scalar(f"Variant/{variant}/high_credit_failure_count", values["high_credit_failure_count"], 0)
    return {
        "logdir": str(logdir),
        "event_files": sorted(path.name for path in logdir.glob("events.out.tfevents.*")),
        "required_provenance": True,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# V21 Paired Smoke Failure Index",
        "",
        "> Development-only read-only audit. Target drift is not inferred without future labels; this report does not open a locked test.",
        "",
        f"Runs: `{report['run_count']}`; episodes: `{report['episode_count']}`; safe capture: `{report['safe_capture_count']}/{report['episode_count']}` (`{report['safe_capture_rate']:.1%}`).",
        f"Safety hard gate: `{report['safety_hard_gate']}`; raw unverified executed steps: `{report['raw_unverified_executed_steps']}`.",
        "",
        "## Primary Causes",
        "",
        "| Primary cause | Failed episodes |",
        "|---|---:|",
    ]
    for cause, count in sorted(report["primary_cause_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{cause}` | {count} |")
    lines.extend(["", "## Diagnostic Labels", "", "| Label | Failed episodes carrying label |", "|---|---:|"])
    for label, count in sorted(report["diagnostic_label_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{label}` | {count} |")
    lines.extend(["", "## By Variant", "", "| Variant | Episodes | Safe capture | Failures | Rank mismatch | High-credit failure | Communication saturated | Fallback | CBF abort |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for variant, values in report["by_variant"].items():
        communication_saturated = sum(
            1 for row in report["rows"]
            if row["variant"] == variant and not row["safe_capture"] and "communication_age_saturated" in row["diagnostic_labels"]
        )
        lines.append(f"| `{variant}` | {values['episodes']} | {values['safe_capture_count']}/{values['episodes']} ({values['safe_capture_rate']:.1%}) | {values['failure_count']} | {values['settled_rank_mismatch_episode_count']} | {values['high_credit_failure_count']} | {communication_saturated} | {values['fallback_episode_count']} | {values['cbf_abort_episode_count']} |")
    lines.extend([
        "",
        "## Provenance",
        "",
        f"Protocol SHA-256: `{report['protocol_sha256']}`  ",
        f"Environment SHA-256: `{report['environment_config_sha256']}`  ",
        f"Settled row coverage: `{json.dumps(report['settled_row_coverage'], sort_keys=True)}`",
    ])
    lines.extend([
        "",
        "## Hard-Replay Selection",
        "",
        "Prioritize failed episodes carrying `candidate_capture_regression`, `high_credit_failure`, `cbf_controlled_abort`, `stale_observation`, `communication_age_saturated`, or `candidate_oscillation`. Replays must remain deterministic and write a new train-only archive if they are later used for training.",
        "",
        "## Interpretation",
        "",
        "- Labels are evidence-based diagnostic categories, not causal claims.",
        "- Settled rows are local action-chunk outcomes; they are not full-episode policy outcomes.",
        "- Keep CBF margins, OOD/stale gates, controlled-abort semantics and the locked-test boundary unchanged.",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--settled-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument(
        "--run-template",
        default="jepa_safe_capture_v21_smoke_{variant}_seed{seed}",
        help="Input run directory template with {seed} and {variant} placeholders.",
    )
    parser.add_argument(
        "--settled-template",
        default="jepa_safe_capture_v21_settled_seed{seed}",
        help="Settled directory template with a {seed} placeholder.",
    )
    parser.add_argument("--development-only", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.development_only:
        raise ValueError("V21 failure indexing requires --development-only")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    report = build_index(
        args.input_root.resolve(),
        args.settled_root.resolve(),
        run_template=args.run_template,
        settled_template=args.settled_template,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "failure_index.csv", report["rows"])
    (output_dir / "failure_index.json").write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output_dir / "provenance.json").write_text(json.dumps(_jsonable({
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "development_only": True,
        "locked_test_opened": False,
        "protocol_sha256": report["protocol_sha256"],
        "environment_config_sha256": report["environment_config_sha256"],
    }), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    tensorboard = write_tensorboard(args.tensorboard_logdir, report)
    report["tensorboard"] = tensorboard
    (output_dir / "failure_index.json").write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable({"safety_hard_gate": report["safety_hard_gate"], "primary_causes": report["primary_cause_counts"], "tensorboard": tensorboard}), indent=2))


if __name__ == "__main__":
    main()
