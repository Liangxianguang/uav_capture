"""Build a causal failure index from paired JEPA safe-capture development runs.

This tool is an offline audit. It reads the existing P7 development summaries
and step traces, never opens a locked-test split, and never modifies a run
directory. It separates episode outcomes (safety and capture) from diagnostic
signals (ledger abstention, stale observations, candidate switching, CBF
intervention, and prediction gaps). A signal is not promoted to a causal claim
unless the trace contains the required evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from aggregate_jepa_safe_capture_v2_paired import (  # noqa: E402
    discover_runs,
    sha256,
)


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
    "visibility_degraded",
    "candidate_oscillation",
    "clearance_prediction_gap",
    "visibility_prediction_gap",
    "unresolved_non_capture",
)


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, default=PROJECT_ROOT / "results/jepa_safe_capture_v3_wp0_baseline_freeze_20260904/manifest.json")
    parser.add_argument("--stage", choices=("smoke", "full"), default="full")
    parser.add_argument("--development-only", action="store_true", required=True)
    return parser.parse_args()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _numeric(values: Iterable[Any]) -> np.ndarray:
    finite = [number for value in values if (number := _finite_float(value)) is not None]
    return np.asarray(finite, dtype=np.float64)


def _mean(values: Iterable[Any]) -> float | None:
    numbers = _numeric(values)
    return float(np.mean(numbers)) if numbers.size else None


def _max(values: Iterable[Any]) -> float | None:
    numbers = _numeric(values)
    return float(np.max(numbers)) if numbers.size else None


def _percentile(values: Iterable[Any], quantile: float) -> float | None:
    numbers = _numeric(values)
    return float(np.quantile(numbers, quantile)) if numbers.size else None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_episode_metadata(path: Path) -> dict[int, dict[str, Any]]:
    """Read the full source episode table without changing aggregate fields."""

    metadata: dict[int, dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            try:
                index = int(raw["episode_index"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid episode_index in {path}: {raw!r}") from error
            if index in metadata:
                raise ValueError(f"Duplicate episode_index in {path}: {index}")
            metadata[index] = dict(raw)
    return metadata


def _validate_freeze_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path.resolve())
    if manifest.get("development_only") is not True or manifest.get("locked_test_opened") is not False:
        raise ValueError(f"WP1 requires a development-only freeze manifest: {path}")
    if manifest.get("freeze_type") != "jepa_safe_capture_v3_next_phase_wp0_baseline":
        raise ValueError(f"Unexpected WP0 freeze manifest type: {path}")
    return manifest


def read_trace(path: Path, episode_index: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Trace row is not an object: {path}:{line_number}")
        if int(value.get("episode_index", -1)) != episode_index:
            raise ValueError(f"Trace episode index mismatch: {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"Trace is empty: {path}")
    return rows


def _flatten_state_values(trace: list[dict[str, Any]], field: str) -> list[Any]:
    result: list[Any] = []
    for row in trace:
        ranking = row.get("candidate_ranking")
        if not isinstance(ranking, Mapping):
            continue
        value = ranking.get(field)
        if isinstance(value, list):
            result.extend(value)
        elif value is not None:
            result.append(value)
    return result


def summarize_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    selected = []
    ranking_modes: list[str] = []
    ledger_states: list[str] = []
    ledger_credits: list[float] = []
    fallback_reasons: list[str] = []
    cbf_modes: list[str] = []
    cbf_unverified = 0
    cbf_infeasible = 0
    cbf_timeouts = 0
    corrections: list[float] = []
    solve_latencies: list[float] = []
    predicted_clearance: list[float] = []
    observed_clearance: list[float] = []
    predicted_visibility: list[float] = []
    observed_visibility: list[float] = []
    observation_ages: list[float] = []
    message_ages: list[float] = []
    for row in trace:
        ranking = row.get("candidate_ranking")
        if isinstance(ranking, Mapping):
            selected_value = ranking.get("selected_index")
            if selected_value is not None:
                selected.append(int(selected_value))
            mode = ranking.get("execution_mode")
            if mode is not None:
                ranking_modes.append(str(mode))
            states = ranking.get("ledger_states")
            if isinstance(states, list):
                ledger_states.extend(str(value) for value in states)
            credits = ranking.get("ledger_credits")
            if isinstance(credits, list):
                ledger_credits.extend(value for value in (_finite_float(item) for item in credits) if value is not None)
            reasons = ranking.get("ledger_fallback_reasons")
            if isinstance(reasons, list):
                fallback_reasons.extend(str(value) for value in reasons if value is not None)
            values = ranking.get("predicted_min_clearance_m")
            if isinstance(values, list):
                finite = [value for value in (_finite_float(item) for item in values) if value is not None]
                if finite:
                    # Compare one conservative prediction with one observed
                    # value per control step; do not flatten candidate values
                    # into a misaligned time series.
                    predicted_clearance.append(min(finite))
            values = ranking.get("predicted_visibility")
            if isinstance(values, list):
                finite = [value for value in (_finite_float(item) for item in values) if value is not None]
                if finite:
                    predicted_visibility.append(float(np.mean(finite)))
        cbf = row.get("cbf")
        if isinstance(cbf, Mapping):
            mode = cbf.get("fallback_mode")
            if mode is not None:
                cbf_modes.append(str(mode))
            if _as_bool(cbf.get("unverified", False)) or not _as_bool(cbf.get("verified_feasible", True)):
                cbf_unverified += 1
            if _as_bool(cbf.get("infeasible", False)):
                cbf_infeasible += 1
            if _as_bool(cbf.get("timed_out", False)):
                cbf_timeouts += 1
            correction = _finite_float(cbf.get("action_correction_norm"))
            if correction is not None:
                corrections.append(correction)
            latency = _finite_float(cbf.get("solve_latency_ms"))
            if latency is not None:
                solve_latencies.append(latency)
        safety = row.get("safety_observables")
        if isinstance(safety, Mapping):
            value = _finite_float(safety.get("minimum_obstacle_clearance_m"))
            if value is not None:
                observed_clearance.append(value)
        observation = row.get("observation")
        if isinstance(observation, Mapping):
            visible = observation.get("target_visible")
            if isinstance(visible, list) and visible:
                observed_visibility.append(float(np.mean([_as_bool(item) for item in visible])))
            for key, destination in (
                ("target_observation_age_steps", observation_ages),
                ("message_age_steps", message_ages),
            ):
                values = observation.get(key)
                if isinstance(values, list):
                    destination.extend(value for value in (_finite_float(item) for item in values) if value is not None)
    switches = sum(left != right for left, right in zip(selected, selected[1:]))
    non_nominal = sum(index != 0 for index in selected)
    clearance_gaps = [predicted - observed for predicted, observed in zip(predicted_clearance, observed_clearance)]
    visibility_gaps = [predicted - observed for predicted, observed in zip(predicted_visibility, observed_visibility)]
    return {
        "trace_steps": len(trace),
        "selected_candidate_indices": selected,
        "candidate_switch_count": int(switches),
        "candidate_switch_rate": float(switches / max(len(selected) - 1, 1)) if selected else 0.0,
        "non_nominal_selection_rate": float(non_nominal / max(len(selected), 1)),
        "ranking_modes": dict(Counter(ranking_modes)),
        "ledger_state_counts": dict(Counter(ledger_states)),
        "ledger_credit_mean": _mean(ledger_credits),
        "ledger_credit_min": min(ledger_credits) if ledger_credits else None,
        "ledger_fallback_reason_counts": dict(Counter(fallback_reasons)),
        "cbf_fallback_mode_counts": dict(Counter(cbf_modes)),
        "cbf_unverified_steps_trace": cbf_unverified,
        "cbf_infeasible_steps_trace": cbf_infeasible,
        "cbf_timeout_steps_trace": cbf_timeouts,
        "cbf_correction_mean_mps": _mean(corrections),
        "cbf_correction_p95_mps": _percentile(corrections, 0.95),
        "cbf_correction_max_mps": _max(corrections),
        "cbf_latency_p95_ms": _percentile(solve_latencies, 0.95),
        "predicted_clearance_mean_m": _mean(predicted_clearance),
        "observed_clearance_mean_m": _mean(observed_clearance),
        "clearance_prediction_gap_mean_m": _mean(clearance_gaps),
        "clearance_overoptimism_max_m": _max(max(gap, 0.0) for gap in clearance_gaps),
        "predicted_visibility_mean": _mean(predicted_visibility),
        "observed_visibility_mean": _mean(observed_visibility),
        "visibility_prediction_gap_mean": _mean(visibility_gaps),
        "observation_age_max_steps": _max(observation_ages),
        "message_age_max_steps": _max(message_ages),
        "prediction_target_drift_observable": False,
    }


def _is_timeout(episode: Mapping[str, Any]) -> bool:
    return str(episode.get("termination_reason", "")) in {"timeout", "truncated"}


def classify_episode(
    episode: Mapping[str, Any],
    trace_summary: Mapping[str, Any],
    *,
    baseline_safe_capture: bool | None,
    variant: str,
) -> tuple[str, list[str]]:
    labels: list[str] = []
    if _as_bool(episode.get("collision")):
        labels.append("collision")
    if _as_bool(episode.get("boundary_violation")):
        labels.append("boundary_violation")
    if _as_bool(episode.get("pairwise_violation")):
        labels.append("pairwise_violation")
    unverified = int(episode.get("cbf_unverified_steps", 0))
    infeasible = int(episode.get("cbf_infeasible_steps", 0))
    controlled_abort = int(episode.get("cbf_controlled_abort_steps", 0)) > 0 or str(episode.get("termination_reason", "")) == "cbf_controlled_abort"
    if controlled_abort:
        labels.append("cbf_controlled_abort")
    if infeasible > 0 or unverified > 0 or int(trace_summary.get("cbf_unverified_steps_trace", 0)) > 0:
        labels.append("cbf_infeasible_or_unverified")
    if _is_timeout(episode):
        labels.append("timeout")
    if variant != "m0" and baseline_safe_capture is True and not _as_bool(episode.get("safe_capture")):
        labels.append("candidate_capture_regression")
    states = trace_summary.get("ledger_state_counts", {})
    if not _as_bool(episode.get("safe_capture")) and isinstance(states, Mapping):
        if int(states.get("trusted", 0)) > 0:
            labels.append("high_credit_failure")
        if int(states.get("fallback_nominal", 0)) > 0 or int(states.get("safe_hold", 0)) > 0:
            labels.append("low_credit_or_nominal_fallback")
    if (trace_summary.get("observation_age_max_steps") or 0.0) > 3.0 or (trace_summary.get("message_age_max_steps") or 0.0) > 3.0:
        labels.append("stale_observation")
    visible = _finite_float(episode.get("mean_visible_fraction"))
    if visible is not None and visible < 0.5:
        labels.append("visibility_degraded")
    if float(trace_summary.get("candidate_switch_rate", 0.0)) > 0.25:
        labels.append("candidate_oscillation")
    if (trace_summary.get("clearance_overoptimism_max_m") or 0.0) > 0.50:
        labels.append("clearance_prediction_gap")
    if abs(trace_summary.get("visibility_prediction_gap_mean") or 0.0) > 0.50:
        labels.append("visibility_prediction_gap")
    if not _as_bool(episode.get("safe_capture")) and not labels:
        labels.append("unresolved_non_capture")
    if _as_bool(episode.get("safe_capture")):
        primary = "safe_capture"
    else:
        priority = (
            "collision", "boundary_violation", "pairwise_violation", "cbf_controlled_abort",
            "cbf_infeasible_or_unverified", "timeout", "candidate_capture_regression",
            "high_credit_failure", "low_credit_or_nominal_fallback", "stale_observation",
            "visibility_degraded", "candidate_oscillation", "clearance_prediction_gap",
            "visibility_prediction_gap", "unresolved_non_capture",
        )
        primary = next((label for label in priority if label in labels), "unresolved_non_capture")
    return primary, labels


def _episode_row(
    run: Mapping[str, Any],
    episode_index: int,
    episode: Mapping[str, Any],
    trace_summary: Mapping[str, Any],
    *,
    baseline_safe_capture: bool | None,
) -> dict[str, Any]:
    primary, labels = classify_episode(
        episode,
        trace_summary,
        baseline_safe_capture=baseline_safe_capture,
        variant=str(run["variant"]),
    )
    row: dict[str, Any] = {
        "training_seed": int(run["seed"]),
        "variant": str(run["variant"]),
        "episode_index": int(episode_index),
        "episode_seed": int(episode["episode_seed"]),
        "layout_seed": int(episode.get("layout_seed", -1)),
        "layout_signature": str(episode.get("layout_signature", "")),
        "scenario": str(episode.get("scenario", "")),
        "observation_condition": str(episode.get("observation_condition", "")),
        "target_motion_mode": str(episode.get("target_motion_mode", "")),
        "obstacle_count": int(episode.get("obstacle_count", -1)),
        "safe_capture": _as_bool(episode.get("safe_capture")),
        "collision": _as_bool(episode.get("collision")),
        "boundary_violation": _as_bool(episode.get("boundary_violation")),
        "pairwise_violation": _as_bool(episode.get("pairwise_violation")),
        "termination_reason": str(episode.get("termination_reason", "")),
        "baseline_safe_capture": baseline_safe_capture,
        "primary_cause": primary,
        "diagnostic_labels": labels,
    }
    row.update(trace_summary)
    row["diagnostic_labels_json"] = json.dumps(labels, ensure_ascii=True, separators=(",", ":"))
    row["selected_candidate_indices_json"] = json.dumps(trace_summary["selected_candidate_indices"], separators=(",", ":"))
    row["ranking_modes_json"] = json.dumps(trace_summary["ranking_modes"], ensure_ascii=True, separators=(",", ":"))
    row["ledger_state_counts_json"] = json.dumps(trace_summary["ledger_state_counts"], ensure_ascii=True, separators=(",", ":"))
    row["cbf_fallback_mode_counts_json"] = json.dumps(trace_summary["cbf_fallback_mode_counts"], ensure_ascii=True, separators=(",", ":"))
    return row


def build_index(runs: list[dict[str, Any]], freeze_manifest: Mapping[str, Any]) -> dict[str, Any]:
    baseline_by_key: dict[tuple[int, int], bool] = {}
    for run in runs:
        if run["variant"] == "m0":
            for index, episode in run["episodes"].items():
                baseline_by_key[(int(run["seed"]), int(index))] = bool(episode["safe_capture"])
    rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    for run in runs:
        run_path = Path(str(run["path"]))
        source_episode_metadata = _read_episode_metadata(run_path / "episodes.csv")
        if sorted(source_episode_metadata) != sorted(run["episodes"]):
            raise ValueError(f"Source episode table does not match aggregate run: {run_path}")
        run_failure_count = 0
        for episode_index, episode in sorted(run["episodes"].items()):
            trace_path = run_path / "step_traces" / f"episode_{episode_index:04d}.jsonl"
            trace_summary = summarize_trace(read_trace(trace_path, episode_index))
            # The paired aggregator intentionally keeps only fields needed for
            # its statistical contract.  Merge the original CSV here so the
            # failure index retains layout, motion, visibility, and safety
            # context for replay triage.  Contract-checked outcome fields from
            # the aggregator remain authoritative.
            enriched_episode = dict(source_episode_metadata[episode_index])
            enriched_episode.update(episode)
            row = _episode_row(
                run,
                episode_index,
                enriched_episode,
                trace_summary,
                baseline_safe_capture=baseline_by_key.get((int(run["seed"]), int(episode_index))),
            )
            rows.append(row)
            if not row["safe_capture"]:
                run_failure_count += 1
        run_records.append(
            {
                "training_seed": int(run["seed"]),
                "variant": str(run["variant"]),
                "path": str(run_path),
                "summary_sha256": str(run["summary_sha256"]),
                "provenance_sha256": str(run["provenance_sha256"]),
                "manifest_sha256": str(run["manifest_sha256"]),
                "episode_count": len(run["episodes"]),
                "failure_count": run_failure_count,
            }
        )
    if not rows:
        raise ValueError("Failure index produced no episodes.")
    safe_rows = [row for row in rows if row["safe_capture"]]
    failures = [row for row in rows if not row["safe_capture"]]
    cause_counts = Counter(row["primary_cause"] for row in failures)
    label_counts = Counter(label for row in failures for label in row["diagnostic_labels"])
    by_variant: dict[str, dict[str, Any]] = {}
    for variant in sorted({str(row["variant"]) for row in rows}):
        subset = [row for row in rows if row["variant"] == variant]
        by_variant[variant] = {
            "episodes": len(subset),
            "safe_capture_count": sum(bool(row["safe_capture"]) for row in subset),
            "safe_capture_rate": float(np.mean([row["safe_capture"] for row in subset])),
            "primary_causes": dict(Counter(row["primary_cause"] for row in subset if not row["safe_capture"])),
            "high_credit_failure_count": sum("high_credit_failure" in row["diagnostic_labels"] for row in subset),
            "fallback_episode_count": sum("low_credit_or_nominal_fallback" in row["diagnostic_labels"] for row in subset),
            "mean_candidate_switch_rate": _mean(row["candidate_switch_rate"] for row in subset),
            "mean_cbf_correction_p95_mps": _mean(row["cbf_correction_p95_mps"] for row in subset),
        }
    by_condition: dict[str, dict[str, Any]] = {}
    for condition in sorted({str(row["observation_condition"]) for row in rows}):
        subset = [row for row in rows if row["observation_condition"] == condition]
        by_condition[condition] = {
            "episodes": len(subset),
            "safe_capture_rate": float(np.mean([row["safe_capture"] for row in subset])),
            "failure_count": sum(not row["safe_capture"] for row in subset),
            "primary_causes": dict(Counter(row["primary_cause"] for row in subset if not row["safe_capture"])),
        }
    return {
        "index_type": "jepa_safe_capture_v3_wp1_failure_index",
        "development_only": True,
        "locked_test_opened": False,
        "freeze_manifest": freeze_manifest.get("inputs", {}).get("next_phase_protocol", {}),
        "run_count": len(runs),
        "episode_count": len(rows),
        "safe_capture_count": len(safe_rows),
        "failure_count": len(failures),
        "safe_capture_rate": float(len(safe_rows) / len(rows)),
        "primary_cause_counts": dict(cause_counts),
        "diagnostic_label_counts": dict(label_counts),
        "by_variant": by_variant,
        "by_observation_condition": by_condition,
        "target_drift_observable": False,
        "target_drift_note": "Current P7 step traces do not expose offline target future labels; target drift is not inferred from proxy fields.",
        "runs": run_records,
        "rows": rows,
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "training_seed", "variant", "episode_index", "episode_seed", "layout_seed", "layout_signature",
        "scenario", "observation_condition", "target_motion_mode", "obstacle_count", "safe_capture",
        "collision", "boundary_violation", "pairwise_violation", "termination_reason", "baseline_safe_capture",
        "primary_cause", "diagnostic_labels_json", "trace_steps", "candidate_switch_count", "candidate_switch_rate",
        "non_nominal_selection_rate", "ranking_modes_json", "ledger_state_counts_json", "ledger_credit_mean",
        "ledger_credit_min", "cbf_fallback_mode_counts_json", "cbf_unverified_steps_trace", "cbf_infeasible_steps_trace",
        "cbf_timeout_steps_trace", "cbf_correction_mean_mps", "cbf_correction_p95_mps", "cbf_correction_max_mps",
        "cbf_latency_p95_ms", "predicted_clearance_mean_m", "observed_clearance_mean_m", "clearance_prediction_gap_mean_m",
        "clearance_overoptimism_max_m", "predicted_visibility_mean", "observed_visibility_mean",
        "visibility_prediction_gap_mean", "observation_age_max_steps", "message_age_max_steps",
        "prediction_target_drift_observable",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# WP1 Failure Index and Causal Replay Audit",
        "",
        "**Status:** development-only; `locked_test_opened=false`  ",
        f"**Runs:** {report['run_count']}  ",
        f"**Episodes:** {report['episode_count']}  ",
        f"**Safe capture:** {report['safe_capture_count']}/{report['episode_count']} = {report['safe_capture_rate']:.1%}  ",
        "**Scope:** existing P7 full development traces; no locked-test data was opened.",
        "",
        "## Primary Causes",
        "",
        "| Primary cause | Episodes |",
        "|---|---:|",
    ]
    for cause, count in sorted(report["primary_cause_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{cause}` | {count} |")
    lines.extend(["", "## Diagnostic Labels", "", "| Label | Episodes |", "|---|---:|"])
    for label, count in sorted(report["diagnostic_label_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{label}` | {count} |")
    lines.extend(["", "## By Variant", "", "| Variant | Episodes | Safe capture | Failure count | High-credit failures | Fallback episodes |", "|---|---:|---:|---:|---:|---:|"])
    for variant, values in sorted(report["by_variant"].items()):
        failure_count = values["episodes"] - values["safe_capture_count"]
        lines.append(
            f"| `{variant}` | {values['episodes']} | {values['safe_capture_rate']:.1%} | {failure_count} | {values['high_credit_failure_count']} | {values['fallback_episode_count']} |"
        )
    lines.extend(["", "## Interpretation and Limits", "", "- Primary outcome labels are derived from episode summary fields and trace solver states.", "- Diagnostic labels describe observed conditions and are not all causal proofs.", "- `target_drift_observable=false`: current traces do not contain offline future target labels, so target drift is not inferred from target clearance proxies.", "- Clearance/visibility gaps are timestamp-aligned diagnostics only; they are not safety certificates and do not replace CBF verification.", "- Every indexed episode retains the source run, episode seed, trace length, ledger state counts, selected candidates, CBF status, and termination reason in `failure_index.csv`.", "", "## Reproducibility Artifacts", "", "- `failure_index.json`", "- `failure_index.csv`", "- `report.md`", "- `provenance.json`", "- `tensorboard/`"])
    return "\n".join(lines) + "\n"


def write_tensorboard(report: Mapping[str, Any], logdir: Path) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard logdir: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/wp1_failure_index", json.dumps({"index_type": report["index_type"], "development_only": True, "locked_test_opened": False}, indent=2), 0)
        writer.add_text("Provenance/runs", json.dumps(report["runs"], indent=2), 0)
        writer.add_text("Provenance/limits", report["target_drift_note"], 0)
        writer.add_scalar("Episodes/total", float(report["episode_count"]), 0)
        writer.add_scalar("Episodes/safe_capture", float(report["safe_capture_count"]), 0)
        writer.add_scalar("Episodes/failure", float(report["failure_count"]), 0)
        writer.add_scalar("Episodes/safe_capture_rate", float(report["safe_capture_rate"]), 0)
        for cause, count in report["primary_cause_counts"].items():
            writer.add_scalar(f"Failure/primary/{cause}", float(count), 0)
        for label, count in report["diagnostic_label_counts"].items():
            writer.add_scalar(f"Failure/diagnostic/{label}", float(count), 0)
        for variant, values in report["by_variant"].items():
            writer.add_scalar(f"Variant/{variant}/safe_capture_rate", float(values["safe_capture_rate"]), 0)
            writer.add_scalar(f"Variant/{variant}/high_credit_failure_count", float(values["high_credit_failure_count"]), 0)
            writer.add_scalar(f"Variant/{variant}/fallback_episode_count", float(values["fallback_episode_count"]), 0)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required_text = {
        "Config/wp1_failure_index/text_summary",
        "Provenance/runs/text_summary",
        "Provenance/limits/text_summary",
    }
    missing = sorted(required_text.difference(tags.get("tensors", [])))
    if missing:
        raise ValueError(f"WP1 TensorBoard provenance is incomplete: {missing}")
    return {
        "logdir": str(logdir),
        "event_files": sorted(path.name for path in logdir.glob("events.out.tfevents.*")),
        "scalar_tag_count": len(tags.get("scalars", [])),
        "text_tag_count": len(tags.get("tensors", [])),
        "required_provenance": not missing,
    }


def main() -> None:
    args = parse_args()
    if not args.development_only:
        raise ValueError("WP1 requires --development-only.")
    freeze_path = args.freeze_manifest.resolve()
    freeze_manifest = _validate_freeze_manifest(freeze_path)
    expected_episodes = 20 if args.stage == "smoke" else 40
    runs = discover_runs(args.input_root.resolve(), stage=args.stage, expected_episodes=expected_episodes)
    report = build_index(runs, freeze_manifest)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "failure_index.csv", report["rows"])
    tensorboard = write_tensorboard(report, args.tensorboard_logdir)
    report["tensorboard"] = tensorboard
    report["provenance"] = {
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tensorboard": version("tensorboard"),
        "input_root": str(args.input_root.resolve()),
        "freeze_manifest": str(freeze_path),
        "freeze_manifest_sha256": sha256(freeze_path),
        "stage": args.stage,
        "command": " ".join(sys.argv),
        "source_hashes": {
            "scripts/index_jepa_safe_capture_failures.py": sha256(Path(__file__).resolve()),
            "scripts/aggregate_jepa_safe_capture_v2_paired.py": sha256(PROJECT_ROOT / "scripts/aggregate_jepa_safe_capture_v2_paired.py"),
        },
    }
    (output_dir / "failure_index.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output_dir / "provenance.json").write_text(json.dumps(report["provenance"], indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"episodes": report["episode_count"], "failures": report["failure_count"], "primary_causes": report["primary_cause_counts"], "tensorboard": tensorboard}, indent=2))


if __name__ == "__main__":
    main()
