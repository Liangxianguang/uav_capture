"""Audit the earliest CBF divergence and communication-age semantics for V21.

This is a read-only development audit.  It consumes the frozen V21 paired-smoke
failure index and the original step traces.  It does not re-simulate episodes,
read target ground truth, alter online decisions, or relax any safety gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
from torch.utils.tensorboard import SummaryWriter


SEEDS = (20260911, 20260912, 20260913)
VARIANTS = ("m0", "m3", "a1", "a2")
EXPECTED_EPISODES = 20
NEGATIVE_SLACK_TOLERANCE = 1e-8
MESSAGE_AGE_SATURATION = 60.0
TARGET_STALE_LIMIT = 45.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"Empty JSONL: {path}")
    return rows


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def finite_numbers(values: Any) -> list[float]:
    if not isinstance(values, (list, tuple)):
        return []
    result: list[float] = []
    for value in values:
        number = finite(value)
        if number is not None:
            result.append(number)
    return result


def min_finite_mapping(values: Any) -> tuple[str | None, float | None]:
    if not isinstance(values, Mapping):
        return None, None
    pairs = [(str(key), number) for key, value in values.items() if (number := finite(value)) is not None]
    if not pairs:
        return None, None
    return min(pairs, key=lambda item: (item[1], item[0]))


def constraint_category(name: str | None) -> str:
    if not name:
        return "unknown"
    lowered = name.lower()
    if lowered.startswith("pairwise"):
        return "pairwise"
    if lowered.startswith("obstacle"):
        return "obstacle"
    if lowered.startswith("boundary") or lowered.startswith("altitude"):
        return "boundary"
    if lowered.startswith("speed") or lowered.startswith("acceleration"):
        return "dynamic_limit"
    if "target" in lowered:
        return "target"
    return "state"


def action_matrix(row: Mapping[str, Any], key: str) -> np.ndarray | None:
    value = row.get(key)
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return array if array.size else None


def action_is_finite(row: Mapping[str, Any], key: str = "requested_action") -> bool:
    value = row.get(key)
    try:
        return bool(np.isfinite(np.asarray(value, dtype=np.float64)).all())
    except (TypeError, ValueError):
        return False


def candidate_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    ranking = row.get("candidate_ranking")
    if not isinstance(ranking, Mapping):
        return {
            "selected_index": None,
            "execution_mode": None,
            "fallback_reason": None,
            "eligible_count": None,
            "all_ineligible": None,
            "valid_count": None,
            "minimum_candidate_separation_m": None,
            "predicted_min_clearance_min_m": None,
            "candidate_eligibility_reasons": [],
        }
    eligible = ranking.get("eligible_mask")
    valid = ranking.get("valid_mask")
    predicted = finite_numbers(ranking.get("predicted_min_clearance_m"))
    reasons = ranking.get("candidate_eligibility_reasons")
    return {
        "selected_index": ranking.get("selected_index"),
        "execution_mode": ranking.get("execution_mode"),
        "fallback_reason": ranking.get("fallback_reason"),
        "eligible_count": sum(bool_value(value) for value in eligible) if isinstance(eligible, list) else None,
        "all_ineligible": all(not bool_value(value) for value in eligible) if isinstance(eligible, list) and eligible else None,
        "valid_count": sum(bool_value(value) for value in valid) if isinstance(valid, list) else None,
        "minimum_candidate_separation_m": finite(ranking.get("minimum_candidate_separation_m")),
        "predicted_min_clearance_min_m": min(predicted) if predicted else None,
        "candidate_eligibility_reasons": reasons if isinstance(reasons, list) else [],
    }


def cbf_failure(row: Mapping[str, Any]) -> bool:
    cbf = row.get("cbf")
    if not isinstance(cbf, Mapping):
        return False
    return (
        str(cbf.get("fallback_mode", "")) == "controlled_abort"
        or not bool_value(cbf.get("verified_feasible", True))
        or bool_value(cbf.get("infeasible", False))
        or bool_value(cbf.get("unverified", False))
    )


def earliest_divergence(trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failure_index = next((index for index, row in enumerate(trace) if cbf_failure(row)), None)
    if failure_index is None:
        return {"has_failure": False, "trace_steps": len(trace)}
    current = trace[failure_index]
    previous = trace[failure_index - 1] if failure_index > 0 else None
    cbf = current.get("cbf") if isinstance(current.get("cbf"), Mapping) else {}
    previous_cbf = previous.get("cbf") if isinstance(previous, Mapping) and isinstance(previous.get("cbf"), Mapping) else {}
    constraint_name, min_slack = min_finite_mapping(cbf.get("constraint_slacks"))
    negative = []
    if isinstance(cbf.get("constraint_slacks"), Mapping):
        negative = sorted(
            ((str(name), number) for name, value in cbf["constraint_slacks"].items() if (number := finite(value)) is not None and number < -NEGATIVE_SLACK_TOLERANCE),
            key=lambda item: (item[1], item[0]),
        )
    previous_constraint_name, previous_min_slack = min_finite_mapping(previous_cbf.get("constraint_slacks"))
    current_finite = action_is_finite(current)
    previous_verified = bool_value(previous_cbf.get("verified_feasible", True)) if previous is not None else None
    solver_status = str(cbf.get("solver_status", ""))
    if not current_finite:
        root_cause = "nonfinite_request"
    elif negative:
        root_cause = "cbf_constraint_infeasible"
    elif "solver_failure" in solver_status or (previous_verified is True and not bool_value(cbf.get("solver_success", True))):
        root_cause = "cbf_solver_failure_or_initialization"
    else:
        root_cause = "cbf_unverified_without_negative_slack"
    current_action = action_matrix(current, "requested_action")
    nominal_action = action_matrix(current, "reachable_nominal_action")
    nominal_match = bool(current_action is not None and nominal_action is not None and current_action.shape == nominal_action.shape and np.allclose(current_action, nominal_action, atol=1e-8, rtol=0.0))
    snapshot = candidate_snapshot(current)
    previous_snapshot = candidate_snapshot(previous) if previous is not None else {}
    observation = current.get("observation") if isinstance(current.get("observation"), Mapping) else {}
    return {
        "has_failure": True,
        "trace_steps": len(trace),
        "first_failure_row_index": failure_index,
        "first_failure_step": current.get("step"),
        "previous_step": previous.get("step") if previous is not None else None,
        "solver_status": solver_status,
        "solver_message": cbf.get("solver_message"),
        "fallback_mode": cbf.get("fallback_mode"),
        "verified_feasible": bool_value(cbf.get("verified_feasible", False)),
        "infeasible": bool_value(cbf.get("infeasible", False)),
        "timed_out": bool_value(cbf.get("timed_out", False)),
        "requested_action_finite": current_finite,
        "root_cause": root_cause,
        "constraint_category": constraint_category(negative[0][0] if negative else constraint_name),
        "minimum_constraint_name": constraint_name,
        "minimum_constraint_slack": min_slack,
        "negative_constraint_count": len(negative),
        "negative_constraints": [{"name": name, "slack": value} for name, value in negative[:8]],
        "previous_minimum_constraint_name": previous_constraint_name,
        "previous_minimum_constraint_slack": previous_min_slack,
        "previous_verified_feasible": previous_verified,
        "active_constraints": cbf.get("active_constraints", []),
        "task_constraint_slacks": cbf.get("task_constraint_slacks", {}),
        "action_correction_norm": finite(cbf.get("action_correction_norm")),
        "solve_latency_ms": finite(cbf.get("solve_latency_ms")),
        "state_safety_violation": bool_value(cbf.get("state_safety_violation", False)),
        "nominal_action_match": nominal_match,
        "candidate": snapshot,
        "previous_candidate": previous_snapshot,
        "target_observation_age_max_steps": max(finite_numbers(observation.get("target_observation_age_steps")), default=None),
        "message_age_max_steps": max(finite_numbers(observation.get("message_age_steps")), default=None),
        "target_visible_count": sum(bool_value(value) for value in observation.get("target_visible", [])) if isinstance(observation.get("target_visible"), list) else None,
    }


def age_episode_summary(trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    message_values: list[float] = []
    target_values: list[float] = []
    visible_count = 0
    visible_total = 0
    message_saturated_rows = 0
    target_stale_rows = 0
    saturation_with_visible_rows = 0
    input_output_age_changes = 0
    for row in trace:
        observation = row.get("observation") if isinstance(row.get("observation"), Mapping) else {}
        input_observation = row.get("input_observation") if isinstance(row.get("input_observation"), Mapping) else {}
        messages = finite_numbers(observation.get("message_age_steps"))
        targets = finite_numbers(observation.get("target_observation_age_steps"))
        message_values.extend(messages)
        target_values.extend(targets)
        visible = observation.get("target_visible")
        if isinstance(visible, list):
            visible_count += sum(bool_value(value) for value in visible)
            visible_total += len(visible)
            if messages and max(messages) >= MESSAGE_AGE_SATURATION and any(bool_value(value) for value in visible):
                saturation_with_visible_rows += 1
        if messages and max(messages) >= MESSAGE_AGE_SATURATION:
            message_saturated_rows += 1
        if targets and max(targets) > TARGET_STALE_LIMIT:
            target_stale_rows += 1
        input_messages = finite_numbers(input_observation.get("message_age_steps"))
        if input_messages and messages and input_messages != messages:
            input_output_age_changes += 1
    return {
        "trace_steps": len(trace),
        "message_age_max_steps": max(message_values, default=None),
        "target_observation_age_max_steps": max(target_values, default=None),
        "message_age_saturated": bool(message_values and max(message_values) >= MESSAGE_AGE_SATURATION),
        "target_observation_stale": bool(target_values and max(target_values) > TARGET_STALE_LIMIT),
        "message_saturated_rows": message_saturated_rows,
        "target_stale_rows": target_stale_rows,
        "saturation_with_target_visible_rows": saturation_with_visible_rows,
        "target_visible_fraction": (visible_count / visible_total) if visible_total else None,
        "input_output_age_change_rows": input_output_age_changes,
    }


def run_path(input_root: Path, seed: int, variant: str) -> Path:
    return (input_root / f"jepa_safe_capture_v21_smoke_{variant}_seed{seed}").resolve()


def validate_run(path: Path, seed: int, variant: str) -> dict[str, Any]:
    required = ("summary.json", "provenance.json", "scene_manifest.jsonl", "episodes.csv")
    for name in required:
        if not (path / name).is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    provenance = json_object(path / "provenance.json")
    if provenance.get("development_only") is not True or provenance.get("locked_test_opened") is not False:
        raise ValueError(f"Development boundary failed: {path}")
    declared = provenance.get("variant")
    if not isinstance(declared, Mapping) or declared.get("variant") != variant:
        raise ValueError(f"Variant provenance mismatch: {path}")
    if int(provenance.get("training_seed", -1)) != seed:
        raise ValueError(f"Seed provenance mismatch: {path}")
    traces = sorted((path / "step_traces").glob("episode_*.jsonl"))
    if len(traces) != EXPECTED_EPISODES:
        raise ValueError(f"Expected {EXPECTED_EPISODES} traces, found {len(traces)}: {path}")
    return {
        "path": str(path),
        "training_seed": seed,
        "variant": variant,
        "manifest_sha256": sha256(path / "scene_manifest.jsonl"),
        "summary_sha256": sha256(path / "summary.json"),
        "provenance_sha256": sha256(path / "provenance.json"),
        "traces": traces,
    }


def load_index(path: Path) -> dict[str, Any]:
    index = json_object(path)
    if index.get("index_type") != "jepa_safe_capture_v21_paired_smoke_failure_index":
        raise ValueError(f"Unexpected index_type: {index.get('index_type')}")
    if index.get("development_only") is not True or index.get("locked_test_opened") is not False:
        raise ValueError("Failure index is outside development-only boundary")
    rows = index.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Failure index rows must be a list")
    return index


def analyze(project_root: Path, input_root: Path, failure_index_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    index = load_index(failure_index_path)
    runs = {(seed, variant): validate_run(run_path(input_root, seed, variant), seed, variant) for seed in SEEDS for variant in VARIANTS}
    index_rows = [row for row in index["rows"] if isinstance(row, Mapping)]
    failures = [row for row in index_rows if str(row.get("primary_cause", "")) == "cbf_controlled_abort"]
    divergence_rows: list[dict[str, Any]] = []
    age_rows: list[dict[str, Any]] = []
    for (seed, variant), run in sorted(runs.items()):
        for trace_path in run["traces"]:
            trace = jsonl_objects(trace_path)
            episode_index = int(trace[0].get("episode_index", -1))
            if any(int(row.get("episode_index", -1)) != episode_index for row in trace):
                raise ValueError(f"Mixed episode trace: {trace_path}")
            age = age_episode_summary(trace)
            age_rows.append({"training_seed": seed, "variant": variant, "episode_index": episode_index, **age})
            matching = next((row for row in failures if int(row.get("training_seed", -1)) == seed and str(row.get("variant")) == variant and int(row.get("episode_index", -1)) == episode_index), None)
            divergence = earliest_divergence(trace)
            if matching is None:
                continue
            for key, expected in (("source_manifest_sha256", run["manifest_sha256"]), ("source_summary_sha256", run["summary_sha256"]), ("source_provenance_sha256", run["provenance_sha256"])):
                if str(matching.get(key, "")) != expected:
                    raise ValueError(f"Failure-index {key} mismatch for {seed}/{variant}/{episode_index}")
            if not divergence.get("has_failure"):
                raise ValueError(f"Failure index says CBF abort but trace has no CBF failure: {trace_path}")
            divergence_rows.append({
                "training_seed": seed,
                "variant": variant,
                "episode_index": episode_index,
                "episode_seed": matching.get("episode_seed"),
                "safe_capture": bool_value(matching.get("safe_capture")),
                "termination_reason": matching.get("termination_reason"),
                "layout_signature": matching.get("layout_signature"),
                "observation_condition": matching.get("observation_condition"),
                "target_motion_mode": matching.get("target_motion_mode"),
                "obstacle_count": matching.get("obstacle_count"),
                "source_trace": str(trace_path),
                "source_trace_sha256": sha256(trace_path),
                **divergence,
            })
    expected_keys = {(int(row.get("training_seed", -1)), str(row.get("variant")), int(row.get("episode_index", -1))) for row in failures}
    actual_keys = {(int(row["training_seed"]), str(row["variant"]), int(row["episode_index"])) for row in divergence_rows}
    if expected_keys != actual_keys:
        raise ValueError(f"Failure/index key mismatch: missing={sorted(expected_keys - actual_keys)} extra={sorted(actual_keys - expected_keys)}")
    source_manifest = {
        f"{seed}:{variant}": {key: value for key, value in run.items() if key != "traces"}
        for (seed, variant), run in sorted(runs.items())
    }
    root_causes = Counter(str(row["root_cause"]) for row in divergence_rows)
    categories = Counter(str(row["constraint_category"]) for row in divergence_rows)
    negative_count = sum(int(row.get("negative_constraint_count", 0)) > 0 for row in divergence_rows)
    nominal_matches = sum(bool_value(row.get("nominal_action_match")) for row in divergence_rows)
    all_ineligible = sum(row.get("candidate", {}).get("all_ineligible") is True for row in divergence_rows)
    saturated = sum(bool_value(row.get("message_age_saturated")) for row in age_rows)
    target_stale = sum(bool_value(row.get("target_observation_stale")) for row in age_rows)
    saturated_visible = sum(int(row.get("saturation_with_target_visible_rows", 0)) > 0 for row in age_rows)
    result: dict[str, Any] = {
        "stage": "jepa_safe_capture_v21_s3_divergence_and_age_audit",
        "input_format": "v21",
        "development_only": True,
        "locked_test_opened": False,
        "project_root": str(project_root.resolve()),
        "failure_index": str(failure_index_path.resolve()),
        "failure_index_sha256": sha256(failure_index_path),
        "run_count": len(runs),
        "episode_count": len(age_rows),
        "cbf_abort_episode_count": len(divergence_rows),
        "divergence_row_count": len(divergence_rows),
        "missing_divergence_count": len(expected_keys - actual_keys),
        "root_cause_counts": dict(sorted(root_causes.items())),
        "constraint_category_counts": dict(sorted(categories.items())),
        "negative_slack_episode_rate": negative_count / max(len(divergence_rows), 1),
        "nominal_action_match_rate": nominal_matches / max(len(divergence_rows), 1),
        "all_candidate_ineligible_rate_at_abort": all_ineligible / max(len(divergence_rows), 1),
        "age_semantics": {
            "message_age_saturated_episode_count": saturated,
            "message_age_saturated_episode_rate": saturated / max(len(age_rows), 1),
            "target_observation_stale_episode_count": target_stale,
            "target_observation_stale_episode_rate": target_stale / max(len(age_rows), 1),
            "saturated_with_target_visible_episode_count": saturated_visible,
            "saturated_with_target_visible_episode_rate": saturated_visible / max(len(age_rows), 1),
            "message_age_saturation_limit": MESSAGE_AGE_SATURATION,
            "target_stale_limit": TARGET_STALE_LIMIT,
            "interpretation": "message_age_saturated and target_observation_stale are separate observables; saturation is not a stale-target claim",
        },
        "safety_invariants": {
            "raw_unverified_executed_steps": int(index.get("raw_unverified_executed_steps", -1)),
            "collision_boundary_pairwise_zero": bool(index.get("safety_hard_gate")) and int(index.get("raw_unverified_executed_steps", -1)) == 0,
            "no_online_target_truth": True,
            "no_cbf_margin_change": True,
        },
        "source_manifest": source_manifest,
        "provenance": {
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "divergence_rows": divergence_rows,
        "age_rows": age_rows,
    }
    result["source_manifest_sha256"] = canonical_sha256(source_manifest)
    return result, divergence_rows, age_rows


def write_outputs(result: Mapping[str, Any], divergence_rows: Sequence[Mapping[str, Any]], age_rows: Sequence[Mapping[str, Any]], output_dir: Path, tensorboard_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    if tensorboard_dir.exists() and any(tensorboard_dir.iterdir()):
        raise FileExistsError(tensorboard_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "s3_divergence_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    divergence_fields = [
        "training_seed", "variant", "episode_index", "episode_seed", "safe_capture", "termination_reason", "layout_signature",
        "observation_condition", "target_motion_mode", "obstacle_count", "source_trace", "source_trace_sha256",
        "first_failure_row_index", "first_failure_step", "previous_step", "solver_status", "fallback_mode", "verified_feasible",
        "infeasible", "timed_out", "requested_action_finite", "root_cause", "constraint_category", "minimum_constraint_name",
        "minimum_constraint_slack", "negative_constraint_count", "previous_minimum_constraint_name", "previous_minimum_constraint_slack",
        "previous_verified_feasible", "action_correction_norm", "solve_latency_ms", "state_safety_violation", "nominal_action_match",
        "target_observation_age_max_steps", "message_age_max_steps", "target_visible_count",
    ]
    with (output_dir / "s3_divergence_cycles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=divergence_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(divergence_rows)
    age_fields = [
        "training_seed", "variant", "episode_index", "trace_steps", "message_age_max_steps", "target_observation_age_max_steps",
        "message_age_saturated", "target_observation_stale", "message_saturated_rows", "target_stale_rows",
        "saturation_with_target_visible_rows", "target_visible_fraction", "input_output_age_change_rows",
    ]
    with (output_dir / "s3_age_semantics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=age_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(age_rows)
    report_lines = [
        "# V21 S3 divergence and communication-age audit",
        "",
        "`development_only=true`; `locked_test_opened=false`. This is a read-only trace audit and does not alter CBF behavior.",
        "",
        f"CBF controlled-abort episodes: `{result['cbf_abort_episode_count']}`; earliest-divergence rows: `{result['divergence_row_count']}`.",
        f"Negative-slack episode rate at first failure: `{result['negative_slack_episode_rate']:.3f}`.",
        f"Nominal-action match rate at first failure: `{result['nominal_action_match_rate']:.3f}`.",
        "",
        "## Root-cause evidence",
        "",
        "| Root-cause label | Episodes |",
        "|---|---:|",
    ]
    for label, count in sorted(result["root_cause_counts"].items()):
        report_lines.append(f"| `{label}` | {count} |")
    report_lines += [
        "",
        "| First negative-slack category | Episodes |",
        "|---|---:|",
    ]
    for label, count in sorted(result["constraint_category_counts"].items()):
        report_lines.append(f"| `{label}` | {count} |")
    age = result["age_semantics"]
    report_lines += [
        "",
        "## Communication-age semantics",
        "",
        f"- `message_age_saturated`: `{age['message_age_saturated_episode_count']}/{result['episode_count']}` episodes.",
        f"- `target_observation_stale`: `{age['target_observation_stale_episode_count']}/{result['episode_count']}` episodes.",
        f"- saturation with at least one visible target: `{age['saturated_with_target_visible_episode_count']}/{result['episode_count']}` episodes.",
        "- These are separate fields; saturated communication age is not labeled as target-observation stale.",
        "",
        "## Safety boundary",
        "",
        "- No raw unverified action is counted as executed.",
        "- CBF margins, OOD/stale thresholds and controlled-abort semantics are unchanged.",
        "- No target future ground truth is consumed by this auditor.",
    ]
    (output_dir / "s3_divergence_audit.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    with SummaryWriter(log_dir=str(tensorboard_dir), flush_secs=1) as writer:
        writer.add_text("Provenance/config", json.dumps({key: result[key] for key in ("stage", "input_format", "failure_index_sha256", "source_manifest_sha256")}, sort_keys=True), 0)
        writer.add_text("Safety/invariants", json.dumps(result["safety_invariants"], sort_keys=True), 0)
        writer.add_text("Age/interpretation", result["age_semantics"]["interpretation"], 0)
        writer.add_scalar("Divergence/cbf_abort_episode_count", result["cbf_abort_episode_count"], 0)
        writer.add_scalar("Divergence/negative_slack_episode_rate", result["negative_slack_episode_rate"], 0)
        writer.add_scalar("Divergence/nominal_action_match_rate", result["nominal_action_match_rate"], 0)
        writer.add_scalar("Divergence/all_candidate_ineligible_rate", result["all_candidate_ineligible_rate_at_abort"], 0)
        for label, count in sorted(result["root_cause_counts"].items()):
            writer.add_scalar(f"Divergence/root_cause/{label}", count, 0)
        for label, count in sorted(result["constraint_category_counts"].items()):
            writer.add_scalar(f"Divergence/constraint_category/{label}", count, 0)
        writer.add_scalar("Age/message_age_saturated_episode_count", age["message_age_saturated_episode_count"], 0)
        writer.add_scalar("Age/target_observation_stale_episode_count", age["target_observation_stale_episode_count"], 0)
        writer.add_scalar("Age/saturated_with_target_visible_episode_count", age["saturated_with_target_visible_episode_count"], 0)
    result_with_tb = dict(result)
    result_with_tb["tensorboard"] = {"logdir": str(tensorboard_dir.resolve()), "event_files": sorted(path.name for path in tensorboard_dir.glob("events.out.tfevents.*"))}
    (output_dir / "s3_divergence_audit.json").write_text(json.dumps(result_with_tb, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "provenance.json").write_text(json.dumps({"stage": result["stage"], "input_format": result["input_format"], "development_only": True, "locked_test_opened": False, "failure_index_sha256": result["failure_index_sha256"], "source_manifest_sha256": result["source_manifest_sha256"], "git_revision": result["provenance"]["git_revision"]}, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--failure-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.development_only:
        raise ValueError("This auditor is development-only")
    project_root = args.project_root.resolve()
    result, divergence_rows, age_rows = analyze(project_root, args.input_root.resolve(), args.failure_index.resolve())
    write_outputs(result, divergence_rows, age_rows, args.output_dir.resolve(), args.tensorboard_dir.resolve())
    print(json.dumps({key: result[key] for key in ("stage", "cbf_abort_episode_count", "divergence_row_count", "root_cause_counts", "constraint_category_counts", "age_semantics")}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
