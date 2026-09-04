"""Compare CUDA and CPU rolling-horizon development replays.

The comparison keeps safety settlement as the primary equivalence criterion.
Small floating-point differences in prediction scores and solver latency are
allowed, while selected candidate, fallback, CBF verification, and all settled
safety fields are checked explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_TYPE = "jepa_safe_capture_v3_wp6_cpu_cuda_replay_audit"
SAFETY_FIELDS = (
    "safe_capture_success",
    "collision",
    "defender_boundary_violation",
    "target_boundary_violation",
    "pairwise_violation",
    "termination_reason",
)
DECISION_FIELDS = (
    "selected_index",
    "execution_mode",
    "valid_mask",
    "eligible_mask",
    "candidate_rejection_reasons",
)
CBF_FIELDS = ("verified_feasible", "infeasible", "timed_out", "fallback_mode")
NUMERIC_TRACE_FIELDS = ("desired_action", "reachable_nominal_action", "requested_action", "executed_action")


def sha256(path: Path) -> str:
    import hashlib

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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _numeric_equal(left: Any, right: Any, *, atol: float = 1e-5) -> bool:
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)) or len(left) != len(right):
            return False
        return all(_numeric_equal(a, b, atol=atol) for a, b in zip(left, right))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping) or set(left) != set(right):
            return False
        return all(_numeric_equal(left[key], right[key], atol=atol) for key in left)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        try:
            return bool(np.isclose(float(left), float(right), rtol=0.0, atol=atol, equal_nan=False))
        except (TypeError, ValueError):
            return False
    return left == right


def _read_episodes(path: Path) -> list[dict[str, str]]:
    with (path / "episodes.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["episode_index"]))
    return rows


def _read_trace(path: Path, episode_index: int) -> list[dict[str, Any]]:
    trace_path = path / "step_traces" / f"episode_{episode_index:04d}.jsonl"
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Trace contains a non-object row: {trace_path}")
    return rows


def _validate_run(path: Path) -> dict[str, Any]:
    summary = _json(path / "summary.json")
    provenance = _json(path / "provenance.json")
    metadata = summary.get("metadata", {})
    if not isinstance(metadata, Mapping) or metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError(f"Replay summary crossed development boundary: {path}")
    if provenance.get("development_only") is not True or provenance.get("locked_test_opened") is not False:
        raise ValueError(f"Replay provenance crossed development boundary: {path}")
    inputs = metadata.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise ValueError(f"Replay inputs are missing: {path}")
    manifest = path / "scene_manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    return {
        "path": str(path.resolve()),
        "summary": summary,
        "metadata": metadata,
        "inputs": dict(inputs),
        "manifest_sha256": sha256(manifest),
        "episodes": _read_episodes(path),
    }


def compare_replays(cuda_path: Path, cpu_path: Path) -> dict[str, Any]:
    cuda = _validate_run(cuda_path.resolve())
    cpu = _validate_run(cpu_path.resolve())
    if len(cuda["episodes"]) != len(cpu["episodes"]):
        raise ValueError("CPU and CUDA episode counts differ")
    input_fields = ("protocol_sha256", "environment_config_sha256", "actor_checkpoint_sha256", "jepa_checkpoint_sha256", "reliability_ledger_sha256", "scene_manifest_sha256")
    input_matches = {field: cuda["inputs"].get(field) == cpu["inputs"].get(field) for field in input_fields}
    input_matches["manifest_file_sha256"] = cuda["manifest_sha256"] == cpu["manifest_sha256"]
    if not all(input_matches.values()):
        raise ValueError(f"CPU/CUDA input provenance differs: {input_matches}")
    paired_episode_seeds = [left.get("episode_seed") == right.get("episode_seed") for left, right in zip(cuda["episodes"], cpu["episodes"])]
    if not all(paired_episode_seeds):
        raise ValueError("CPU/CUDA episode seeds differ")
    safety_equal = 0
    decision_equal = 0
    cbf_equal = 0
    numeric_equal = 0
    step_total = 0
    step_decision_equal = 0
    step_cbf_equal = 0
    step_numeric_equal = 0
    raw_unverified_executed = 0
    rejection_reason_steps = 0
    cpu_raw_unverified_executed = 0
    episode_records: list[dict[str, Any]] = []
    for left_episode, right_episode in zip(cuda["episodes"], cpu["episodes"]):
        episode_index = int(left_episode["episode_index"])
        safety_ok = all(str(left_episode.get(field)) == str(right_episode.get(field)) for field in SAFETY_FIELDS)
        decision_keys = ("selected_candidate_indices", "termination_reason")
        decision_ok = all(str(left_episode.get(field)) == str(right_episode.get(field)) for field in decision_keys)
        left_trace = _read_trace(cuda_path, episode_index)
        right_trace = _read_trace(cpu_path, episode_index)
        if len(left_trace) != len(right_trace):
            raise ValueError(f"Trace length differs for episode {episode_index}")
        episode_step_decision = episode_step_cbf = episode_step_numeric = 0
        for left_step, right_step in zip(left_trace, right_trace):
            step_total += 1
            left_rank = left_step.get("candidate_ranking")
            right_rank = right_step.get("candidate_ranking")
            step_decision = False
            if isinstance(left_rank, Mapping) and isinstance(right_rank, Mapping):
                step_decision = all(left_rank.get(field) == right_rank.get(field) for field in DECISION_FIELDS)
                if "candidate_rejection_reasons" in left_rank and "candidate_rejection_reasons" in right_rank:
                    rejection_reason_steps += 1
            elif left_rank is None and right_rank is None:
                step_decision = True
            left_cbf = left_step.get("cbf")
            right_cbf = right_step.get("cbf")
            step_cbf = isinstance(left_cbf, Mapping) and isinstance(right_cbf, Mapping) and all(left_cbf.get(field) == right_cbf.get(field) for field in CBF_FIELDS)
            step_numeric = all(_numeric_equal(left_step.get(field), right_step.get(field), atol=1e-4) for field in NUMERIC_TRACE_FIELDS)
            if isinstance(left_cbf, Mapping) and not _as_bool(left_cbf.get("verified_feasible")):
                if _numeric_equal(left_step.get("requested_action"), left_step.get("executed_action"), atol=1e-10):
                    raw_unverified_executed += 1
            if isinstance(right_cbf, Mapping) and not _as_bool(right_cbf.get("verified_feasible")):
                if _numeric_equal(right_step.get("requested_action"), right_step.get("executed_action"), atol=1e-10):
                    cpu_raw_unverified_executed += 1
            step_decision_equal += int(step_decision)
            step_cbf_equal += int(step_cbf)
            step_numeric_equal += int(step_numeric)
            episode_step_decision += int(step_decision)
            episode_step_cbf += int(step_cbf)
            episode_step_numeric += int(step_numeric)
        safety_equal += int(safety_ok)
        decision_equal += int(decision_ok)
        numeric_ok = all(_numeric_equal(left_episode.get(field), right_episode.get(field), atol=1e-4) for field in ("mean_cbf_action_correction_norm", "mean_cbf_solve_latency_ms"))
        numeric_equal += int(numeric_ok)
        cbf_equal += int(str(left_episode.get("cbf_unverified_steps")) == str(right_episode.get("cbf_unverified_steps")))
        episode_records.append({"episode_index": episode_index, "safety_equal": safety_ok, "decision_equal": decision_ok, "trace_steps": len(left_trace), "trace_decision_equal": episode_step_decision == len(left_trace), "trace_cbf_equal": episode_step_cbf == len(left_trace), "trace_numeric_equal": episode_step_numeric == len(left_trace)})
    cuda_overall = cuda["summary"].get("overall", {})
    cpu_overall = cpu["summary"].get("overall", {})
    same_safety = safety_equal == len(cuda["episodes"])
    same_decisions = decision_equal == len(cuda["episodes"]) and step_decision_equal == step_total
    both_safe = all(int(cuda_overall.get(field, 0)) == 0 and int(cpu_overall.get(field, 0)) == 0 for field in ("collision_count", "boundary_violation_count", "pairwise_violation_count"))
    same_decisions = decision_equal == len(cuda["episodes"]) and step_decision_equal == step_total
    classification = (
        "cpu_cuda_safety_and_decision_equivalent"
        if same_safety and same_decisions and both_safe
        else "cpu_cuda_safety_settlement_equivalent_decision_drift"
        if same_safety and both_safe
        else "cpu_cuda_safety_settlement_requires_review"
    )
    return {
        "audit_type": AUDIT_TYPE,
        "development_only": True,
        "locked_test_opened": False,
        "cuda_run": cuda["path"],
        "cpu_run": cpu["path"],
        "episode_count": len(cuda["episodes"]),
        "paired_episode_seeds": True,
        "input_matches": input_matches,
        "episode_records": episode_records,
        "counts": {"safety_equal": safety_equal, "decision_equal": decision_equal, "cbf_equal": cbf_equal, "numeric_equal": numeric_equal, "step_total": step_total, "step_decision_equal": step_decision_equal, "step_cbf_equal": step_cbf_equal, "step_numeric_equal": step_numeric_equal, "raw_unverified_executed_cuda": raw_unverified_executed, "raw_unverified_executed_cpu": cpu_raw_unverified_executed, "rejection_reason_steps": rejection_reason_steps},
        "gates": {
            "development_boundary_both": True,
            "input_provenance_equal": all(input_matches.values()),
            "paired_episode_seeds": True,
            "settled_safety_outcomes_equal": same_safety,
            "settled_safety_zero_in_both": both_safe,
            "candidate_decisions_equal": same_decisions,
            "cbf_verification_counts_equal": cbf_equal == len(cuda["episodes"]),
            "raw_unverified_execution_zero_in_both": raw_unverified_executed == 0 and cpu_raw_unverified_executed == 0,
            "candidate_rejection_reason_schema_present": rejection_reason_steps == step_total,
            "p95_cbf_latency_under_100ms_both": float(cuda_overall.get("max_cbf_p95_solve_latency_ms", 0.0)) <= 100.0 and float(cpu_overall.get("max_cbf_p95_solve_latency_ms", 0.0)) <= 100.0,
        },
        "classification": classification,
        "provenance": {"git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(), "python": platform.python_version(), "numpy": np.__version__},
    }


def _write(report: dict[str, Any], output_dir: Path, tensorboard_dir: Path) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    if tensorboard_dir.exists() and any(tensorboard_dir.iterdir()):
        raise FileExistsError(tensorboard_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "device_replay_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    lines = ["# WP-6 CPU/CUDA Replay Audit", "", f"**Classification:** `{report['classification']}`", "", "| Gate | Result |", "|---|---:|"]
    lines.extend(f"| `{name}` | {str(value).lower()} |" for name, value in report["gates"].items())
    lines.extend(["", f"Safety-equal episodes: {report['counts']['safety_equal']}/{report['episode_count']}", f"Decision-equal trace steps: {report['counts']['step_decision_equal']}/{report['counts']['step_total']}", f"CBF-equal trace steps: {report['counts']['step_cbf_equal']}/{report['counts']['step_total']}"])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with SummaryWriter(log_dir=str(tensorboard_dir), flush_secs=1) as writer:
        writer.add_text("Config/device_replay", json.dumps({"audit_type": AUDIT_TYPE, "development_only": True, "locked_test_opened": False}, indent=2), 0)
        writer.add_text("Provenance/input_matches", json.dumps(report["input_matches"], indent=2), 0)
        writer.add_text("Gates/status", json.dumps(report["gates"], indent=2), 0)
        writer.add_scalar("Replay/safety_equal_episodes", report["counts"]["safety_equal"], 0)
        writer.add_scalar("Replay/decision_equal_steps", report["counts"]["step_decision_equal"], 0)
        writer.add_scalar("Replay/cbf_equal_steps", report["counts"]["step_cbf_equal"], 0)
        writer.add_scalar("Replay/numeric_equal_steps", report["counts"]["step_numeric_equal"], 0)
    accumulator = EventAccumulator(str(tensorboard_dir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required = {"Config/device_replay/text_summary", "Provenance/input_matches/text_summary", "Gates/status/text_summary"}
    missing = sorted(required.difference(tags.get("tensors", [])))
    events = sorted(path.name for path in tensorboard_dir.glob("events.out.tfevents.*"))
    if missing or not events:
        raise ValueError(f"Device replay TensorBoard validation failed: missing={missing}, events={events}")
    report["tensorboard"] = {"logdir": str(tensorboard_dir), "event_files": events, "scalar_tag_count": len(tags.get("scalars", [])), "text_tag_count": len(tags.get("tensors", [])), "required_provenance": True}
    (output_dir / "device_replay_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    manifest = {str(path.relative_to(output_dir)).replace("\\", "/"): sha256(path) for path in sorted(output_dir.rglob("*")) if path.is_file() and path.name != "hash_manifest.json"}
    (output_dir / "hash_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-run", type=Path, required=True)
    parser.add_argument("--cpu-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    args = parser.parse_args()
    if not args.development_only:
        raise ValueError("Device replay audit requires --development-only")
    report = compare_replays(args.cuda_run, args.cpu_run)
    _write(report, args.output_dir, args.tensorboard_logdir)
    print(json.dumps({"classification": report["classification"], "gates": report["gates"], "tensorboard": report["tensorboard"]}, indent=2))


if __name__ == "__main__":
    main()
