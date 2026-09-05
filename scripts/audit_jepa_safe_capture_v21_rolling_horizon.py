"""Audit V21 rolling-horizon replay and Joint CBF execution invariants.

The input runs are already-produced development traces.  This audit never
changes their outcomes or uses target truth online.  It checks long-sequence
coverage, first-step replan metadata, trace safety fields, deterministic
repeat equivalence, and latency quantiles, then records the result in a new
TensorBoard logdir.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LATENCY_FIELDS = (
    "actor", "candidate_generation", "jepa_inference", "ledger_route",
    "ranker_compute", "rank_total", "cbf_filter_wall", "cbf_solver",
    "env_step", "cycle_total",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite_nonnegative(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number >= 0.0 else None


def _quantiles(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "max_ms": float(np.max(array)),
    }


def _read_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    summary = _json(run_dir / "summary.json")
    provenance = _json(run_dir / "provenance.json")
    metadata = summary.get("metadata")
    overall = summary.get("overall")
    if not isinstance(metadata, Mapping) or not isinstance(overall, Mapping):
        raise ValueError(f"Invalid replay summary: {run_dir}")
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError(f"Run crossed development boundary: {run_dir}")
    if provenance.get("development_only") is not True or provenance.get("locked_test_opened") is not False:
        raise ValueError(f"Provenance crossed development boundary: {run_dir}")
    episodes_path = run_dir / "episodes.csv"
    with episodes_path.open("r", newline="", encoding="utf-8") as handle:
        episodes = list(csv.DictReader(handle))
    trace_paths = sorted((run_dir / "step_traces").glob("episode_*.jsonl"))
    if not episodes or len(trace_paths) != len(episodes):
        raise ValueError(f"Episode/trace mismatch: {run_dir}")
    traces: list[dict[str, Any]] = []
    for trace_path in trace_paths:
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Trace row is not an object: {trace_path}")
                traces.append(row)
    return {"path": str(run_dir), "summary": summary, "metadata": metadata, "overall": overall, "episodes": episodes, "traces": traces}


def _audit_run(run: Mapping[str, Any], *, minimum_cycles: int) -> dict[str, Any]:
    metadata = run["metadata"]
    overall = run["overall"]
    traces = run["traces"]
    missing_latency = 0
    bad_latency = 0
    missing_required = 0
    raw_unverified = 0
    unverified_equal_requested = 0
    cbf_failures_without_fallback = 0
    first_step_replan_flags = 0
    ranking_steps = 0
    cycle_values: list[float] = []
    stage_values: dict[str, list[float]] = {field: [] for field in LATENCY_FIELDS}
    for trace in traces:
        required = ("candidate_ranking", "cbf", "executed_action", "raw_unverified_executed", "latency_ms")
        missing_required += int(any(field not in trace for field in required))
        ranking = trace.get("candidate_ranking")
        if isinstance(ranking, Mapping):
            ranking_steps += 1
        if trace.get("raw_unverified_executed") is True:
            raw_unverified += 1
        cbf = trace.get("cbf")
        if isinstance(cbf, Mapping):
            verified = _bool(cbf.get("verified_feasible"))
            fallback = str(cbf.get("fallback_mode", "none"))
            if not verified:
                if fallback in {"none", ""}:
                    cbf_failures_without_fallback += 1
                requested = trace.get("requested_action")
                executed = trace.get("executed_action")
                if requested == executed:
                    unverified_equal_requested += 1
        contract = metadata.get("candidate_contract", {})
        if isinstance(contract, Mapping) and contract.get("execute_first_step_then_replan") is True:
            first_step_replan_flags += 1
        latency = trace.get("latency_ms")
        if not isinstance(latency, Mapping):
            missing_latency += 1
            continue
        for field in LATENCY_FIELDS:
            value = _finite_nonnegative(latency.get(field))
            if value is None:
                bad_latency += 1
            else:
                stage_values[field].append(value)
        if (value := _finite_nonnegative(latency.get("cycle_total"))) is not None:
            cycle_values.append(value)
    control_cycles = int(overall.get("control_cycles", -1))
    episode_safe = int(overall.get("safe_capture_count", -1))
    safety_zero = all(int(overall.get(field, 0)) == 0 for field in ("collision_count", "boundary_violation_count", "pairwise_violation_count"))
    gates = {
        "development_only": True,
        "locked_test_not_opened": True,
        "minimum_cycle_coverage": control_cycles >= minimum_cycles,
        "summary_trace_cycle_count_match": control_cycles == len(traces),
        "episode_count_trace_file_match": len(run["episodes"]) == len([p for p in (Path(run["path"]) / "step_traces").glob("episode_*.jsonl")]),
        "required_trace_fields_present": missing_required == 0,
        "ranking_trace_present": ranking_steps == len(traces),
        "first_step_replan_contract": first_step_replan_flags == len(traces),
        "latency_fields_present": missing_latency == 0 and bad_latency == 0 and all(len(values) == len(traces) for values in stage_values.values()),
        "safe_settlement_hard_gates_zero": safety_zero,
        "raw_unverified_zero": raw_unverified == 0 and int(overall.get("raw_unverified_executed_steps", -1)) == 0,
        "cbf_failures_have_fallback": cbf_failures_without_fallback == 0,
        "nonfinite_unverified_not_executed": unverified_equal_requested == 0,
        "cycle_p95_under_100ms": _quantiles(cycle_values)["p95_ms"] <= 100.0,
    }
    return {
        "run_dir": run["path"],
        "training_seed": metadata.get("training_seed"),
        "variant": metadata.get("variant"),
        "episodes": len(run["episodes"]),
        "safe_capture_count": episode_safe,
        "safe_capture_rate": float(overall.get("safe_capture_rate", 0.0)),
        "control_cycles": control_cycles,
        "trace_rows": len(traces),
        "ranking_steps": ranking_steps,
        "raw_unverified_trace_count": raw_unverified,
        "cbf_failures_without_fallback": cbf_failures_without_fallback,
        "stage_latency": {field: _quantiles(values) for field, values in stage_values.items()},
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "summary_sha256": sha256(Path(run["path"]) / "summary.json"),
        "provenance_sha256": sha256(Path(run["path"]) / "provenance.json"),
        "protocol_sha256": metadata.get("inputs", {}).get("protocol_sha256"),
        "ledger_sha256": metadata.get("inputs", {}).get("reliability_ledger_sha256"),
        "scene_manifest_sha256": metadata.get("inputs", {}).get("scene_manifest_sha256"),
    }


_NONDETERMINISTIC_TRACE_FIELDS = {
    "latency_ms",
    "jepa_inference_latency_ms",
    "ledger_route_latency_ms",
    "ranker_compute_latency_ms",
    "rank_total_latency_ms",
    "solve_latency_ms",
}


def _deterministic_value(value: Any) -> Any:
    """Remove wall-clock measurements before comparing replay semantics."""

    if isinstance(value, Mapping):
        return {
            str(key): _deterministic_value(item)
            for key, item in value.items()
            if str(key) not in _NONDETERMINISTIC_TRACE_FIELDS
        }
    if isinstance(value, list):
        return [_deterministic_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_deterministic_value(item) for item in value)
    return value


def _compare_runs(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_traces = left["traces"]
    right_traces = right["traces"]
    fields = ("requested_action", "executed_action", "raw_unverified_executed", "candidate_ranking", "cbf")
    differences = 0
    if len(left_traces) != len(right_traces):
        return {"trace_rows_equal": False, "compared_rows": min(len(left_traces), len(right_traces)), "field_difference_count": None, "passed": False}
    for a, b in zip(left_traces, right_traces):
        for field in fields:
            if _deterministic_value(a.get(field)) != _deterministic_value(b.get(field)):
                differences += 1
    return {"trace_rows_equal": True, "compared_rows": len(left_traces), "field_difference_count": differences, "passed": differences == 0}


def _write_tensorboard(logdir: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite TensorBoard directory: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/rolling_horizon", json.dumps(report["policy"], indent=2, sort_keys=True), 0)
        writer.add_text("Provenance/runs", json.dumps(report["runs"], indent=2, sort_keys=True), 0)
        writer.add_text("Replay/repeat_comparison", json.dumps(report["repeat_comparison"], indent=2), 0)
        writer.add_text("Gates/status", json.dumps(report["gates"], indent=2), 0)
        for row in report["runs"]:
            prefix = str(row["training_seed"])
            writer.add_scalar(f"Safety/{prefix}/safe_capture", row["safe_capture_rate"], 0)
            writer.add_scalar(f"Replay/{prefix}/control_cycles", row["control_cycles"], 0)
            writer.add_scalar(f"Safety/{prefix}/raw_unverified", row["raw_unverified_trace_count"], 0)
            for field, values in row["stage_latency"].items():
                writer.add_scalar(f"Latency/{prefix}/{field}/p50_ms", values["p50_ms"], 0)
                writer.add_scalar(f"Latency/{prefix}/{field}/p95_ms", values["p95_ms"], 0)
                writer.add_scalar(f"Latency/{prefix}/{field}/p99_ms", values["p99_ms"], 0)
            for name, passed in row["gates"].items():
                writer.add_scalar(f"Audit/{prefix}/{name}", float(bool(passed)), 0)
        for name, passed in report["gates"].items():
            writer.add_scalar(f"Audit/aggregate/{name}", float(bool(passed)), 0)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required_text = {"Config/rolling_horizon/text_summary", "Provenance/runs/text_summary", "Replay/repeat_comparison/text_summary", "Gates/status/text_summary"}
    # The aggregate gate names are the keys written below.  Keep this check
    # bound to those names rather than to per-run aliases; older TensorBoard
    # versions otherwise make a valid event file look incomplete.
    required_scalars = {"Audit/aggregate/at_least_100_cycles", "Audit/aggregate/all_run_gates_pass"}
    missing = sorted(required_text.difference(tags.get("tensors", []))) + sorted(required_scalars.difference(tags.get("scalars", [])))
    events = sorted(path.name for path in logdir.glob("events.out.tfevents.*"))
    if missing or not events:
        raise ValueError(f"Rolling TensorBoard validation failed: missing={missing}, events={events}")
    return {"logdir": str(logdir), "event_files": events, "scalar_tag_count": len(tags.get("scalars", [])), "text_tag_count": len(tags.get("tensors", []))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--minimum-cycles", type=int, default=100)
    parser.add_argument("--hard-context-cycles", type=int, default=500)
    parser.add_argument("--development-only", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.development_only:
        raise ValueError("V21 rolling audit requires --development-only")
    if args.minimum_cycles <= 0 or args.hard_context_cycles < args.minimum_cycles:
        raise ValueError("Invalid cycle thresholds")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    if len(args.run) < 2:
        raise ValueError("Provide at least two repeated runs for deterministic comparison")
    runs = [_read_run(path) for path in args.run]
    rows = [_audit_run(run, minimum_cycles=args.minimum_cycles) for run in runs]
    repeat_comparison = _compare_runs(runs[0], runs[1])
    cycle_max = max(row["control_cycles"] for row in rows)
    gates = {
        "development_only": True,
        "locked_test_not_opened": True,
        "repeat_count_at_least_two": len(runs) >= 2,
        "all_run_gates_pass": bool(all(row["all_gates_pass"] for row in rows)),
        "at_least_100_cycles": cycle_max >= args.minimum_cycles,
        # A long trace is not evidence that hard-context examples were
        # covered.  This gate only records total control-cycle coverage; the
        # report explicitly leaves hard-context coverage unclaimed.
        "at_least_500_control_cycles": cycle_max >= args.hard_context_cycles,
        "repeat_decision_trace_equal": bool(repeat_comparison["passed"]),
        "repeat_trace_rows_equal": bool(repeat_comparison["trace_rows_equal"]),
        "safe_capture_is_episode_level_only": True,
    }
    report: dict[str, Any] = {
        "audit_type": "jepa_safe_capture_v21_rolling_horizon",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "development_only": True,
        "locked_test_opened": False,
        "policy": {
            "minimum_cycles": args.minimum_cycles,
            "hard_context_cycles": args.hard_context_cycles,
            "hard_context_coverage": "not_inferred_from_total_control_cycles",
            "execute_first_step_then_replan": True,
            "world_model_role": "candidate_trajectory_evaluator_only",
            "cbf_final_safety_filter": True,
            "raw_unverified_allowed": False,
            "mean_capture_time_is_diagnostic_only": True,
        },
        "runs": rows,
        "repeat_comparison": repeat_comparison,
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "provenance": {
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
            "source_sha256": sha256(Path(__file__).resolve()),
        },
    }
    report["tensorboard"] = _write_tensorboard(args.tensorboard_logdir.resolve(), report)
    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    (args.output_dir.resolve() / "rolling_horizon_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {
        "audit_script": report["provenance"]["source_sha256"],
        "runs": {str(row["training_seed"]): {"summary": row["summary_sha256"], "provenance": row["provenance_sha256"], "scene_manifest": row["scene_manifest_sha256"], "protocol": row["protocol_sha256"], "ledger": row["ledger_sha256"]} for row in rows},
    }
    (args.output_dir.resolve() / "hash_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# V21 Rolling-Horizon / Joint CBF Audit", "", f"All gates pass: `{report['all_gates_pass']}`.", "", "| Run | Episodes | Cycles | Safe capture | Raw unverified | Cycle p95 ms | Run gates |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['run_dir']} | {row['episodes']} | {row['control_cycles']} | {row['safe_capture_rate']:.3f} | {row['raw_unverified_trace_count']} | {row['stage_latency']['cycle_total']['p95_ms']:.3f} | {row['all_gates_pass']} |")
    lines.extend(["", f"Repeat decision trace equal: `{repeat_comparison['passed']}`.", "", "The 500-cycle field is total control-cycle coverage only; hard-context coverage is not inferred from cycle count.", "", "This is a development replay audit, not a safe-capture improvement claim."])
    (args.output_dir.resolve() / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"all_gates_pass": report["all_gates_pass"], "gates": gates, "repeat_comparison": repeat_comparison, "tensorboard": report["tensorboard"]}, indent=2))


if __name__ == "__main__":
    main()
