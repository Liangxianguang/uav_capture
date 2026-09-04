"""Audit full-chain latency and queue-age instrumentation for a development run.

The audit is read-only with respect to the episode traces.  It validates the
trace schema, checks that every control cycle has a finite latency breakdown,
and writes a separate TensorBoard audit.  Runtime measurements are diagnostic
only; they never change action selection or safe-capture settlement.
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
AUDIT_TYPE = "jepa_safe_capture_v2_full_chain_latency_audit"
LATENCY_STAGES = (
    "actor",
    "candidate_generation",
    "jepa_inference",
    "ledger_route",
    "ranker_compute",
    "rank_total",
    "cbf_filter_wall",
    "cbf_solver",
    "env_step",
    "cycle_total",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _finite_nonnegative(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric.") from error
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return number


def _stats(values: list[float], *, suffix: str = "ms") -> dict[str, float | int]:
    if not values:
        return {"count": 0, f"mean_{suffix}": 0.0, f"p50_{suffix}": 0.0, f"p95_{suffix}": 0.0, f"p99_{suffix}": 0.0, f"max_{suffix}": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        f"mean_{suffix}": float(np.mean(array)),
        f"p50_{suffix}": float(np.percentile(array, 50)),
        f"p95_{suffix}": float(np.percentile(array, 95)),
        f"p99_{suffix}": float(np.percentile(array, 99)),
        f"max_{suffix}": float(np.max(array)),
    }


def _read_trace(run_dir: Path) -> list[dict[str, Any]]:
    trace_dir = run_dir / "step_traces"
    files = sorted(trace_dir.glob("episode_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No step traces found: {trace_dir}")
    records: list[dict[str, Any]] = []
    for path in files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Trace record is not an object: {path}:{line_number}")
            value["_trace_file"] = path.name
            records.append(value)
    return records


def _episode_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "episodes.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"episodes.csv is empty: {path}")
    return rows


def audit_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    summary = _load_json(run_dir / "summary.json")
    provenance = _load_json(run_dir / "provenance.json")
    metadata = summary.get("metadata")
    overall = summary.get("overall")
    if not isinstance(metadata, Mapping) or not isinstance(overall, Mapping):
        raise ValueError(f"Invalid paired evaluator summary: {run_dir}")
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError(f"Run crossed the development boundary: {run_dir}")
    if provenance.get("development_only") is not True or provenance.get("locked_test_opened") is not False:
        raise ValueError(f"Run provenance crossed the development boundary: {run_dir}")
    if int(metadata.get("trace_schema_version", -1)) != 2:
        raise ValueError("Full-chain latency audit requires trace_schema_version=2.")
    latency_contract = metadata.get("latency_contract")
    if not isinstance(latency_contract, Mapping):
        raise ValueError("Run is missing latency_contract metadata.")
    declared_stages = tuple(latency_contract.get("per_step_fields", ()))
    if declared_stages != LATENCY_STAGES:
        raise ValueError(f"Latency stage contract mismatch: {declared_stages}")

    records = _read_trace(run_dir)
    episode_rows = _episode_rows(run_dir)
    stage_values: dict[str, list[float]] = {stage: [] for stage in LATENCY_STAGES}
    queue_values: list[float] = []
    missing_latency = 0
    missing_queue_age = 0
    raw_trace_count = 0
    ranking_steps = 0
    for record in records:
        latency = record.get("latency_ms")
        if not isinstance(latency, Mapping):
            missing_latency += 1
        else:
            for stage in LATENCY_STAGES:
                if stage not in latency:
                    missing_latency += 1
                    continue
                stage_values[stage].append(_finite_nonnegative(latency[stage], f"latency_ms.{stage}"))
        input_observation = record.get("input_observation")
        if not isinstance(input_observation, Mapping) or "queue_age_steps" not in input_observation:
            missing_queue_age += 1
        else:
            queue_values.append(_finite_nonnegative(input_observation["queue_age_steps"], "queue_age_steps"))
        raw_flag = record.get("raw_unverified_executed")
        if bool(raw_flag):
            raw_trace_count += 1
        if isinstance(record.get("candidate_ranking"), Mapping):
            ranking_steps += 1

    expected_cycles = int(overall.get("control_cycles", -1))
    summary_cycles_match = expected_cycles == len(records)
    rows_cycles_match = all(int(row.get("control_cycle_count", -1)) == len([item for item in records if int(item.get("episode_index", -1)) == int(row.get("episode_index", -2))]) for row in episode_rows)
    summary_raw = int(overall.get("raw_unverified_executed_steps", -1))
    raw_match = summary_raw == raw_trace_count
    latency = {stage: _stats(values) for stage, values in stage_values.items()}
    cycle_p95 = float(latency["cycle_total"]["p95_ms"])
    trace_files = sorted((run_dir / "step_traces").glob("episode_*.jsonl"))
    gates = {
        "development_only": True,
        "locked_test_not_opened": True,
        "trace_schema_version": True,
        "latency_contract": True,
        "episode_csv_present": len(episode_rows) == int(overall.get("episodes", -1)),
        "trace_files_match_episodes": len(trace_files) == len(episode_rows),
        "control_cycle_count_matches_summary": summary_cycles_match,
        "control_cycle_count_matches_rows": rows_cycles_match,
        "all_latency_fields_present": missing_latency == 0 and all(len(values) == len(records) for values in stage_values.values()),
        "all_latency_values_finite_nonnegative": True,
        "all_queue_age_fields_present": missing_queue_age == 0,
        "raw_trace_matches_summary": raw_match,
        "raw_unverified_zero": raw_trace_count == 0 and summary_raw == 0,
        "cycle_p95_under_100ms": cycle_p95 <= 100.0,
        "ranking_trace_observable": ranking_steps > 0 if bool(metadata.get("variant", {}).get("use_jepa", False)) else True,
    }
    return {
        "audit_type": AUDIT_TYPE,
        "development_only": True,
        "locked_test_opened": False,
        "run_dir": str(run_dir),
        "run_summary_sha256": sha256(run_dir / "summary.json"),
        "run_provenance_sha256": sha256(run_dir / "provenance.json"),
        "git_revision": metadata.get("git_revision"),
        "variant": metadata.get("variant"),
        "training_seed": metadata.get("training_seed"),
        "episodes": len(episode_rows),
        "control_cycles": len(records),
        "ranking_steps": ranking_steps,
        "missing_latency_fields": missing_latency,
        "missing_queue_age_fields": missing_queue_age,
        "raw_trace_count": raw_trace_count,
        "latency": latency,
        "queue_age_steps": _stats(queue_values, suffix="steps"),
        "gates": gates,
    }


def _write_tensorboard(logdir: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite TensorBoard directory: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/latency_audit", json.dumps({"audit_type": AUDIT_TYPE, "development_only": True, "locked_test_opened": False}, indent=2), 0)
        writer.add_text("Provenance/run", json.dumps({"run_dir": report["run_dir"], "run_summary_sha256": report["run_summary_sha256"], "run_provenance_sha256": report["run_provenance_sha256"], "git_revision": report.get("git_revision")}, indent=2), 0)
        writer.add_text("Gates/status", json.dumps(report["gates"], indent=2), 0)
        for stage, values in report["latency"].items():
            for quantile in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"):
                writer.add_scalar(f"Latency/{stage}/{quantile}", float(values[quantile]), 0)
        for quantile in ("mean_steps", "p50_steps", "p95_steps", "p99_steps", "max_steps"):
            writer.add_scalar(f"Queue/age_steps/{quantile}", float(report["queue_age_steps"].get(quantile, 0.0)), 0)
        for name, passed in report["gates"].items():
            writer.add_scalar(f"Audit/gate/{name}", float(bool(passed)), 0)
        writer.add_scalar("Audit/control_cycles", float(report["control_cycles"]), 0)
        writer.add_scalar("Audit/ranking_steps", float(report["ranking_steps"]), 0)
        writer.flush()
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required_text = {
        "Config/latency_audit/text_summary",
        "Provenance/run/text_summary",
        "Gates/status/text_summary",
    }
    required_scalars = {
        "Latency/cycle_total/p95_ms",
        "Latency/jepa_inference/p95_ms",
        "Latency/ledger_route/p95_ms",
        "Latency/ranker_compute/p95_ms",
        "Latency/cbf_solver/p95_ms",
        "Queue/age_steps/p95_steps",
        "Audit/control_cycles",
    }
    missing = sorted(required_text.difference(tags.get("tensors", [])))
    missing.extend(sorted(required_scalars.difference(tags.get("scalars", []))))
    events = sorted(path.name for path in logdir.glob("events.out.tfevents.*"))
    if missing or not events:
        raise ValueError(f"Latency TensorBoard validation failed: missing={missing}, events={events}")
    return {"logdir": str(logdir), "event_files": events, "scalar_tag_count": len(tags.get("scalars", [])), "text_tag_count": len(tags.get("tensors", []))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    args = parser.parse_args()
    if not args.development_only:
        raise ValueError("Full-chain latency audit requires --development-only.")
    run_dir = args.run.resolve()
    output = (args.output.resolve() if args.output else run_dir / "latency_audit.json")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite latency audit: {output}")
    report = audit_run(run_dir)
    report["auditor"] = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "platform": platform.platform(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
    }
    report["tensorboard"] = _write_tensorboard(args.tensorboard_logdir, report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not all(bool(value) for value in report["gates"].values()):
        raise SystemExit("Full-chain latency audit failed.")


if __name__ == "__main__":
    main()
