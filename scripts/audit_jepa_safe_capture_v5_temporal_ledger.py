"""Audit temporal and out-of-distribution routing for the safe-capture ledger.

The audit is intentionally read-only.  It inspects closed-loop traces for
sequential drift signals and exercises the immutable checkpoint-bound ledger
with explicit fault contexts.  It does not fit thresholds from development
episodes, update credits online, or claim a control improvement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from encirclement3d.reliability import SafeCaptureReliabilityLedger


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _array_min(record: Mapping[str, Any], key: str) -> float | None:
    values = record.get(key)
    if not isinstance(values, list):
        return None
    finite = [number for item in values if (number := _finite(item)) is not None]
    return min(finite) if finite else None


def _array_max(record: Mapping[str, Any], key: str) -> float | None:
    values = record.get(key)
    if not isinstance(values, list):
        return None
    finite = [number for item in values if (number := _finite(item)) is not None]
    return max(finite) if finite else None


def _temporal_signals(
    ranking: Mapping[str, Any],
    observation: Mapping[str, Any],
    previous: Mapping[str, float | None] | None,
    *,
    margin_threshold: float,
    clearance_floor: float,
    uncertainty_threshold: float,
    risk_threshold: float,
    ttc_threshold: float,
    stale_age_steps: float,
    uncertainty_spike_threshold: float,
    risk_spike_threshold: float,
) -> tuple[dict[str, bool], dict[str, float | None]]:
    uncertainty = _array_max(ranking, "predicted_uncertainty")
    risk = _array_max(ranking, "predicted_cbf_risk")
    clearance = _array_min(ranking, "predicted_min_clearance_m")
    ttc = _array_min(ranking, "predicted_min_ttc_s")
    margin = _finite(ranking.get("top_two_margin_m"))
    ages = observation.get("target_observation_age_steps")
    age_values = [number for item in ages if (number := _finite(item)) is not None] if isinstance(ages, list) else []
    max_age = max(age_values) if age_values else None
    signals = {
        "low_margin": margin is not None and margin <= margin_threshold,
        "clearance_floor": clearance is not None and clearance < clearance_floor,
        "uncertainty_high": uncertainty is not None and uncertainty > uncertainty_threshold,
        "risk_ttc_high": (
            risk is not None
            and ttc is not None
            and risk >= risk_threshold
            and ttc < ttc_threshold
        ),
        "stale_observation": max_age is not None and max_age > stale_age_steps,
        "uncertainty_spike": (
            previous is not None
            and uncertainty is not None
            and previous.get("uncertainty") is not None
            and uncertainty - float(previous["uncertainty"]) >= uncertainty_spike_threshold
        ),
        "risk_spike": (
            previous is not None
            and risk is not None
            and previous.get("risk") is not None
            and risk - float(previous["risk"]) >= risk_spike_threshold
        ),
    }
    values = {
        "uncertainty": uncertainty,
        "risk": risk,
        "clearance": clearance,
        "ttc": ttc,
        "margin": margin,
        "max_observation_age": max_age,
    }
    return signals, values


def _episode_outcomes(run_dir: Path) -> dict[int, bool]:
    with (run_dir / "episodes.csv").open("r", newline="", encoding="utf-8") as handle:
        return {
            int(row["episode_index"]): _bool(row.get("safe_capture_success"))
            for row in csv.DictReader(handle)
        }


def _trace_records(run_dir: Path) -> dict[int, list[dict[str, Any]]]:
    records: dict[int, list[dict[str, Any]]] = {}
    for path in sorted((run_dir / "step_traces").glob("episode_*.jsonl")):
        episode = int(path.stem.split("_")[-1])
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"Invalid trace record: {path}")
        records[episode] = rows
    if not records:
        raise FileNotFoundError(f"No step traces found: {run_dir / 'step_traces'}")
    return records


def _audit_run(
    run_dir: Path,
    *,
    margin_threshold: float,
    clearance_floor: float,
    uncertainty_threshold: float,
    risk_threshold: float,
    ttc_threshold: float,
    stale_age_steps: float,
    uncertainty_spike_threshold: float,
    risk_spike_threshold: float,
) -> dict[str, Any]:
    summary = _load_json(run_dir / "summary.json")
    metadata = summary.get("metadata")
    overall = summary.get("overall")
    if not isinstance(metadata, dict) or not isinstance(overall, dict):
        raise ValueError(f"Invalid paired evaluator summary: {run_dir}")
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError(f"Run crossed the locked-test boundary: {run_dir}")
    outcomes = _episode_outcomes(run_dir)
    traces = _trace_records(run_dir)
    if set(outcomes) != set(traces):
        raise ValueError(f"Episode outcome/trace mismatch: {run_dir}")
    signal_counts: Counter[str] = Counter()
    episode_rows: list[dict[str, Any]] = []
    raw_trace_count = 0
    missing_raw_trace_count = 0
    finite_prediction_steps = 0
    for episode_index in sorted(traces):
        previous: dict[str, float | None] | None = None
        triggered_steps = 0
        max_consecutive = 0
        consecutive = 0
        first_trigger_step: int | None = None
        step_count = 0
        for record in traces[episode_index]:
            ranking = record.get("candidate_ranking")
            if not isinstance(ranking, dict):
                continue
            observation = record.get("observation")
            if not isinstance(observation, dict):
                observation = {}
            signals, values = _temporal_signals(
                ranking,
                observation,
                previous,
                margin_threshold=margin_threshold,
                clearance_floor=clearance_floor,
                uncertainty_threshold=uncertainty_threshold,
                risk_threshold=risk_threshold,
                ttc_threshold=ttc_threshold,
                stale_age_steps=stale_age_steps,
                uncertainty_spike_threshold=uncertainty_spike_threshold,
                risk_spike_threshold=risk_spike_threshold,
            )
            step_count += 1
            if all(values[key] is not None for key in ("uncertainty", "risk", "clearance", "ttc")):
                finite_prediction_steps += 1
            active = [name for name, enabled in signals.items() if enabled]
            for name in active:
                signal_counts[name] += 1
            if active:
                triggered_steps += 1
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
                if first_trigger_step is None:
                    first_trigger_step = int(record.get("step", step_count))
            else:
                consecutive = 0
            raw_flag = record.get("raw_unverified_executed")
            if raw_flag is None:
                missing_raw_trace_count += 1
            elif _bool(raw_flag):
                raw_trace_count += 1
            previous = {"uncertainty": values["uncertainty"], "risk": values["risk"]}
        episode_rows.append(
            {
                "episode_index": episode_index,
                "safe_capture": bool(outcomes[episode_index]),
                "ranking_steps": step_count,
                "temporal_trigger_steps": triggered_steps,
                "max_consecutive_trigger_steps": max_consecutive,
                "first_trigger_step": first_trigger_step,
            }
        )
    triggered_episodes = [row for row in episode_rows if row["temporal_trigger_steps"] > 0]
    safe_triggered = [row for row in triggered_episodes if row["safe_capture"]]
    unsafe_triggered = [row for row in triggered_episodes if not row["safe_capture"]]
    return {
        "run_dir": str(run_dir.resolve()),
        "variant": metadata.get("variant", {}).get("variant"),
        "training_seed": int(metadata.get("training_seed", -1)),
        "episodes": len(episode_rows),
        "safe_capture_rate": float(overall.get("safe_capture_rate", 0.0)),
        "cbf_unverified_steps": int(overall.get("cbf_unverified_steps", 0)),
        "raw_unverified_executed_steps": int(overall.get("raw_unverified_executed_steps", 0)),
        "cbf_controlled_abort_steps": int(overall.get("cbf_controlled_abort_steps", 0)),
        "signal_counts": dict(sorted(signal_counts.items())),
        "triggered_episode_count": len(triggered_episodes),
        "triggered_safe_capture_rate": float(np.mean([row["safe_capture"] for row in triggered_episodes])) if triggered_episodes else None,
        "triggered_safe_episode_count": len(safe_triggered),
        "triggered_unsafe_episode_count": len(unsafe_triggered),
        "max_consecutive_trigger_steps": max((row["max_consecutive_trigger_steps"] for row in episode_rows), default=0),
        "ranking_steps": int(sum(row["ranking_steps"] for row in episode_rows)),
        "finite_prediction_steps": finite_prediction_steps,
        "raw_trace_count": raw_trace_count,
        "missing_raw_trace_count": missing_raw_trace_count,
        "episode_rows": episode_rows,
        "summary_sha256": sha256(run_dir / "summary.json"),
        "provenance_sha256": sha256(run_dir / "provenance.json"),
        "protocol_sha256": metadata.get("inputs", {}).get("protocol_sha256"),
        "scene_manifest_sha256": metadata.get("inputs", {}).get("scene_manifest_sha256"),
    }


def _fault_contexts(ledger: SafeCaptureReliabilityLedger) -> dict[str, dict[str, Any]]:
    base: dict[str, Any] = {
        "visibility_condition": 1.0,
        "observation_age_steps": 0.0,
        "obstacle_count": 3,
        "layout_signature": "scenario_0",
        "target_motion_mode": "flee_persistence",
        "minimum_clearance_m": 1.0,
        "pairwise_ttc_s": 2.0,
        "uncertainty": 0.05,
        "cbf_risk": 0.10,
        "candidate_separation_m": 0.30,
    }
    return {
        "ood": {**base, "ood": True},
        "stale": {**base, "observation_age_steps": ledger.maximum_observation_age_steps + 1.0},
        "non_finite": {**base, "uncertainty": float("nan")},
        "uncertainty_high": {**base, "uncertainty": ledger.safe_hold_uncertainty_threshold + 0.01},
        "joint_ttc_cbf_risk": {**base, "pairwise_ttc_s": ledger.safe_hold_ttc_seconds - 0.01, "cbf_risk": 0.80},
        "unknown_horizon": {**base},
    }


def _fault_audit(ledger_path: Path) -> dict[str, Any]:
    payload = _load_json(ledger_path)
    before = sha256(ledger_path)
    ledger = SafeCaptureReliabilityLedger(payload)
    expected = {
        "ood": "ood",
        "stale": "stale_observation",
        "non_finite": "non_finite_context",
        "uncertainty_high": "uncertainty_high",
        "joint_ttc_cbf_risk": "joint_ttc_cbf_risk",
        "unknown_horizon": "missing_bucket",
    }
    results: dict[str, dict[str, Any]] = {}
    for name, context in _fault_contexts(ledger).items():
        horizon = 999 if name == "unknown_horizon" else 2
        try:
            decision = ledger.decision(horizon, context)
            results[name] = {
                "state": decision.state,
                "fallback_reason": decision.fallback_reason,
                "credit": decision.credit,
                "sample_count": decision.sample_count,
                "passed": bool(decision.state == "safe_hold" and decision.fallback_reason == expected[name]),
            }
        except Exception as error:  # pragma: no cover - defensive audit record
            results[name] = {"state": "exception", "fallback_reason": None, "error": repr(error), "passed": False}
    after = sha256(ledger_path)
    return {
        "ledger_path": str(ledger_path.resolve()),
        "ledger_sha256_before": before,
        "ledger_sha256_after": after,
        "immutable_file_hash": before == after,
        "cases": results,
        "all_required_fallbacks_pass": bool(all(item.get("passed") for item in results.values())),
    }


def _write_tensorboard(logdir: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite TensorBoard directory: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text(
            "Config/temporal_ledger_audit",
            json.dumps(report["policy"], indent=2, sort_keys=True),
            0,
        )
        writer.add_text("Provenance/runs", json.dumps(report["runs"], indent=2), 0)
        writer.add_text("Reliability/fault_audit", json.dumps(report["fault_audit"], indent=2), 0)
        writer.add_text("Gates/status", json.dumps(report["gates"], indent=2), 0)
        for run in report["runs"]:
            prefix = str(run["variant"]).upper()
            writer.add_scalar(f"Reliability/{prefix}/triggered_episode_count", run["triggered_episode_count"], 0)
            writer.add_scalar(f"Reliability/{prefix}/triggered_safe_capture_rate", float(run["triggered_safe_capture_rate"] or 0.0), 0)
            writer.add_scalar(f"Reliability/{prefix}/raw_unverified_executed_steps", run["raw_unverified_executed_steps"], 0)
            writer.add_scalar(f"Reliability/{prefix}/controlled_abort_steps", run["cbf_controlled_abort_steps"], 0)
            writer.add_scalar(f"Reliability/{prefix}/max_consecutive_trigger_steps", run["max_consecutive_trigger_steps"], 0)
            for name, count in run["signal_counts"].items():
                writer.add_scalar(f"Reliability/{prefix}/signal_{name}", count, 0)
        for name, result in report["fault_audit"]["cases"].items():
            writer.add_scalar(f"Fault/{name}/passed", float(bool(result.get("passed"))), 0)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required = {
        "Config/temporal_ledger_audit/text_summary",
        "Provenance/runs/text_summary",
        "Reliability/fault_audit/text_summary",
        "Gates/status/text_summary",
    }
    missing = sorted(required.difference(tags.get("tensors", [])))
    events = sorted(path.name for path in logdir.glob("events.out.tfevents.*"))
    if missing or not events:
        raise ValueError(f"Temporal ledger TensorBoard validation failed: missing={missing}, events={events}")
    return {
        "logdir": str(logdir),
        "event_files": events,
        "scalar_tag_count": len(tags.get("scalars", [])),
        "text_tag_count": len(tags.get("tensors", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    parser.add_argument("--margin-threshold", type=float, default=0.0015)
    parser.add_argument("--clearance-floor", type=float, default=0.35)
    parser.add_argument("--uncertainty-threshold", type=float, default=0.40)
    parser.add_argument("--risk-threshold", type=float, default=0.60)
    parser.add_argument("--ttc-threshold", type=float, default=0.30)
    parser.add_argument("--stale-age-steps", type=float, default=45.0)
    parser.add_argument("--uncertainty-spike-threshold", type=float, default=0.05)
    parser.add_argument("--risk-spike-threshold", type=float, default=0.10)
    args = parser.parse_args()
    if not args.development_only:
        raise ValueError("Temporal ledger audit requires --development-only.")
    if args.protocol.resolve().is_file() is False or args.ledger.resolve().is_file() is False:
        raise FileNotFoundError("Protocol and ledger files are required.")
    protocol = yaml.safe_load(args.protocol.resolve().read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or protocol.get("phase") != "development_only" or protocol.get("locked_test_opened") is not False:
        raise ValueError("Temporal ledger audit requires a closed development protocol.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    policy = {
        "margin_threshold": args.margin_threshold,
        "clearance_floor": args.clearance_floor,
        "uncertainty_threshold": args.uncertainty_threshold,
        "risk_threshold": args.risk_threshold,
        "ttc_threshold": args.ttc_threshold,
        "stale_age_steps": args.stale_age_steps,
        "uncertainty_spike_threshold": args.uncertainty_spike_threshold,
        "risk_spike_threshold": args.risk_spike_threshold,
        "thresholds_fit_from_development": False,
    }
    runs = [
        _audit_run(
            run.resolve(),
            margin_threshold=args.margin_threshold,
            clearance_floor=args.clearance_floor,
            uncertainty_threshold=args.uncertainty_threshold,
            risk_threshold=args.risk_threshold,
            ttc_threshold=args.ttc_threshold,
            stale_age_steps=args.stale_age_steps,
            uncertainty_spike_threshold=args.uncertainty_spike_threshold,
            risk_spike_threshold=args.risk_spike_threshold,
        )
        for run in args.run
    ]
    fault_audit = _fault_audit(args.ledger.resolve())
    gates = {
        "development_only": True,
        "locked_test_not_opened": True,
        "fault_fallbacks_pass": fault_audit["all_required_fallbacks_pass"],
        "ledger_file_immutable": fault_audit["immutable_file_hash"],
        "raw_unverified_zero": all(run["raw_unverified_executed_steps"] == 0 for run in runs),
        "raw_trace_observable": all(run["missing_raw_trace_count"] == 0 for run in runs),
        "finite_prediction_trace": all(run["finite_prediction_steps"] == run["ranking_steps"] for run in runs),
    }
    report: dict[str, Any] = {
        "audit_type": "jepa_safe_capture_v5_temporal_ledger",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "policy": policy,
        "runs": runs,
        "fault_audit": fault_audit,
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "provenance": {
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
            "source_hashes": {
                "scripts/audit_jepa_safe_capture_v5_temporal_ledger.py": sha256(Path(__file__).resolve()),
                "protocol": sha256(args.protocol.resolve()),
                "ledger": sha256(args.ledger.resolve()),
            },
        },
    }
    report["tensorboard"] = _write_tensorboard(args.tensorboard_logdir, report)
    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    (args.output_dir.resolve() / "temporal_ledger_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# P12 Temporal Reliability Ledger Audit",
        "",
        "> Read-only development audit; thresholds are fixed inputs and no online credit update is performed.",
        "",
        f"All gates pass: `{report['all_gates_pass']}`.",
        "",
        "| Variant | Episodes | Safe capture | Triggered episodes | Triggered safe rate | Raw unverified | Controlled abort | Max consecutive triggers |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        lines.append(
            f"| {run['variant']} | {run['episodes']} | {run['safe_capture_rate']:.3f} | "
            f"{run['triggered_episode_count']} | "
            f"{(run['triggered_safe_capture_rate'] if run['triggered_safe_capture_rate'] is not None else float('nan')):.3f} | "
            f"{run['raw_unverified_executed_steps']} | {run['cbf_controlled_abort_steps']} | "
            f"{run['max_consecutive_trigger_steps']} |"
        )
    lines.extend(
        [
            "",
            "Fault cases: " + ", ".join(
                f"{name}={result['state']}/{result['fallback_reason']}"
                for name, result in fault_audit["cases"].items()
            ),
            "",
            "`locked_test_opened=false`; triggered signals are diagnostics for future temporal routing, not a control-improvement claim.",
        ]
    )
    (args.output_dir.resolve() / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"all_gates_pass": report["all_gates_pass"], "gates": gates, "tensorboard": report["tensorboard"]}, indent=2))


if __name__ == "__main__":
    main()
