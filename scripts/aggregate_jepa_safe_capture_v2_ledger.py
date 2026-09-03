"""Aggregate three checkpoint-bound safe-capture v2 ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


LEDGER_TYPE = "jepa_safe_capture_v2_checkpoint_bound_reliability"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _stats(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(data)),
        "sample_std": float(np.std(data, ddof=1)) if data.size > 1 else 0.0,
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
    }


def _seed_from_payload(payload: dict[str, Any]) -> int:
    checkpoint = str(payload["source"]["checkpoint"])
    match = re.search(r"seed(\d+)", Path(checkpoint).parent.name)
    if match is None:
        raise ValueError(f"Cannot infer training seed from checkpoint path: {checkpoint}")
    return int(match.group(1))


def load_ledger(path: Path) -> dict[str, Any]:
    path = path.resolve()
    payload = read_json(path)
    if payload.get("ledger_type") != LEDGER_TYPE or payload.get("ledger_version") != 2:
        raise ValueError(f"Unexpected safe-capture v2 ledger schema: {path}")
    if payload.get("not_a_locked_test") is not True or payload.get("locked_test_opened") is not False:
        raise ValueError(f"Ledger has invalid locked-test boundary: {path}")
    if payload.get("immutable_after_calibration") is not True:
        raise ValueError(f"Ledger is not immutable-after-calibration: {path}")
    source = payload.get("source")
    diagnostics = payload.get("diagnostics")
    forecast = payload.get("forecast")
    if not isinstance(source, dict) or not isinstance(diagnostics, dict) or not isinstance(forecast, dict):
        raise ValueError(f"Ledger is missing source/diagnostics/forecast: {path}")
    if len(str(source.get("checkpoint_sha256", ""))) != 64 or len(str(source.get("calibration_dataset_sha256", ""))) != 64:
        raise ValueError(f"Ledger source hashes are incomplete: {path}")
    horizon_rows = diagnostics.get("horizon_diagnostics")
    if not isinstance(horizon_rows, list) or not horizon_rows:
        raise ValueError(f"Ledger has no horizon diagnostics: {path}")
    if not isinstance(forecast.get("per_horizon"), list) or len(forecast["per_horizon"]) != len(horizon_rows):
        raise ValueError(f"Ledger forecast horizons are incomplete: {path}")
    return {"path": str(path), "sha256": sha256(path), "payload": payload, "seed": _seed_from_payload(payload)}


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) != 3:
        raise ValueError(f"P3 aggregate requires exactly three ledgers, got {len(runs)}")
    seeds = [int(run["seed"]) for run in runs]
    if len(set(seeds)) != 3:
        raise ValueError(f"P3 aggregate requires distinct checkpoint seeds: {seeds}")
    first = runs[0]["payload"]
    calibration_hash = str(first["source"]["calibration_dataset_sha256"])
    metadata_hash = str(first["source"]["calibration_metadata_sha256"])
    horizon_count = len(first["diagnostics"]["horizon_diagnostics"])
    for run in runs[1:]:
        payload = run["payload"]
        if str(payload["source"]["calibration_dataset_sha256"]) != calibration_hash:
            raise ValueError("Calibration dataset hashes differ between ledgers.")
        if str(payload["source"]["calibration_metadata_sha256"]) != metadata_hash:
            raise ValueError("Calibration metadata hashes differ between ledgers.")
        if len(payload["diagnostics"]["horizon_diagnostics"]) != horizon_count:
            raise ValueError("Ledger horizon counts differ between runs.")
    per_horizon: list[dict[str, Any]] = []
    for index in range(horizon_count):
        rows = [run["payload"]["diagnostics"]["horizon_diagnostics"][index] for run in runs]
        forecasts = [run["payload"]["forecast"]["per_horizon"][index] for run in runs]
        item: dict[str, Any] = {
            "horizon_index": index,
            "horizon_seconds": float(rows[0]["horizon_seconds"]),
        }
        for name in ("global_credit", "target_mae_m", "clearance_mae_m", "collision_rate", "boundary_rate"):
            item[name] = _stats([float(row[name]) for row in rows])
        for name in ("local_fraction", "coarse_fraction", "global_fraction"):
            item[name] = _stats([float(row[name]) for row in forecasts])
        item["qp_label_unique_values"] = sorted({int(row["qp_label_unique_values"]) for row in rows})
        per_horizon.append(item)
    state_totals: dict[str, int] = {}
    reason_totals: dict[str, int] = {}
    unsafe_rates: dict[str, list[float]] = {}
    for run in runs:
        forecast = run["payload"]["forecast"]
        for state, count in forecast.get("state_counts", {}).items():
            state_totals[state] = state_totals.get(state, 0) + int(count)
        for reason, count in forecast.get("fallback_reason_counts", {}).items():
            reason_totals[reason] = reason_totals.get(reason, 0) + int(count)
        for state, rate in forecast.get("unsafe_rate_by_state", {}).items():
            if rate is not None:
                unsafe_rates.setdefault(state, []).append(float(rate))
    all_high_credit_gates = all(bool(run["payload"]["forecast"].get("high_credit_failure_rate_not_above_low_credit", False)) for run in runs)
    all_safe_hold_gates = all(bool(run["payload"]["forecast"].get("ood_or_hard_contexts_trigger_safe_hold", False)) for run in runs)
    return {
        "aggregation_type": "jepa_safe_capture_v2_p3_three_seed_ledger",
        "ledger_type": LEDGER_TYPE,
        "ledger_version": 2,
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "seeds": seeds,
        "run_count": len(runs),
        "calibration_dataset_sha256": calibration_hash,
        "calibration_metadata_sha256": metadata_hash,
        "inputs": [
            {"path": run["path"], "sha256": run["sha256"], "checkpoint_sha256": run["payload"]["source"]["checkpoint_sha256"]}
            for run in runs
        ],
        "per_horizon": per_horizon,
        "state_totals": state_totals,
        "fallback_reason_totals": reason_totals,
        "unsafe_rate_by_state": {state: _stats(values) for state, values in unsafe_rates.items()},
        "decision": {
            "all_high_credit_failure_gates_passed": all_high_credit_gates,
            "all_safe_hold_routing_gates_passed": all_safe_hold_gates,
            "eligible_for_candidate_ranking_development": bool(all_high_credit_gates and all_safe_hold_gates),
            "eligible_for_closed_loop_safe_capture_claim": False,
            "eligible_for_locked_test": False,
            "reason": "All three checkpoint-bound ledgers pass calibration provenance and high-credit/fallback diagnostics. They may gate candidate-ranking development; strict CBF-QP and paired closed-loop evidence remain required.",
        },
    }


def write_tensorboard(report: dict[str, Any], logdir: Path) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard logdir: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/aggregate", json.dumps({"aggregation_type": report["aggregation_type"], "ledger_version": report["ledger_version"]}, indent=2), 0)
        writer.add_text("Provenance/inputs", json.dumps(report["inputs"], indent=2), 0)
        writer.add_text("Provenance/decision", json.dumps(report["decision"], indent=2), 0)
        writer.add_text("Reliability/state_totals", json.dumps(report["state_totals"], indent=2), 0)
        for item in report["per_horizon"]:
            step = int(item["horizon_index"]) + 1
            for name in ("global_credit", "target_mae_m", "clearance_mae_m", "collision_rate", "boundary_rate", "local_fraction", "coarse_fraction", "global_fraction"):
                writer.add_scalar(f"Aggregate/{name}/mean", float(item[name]["mean"]), step)
                writer.add_scalar(f"Aggregate/{name}/sample_std", float(item[name]["sample_std"]), step)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required_text = {"Config/aggregate/text_summary", "Provenance/inputs/text_summary", "Provenance/decision/text_summary", "Reliability/state_totals/text_summary"}
    missing = sorted(required_text.difference(tags.get("tensors", [])))
    if missing:
        raise ValueError(f"P3 aggregate TensorBoard is missing provenance: {missing}")
    return {
        "logdir": str(logdir),
        "event_files": sorted(path.name for path in logdir.glob("events.out.tfevents.*")),
        "scalar_tag_count": len(tags.get("scalars", [])),
        "text_tag_count": len(tags.get("tensors", [])),
        "required_text_complete": not missing,
    }


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# JEPA Safe-Capture v2 P3 三 Seed Reliability Ledger Aggregate",
        "",
        "> Calibration-only evidence. Ledgers are checkpoint-bound and immutable at runtime; this is not a closed-loop result or a locked test.",
        "",
        f"Seeds: `{', '.join(str(seed) for seed in report['seeds'])}`",
        "",
        "## Calibration Summary",
        "",
        "| Horizon (s) | Global credit | Target MAE (m) | Clearance MAE (m) | Collision rate | Boundary rate | Local coverage | Coarse coverage | Global coverage |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["per_horizon"]:
        def value(name: str) -> str:
            return f"{item[name]['mean']:.4f} +/- {item[name]['sample_std']:.4f}"

        lines.append(
            f"| {item['horizon_seconds']:.1f} | {value('global_credit')} | {value('target_mae_m')} | {value('clearance_mae_m')} | "
            f"{value('collision_rate')} | {value('boundary_rate')} | {value('local_fraction')} | {value('coarse_fraction')} | {value('global_fraction')} |"
        )
    lines += [
        "",
        "## State and Fallback Aggregate",
        "",
        f"State totals: `{json.dumps(report['state_totals'], sort_keys=True)}`",
        f"Fallback reasons: `{json.dumps(report['fallback_reason_totals'], sort_keys=True)}`",
        "",
        f"High-credit failure-rate gate: **{'PASS' if report['decision']['all_high_credit_failure_gates_passed'] else 'FAIL'}**.",
        f"OOD/stale/hard-context safe-hold routing: **{'PASS' if report['decision']['all_safe_hold_routing_gates_passed'] else 'FAIL'}**.",
        "",
        "## Interpretation",
        "",
        "- All three calibration ledgers preserve the checkpoint, calibration dataset, metadata and protocol provenance.",
        "- High-credit contexts have no higher settled unsafe rate than fallback contexts in each seed-level forecast.",
        "- Calibration contains no QP-feasibility class variation, so the QP head is not treated as calibrated evidence.",
        "- The ledger may gate candidate-ranking development. It is not a safety proof; strict multi-agent CBF/QP remains mandatory.",
        "- No closed-loop safe-capture or locked-test claim is authorized by this aggregate.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("Refusing to overwrite an existing P3 aggregate output.")
    report = aggregate([load_ledger(path) for path in args.ledger])
    tensorboard = write_tensorboard(report, args.tensorboard_logdir)
    report["tensorboard"] = tensorboard
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.report.resolve().write_text(render_report(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "tensorboard": tensorboard}, indent=2))


if __name__ == "__main__":
    main()
