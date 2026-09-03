"""Audit and aggregate three checkpoint-bound v3 Reliability Ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter

from encirclement3d.reliability import SafeCaptureReliabilityLedger, make_safe_capture_global_key


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEDGER_TYPE = SafeCaptureReliabilityLedger.LEDGER_TYPE_V3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def mean_sd(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def audit_ledger(path: Path) -> dict[str, Any]:
    path = path.resolve()
    payload = read_json(path)
    if payload.get("ledger_type") != LEDGER_TYPE or payload.get("ledger_version") != 3:
        raise ValueError(f"Not a v3 ledger: {path}")
    if payload.get("not_a_locked_test") is not True or payload.get("locked_test_opened") is not False:
        raise ValueError(f"Ledger is not development-only: {path}")
    if payload.get("immutable_after_calibration") is not True:
        raise ValueError(f"Ledger is mutable or missing immutable marker: {path}")
    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"Ledger source provenance missing: {path}")
    checkpoint = Path(str(source.get("checkpoint", ""))).resolve()
    dataset = Path(str(source.get("calibration_dataset", ""))).resolve()
    metadata = Path(str(source.get("calibration_metadata", ""))).resolve()
    protocol = Path(str(source.get("protocol", ""))).resolve()
    for item in (checkpoint, dataset, metadata, protocol):
        if not item.is_file():
            raise FileNotFoundError(f"Ledger provenance path is missing: {item}")
    for key, item in (
        ("checkpoint_sha256", checkpoint),
        ("calibration_dataset_sha256", dataset),
        ("calibration_metadata_sha256", metadata),
        ("protocol_sha256", protocol),
    ):
        if source.get(key) != sha256(item):
            raise ValueError(f"Ledger provenance hash mismatch for {key}: {path}")
    protocol_config = yaml.safe_load(protocol.read_text(encoding="utf-8"))
    if protocol_config.get("phase") != "development_only" or protocol_config.get("locked_test_opened") is not False:
        raise ValueError(f"Ledger protocol is not closed development: {protocol}")
    ledger = SafeCaptureReliabilityLedger(payload)
    fallback_audit = payload.get("fallback_audit", {})
    if fallback_audit.get("all_required_fallbacks_pass") is not True:
        raise ValueError(f"Ledger fallback audit failed: {path}")
    forecast = payload.get("forecast", {})
    if forecast.get("high_credit_failure_rate_not_above_low_credit") is not True:
        raise ValueError(f"High-credit reliability gate failed: {path}")
    if forecast.get("ood_or_hard_contexts_trigger_safe_hold") is not True:
        raise ValueError(f"OOD/stale safe-hold gate failed: {path}")
    return {
        "path": str(path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": source["checkpoint_sha256"],
        "calibration_dataset_sha256": source["calibration_dataset_sha256"],
        "calibration_metadata_sha256": source["calibration_metadata_sha256"],
        "protocol_sha256": source["protocol_sha256"],
        "training_variant": payload.get("training_variant"),
        "entry_count": len(payload.get("entries", {})),
        "fallback_audit": fallback_audit,
        "forecast": forecast,
        "policy": payload.get("decision_policy", {}),
        "horizon_seconds": source.get("horizon_seconds", []),
        "global_entries": [payload["entries"][make_safe_capture_global_key(index)] for index in range(len(source.get("horizon_seconds", [])))],
        "ledger_runtime_valid": True,
        "_ledger": ledger,
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) < 3:
        raise ValueError("v3 ledger aggregation requires at least three independent checkpoints.")
    if len({item["checkpoint_sha256"] for item in runs}) != len(runs):
        raise ValueError("v3 ledgers must bind distinct checkpoints.")
    horizon_count = len(runs[0]["global_entries"])
    if any(len(item["global_entries"]) != horizon_count for item in runs):
        raise ValueError("v3 ledger horizon counts differ.")
    metrics_by_horizon: list[dict[str, Any]] = []
    for index in range(horizon_count):
        rows = [item["global_entries"][index] for item in runs]
        metrics_by_horizon.append(
            {
                "horizon_index": index,
                "horizon_seconds": float(runs[0]["horizon_seconds"][index]),
                "sample_count": mean_sd([float(row["sample_count"]) for row in rows]),
                "credit": mean_sd([float(row["credit"]) for row in rows]),
                "target_mae_m": mean_sd([float(row["target_mae_m"]) for row in rows]),
                "clearance_mae_m": mean_sd([float(row["clearance_mae_m"]) for row in rows]),
                "collision_rate": mean_sd([float(row["collision_rate"]) for row in rows]),
                "boundary_rate": mean_sd([float(row["boundary_rate"]) for row in rows]),
                "candidate_ranking_win_rate": mean_sd([float(row["candidate_ranking_win_rate"]) for row in rows]),
            }
        )
    state_rates: dict[str, dict[str, float]] = {}
    for state in ("trusted", "fallback_nominal", "safe_hold"):
        values = [float(item["forecast"].get("unsafe_rate_by_state", {}).get(state, 0.0) or 0.0) for item in runs]
        state_rates[state] = mean_sd(values)
    fallback_reason_counts: dict[str, dict[str, float]] = {}
    reasons = {reason for item in runs for reason in item["forecast"].get("fallback_reason_counts", {})}
    for reason in sorted(reasons):
        fallback_reason_counts[reason] = mean_sd([float(item["forecast"].get("fallback_reason_counts", {}).get(reason, 0)) for item in runs])
    return {
        "aggregation_type": "jepa_safe_capture_v3_reliability_ledger_three_seed",
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "run_count": len(runs),
        "runs": [{key: value for key, value in item.items() if key != "_ledger"} for item in runs],
        "metrics_by_horizon": metrics_by_horizon,
        "unsafe_rate_by_state": state_rates,
        "fallback_reason_counts": fallback_reason_counts,
        "gates": {
            "all_ledgers_runtime_valid": all(item["ledger_runtime_valid"] for item in runs),
            "all_fallback_audits_pass": all(item["fallback_audit"]["all_required_fallbacks_pass"] for item in runs),
            "high_credit_failure_rate_not_above_low_credit": all(item["forecast"]["high_credit_failure_rate_not_above_low_credit"] for item in runs),
            "ood_stale_nonfinite_fallback_100_percent": all(item["fallback_audit"]["all_required_fallbacks_pass"] for item in runs),
            "eligible_for_closed_loop_smoke": True,
            "eligible_for_locked_test": False,
        },
    }


def write_tensorboard(report: dict[str, Any], logdir: Path) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite TensorBoard logdir: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/aggregation", json.dumps({"aggregation_type": report["aggregation_type"], "run_count": report["run_count"]}, indent=2), 0)
        writer.add_text("Provenance/runs", json.dumps(report["runs"], indent=2), 0)
        writer.add_text("Reliability/gates", json.dumps(report["gates"], indent=2), 0)
        writer.add_text("Reliability/fallback_reason_counts", json.dumps(report["fallback_reason_counts"], indent=2), 0)
        for item in report["metrics_by_horizon"]:
            step = int(item["horizon_index"]) + 1
            writer.add_scalar("Calibration/global_credit_mean", item["credit"]["mean"], step)
            writer.add_scalar("Calibration/global_credit_std", item["credit"]["sample_std"], step)
            writer.add_scalar("Calibration/target_mae_m_mean", item["target_mae_m"]["mean"], step)
            writer.add_scalar("Calibration/clearance_mae_m_mean", item["clearance_mae_m"]["mean"], step)
            writer.add_scalar("Calibration/collision_rate_mean", item["collision_rate"]["mean"], step)
            writer.add_scalar("Calibration/boundary_rate_mean", item["boundary_rate"]["mean"], step)
            writer.add_scalar("Calibration/ranking_win_rate_mean", item["candidate_ranking_win_rate"]["mean"], step)
        for state, values in report["unsafe_rate_by_state"].items():
            writer.add_scalar(f"Reliability/unsafe_rate_by_state/{state}", values["mean"], 0)
    return {
        "path": str(logdir),
        "event_files": sorted(path.name for path in logdir.glob("events.out.tfevents.*")),
        "logged": True,
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# JEPA Safe-Capture v3 Reliability Ledger Aggregate",
        "",
        "> Development-only calibration evidence. Ledgers are immutable, checkpoint-bound, and not safety certificates; all actions still require CBF.",
        "",
        f"Independent ledgers: `{report['run_count']}`; locked test opened: `{report['locked_test_opened']}`.",
        "",
        "| Horizon (s) | Samples | Credit | Target MAE (m) | Clearance MAE (m) | Collision rate | Boundary rate | Ranking win rate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["metrics_by_horizon"]:
        def value(name: str) -> str:
            metric = item[name]
            return f"{metric['mean']:.4f} +/- {metric['sample_std']:.4f}"
        lines.append(
            f"| {item['horizon_seconds']:.1f} | {value('sample_count')} | {value('credit')} | "
            f"{value('target_mae_m')} | {value('clearance_mae_m')} | "
            f"{value('collision_rate')} | {value('boundary_rate')} | {value('candidate_ranking_win_rate')} |"
        )
    lines.extend([
        "",
        "## Gates",
        "",
        f"- Runtime-valid ledgers: **{'PASS' if report['gates']['all_ledgers_runtime_valid'] else 'FAIL'}**.",
        f"- OOD/stale/non-finite fallback audit: **{'PASS' if report['gates']['ood_stale_nonfinite_fallback_100_percent'] else 'FAIL'}**.",
        f"- High-credit failure-rate ordering: **{'PASS' if report['gates']['high_credit_failure_rate_not_above_low_credit'] else 'FAIL'}**.",
        "- Eligible for closed-loop smoke only; this artifact does not authorize a locked test.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate([audit_ledger(path) for path in args.ledger])
    report["tensorboard"] = write_tensorboard(report, args.tensorboard_logdir)
    args.output_json.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_md.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_json.resolve().write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.output_md.resolve().write_text(render(report), encoding="utf-8")
    print(json.dumps(report["gates"], indent=2))


if __name__ == "__main__":
    main()
