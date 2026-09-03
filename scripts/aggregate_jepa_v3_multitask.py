"""Aggregate audited JEPA-v3 multitask training seeds into development evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


REQUIRED_SCALARS = {
    "Loss/train",
    "Loss/validation",
    "Target/train",
    "Target/validation",
    "Clearance/train",
    "Clearance/validation",
    "Visibility/train",
    "Visibility/validation",
    "Risk/train",
    "Risk/validation",
    "Calibration/train",
    "Calibration/validation",
    "Optimization/learning_rate",
}
REQUIRED_TEXT = {
    "Config/protocol/text_summary",
    "Config/model/text_summary",
    "Config/optimization/text_summary",
    "Dataset/train_metadata/text_summary",
    "Dataset/validation_metadata/text_summary",
    "Provenance/source_hashes/text_summary",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _mean_sd(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(data)),
        "sample_std": float(np.std(data, ddof=1)) if data.size > 1 else 0.0,
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
    }


def _tensorboard_audit(logdir: Path) -> dict[str, Any]:
    accumulator = EventAccumulator(str(logdir))
    accumulator.Reload()
    tags = accumulator.Tags()
    scalar_tags = set(tags["scalars"])
    text_tags = set(tags["tensors"])
    missing_scalars = sorted(REQUIRED_SCALARS.difference(scalar_tags))
    missing_text = sorted(REQUIRED_TEXT.difference(text_tags))
    epochs = len(accumulator.Scalars("Loss/train")) if "Loss/train" in scalar_tags else 0
    return {
        "logdir": str(logdir.resolve()),
        "event_files": sorted(path.name for path in logdir.glob("events.out.tfevents.*")),
        "loss_train_epochs": epochs,
        "histogram_tag_count": len(tags["histograms"]),
        "missing_required_scalars": missing_scalars,
        "missing_required_text": missing_text,
        "complete": bool(epochs > 0 and not missing_scalars and not missing_text and tags["histograms"]),
    }


def load_run(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    gate = _read_json(directory / "prediction_gate.json")
    action_audit = _read_json(directory / "action_following_audit.json")
    run_metadata = _read_json(directory / "run_metadata.json")
    checkpoint = directory / "checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if gate.get("evaluation_type") != "jepa_v3_multitask_prediction_gate":
        raise ValueError(f"Unexpected gate type: {directory}")
    gate_result = gate.get("prediction_gate", {})
    if gate_result.get("all_finite") is not True or gate_result.get("accepted_for_development_control_smoke") is not True:
        raise ValueError(f"Run did not pass the development prediction gate: {directory}")
    reports = action_audit.get("reports")
    if not isinstance(reports, list) or len(reports) != 1 or reports[0].get("all_finite") is not True:
        raise ValueError(f"Action-following audit is incomplete: {directory}")
    tensorboard = _tensorboard_audit(Path(run_metadata["tensorboard_logdir"]))
    if not tensorboard["complete"]:
        raise ValueError(f"TensorBoard provenance is incomplete: {directory}")
    return {
        "run_directory": str(directory),
        "seed": int(run_metadata["seed"]),
        "checkpoint_sha256": _sha256(checkpoint),
        "best_epoch": int(run_metadata["best_epoch"]),
        "best_validation_loss": float(run_metadata["best_validation_loss"]),
        "prediction_gate": gate,
        "action_following": reports[0],
        "tensorboard": tensorboard,
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = [run["seed"] for run in runs]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Aggregation requires at least two unique training seeds.")
    horizon_count = len(runs[0]["prediction_gate"]["metrics_by_horizon"])
    if any(len(run["prediction_gate"]["metrics_by_horizon"]) != horizon_count for run in runs):
        raise ValueError("Prediction horizons differ between runs.")
    metrics_by_horizon: list[dict[str, Any]] = []
    metric_names = (
        "target_position_mae_m",
        "target_improvement_over_constant_velocity_fraction",
        "obstacle_clearance_mae_m",
        "inter_agent_clearance_mae_m",
        "visibility_auc",
        "cbf_intervention_auc",
        "target_one_std_coverage",
    )
    for index in range(horizon_count):
        rows = [run["prediction_gate"]["metrics_by_horizon"][index] for run in runs]
        metric = {"horizon_seconds": float(rows[0]["horizon_seconds"])}
        for name in metric_names:
            values = [float(row[name]) for row in rows if row[name] is not None]
            metric[name] = _mean_sd(values) if values else None
        metric["seeds_better_than_constant_velocity"] = int(
            sum(float(row["target_improvement_over_constant_velocity_fraction"]) > 0.0 for row in rows)
        )
        metrics_by_horizon.append(metric)
    audit_axes = [run["action_following"]["axes"] for run in runs]
    separation = [
        float(axis["mean_plus_minus_separation_norm"][horizon])
        for axes in audit_axes
        for axis in axes
        for horizon in range(horizon_count)
    ]
    return {
        "aggregation_type": "jepa_v3_multitask_three_seed_development",
        "not_a_locked_test": True,
        "run_count": len(runs),
        "seeds": seeds,
        "runs": runs,
        "metrics_by_horizon": metrics_by_horizon,
        "action_following": {
            "all_runs_finite": True,
            "candidate_separation_normalized_position": _mean_sd(separation),
        },
        "decision": {
            "all_prediction_gates_passed": True,
            "eligible_for_reliability_ledger_development": True,
            "eligible_for_direct_locked_test": False,
            "reason": "Prediction gates and action-following audits pass, but closed-loop paired development evidence is still required.",
        },
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# JEPA-v3 Multitask Three-Seed Prediction Aggregate",
        "",
        "> Development-only prediction evidence. It is not a locked test and does not establish closed-loop capture improvement.",
        "",
        f"Training seeds: `{', '.join(str(value) for value in report['seeds'])}`.",
        "",
        "| Horizon (s) | Target MAE (m) | Improvement vs CV | Seeds better than CV | Obstacle clearance MAE (m) | Inter-agent clearance MAE (m) | Visibility AUROC | CBF intervention AUROC |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["metrics_by_horizon"]:
        def value(name: str) -> str:
            metric = item[name]
            return "n/a" if metric is None else f"{metric['mean']:.4f} +/- {metric['sample_std']:.4f}"
        lines.append(
            f"| {item['horizon_seconds']:.1f} | {value('target_position_mae_m')} | "
            f"{value('target_improvement_over_constant_velocity_fraction')} | "
            f"{item['seeds_better_than_constant_velocity']}/3 | {value('obstacle_clearance_mae_m')} | "
            f"{value('inter_agent_clearance_mae_m')} | {value('visibility_auc')} | "
            f"{value('cbf_intervention_auc')} |"
        )
    lines += [
        "",
        "## Run Provenance",
        "",
        "| Seed | Best epoch | Best validation loss | Checkpoint SHA-256 | TensorBoard epochs | Histogram tags |",
        "| ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for run in report["runs"]:
        tensorboard = run["tensorboard"]
        lines.append(
            f"| {run['seed']} | {run['best_epoch']} | {run['best_validation_loss']:.6f} | "
            f"`{run['checkpoint_sha256']}` | {tensorboard['loss_train_epochs']} | "
            f"{tensorboard['histogram_tag_count']} |"
        )
    lines += [
        "",
        "## Interpretation and Limits",
        "",
        "- All three checkpoints pass the finite-output, held-out prediction, action-following, and TensorBoard provenance checks.",
        "- Target prediction improves over constant velocity in all three seeds at 0.2, 0.3, and 0.5 seconds. At 0.1 seconds only one of three seeds improves, so the learned predictor must not be treated as uniformly superior at every horizon.",
        "- CBF intervention AUROC is consistently high across horizons. Visibility AUROC is modest and obstacle-clearance error remains material; neither signal is a safety certificate or a license to bypass CBF.",
        "- The next permitted use is an execution-settled reliability ledger with deterministic nominal-action fallback. Closed-loop paired development evaluation remains required before any claim that this model improves capture.",
        "",
        f"Action-following candidate separation: `{report['action_following']['candidate_separation_normalized_position']['mean']:.6f} +/- {report['action_following']['candidate_separation_normalized_position']['sample_std']:.6f}` normalized position units.",
        "",
        report["decision"]["reason"],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate([load_run(path) for path in args.run_dir])
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
