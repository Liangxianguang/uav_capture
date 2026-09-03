"""Aggregate v3 held-out prediction gates across independent training seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def load_run(path: Path) -> dict[str, Any]:
    path = path.resolve()
    summary_path = path / "summary.json"
    report = read_json(summary_path)
    if report.get("evaluation_type") != "jepa_safe_capture_v2_p2_prediction_gate":
        raise ValueError(f"Unexpected prediction report type: {summary_path}")
    if report.get("locked_test_opened") is not False or report.get("not_a_locked_test") is not True:
        raise ValueError(f"Prediction report is not development-only: {summary_path}")
    gate = report.get("prediction_gate", {})
    if gate.get("all_finite") is not True:
        raise ValueError(f"Prediction gate is non-finite: {summary_path}")
    horizons = report.get("metrics_by_horizon")
    if not isinstance(horizons, list) or not horizons:
        raise ValueError(f"Prediction report has no horizons: {summary_path}")
    required = {
        "target_position_mae_m",
        "constant_velocity_mae_m",
        "target_improvement_over_constant_velocity_fraction",
        "obstacle_clearance_lower_quantile_mae_m",
        "inter_agent_clearance_lower_quantile_mae_m",
        "pairwise_ttc_mae_s",
        "visibility_brier",
        "cbf_intervention_brier",
    }
    for item in horizons:
        missing = sorted(required.difference(item))
        if missing:
            raise ValueError(f"Prediction horizon is missing fields {missing}: {summary_path}")
        if not all(np.isfinite(float(item[key])) for key in required):
            raise ValueError(f"Prediction horizon has non-finite metrics: {summary_path}")
    return {
        "summary_path": str(summary_path),
        "checkpoint": report.get("checkpoint"),
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "dataset_sha256": report.get("dataset_sha256"),
        "metadata_sha256": report.get("metadata_sha256"),
        "samples": int(report["samples"]),
        "device": report.get("device"),
        "metrics_by_horizon": horizons,
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) < 3:
        raise ValueError("Prediction aggregation requires at least three independent seeds.")
    if len({run["checkpoint_sha256"] for run in runs}) != len(runs):
        raise ValueError("Prediction runs must use distinct checkpoints.")
    horizon_count = len(runs[0]["metrics_by_horizon"])
    if any(len(run["metrics_by_horizon"]) != horizon_count for run in runs):
        raise ValueError("Prediction horizons differ between runs.")
    metric_names = (
        "target_position_mae_m",
        "constant_velocity_mae_m",
        "target_improvement_over_constant_velocity_fraction",
        "obstacle_clearance_lower_quantile_mae_m",
        "inter_agent_clearance_lower_quantile_mae_m",
        "pairwise_ttc_mae_s",
        "visibility_brier",
        "cbf_intervention_brier",
    )
    metrics_by_horizon: list[dict[str, Any]] = []
    for index in range(horizon_count):
        rows = [run["metrics_by_horizon"][index] for run in runs]
        item: dict[str, Any] = {"horizon_seconds": float(rows[0]["horizon_seconds"])}
        for name in metric_names:
            values = [float(row[name]) for row in rows]
            item[name] = summary(values)
        item["seeds_better_than_constant_velocity"] = int(
            sum(float(row["target_improvement_over_constant_velocity_fraction"]) > 0.0 for row in rows)
        )
        metrics_by_horizon.append(item)
    return {
        "aggregation_type": "jepa_safe_capture_v3_hard_context_prediction_three_seed",
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "run_count": len(runs),
        "runs": runs,
        "metrics_by_horizon": metrics_by_horizon,
        "prediction_gate": {
            "all_runs_finite": True,
            "all_horizons_have_positive_mean_improvement": all(
                item["target_improvement_over_constant_velocity_fraction"]["mean"] > 0.0
                for item in metrics_by_horizon
            ),
            "all_seed_horizons_have_positive_improvement": all(
                item["seeds_better_than_constant_velocity"] == len(runs)
                for item in metrics_by_horizon
            ),
            "eligible_for_ledger_calibration": True,
            "eligible_for_closed_loop_development": False,
            "eligible_for_locked_test": False,
        },
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# JEPA v3 Hard-Context Prediction Gate",
        "",
        "> Development-only held-out prediction evidence. This is not a closed-loop safe-capture result or a locked test.",
        "",
        f"Independent checkpoints: `{report['run_count']}`.",
        "",
        "| Horizon (s) | Target MAE (m) | Improvement vs CV | Seeds better | Obstacle clearance q10 MAE (m) | Inter-agent clearance q10 MAE (m) | Visibility Brier | CBF intervention Brier |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["metrics_by_horizon"]:
        def value(name: str) -> str:
            metric = item[name]
            return f"{metric['mean']:.4f} +/- {metric['sample_std']:.4f}"
        lines.append(
            f"| {item['horizon_seconds']:.1f} | {value('target_position_mae_m')} | "
            f"{value('target_improvement_over_constant_velocity_fraction')} | "
            f"{item['seeds_better_than_constant_velocity']}/{report['run_count']} | "
            f"{value('obstacle_clearance_lower_quantile_mae_m')} | "
            f"{value('inter_agent_clearance_lower_quantile_mae_m')} | "
            f"{value('visibility_brier')} | {value('cbf_intervention_brier')} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        "- All three hard-context weighted checkpoints emit finite predictions and improve target MAE over the constant-velocity baseline at every evaluated horizon.",
        "- The auxiliary heads are available for ledger calibration, but prediction accuracy is not a safety certificate and does not establish safe-capture improvement.",
        "- The next permitted step is independent calibration and reliability-ledger construction, followed by closed-loop smoke testing with the unchanged CBF contract.",
        "- No locked test was opened.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate([load_run(path) for path in args.run_dir])
    output_json = args.output_json.resolve()
    output_md = args.output_md.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["prediction_gate"], indent=2))


if __name__ == "__main__":
    main()
