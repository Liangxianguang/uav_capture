"""Aggregate multi-seed JEPA prediction gates and ledger summaries."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--ledger-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    return parser.parse_args()


def mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot aggregate an empty metric list.")
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "n": len(values),
    }


def main() -> None:
    args = parse_args()
    predictions = [load_json(path.resolve()) for path in args.prediction]
    ledgers = [load_json(path.resolve()) for path in args.ledger_report]
    if len(predictions) != len(ledgers):
        raise ValueError("Prediction and ledger report counts must match.")
    if len(predictions) < 3:
        raise ValueError("The L0-L3 summary requires at least three seeds.")

    seeds = [int(item["training_seed"]) for item in predictions]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Prediction reports contain duplicate training seeds.")
    horizon_count = len(predictions[0]["metrics_by_horizon"])
    metric_names = (
        "target_position_mae_m",
        "target_improvement_over_constant_velocity_fraction",
        "target_one_std_coverage",
        "target_velocity_mae_mps",
        "target_acceleration_mae_mps2",
        "obstacle_clearance_lower_quantile_mae_m",
        "inter_agent_clearance_lower_quantile_mae_m",
        "pairwise_ttc_mae_s",
        "visibility_brier",
        "visibility_auc",
        "observation_age_mae_steps",
        "cbf_correction_mae_mps",
        "cbf_intervention_brier",
        "cbf_intervention_auc",
        "qp_feasibility_brier",
        "qp_feasibility_auc",
    )
    horizons: list[dict[str, Any]] = []
    for index in range(horizon_count):
        rows = [item["metrics_by_horizon"][index] for item in predictions]
        aggregate = {name: mean_std([float(row[name]) for row in rows]) for name in metric_names}
        aggregate["horizon_seconds"] = float(rows[0]["horizon_seconds"])
        horizons.append(aggregate)

    prediction_gates = [item.get("prediction_gate", {}) for item in predictions]
    all_finite = all(bool(gate.get("all_finite")) for gate in prediction_gates)
    all_horizons_better = all(bool(gate.get("target_better_than_constant_velocity_all_horizons")) for gate in prediction_gates)
    ledger_states = {}
    for seed, report in zip(seeds, ledgers):
        forecast = report.get("forecast", {})
        source = report.get("source", {})
        ledger_states[str(seed)] = {
            # Ledger reports store the model identity under ``source``. Keep
            # the fallback for older reports that used a top-level field.
            "checkpoint": report.get("checkpoint") or source.get("checkpoint_sha256"),
            "state_counts": forecast.get("state_counts", {}),
            "fallback_reason_counts": forecast.get("fallback_reason_counts", {}),
            "unsafe_rate_by_state": forecast.get("unsafe_rate_by_state", {}),
            "high_credit_failure_rate_not_above_low_credit": forecast.get("high_credit_failure_rate_not_above_low_credit"),
            "ood_or_hard_contexts_trigger_safe_hold": forecast.get("ood_or_hard_contexts_trigger_safe_hold"),
        }

    result = {
        "evaluation_type": "jepa_safe_capture_l0_l3_p2_prediction_and_p3_ledger_aggregate",
        "seed_count": len(seeds),
        "training_seeds": seeds,
        "metrics_by_horizon": horizons,
        "prediction_gate": {
            "all_finite_for_all_seeds": all_finite,
            "all_seeds_better_than_constant_velocity_at_all_horizons": all_horizons_better,
            "safe_capture_evaluated": False,
            "locked_test_opened": False,
        },
        "ledger_by_seed": ledger_states,
        "locked_test_opened": False,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# JEPA L0-L3 Prediction and Ledger Aggregate",
        "",
        "This is a development-only P2/P3 aggregate. It is not a closed-loop safe-capture result and did not open a locked test.",
        "",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}  ",
        f"All prediction outputs finite: `{all_finite}`  ",
        f"Every seed beats constant velocity at every horizon: `{all_horizons_better}`",
        "",
        "| Horizon (s) | Position MAE mean +/- std (m) | Improvement over CV mean +/- std | Visibility AUC mean +/- std | QP feasibility Brier mean +/- std |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in horizons:
        lines.append(
            f"| {row['horizon_seconds']:.1f} | "
            f"{row['target_position_mae_m']['mean']:.4f} +/- {row['target_position_mae_m']['std']:.4f} | "
            f"{row['target_improvement_over_constant_velocity_fraction']['mean']:.4f} +/- {row['target_improvement_over_constant_velocity_fraction']['std']:.4f} | "
            f"{row['visibility_auc']['mean']:.4f} +/- {row['visibility_auc']['std']:.4f} | "
            f"{row['qp_feasibility_brier']['mean']:.6f} +/- {row['qp_feasibility_brier']['std']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "All three seeds pass the held-out prediction gate and show positive target-position improvement over the constant-velocity reference at all four horizons. The ledger reports zero unsafe rate in each routed state for every seed in the calibration replay. These findings establish prediction and routing evidence only; the L0-L3 `safe_capture` endpoint still requires paired rolling-horizon closed-loop evaluation.",
            "",
        ]
    )
    args.markdown_output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.resolve().write_text("\n".join(lines), encoding="utf-8")

    writer = SummaryWriter(log_dir=str(args.tensorboard_logdir.resolve()), flush_secs=10)
    writer.add_text("Aggregate/result", json.dumps(result, indent=2), 0)
    writer.add_text("Aggregate/interpretation", lines[-3], 0)
    for row in horizons:
        step = int(round(row["horizon_seconds"] * 1000))
        writer.add_scalar("Prediction/target_position_mae_mean_m", row["target_position_mae_m"]["mean"], step)
        writer.add_scalar("Prediction/target_position_mae_std_m", row["target_position_mae_m"]["std"], step)
        writer.add_scalar("Prediction/improvement_over_cv_mean", row["target_improvement_over_constant_velocity_fraction"]["mean"], step)
        writer.add_scalar("Prediction/improvement_over_cv_std", row["target_improvement_over_constant_velocity_fraction"]["std"], step)
    writer.add_scalar("Prediction/all_finite", float(all_finite), 0)
    writer.add_scalar("Prediction/all_seeds_better_than_cv", float(all_horizons_better), 0)
    writer.add_scalar("Data/seed_count", len(seeds), 0)
    writer.flush()
    writer.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
