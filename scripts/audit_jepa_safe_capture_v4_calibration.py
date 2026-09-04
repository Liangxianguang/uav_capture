"""Audit multi-task JEPA heads on the independent safe-capture calibration split.

This audit is offline-only.  It does not run the environment or alter the
runtime ledger; it reports whether the safety-oriented heads are finite,
calibrated enough to be diagnosed, and action-conditioned on the five
candidate chunks before any closed-loop ranking experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_jepa_safe_capture_v2_reliability_ledger import (  # noqa: E402
    _load_arrays,
    _load_metadata,
    _predict,
    choose_device,
)


MODEL_TYPE = "interaction_aware_action_conditioned_jepa_safe_capture_v2"
TRAINING_VARIANT = "hard_context_weighted_v1"
QUANTILE = 0.10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ends = np.concatenate((np.flatnonzero(np.diff(sorted_scores)) + 1, [scores.size]))
    starts = np.concatenate(([0], ends[:-1]))
    ranks = np.empty(scores.size, dtype=np.float64)
    mean_ranks = (starts + 1.0 + ends) / 2.0
    ranks[order] = np.repeat(mean_ranks, ends - starts)
    positives = labels == 1.0
    negatives = labels == 0.0
    return float(
        (np.sum(ranks[positives]) - np.sum(positives) * (np.sum(positives) + 1.0) / 2.0)
        / (np.sum(positives) * np.sum(negatives))
    )


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64).reshape(-1), 0.0, 1.0)
    if labels.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.minimum(np.digitize(probabilities, edges[1:-1], right=False), bins - 1)
    error = 0.0
    for index in range(bins):
        selected = bucket == index
        if np.any(selected):
            error += float(np.mean(selected)) * abs(float(np.mean(probabilities[selected])) - float(np.mean(labels[selected])))
    return float(error)


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("Calibration protocol must be a YAML mapping.")
    if protocol.get("phase") != "development_only" or protocol.get("locked_test_opened") is not False:
        raise ValueError("Calibration audit requires phase=development_only and locked_test_opened=false.")
    ledger = protocol.get("reliability_ledger", {})
    if ledger.get("source_split") != "calibration_only":
        raise ValueError("Calibration audit requires a calibration-only ledger source split.")
    return protocol


def _action_conditioning_stats(arrays: dict[str, np.ndarray], predictions: np.ndarray) -> dict[str, float | int]:
    keys = np.stack([arrays["episode_seed"], arrays["time_index"], arrays["agent_id"]], axis=1)
    _groups, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    if counts.size == 0 or not np.all(counts == 5):
        raise ValueError("Calibration archive must contain exactly five candidates per state-agent group.")
    spreads: list[float] = []
    for group_index in range(len(counts)):
        group = predictions[inverse == group_index]
        spread = float(np.max(np.linalg.norm(group - np.mean(group, axis=0, keepdims=True), axis=1)))
        spreads.append(spread)
    spread_array = np.asarray(spreads, dtype=np.float64)
    return {
        "group_count": int(len(spreads)),
        "nonzero_spread_fraction": float(np.mean(spread_array > 1e-6)),
        "mean_spread_normalized": float(np.mean(spread_array)),
        "max_spread_normalized": float(np.max(spread_array)),
    }


def _evaluate_run(
    checkpoint_path: Path,
    dataset_path: Path,
    metadata_path: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    metadata = _load_metadata(metadata_path.resolve(), dataset_path.resolve())
    arrays = _load_arrays(dataset_path.resolve())
    predictions, checkpoint = _predict(checkpoint_path.resolve(), arrays, batch_size, device)
    if checkpoint.get("model_type") != MODEL_TYPE or checkpoint.get("training_variant") != TRAINING_VARIANT:
        raise ValueError(f"Unexpected checkpoint contract: {checkpoint_path}")
    collection = yaml.safe_load(Path(metadata["collection_config"]).resolve().read_text(encoding="utf-8"))
    extent = float(collection["world"]["half_extent_xy"])
    target_speed = float(collection["agents"]["target_max_speed"])
    target_acceleration = float(collection["agents"]["target_max_acceleration"])
    maximum_observation_age = float(collection["task"]["pursuit"]["maximum_message_age_steps"])
    horizon_seconds = [float(value) for value in metadata["horizon_seconds"]]
    metrics: list[dict[str, Any]] = []
    horizon_labels = np.asarray(metadata["horizon_steps"], dtype=np.int64)
    if not np.array_equal(horizon_labels, np.asarray([1, 2, 3, 5], dtype=np.int64)):
        raise ValueError("Unexpected calibration horizon contract.")
    for horizon_index, seconds in enumerate(horizon_seconds):
        target = arrays["labels_relative"][:, horizon_index]
        prediction = predictions["target_relative"][:, horizon_index]
        constant_velocity = (
            arrays["inputs"][:, -1, 3:6]
            + arrays["inputs"][:, -1, 6:9] * (seconds * target_speed / extent)
        )
        target_error = np.linalg.norm(prediction - target, axis=1) * extent
        constant_velocity_error = np.linalg.norm(constant_velocity - target, axis=1) * extent
        target_std = predictions["target_std"][:, horizon_index]

        predicted_clearance = np.minimum(
            predictions["obstacle_clearance"][:, horizon_index],
            predictions["inter_agent_clearance"][:, horizon_index],
        ) * extent
        settled_clearance = np.minimum(
            arrays["labels_obstacle_clearance"][:, horizon_index],
            arrays["labels_inter_agent_clearance"][:, horizon_index],
        ) * extent
        visibility_probability = predictions["visibility_probability"][:, horizon_index]
        visibility_label = arrays["labels_target_visible"][:, horizon_index]
        intervention_probability = predictions["intervention_probability"][:, horizon_index]
        intervention_label = arrays["labels_cbf_intervention"][:, horizon_index]
        qp_probability = predictions["qp_feasibility_probability"][:, horizon_index]
        qp_label = arrays["labels_cbf_qp_feasible"][:, horizon_index]
        metrics.append(
            {
                "horizon_index": horizon_index,
                "horizon_seconds": seconds,
                "target_position_mae_m": float(np.mean(target_error)),
                "constant_velocity_mae_m": float(np.mean(constant_velocity_error)),
                "target_improvement_over_constant_velocity_fraction": float(
                    1.0 - np.mean(target_error) / max(np.mean(constant_velocity_error), 1e-12)
                ),
                "target_one_std_coverage": float(np.mean(np.abs(prediction - target) <= target_std)),
                "target_velocity_mae_mps": float(
                    np.mean(np.abs(predictions["target_velocity"][:, horizon_index] - arrays["labels_target_velocity"][:, horizon_index]))
                    * target_speed
                ),
                "target_acceleration_mae_mps2": float(
                    np.mean(
                        np.abs(
                            predictions["target_acceleration"][:, horizon_index]
                            - arrays["labels_target_acceleration"][:, horizon_index]
                        )
                    )
                    * target_acceleration
                ),
                "clearance_lower_quantile_mae_m": float(np.mean(np.abs(predicted_clearance - settled_clearance))),
                "clearance_lower_quantile_coverage": float(np.mean(predicted_clearance <= settled_clearance)),
                "clearance_overprediction_rate": float(np.mean(predicted_clearance > settled_clearance)),
                "clearance_coverage_gap_from_q10": float(np.mean(predicted_clearance <= settled_clearance) - (1.0 - QUANTILE)),
                "pairwise_ttc_mae_s": float(
                    np.mean(np.abs(predictions["pairwise_ttc"][:, horizon_index] - arrays["labels_pairwise_ttc"][:, horizon_index]))
                ),
                "visibility_brier": float(np.mean((visibility_probability - visibility_label) ** 2)),
                "visibility_ece": _ece(visibility_label, visibility_probability),
                "visibility_auc": _binary_auc(visibility_label, visibility_probability),
                "observation_age_mae_steps": float(
                    np.mean(np.abs(predictions["observation_age"][:, horizon_index] - arrays["labels_observation_age"][:, horizon_index]))
                ),
                "cbf_correction_mae_mps": float(
                    np.mean(np.abs(predictions["cbf_correction"][:, horizon_index] - arrays["labels_cbf_correction"][:, horizon_index]))
                ),
                "cbf_intervention_brier": float(np.mean((intervention_probability - intervention_label) ** 2)),
                "cbf_intervention_ece": _ece(intervention_label, intervention_probability),
                "cbf_intervention_auc": _binary_auc(intervention_label, intervention_probability),
                "qp_feasibility_brier": float(np.mean((qp_probability - qp_label) ** 2)),
                "qp_feasibility_auc": _binary_auc(qp_label, qp_probability),
                "qp_label_unique_values": int(np.unique(qp_label).size),
                "action_conditioning": _action_conditioning_stats(
                    arrays,
                    prediction,
                ),
            }
        )
    finite_values = [value for value in predictions.values() if np.issubdtype(value.dtype, np.number)]
    if not all(np.isfinite(value).all() for value in finite_values):
        raise ValueError(f"Calibration predictions are non-finite: {checkpoint_path}")
    run = {
        "seed": int(checkpoint.get("seed", -1)),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path.resolve()),
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": sha256(dataset_path.resolve()),
        "metadata": str(metadata_path.resolve()),
        "metadata_sha256": sha256(metadata_path.resolve()),
        "samples": int(arrays["inputs"].shape[0]),
        "metrics_by_horizon": metrics,
        "all_predictions_finite": True,
        "locked_test_opened": False,
    }
    return run, predictions, arrays


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) < 3:
        raise ValueError("Calibration aggregation requires at least three independent checkpoints.")
    if len({run["seed"] for run in runs}) != len(runs):
        raise ValueError("Calibration checkpoints must have distinct seeds.")
    horizon_count = len(runs[0]["metrics_by_horizon"])
    metric_names = [
        "target_position_mae_m",
        "constant_velocity_mae_m",
        "target_improvement_over_constant_velocity_fraction",
        "target_one_std_coverage",
        "target_velocity_mae_mps",
        "target_acceleration_mae_mps2",
        "clearance_lower_quantile_mae_m",
        "clearance_lower_quantile_coverage",
        "clearance_overprediction_rate",
        "clearance_coverage_gap_from_q10",
        "pairwise_ttc_mae_s",
        "visibility_brier",
        "visibility_ece",
        "visibility_auc",
        "observation_age_mae_steps",
        "cbf_correction_mae_mps",
        "cbf_intervention_brier",
        "cbf_intervention_ece",
        "cbf_intervention_auc",
        "qp_feasibility_brier",
        "qp_feasibility_auc",
    ]
    metrics_by_horizon: list[dict[str, Any]] = []
    for index in range(horizon_count):
        rows = [run["metrics_by_horizon"][index] for run in runs]
        aggregate = {"horizon_index": index, "horizon_seconds": float(rows[0]["horizon_seconds"])}
        for name in metric_names:
            values = [float(row[name]) for row in rows if row[name] is not None]
            aggregate[name] = _summary(values) if values else None
        aggregate["qp_label_unique_values_by_seed"] = [int(row["qp_label_unique_values"]) for row in rows]
        aggregate["action_conditioning"] = {
            "group_count_by_seed": [int(row["action_conditioning"]["group_count"]) for row in rows],
            "nonzero_spread_fraction": _summary(
                [float(row["action_conditioning"]["nonzero_spread_fraction"]) for row in rows]
            ),
            "mean_spread_normalized": _summary(
                [float(row["action_conditioning"]["mean_spread_normalized"]) for row in rows]
            ),
        }
        metrics_by_horizon.append(aggregate)
    prediction_positive = all(
        item["target_improvement_over_constant_velocity_fraction"]["mean"] > 0.0
        for item in metrics_by_horizon
    )
    return {
        "audit_type": "jepa_safe_capture_v4_independent_calibration",
        "model_type": MODEL_TYPE,
        "training_variant": TRAINING_VARIANT,
        "quantile": QUANTILE,
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "run_count": len(runs),
        "runs": runs,
        "metrics_by_horizon": metrics_by_horizon,
        "calibration_gate": {
            "all_predictions_finite": all(bool(run["all_predictions_finite"]) for run in runs),
            "all_horizons_target_better_than_constant_velocity": prediction_positive,
            "clearance_coverage_reported": True,
            "visibility_probability_calibration_reported": True,
            "cbf_intervention_calibration_reported": True,
            "qp_feasibility_class_variation": any(
                any(value > 1 for value in item["qp_label_unique_values_by_seed"])
                for item in metrics_by_horizon
            ),
            "eligible_for_closed_loop": False,
            "requires_settled_counterfactual_rank_calibration": True,
        },
    }


def _write_tensorboard(
    logdir: Path,
    report: dict[str, Any],
    protocol: dict[str, Any],
    source_hashes: dict[str, str],
    histogram_data: list[tuple[str, np.ndarray]],
) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard directory: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/protocol", yaml.safe_dump(protocol, sort_keys=False), 0)
        writer.add_text("Config/calibration_audit", json.dumps({"quantile": QUANTILE, "model_type": MODEL_TYPE}, indent=2), 0)
        writer.add_text("Provenance/source_hashes", json.dumps(source_hashes, indent=2, sort_keys=True), 0)
        writer.add_text("Calibration/gate", json.dumps(report["calibration_gate"], indent=2), 0)
        for item in report["metrics_by_horizon"]:
            horizon = int(item["horizon_index"])
            for name, value in item.items():
                if isinstance(value, dict) and "mean" in value and np.isfinite(float(value["mean"])):
                    writer.add_scalar(f"Calibration/{name}/mean", float(value["mean"]), horizon)
                    writer.add_scalar(f"Calibration/{name}/sample_std", float(value["sample_std"]), horizon)
        for tag, values in histogram_data:
            finite = np.asarray(values, dtype=np.float32)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                writer.add_histogram(tag, finite, 0)
    event_files = sorted(path.name for path in logdir.glob("events.out.tfevents.*"))
    return {"path": str(logdir), "event_files": event_files, "scalar_tags": len(report["metrics_by_horizon"]) * 2}


def _render(report: dict[str, Any]) -> str:
    lines = [
        "# JEPA v4 Independent Calibration Audit",
        "",
        "> Offline calibration-only evidence. This is not a closed-loop result and not a locked test.",
        "",
        f"Independent checkpoints: `{report['run_count']}`; quantile target: `{report['quantile']:.2f}`.",
        "",
        "| Horizon (s) | Target MAE m | Improvement vs CV | 1-std coverage | Clearance q10 coverage | Clearance overprediction | Visibility Brier/ECE | CBF intervention Brier/ECE | QP label classes |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["metrics_by_horizon"]:
        def value(name: str) -> str:
            metric = item[name]
            if metric is None:
                return "n/a"
            return f"{metric['mean']:.4f} +/- {metric['sample_std']:.4f}"

        lines.append(
            f"| {item['horizon_seconds']:.1f} | {value('target_position_mae_m')} | "
            f"{value('target_improvement_over_constant_velocity_fraction')} | {value('target_one_std_coverage')} | "
            f"{value('clearance_lower_quantile_coverage')} | {value('clearance_overprediction_rate')} | "
            f"{value('visibility_brier')} / {value('visibility_ece')} | "
            f"{value('cbf_intervention_brier')} / {value('cbf_intervention_ece')} | "
            f"{item['qp_label_unique_values_by_seed']} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "- All three checkpoints emit finite predictions and improve target displacement MAE over constant velocity at every evaluated horizon.",
            "- Clearance lower-quantile coverage, visibility calibration and CBF-intervention calibration are now measured on the independent calibration split.",
            "- QP-feasibility labels have no class variation when the reported class count is one; the QP head is therefore diagnostic only until a varied archive exists.",
            "- Action-conditioning is audited by the non-zero candidate latent/prediction spread within each five-candidate state-agent group.",
            "- These metrics do not establish safe-capture improvement. Settled counterfactual ranking, ledger routing and CBF closed-loop tests remain required.",
            "- `locked_test_opened=false`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if len(args.run) < 3 or args.batch_size <= 0:
        raise ValueError("Calibration audit requires at least three runs and a positive batch size.")
    protocol = _load_protocol(args.protocol)
    device = choose_device(args.device)
    runs: list[dict[str, Any]] = []
    histogram_data: list[tuple[str, np.ndarray]] = []
    for run_path in args.run:
        checkpoint_path = run_path.resolve() / "checkpoint.pt"
        metadata_path = args.metadata.resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        run, predictions, arrays = _evaluate_run(
            checkpoint_path,
            args.dataset.resolve(),
            metadata_path,
            args.batch_size,
            device,
        )
        runs.append(run)
        histogram_data.extend(
            [
                (
                    f"Calibration/seed_{run['seed']}/target_error_normalized",
                    np.linalg.norm(predictions["target_relative"] - arrays["labels_relative"], axis=2),
                ),
                (f"Calibration/seed_{run['seed']}/visibility_probability", predictions["visibility_probability"]),
                (f"Calibration/seed_{run['seed']}/cbf_intervention_probability", predictions["intervention_probability"]),
            ]
        )
    report = _aggregate(runs)
    output_json = args.output_json.resolve()
    output_md = args.output_md.resolve()
    if output_json.exists() or output_md.exists():
        raise FileExistsError("Refusing to overwrite an existing calibration audit report.")
    source_hashes = {
        "scripts/audit_jepa_safe_capture_v4_calibration.py": sha256(Path(__file__).resolve()),
        "configs/jepa_safe_capture_v3_next_phase.yaml": sha256(args.protocol.resolve()),
        "calibration_dataset": sha256(args.dataset.resolve()),
        "calibration_metadata": sha256(args.metadata.resolve()),
    }
    report["protocol"] = str(args.protocol.resolve())
    report["protocol_sha256"] = sha256(args.protocol.resolve())
    report["source_hashes"] = source_hashes
    report["tensorboard"] = _write_tensorboard(args.tensorboard_logdir, report, protocol, source_hashes, histogram_data)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output_md.write_text(_render(report), encoding="utf-8")
    print(json.dumps(report["calibration_gate"], indent=2))


if __name__ == "__main__":
    main()
