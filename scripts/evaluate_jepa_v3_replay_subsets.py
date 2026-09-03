"""Compare JEPA-v3 replay checkpoints on train-defined held-out subsets.

The replay weights remain a train-only artifact.  This script applies their
fixed predicate to the independent validation labels for reporting only.  It
does not create weights, tune the predicate, or read development-S3 episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.prediction import (  # noqa: E402
    InteractionAwareActionConditionedMultitaskJEPAPredictor,
    build_action_conditioned_predictor,
)
from scripts.build_jepa_v3_hard_replay_weights import hard_example_masks  # noqa: E402
from scripts.evaluate_jepa_v3_multitask import binary_auc, choose_device, constant_velocity  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        required=True,
        help="Exactly one replay-off and one replay-on checkpoint.",
    )
    parser.add_argument("--replay-manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def predict(checkpoint_path: Path, inputs: np.ndarray, actions: np.ndarray, batch_size: int, device: torch.device) -> dict[str, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != "interaction_aware_action_conditioned_jepa_multitask":
        raise ValueError("Replay subset evaluation requires a JEPA-v3 multitask checkpoint.")
    model = build_action_conditioned_predictor(str(checkpoint["model_type"]), checkpoint["model"])
    if not isinstance(model, InteractionAwareActionConditionedMultitaskJEPAPredictor):
        raise RuntimeError("Checkpoint factory did not return the JEPA-v3 multitask predictor.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            stop = min(start + batch_size, inputs.shape[0])
            mean, log_variance, _latent, auxiliary = model.forward_multitask(
                torch.as_tensor(inputs[start:stop], device=device),
                torch.as_tensor(actions[start:stop], device=device),
            )
            parts["target_relative"].append(mean.cpu().numpy())
            parts["target_std"].append(torch.exp(0.5 * log_variance).cpu().numpy())
            parts["obstacle_clearance"].append(auxiliary["obstacle_clearance"].cpu().numpy())
            parts["inter_agent_clearance"].append(auxiliary["inter_agent_clearance"].cpu().numpy())
            parts["visibility_probability"].append(torch.sigmoid(auxiliary["target_visibility_logit"]).cpu().numpy())
            parts["cbf_correction"].append(auxiliary["cbf_correction"].cpu().numpy())
            parts["intervention_probability"].append(torch.sigmoid(auxiliary["cbf_intervention_logit"]).cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in parts.items()}


def subset_metrics(
    arrays: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    baseline: np.ndarray,
    indices: np.ndarray,
    extent: float,
    seconds: list[float],
) -> list[dict[str, Any]]:
    """Return held-out metrics for a fixed, non-empty sample subset."""
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("Replay subset metrics require a non-empty one-dimensional index array.")
    rows: list[dict[str, Any]] = []
    for horizon, horizon_seconds in enumerate(seconds):
        error = np.linalg.norm(predictions["target_relative"][indices, horizon] - arrays["labels_relative"][indices, horizon], axis=1)
        baseline_error = np.linalg.norm(baseline[indices, horizon] - arrays["labels_relative"][indices, horizon], axis=1)
        rows.append(
            {
                "horizon_seconds": float(horizon_seconds),
                "target_position_mae_m": float(np.mean(error) * extent),
                "constant_velocity_mae_m": float(np.mean(baseline_error) * extent),
                "target_improvement_over_constant_velocity_fraction": float(1.0 - np.mean(error) / max(np.mean(baseline_error), 1e-9)),
                "target_p90_error_m": float(np.quantile(error, 0.9) * extent),
                "target_one_std_coverage": float(
                    np.mean(np.abs(predictions["target_relative"][indices, horizon] - arrays["labels_relative"][indices, horizon]) <= predictions["target_std"][indices, horizon])
                ),
                "obstacle_clearance_mae_m": float(
                    np.mean(np.abs(predictions["obstacle_clearance"][indices, horizon] - arrays["labels_obstacle_clearance"][indices, horizon])) * extent
                ),
                "inter_agent_clearance_mae_m": float(
                    np.mean(np.abs(predictions["inter_agent_clearance"][indices, horizon] - arrays["labels_inter_agent_clearance"][indices, horizon])) * extent
                ),
                "visibility_brier": float(
                    np.mean((predictions["visibility_probability"][indices, horizon] - arrays["labels_target_visible"][indices, horizon]) ** 2)
                ),
                "visibility_auc": binary_auc(arrays["labels_target_visible"][indices, horizon], predictions["visibility_probability"][indices, horizon]),
                "cbf_correction_mae_mps": float(
                    np.mean(np.abs(predictions["cbf_correction"][indices, horizon] - arrays["labels_cbf_correction"][indices, horizon]))
                ),
                "cbf_intervention_brier": float(
                    np.mean((predictions["intervention_probability"][indices, horizon] - arrays["labels_cbf_intervention"][indices, horizon]) ** 2)
                ),
                "cbf_intervention_auc": binary_auc(arrays["labels_cbf_intervention"][indices, horizon], predictions["intervention_probability"][indices, horizon]),
            }
        )
    return rows


def group_ranking_metrics(
    arrays: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    sample_hard: np.ndarray,
    metadata: dict[str, Any],
    extent: float,
) -> list[dict[str, Any]]:
    """Score complete candidate groups, separating groups with any hard action.

    A group is a common `(episode_seed, time_index, agent_id)` state with all
    five counterfactual candidates.  Sample-level hard labels identify which
    candidate outcomes are challenging.  Ranking must nevertheless operate on
    complete groups, so an entire group is called hard when any candidate is
    hard.  This avoids comparing rankings after deleting the alternatives that
    a controller would have been able to choose.
    """
    candidate_count = int(metadata["candidate_count"])
    keys = np.stack([arrays["episode_seed"], arrays["time_index"], arrays["agent_id"]], axis=1)
    _unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    ends = np.concatenate((np.flatnonzero(np.diff(inverse[order])) + 1, [order.size]))
    starts = np.concatenate(([0], ends[:-1]))
    groups = [order[start:stop] for start, stop in zip(starts, ends)]
    for group in groups:
        if group.size != candidate_count or not np.array_equal(np.sort(arrays["candidate_index"][group]), np.arange(candidate_count)):
            raise ValueError("Counterfactual candidates are incomplete; ranking comparison would be invalid.")
    rows: list[dict[str, Any]] = []
    for horizon, horizon_seconds in enumerate(metadata["horizon_seconds"]):
        predicted_cost = np.linalg.norm(
            predictions["target_relative"][:, horizon] * extent
            - arrays["action_history"][:, -1] * float(metadata["action_scale"]) * float(horizon_seconds),
            axis=1,
        )
        settled_cost = np.linalg.norm(
            arrays["labels_relative"][:, horizon] * extent
            - arrays["action_history"][:, -1] * float(metadata["action_scale"]) * float(horizon_seconds),
            axis=1,
        )
        grouped: dict[str, list[tuple[float, float]]] = {"hard_group": [], "non_hard_group": []}
        for group in groups:
            selected = group[int(np.argmin(predicted_cost[group]))]
            best = float(np.min(settled_cost[group]))
            worst = float(np.max(settled_cost[group]))
            regret = float(settled_cost[selected] - best)
            credit = 1.0 if worst - best <= 1e-9 else float(np.clip(1.0 - regret / (worst - best), 0.0, 1.0))
            winner = float(regret <= 1e-8)
            key = "hard_group" if bool(np.any(sample_hard[group])) else "non_hard_group"
            grouped[key].append((credit, winner))
        row: dict[str, Any] = {"horizon_seconds": float(horizon_seconds)}
        for name, values in grouped.items():
            credit, winner = np.asarray(values, dtype=np.float64).T
            row[name] = {
                "groups": int(len(values)),
                "candidate_ranking_credit": float(np.mean(credit)),
                "candidate_ranking_win_rate": float(np.mean(winner)),
            }
        rows.append(row)
    return rows


def compare_metrics(replay_off: dict[str, Any], replay_on: dict[str, Any]) -> dict[str, Any]:
    """Compute direction-labelled deltas so lower-is-better metrics stay clear."""
    lower_is_better = (
        "target_position_mae_m",
        "target_p90_error_m",
        "obstacle_clearance_mae_m",
        "inter_agent_clearance_mae_m",
        "visibility_brier",
        "cbf_correction_mae_mps",
        "cbf_intervention_brier",
    )
    higher_is_better = (
        "target_improvement_over_constant_velocity_fraction",
        "visibility_auc",
        "cbf_intervention_auc",
    )
    comparisons: dict[str, Any] = {}
    for subset in ("overall", "hard_samples", "non_hard_samples"):
        rows: list[dict[str, Any]] = []
        for off_row, on_row in zip(replay_off["sample_metrics"][subset], replay_on["sample_metrics"][subset]):
            delta = {"horizon_seconds": float(off_row["horizon_seconds"])}
            for name in lower_is_better:
                delta[f"{name}_reduction_replay_on_minus_off"] = float(off_row[name] - on_row[name])
            for name in higher_is_better:
                off_value, on_value = off_row[name], on_row[name]
                delta[f"{name}_increase_replay_on_minus_off"] = None if off_value is None or on_value is None else float(on_value - off_value)
            rows.append(delta)
        comparisons[subset] = rows
    ranking: list[dict[str, Any]] = []
    for off_row, on_row in zip(replay_off["group_ranking_metrics"], replay_on["group_ranking_metrics"]):
        row: dict[str, Any] = {"horizon_seconds": float(off_row["horizon_seconds"])}
        for subset in ("hard_group", "non_hard_group"):
            row[subset] = {
                "candidate_ranking_credit_increase_replay_on_minus_off": float(
                    on_row[subset]["candidate_ranking_credit"] - off_row[subset]["candidate_ranking_credit"]
                ),
                "candidate_ranking_win_rate_increase_replay_on_minus_off": float(
                    on_row[subset]["candidate_ranking_win_rate"] - off_row[subset]["candidate_ranking_win_rate"]
                ),
            }
        ranking.append(row)
    return {"sample_metrics": comparisons, "group_ranking_metrics": ranking}


def _block_inverse(arrays: dict[str, np.ndarray], indices: np.ndarray) -> tuple[np.ndarray, int]:
    keys = np.stack(
        [arrays["episode_seed"][indices], arrays["time_index"][indices], arrays["agent_id"][indices]],
        axis=1,
    )
    groups, inverse = np.unique(keys, axis=0, return_inverse=True)
    return inverse, int(groups.shape[0])


def paired_block_bootstrap_interval(
    off_values: np.ndarray,
    on_values: np.ndarray,
    inverse: np.ndarray,
    group_count: int,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Interval for replay-on improvement under complete state-agent blocks.

    Positive values mean a lower replay-on error. The sample denominator is
    resampled with the numerator because hard groups can contain different
    numbers of hard candidates.
    """
    if off_values.shape != on_values.shape or off_values.ndim != 1 or inverse.shape != off_values.shape:
        raise ValueError("Paired block bootstrap inputs must be aligned one-dimensional arrays.")
    if group_count <= 0 or replicates <= 0:
        raise ValueError("Paired block bootstrap requires positive group and replicate counts.")
    difference_sum = np.bincount(inverse, weights=off_values - on_values, minlength=group_count)
    sample_count = np.bincount(inverse, minlength=group_count).astype(np.float64)
    batches: list[np.ndarray] = []
    for start in range(0, replicates, 64):
        count = min(64, replicates - start)
        draw = rng.integers(0, group_count, size=(count, group_count), endpoint=False)
        batches.append(np.sum(difference_sum[draw], axis=1) / np.sum(sample_count[draw], axis=1))
    distribution = np.concatenate(batches)
    return {
        "point_estimate_replay_on_error_reduction": float(np.sum(difference_sum) / np.sum(sample_count)),
        "ci95_low": float(np.quantile(distribution, 0.025)),
        "ci95_high": float(np.quantile(distribution, 0.975)),
        "state_agent_groups": int(group_count),
        "replicates": int(replicates),
    }


def paired_error_bootstrap(
    arrays: dict[str, np.ndarray],
    off_predictions: dict[str, np.ndarray],
    on_predictions: dict[str, np.ndarray],
    indices: np.ndarray,
    extent: float,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Bootstrap lower-is-better prediction-error reductions by horizon."""
    inverse, group_count = _block_inverse(arrays, indices)
    rows: list[dict[str, Any]] = []
    for horizon in range(off_predictions["target_relative"].shape[1]):
        off_values = {
            "target_position_mae_m": np.linalg.norm(
                off_predictions["target_relative"][indices, horizon] - arrays["labels_relative"][indices, horizon], axis=1
            ) * extent,
            "obstacle_clearance_mae_m": np.abs(
                off_predictions["obstacle_clearance"][indices, horizon] - arrays["labels_obstacle_clearance"][indices, horizon]
            ) * extent,
            "inter_agent_clearance_mae_m": np.abs(
                off_predictions["inter_agent_clearance"][indices, horizon] - arrays["labels_inter_agent_clearance"][indices, horizon]
            ) * extent,
            "cbf_correction_mae_mps": np.abs(
                off_predictions["cbf_correction"][indices, horizon] - arrays["labels_cbf_correction"][indices, horizon]
            ),
            "visibility_brier": (
                off_predictions["visibility_probability"][indices, horizon] - arrays["labels_target_visible"][indices, horizon]
            ) ** 2,
            "cbf_intervention_brier": (
                off_predictions["intervention_probability"][indices, horizon] - arrays["labels_cbf_intervention"][indices, horizon]
            ) ** 2,
        }
        on_values = {
            "target_position_mae_m": np.linalg.norm(
                on_predictions["target_relative"][indices, horizon] - arrays["labels_relative"][indices, horizon], axis=1
            ) * extent,
            "obstacle_clearance_mae_m": np.abs(
                on_predictions["obstacle_clearance"][indices, horizon] - arrays["labels_obstacle_clearance"][indices, horizon]
            ) * extent,
            "inter_agent_clearance_mae_m": np.abs(
                on_predictions["inter_agent_clearance"][indices, horizon] - arrays["labels_inter_agent_clearance"][indices, horizon]
            ) * extent,
            "cbf_correction_mae_mps": np.abs(
                on_predictions["cbf_correction"][indices, horizon] - arrays["labels_cbf_correction"][indices, horizon]
            ),
            "visibility_brier": (
                on_predictions["visibility_probability"][indices, horizon] - arrays["labels_target_visible"][indices, horizon]
            ) ** 2,
            "cbf_intervention_brier": (
                on_predictions["intervention_probability"][indices, horizon] - arrays["labels_cbf_intervention"][indices, horizon]
            ) ** 2,
        }
        metrics: dict[str, Any] = {}
        for metric_index, name in enumerate(off_values):
            metrics[name] = paired_block_bootstrap_interval(
                off_values[name],
                on_values[name],
                inverse,
                group_count,
                replicates,
                np.random.default_rng(seed + 100 * horizon + metric_index),
            )
        rows.append({"horizon_index": horizon, "metrics": metrics})
    return rows


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite replay subset evaluation output.")
    if args.batch_size <= 0 or args.bootstrap_replicates <= 0:
        raise ValueError("batch-size and bootstrap-replicates must be positive.")
    checkpoints = {label: Path(path).resolve() for label, path in args.checkpoint}
    if set(checkpoints) != {"replay-off", "replay-on"} or len(checkpoints) != 2:
        raise ValueError("Supply exactly --checkpoint replay-off PATH and --checkpoint replay-on PATH.")
    manifest = json.loads(args.replay_manifest.resolve().read_text(encoding="utf-8"))
    if manifest.get("replay_type") != "jepa_v3_train_only_hard_example_weights" or manifest.get("source_split") != "train":
        raise ValueError("Replay subset evaluator requires a train-only JEPA-v3 replay manifest.")
    metadata = json.loads(args.metadata.resolve().read_text(encoding="utf-8"))
    if metadata.get("split") != "validation":
        raise ValueError("Replay subset evaluation must use the held-out validation split.")
    if metadata.get("information_boundary", {}).get("development_s3_or_locked_data_used_for_training") is not False:
        raise ValueError("Validation metadata does not prove development/locked separation.")
    with np.load(args.dataset.resolve()) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    config = yaml.safe_load(Path(metadata["collection_config"]).read_text(encoding="utf-8"))
    extent = float(config["world"]["half_extent_xy"])
    masks = hard_example_masks(arrays, extent, manifest["policy"])
    hard_indices = np.flatnonzero(masks["hard"])
    non_hard_indices = np.flatnonzero(~masks["hard"])
    if hard_indices.size == 0 or non_hard_indices.size == 0:
        raise ValueError("Held-out data must contain both hard and non-hard samples for a replay comparison.")
    baseline = constant_velocity(arrays["inputs"].astype(np.float32), metadata)
    device = choose_device(args.device)
    reports: dict[str, Any] = {}
    predictions_by_label: dict[str, dict[str, np.ndarray]] = {}
    for label, checkpoint in checkpoints.items():
        predictions = predict(checkpoint, arrays["inputs"].astype(np.float32), arrays["action_history"].astype(np.float32), args.batch_size, device)
        predictions_by_label[label] = predictions
        reports[label] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "all_finite": bool(all(np.isfinite(value).all() for value in predictions.values())),
            "sample_metrics": {
                "overall": subset_metrics(arrays, predictions, baseline, np.arange(arrays["inputs"].shape[0]), extent, metadata["horizon_seconds"]),
                "hard_samples": subset_metrics(arrays, predictions, baseline, hard_indices, extent, metadata["horizon_seconds"]),
                "non_hard_samples": subset_metrics(arrays, predictions, baseline, non_hard_indices, extent, metadata["horizon_seconds"]),
            },
            "group_ranking_metrics": group_ranking_metrics(arrays, predictions, masks["hard"], metadata, extent),
        }
    summary = {
        "evaluation_type": "jepa_v3_replay_held_out_subset_comparison",
        "not_a_locked_test": True,
        "device": str(device),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset.resolve()),
        "metadata_sha256": sha256(args.metadata.resolve()),
        "replay_manifest": str(args.replay_manifest.resolve()),
        "replay_manifest_sha256": sha256(args.replay_manifest.resolve()),
        "subset_definition": {
            "sample_hard": "same fixed predicates as the train-only replay manifest, applied read-only to held-out validation labels",
            "hard_group": "a complete five-candidate state-agent group with at least one hard candidate",
            "policy": manifest["policy"],
        },
        "subset_counts": {
            "overall_samples": int(arrays["inputs"].shape[0]),
            "hard_samples": int(hard_indices.size),
            "non_hard_samples": int(non_hard_indices.size),
            "low_clearance_samples": int(np.sum(masks["low_clearance"])),
            "high_cbf_correction_samples": int(np.sum(masks["high_cbf_correction"])),
            "collision_or_boundary_samples": int(np.sum(masks["collision_or_boundary"])),
        },
        "reports": reports,
        "replay_on_minus_off": compare_metrics(reports["replay-off"], reports["replay-on"]),
        "paired_state_agent_block_bootstrap": {
            "definition": "Resample complete (episode_seed, time_index, agent_id) blocks; positive reductions favour replay-on for lower-is-better error metrics.",
            "hard_samples": paired_error_bootstrap(
                arrays, predictions_by_label["replay-off"], predictions_by_label["replay-on"], hard_indices, extent, args.bootstrap_replicates, args.bootstrap_seed
            ),
            "non_hard_samples": paired_error_bootstrap(
                arrays, predictions_by_label["replay-off"], predictions_by_label["replay-on"], non_hard_indices, extent, args.bootstrap_replicates, args.bootstrap_seed + 10000
            ),
        },
        "interpretation": "This is a held-out prediction and candidate-ranking comparison. It does not estimate capture rate, does not tune replay policy, and does not replace CBF or paired closed-loop evaluation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"subset_counts": summary["subset_counts"], "all_finite": {label: report["all_finite"] for label, report in reports.items()}}, indent=2))


if __name__ == "__main__":
    main()
