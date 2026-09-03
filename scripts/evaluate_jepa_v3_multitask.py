"""Evaluate JEPA-v3 multi-task checkpoints on held-out counterfactual data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.prediction import (  # noqa: E402
    InteractionAwareActionConditionedMultitaskJEPAPredictor,
    build_action_conditioned_predictor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu")


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # Average tied ranks for a deterministic Mann-Whitney AUROC.
    for value in np.unique(scores):
        tied = scores == value
        if np.sum(tied) > 1:
            ranks[tied] = np.mean(ranks[tied])
    return float((np.sum(ranks[labels == 1]) - positives * (positives + 1) / 2.0) / (positives * negatives))


def constant_velocity(inputs: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    config_path = Path(metadata["collection_config"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    extent = float(config["world"]["half_extent_xy"])
    target_max_speed = float(config["agents"]["target_max_speed"])
    seconds = np.asarray(metadata["horizon_seconds"], dtype=np.float32)
    relative = inputs[:, -1, 3:6]
    velocity = inputs[:, -1, 6:9]
    return relative[:, None, :] + velocity[:, None, :] * (seconds[None, :, None] * target_max_speed / extent)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    metadata = json.loads(args.metadata.resolve().read_text(encoding="utf-8"))
    if metadata.get("split") != "validation":
        raise ValueError("JEPA-v3 prediction gate must use the held-out validation dataset.")
    arrays = np.load(args.dataset.resolve())
    inputs = np.asarray(arrays["inputs"], dtype=np.float32)
    actions = np.asarray(arrays["action_history"], dtype=np.float32)
    target = np.asarray(arrays["labels_relative"], dtype=np.float32)
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != "interaction_aware_action_conditioned_jepa_multitask":
        raise ValueError("Checkpoint is not an interaction-aware JEPA-v3 multitask model.")
    model = build_action_conditioned_predictor(str(checkpoint["model_type"]), checkpoint["model"])
    if not isinstance(model, InteractionAwareActionConditionedMultitaskJEPAPredictor):
        raise RuntimeError("Checkpoint factory did not return the JEPA-v3 multitask model.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = choose_device(args.device)
    model = model.to(device).eval()
    target_mean: list[np.ndarray] = []
    target_std: list[np.ndarray] = []
    outputs: dict[str, list[np.ndarray]] = {
        "obstacle_clearance": [],
        "inter_agent_clearance": [],
        "target_visibility_probability": [],
        "cbf_correction": [],
        "cbf_intervention_probability": [],
    }
    with torch.no_grad():
        for start in range(0, inputs.shape[0], args.batch_size):
            stop = min(start + args.batch_size, inputs.shape[0])
            mean, log_variance, _latent, auxiliary = model.forward_multitask(
                torch.as_tensor(inputs[start:stop], device=device),
                torch.as_tensor(actions[start:stop], device=device),
            )
            target_mean.append(mean.cpu().numpy())
            target_std.append(torch.exp(0.5 * log_variance).cpu().numpy())
            outputs["obstacle_clearance"].append(auxiliary["obstacle_clearance"].cpu().numpy())
            outputs["inter_agent_clearance"].append(auxiliary["inter_agent_clearance"].cpu().numpy())
            outputs["target_visibility_probability"].append(torch.sigmoid(auxiliary["target_visibility_logit"]).cpu().numpy())
            outputs["cbf_correction"].append(auxiliary["cbf_correction"].cpu().numpy())
            outputs["cbf_intervention_probability"].append(torch.sigmoid(auxiliary["cbf_intervention_logit"]).cpu().numpy())
    prediction = np.concatenate(target_mean, axis=0)
    std = np.concatenate(target_std, axis=0)
    decoded = {key: np.concatenate(parts, axis=0) for key, parts in outputs.items()}
    baseline = constant_velocity(inputs, metadata)
    extent = float(yaml.safe_load(Path(metadata["collection_config"]).read_text(encoding="utf-8"))["world"]["half_extent_xy"])
    horizons: list[dict[str, Any]] = []
    for index, seconds in enumerate(metadata["horizon_seconds"]):
        error = np.linalg.norm(prediction[:, index] - target[:, index], axis=1)
        baseline_error = np.linalg.norm(baseline[:, index] - target[:, index], axis=1)
        coverage = np.mean(np.abs(prediction[:, index] - target[:, index]) <= std[:, index])
        horizons.append(
            {
                "horizon_seconds": float(seconds),
                "target_position_mae_m": float(np.mean(error) * extent),
                "constant_velocity_mae_m": float(np.mean(baseline_error) * extent),
                "target_improvement_over_constant_velocity_fraction": float(1.0 - np.mean(error) / max(np.mean(baseline_error), 1e-9)),
                "target_p90_error_m": float(np.quantile(error, 0.9) * extent),
                "target_one_std_coverage": float(coverage),
                "obstacle_clearance_mae_m": float(np.mean(np.abs(decoded["obstacle_clearance"][:, index] - arrays["labels_obstacle_clearance"][:, index])) * extent),
                "inter_agent_clearance_mae_m": float(np.mean(np.abs(decoded["inter_agent_clearance"][:, index] - arrays["labels_inter_agent_clearance"][:, index])) * extent),
                "visibility_brier": float(np.mean((decoded["target_visibility_probability"][:, index] - arrays["labels_target_visible"][:, index]) ** 2)),
                "visibility_auc": binary_auc(arrays["labels_target_visible"][:, index], decoded["target_visibility_probability"][:, index]),
                "cbf_correction_mae_mps": float(np.mean(np.abs(decoded["cbf_correction"][:, index] - arrays["labels_cbf_correction"][:, index]))),
                "cbf_intervention_brier": float(np.mean((decoded["cbf_intervention_probability"][:, index] - arrays["labels_cbf_intervention"][:, index]) ** 2)),
                "cbf_intervention_auc": binary_auc(arrays["labels_cbf_intervention"][:, index], decoded["cbf_intervention_probability"][:, index]),
            }
        )
    all_finite = bool(np.isfinite(prediction).all() and np.isfinite(std).all() and all(np.isfinite(value).all() for value in decoded.values()))
    target_better = [item["target_improvement_over_constant_velocity_fraction"] > 0.0 for item in horizons]
    summary = {
        "evaluation_type": "jepa_v3_multitask_prediction_gate",
        "not_a_locked_test": True,
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "samples": int(inputs.shape[0]),
        "device": str(device),
        "metrics_by_horizon": horizons,
        "prediction_gate": {
            "all_finite": all_finite,
            "target_better_than_constant_velocity_at_any_horizon": bool(any(target_better)),
            "target_better_than_constant_velocity_all_horizons": bool(all(target_better)),
            "auxiliary_tasks_present": True,
            "accepted_for_development_control_smoke": bool(all_finite and any(target_better)),
        },
    }
    args.output.resolve().write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
