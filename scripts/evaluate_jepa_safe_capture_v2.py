"""Evaluate a P2 safe-capture JEPA checkpoint on held-out P1 validation data."""

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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.prediction import (  # noqa: E402
    InteractionAwareActionConditionedSafeCaptureJEPAPredictor,
    build_action_conditioned_predictor,
)


MODEL_TYPE = "interaction_aware_action_conditioned_jepa_safe_capture_v2"
SUPPORTED_DATASET_VERSIONS = {
    "jepa_safe_capture_v2_p1",
    "jepa_safe_capture_v2_p1_corrected_frame",
    "jepa_safe_capture_l0_l3_v1",
    "jepa_safe_capture_l0_l3_v2",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(order, dtype=np.float64)
    ends = np.concatenate((np.flatnonzero(np.diff(sorted_scores)) + 1, [scores.size]))
    starts = np.concatenate(([0], ends[:-1]))
    mean_ranks = (starts + 1.0 + ends) / 2.0
    ranks[order] = np.repeat(mean_ranks, ends - starts)
    return float((np.sum(ranks[labels == 1]) - positives * (positives + 1) / 2.0) / (positives * negatives))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu")


def load_metadata(path: Path) -> dict[str, Any]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("dataset_version") not in SUPPORTED_DATASET_VERSIONS or metadata.get("split") != "validation":
        raise ValueError("P2 evaluation requires a supported validation metadata file.")
    if metadata.get("dataset_version") == "jepa_safe_capture_v2_p1_corrected_frame":
        if metadata.get("target_relative_frame") != "post_action_defender_position" or int(metadata.get("label_frame_correction_version", 0)) < 1:
            raise ValueError("Corrected-frame validation metadata is incomplete.")
    if metadata.get("information_boundary", {}).get("locked_test_opened") is not False:
        raise ValueError("P2 evaluation refuses metadata that opened locked test.")
    return metadata


def constant_velocity(inputs: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    collection = yaml.safe_load(Path(metadata["collection_config"]).read_text(encoding="utf-8"))
    extent = float(collection["world"]["half_extent_xy"])
    target_speed = float(collection["agents"]["target_max_speed"])
    seconds = np.asarray(metadata["horizon_seconds"], dtype=np.float32)
    relative = inputs[:, -1, 3:6]
    velocity = inputs[:, -1, 6:9]
    return relative[:, None, :] + velocity[:, None, :] * (seconds[None, :, None] * target_speed / extent)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    metadata = load_metadata(args.metadata.resolve())
    with np.load(args.dataset.resolve()) as archive:
        arrays = {name: np.asarray(archive[name], dtype=np.float32) for name in archive.files}
    required = {
        "inputs", "action_history", "labels_relative", "labels_target_velocity", "labels_target_acceleration",
        "labels_obstacle_clearance", "labels_inter_agent_clearance", "labels_pairwise_ttc",
        "labels_target_visible", "labels_observation_age", "labels_cbf_correction", "labels_cbf_intervention",
        "labels_cbf_qp_feasible",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"Validation dataset is missing arrays: {missing}")
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != MODEL_TYPE:
        raise ValueError("Checkpoint is not the P2 safe-capture model type.")
    model = build_action_conditioned_predictor(str(checkpoint["model_type"]), checkpoint["model"])
    if not isinstance(model, InteractionAwareActionConditionedSafeCaptureJEPAPredictor):
        raise RuntimeError("P2 checkpoint factory returned an unexpected model.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = choose_device(args.device)
    model = model.to(device).eval()
    inputs = arrays["inputs"]
    actions = arrays["action_history"]
    prediction_parts: dict[str, list[np.ndarray]] = {"target": [], "std": []}
    auxiliary_parts: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, inputs.shape[0], args.batch_size):
            stop = min(start + args.batch_size, inputs.shape[0])
            mean, log_variance, _latent, auxiliary = model.forward_multitask(
                torch.as_tensor(inputs[start:stop], device=device),
                torch.as_tensor(actions[start:stop], device=device),
            )
            prediction_parts["target"].append(mean.cpu().numpy())
            prediction_parts["std"].append(torch.exp(0.5 * log_variance).cpu().numpy())
            for key, value in auxiliary.items():
                auxiliary_parts.setdefault(key, []).append(value.cpu().numpy())
    prediction = np.concatenate(prediction_parts["target"], axis=0)
    std = np.concatenate(prediction_parts["std"], axis=0)
    auxiliary = {key: np.concatenate(values, axis=0) for key, values in auxiliary_parts.items()}
    if not np.isfinite(prediction).all() or not np.isfinite(std).all() or not all(np.isfinite(value).all() for value in auxiliary.values()):
        raise ValueError("P2 validation emitted non-finite predictions.")
    collection = yaml.safe_load(Path(metadata["collection_config"]).read_text(encoding="utf-8"))
    extent = float(collection["world"]["half_extent_xy"])
    target_max_speed = float(collection["agents"]["target_max_speed"])
    target_max_accel = float(collection["agents"]["target_max_acceleration"])
    baseline = constant_velocity(inputs, metadata)
    horizons: list[dict[str, Any]] = []
    for index, seconds in enumerate(metadata["horizon_seconds"]):
        target = arrays["labels_relative"][:, index]
        target_error = np.linalg.norm(prediction[:, index] - target, axis=1)
        baseline_error = np.linalg.norm(baseline[:, index] - target, axis=1)
        visibility_probability = 1.0 / (1.0 + np.exp(-auxiliary["target_visibility_logit"][:, index]))
        intervention_probability = 1.0 / (1.0 + np.exp(-auxiliary["cbf_intervention_logit"][:, index]))
        qp_probability = 1.0 / (1.0 + np.exp(-auxiliary["cbf_qp_feasibility_logit"][:, index]))
        horizons.append(
            {
                "horizon_seconds": float(seconds),
                "target_position_mae_m": float(np.mean(target_error) * extent),
                "constant_velocity_mae_m": float(np.mean(baseline_error) * extent),
                "target_improvement_over_constant_velocity_fraction": float(1.0 - np.mean(target_error) / max(np.mean(baseline_error), 1e-9)),
                "target_one_std_coverage": float(np.mean(np.abs(prediction[:, index] - target) <= std[:, index])),
                "target_velocity_mae_mps": float(np.mean(np.abs(auxiliary["target_velocity"][:, index] - arrays["labels_target_velocity"][:, index])) * target_max_speed),
                "target_acceleration_mae_mps2": float(np.mean(np.abs(auxiliary["target_acceleration"][:, index] - arrays["labels_target_acceleration"][:, index])) * target_max_accel),
                "obstacle_clearance_lower_quantile_mae_m": float(np.mean(np.abs(auxiliary["obstacle_clearance_lower_quantile"][:, index] - arrays["labels_obstacle_clearance"][:, index])) * extent),
                "inter_agent_clearance_lower_quantile_mae_m": float(np.mean(np.abs(auxiliary["inter_agent_clearance_lower_quantile"][:, index] - arrays["labels_inter_agent_clearance"][:, index])) * extent),
                "pairwise_ttc_mae_s": float(np.mean(np.abs(auxiliary["pairwise_ttc"][:, index] - arrays["labels_pairwise_ttc"][:, index]))),
                "visibility_brier": float(np.mean((visibility_probability - arrays["labels_target_visible"][:, index]) ** 2)),
                "visibility_auc": binary_auc(arrays["labels_target_visible"][:, index], visibility_probability),
                "observation_age_mae_steps": float(np.mean(np.abs(auxiliary["observation_age"][:, index] - arrays["labels_observation_age"][:, index]))),
                "cbf_correction_mae_mps": float(np.mean(np.abs(auxiliary["cbf_correction"][:, index] - arrays["labels_cbf_correction"][:, index]))),
                "cbf_intervention_brier": float(np.mean((intervention_probability - arrays["labels_cbf_intervention"][:, index]) ** 2)),
                "cbf_intervention_auc": binary_auc(arrays["labels_cbf_intervention"][:, index], intervention_probability),
                "qp_feasibility_brier": float(np.mean((qp_probability - arrays["labels_cbf_qp_feasible"][:, index]) ** 2)),
                "qp_feasibility_auc": binary_auc(arrays["labels_cbf_qp_feasible"][:, index], qp_probability),
            }
        )
    summary = {
        "evaluation_type": "jepa_safe_capture_v2_p2_prediction_gate",
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "training_seed": int(checkpoint.get("seed", -1)),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset.resolve()),
        "metadata_sha256": sha256(args.metadata.resolve()),
        "samples": int(inputs.shape[0]),
        "device": str(device),
        "metrics_by_horizon": horizons,
        "prediction_gate": {
            "all_finite": True,
            "target_better_than_constant_velocity_at_any_horizon": bool(any(item["target_improvement_over_constant_velocity_fraction"] > 0.0 for item in horizons)),
            "target_better_than_constant_velocity_all_horizons": bool(all(item["target_improvement_over_constant_velocity_fraction"] > 0.0 for item in horizons)),
            "auxiliary_tasks_present": sorted(auxiliary),
        },
    }
    args.tensorboard_logdir.resolve().mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.tensorboard_logdir.resolve()), flush_secs=10)
    writer.add_text("Config/evaluation", json.dumps(vars(args), default=str, indent=2), 0)
    writer.add_text("Provenance/summary", json.dumps(summary, indent=2), 0)
    for index, item in enumerate(horizons):
        step = index + 1
        for name, value in item.items():
            if isinstance(value, (int, float)) and value is not None:
                writer.add_scalar(f"Eval/{name}", float(value), step)
    writer.flush()
    writer.close()
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
