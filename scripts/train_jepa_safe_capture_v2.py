"""Train the P2 safe-capture JEPA evaluator on the frozen P1 archive.

Only the train archive reaches the optimizer.  Validation is held out for the
prediction gate and calibration metadata is recorded as provenance for P3; no
calibration samples are loaded into this trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SUPPORTED_DATASET_VERSIONS = {
    "jepa_safe_capture_v2_p1",
    "jepa_safe_capture_v2_p1_corrected_frame",
    "jepa_safe_capture_l0_l3_v1",
    "jepa_safe_capture_l0_l3_v2",
}

from encirclement3d.prediction import (  # noqa: E402
    InteractionAwareActionConditionedSafeCaptureJEPAPredictor,
    build_action_conditioned_predictor,
    deterministic_mse,
    gaussian_nll,
)


MODEL_TYPE = "interaction_aware_action_conditioned_jepa_safe_capture_v2"
REQUIRED_ARRAYS = (
    "inputs",
    "action_history",
    "labels_relative",
    "labels_target_velocity",
    "labels_target_acceleration",
    "labels_obstacle_clearance",
    "labels_inter_agent_clearance",
    "labels_pairwise_ttc",
    "labels_target_visible",
    "labels_observation_age",
    "labels_cbf_correction",
    "labels_cbf_intervention",
    "labels_cbf_qp_feasible",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs/jepa_safe_capture_v2_protocol.yaml")
    parser.add_argument("--training-config", type=Path, default=PROJECT_ROOT / "configs/jepa_safe_capture_v2_training.yaml")
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--validation-metadata", type=Path, required=True)
    parser.add_argument("--calibration-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--latent-loss-weight", type=float, default=1.0)
    parser.add_argument("--clearance-loss-weight", type=float, default=0.5)
    parser.add_argument("--visibility-loss-weight", type=float, default=0.25)
    parser.add_argument("--cbf-correction-loss-weight", type=float, default=0.25)
    parser.add_argument("--cbf-intervention-loss-weight", type=float, default=0.25)
    parser.add_argument("--velocity-loss-weight", type=float, default=0.50)
    parser.add_argument("--acceleration-loss-weight", type=float, default=0.35)
    parser.add_argument("--quantile-loss-weight", type=float, default=0.50)
    parser.add_argument("--ttc-loss-weight", type=float, default=0.25)
    parser.add_argument("--observation-age-loss-weight", type=float, default=0.20)
    parser.add_argument("--qp-feasibility-loss-weight", type=float, default=0.35)
    parser.add_argument("--action-consistency-loss-weight", type=float, default=0.25)
    parser.add_argument("--quantile", type=float, default=0.10)
    parser.add_argument("--histogram-interval", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON mapping in {path}.")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return value


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("P2 protocol must be a mapping.")
    if protocol.get("phase") != "development_only" or protocol.get("locked_test_opened") is not False:
        raise ValueError("P2 training is only permitted in the closed development protocol.")
    if protocol.get("training", {}).get("tensorboard", {}).get("required") is not True:
        raise ValueError("P2 requires TensorBoard records.")
    return protocol


def _validate_metadata(metadata: dict[str, Any], expected_split: str) -> None:
    if metadata.get("dataset_version") not in SUPPORTED_DATASET_VERSIONS:
        raise ValueError("P2 requires a supported safe-capture v2 archive version.")
    if metadata.get("split") != expected_split:
        raise ValueError(f"Expected {expected_split} metadata, got {metadata.get('split')!r}.")
    boundary = metadata.get("information_boundary", {})
    if boundary.get("target_truth_used_only_for_offline_labels") is not True:
        raise ValueError("P2 metadata does not prove target truth is offline-only.")
    if boundary.get("locked_test_opened") is not False:
        raise ValueError("P2 metadata does not prove locked test is closed.")
    if int(metadata.get("history_length", 0)) != 8 or int(metadata.get("candidate_count", 0)) != 5:
        raise ValueError("P2 requires the frozen 8-step, 5-candidate P1 contract.")
    if metadata.get("candidate_action_semantics") != "constant_desired_action_chunk_execute_first_step_then_replan":
        raise ValueError("P2 candidate semantics differ from the P1 contract.")
    frame = metadata.get("target_relative_frame")
    if frame is not None and frame != "post_action_defender_position":
        raise ValueError("P2 target-relative labels must use the post-action defender frame.")
    if metadata.get("dataset_version") == "jepa_safe_capture_v2_p1_corrected_frame" and int(metadata.get("label_frame_correction_version", 0)) < 1:
        raise ValueError("Corrected-frame archive is missing label_frame_correction_version.")


def load_dataset(path: Path, metadata_path: Path, expected_split: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    metadata = _load_json(metadata_path)
    _validate_metadata(metadata, expected_split)
    with np.load(path) as archive:
        missing = set(REQUIRED_ARRAYS).difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing P2 arrays: {sorted(missing)}")
        tensors = {name: torch.from_numpy(np.asarray(archive[name], dtype=np.float32)) for name in REQUIRED_ARRAYS}
    samples = int(tensors["inputs"].shape[0])
    if samples <= 0 or tensors["inputs"].shape[1:] != (8, 63) or tensors["action_history"].shape[1:] != (8, 3):
        raise ValueError("P2 archive does not match the frozen input contract.")
    horizon_count = len(metadata.get("horizon_steps", []))
    for name, tensor in tensors.items():
        if tensor.shape[0] != samples or not torch.isfinite(tensor).all():
            raise ValueError(f"P2 tensor {name} is inconsistent or non-finite.")
        if name in {"labels_relative", "labels_target_velocity", "labels_target_acceleration"}:
            if tensor.shape[1:] != (horizon_count, 3):
                raise ValueError(f"P2 tensor {name} has shape {tuple(tensor.shape)}.")
        elif name.startswith("labels_") and tensor.shape[1:] != (horizon_count,):
            raise ValueError(f"P2 tensor {name} has shape {tuple(tensor.shape)}.")
    return tensors, metadata


def validate_train_validation_contract(train: dict[str, Any], validation: dict[str, Any]) -> None:
    fields = (
        "history_length",
        "horizon_steps",
        "candidate_count",
        "candidate_perturbation_mps",
        "sample_stride",
        "chunk_length_steps",
        "candidate_action_semantics",
        "candidate_chunk_is_constant",
        "action_history_normalization",
        "action_scale",
        "label_units",
    )
    mismatch = {field: {"train": train.get(field), "validation": validation.get(field)} for field in fields if train.get(field) != validation.get(field)}
    if mismatch:
        raise ValueError(f"P2 train/validation contracts differ: {mismatch}")
    overlap = sorted(set(train.get("episode_seeds", [])).intersection(validation.get("episode_seeds", [])))
    if overlap:
        raise ValueError(f"P2 train/validation episode seeds overlap: {overlap[:8]}")


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    if prediction.shape != target.shape or not 0.0 < quantile < 1.0:
        raise ValueError("Pinball prediction/target shapes or quantile are invalid.")
    error = target - prediction
    return torch.maximum(float(quantile) * error, (float(quantile) - 1.0) * error).mean()


def _losses(
    model: InteractionAwareActionConditionedSafeCaptureJEPAPredictor,
    batch: tuple[torch.Tensor, ...],
    weights: dict[str, float],
    quantile: float,
    ttc_clip_seconds: float,
    maximum_observation_age_steps: float,
) -> dict[str, torch.Tensor]:
    (
        inputs,
        actions,
        target,
        target_velocity,
        target_acceleration,
        obstacle_clearance,
        inter_agent_clearance,
        pairwise_ttc,
        target_visible,
        observation_age,
        cbf_correction,
        cbf_intervention,
        cbf_qp_feasible,
    ) = batch
    mean, log_variance, latent, auxiliary = model.forward_multitask(inputs, actions)
    target_nll = gaussian_nll(mean, log_variance, target)
    target_mse = deterministic_mse(mean, target)
    latent_mse = deterministic_mse(torch.tanh(latent), model.target_latent(target))
    obstacle_quantile = pinball_loss(auxiliary["obstacle_clearance_lower_quantile"], obstacle_clearance, quantile)
    inter_agent_quantile = pinball_loss(auxiliary["inter_agent_clearance_lower_quantile"], inter_agent_clearance, quantile)
    clearance_quantile = 0.5 * (obstacle_quantile + inter_agent_quantile)
    visibility_bce = F.binary_cross_entropy_with_logits(auxiliary["target_visibility_logit"], target_visible)
    cbf_correction_mse = F.mse_loss(auxiliary["cbf_correction"], cbf_correction)
    intervention_bce = F.binary_cross_entropy_with_logits(auxiliary["cbf_intervention_logit"], cbf_intervention)
    velocity_mse = F.mse_loss(auxiliary["target_velocity"], target_velocity)
    acceleration_mse = F.mse_loss(auxiliary["target_acceleration"], target_acceleration)
    ttc_mse = F.smooth_l1_loss(auxiliary["pairwise_ttc"] / ttc_clip_seconds, pairwise_ttc / ttc_clip_seconds)
    age_mse = F.smooth_l1_loss(auxiliary["observation_age"] / maximum_observation_age_steps, observation_age / maximum_observation_age_steps)
    qp_bce = F.binary_cross_entropy_with_logits(auxiliary["cbf_qp_feasibility_logit"], cbf_qp_feasible)
    action_consistency_mse = F.mse_loss(auxiliary["action_consistency"], actions[:, -1])
    visibility_probability = torch.sigmoid(auxiliary["target_visibility_logit"])
    intervention_probability = torch.sigmoid(auxiliary["cbf_intervention_logit"])
    qp_probability = torch.sigmoid(auxiliary["cbf_qp_feasibility_logit"])
    target_std = torch.exp(0.5 * log_variance)
    loss = (
        target_nll
        + weights["latent"] * latent_mse
        + (weights["clearance"] + weights["quantile"]) * clearance_quantile
        + weights["visibility"] * visibility_bce
        + weights["cbf_correction"] * cbf_correction_mse
        + weights["cbf_intervention"] * intervention_bce
        + weights["velocity"] * velocity_mse
        + weights["acceleration"] * acceleration_mse
        + weights["ttc"] * ttc_mse
        + weights["observation_age"] * age_mse
        + weights["qp_feasibility"] * qp_bce
        + weights["action_consistency"] * action_consistency_mse
    )
    return {
        "loss": loss,
        "target_nll": target_nll,
        "target_mse": target_mse,
        "latent_mse": latent_mse,
        "clearance_quantile": clearance_quantile,
        "obstacle_quantile": obstacle_quantile,
        "inter_agent_quantile": inter_agent_quantile,
        "visibility_bce": visibility_bce,
        "visibility_brier": F.mse_loss(visibility_probability, target_visible),
        "cbf_correction_mse": cbf_correction_mse,
        "cbf_intervention_bce": intervention_bce,
        "cbf_intervention_brier": F.mse_loss(intervention_probability, cbf_intervention),
        "velocity_mse": velocity_mse,
        "acceleration_mse": acceleration_mse,
        "ttc_mse": ttc_mse,
        "observation_age_mse": age_mse,
        "qp_feasibility_bce": qp_bce,
        "qp_feasibility_brier": F.mse_loss(qp_probability, cbf_qp_feasible),
        "action_consistency_mse": action_consistency_mse,
        "target_one_std_coverage": (torch.abs(mean - target) <= target_std).float().mean(),
        "target_mean_std": target_std.mean(),
    }


def run_epoch(
    model: InteractionAwareActionConditionedSafeCaptureJEPAPredictor,
    loader: DataLoader[Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    weights: dict[str, float],
    quantile: float,
    ttc_clip_seconds: float,
    maximum_observation_age_steps: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    for source_batch in loader:
        batch = tuple(tensor.to(device, non_blocking=True) for tensor in source_batch)
        with torch.set_grad_enabled(training):
            metrics = _losses(model, batch, weights, quantile, ttc_clip_seconds, maximum_observation_age_steps)
            if training:
                optimizer.zero_grad(set_to_none=True)
                metrics["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        batch_size = int(batch[0].shape[0])
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
        count += batch_size
    if count == 0:
        raise RuntimeError("P2 data loader is empty.")
    return {name: value / count for name, value in totals.items()}


def _loader(tensors: dict[str, torch.Tensor], batch_size: int, shuffle: bool, seed: int, pin_memory: bool) -> DataLoader[Any]:
    ordered = tuple(tensors[name] for name in REQUIRED_ARRAYS)
    return DataLoader(
        TensorDataset(*ordered),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=pin_memory,
    )


def source_hashes(protocol_path: Path, training_config_path: Path) -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/encirclement3d/prediction.py",
        protocol_path.resolve(),
        PROJECT_ROOT / "configs/jepa_safe_capture_v2_collection.yaml",
        training_config_path.resolve(),
    )
    return {str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path) for path in paths}


METRIC_TAGS = {
    "loss": "Loss/total",
    "target_nll": "Loss/target",
    "velocity_mse": "Loss/velocity",
    "acceleration_mse": "Loss/acceleration",
    "clearance_quantile": "Loss/clearance",
    "inter_agent_quantile": "Loss/inter_agent",
    "visibility_bce": "Loss/visibility",
    "cbf_intervention_bce": "Loss/cbf_risk",
    "action_consistency_mse": "Loss/action_consistency",
    "target_one_std_coverage": "Calibration/target_one_std_coverage",
    "visibility_brier": "Calibration/visibility_brier",
    "cbf_intervention_brier": "Calibration/cbf_intervention_brier",
    "qp_feasibility_brier": "Calibration/qp_feasibility_brier",
    "target_mean_std": "Uncertainty/target_mean_std",
}


def main() -> None:
    args = parse_args()
    dimensions = (args.epochs, args.batch_size, args.hidden_dim, args.latent_dim, args.num_layers, args.histogram_interval)
    if min(dimensions) <= 0:
        raise ValueError("epochs, batch size, dimensions, layers, and histogram interval must be positive.")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0 or not 0.0 < args.quantile < 1.0:
        raise ValueError("Learning rate/weight decay/quantile values are invalid.")
    weights = {
        "latent": args.latent_loss_weight,
        "clearance": args.clearance_loss_weight,
        "visibility": args.visibility_loss_weight,
        "cbf_correction": args.cbf_correction_loss_weight,
        "cbf_intervention": args.cbf_intervention_loss_weight,
        "velocity": args.velocity_loss_weight,
        "acceleration": args.acceleration_loss_weight,
        "quantile": args.quantile_loss_weight,
        "ttc": args.ttc_loss_weight,
        "observation_age": args.observation_age_loss_weight,
        "qp_feasibility": args.qp_feasibility_loss_weight,
        "action_consistency": args.action_consistency_loss_weight,
    }
    if any(value < 0.0 for value in weights.values()):
        raise ValueError("Task loss weights must be non-negative.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output}")
    if args.tensorboard_logdir.exists() and any(args.tensorboard_logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard directory: {args.tensorboard_logdir}")
    protocol = _load_protocol(args.protocol.resolve())
    training_config = _load_yaml(args.training_config.resolve())
    if training_config.get("model_type") != MODEL_TYPE or training_config.get("phase") != "development_only" or training_config.get("locked_test_opened") is not False:
        raise ValueError("P2 training config model/phase/locked contract is invalid.")
    configured_seeds = [int(value) for value in training_config.get("seeds", [])]
    if args.seed not in configured_seeds:
        raise ValueError(f"Seed {args.seed} is not declared in the P2 training config: {configured_seeds}")
    calibration_metadata = _load_json(args.calibration_metadata.resolve())
    _validate_metadata(calibration_metadata, "calibration")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    train_tensors, train_metadata = load_dataset(args.train_dataset.resolve(), args.train_metadata.resolve(), "train")
    validation_tensors, validation_metadata = load_dataset(args.validation_dataset.resolve(), args.validation_metadata.resolve(), "validation")
    validate_train_validation_contract(train_metadata, validation_metadata)
    if set(train_metadata["episode_seeds"]).intersection(calibration_metadata["episode_seeds"]):
        raise ValueError("P2 train/calibration episode seeds overlap.")
    if set(validation_metadata["episode_seeds"]).intersection(calibration_metadata["episode_seeds"]):
        raise ValueError("P2 validation/calibration episode seeds overlap.")
    ttc_clip_seconds = float(train_metadata["ttc_clip_seconds"])
    # The archive contract stores age in integer steps; use the environment's
    # frozen maximum message age rather than infer it from labels.
    # Use the collection bound to the archive rather than a legacy default.
    collection_path = Path(train_metadata["collection_config"])
    if not collection_path.is_absolute():
        collection_path = PROJECT_ROOT / collection_path
    collection = _load_yaml(collection_path.resolve())
    maximum_observation_age_steps = float(collection["task"]["pursuit"]["maximum_message_age_steps"])
    model_config: dict[str, Any] = {
        "input_dim": 63,
        "horizon_count": int(train_tensors["labels_relative"].shape[1]),
        "action_dim": 3,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "num_layers": args.num_layers,
        "interaction_group_slices": ((0, 15), (15, 33), (33, 48), (48, 63)),
        "ttc_clip_seconds": ttc_clip_seconds,
        "maximum_observation_age_steps": maximum_observation_age_steps,
    }
    model = build_action_conditioned_predictor(MODEL_TYPE, model_config).to(device)
    if not isinstance(model, InteractionAwareActionConditionedSafeCaptureJEPAPredictor):
        raise RuntimeError("P2 predictor factory returned the wrong model type.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    args.output.mkdir(parents=True, exist_ok=True)
    args.tensorboard_logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.tensorboard_logdir), flush_secs=10)
    hashes = source_hashes(args.protocol, args.training_config)
    writer.add_text("Config/protocol", yaml.safe_dump(protocol, sort_keys=False), 0)
    writer.add_text("Config/training_config", yaml.safe_dump(training_config, sort_keys=False), 0)
    writer.add_text("Config/model", json.dumps(model_config, indent=2), 0)
    writer.add_text("Config/optimization", json.dumps({**vars(args), "weights": weights}, default=str, indent=2), 0)
    writer.add_text("Dataset/train_metadata", json.dumps(train_metadata, indent=2), 0)
    writer.add_text("Dataset/validation_metadata", json.dumps(validation_metadata, indent=2), 0)
    writer.add_text("Dataset/calibration_metadata", json.dumps(calibration_metadata, indent=2), 0)
    writer.add_text("Provenance/source_hashes", json.dumps(hashes, indent=2), 0)
    writer.add_scalar("Data/train_samples", int(train_tensors["inputs"].shape[0]), 0)
    writer.add_scalar("Data/validation_samples", int(validation_tensors["inputs"].shape[0]), 0)
    writer.add_scalar("Data/calibration_episodes", int(calibration_metadata["episode_seed_count"]), 0)
    writer.add_scalar("Data/train_nominal_fraction", float(train_metadata["candidate_is_nominal_fraction"]), 0)
    train_loader = _loader(train_tensors, args.batch_size, True, args.seed, device.type == "cuda")
    validation_loader = _loader(validation_tensors, args.batch_size, False, args.seed, device.type == "cuda")
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = -1
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, weights, args.quantile, ttc_clip_seconds, maximum_observation_age_steps)
        with torch.no_grad():
            validation_metrics = run_epoch(model, validation_loader, device, None, weights, args.quantile, ttc_clip_seconds, maximum_observation_age_steps)
        record: dict[str, float | int] = {"epoch": epoch}
        for prefix, metrics in (("train", train_metrics), ("validation", validation_metrics)):
            for name, value in metrics.items():
                record[f"{prefix}_{name}"] = value
                tag = METRIC_TAGS.get(name)
                if tag is not None:
                    writer.add_scalar(f"{tag}/{prefix}", value, epoch)
            writer.add_scalar(f"Risk/{prefix}_qp_feasibility_bce", metrics["qp_feasibility_bce"], epoch)
            writer.add_scalar(f"Target/{prefix}_velocity_mse", metrics["velocity_mse"], epoch)
            writer.add_scalar(f"Target/{prefix}_acceleration_mse", metrics["acceleration_mse"], epoch)
            writer.add_scalar(f"Clearance/{prefix}_quantile_loss", metrics["clearance_quantile"], epoch)
            writer.add_scalar(f"InterAgent/{prefix}_quantile_loss", metrics["inter_agent_quantile"], epoch)
            writer.add_scalar(f"Visibility/{prefix}_brier", metrics["visibility_brier"], epoch)
            writer.add_scalar(f"CBF/{prefix}_intervention_brier", metrics["cbf_intervention_brier"], epoch)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        writer.add_scalar("Optimization/learning_rate", learning_rate, epoch)
        record["learning_rate"] = learning_rate
        history.append(record)
        if epoch % args.histogram_interval == 0 or epoch == 1:
            for name, parameter in model.named_parameters():
                writer.add_histogram(f"Parameters/{name}", parameter.detach(), epoch)
                if parameter.grad is not None:
                    writer.add_histogram(f"Gradients/{name}", parameter.grad.detach(), epoch)
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_type": MODEL_TYPE,
                    "model_state_dict": model.state_dict(),
                    "model": model_config,
                    "seed": args.seed,
                    "protocol": str(args.protocol.resolve()),
                    "train_metadata": train_metadata,
                    "validation_metadata": validation_metadata,
                    "calibration_metadata": calibration_metadata,
                    "task_weights": weights,
                    "quantile": args.quantile,
                    "source_hashes": hashes,
                },
                args.output / "checkpoint.pt",
            )
        writer.flush()
    elapsed_seconds = time.perf_counter() - started
    hparams = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "quantile": args.quantile,
        **{f"weight_{name}": value for name, value in weights.items()},
    }
    hparam_metrics = {"hparam/best_validation_loss": best_validation_loss}
    try:
        writer.add_hparams(hparams, hparam_metrics)
    except (TypeError, ValueError, RuntimeError):
        # TensorBoard 2.4 with protobuf 4+ can reject the legacy hparams proto
        # at shutdown. Config text and scalar tags remain the authoritative
        # parameter record, so keep a compatible fallback instead of losing a
        # completed training run at the final flush.
        writer.add_text("Config/hparams_fallback", json.dumps(hparams, sort_keys=True), 0)
        for name, value in hparam_metrics.items():
            writer.add_scalar(name, float(value), 0)
    writer.close()
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    run_metadata = {
        "model_type": MODEL_TYPE,
        "device": str(device),
        "torch": version("torch"),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "model": model_config,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "task_weights": weights,
        "quantile": args.quantile,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "elapsed_seconds": elapsed_seconds,
        "tensorboard_logdir": str(args.tensorboard_logdir.resolve()),
        "train_dataset": str(args.train_dataset.resolve()),
        "train_dataset_sha256": _sha256(args.train_dataset.resolve()),
        "validation_dataset": str(args.validation_dataset.resolve()),
        "validation_dataset_sha256": _sha256(args.validation_dataset.resolve()),
        "calibration_metadata": str(args.calibration_metadata.resolve()),
        "calibration_metadata_sha256": _sha256(args.calibration_metadata.resolve()),
        "training_config": str(args.training_config.resolve()),
        "training_config_sha256": _sha256(args.training_config.resolve()),
        "source_hashes": hashes,
        "locked_test_opened": False,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model_type": MODEL_TYPE, "seed": args.seed, "best_epoch": best_epoch, "best_validation_loss": best_validation_loss, "device": str(device), "tensorboard_logdir": str(args.tensorboard_logdir.resolve()), "locked_test_opened": False}, indent=2))


if __name__ == "__main__":
    main()
