"""Train a hard-context weighted JEPA evaluator for safe capture.

This is a development-only continuation of the v2 model. It keeps the exact
v2 serialized model type and runtime shape, but gives more optimizer weight to
offline-labelled difficult contexts. No development/locked rollouts are read,
and no CBF or evaluation threshold is changed by this trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping

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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.prediction import (  # noqa: E402
    InteractionAwareActionConditionedSafeCaptureJEPAPredictor,
    build_action_conditioned_predictor,
    deterministic_mse,
    gaussian_nll,
)
from train_jepa_safe_capture_v2 import (  # noqa: E402
    REQUIRED_ARRAYS,
    _load_json,
    _load_yaml,
    _sha256,
    _validate_metadata,
    choose_device,
    load_dataset,
    validate_train_validation_contract,
)


MODEL_TYPE = "interaction_aware_action_conditioned_jepa_safe_capture_v2"
TRAINING_VARIANT = "hard_context_weighted_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs/jepa_safe_capture_v3_next_phase.yaml")
    parser.add_argument("--training-config", type=Path, default=PROJECT_ROOT / "configs/jepa_safe_capture_v3_training.yaml")
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
    parser.add_argument("--hard-weight-cap", type=float, default=8.0)
    parser.add_argument("--occlusion-weight", type=float, default=1.5)
    parser.add_argument("--stale-weight", type=float, default=1.5)
    parser.add_argument("--obstacle-clearance-weight", type=float, default=2.0)
    parser.add_argument("--inter-agent-clearance-weight", type=float, default=2.0)
    parser.add_argument("--ttc-weight", type=float, default=2.0)
    parser.add_argument("--cbf-intervention-weight", type=float, default=1.0)
    parser.add_argument("--histogram-interval", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.ndim == 0 or weights.ndim != 1 or values.shape[0] != weights.shape[0]:
        raise ValueError("Per-sample values and weights have incompatible shapes.")
    return (values.reshape(values.shape[0], -1).mean(dim=1) * weights).sum() / weights.sum().clamp_min(1e-9)


def hard_context_weights(
    batch: Mapping[str, torch.Tensor],
    *,
    cap: float = 8.0,
    occlusion_weight: float = 1.5,
    stale_weight: float = 1.5,
    obstacle_clearance_weight: float = 2.0,
    inter_agent_clearance_weight: float = 2.0,
    ttc_weight: float = 2.0,
    cbf_intervention_weight: float = 1.0,
) -> torch.Tensor:
    """Compute bounded sample weights from offline safety-context labels."""

    required = {
        "labels_target_visible",
        "labels_observation_age",
        "labels_obstacle_clearance",
        "labels_inter_agent_clearance",
        "labels_pairwise_ttc",
        "labels_cbf_intervention",
    }
    missing = sorted(required.difference(batch))
    if missing:
        raise ValueError(f"Hard-context weighting is missing labels: {missing}")
    if cap < 1.0 or any(value < 0.0 for value in (occlusion_weight, stale_weight, obstacle_clearance_weight, inter_agent_clearance_weight, ttc_weight, cbf_intervention_weight)):
        raise ValueError("Hard-context cap must be >= 1 and weights must be non-negative.")
    visible = batch["labels_target_visible"].float()
    age = batch["labels_observation_age"].float()
    obstacle = batch["labels_obstacle_clearance"].float()
    inter_agent = batch["labels_inter_agent_clearance"].float()
    ttc = batch["labels_pairwise_ttc"].float()
    intervention = batch["labels_cbf_intervention"].float()
    shape = visible.shape
    if any(value.shape != shape for value in (age, obstacle, inter_agent, ttc, intervention)):
        raise ValueError("Hard-context labels must have a common [batch, horizon] shape.")
    occlusion = (1.0 - visible).mean(dim=1)
    stale = torch.clamp(age / 3.0, 0.0, 1.0).mean(dim=1)
    low_obstacle = torch.clamp((0.10 - obstacle) / 0.10, 0.0, 1.0).mean(dim=1)
    low_inter_agent = torch.clamp((0.08 - inter_agent) / 0.08, 0.0, 1.0).mean(dim=1)
    low_ttc = torch.clamp((2.0 - ttc) / 2.0, 0.0, 1.0).mean(dim=1)
    cbf = intervention.mean(dim=1)
    weights = 1.0 + (
        float(occlusion_weight) * occlusion
        + float(stale_weight) * stale
        + float(obstacle_clearance_weight) * low_obstacle
        + float(inter_agent_clearance_weight) * low_inter_agent
        + float(ttc_weight) * low_ttc
        + float(cbf_intervention_weight) * cbf
    )
    return weights.clamp_min(1.0).clamp_max(float(cap))


def pinball_per_sample(prediction: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    if prediction.shape != target.shape or not 0.0 < quantile < 1.0:
        raise ValueError("Pinball prediction/target shapes or quantile are invalid.")
    error = target - prediction
    return torch.maximum(float(quantile) * error, (float(quantile) - 1.0) * error).reshape(prediction.shape[0], -1).mean(dim=1)


def _per_sample_losses(
    model: InteractionAwareActionConditionedSafeCaptureJEPAPredictor,
    batch: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    quantile: float,
    ttc_clip_seconds: float,
    maximum_observation_age_steps: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    mean, log_variance, latent, auxiliary = model.forward_multitask(batch["inputs"], batch["action_history"])
    target = batch["labels_relative"]
    target_velocity = batch["labels_target_velocity"]
    target_acceleration = batch["labels_target_acceleration"]
    obstacle_clearance = batch["labels_obstacle_clearance"]
    inter_agent_clearance = batch["labels_inter_agent_clearance"]
    pairwise_ttc = batch["labels_pairwise_ttc"]
    target_visible = batch["labels_target_visible"]
    observation_age = batch["labels_observation_age"]
    cbf_correction = batch["labels_cbf_correction"]
    cbf_intervention = batch["labels_cbf_intervention"]
    cbf_qp_feasible = batch["labels_cbf_qp_feasible"]
    target_nll = 0.5 * (log_variance + (target - mean).square() / torch.exp(log_variance)).reshape(target.shape[0], -1).mean(dim=1)
    target_mse = (mean - target).square().reshape(target.shape[0], -1).mean(dim=1)
    latent_mse = (torch.tanh(latent) - model.target_latent(target)).square().reshape(target.shape[0], -1).mean(dim=1)
    clearance_quantile = 0.5 * (pinball_per_sample(auxiliary["obstacle_clearance_lower_quantile"], obstacle_clearance, quantile) + pinball_per_sample(auxiliary["inter_agent_clearance_lower_quantile"], inter_agent_clearance, quantile))
    visibility_bce = F.binary_cross_entropy_with_logits(auxiliary["target_visibility_logit"], target_visible, reduction="none").reshape(target.shape[0], -1).mean(dim=1)
    cbf_correction_mse = (auxiliary["cbf_correction"] - cbf_correction).square().reshape(target.shape[0], -1).mean(dim=1)
    intervention_bce = F.binary_cross_entropy_with_logits(auxiliary["cbf_intervention_logit"], cbf_intervention, reduction="none").reshape(target.shape[0], -1).mean(dim=1)
    velocity_mse = (auxiliary["target_velocity"] - target_velocity).square().reshape(target.shape[0], -1).mean(dim=1)
    acceleration_mse = (auxiliary["target_acceleration"] - target_acceleration).square().reshape(target.shape[0], -1).mean(dim=1)
    ttc_mse = F.smooth_l1_loss(auxiliary["pairwise_ttc"] / ttc_clip_seconds, pairwise_ttc / ttc_clip_seconds, reduction="none").reshape(target.shape[0], -1).mean(dim=1)
    age_mse = F.smooth_l1_loss(auxiliary["observation_age"] / maximum_observation_age_steps, observation_age / maximum_observation_age_steps, reduction="none").reshape(target.shape[0], -1).mean(dim=1)
    qp_bce = F.binary_cross_entropy_with_logits(auxiliary["cbf_qp_feasibility_logit"], cbf_qp_feasible, reduction="none").reshape(target.shape[0], -1).mean(dim=1)
    action_consistency_mse = (auxiliary["action_consistency"] - batch["action_history"][:, -1]).square().reshape(target.shape[0], -1).mean(dim=1)
    losses = {
        "target_nll": target_nll,
        "target_mse": target_mse,
        "latent_mse": latent_mse,
        "clearance_quantile": clearance_quantile,
        "visibility_bce": visibility_bce,
        "cbf_correction_mse": cbf_correction_mse,
        "cbf_intervention_bce": intervention_bce,
        "velocity_mse": velocity_mse,
        "acceleration_mse": acceleration_mse,
        "ttc_mse": ttc_mse,
        "observation_age_mse": age_mse,
        "qp_feasibility_bce": qp_bce,
        "action_consistency_mse": action_consistency_mse,
    }
    losses["loss"] = (
        target_nll
        + float(weights["latent"]) * latent_mse
        + (float(weights["clearance"]) + float(weights["quantile"])) * clearance_quantile
        + float(weights["visibility"]) * visibility_bce
        + float(weights["cbf_correction"]) * cbf_correction_mse
        + float(weights["cbf_intervention"]) * intervention_bce
        + float(weights["velocity"]) * velocity_mse
        + float(weights["acceleration"]) * acceleration_mse
        + float(weights["ttc"]) * ttc_mse
        + float(weights["observation_age"]) * age_mse
        + float(weights["qp_feasibility"]) * qp_bce
        + float(weights["action_consistency"]) * action_consistency_mse
    )
    losses["visibility_brier"] = (torch.sigmoid(auxiliary["target_visibility_logit"]) - target_visible).square().reshape(target.shape[0], -1).mean(dim=1)
    losses["cbf_intervention_brier"] = (torch.sigmoid(auxiliary["cbf_intervention_logit"]) - cbf_intervention).square().reshape(target.shape[0], -1).mean(dim=1)
    losses["qp_feasibility_brier"] = (torch.sigmoid(auxiliary["cbf_qp_feasibility_logit"]) - cbf_qp_feasible).square().reshape(target.shape[0], -1).mean(dim=1)
    losses["target_one_std_coverage"] = (torch.abs(mean - target) <= torch.exp(0.5 * log_variance)).float().reshape(target.shape[0], -1).mean(dim=1)
    losses["target_mean_std"] = torch.exp(0.5 * log_variance).reshape(target.shape[0], -1).mean(dim=1)
    return losses, hard_context_weights(batch, cap=8.0)


def run_epoch(
    model: InteractionAwareActionConditionedSafeCaptureJEPAPredictor,
    loader: DataLoader[Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    task_weights: Mapping[str, float],
    hard_weight_kwargs: Mapping[str, float],
    quantile: float,
    ttc_clip_seconds: float,
    maximum_observation_age_steps: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    weight_total = 0.0
    for source_batch in loader:
        batch = {name: tensor.to(device, non_blocking=True) for name, tensor in zip(REQUIRED_ARRAYS, source_batch)}
        with torch.set_grad_enabled(training):
            losses, hard_weights = _per_sample_losses(model, batch, task_weights, quantile, ttc_clip_seconds, maximum_observation_age_steps)
            # Recompute with the declared hard-context parameters; keeping this
            # explicit makes the weighting contract visible in the trace.
            hard_weights = hard_context_weights(batch, **hard_weight_kwargs)
            loss = _weighted_mean(losses["loss"], hard_weights)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        batch_size = int(batch["inputs"].shape[0])
        for name, value in losses.items():
            metric_value = _weighted_mean(value, hard_weights)
            totals[name] = totals.get(name, 0.0) + float(metric_value.detach()) * batch_size
        totals["hard_weight_mean"] = totals.get("hard_weight_mean", 0.0) + float(hard_weights.mean().detach()) * batch_size
        totals["hard_weight_p95"] = totals.get("hard_weight_p95", 0.0) + float(torch.quantile(hard_weights.detach(), 0.95)) * batch_size
        weight_total += float(hard_weights.sum().detach())
        count += batch_size
    if count == 0:
        raise RuntimeError("Hard-context data loader is empty.")
    return {name: value / count for name, value in totals.items()} | {"hard_weight_total": weight_total, "samples": float(count)}


def _loader(tensors: dict[str, torch.Tensor], batch_size: int, shuffle: bool, seed: int, pin_memory: bool) -> DataLoader[Any]:
    ordered = tuple(tensors[name] for name in REQUIRED_ARRAYS)
    return DataLoader(TensorDataset(*ordered), batch_size=batch_size, shuffle=shuffle, generator=torch.Generator().manual_seed(seed), num_workers=0, pin_memory=pin_memory)


def _git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def source_hashes(protocol_path: Path, training_config_path: Path) -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/encirclement3d/prediction.py",
        PROJECT_ROOT / "scripts/train_jepa_safe_capture_v2.py",
        protocol_path.resolve(),
        training_config_path.resolve(),
    )
    return {str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path) for path in paths}


METRIC_TAGS = {
    "loss": "Loss/total",
    "target_nll": "Loss/target",
    "velocity_mse": "Loss/velocity",
    "acceleration_mse": "Loss/acceleration",
    "clearance_quantile": "Loss/clearance",
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
    if min(args.epochs, args.batch_size, args.hidden_dim, args.latent_dim, args.num_layers, args.histogram_interval) <= 0:
        raise ValueError("epochs, batch size, dimensions, layers, and histogram interval must be positive.")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0 or not 0.0 < args.quantile < 1.0:
        raise ValueError("Learning rate/weight decay/quantile values are invalid.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output}")
    if args.tensorboard_logdir.exists() and any(args.tensorboard_logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard directory: {args.tensorboard_logdir}")
    protocol = _load_yaml(args.protocol.resolve())
    if protocol.get("phase") != "development_only" or protocol.get("locked_test_opened") is not False:
        raise ValueError("v3 training is only permitted in the closed development protocol.")
    if protocol.get("provenance", {}).get("require_tensorboard") is not True:
        raise ValueError("v3 protocol must require TensorBoard records.")
    training_config = _load_yaml(args.training_config.resolve())
    if training_config.get("model_type") != MODEL_TYPE or training_config.get("training_variant") != TRAINING_VARIANT or training_config.get("phase") != "development_only" or training_config.get("locked_test_opened") is not False:
        raise ValueError("v3 training config model/variant/phase/locked contract is invalid.")
    configured_seeds = [int(value) for value in training_config.get("seeds", [])]
    if args.seed not in configured_seeds:
        raise ValueError(f"Seed {args.seed} is not declared in training config: {configured_seeds}")
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
        raise ValueError("Train/calibration episode seeds overlap.")
    if set(validation_metadata["episode_seeds"]).intersection(calibration_metadata["episode_seeds"]):
        raise ValueError("Validation/calibration episode seeds overlap.")
    collection = _load_yaml(PROJECT_ROOT / "configs/jepa_safe_capture_v2_collection.yaml")
    ttc_clip_seconds = float(train_metadata["ttc_clip_seconds"])
    maximum_observation_age_steps = float(collection["task"]["pursuit"]["maximum_message_age_steps"])
    task_weights = {
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
    hard_weight_kwargs = {
        "cap": args.hard_weight_cap,
        "occlusion_weight": args.occlusion_weight,
        "stale_weight": args.stale_weight,
        "obstacle_clearance_weight": args.obstacle_clearance_weight,
        "inter_agent_clearance_weight": args.inter_agent_clearance_weight,
        "ttc_weight": args.ttc_weight,
        "cbf_intervention_weight": args.cbf_intervention_weight,
    }
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
        raise RuntimeError("v3 predictor factory returned the wrong model type.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    args.output.mkdir(parents=True, exist_ok=True)
    args.tensorboard_logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.tensorboard_logdir), flush_secs=10)
    hashes = source_hashes(args.protocol, args.training_config)
    writer.add_text("Config/protocol", yaml.safe_dump(protocol, sort_keys=False), 0)
    writer.add_text("Config/training_config", yaml.safe_dump(training_config, sort_keys=False), 0)
    writer.add_text("Config/model", json.dumps(model_config, indent=2), 0)
    writer.add_text("Config/hard_context_weights", json.dumps(hard_weight_kwargs, indent=2), 0)
    writer.add_text("Config/optimization", json.dumps({**vars(args), "task_weights": task_weights}, default=str, indent=2), 0)
    writer.add_text("Dataset/train_metadata", json.dumps(train_metadata, indent=2), 0)
    writer.add_text("Dataset/validation_metadata", json.dumps(validation_metadata, indent=2), 0)
    writer.add_text("Dataset/calibration_metadata", json.dumps(calibration_metadata, indent=2), 0)
    writer.add_text("Provenance/source_hashes", json.dumps(hashes, indent=2), 0)
    writer.add_scalar("Data/train_samples", int(train_tensors["inputs"].shape[0]), 0)
    writer.add_scalar("Data/validation_samples", int(validation_tensors["inputs"].shape[0]), 0)
    writer.add_scalar("Data/calibration_episodes", int(calibration_metadata["episode_seed_count"]), 0)
    train_loader = _loader(train_tensors, args.batch_size, True, args.seed, device.type == "cuda")
    validation_loader = _loader(validation_tensors, args.batch_size, False, args.seed, device.type == "cuda")
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = -1
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, task_weights, hard_weight_kwargs, args.quantile, ttc_clip_seconds, maximum_observation_age_steps)
        with torch.no_grad():
            validation_metrics = run_epoch(model, validation_loader, device, None, task_weights, hard_weight_kwargs, args.quantile, ttc_clip_seconds, maximum_observation_age_steps)
        record: dict[str, float | int] = {"epoch": epoch}
        for prefix, metrics in (("train", train_metrics), ("validation", validation_metrics)):
            for name, value in metrics.items():
                if name in {"samples", "hard_weight_total"}:
                    continue
                record[f"{prefix}_{name}"] = value
                tag = METRIC_TAGS.get(name)
                if tag is not None:
                    writer.add_scalar(f"{tag}/{prefix}", value, epoch)
            writer.add_scalar(f"HardContext/{prefix}_weight_mean", metrics["hard_weight_mean"], epoch)
            writer.add_scalar(f"HardContext/{prefix}_weight_p95", metrics["hard_weight_p95"], epoch)
            writer.add_scalar(f"Risk/{prefix}_qp_feasibility_bce", metrics["qp_feasibility_bce"], epoch)
            writer.add_scalar(f"Target/{prefix}_velocity_mse", metrics["velocity_mse"], epoch)
            writer.add_scalar(f"Target/{prefix}_acceleration_mse", metrics["acceleration_mse"], epoch)
            writer.add_scalar(f"Clearance/{prefix}_quantile_loss", metrics["clearance_quantile"], epoch)
            writer.add_scalar(f"Visibility/{prefix}_brier", metrics["visibility_brier"], epoch)
            writer.add_scalar(f"CBF/{prefix}_intervention_brier", metrics["cbf_intervention_brier"], epoch)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        writer.add_scalar("Optimization/learning_rate", learning_rate, epoch)
        record["learning_rate"] = learning_rate
        history.append(record)
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            torch.save({
                "model_type": MODEL_TYPE,
                "model_state_dict": model.state_dict(),
                "model": model_config,
                "seed": args.seed,
                "training_variant": TRAINING_VARIANT,
                "protocol": str(args.protocol.resolve()),
                "train_metadata": train_metadata,
                "validation_metadata": validation_metadata,
                "calibration_metadata": calibration_metadata,
                "task_weights": task_weights,
                "hard_weight_kwargs": hard_weight_kwargs,
                "quantile": args.quantile,
                "source_hashes": hashes,
            }, args.output / "checkpoint.pt")
        writer.flush()
        if epoch % args.histogram_interval == 0 or epoch == 1:
            for name, parameter in model.named_parameters():
                writer.add_histogram(f"Parameters/{name}", parameter.detach(), epoch)
                if parameter.grad is not None:
                    writer.add_histogram(f"Gradients/{name}", parameter.grad.detach(), epoch)
    elapsed_seconds = time.perf_counter() - started
    writer.add_hparams({"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "hard_weight_cap": args.hard_weight_cap}, {"hparam/best_validation_loss": best_validation_loss})
    writer.close()
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    run_metadata = {
        "model_type": MODEL_TYPE,
        "training_variant": TRAINING_VARIANT,
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
        "task_weights": task_weights,
        "hard_weight_kwargs": hard_weight_kwargs,
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
        "git_revision": _git_revision(),
        "locked_test_opened": False,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model_type": MODEL_TYPE, "training_variant": TRAINING_VARIANT, "seed": args.seed, "best_epoch": best_epoch, "best_validation_loss": best_validation_loss, "device": str(device), "tensorboard_logdir": str(args.tensorboard_logdir.resolve()), "locked_test_opened": False}, indent=2))


if __name__ == "__main__":
    main()
