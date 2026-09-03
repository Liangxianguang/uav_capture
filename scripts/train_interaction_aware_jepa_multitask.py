"""Train JEPA-v3 multi-task interaction-aware predictors with TensorBoard logs."""

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

from encirclement3d.prediction import (  # noqa: E402
    InteractionAwareActionConditionedMultitaskJEPAPredictor,
    build_action_conditioned_predictor,
    deterministic_mse,
    gaussian_nll,
)


MODEL_TYPE = "interaction_aware_action_conditioned_jepa_multitask"
REQUIRED_DATASET_ARRAYS = (
    "inputs",
    "action_history",
    "labels_relative",
    "labels_obstacle_clearance",
    "labels_inter_agent_clearance",
    "labels_target_visible",
    "labels_cbf_correction",
    "labels_cbf_intervention",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs/jepa_v3_development_protocol.yaml")
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--validation-metadata", type=Path, required=True)
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
    parser.add_argument("--histogram-interval", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("JEPA-v3 protocol must be a mapping.")
    if protocol.get("phase") != "development_only" or protocol.get("locked_test_opened") is not False:
        raise ValueError("JEPA-v3 training is only permitted in the closed development protocol.")
    if protocol.get("training", {}).get("tensorboard", {}).get("required") is not True:
        raise ValueError("JEPA-v3 protocol requires TensorBoard logging.")
    return protocol


def load_dataset(path: Path, metadata_path: Path, required_split: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("split") != required_split:
        raise ValueError(f"Expected {required_split} metadata, got {metadata.get('split')!r}.")
    boundary = metadata.get("information_boundary", {})
    if boundary.get("development_s3_or_locked_data_used_for_training") is not False:
        raise ValueError("Dataset does not establish development/locked data exclusion.")
    with np.load(path) as archive:
        missing = set(REQUIRED_DATASET_ARRAYS).difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        tensors = {name: torch.from_numpy(np.asarray(archive[name], dtype=np.float32)) for name in REQUIRED_DATASET_ARRAYS}
    samples = tensors["inputs"].shape[0]
    if tensors["inputs"].shape[1:] != (8, 63) or tensors["action_history"].shape[1:] != (8, 3):
        raise ValueError("Dataset does not match the frozen JEPA-v3 input contract.")
    for name, tensor in tensors.items():
        if tensor.shape[0] != samples or not torch.isfinite(tensor).all():
            raise ValueError(f"Dataset tensor {name} is invalid.")
    return tensors, metadata


def _loader(tensors: dict[str, torch.Tensor], batch_size: int, shuffle: bool, seed: int, pin_memory: bool) -> DataLoader[Any]:
    ordered = tuple(tensors[name] for name in REQUIRED_DATASET_ARRAYS)
    return DataLoader(
        TensorDataset(*ordered),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=pin_memory,
    )


def _losses(
    model: InteractionAwareActionConditionedMultitaskJEPAPredictor,
    batch: tuple[torch.Tensor, ...],
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    (
        inputs,
        actions,
        target,
        obstacle_clearance,
        inter_agent_clearance,
        target_visible,
        cbf_correction,
        cbf_intervention,
    ) = batch
    mean, log_variance, latent, auxiliary = model.forward_multitask(inputs, actions)
    target_nll = gaussian_nll(mean, log_variance, target)
    target_mse = deterministic_mse(mean, target)
    latent_mse = deterministic_mse(torch.tanh(latent), model.target_latent(target))
    obstacle_mse = F.mse_loss(auxiliary["obstacle_clearance"], obstacle_clearance)
    inter_agent_mse = F.mse_loss(auxiliary["inter_agent_clearance"], inter_agent_clearance)
    clearance_mse = 0.5 * (obstacle_mse + inter_agent_mse)
    visibility_bce = F.binary_cross_entropy_with_logits(auxiliary["target_visibility_logit"], target_visible)
    cbf_correction_mse = F.mse_loss(auxiliary["cbf_correction"], cbf_correction)
    intervention_bce = F.binary_cross_entropy_with_logits(auxiliary["cbf_intervention_logit"], cbf_intervention)
    visibility_probability = torch.sigmoid(auxiliary["target_visibility_logit"])
    intervention_probability = torch.sigmoid(auxiliary["cbf_intervention_logit"])
    target_std = torch.exp(0.5 * log_variance)
    loss = (
        target_nll
        + weights["latent"] * latent_mse
        + weights["clearance"] * clearance_mse
        + weights["visibility"] * visibility_bce
        + weights["cbf_correction"] * cbf_correction_mse
        + weights["cbf_intervention"] * intervention_bce
    )
    return {
        "loss": loss,
        "target_nll": target_nll,
        "target_mse": target_mse,
        "latent_mse": latent_mse,
        "clearance_mse": clearance_mse,
        "obstacle_clearance_mse": obstacle_mse,
        "inter_agent_clearance_mse": inter_agent_mse,
        "visibility_bce": visibility_bce,
        "visibility_brier": F.mse_loss(visibility_probability, target_visible),
        "cbf_correction_mse": cbf_correction_mse,
        "cbf_intervention_bce": intervention_bce,
        "cbf_intervention_brier": F.mse_loss(intervention_probability, cbf_intervention),
        "target_one_std_coverage": (torch.abs(mean - target) <= target_std).float().mean(),
    }


def run_epoch(
    model: InteractionAwareActionConditionedMultitaskJEPAPredictor,
    loader: DataLoader[Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    weights: dict[str, float],
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    for source_batch in loader:
        batch = tuple(tensor.to(device, non_blocking=True) for tensor in source_batch)
        with torch.set_grad_enabled(training):
            metrics = _losses(model, batch, weights)
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
        raise RuntimeError("Empty JEPA-v3 data loader.")
    return {name: value / count for name, value in totals.items()}


def source_hashes() -> dict[str, str]:
    paths = (
        PROJECT_ROOT / "scripts" / "train_interaction_aware_jepa_multitask.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "prediction.py",
        PROJECT_ROOT / "configs" / "jepa_v3_development_protocol.yaml",
    )
    return {str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path) for path in paths}


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.batch_size, args.hidden_dim, args.latent_dim, args.num_layers, args.histogram_interval) <= 0:
        raise ValueError("epochs, batch size, dimensions, layers, and histogram interval must be positive.")
    weights = {
        "latent": args.latent_loss_weight,
        "clearance": args.clearance_loss_weight,
        "visibility": args.visibility_loss_weight,
        "cbf_correction": args.cbf_correction_loss_weight,
        "cbf_intervention": args.cbf_intervention_loss_weight,
    }
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0 or any(value < 0.0 for value in weights.values()):
        raise ValueError("Learning rate must be positive; regularization and task weights must be non-negative.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output}")
    if args.tensorboard_logdir.exists() and any(args.tensorboard_logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard log directory: {args.tensorboard_logdir}")
    protocol = _load_protocol(args.protocol.resolve())
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    train_tensors, train_metadata = load_dataset(args.train_dataset.resolve(), args.train_metadata.resolve(), "train")
    validation_tensors, validation_metadata = load_dataset(
        args.validation_dataset.resolve(), args.validation_metadata.resolve(), "validation"
    )
    train_loader = _loader(train_tensors, args.batch_size, True, args.seed, device.type == "cuda")
    validation_loader = _loader(validation_tensors, args.batch_size, False, args.seed, device.type == "cuda")
    model_config: dict[str, Any] = {
        "input_dim": 63,
        "horizon_count": int(train_tensors["labels_relative"].shape[1]),
        "action_dim": 3,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "num_layers": args.num_layers,
        "interaction_group_slices": protocol["model_contract"]["interaction_group_slices"],
    }
    model = build_action_conditioned_predictor(MODEL_TYPE, model_config).to(device)
    if not isinstance(model, InteractionAwareActionConditionedMultitaskJEPAPredictor):
        raise RuntimeError("Predictor factory did not create the JEPA-v3 multitask model.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    args.output.mkdir(parents=True, exist_ok=True)
    args.tensorboard_logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.tensorboard_logdir), flush_secs=10)
    writer.add_text("Config/protocol", yaml.safe_dump(protocol, sort_keys=True), 0)
    writer.add_text("Config/model", json.dumps(model_config, indent=2), 0)
    writer.add_text("Config/optimization", json.dumps({**vars(args), "weights": weights}, default=str, indent=2), 0)
    writer.add_text("Dataset/train_metadata", json.dumps(train_metadata, indent=2), 0)
    writer.add_text("Dataset/validation_metadata", json.dumps(validation_metadata, indent=2), 0)
    writer.add_text("Provenance/source_hashes", json.dumps(source_hashes(), indent=2), 0)
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = -1
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, weights)
        with torch.no_grad():
            validation_metrics = run_epoch(model, validation_loader, device, None, weights)
        record: dict[str, float | int] = {"epoch": epoch}
        for name, value in train_metrics.items():
            record[f"train_{name}"] = value
            writer.add_scalar(f"{_tensorboard_group(name)}/train", value, epoch)
        for name, value in validation_metrics.items():
            record[f"validation_{name}"] = value
            writer.add_scalar(f"{_tensorboard_group(name)}/validation", value, epoch)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        record["learning_rate"] = learning_rate
        writer.add_scalar("Optimization/learning_rate", learning_rate, epoch)
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
                    "task_weights": weights,
                    "source_hashes": source_hashes(),
                },
                args.output / "checkpoint.pt",
            )
        writer.flush()
    elapsed_seconds = time.perf_counter() - started
    writer.add_hparams(
        {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            **{f"weight_{name}": value for name, value in weights.items()},
        },
        {"hparam/best_validation_loss": best_validation_loss},
    )
    writer.close()
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (args.output / "run_metadata.json").write_text(
        json.dumps(
            {
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
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "elapsed_seconds": elapsed_seconds,
                "tensorboard_logdir": str(args.tensorboard_logdir.resolve()),
                "source_hashes": source_hashes(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"best_epoch": best_epoch, "best_validation_loss": best_validation_loss, "device": str(device), "tensorboard_logdir": str(args.tensorboard_logdir.resolve())}, indent=2))


def _tensorboard_group(metric: str) -> str:
    if "brier" in metric or "coverage" in metric:
        return "Calibration"
    if metric in {"loss", "target_nll", "latent_mse"}:
        return "Loss" if metric == "loss" else "Target"
    if "target" in metric:
        return "Target"
    if "clearance" in metric:
        return "Clearance"
    if "visibility" in metric:
        return "Visibility"
    if "cbf" in metric or "intervention" in metric:
        return "Risk"
    return "Metrics"


if __name__ == "__main__":
    main()
