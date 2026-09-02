"""Train the structured action-conditioned JEPA target-prediction pilot."""

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
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.prediction import (  # noqa: E402
    ActionConditionedJEPAPredictor,
    deterministic_mse,
    gaussian_nll,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--validation-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--latent-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu")


def load_dataset(path: Path, metadata_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    arrays = np.load(path)
    required = {"inputs", "action_history", "labels_relative"}
    missing = required.difference(arrays.files)
    if missing:
        raise ValueError(f"{path} is missing required arrays: {sorted(missing)}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    inputs = torch.from_numpy(np.asarray(arrays["inputs"], dtype=np.float32))
    actions = torch.from_numpy(np.asarray(arrays["action_history"], dtype=np.float32))
    labels = torch.from_numpy(np.asarray(arrays["labels_relative"], dtype=np.float32))
    if inputs.ndim != 3 or actions.ndim != 3 or labels.ndim != 3:
        raise ValueError("Action-conditioned datasets must contain rank-3 input/action/label arrays.")
    if inputs.shape[0] != actions.shape[0] or inputs.shape[0] != labels.shape[0] or inputs.shape[1] != actions.shape[1]:
        raise ValueError("Dataset sample and history dimensions do not agree.")
    return inputs, actions, labels, metadata


def run_epoch(
    model: ActionConditionedJEPAPredictor,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    latent_loss_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "nll": 0.0, "mse": 0.0, "latent_mse": 0.0}
    count = 0
    for inputs, actions, labels in loader:
        inputs, actions, labels = inputs.to(device), actions.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            mean, log_variance, latent = model(inputs, actions)
            nll = gaussian_nll(mean, log_variance, labels)
            mse = deterministic_mse(mean, labels)
            latent_mse = deterministic_mse(torch.tanh(latent), model.target_latent(labels))
            loss = nll + float(latent_loss_weight) * latent_mse
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        batch = int(inputs.shape[0])
        totals["loss"] += float(loss.detach()) * batch
        totals["nll"] += float(nll.detach()) * batch
        totals["mse"] += float(mse.detach()) * batch
        totals["latent_mse"] += float(latent_mse.detach()) * batch
        count += batch
    if count == 0:
        raise RuntimeError("Empty action-conditioned data loader.")
    return {key: value / count for key, value in totals.items()}


def source_hashes() -> dict[str, str]:
    paths = [
        PROJECT_ROOT / "scripts" / "train_action_conditioned_jepa_predictor.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "prediction.py",
    ]
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.batch_size, args.hidden_dim, args.latent_dim, args.num_layers) <= 0:
        raise ValueError("epochs, batch-size, hidden-dim, latent-dim, and num-layers must be positive.")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0 or args.latent_loss_weight < 0.0:
        raise ValueError("learning-rate must be positive; regularization weights must be non-negative.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    train_inputs, train_actions, train_labels, train_metadata = load_dataset(args.train_dataset, args.train_metadata)
    val_inputs, val_actions, val_labels, val_metadata = load_dataset(args.validation_dataset, args.validation_metadata)
    if train_inputs.shape[1:] != val_inputs.shape[1:] or train_actions.shape[1:] != val_actions.shape[1:]:
        raise ValueError("Train and validation action-conditioned input dimensions do not match.")
    if train_labels.shape[1:] != val_labels.shape[1:]:
        raise ValueError("Train and validation label dimensions do not match.")
    train_loader = DataLoader(
        TensorDataset(train_inputs, train_actions, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(val_inputs, val_actions, val_labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = ActionConditionedJEPAPredictor(
        input_dim=int(train_inputs.shape[-1]),
        horizon_count=int(train_labels.shape[1]),
        action_dim=int(train_actions.shape[-1]),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    writer = SummaryWriter(log_dir=str(args.output / "tensorboard"), flush_secs=10)
    writer.add_text("Dataset/train_metadata", json.dumps(train_metadata, indent=2), 0)
    writer.add_text("Dataset/validation_metadata", json.dumps(val_metadata, indent=2), 0)
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = -1
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, args.latent_loss_weight)
        with torch.no_grad():
            validation_metrics = run_epoch(model, val_loader, device, None, args.latent_loss_weight)
        record = {"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()}, **{f"validation_{key}": value for key, value in validation_metrics.items()}}
        history.append(record)
        for key, value in record.items():
            if key != "epoch":
                writer.add_scalar(f"ActionConditionedJEPA/{key}", float(value), epoch)
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_type": "action_conditioned_jepa",
                    "model_state_dict": model.state_dict(),
                    "model": {
                        "input_dim": int(train_inputs.shape[-1]),
                        "horizon_count": int(train_labels.shape[1]),
                        "action_dim": int(train_actions.shape[-1]),
                        "hidden_dim": args.hidden_dim,
                        "latent_dim": args.latent_dim,
                        "num_layers": args.num_layers,
                    },
                    "seed": args.seed,
                    "train_metadata": train_metadata,
                    "validation_metadata": val_metadata,
                    "source_hashes": source_hashes(),
                },
                args.output / "checkpoint.pt",
            )
        writer.flush()
    writer.close()
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output / "run_metadata.json").write_text(
        json.dumps(
            {
                "model_type": "action_conditioned_jepa",
                "device": str(device),
                "torch": version("torch"),
                "python": sys.version.replace(chr(10), " "),
                "platform": platform.platform(),
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "hidden_dim": args.hidden_dim,
                "latent_dim": args.latent_dim,
                "num_layers": args.num_layers,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "latent_loss_weight": args.latent_loss_weight,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "elapsed_seconds": time.perf_counter() - started,
                "source_hashes": source_hashes(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"best_epoch": best_epoch, "best_validation_loss": best_validation_loss, "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
