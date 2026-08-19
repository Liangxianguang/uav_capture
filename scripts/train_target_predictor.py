"""Train a GRU target trajectory predictor on local-history datasets."""

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
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.prediction import HistoryTargetPredictor, deterministic_mse, gaussian_nll


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--validation-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(path: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    arrays = np.load(path)
    required = {"inputs", "labels_relative"}
    missing = required.difference(arrays.files)
    if missing:
        raise ValueError(f"{path} is missing required arrays: {sorted(missing)}")
    metadata_path = path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    inputs = torch.from_numpy(np.asarray(arrays["inputs"], dtype=np.float32))
    labels = torch.from_numpy(np.asarray(arrays["labels_relative"], dtype=np.float32))
    return inputs, labels, metadata


def run_epoch(
    model: HistoryTargetPredictor,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    nll_total = 0.0
    mse_total = 0.0
    count = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(training):
            mean, log_variance = model(inputs)
            nll = gaussian_nll(mean, log_variance, labels)
            mse = deterministic_mse(mean, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                nll.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        batch = int(inputs.shape[0])
        nll_total += float(nll.detach().cpu()) * batch
        mse_total += float(mse.detach().cpu()) * batch
        count += batch
    if count == 0:
        raise RuntimeError("Empty prediction data loader.")
    return {"nll": nll_total / count, "mse": mse_total / count}


def source_hashes() -> dict[str, str]:
    paths = [
        PROJECT_ROOT / "scripts" / "train_target_predictor.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "prediction.py",
    ]
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.hidden_dim <= 0 or args.num_layers <= 0:
        raise ValueError("epochs, batch-size, hidden-dim, and num-layers must be positive.")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("learning-rate must be positive and weight-decay must be non-negative.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)

    train_inputs, train_labels, train_metadata = load_dataset(args.train_dataset)
    val_inputs, val_labels, val_metadata = load_dataset(args.validation_dataset)
    if train_inputs.shape[-1] != val_inputs.shape[-1] or train_labels.shape[-1] != val_labels.shape[-1]:
        raise ValueError("Train and validation feature/label dimensions do not match.")
    if train_labels.shape[1] != val_labels.shape[1]:
        raise ValueError("Train and validation horizon counts do not match.")

    train_loader = DataLoader(
        TensorDataset(train_inputs, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(val_inputs, val_labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = HistoryTargetPredictor(
        input_dim=int(train_inputs.shape[-1]),
        horizon_count=int(train_labels.shape[1]),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    writer = SummaryWriter(log_dir=str(args.output / "tensorboard"), flush_secs=10)
    writer.add_text("Dataset/train_metadata", json.dumps(train_metadata, indent=2), 0)
    writer.add_text("Dataset/validation_metadata", json.dumps(val_metadata, indent=2), 0)
    history: list[dict[str, float | int]] = []
    best_val = float("inf")
    best_epoch = -1
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device, optimizer=None)
        record = {
            "epoch": epoch,
            "train_nll": train_metrics["nll"],
            "train_mse": train_metrics["mse"],
            "validation_nll": val_metrics["nll"],
            "validation_mse": val_metrics["mse"],
        }
        history.append(record)
        for key, value in record.items():
            if key != "epoch":
                writer.add_scalar(f"Prediction/{key}", float(value), epoch)
        writer.add_scalar("Prediction/learning_rate", float(optimizer.param_groups[0]["lr"]), epoch)
        if val_metrics["nll"] < best_val:
            best_val = val_metrics["nll"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model": {
                        "input_dim": int(train_inputs.shape[-1]),
                        "horizon_count": int(train_labels.shape[1]),
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                    },
                    "seed": args.seed,
                    "train_metadata": train_metadata,
                    "validation_metadata": val_metadata,
                    "source_hashes": source_hashes(),
                },
                args.output / "checkpoint.pt",
            )
            writer.add_scalar("Selection/best_epoch", best_epoch, epoch)
        writer.flush()

    writer.close()
    args.output.joinpath("history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    args.output.joinpath("run_metadata.json").write_text(
        json.dumps(
            {
                "device": str(device),
                "torch": version("torch"),
                "python": sys.version.replace(chr(10), " "),
                "platform": platform.platform(),
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "best_epoch": best_epoch,
                "best_validation_nll": best_val,
                "elapsed_seconds": time.perf_counter() - started,
                "source_hashes": source_hashes(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"best_epoch": best_epoch, "best_validation_nll": best_val, "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
