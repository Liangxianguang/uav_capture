"""Evaluate action-conditioned predictors against a constant-velocity baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.prediction import build_action_conditioned_predictor  # noqa: E402


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


def load(path: Path, metadata_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    arrays = np.load(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    inputs = np.asarray(arrays["inputs"], dtype=np.float32)
    actions = np.asarray(arrays["action_history"], dtype=np.float32)
    labels = np.asarray(arrays["labels_relative"], dtype=np.float32)
    if inputs.ndim != 3 or actions.ndim != 3 or labels.ndim != 3:
        raise ValueError("Dataset arrays must all be rank-3.")
    return inputs, actions, labels, metadata


def constant_velocity_baseline(inputs: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    config = metadata.get("config", {})
    world = config.get("world", {})
    agents = config.get("agents", {})
    extent = float(world.get("half_extent_xy", 10.0))
    target_max_speed = float(agents.get("target_max_speed", 3.6))
    seconds = np.asarray(metadata["horizon_seconds"], dtype=np.float32)
    relative = inputs[:, -1, 3:6]
    velocity_normalized = inputs[:, -1, 6:9]
    return relative[:, None, :] + velocity_normalized[:, None, :] * (
        seconds[None, :, None] * target_max_speed / extent
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    inputs, actions, labels, metadata = load(args.dataset.resolve(), args.metadata.resolve())
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=True)
    model = build_action_conditioned_predictor(str(checkpoint["model_type"]), checkpoint["model"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = choose_device(args.device)
    model = model.to(device).eval()
    prediction_chunks: list[np.ndarray] = []
    std_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], args.batch_size):
            stop = min(start + args.batch_size, inputs.shape[0])
            mean, log_variance, _latent = model(
                torch.as_tensor(inputs[start:stop], device=device),
                torch.as_tensor(actions[start:stop], device=device),
            )
            prediction_chunks.append(mean.detach().cpu().numpy())
            std_chunks.append(torch.exp(0.5 * log_variance).detach().cpu().numpy())
    prediction = np.concatenate(prediction_chunks, axis=0)
    predicted_std = np.concatenate(std_chunks, axis=0)
    baseline = constant_velocity_baseline(inputs, metadata)
    horizon_seconds = [float(value) for value in metadata["horizon_seconds"]]
    metrics: list[dict[str, float | int]] = []
    for index, seconds in enumerate(horizon_seconds):
        error = np.linalg.norm(prediction[:, index] - labels[:, index], axis=1)
        baseline_error = np.linalg.norm(baseline[:, index] - labels[:, index], axis=1)
        std_norm = np.linalg.norm(predicted_std[:, index], axis=1)
        metrics.append(
            {
                "horizon_index": index,
                "horizon_seconds": seconds,
                "predictor_mean_position_error_m": float(np.mean(error) * 10.0),
                "predictor_median_position_error_m": float(np.median(error) * 10.0),
                "constant_velocity_mean_position_error_m": float(np.mean(baseline_error) * 10.0),
                "improvement_over_constant_velocity_fraction": float(
                    1.0 - np.mean(error) / max(np.mean(baseline_error), 1e-9)
                ),
                "p90_position_error_m": float(np.quantile(error, 0.90) * 10.0),
                "mean_predicted_std_m": float(np.mean(std_norm) * 10.0),
                "finite_fraction": float(np.mean(np.isfinite(prediction[:, index]))),
            }
        )
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "model_type": checkpoint["model_type"],
        "device": str(device),
        "samples": int(labels.shape[0]),
        "metrics_by_horizon": metrics,
        "prediction_gate": {
            "all_finite": bool(np.isfinite(prediction).all() and np.isfinite(predicted_std).all()),
            "better_than_constant_velocity_at_any_horizon": bool(
                any(item["improvement_over_constant_velocity_fraction"] > 0.0 for item in metrics)
            ),
            "better_than_constant_velocity_at_all_horizons": bool(
                all(item["improvement_over_constant_velocity_fraction"] > 0.0 for item in metrics)
            ),
        },
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
