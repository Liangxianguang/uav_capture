"""Evaluate a trained GRU target predictor on a locked dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.prediction import HistoryTargetPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    arrays_npz = np.load(args.dataset)
    inputs = np.asarray(arrays_npz["inputs"], dtype=np.float32)
    labels = np.asarray(arrays_npz["labels_relative"], dtype=np.float32)
    scenarios = np.asarray(arrays_npz["scenario_index"], dtype=np.int64)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint["model"]
    model = HistoryTargetPredictor(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = choose_device(args.device)
    model.to(device)
    model.eval()

    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], args.batch_size):
            batch = torch.from_numpy(inputs[start : start + args.batch_size]).to(device)
            mean, _log_variance = model(batch)
            predictions.append(mean.cpu().numpy())
    prediction = np.concatenate(predictions, axis=0)
    extent = float(metadata["normalization"]["position_extent_m"])
    errors = np.linalg.norm(prediction - labels, axis=2) * extent
    horizon_steps = [int(step) for step in metadata["horizon_steps"]]
    output: dict[str, Any] = {
        "task": metadata["task"],
        "split": metadata["split"],
        "checkpoint": str(args.checkpoint),
        "samples": int(labels.shape[0]),
        "device": str(device),
        "horizon_steps": horizon_steps,
        "horizon_seconds": [float(step * metadata["config"]["world"]["dt"]) for step in horizon_steps],
        "overall": {},
        "by_scenario_index": {},
    }
    for index, step in enumerate(horizon_steps):
        output["overall"][str(step)] = {
            "horizon_seconds": float(step * metadata["config"]["world"]["dt"]),
            "mean_position_error_m": float(np.mean(errors[:, index])),
            "median_position_error_m": float(np.median(errors[:, index])),
            "p90_position_error_m": float(np.quantile(errors[:, index], 0.90)),
            "n": int(errors.shape[0]),
        }
    for scenario in sorted(set(int(value) for value in scenarios)):
        mask = scenarios == scenario
        output["by_scenario_index"][str(scenario)] = {
            str(step): {
                "mean_position_error_m": float(np.mean(errors[mask, index])),
                "n": int(np.sum(mask)),
            }
            for index, step in enumerate(horizon_steps)
        }
    if args.baseline_json is not None:
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        output["constant_velocity_improvement_over_gru_percent"] = {
            str(step): float(
                100.0
                * (
                    baseline["baselines"]["constant_velocity"]["overall"][str(step)]["mean_position_error_m"]
                    - output["overall"][str(step)]["mean_position_error_m"]
                )
                / max(
                    baseline["baselines"]["constant_velocity"]["overall"][str(step)]["mean_position_error_m"],
                    1e-9,
                )
            )
            for step in horizon_steps
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
