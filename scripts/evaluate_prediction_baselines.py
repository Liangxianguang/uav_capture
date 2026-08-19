"""Evaluate zero-velocity and constant-velocity prediction baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def evaluate(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    labels = np.asarray(arrays["labels_relative"], dtype=np.float64)
    belief = np.asarray(arrays["belief_relative"], dtype=np.float64)
    velocity = np.asarray(arrays["belief_velocity"], dtype=np.float64)
    horizon_steps = [int(step) for step in metadata["horizon_steps"]]
    dt = float(metadata["config"]["world"]["dt"])
    target_speed = float(metadata["normalization"]["target_velocity_scale_mps"])
    extent = float(metadata["normalization"]["position_extent_m"])
    scenarios = np.asarray(arrays["scenario_index"], dtype=np.int64)
    outputs: dict[str, Any] = {
        "task": metadata["task"],
        "split": metadata["split"],
        "samples": int(labels.shape[0]),
        "horizon_steps": horizon_steps,
        "horizon_seconds": [float(step * dt) for step in horizon_steps],
        "normalization": metadata["normalization"],
        "baselines": {},
    }
    for name in ("zero_velocity", "constant_velocity"):
        errors = []
        for horizon_index, step in enumerate(horizon_steps):
            if name == "zero_velocity":
                prediction = np.repeat(belief[:, None, :], len(horizon_steps), axis=1)[:, horizon_index]
            else:
                prediction = belief + velocity * (float(step) * dt * target_speed / extent)
            error = np.linalg.norm(prediction - labels[:, horizon_index], axis=1) * extent
            errors.append(error)
        per_horizon = {}
        for horizon_index, step in enumerate(horizon_steps):
            error = errors[horizon_index]
            per_horizon[str(step)] = {
                "horizon_seconds": float(step * dt),
                "mean_position_error_m": float(np.mean(error)),
                "median_position_error_m": float(np.median(error)),
                "p90_position_error_m": float(np.quantile(error, 0.90)),
                "n": int(error.size),
            }
        scenario_summary: dict[str, Any] = {}
        for scenario_index in sorted(set(int(x) for x in scenarios)):
            mask = scenarios == scenario_index
            scenario_summary[str(scenario_index)] = {
                str(step): {
                    "mean_position_error_m": float(np.mean(errors[horizon_index][mask])),
                    "n": int(np.sum(mask)),
                }
                for horizon_index, step in enumerate(horizon_steps)
            }
        outputs["baselines"][name] = {
            "overall": per_horizon,
            "by_scenario_index": scenario_summary,
        }
    outputs["improvement_constant_vs_zero_percent"] = {
        str(step): float(
            100.0
            * (
                outputs["baselines"]["zero_velocity"]["overall"][str(step)]["mean_position_error_m"]
                - outputs["baselines"]["constant_velocity"]["overall"][str(step)]["mean_position_error_m"]
            )
            / max(outputs["baselines"]["zero_velocity"]["overall"][str(step)]["mean_position_error_m"], 1e-9)
        )
        for step in horizon_steps
    }
    return outputs


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    arrays_npz = np.load(args.dataset)
    arrays = {key: arrays_npz[key] for key in arrays_npz.files}
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    result = evaluate(arrays, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
