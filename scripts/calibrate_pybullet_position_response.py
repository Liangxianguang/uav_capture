"""Identify a one-step response model for the PyBullet position-PID interface.

The learned relation is deliberately empirical: the policy action is first
passed through the configured command-rate governor and then translated into a
position reference for ``DSLPIDControl``.  A kinematic CBF cannot assume that
this requested velocity becomes the vehicle's next velocity.  This calibration
records the actual response before constructing a safety shield around it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.pybullet_env import PYBULLET_DRONES_ROOT, PyBulletEncirclement3DEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_artifacts(output: Path, document: dict[str, Any], environment: dict[str, Any], settings: dict[str, Any]) -> None:
    output.joinpath("config.yaml").write_text(
        yaml.safe_dump(
            {
                "calibration_document": document,
                "effective_calibration": settings,
                "environment": environment,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
    output.joinpath("environment.txt").write_text(
        "\n".join(
            [
                f"python={sys.version.replace(chr(10), ' ')}",
                f"platform={platform.platform()}",
                f"numpy={package_version('numpy')}",
                f"torch={package_version('torch')}",
                f"tensorboard={package_version('tensorboard')}",
                f"pybullet={package_version('pybullet')}",
                f"cuda_available={torch.cuda.is_available()}",
                f"cuda_runtime={torch.version.cuda}",
                f"device_name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
                "",
                "pip_freeze:",
                freeze.rstrip(),
                "",
            ]
        ),
        encoding="utf-8",
    )
    source_paths = [
        PROJECT_ROOT / "scripts" / "calibrate_pybullet_position_response.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pybullet_env.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "environment.py",
        PYBULLET_DRONES_ROOT / "gym_pybullet_drones" / "envs" / "BaseAviary.py",
        PYBULLET_DRONES_ROOT / "gym_pybullet_drones" / "envs" / "CtrlAviary.py",
        PYBULLET_DRONES_ROOT / "gym_pybullet_drones" / "control" / "DSLPIDControl.py",
    ]
    output.joinpath("source_hashes.json").write_text(
        json.dumps(
            {str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(path) for path in source_paths},
            indent=2,
        ),
        encoding="utf-8",
    )


def command_for_step(step: int, episode: int, settings: dict[str, Any], n_defenders: int) -> np.ndarray:
    directions = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=np.float64,
    )
    segment = step // int(settings["segment_steps"])
    # Opposite direction pairs use equal magnitude to keep the swarm inside
    # the workspace while still exciting all response axes.
    magnitude_index = (episode + segment // len(directions)) % len(settings["command_magnitudes"])
    magnitude = float(settings["command_magnitudes"][magnitude_index])
    return np.repeat((directions[segment % len(directions)] * magnitude)[None, :], n_defenders, axis=0)


def fit_response(rows: list[dict[str, float | int]]) -> dict[str, float | list[float]]:
    previous_velocity = np.asarray([row["velocity_before"] for row in rows], dtype=np.float64)
    executed_command = np.asarray([row["executed_command"] for row in rows], dtype=np.float64)
    features = np.column_stack([previous_velocity, executed_command])

    def fit(target_name: str) -> dict[str, float | list[float]]:
        target = np.asarray([row[target_name] for row in rows], dtype=np.float64)
        coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(features, target, rcond=None)
        prediction = features @ coefficients
        residual = target - prediction
        total = float(np.sum((target - np.mean(target)) ** 2))
        explained = 1.0 - float(np.sum(residual**2)) / total if total > 1e-12 else 1.0
        return {
            "velocity_coefficient": float(coefficients[0]),
            "command_coefficient": float(coefficients[1]),
            "r2": explained,
            "mae": float(np.mean(np.abs(residual))),
            "absolute_error_p95": float(np.quantile(np.abs(residual), 0.95)),
        }

    return {
        "velocity_model": fit("velocity_after"),
        "displacement_velocity_model": fit("displacement_velocity"),
        "component_samples": len(rows),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    document = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    environment_path = Path(document["environment_config"])
    if not environment_path.is_absolute():
        environment_path = (args.config.parent / environment_path).resolve()
    environment = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    environment.setdefault("dynamics", {})["backend"] = "pybullet"
    settings = dict(document["calibration"])
    if args.seed is not None:
        settings["seed"] = args.seed
    required = {
        "seed",
        "episodes",
        "steps_per_episode",
        "segment_steps",
        "command_magnitudes",
        "obstacle_count",
        "target_speed_scale",
        "deterministic_algorithms",
    }
    missing = sorted(required.difference(settings))
    if missing:
        raise ValueError(f"Missing calibration settings: {', '.join(missing)}")
    if int(settings["episodes"]) <= 0 or int(settings["steps_per_episode"]) <= 0:
        raise ValueError("episodes and steps_per_episode must be positive")
    if int(settings["segment_steps"]) <= 0:
        raise ValueError("segment_steps must be positive")
    if int(settings["steps_per_episode"]) % int(settings["segment_steps"]) != 0:
        raise ValueError("steps_per_episode must be divisible by segment_steps")
    if int(settings["steps_per_episode"]) // int(settings["segment_steps"]) % 6 != 0:
        raise ValueError("the command schedule must contain complete six-direction cycles")
    if any(float(magnitude) <= 0.0 for magnitude in settings["command_magnitudes"]):
        raise ValueError("command_magnitudes must be positive")

    np.random.seed(int(settings["seed"]))
    torch.manual_seed(int(settings["seed"]))
    torch.use_deterministic_algorithms(bool(settings["deterministic_algorithms"]), warn_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    write_artifacts(args.output, document, environment, settings)
    writer = SummaryWriter(str(args.output / "tensorboard"), flush_secs=10)
    writer.add_text("Config/effective_calibration", f"```yaml\n{yaml.safe_dump(settings, sort_keys=False)}```", 0)
    samples: list[dict[str, float | int]] = []
    started = time.perf_counter()
    env = PyBulletEncirclement3DEnv(
        environment,
        obstacle_count=int(settings["obstacle_count"]),
        target_speed_scale=float(settings["target_speed_scale"]),
    )
    try:
        for episode in range(int(settings["episodes"])):
            observation = env.reset(seed=int(settings["seed"]) + episode, record_history=False)
            for step in range(int(settings["steps_per_episode"])):
                requested = command_for_step(step, episode, settings, env.n_defenders)
                position_before = np.asarray(observation["defender_positions"], dtype=np.float64).copy()
                velocity_before = np.asarray(observation["defender_velocities"], dtype=np.float64).copy()
                observation, _reward, terminated, truncated, info = env.step(requested, record_history=False)
                if terminated or truncated:
                    raise RuntimeError(
                        "Calibration excitation unexpectedly reached a terminal state; "
                        f"episode={episode}, step={step}, info={info}"
                    )
                position_after = np.asarray(observation["defender_positions"], dtype=np.float64)
                velocity_after = np.asarray(observation["defender_velocities"], dtype=np.float64)
                for agent in range(env.n_defenders):
                    for axis in range(3):
                        samples.append(
                            {
                                "episode": episode,
                                "seed": int(settings["seed"]) + episode,
                                "step": step,
                                "agent": agent,
                                "axis": axis,
                                "requested_command": float(requested[agent, axis]),
                                "executed_command": float(env.last_executed_defender_actions[agent, axis]),
                                "velocity_before": float(velocity_before[agent, axis]),
                                "velocity_after": float(velocity_after[agent, axis]),
                                "displacement_velocity": float(
                                    (position_after[agent, axis] - position_before[agent, axis]) / env.control_dt
                                ),
                            }
                        )
    finally:
        env.close()

    response_model = fit_response(samples)
    response_model.update(
        {
            "control_dt": env.control_dt,
            "position_reference_horizon": float(environment["dynamics"]["pybullet_position_horizon"]),
            "command_max_acceleration": float(environment["dynamics"].get("pybullet_command_max_acceleration", 0.0)),
            "sample_protocol": "common translational, axis-paired commands with zero target speed and no obstacles",
        }
    )
    with (args.output / "samples.csv").open("w", encoding="utf-8", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(samples[0].keys()))
        csv_writer.writeheader()
        csv_writer.writerows(samples)
    args.output.joinpath("response_model.json").write_text(json.dumps(response_model, indent=2), encoding="utf-8")
    for model_name in ("velocity_model", "displacement_velocity_model"):
        model = response_model[model_name]
        assert isinstance(model, dict)
        for metric, value in model.items():
            writer.add_scalar(f"Calibration/{model_name}/{metric}", float(value), 0)
    writer.flush()
    writer.close()
    args.output.joinpath("run_metadata.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": time.perf_counter() - started,
                "samples": len(samples),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "calibration_only": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(response_model, indent=2))


if __name__ == "__main__":
    main()
