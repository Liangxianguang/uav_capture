"""Characterize PyBullet vertical braking from controlled falling states.

This is an interface-identification experiment, not a real-flight experiment.
Each condition resets a level Crazyflie state at a safe altitude, injects a
specified downward velocity, and applies the low-level vertical recovery
reference. The resulting stopping distances define only a simulator-specific
emergency envelope for the pinned PyBullet dynamics backend.
"""

from __future__ import annotations

import argparse
import copy
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
        PROJECT_ROOT / "scripts" / "calibrate_pybullet_vertical_braking.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pybullet_env.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "environment.py",
        PYBULLET_DRONES_ROOT / "gym_pybullet_drones" / "envs" / "BaseAviary.py",
        PYBULLET_DRONES_ROOT / "gym_pybullet_drones" / "envs" / "CtrlAviary.py",
        PYBULLET_DRONES_ROOT / "gym_pybullet_drones" / "control" / "DSLPIDControl.py",
    ]
    output.joinpath("source_hashes.json").write_text(
        json.dumps({str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(path) for path in source_paths}, indent=2),
        encoding="utf-8",
    )


def load(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    document = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    environment_path = Path(document["environment_config"])
    if not environment_path.is_absolute():
        environment_path = (args.config.parent / environment_path).resolve()
    environment = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    settings = dict(document["calibration"])
    if args.seed is not None:
        settings["seed"] = args.seed
    required = {
        "seed",
        "initial_altitude",
        "initial_descent_speeds",
        "emergency_climb_heights",
        "steps_per_condition",
        "obstacle_count",
        "target_speed_scale",
        "deterministic_algorithms",
    }
    missing = sorted(required.difference(settings))
    if missing:
        raise ValueError(f"Missing calibration settings: {', '.join(missing)}")
    if int(settings["steps_per_condition"]) <= 0:
        raise ValueError("steps_per_condition must be positive")
    if float(settings["initial_altitude"]) <= 0.0:
        raise ValueError("initial_altitude must be positive")
    if any(float(value) <= 0.0 for value in settings["initial_descent_speeds"]):
        raise ValueError("initial_descent_speeds must be positive magnitudes")
    if any(float(value) <= 0.0 for value in settings["emergency_climb_heights"]):
        raise ValueError("emergency_climb_heights must be positive")
    return document, environment, settings


def inject_level_falling_state(env: PyBulletEncirclement3DEnv, altitude: float, descent_speed: float) -> None:
    if env.pybullet is None or env.aviary is None:
        raise RuntimeError("PyBullet environment has not been initialized.")
    positions_xy = np.asarray(((-4.0, -4.0), (-4.0, 4.0), (4.0, -4.0), (4.0, 4.0)), dtype=np.float64)
    for index, body_id in enumerate(env.aviary.DRONE_IDS):
        env.pybullet.resetBasePositionAndOrientation(
            int(body_id),
            [float(positions_xy[index, 0]), float(positions_xy[index, 1]), altitude],
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=env.aviary.CLIENT,
        )
        env.pybullet.resetBaseVelocity(
            int(body_id),
            linearVelocity=[0.0, 0.0, -descent_speed],
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=env.aviary.CLIENT,
        )
    env.aviary._updateAndStoreKinematicInformation()
    env._sync_defender_state()
    env.last_pid_target_positions = env.defender_positions.copy()


def summarize_condition(rows: list[dict[str, float | int | bool]], initial_altitude: float) -> dict[str, float | int | bool | None]:
    z_values = [initial_altitude] + [float(row["z_after"]) for row in rows]
    decelerations = [float(row["vertical_deceleration"]) for row in rows if float(row["vz_before"]) < 0.0]
    stop_rows = [row for row in rows if float(row["vz_after"]) >= 0.0]
    first_stop = stop_rows[0] if stop_rows else None
    return {
        "condition": int(rows[0]["condition"]),
        "initial_descent_speed": float(rows[0]["initial_descent_speed"]),
        "emergency_climb_height": float(rows[0]["emergency_climb_height"]),
        "steps_recorded": len(rows),
        "minimum_altitude": float(min(z_values)),
        "maximum_vertical_deceleration": float(max(decelerations)) if decelerations else None,
        "mean_vertical_deceleration": float(np.mean(decelerations)) if decelerations else None,
        "stopped_before_floor": first_stop is not None and float(min(z_values)) >= float(rows[0]["floor_altitude"]),
        "stopping_distance": float(initial_altitude - min(z_values)) if first_stop is not None else None,
        "first_stop_step": int(first_stop["step"]) if first_stop is not None else None,
        "terminal_world_violation": bool(rows[-1]["world_violation"]),
        "terminal_physical_contact": bool(rows[-1]["physical_contact"]),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    document, environment, settings = load(args)
    lower_altitude = float(environment["world"]["minimum_altitude"])
    upper_altitude = float(environment["world"]["height"])
    initial_altitude = float(settings["initial_altitude"])
    if not lower_altitude < initial_altitude < upper_altitude:
        raise ValueError("initial_altitude must lie inside the configured flight volume")

    np.random.seed(int(settings["seed"]))
    torch.manual_seed(int(settings["seed"]))
    torch.use_deterministic_algorithms(bool(settings["deterministic_algorithms"]), warn_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    write_artifacts(args.output, document, environment, settings)
    writer = SummaryWriter(str(args.output / "tensorboard"), flush_secs=10)
    writer.add_text("Config/effective_calibration", f"```yaml\n{yaml.safe_dump(settings, sort_keys=False)}```", 0)
    sample_rows: list[dict[str, float | int | bool]] = []
    condition_rows: list[dict[str, float | int | bool | None]] = []
    started = time.perf_counter()
    condition = 0
    for descent_speed in (float(value) for value in settings["initial_descent_speeds"]):
        for climb_height in (float(value) for value in settings["emergency_climb_heights"]):
            condition_environment = copy.deepcopy(environment)
            dynamics = condition_environment.setdefault("dynamics", {})
            dynamics["backend"] = "pybullet"
            dynamics["pybullet_vertical_recovery_enabled"] = True
            dynamics["pybullet_vertical_recovery_altitude"] = upper_altitude - 0.05
            dynamics["pybullet_vertical_recovery_descend_speed"] = 0.01
            dynamics["pybullet_vertical_recovery_climb_height"] = climb_height
            env = PyBulletEncirclement3DEnv(
                condition_environment,
                obstacle_count=int(settings["obstacle_count"]),
                target_speed_scale=float(settings["target_speed_scale"]),
            )
            rows_for_condition: list[dict[str, float | int | bool]] = []
            try:
                env.reset(seed=int(settings["seed"]) + condition, record_history=False)
                inject_level_falling_state(env, initial_altitude, descent_speed)
                action = np.zeros((env.n_defenders, 3), dtype=np.float64)
                for step in range(int(settings["steps_per_condition"])):
                    z_before = float(env.defender_positions[0, 2])
                    vz_before = float(env.defender_velocities[0, 2])
                    _observation, _reward, terminated, truncated, info = env.step(action, record_history=False)
                    row: dict[str, float | int | bool] = {
                        "condition": condition,
                        "step": step + 1,
                        "initial_descent_speed": descent_speed,
                        "emergency_climb_height": climb_height,
                        "floor_altitude": lower_altitude,
                        "z_before": z_before,
                        "vz_before": vz_before,
                        "z_after": float(env.defender_positions[0, 2]),
                        "vz_after": float(env.defender_velocities[0, 2]),
                        "vertical_deceleration": float((env.defender_velocities[0, 2] - vz_before) / env.control_dt),
                        "roll_after": float(env.aviary.rpy[0, 0]) if env.aviary is not None else 0.0,
                        "pitch_after": float(env.aviary.rpy[0, 1]) if env.aviary is not None else 0.0,
                        "recovery_active": bool(info["vertical_recovery_active_agents"][0]),
                        "world_violation": bool(info["world_violation"]),
                        "physical_contact": bool(info["physical_contact"]),
                    }
                    rows_for_condition.append(row)
                    sample_rows.append(row)
                    if terminated or truncated:
                        break
            finally:
                env.close()
            summary = summarize_condition(rows_for_condition, initial_altitude)
            condition_rows.append(summary)
            for metric, value in summary.items():
                if isinstance(value, (int, float, bool)):
                    writer.add_scalar(f"Condition/{metric}", value, condition)
            writer.flush()
            condition += 1

    successful_decelerations = [
        float(row["maximum_vertical_deceleration"])
        for row in condition_rows
        if bool(row["stopped_before_floor"]) and row["maximum_vertical_deceleration"] is not None
    ]
    aggregate: dict[str, float | int | None] = {
        "conditions": len(condition_rows),
        "stopped_before_floor_conditions": int(sum(bool(row["stopped_before_floor"]) for row in condition_rows)),
        "conservative_vertical_deceleration_p05": float(np.quantile(successful_decelerations, 0.05))
        if successful_decelerations
        else None,
        "maximum_tested_descent_speed": float(max(settings["initial_descent_speeds"])),
    }
    for metric, value in aggregate.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"Aggregate/{metric}", value, 0)
    writer.flush()
    writer.close()

    with (args.output / "samples.csv").open("w", encoding="utf-8", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0].keys()))
        csv_writer.writeheader()
        csv_writer.writerows(sample_rows)
    with (args.output / "conditions.csv").open("w", encoding="utf-8", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(condition_rows[0].keys()))
        csv_writer.writeheader()
        csv_writer.writerows(condition_rows)
    args.output.joinpath("braking_model.json").write_text(
        json.dumps(
            {
                **aggregate,
                "control_dt": float(environment["world"]["dt"]),
                "initial_altitude": initial_altitude,
                "protocol": "Level attitude state reset, injected vertical velocity, persistent emergency position reference.",
                "scope": "Simulator interface identification only; not real-flight validation.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    args.output.joinpath("run_metadata.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": time.perf_counter() - started,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "calibration_only": True,
                "state_reset_protocol": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
