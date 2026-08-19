"""Run reproducible partial-observation 3D capture-radius pursuit baselines."""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import os
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

# Keep the standalone evaluator consistent with the training launchers on
# Windows, where NumPy/Matplotlib and PyTorch can otherwise load two OpenMP
# runtimes in one process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.pursuit_controllers import (
    DynamicEncirclementController,
    PredictionPursuitController,
    PurePursuitController,
    SafetyFilteredPursuitController,
)
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv
from encirclement3d.reporting import plot_pursuit_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--controller",
        choices=("pure", "prediction", "encirclement", "pure_cbf", "prediction_cbf", "encirclement_cbf"),
        default="encirclement_cbf",
    )
    parser.add_argument("--episodes", type=int, help="Optional override for every scenario.")
    parser.add_argument("--scenario", type=str, help="Run one configured scenario.")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard event logging.")
    return parser.parse_args()


def controller_for(name: str, env: CaptureRadiusPursuit3DEnv) -> Any:
    use_safety_filter = name.endswith("_cbf")
    base_name = name.removesuffix("_cbf")
    mapping = {
        "pure": PurePursuitController,
        "prediction": PredictionPursuitController,
        "encirclement": DynamicEncirclementController,
    }
    controller = mapping[base_name](env)
    return SafetyFilteredPursuitController(controller) if use_safety_filter else controller


def run_episode(
    env: CaptureRadiusPursuit3DEnv,
    controller: Any,
    seed: int,
    record_history: bool,
) -> dict[str, Any]:
    observation = env.reset(seed=seed, record_history=record_history)
    correction_norms: list[float] = []
    barrier_values: list[float] = []
    visible_fractions: list[float] = []
    message_ages: list[float] = []
    final_info: dict[str, Any] = {}
    while True:
        action = controller.act(observation)
        diagnostics = getattr(controller, "last_diagnostics", None)
        if diagnostics is not None:
            correction_norms.append(float(diagnostics.action_correction_norm))
            barrier_values.append(float(diagnostics.minimum_barrier_value))
        observation, _reward, terminated, truncated, final_info = env.step(action, record_history=record_history)
        visible_fractions.append(float(final_info["target_visible_fraction"]))
        message_ages.append(float(final_info["mean_message_age_steps"]))
        if terminated or truncated:
            break

    return {
        "seed": seed,
        "success": bool(final_info["success"]),
        "safe_capture_success": bool(final_info["safe_capture_success"]),
        "capture_event": bool(final_info["capture_event"]),
        "capture_time_seconds": final_info["capture_time_seconds"],
        "capturing_defender_id": final_info["capturing_defender_id"],
        "relative_speed_at_capture": final_info["relative_speed_at_capture"],
        "steps": int(env.step_count),
        "termination_reason": str(final_info["termination_reason"]),
        "collision_steps": int(final_info["collision_steps"]),
        "world_violation_steps": int(final_info["world_violation_steps"]),
        "physical_target_contact": bool(final_info["physical_target_contact"]),
        "min_clearance": float(final_info["min_clearance_so_far"]),
        "nearest_target_distance": float(final_info["nearest_target_distance"]),
        "mean_target_visible_fraction": float(np.mean(visible_fractions)) if visible_fractions else 0.0,
        "mean_message_age_steps": float(np.mean(message_ages)) if message_ages else 0.0,
        "mean_observation_confidence": float(final_info.get("mean_observation_confidence", 0.0)),
        "mean_observation_age_steps": float(final_info.get("mean_observation_age_steps", 0.0)),
        "mean_observation_covariance_trace": float(
            final_info.get("mean_observation_covariance_trace", 0.0)
        ),
        "mean_action_correction": float(np.mean(correction_norms)) if correction_norms else 0.0,
        "minimum_cbf_barrier": float(min(barrier_values)) if barrier_values else None,
    }


def config_for_experiment(config: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    """Apply per-scenario pursuit overrides without mutating the base config."""
    scenario_config = copy.deepcopy(config)
    overrides = experiment.get("pursuit_overrides", {})
    if overrides:
        if not isinstance(overrides, dict):
            raise ValueError("experiment.pursuit_overrides must be a mapping.")
        scenario_config.setdefault("task", {}).setdefault("pursuit", {}).update(overrides)
    return scenario_config


def write_artifacts(output: Path, config: dict[str, Any]) -> None:
    output.joinpath("config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output.joinpath("environment.txt").write_text(
        "\n".join(
            [
                f"python={sys.version.replace(chr(10), ' ')}",
                f"platform={platform.platform()}",
                f"numpy={version('numpy')}",
                f"matplotlib={version('matplotlib')}",
                f"PyYAML={version('PyYAML')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source_paths = [
        PROJECT_ROOT / "scripts" / "run_capture_radius_pursuit.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_env.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_controllers.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "reporting.py",
    ]
    hashes = {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    output.joinpath("source_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    payload: dict[str, dict[str, float | int | None]] = {}
    for scenario in sorted({str(row["scenario"]) for row in rows}):
        subset = [row for row in rows if row["scenario"] == scenario]
        capture_times = [float(row["capture_time_seconds"]) for row in subset if row["capture_time_seconds"] is not None]
        relative_speeds = [
            float(row["relative_speed_at_capture"]) for row in subset if row["relative_speed_at_capture"] is not None
        ]
        barriers = [float(row["minimum_cbf_barrier"]) for row in subset if row["minimum_cbf_barrier"] is not None]
        payload[scenario] = {
            "episodes": len(subset),
            "capture_rate": sum(bool(row["capture_event"]) for row in subset) / len(subset),
            "safe_capture_rate": sum(bool(row["safe_capture_success"]) for row in subset) / len(subset),
            "collision_episode_rate": sum(int(row["collision_steps"]) > 0 for row in subset) / len(subset),
            "world_violation_episode_rate": sum(int(row["world_violation_steps"]) > 0 for row in subset) / len(subset),
            "mean_capture_time_seconds": float(np.mean(capture_times)) if capture_times else None,
            "mean_relative_speed_at_capture": float(np.mean(relative_speeds)) if relative_speeds else None,
            "mean_minimum_clearance_m": float(np.mean([float(row["min_clearance"]) for row in subset])),
            "worst_minimum_clearance_m": float(min(float(row["min_clearance"]) for row in subset)),
            "mean_visible_fraction": float(np.mean([float(row["mean_target_visible_fraction"]) for row in subset])),
            "mean_message_age_steps": float(np.mean([float(row["mean_message_age_steps"]) for row in subset])),
            "mean_observation_confidence": float(
                np.mean([float(row["mean_observation_confidence"]) for row in subset])
            ),
            "mean_observation_age_steps": float(
                np.mean([float(row["mean_observation_age_steps"]) for row in subset])
            ),
            "mean_observation_covariance_trace": float(
                np.mean([float(row["mean_observation_covariance_trace"]) for row in subset])
            ),
            "mean_action_correction": float(np.mean([float(row["mean_action_correction"]) for row in subset])),
            "worst_cbf_barrier": float(min(barriers)) if barriers else None,
        }
    return payload


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.episodes is not None:
        if args.episodes <= 0:
            raise ValueError("--episodes must be positive.")
        for experiment in config["experiments"]:
            experiment["episodes"] = args.episodes
    if args.scenario is not None:
        config["experiments"] = [item for item in config["experiments"] if item["name"] == args.scenario]
        if not config["experiments"]:
            raise ValueError(f"Unknown scenario: {args.scenario}")
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_artifacts(output, config)
    tensorboard = None if args.no_tensorboard else SummaryWriter(
        log_dir=str(output / "tensorboard"),
        flush_secs=10,
    )
    if tensorboard is not None:
        tensorboard.add_text("Config/effective_benchmark", yaml.safe_dump(config, sort_keys=False), 0)

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    seed_blocks = config.get("seed_blocks", {})
    if not isinstance(seed_blocks, dict):
        raise ValueError("seed_blocks must be a mapping when provided.")
    evaluation_seed = int(seed_blocks.get("locked_test", config["seed"]))
    for scenario_index, experiment in enumerate(config["experiments"]):
        scenario_config = config_for_experiment(config, experiment)
        for episode_index in range(int(experiment["episodes"])):
            env = CaptureRadiusPursuit3DEnv(
                scenario_config,
                obstacle_count=int(experiment["obstacle_count"]),
                target_speed_scale=float(experiment["target_speed_scale"]),
            )
            controller = controller_for(args.controller, env)
            seed = evaluation_seed + scenario_index * 10_000 + episode_index
            row = run_episode(env, controller, seed=seed, record_history=episode_index == 0)
            row["scenario"] = str(experiment["name"])
            row["controller"] = args.controller
            row["target_speed_scale"] = float(experiment["target_speed_scale"])
            row["obstacle_count"] = int(experiment["obstacle_count"])
            row["target_motion_mode"] = str(scenario_config["task"]["pursuit"]["target_motion_mode"])
            row["obstacle_profile"] = str(scenario_config["task"]["pursuit"]["obstacle_profile"])
            rows.append(row)
            if tensorboard is not None:
                step = len(rows)
                for metric in (
                    "success",
                    "safe_capture_success",
                    "capture_event",
                    "collision_steps",
                    "world_violation_steps",
                    "min_clearance",
                    "mean_target_visible_fraction",
                    "mean_message_age_steps",
                    "mean_observation_confidence",
                    "mean_observation_age_steps",
                    "mean_observation_covariance_trace",
                    "mean_action_correction",
                ):
                    tensorboard.add_scalar(
                        f"Episode/{experiment['name']}/{metric}",
                        float(row[metric]),
                        step,
                    )
            if episode_index == 0:
                plot_pursuit_trajectory(
                    env,
                    output / f"trajectory_{experiment['name']}.png",
                    f"{experiment['name']} {args.controller} seed {seed}",
                )

    fieldnames = list(rows[0].keys())
    with output.joinpath("episodes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    output.joinpath("summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output.joinpath("capture_radius_summary.json").write_text(
        json.dumps(
            {
                "task_name": "partial_observable_3d_capture_radius_pursuit",
                "success_definition": "At least one defender enters r_capture before timeout without a safety failure.",
                "not_claimed": ["physical target contact", "virtual cage closure", "soft-body net capture", "real-flight capture"],
                "scenarios": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output.joinpath("run_metadata.json").write_text(
        json.dumps(
            {
                "controller": args.controller,
                "evaluation_seed": evaluation_seed,
                "seed_blocks": seed_blocks,
                "elapsed_seconds": time.perf_counter() - started,
                "episodes": len(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if tensorboard is not None:
        for scenario, values in summary.items():
            for metric, value in values.items():
                if isinstance(value, (int, float)) and value is not None:
                    tensorboard.add_scalar(f"Summary/{scenario}/{metric}", float(value), len(rows))
        tensorboard.flush()
        tensorboard.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
