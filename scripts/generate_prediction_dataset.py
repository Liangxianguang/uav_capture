"""Generate local-history target-prediction data for the 3D capture task.

The predictor inputs are policy-safe observations only. Simulator target truth is
used exclusively to create offline labels and is never serialized as an input
feature.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.pursuit_controllers import (
    DynamicEncirclementController,
    PredictionPursuitController,
    PurePursuitController,
    SafetyFilteredPursuitController,
)
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "locked_test"), default="train")
    parser.add_argument("--episodes-per-scenario", type=int, help="Override configured episodes for this split.")
    parser.add_argument(
        "--controller",
        choices=("pure", "prediction", "encirclement", "pure_cbf", "prediction_cbf", "encirclement_cbf"),
        default="pure_cbf",
    )
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--horizon-steps", type=int, nargs="+", default=[1, 3, 5, 10])
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


def config_for_experiment(config: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    scenario_config = copy.deepcopy(config)
    overrides = experiment.get("pursuit_overrides", {})
    if overrides:
        if not isinstance(overrides, dict):
            raise ValueError("experiment.pursuit_overrides must be a mapping.")
        scenario_config.setdefault("task", {}).setdefault("pursuit", {}).update(overrides)
    return scenario_config


def seed_for_split(config: dict[str, Any], split: str) -> int:
    blocks = config.get("seed_blocks", {})
    if blocks and split not in blocks:
        raise ValueError(f"Config seed_blocks does not contain split {split!r}.")
    return int(blocks.get(split, config["seed"]))


def _append_episode(
    episode_records: list[dict[str, Any]],
    env: CaptureRadiusPursuit3DEnv,
    controller: Any,
    seed: int,
) -> dict[str, Any]:
    observation = env.reset(seed=seed)
    policy_history: list[np.ndarray] = []
    belief_history: list[np.ndarray] = []
    velocity_history: list[np.ndarray] = []
    confidence_history: list[np.ndarray] = []
    covariance_history: list[np.ndarray] = []
    message_age_history: list[np.ndarray] = []
    target_positions: list[np.ndarray] = []
    defender_positions: list[np.ndarray] = []

    def record(current_observation: dict[str, Any]) -> None:
        policy_history.append(env.policy_observations(current_observation).copy())
        belief_history.append(
            (
                np.asarray(current_observation["target_belief_positions"], dtype=np.float32)
                - np.asarray(current_observation["defender_positions"], dtype=np.float32)
            ).copy()
        )
        velocity_history.append(np.asarray(current_observation["target_belief_velocities"], dtype=np.float32).copy())
        confidence_history.append(
            np.asarray(current_observation["target_observation_confidence"], dtype=np.float32).copy()
        )
        covariance_history.append(
            np.asarray(current_observation["target_observation_covariance"], dtype=np.float32).copy()
        )
        message_age_history.append(np.asarray(current_observation["message_age_steps"], dtype=np.int64).copy())
        target_positions.append(env.target_position.astype(np.float32, copy=True))
        defender_positions.append(env.defender_positions.astype(np.float32, copy=True))

    record(observation)
    while True:
        action = controller.act(observation)
        observation, _reward, terminated, truncated, _info = env.step(action)
        record(observation)
        if terminated or truncated:
            break

    episode_records.append(
        {
            "policy": np.asarray(policy_history, dtype=np.float32),
            "belief_relative": np.asarray(belief_history, dtype=np.float32),
            "belief_velocity": np.asarray(velocity_history, dtype=np.float32),
            "confidence": np.asarray(confidence_history, dtype=np.float32),
            "covariance": np.asarray(covariance_history, dtype=np.float32),
            "message_age": np.asarray(message_age_history, dtype=np.int64),
            "target_positions": np.asarray(target_positions, dtype=np.float32),
            "defender_positions": np.asarray(defender_positions, dtype=np.float32),
            "seed": int(seed),
        }
    )
    return episode_records[-1]


def assemble_prediction_samples(
    episodes: list[dict[str, Any]],
    history_length: int,
    horizon_steps: list[int],
    extent: float,
    target_max_speed: float,
    dt: float,
) -> dict[str, np.ndarray]:
    if history_length <= 0:
        raise ValueError("history_length must be positive.")
    if not horizon_steps or any(step <= 0 for step in horizon_steps):
        raise ValueError("horizon_steps must contain positive integers.")
    horizon_steps = sorted(set(int(step) for step in horizon_steps))
    samples: dict[str, list[np.ndarray]] = {
        "inputs": [],
        "belief_relative": [],
        "belief_velocity": [],
        "confidence": [],
        "covariance": [],
        "message_age": [],
        "labels_relative": [],
        "agent_id": [],
        "time_index": [],
        "episode_seed": [],
        "scenario_index": [],
    }
    for episode in episodes:
        scenario_index = int(episode.get("scenario_index", 0))
        length = int(episode["policy"].shape[0])
        max_horizon = max(horizon_steps)
        if length < history_length + max_horizon:
            continue
        for time_index in range(history_length - 1, length - max_horizon):
            for agent in range(int(episode["policy"].shape[1])):
                samples["inputs"].append(episode["policy"][time_index - history_length + 1 : time_index + 1, agent])
                samples["belief_relative"].append(episode["belief_relative"][time_index, agent] / float(extent))
                samples["belief_velocity"].append(episode["belief_velocity"][time_index, agent] / float(target_max_speed))
                samples["confidence"].append(episode["confidence"][time_index, agent])
                samples["covariance"].append(np.diag(episode["covariance"][time_index, agent]) / float(extent**2))
                samples["message_age"].append(episode["message_age"][time_index, agent])
                labels = [
                    (episode["target_positions"][time_index + horizon, 0:3] - episode["defender_positions"][time_index, agent])
                    / float(extent)
                    for horizon in horizon_steps
                ]
                samples["labels_relative"].append(np.asarray(labels, dtype=np.float32))
                samples["agent_id"].append(agent)
                samples["time_index"].append(time_index)
                samples["episode_seed"].append(episode["seed"])
                samples["scenario_index"].append(scenario_index)
    if not samples["inputs"]:
        raise RuntimeError("No prediction samples were generated; increase episodes or reduce horizons.")
    return {
        "inputs": np.asarray(samples["inputs"], dtype=np.float32),
        "belief_relative": np.asarray(samples["belief_relative"], dtype=np.float32),
        "belief_velocity": np.asarray(samples["belief_velocity"], dtype=np.float32),
        "confidence": np.asarray(samples["confidence"], dtype=np.float32),
        "covariance": np.asarray(samples["covariance"], dtype=np.float32),
        "message_age": np.asarray(samples["message_age"], dtype=np.int64),
        "labels_relative": np.asarray(samples["labels_relative"], dtype=np.float32),
        "agent_id": np.asarray(samples["agent_id"], dtype=np.int64),
        "time_index": np.asarray(samples["time_index"], dtype=np.int64),
        "episode_seed": np.asarray(samples["episode_seed"], dtype=np.int64),
        "scenario_index": np.asarray(samples["scenario_index"], dtype=np.int64),
    }


def write_metadata(
    output: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    split: str,
    seed: int,
    sample_arrays: dict[str, np.ndarray],
    episodes: list[dict[str, Any]],
) -> None:
    source_paths = [
        PROJECT_ROOT / "scripts" / "generate_prediction_dataset.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_env.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_controllers.py",
    ]
    hashes = {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    metadata = {
        "task": "local_history_target_prediction_for_3d_capture_radius_pursuit",
        "split": split,
        "seed_block": seed,
        "controller": args.controller,
        "history_length": args.history_length,
        "horizon_steps": args.horizon_steps,
        "horizon_seconds": [float(step * config["world"]["dt"]) for step in args.horizon_steps],
        "input_shape": list(sample_arrays["inputs"].shape),
        "label_shape": list(sample_arrays["labels_relative"].shape),
        "input_feature_dimension": int(sample_arrays["inputs"].shape[-1]),
        "episodes_recorded": len(episodes),
        "samples": int(sample_arrays["inputs"].shape[0]),
        "normalization": {
            "position_extent_m": float(config["world"]["half_extent_xy"]),
            "target_velocity_scale_mps": float(config["agents"]["target_max_speed"]),
            "labels_are_target_position_relative_to_current_defender": True,
        },
        "information_boundary": {
            "predictor_inputs_use_hidden_target_truth": False,
            "target_truth_used_only_for_offline_labels": True,
            "centralized_critic_or_global_state_in_inputs": False,
        },
        "source_hashes": hashes,
        "config": config,
        "environment": {
            "python": sys.version.replace(chr(10), " "),
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "PyYAML": version("PyYAML"),
        },
    }
    output.joinpath("metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.history_length <= 0 or any(step <= 0 for step in args.horizon_steps):
        raise ValueError("history-length and horizon-steps must be positive.")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.episodes_per_scenario is not None and args.episodes_per_scenario <= 0:
        raise ValueError("--episodes-per-scenario must be positive.")
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed = seed_for_split(config, args.split)
    episodes: list[dict[str, Any]] = []
    episodes_per_scenario = args.episodes_per_scenario
    for scenario_index, experiment in enumerate(config["experiments"]):
        scenario_config = config_for_experiment(config, experiment)
        count = int(experiment["episodes"]) if episodes_per_scenario is None else episodes_per_scenario
        for episode_index in range(count):
            env = CaptureRadiusPursuit3DEnv(
                scenario_config,
                obstacle_count=int(experiment["obstacle_count"]),
                target_speed_scale=float(experiment["target_speed_scale"]),
            )
            controller = controller_for(args.controller, env)
            episode_seed = seed + scenario_index * 10_000 + episode_index
            record = _append_episode(episodes, env, controller, episode_seed)
            record["scenario_index"] = int(scenario_index)
            record["scenario_name"] = str(experiment["name"])
            record["target_motion_mode"] = str(scenario_config["task"]["pursuit"]["target_motion_mode"])
            record["obstacle_profile"] = str(scenario_config["task"]["pursuit"]["obstacle_profile"])
    arrays = assemble_prediction_samples(
        episodes,
        history_length=args.history_length,
        horizon_steps=args.horizon_steps,
        extent=float(config["world"]["half_extent_xy"]),
        target_max_speed=float(config["agents"]["target_max_speed"]),
        dt=float(config["world"]["dt"]),
    )
    np.savez_compressed(output / "prediction_dataset.npz", **arrays)
    output.joinpath("scenario_manifest.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "seed_block": seed,
                "episodes_per_scenario": episodes_per_scenario,
                "scenarios": [
                    {
                        "name": str(item["name"]),
                        "obstacle_count": int(item["obstacle_count"]),
                        "target_speed_scale": float(item["target_speed_scale"]),
                        "target_motion_mode": str(config_for_experiment(config, item)["task"]["pursuit"]["target_motion_mode"]),
                        "obstacle_profile": str(config_for_experiment(config, item)["task"]["pursuit"]["obstacle_profile"]),
                    }
                    for item in config["experiments"]
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_metadata(output, config, args, args.split, seed, arrays, episodes)
    print(
        json.dumps(
            {
                "split": args.split,
                "seed_block": seed,
                "episodes": len(episodes),
                "samples": int(arrays["inputs"].shape[0]),
                "input_shape": list(arrays["inputs"].shape),
                "label_shape": list(arrays["labels_relative"].shape),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
