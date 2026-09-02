"""Generate structured action-conditioned JEPA pilot data.

Each sample contains a local policy-safe observation history, the executed
action history for that defender, and future target displacement labels. The
simulator target state is used only to create offline labels.
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

from encirclement3d.pursuit_controllers import (  # noqa: E402
    DynamicEncirclementController,
    PredictionPursuitController,
    PurePursuitController,
    SafetyFilteredPursuitController,
)
from encirclement3d.observation_encoding import policy_observations  # noqa: E402
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "locked_test"), default="train")
    parser.add_argument("--episodes-per-scenario", type=int)
    parser.add_argument(
        "--controller",
        choices=("pure", "prediction", "encirclement", "pure_cbf", "prediction_cbf", "encirclement_cbf"),
        default="encirclement_cbf",
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
    if not isinstance(overrides, dict):
        raise ValueError("experiment.pursuit_overrides must be a mapping.")
    scenario_config.setdefault("task", {}).setdefault("pursuit", {}).update(overrides)
    return scenario_config


def seed_for_split(config: dict[str, Any], split: str) -> int:
    blocks = config.get("seed_blocks", {})
    if blocks and split not in blocks:
        raise ValueError(f"Config seed_blocks does not contain split {split!r}.")
    return int(blocks.get(split, config["seed"]))


def collect_episode(
    config: dict[str, Any],
    experiment: dict[str, Any],
    seed: int,
    controller_name: str,
) -> dict[str, np.ndarray | int]:
    # Continue successful episodes only for offline label coverage.  This is a
    # data-collection override and never changes the deployment/evaluation
    # contract, where capture remains a terminal event.
    config = copy.deepcopy(config)
    config.setdefault("task", {}).setdefault("pursuit", {})["terminate_on_capture"] = False
    env = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=int(experiment["obstacle_count"]),
        target_speed_scale=float(experiment["target_speed_scale"]),
    )
    controller = controller_for(controller_name, env)
    observation = env.reset(seed=seed)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    target_positions: list[np.ndarray] = []
    defender_positions: list[np.ndarray] = []
    pending_action = np.zeros((env.n_defenders, 3), dtype=np.float32)
    while True:
        observations.append(policy_observations(env, observation).copy())
        actions.append(pending_action.copy())
        target_positions.append(env.target_position.astype(np.float32, copy=True))
        defender_positions.append(env.defender_positions.astype(np.float32, copy=True))
        pending_action = np.asarray(controller.act(observation), dtype=np.float32)
        observation, _reward, terminated, truncated, _info = env.step(pending_action)
        if terminated or truncated:
            # The final observation has no action transition; retaining a zero
            # action for it keeps all arrays aligned and excludes it from most
            # training windows through the future-horizon check.
            break
    return {
        "policy": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "target_positions": np.asarray(target_positions, dtype=np.float32),
        "defender_positions": np.asarray(defender_positions, dtype=np.float32),
        "seed": int(seed),
    }


def assemble_action_conditioned_samples(
    episodes: list[dict[str, np.ndarray | int]],
    history_length: int,
    horizon_steps: list[int],
    extent: float,
) -> dict[str, np.ndarray]:
    if history_length <= 0:
        raise ValueError("history-length must be positive.")
    horizon_steps = sorted(set(int(step) for step in horizon_steps))
    if not horizon_steps or any(step <= 0 for step in horizon_steps):
        raise ValueError("horizon-steps must contain positive integers.")
    samples: dict[str, list[np.ndarray | int]] = {
        "inputs": [],
        "action_history": [],
        "labels_relative": [],
        "agent_id": [],
        "time_index": [],
        "episode_seed": [],
        "scenario_index": [],
    }
    max_horizon = max(horizon_steps)
    for episode in episodes:
        policy = np.asarray(episode["policy"], dtype=np.float32)
        actions = np.asarray(episode["actions"], dtype=np.float32)
        target_positions = np.asarray(episode["target_positions"], dtype=np.float32)
        defender_positions = np.asarray(episode["defender_positions"], dtype=np.float32)
        length = int(policy.shape[0])
        if length < history_length + max_horizon:
            continue
        scenario_index = int(episode.get("scenario_index", 0))
        for time_index in range(history_length - 1, length - max_horizon):
            for agent in range(int(policy.shape[1])):
                samples["inputs"].append(policy[time_index - history_length + 1 : time_index + 1, agent])
                # `actions[k]` is the action that produced observation `k`
                # (with an initial zero action at k=0).  The world-model
                # input instead conditions observation O_t on the outgoing
                # action A_t, which is stored at actions[t + 1].  Shifting
                # this window by one makes the final slot a valid
                # counterfactual candidate at deployment rather than the
                # action already executed to reach the current observation.
                samples["action_history"].append(actions[time_index - history_length + 2 : time_index + 2, agent])
                labels = [
                    (target_positions[time_index + horizon] - defender_positions[time_index, agent]) / float(extent)
                    for horizon in horizon_steps
                ]
                samples["labels_relative"].append(np.asarray(labels, dtype=np.float32))
                samples["agent_id"].append(agent)
                samples["time_index"].append(time_index)
                samples["episode_seed"].append(int(episode["seed"]))
                samples["scenario_index"].append(scenario_index)
    if not samples["inputs"]:
        raise RuntimeError("No action-conditioned samples were generated.")
    return {
        "inputs": np.asarray(samples["inputs"], dtype=np.float32),
        "action_history": np.asarray(samples["action_history"], dtype=np.float32),
        "labels_relative": np.asarray(samples["labels_relative"], dtype=np.float32),
        "agent_id": np.asarray(samples["agent_id"], dtype=np.int64),
        "time_index": np.asarray(samples["time_index"], dtype=np.int64),
        "episode_seed": np.asarray(samples["episode_seed"], dtype=np.int64),
        "scenario_index": np.asarray(samples["scenario_index"], dtype=np.int64),
    }


def main() -> None:
    args = parse_args()
    if args.history_length <= 0:
        raise ValueError("history-length must be positive.")
    if args.episodes_per_scenario is not None and args.episodes_per_scenario <= 0:
        raise ValueError("episodes-per-scenario must be positive.")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("experiments"), list):
        raise ValueError("Config must contain an experiments list.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    seed = seed_for_split(config, args.split)
    episodes: list[dict[str, np.ndarray | int]] = []
    for scenario_index, experiment in enumerate(config["experiments"]):
        scenario_config = config_for_experiment(config, experiment)
        count = int(experiment["episodes"]) if args.episodes_per_scenario is None else args.episodes_per_scenario
        for episode_index in range(count):
            record = collect_episode(
                scenario_config,
                experiment,
                seed + scenario_index * 10_000 + episode_index,
                args.controller,
            )
            record["scenario_index"] = scenario_index
            episodes.append(record)
    arrays = assemble_action_conditioned_samples(
        episodes,
        history_length=args.history_length,
        horizon_steps=args.horizon_steps,
        extent=float(config["world"]["half_extent_xy"]),
    )
    np.savez_compressed(args.output / "action_conditioned_prediction_dataset.npz", **arrays)
    source_paths = [
        PROJECT_ROOT / "scripts" / "generate_action_conditioned_prediction_dataset.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "prediction.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_env.py",
    ]
    metadata = {
        "task": "structured_action_conditioned_jepa_target_prediction_for_3d_capture",
        "split": args.split,
        "seed_block": seed,
        "controller": args.controller,
        "history_length": args.history_length,
        "horizon_steps": args.horizon_steps,
        "horizon_seconds": [float(step * config["world"]["dt"]) for step in args.horizon_steps],
        "input_shape": list(arrays["inputs"].shape),
        "action_shape": list(arrays["action_history"].shape),
        "label_shape": list(arrays["labels_relative"].shape),
        "input_feature_dimension": int(arrays["inputs"].shape[-1]),
        "information_boundary": {
            "target_truth_used_only_for_offline_labels": True,
            "centralized_state_in_inputs": False,
            "action_history_alignment": "outgoing_action_for_each_observation; final_action_is_current_expert_candidate",
        },
        "source_hashes": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "config": config,
        "environment": {
            "python": sys.version.replace(chr(10), " "),
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "PyYAML": version("PyYAML"),
        },
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.output / "scenario_manifest.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "seed_block": seed,
                "episodes": len(episodes),
                "scenarios": [str(item["name"]) for item in config["experiments"]],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "split": args.split,
                "episodes": len(episodes),
                "samples": int(arrays["inputs"].shape[0]),
                "input_shape": list(arrays["inputs"].shape),
                "action_shape": list(arrays["action_history"].shape),
                "label_shape": list(arrays["labels_relative"].shape),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
