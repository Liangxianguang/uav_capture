"""Collect train/validation-only counterfactual data for JEPA-v3.

For each retained policy-safe state, this generator evaluates a nominal action
and deterministic local alternatives in cloned simulator states.  Simulator
truth is used solely to form offline labels. The deployed model receives only
the local observation history, past executed actions, and the first desired
action of a constant candidate chunk; the label records the consequence of
holding that desired action for the configured short chunk. CBF remains
outside the learned model.
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
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.observation_encoding import policy_observations  # noqa: E402
from encirclement3d.prediction import make_constant_action_chunks  # noqa: E402
from encirclement3d.pursuit_controllers import DynamicEncirclementController, PursuitCBFSafetyFilter  # noqa: E402
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-config", type=Path, required=True)
    parser.add_argument("--v3-protocol", type=Path, default=PROJECT_ROOT / "configs/jepa_v3_development_protocol.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--episodes-per-scenario", type=int)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--horizon-steps", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--perturbation-mps", type=float, default=0.10)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--chunk-length-steps", type=int, default=1)
    parser.add_argument("--clearance-clip-m", type=float, default=5.0)
    parser.add_argument(
        "--action-scale",
        type=float,
        help="Expected physical action scale from the frozen V5 actor checkpoint. "
        "When omitted, read and verify the scale from the checkpoint named in the v3 protocol.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    result = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return result


def _validate_v3_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("phase") != "development_only" or protocol.get("locked_test_opened") is not False:
        raise ValueError("Counterfactual collection requires a closed JEPA-v3 development protocol.")
    contract = protocol.get("data_contract", {})
    if contract.get("train_only_counterfactual_collection") is not True:
        raise ValueError("Protocol must explicitly require train-only counterfactual collection.")
    if contract.get("validation_and_development_excluded_from_training") is not True:
        raise ValueError("Protocol must preserve validation/development separation.")


def _frozen_actor_action_scale(protocol: dict[str, Any], supplied_scale: float | None) -> tuple[Path, float, str]:
    """Resolve the runtime action normalization from the immutable actor contract."""
    frozen_baseline = protocol.get("frozen_baseline")
    if not isinstance(frozen_baseline, dict) or not frozen_baseline.get("actor_checkpoint"):
        raise ValueError("JEPA-v3 protocol must name its frozen actor checkpoint.")
    checkpoint_path = (PROJECT_ROOT / str(frozen_baseline["actor_checkpoint"])).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Frozen actor checkpoint is missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_scale = float(checkpoint["action_scale"])
    if checkpoint_scale <= 0.0:
        raise ValueError("Frozen actor checkpoint has an invalid action_scale.")
    action_scale = checkpoint_scale if supplied_scale is None else float(supplied_scale)
    if action_scale <= 0.0:
        raise ValueError("action-scale must be positive.")
    if not np.isclose(action_scale, checkpoint_scale, rtol=0.0, atol=1e-7):
        raise ValueError(
            f"action-scale {action_scale} does not match frozen actor action_scale {checkpoint_scale}."
        )
    return checkpoint_path, checkpoint_scale, _sha256(checkpoint_path)


def _scenario_config(config: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    scenario = copy.deepcopy(config)
    overrides = experiment.get("pursuit_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("experiment.pursuit_overrides must be a mapping.")
    scenario.setdefault("task", {}).setdefault("pursuit", {}).update(overrides)
    # Collection continues through capture to preserve fixed-horizon labels.
    scenario["task"]["pursuit"]["terminate_on_capture"] = False
    return scenario


def _episode_seed(config: dict[str, Any], split: str, scenario_index: int, episode_index: int) -> int:
    blocks = config.get("seed_blocks")
    if not isinstance(blocks, dict) or split not in blocks:
        raise ValueError("Collection config must define independent train and validation seed blocks.")
    return int(blocks[split]) + scenario_index * 10_000 + episode_index


def _per_defender_clearances(env: CaptureRadiusPursuit3DEnv, clearance_clip_m: float) -> tuple[np.ndarray, np.ndarray]:
    radius = float(env.agents["drone_radius"])
    obstacle = np.full(env.n_defenders, clearance_clip_m, dtype=np.float32)
    teammate = np.full(env.n_defenders, clearance_clip_m, dtype=np.float32)
    for agent, position in enumerate(env.defender_positions):
        if env.obstacles:
            obstacle[agent] = min(
                float(env._obstacle_clearance(position, item) - radius) for item in env.obstacles
            )
        distances = [
            float(np.linalg.norm(position - other) - 2.0 * radius)
            for other_index, other in enumerate(env.defender_positions)
            if other_index != agent
        ]
        teammate[agent] = min(distances)
    return np.clip(obstacle, -clearance_clip_m, clearance_clip_m), np.clip(
        teammate, -clearance_clip_m, clearance_clip_m
    )


def _clone_controller(source: DynamicEncirclementController, env: CaptureRadiusPursuit3DEnv) -> DynamicEncirclementController:
    clone = DynamicEncirclementController(env, horizon_seconds=source.horizon_seconds)
    clone.interceptor_id = source.interceptor_id
    return clone


def _roll_counterfactual(
    env: CaptureRadiusPursuit3DEnv,
    controller: DynamicEncirclementController,
    observation: dict[str, Any],
    candidate_chunk: np.ndarray,
    horizon_steps: list[int],
    chunk_length_steps: int,
    clearance_clip_m: float,
    extent: float,
) -> dict[str, np.ndarray]:
    """Execute a constant desired-action chunk then a policy continuation."""
    # Preserve the speed-feasible float64 action passed to CBF. The model's
    # recorded action history remains float32, but changing rollout precision
    # here would make its labels differ from runtime zero-perturbation control.
    chunk = np.asarray(candidate_chunk)
    if chunk.shape != (chunk_length_steps, env.n_defenders, 3):
        raise ValueError(
            "Counterfactual candidate chunk shape must be "
            f"({chunk_length_steps}, {env.n_defenders}, 3), got {chunk.shape}."
        )
    if not np.allclose(chunk, chunk[:1], rtol=0.0, atol=1e-7):
        raise ValueError("JEPA-v3 chunk collection requires constant desired-action chunks.")
    clone = copy.deepcopy(env)
    clone_observation = copy.deepcopy(observation)
    continuation = _clone_controller(controller, clone)
    safety_filter = PursuitCBFSafetyFilter(clone)
    maximum_horizon = max(horizon_steps)
    by_step: dict[int, dict[str, np.ndarray]] = {}
    for step in range(1, maximum_horizon + 1):
        desired = np.asarray(chunk[step - 1], dtype=np.float64) if step <= chunk_length_steps else continuation.act(clone_observation)
        executed, diagnostics = safety_filter.filter(desired, clone_observation)
        clone_observation, _reward, _terminated, _truncated, info = clone.step(executed)
        obstacle, teammate = _per_defender_clearances(clone, clearance_clip_m)
        by_step[step] = {
            "target_relative": ((clone.target_position[None, :] - env.defender_positions) / extent).astype(np.float32),
            "obstacle_clearance": (obstacle / extent).astype(np.float32),
            "inter_agent_clearance": (teammate / extent).astype(np.float32),
            "target_visible": clone.target_visible.astype(np.float32),
            "cbf_correction": np.linalg.norm(executed - desired, axis=1).astype(np.float32),
            "cbf_intervention": (np.linalg.norm(executed - desired, axis=1) > 1e-6).astype(np.float32),
            "collision": np.full(clone.n_defenders, float(bool(info["collision"])), dtype=np.float32),
            "boundary": np.full(clone.n_defenders, float(int(info["world_violation_steps"]) > 0), dtype=np.float32),
        }
    return {
        key: np.stack([by_step[step][key] for step in horizon_steps], axis=1)
        for key in next(iter(by_step.values()))
    }


def _empty_samples() -> dict[str, list[Any]]:
    return {
        "inputs": [],
        "action_history": [],
        "labels_relative": [],
        "labels_obstacle_clearance": [],
        "labels_inter_agent_clearance": [],
        "labels_target_visible": [],
        "labels_cbf_correction": [],
        "labels_cbf_intervention": [],
        "labels_collision": [],
        "labels_boundary": [],
        "agent_id": [],
        "time_index": [],
        "episode_seed": [],
        "scenario_index": [],
        "candidate_index": [],
        "candidate_is_nominal": [],
        "candidate_action_norm_mps": [],
        "chunk_length_steps": [],
    }


def _append_candidate_samples(
    samples: dict[str, list[Any]],
    observation_history: list[np.ndarray],
    executed_action_history: list[np.ndarray],
    candidate: np.ndarray,
    labels: dict[str, np.ndarray],
    episode_seed: int,
    scenario_index: int,
    time_index: int,
    candidate_index: int,
    chunk_length_steps: int,
    action_scale: float,
) -> None:
    if len(observation_history) < 8 or len(executed_action_history) != len(observation_history) - 1:
        raise ValueError("Counterfactual histories are not causally aligned.")
    inputs = np.stack(observation_history[-8:], axis=0)
    past_actions = np.stack(executed_action_history[-7:], axis=0)
    # This is intentionally the same unitless representation consumed by
    # ActionConditionedCandidateHistory at runtime.
    actions = np.concatenate([past_actions, candidate[None, ...]], axis=0) / action_scale
    for agent in range(inputs.shape[1]):
        samples["inputs"].append(inputs[:, agent].copy())
        samples["action_history"].append(actions[:, agent].copy())
        label_sources = {
            "labels_relative": "target_relative",
            "labels_obstacle_clearance": "obstacle_clearance",
            "labels_inter_agent_clearance": "inter_agent_clearance",
            "labels_target_visible": "target_visible",
            "labels_cbf_correction": "cbf_correction",
            "labels_cbf_intervention": "cbf_intervention",
            "labels_collision": "collision",
            "labels_boundary": "boundary",
        }
        for key, source_key in label_sources.items():
            samples[key].append(labels[source_key][agent].copy())
        samples["agent_id"].append(agent)
        samples["time_index"].append(time_index)
        samples["episode_seed"].append(episode_seed)
        samples["scenario_index"].append(scenario_index)
        samples["candidate_index"].append(candidate_index)
        samples["candidate_is_nominal"].append(candidate_index == 0)
        samples["candidate_action_norm_mps"].append(float(np.linalg.norm(candidate[agent])))
        samples["chunk_length_steps"].append(chunk_length_steps)


def collect(
    config: dict[str, Any],
    split: str,
    episodes_per_scenario: int | None,
    history_length: int,
    horizon_steps: list[int],
    candidate_count: int,
    perturbation_mps: float,
    sample_stride: int,
    chunk_length_steps: int,
    clearance_clip_m: float,
    action_scale: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if history_length != 8:
        raise ValueError("JEPA-v3 currently fixes history-length at 8 to preserve the frozen contract.")
    samples = _empty_samples()
    scenario_counts: dict[str, int] = {}
    extent = float(config["world"]["half_extent_xy"])
    for scenario_index, experiment in enumerate(config["experiments"]):
        scenario = _scenario_config(config, experiment)
        count = int(experiment["episodes"]) if episodes_per_scenario is None else int(episodes_per_scenario)
        scenario_counts[str(experiment["name"])] = count
        for episode_index in range(count):
            seed = _episode_seed(config, split, scenario_index, episode_index)
            env = CaptureRadiusPursuit3DEnv(
                scenario,
                obstacle_count=int(experiment["obstacle_count"]),
                target_speed_scale=float(experiment["target_speed_scale"]),
            )
            controller = DynamicEncirclementController(env)
            safety_filter = PursuitCBFSafetyFilter(env)
            observation = env.reset(seed=seed)
            observations = [policy_observations(env, observation).copy()]
            executed_actions: list[np.ndarray] = []
            for time_index in range(env.max_steps):
                desired = np.asarray(controller.act(observation), dtype=np.float32)
                if time_index >= history_length - 1 and (time_index - (history_length - 1)) % sample_stride == 0:
                    candidate_chunks = make_constant_action_chunks(
                        desired,
                        chunk_length_steps=chunk_length_steps,
                        perturbation_mps=perturbation_mps,
                        candidate_count=candidate_count,
                        max_speed_mps=float(env.agents["defender_max_speed"]),
                    )
                    for candidate_index, candidate_chunk in enumerate(candidate_chunks):
                        labels = _roll_counterfactual(
                            env,
                            controller,
                            observation,
                            candidate_chunk,
                            horizon_steps,
                            chunk_length_steps,
                            clearance_clip_m,
                            extent,
                        )
                        _append_candidate_samples(
                            samples,
                            observations,
                            executed_actions,
                            candidate_chunk[0],
                            labels,
                            seed,
                            scenario_index,
                            time_index,
                            candidate_index,
                            chunk_length_steps,
                            action_scale,
                        )
                executed, _diagnostics = safety_filter.filter(desired, observation)
                observation, _reward, terminated, truncated, _info = env.step(executed)
                executed_actions.append(executed.astype(np.float32, copy=True))
                if terminated or truncated:
                    break
                observations.append(policy_observations(env, observation).copy())
    if not samples["inputs"]:
        raise RuntimeError("No counterfactual samples were collected.")
    arrays = {
        key: np.asarray(value, dtype=np.float32)
        if key not in {"agent_id", "time_index", "episode_seed", "scenario_index", "candidate_index", "chunk_length_steps"}
        else np.asarray(value, dtype=np.int64)
        for key, value in samples.items()
    }
    metadata = {
        "task": "jepa_v3_train_or_validation_only_counterfactual_multitask_collection",
        "split": split,
        "episodes_by_scenario": scenario_counts,
        "history_length": history_length,
        "horizon_steps": horizon_steps,
        "horizon_seconds": [float(step * config["world"]["dt"]) for step in horizon_steps],
        "candidate_count": candidate_count,
        "candidate_perturbation_mps": perturbation_mps,
        "sample_stride": sample_stride,
        "chunk_length_steps": chunk_length_steps,
        "candidate_action_semantics": "constant_desired_action_chunk_execute_first_step_then_replan",
        "candidate_chunk_is_constant": True,
        "clearance_clip_m": clearance_clip_m,
        "action_history_normalization": "actions_divided_by_frozen_actor_action_scale",
        "action_scale": action_scale,
        "input_shape": list(arrays["inputs"].shape),
        "action_shape": list(arrays["action_history"].shape),
        "target_label_shape": list(arrays["labels_relative"].shape),
        "counterfactual_label_shapes": {
            key: list(value.shape) for key, value in arrays.items() if key.startswith("labels_")
        },
        "candidate_is_nominal_fraction": float(np.mean(arrays["candidate_is_nominal"])),
        "information_boundary": {
            "target_truth_used_only_for_offline_labels": True,
            "centralized_state_in_inputs": False,
            "action_history_alignment": "past_executed_actions_then_final_counterfactual_desired_action",
            "development_s3_or_locked_data_used_for_training": False,
        },
    }
    return arrays, metadata


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output}")
    if min(args.history_length, args.candidate_count, args.sample_stride, args.chunk_length_steps) <= 0:
        raise ValueError("history/candidate/stride/chunk values must be positive.")
    if args.perturbation_mps < 0.0 or args.clearance_clip_m <= 0.0:
        raise ValueError("perturbation must be non-negative and clearance clip positive.")
    horizon_steps = sorted(set(int(value) for value in args.horizon_steps))
    if not horizon_steps or horizon_steps[0] <= 0:
        raise ValueError("horizon-steps must be positive.")
    v3_protocol = _load_yaml(args.v3_protocol.resolve())
    _validate_v3_protocol(v3_protocol)
    checkpoint_path, checkpoint_action_scale, checkpoint_sha256 = _frozen_actor_action_scale(
        v3_protocol,
        args.action_scale,
    )
    collection_config = _load_yaml(args.collection_config.resolve())
    if not isinstance(collection_config.get("experiments"), list):
        raise ValueError("Collection config requires experiments.")
    arrays, metadata = collect(
        collection_config,
        args.split,
        args.episodes_per_scenario,
        args.history_length,
        horizon_steps,
        args.candidate_count,
        args.perturbation_mps,
        args.sample_stride,
        args.chunk_length_steps,
        args.clearance_clip_m,
        checkpoint_action_scale,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "counterfactual_multitask_dataset.npz", **arrays)
    metadata.update(
        {
            "v3_protocol": str(args.v3_protocol.resolve()),
            "collection_config": str(args.collection_config.resolve()),
            "source_hashes": {
                str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path)
                for path in (
                    PROJECT_ROOT / "scripts" / "generate_jepa_v3_counterfactual_dataset.py",
                    PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_env.py",
                    PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_controllers.py",
                    PROJECT_ROOT / "src" / "encirclement3d" / "observation_encoding.py",
                    PROJECT_ROOT / "src" / "encirclement3d" / "prediction.py",
                )
            },
            "collection_config_sha256": _sha256(args.collection_config.resolve()),
            "v3_protocol_sha256": _sha256(args.v3_protocol.resolve()),
            "frozen_actor_checkpoint": str(checkpoint_path),
            "frozen_actor_checkpoint_sha256": checkpoint_sha256,
            "frozen_actor_action_scale": checkpoint_action_scale,
            "environment": {
                "python": sys.version.replace("\n", " "),
                "platform": platform.platform(),
                "numpy": version("numpy"),
                "PyYAML": version("PyYAML"),
            },
        }
    )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (args.output / "scenario_manifest.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "episodes_by_scenario": metadata["episodes_by_scenario"],
                "sample_count": int(arrays["inputs"].shape[0]),
                "candidate_count": args.candidate_count,
                "development_or_locked_data_in_training": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"split": args.split, "samples": int(arrays["inputs"].shape[0]), "input_shape": list(arrays["inputs"].shape), "candidate_nominal_fraction": metadata["candidate_is_nominal_fraction"]}, indent=2))


if __name__ == "__main__":
    main()
