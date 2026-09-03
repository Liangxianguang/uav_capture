"""Collect a versioned P1 archive for safe-capture JEPA calibration.

The simulator is used only offline to settle counterfactual labels.  The
resulting observation/action arrays stay within the policy-safe contract.  The
script deliberately refuses development and locked splits, and never writes
over an existing archive or TensorBoard directory.
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
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.observation_encoding import policy_observations  # noqa: E402
from encirclement3d.prediction import make_constant_action_chunks  # noqa: E402
from encirclement3d.pursuit_controllers import (  # noqa: E402
    DynamicEncirclementController,
    PursuitCBFSafetyFilter,
)
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402


ALLOWED_SPLITS = ("train", "validation", "calibration")
INDEX_ARRAYS = {
    "agent_id",
    "time_index",
    "episode_seed",
    "scenario_index",
    "candidate_index",
    "chunk_length_steps",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return value


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_contract(protocol: dict[str, Any], collection: dict[str, Any], split: str) -> None:
    if protocol.get("phase") != "development_only" or protocol.get("locked_test_opened") is not False:
        raise ValueError("P1 collection requires a closed development-only protocol.")
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"P1 collection only permits {ALLOWED_SPLITS}; refusing {split!r}.")
    archive = collection.get("archive_contract")
    if not isinstance(archive, dict) or archive.get("locked_test_collection_forbidden") is not True:
        raise ValueError("Protocol must explicitly forbid locked-test collection.")
    if archive.get("target_truth_used_only_for_offline_labels") is not True:
        raise ValueError("Protocol must keep target truth offline-only.")
    blocks = collection.get("seed_blocks")
    if not isinstance(blocks, dict) or any(name not in blocks for name in (*ALLOWED_SPLITS, "development", "locked")):
        raise ValueError("Collection must declare disjoint train/validation/calibration/development/locked seed blocks.")
    experiments = collection.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Collection must define at least one scenario experiment.")
    names = [str(item.get("name")) for item in experiments]
    if len(set(names)) != len(names) or any(name in {"None", ""} for name in names):
        raise ValueError("Scenario names must be unique and non-empty.")
    for experiment in experiments:
        if int(experiment.get("episodes", 0)) <= 0 or int(experiment.get("obstacle_count", -1)) < 0:
            raise ValueError(f"Invalid episode/obstacle count in {experiment!r}.")


def _frozen_actor_action_scale(protocol: dict[str, Any]) -> tuple[Path, float, str]:
    frozen = protocol.get("frozen_inputs", {})
    if not isinstance(frozen, dict) or not frozen.get("actor_checkpoint"):
        raise ValueError("P1 protocol must name frozen_inputs.actor_checkpoint.")
    checkpoint = (PROJECT_ROOT / str(frozen["actor_checkpoint"])).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Frozen actor checkpoint is missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    scale = float(payload.get("action_scale", 0.0))
    if scale <= 0.0:
        raise ValueError("Frozen actor checkpoint has an invalid action_scale.")
    return checkpoint, scale, _sha256(checkpoint)


def _scenario_config(config: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    scenario = copy.deepcopy(config)
    overrides = experiment.get("pursuit_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("experiment.pursuit_overrides must be a mapping.")
    scenario.setdefault("task", {}).setdefault("pursuit", {}).update(overrides)
    scenario["task"]["pursuit"]["terminate_on_capture"] = False
    return scenario


def episode_seed(collection: dict[str, Any], split: str, scenario_index: int, episode_index: int) -> int:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Unsupported P1 split: {split!r}")
    blocks = collection.get("seed_blocks", {})
    return int(blocks[split]) + scenario_index * 10_000 + episode_index


def _per_defender_clearances(env: CaptureRadiusPursuit3DEnv, clip_m: float) -> tuple[np.ndarray, np.ndarray]:
    radius = float(env.agents["drone_radius"])
    obstacle = np.full(env.n_defenders, clip_m, dtype=np.float32)
    teammate = np.full(env.n_defenders, clip_m, dtype=np.float32)
    for agent, position in enumerate(env.defender_positions):
        if env.obstacles:
            obstacle[agent] = min(
                float(env._obstacle_clearance(position, item) - radius) for item in env.obstacles
            )
        teammate[agent] = min(
            float(np.linalg.norm(position - other) - 2.0 * radius)
            for other_index, other in enumerate(env.defender_positions)
            if other_index != agent
        )
    return np.clip(obstacle, -clip_m, clip_m), np.clip(teammate, -clip_m, clip_m)


def pairwise_time_to_collision(
    positions: np.ndarray,
    velocities: np.ndarray,
    radius: float,
    margin: float,
    clip_seconds: float,
) -> np.ndarray:
    """Return each defender's conservative earliest pairwise TTC."""
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    if positions.ndim != 2 or positions.shape != velocities.shape or positions.shape[1] != 3:
        raise ValueError("positions and velocities must have shape [defenders, 3].")
    result = np.full(positions.shape[0], float(clip_seconds), dtype=np.float32)
    safe_distance = 2.0 * float(radius) + float(margin)
    for first in range(positions.shape[0]):
        for second in range(first + 1, positions.shape[0]):
            relative_position = positions[first] - positions[second]
            relative_velocity = velocities[first] - velocities[second]
            speed_squared = float(np.dot(relative_velocity, relative_velocity))
            if speed_squared <= 1e-12:
                continue
            closing = float(np.dot(relative_position, relative_velocity))
            if closing >= 0.0:
                continue
            # Solve ||r + t v||^2 = safe_distance^2 and use the first
            # non-negative root.  Time-to-closest-approach would systematically
            # overestimate the warning horizon for an approaching pair.
            c = float(np.dot(relative_position, relative_position) - safe_distance**2)
            if c <= 0.0:
                time = 0.0
            else:
                discriminant = closing**2 - speed_squared * c
                if discriminant < 0.0:
                    continue
                time = (-closing - float(np.sqrt(discriminant))) / speed_squared
            if 0.0 <= time <= clip_seconds:
                result[first] = min(result[first], float(time))
                result[second] = min(result[second], float(time))
    return result


def _clone_controller(source: DynamicEncirclementController, env: CaptureRadiusPursuit3DEnv) -> DynamicEncirclementController:
    clone = DynamicEncirclementController(env, horizon_seconds=source.horizon_seconds)
    clone.interceptor_id = source.interceptor_id
    return clone


def _roll_counterfactual_v2(
    env: CaptureRadiusPursuit3DEnv,
    controller: DynamicEncirclementController,
    observation: dict[str, Any],
    candidate_chunk: np.ndarray,
    horizon_steps: list[int],
    chunk_length_steps: int,
    clip_m: float,
    extent: float,
    ttc_clip_seconds: float,
    cbf_max_correction_norm_mps: float,
) -> dict[str, np.ndarray]:
    chunk = np.asarray(candidate_chunk)
    expected = (chunk_length_steps, env.n_defenders, 3)
    if chunk.shape != expected or not np.allclose(chunk, chunk[:1], rtol=0.0, atol=1e-7):
        raise ValueError(f"P1 requires constant action chunks with shape {expected}.")
    clone = copy.deepcopy(env)
    clone_observation = copy.deepcopy(observation)
    continuation = _clone_controller(controller, clone)
    safety_filter = PursuitCBFSafetyFilter(clone)
    by_step: dict[int, dict[str, np.ndarray]] = {}
    for step in range(1, max(horizon_steps) + 1):
        previous_target_velocity = clone.target_velocity.copy()
        desired = (
            np.asarray(chunk[step - 1], dtype=np.float64)
            if step <= chunk_length_steps
            else continuation.act(clone_observation)
        )
        executed, diagnostics = safety_filter.filter(desired, clone_observation)
        clone_observation, _reward, _terminated, _truncated, info = clone.step(executed)
        obstacle, teammate = _per_defender_clearances(clone, clip_m)
        correction = np.linalg.norm(executed - desired, axis=1).astype(np.float32)
        qp_feasible = bool(
            np.isfinite(executed).all()
            and np.isfinite(float(diagnostics.action_correction_norm))
            and float(np.max(correction)) <= cbf_max_correction_norm_mps + 1e-7
        )
        observation_age = np.asarray(clone_observation["target_observation_age_steps"], dtype=np.float32)
        by_step[step] = {
            # Relative target is measured from the current defender position,
            # matching the runtime candidate reranker contract.
            "target_relative": ((clone.target_position[None, :] - env.defender_positions) / extent).astype(np.float32),
            "target_velocity": np.repeat(
                (clone.target_velocity / float(clone.agents["target_max_speed"]))[None, :],
                clone.n_defenders,
                axis=0,
            ).astype(np.float32),
            "target_acceleration": np.repeat(
                (
                    (clone.target_velocity - previous_target_velocity)
                    / max(float(clone.agents["target_max_acceleration"]), 1e-9)
                )[None, :],
                clone.n_defenders,
                axis=0,
            ).astype(np.float32),
            "obstacle_clearance": (obstacle / extent).astype(np.float32),
            "inter_agent_clearance": (teammate / extent).astype(np.float32),
            "pairwise_ttc": pairwise_time_to_collision(
                clone.defender_positions,
                clone.defender_velocities,
                float(clone.agents["drone_radius"]),
                float(clone.pursuit["safety_margin"]),
                ttc_clip_seconds,
            ),
            "target_visible": clone.target_visible.astype(np.float32),
            "observation_age": np.clip(observation_age, 0.0, float(clone.pursuit["maximum_message_age_steps"])),
            "cbf_correction": correction,
            "cbf_intervention": (correction > 1e-6).astype(np.float32),
            "cbf_qp_feasible": np.full(clone.n_defenders, float(qp_feasible), dtype=np.float32),
            "collision": np.full(clone.n_defenders, float(bool(info["collision"])), dtype=np.float32),
            "boundary": np.full(
                clone.n_defenders,
                float(bool(info.get("defender_boundary_violation", False))),
                dtype=np.float32,
            ),
        }
    return {
        key: np.stack([by_step[step][key] for step in horizon_steps], axis=1)
        for key in by_step[1]
    }


def _empty_samples() -> dict[str, list[Any]]:
    return {
        "inputs": [],
        "action_history": [],
        "labels_relative": [],
        "labels_target_velocity": [],
        "labels_target_acceleration": [],
        "labels_obstacle_clearance": [],
        "labels_inter_agent_clearance": [],
        "labels_pairwise_ttc": [],
        "labels_target_visible": [],
        "labels_observation_age": [],
        "labels_cbf_correction": [],
        "labels_cbf_intervention": [],
        "labels_cbf_qp_feasible": [],
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


def _append_candidate(
    samples: dict[str, list[Any]],
    observation_history: list[np.ndarray],
    executed_action_history: list[np.ndarray],
    candidate: np.ndarray,
    labels: dict[str, np.ndarray],
    episode_seed_value: int,
    scenario_index: int,
    time_index: int,
    candidate_index: int,
    chunk_length_steps: int,
    action_scale: float,
) -> None:
    if len(observation_history) < 8 or len(executed_action_history) != len(observation_history) - 1:
        raise ValueError("P1 histories are not causally aligned.")
    inputs = np.stack(observation_history[-8:], axis=0)
    past_actions = np.stack(executed_action_history[-7:], axis=0)
    actions = np.concatenate([past_actions, np.asarray(candidate)[None, ...]], axis=0) / action_scale
    sources = {
        "labels_relative": "target_relative",
        "labels_target_velocity": "target_velocity",
        "labels_target_acceleration": "target_acceleration",
        "labels_obstacle_clearance": "obstacle_clearance",
        "labels_inter_agent_clearance": "inter_agent_clearance",
        "labels_pairwise_ttc": "pairwise_ttc",
        "labels_target_visible": "target_visible",
        "labels_observation_age": "observation_age",
        "labels_cbf_correction": "cbf_correction",
        "labels_cbf_intervention": "cbf_intervention",
        "labels_cbf_qp_feasible": "cbf_qp_feasible",
        "labels_collision": "collision",
        "labels_boundary": "boundary",
    }
    for agent in range(inputs.shape[1]):
        samples["inputs"].append(inputs[:, agent].copy())
        samples["action_history"].append(actions[:, agent].copy())
        for key, source in sources.items():
            samples[key].append(labels[source][agent].copy())
        samples["agent_id"].append(agent)
        samples["time_index"].append(time_index)
        samples["episode_seed"].append(episode_seed_value)
        samples["scenario_index"].append(scenario_index)
        samples["candidate_index"].append(candidate_index)
        samples["candidate_is_nominal"].append(float(candidate_index == 0))
        samples["candidate_action_norm_mps"].append(float(np.linalg.norm(candidate[agent])))
        samples["chunk_length_steps"].append(chunk_length_steps)


def collect(
    collection: dict[str, Any],
    protocol: dict[str, Any],
    split: str,
    episodes_per_scenario: int | None,
    tensorboard_logdir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _validate_contract(protocol, collection, split)
    archive = collection["archive_contract"]
    history_length = int(archive["history_length"])
    horizon_steps = [int(value) for value in archive["horizon_steps"]]
    candidate_count = int(archive["candidate_count"])
    perturbation_mps = float(archive["perturbation_mps"])
    sample_stride = int(archive["sample_stride"])
    chunk_length_steps = int(archive["chunk_length_steps"])
    clip_m = float(protocol["world"]["half_extent_xy_m"])
    ttc_clip = float(archive["ttc_clip_seconds"])
    max_cbf_correction = float(archive["cbf_max_correction_norm_mps"])
    actor_checkpoint, action_scale, actor_hash = _frozen_actor_action_scale(protocol)
    samples = _empty_samples()
    scenario_records: list[dict[str, Any]] = []
    for scenario_index, experiment in enumerate(collection["experiments"]):
        count = int(experiment["episodes"]) if episodes_per_scenario is None else int(episodes_per_scenario)
        scenario = _scenario_config(collection, experiment)
        experiment_hash = _canonical_hash(experiment)
        seeds = [episode_seed(collection, split, scenario_index, episode_index) for episode_index in range(count)]
        scenario_records.append(
            {
                "scenario_index": scenario_index,
                "name": str(experiment["name"]),
                "difficulty": str(experiment.get("difficulty", "unspecified")),
                "experiment_sha256": experiment_hash,
                "episodes": count,
                "episode_seeds": seeds,
                "obstacle_count": int(experiment["obstacle_count"]),
                "target_speed_scale": float(experiment["target_speed_scale"]),
                "target_motion_mode": str(scenario["task"]["pursuit"]["target_motion_mode"]),
            }
        )
        for seed in seeds:
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
                if time_index >= history_length - 1 and (time_index - history_length + 1) % sample_stride == 0:
                    candidate_chunks = make_constant_action_chunks(
                        desired,
                        chunk_length_steps=chunk_length_steps,
                        perturbation_mps=perturbation_mps,
                        candidate_count=candidate_count,
                        max_speed_mps=float(env.agents["defender_max_speed"]),
                    )
                    for candidate_index, candidate_chunk in enumerate(candidate_chunks):
                        labels = _roll_counterfactual_v2(
                            env,
                            controller,
                            observation,
                            candidate_chunk,
                            horizon_steps,
                            chunk_length_steps,
                            clip_m,
                            float(collection["world"]["half_extent_xy"]),
                            ttc_clip,
                            max_cbf_correction,
                        )
                        _append_candidate(
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
        raise RuntimeError("P1 collection produced no samples.")
    arrays = {
        key: np.asarray(value, dtype=np.int64 if key in INDEX_ARRAYS else np.float32)
        for key, value in samples.items()
    }
    metadata = {
        "task": "jepa_safe_capture_v2_p1_offline_counterfactual_archive",
        "dataset_version": str(archive["dataset_version"]),
        "split": split,
        "history_length": history_length,
        "horizon_steps": horizon_steps,
        "horizon_seconds": [float(step * collection["world"]["dt"]) for step in horizon_steps],
        "candidate_count": candidate_count,
        "candidate_perturbation_mps": perturbation_mps,
        "sample_stride": sample_stride,
        "chunk_length_steps": chunk_length_steps,
        "candidate_action_semantics": str(archive["candidate_semantics"]),
        "candidate_chunk_is_constant": True,
        "clearance_clip_m": clip_m,
        "ttc_clip_seconds": ttc_clip,
        "action_history_normalization": "actions_divided_by_frozen_actor_action_scale",
        "action_scale": action_scale,
        "input_shape": list(arrays["inputs"].shape),
        "action_shape": list(arrays["action_history"].shape),
        "label_shapes": {key: list(value.shape) for key, value in arrays.items() if key.startswith("labels_")},
        "candidate_is_nominal_fraction": float(np.mean(arrays["candidate_is_nominal"])),
        "scenario_records": scenario_records,
        "episode_seed_count": int(sum(len(item["episode_seeds"]) for item in scenario_records)),
        "episode_seeds": sorted(seed for item in scenario_records for seed in item["episode_seeds"]),
        "label_units": archive["label_units"],
        "information_boundary": {
            "target_truth_used_only_for_offline_labels": True,
            "centralized_state_in_inputs": False,
            "development_or_locked_data_used_for_training": False,
            "locked_test_opened": False,
        },
        "collection_config": str((PROJECT_ROOT / "configs/jepa_safe_capture_v2_collection.yaml").resolve()),
        "protocol": str((PROJECT_ROOT / "configs/jepa_safe_capture_v2_protocol.yaml").resolve()),
        "tensorboard_logdir": str(tensorboard_logdir.resolve()),
        "frozen_actor_checkpoint": str(actor_checkpoint),
        "frozen_actor_checkpoint_sha256": actor_hash,
        "frozen_actor_action_scale": action_scale,
        "source_hashes": {
            "scripts/generate_jepa_safe_capture_v2_archive.py": _sha256(Path(__file__).resolve()),
            "src/encirclement3d/pursuit_env.py": _sha256(PROJECT_ROOT / "src/encirclement3d/pursuit_env.py"),
            "src/encirclement3d/pursuit_controllers.py": _sha256(PROJECT_ROOT / "src/encirclement3d/pursuit_controllers.py"),
            "src/encirclement3d/observation_encoding.py": _sha256(PROJECT_ROOT / "src/encirclement3d/observation_encoding.py"),
            "src/encirclement3d/prediction.py": _sha256(PROJECT_ROOT / "src/encirclement3d/prediction.py"),
        },
        "collection_config_sha256": _sha256(PROJECT_ROOT / "configs/jepa_safe_capture_v2_collection.yaml"),
        "protocol_sha256": _sha256(PROJECT_ROOT / "configs/jepa_safe_capture_v2_protocol.yaml"),
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "torch": version("torch"),
            "tensorboard": version("tensorboard"),
            "yaml": version("PyYAML"),
        },
    }
    tensorboard_logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_logdir), flush_secs=10)
    writer.add_text("Config/protocol", yaml.safe_dump(protocol, sort_keys=False), 0)
    writer.add_text("Config/collection", yaml.safe_dump(collection, sort_keys=False), 0)
    writer.add_text("Data/metadata", json.dumps(metadata, indent=2), 0)
    writer.add_text("Provenance/source_hashes", json.dumps(metadata["source_hashes"], indent=2), 0)
    writer.add_scalar("Data/sample_count", int(arrays["inputs"].shape[0]), 0)
    writer.add_scalar("Data/episode_count", metadata["episode_seed_count"], 0)
    writer.add_scalar("Data/scenario_count", len(scenario_records), 0)
    writer.add_scalar("Data/candidate_count", candidate_count, 0)
    writer.add_scalar("Data/nominal_fraction", metadata["candidate_is_nominal_fraction"], 0)
    for name, values in arrays.items():
        if not name.startswith("labels_"):
            continue
        finite = float(np.isfinite(values).all())
        writer.add_scalar(f"Data/{name}/finite", finite, 0)
        writer.add_scalar(f"Data/{name}/coverage", float(np.isfinite(values).mean()), 0)
        writer.add_histogram(f"Data/{name}", values.reshape(-1), 0)
    writer.flush()
    writer.close()
    return arrays, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-config", type=Path, default=PROJECT_ROOT / "configs/jepa_safe_capture_v2_collection.yaml")
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs/jepa_safe_capture_v2_protocol.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--split", choices=ALLOWED_SPLITS, required=True)
    parser.add_argument("--episodes-per-scenario", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes_per_scenario is not None and args.episodes_per_scenario <= 0:
        raise ValueError("episodes-per-scenario must be positive.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output}")
    if args.tensorboard_logdir.exists() and any(args.tensorboard_logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard directory: {args.tensorboard_logdir}")
    protocol = _load_yaml(args.protocol.resolve())
    collection = _load_yaml(args.collection_config.resolve())
    arrays, metadata = collect(collection, protocol, args.split, args.episodes_per_scenario, args.tensorboard_logdir.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = args.output / "counterfactual_safe_capture_v2.npz"
    np.savez_compressed(dataset, **arrays)
    metadata["collection_config"] = str(args.collection_config.resolve())
    metadata["collection_config_sha256"] = _sha256(args.collection_config.resolve())
    metadata["protocol"] = str(args.protocol.resolve())
    metadata["protocol_sha256"] = _sha256(args.protocol.resolve())
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (args.output / "scenario_manifest.json").write_text(
        json.dumps({"split": args.split, "scenarios": metadata["scenario_records"], "episode_seeds": metadata["episode_seeds"]}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset_version": metadata["dataset_version"],
        "split": args.split,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": _sha256(dataset),
        "metadata": str(metadata_path.resolve()),
        "metadata_sha256": _sha256(metadata_path),
        "scenario_manifest": str((args.output / "scenario_manifest.json").resolve()),
        "scenario_manifest_sha256": _sha256(args.output / "scenario_manifest.json"),
        "tensorboard_logdir": str(args.tensorboard_logdir.resolve()),
        "episode_seeds": metadata["episode_seeds"],
        "locked_test_opened": False,
    }
    (args.output / "archive_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"split": args.split, "samples": int(arrays["inputs"].shape[0]), "episodes": metadata["episode_seed_count"], "dataset_sha256": manifest["dataset_sha256"], "locked_test_opened": False}, indent=2))


if __name__ == "__main__":
    main()
