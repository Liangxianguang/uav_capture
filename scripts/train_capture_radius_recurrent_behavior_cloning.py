"""Initialize a recurrent decentralized capture actor from a rule expert."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.learning import RecurrentCentralizedSharedActorCritic
from encirclement3d.observation_encoding import policy_observations
from encirclement3d.prediction import HistoryTargetPredictor, LearnedPredictionObserver
from encirclement3d.pursuit_controllers import DynamicEncirclementController, SafetyFilteredPursuitController
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv
from encirclement3d.showcase import sample_training_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--sequence-batch-size", type=int, default=16)
    parser.add_argument(
        "--expert-dataset",
        type=Path,
        help="Reuse a prior expert_sequence_dataset.npz after verifying its manifest.",
    )
    parser.add_argument(
        "--resume-expert-collection",
        action="store_true",
        help="Resume an interrupted locally collected expert archive in --output.",
    )
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Initialize the recurrent actor from a compatible audited checkpoint.",
    )
    parser.add_argument("--prediction-checkpoint", type=Path)
    parser.add_argument("--prediction-history-length", type=int, default=8)
    parser.add_argument("--prediction-horizon-index", type=int, default=2)
    return parser.parse_args()


def load_configuration(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    document = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "imitation" not in document:
        raise ValueError("Behavior-cloning YAML must contain imitation.")
    environment_path = Path(document["environment_config"])
    if not environment_path.is_absolute():
        environment_path = args.config.parent / environment_path
    environment = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    settings = dict(document["imitation"])
    for name in ("seed", "device"):
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
    required = {
        "seed",
        "device",
        "episodes",
        "training_obstacle_counts",
        "training_target_speed_scales",
        "epochs",
        "learning_rate",
        "hidden_dim",
        "validation_episodes",
    }
    missing = sorted(required.difference(settings))
    if missing:
        raise ValueError(f"Missing imitation settings: {', '.join(missing)}")
    return document, environment, settings


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else requested)


def action_scale_for_settings(config: dict[str, Any], settings: dict[str, Any]) -> float:
    """Return the actor action scale while preserving older checkpoint semantics."""
    mode = str(settings.get("action_scale_mode", "per_axis_safe"))
    max_speed = float(config["agents"]["defender_max_speed"])
    if mode == "per_axis_safe":
        return max_speed / np.sqrt(3.0)
    if mode == "full_range":
        return max_speed
    raise ValueError("action_scale_mode must be 'per_axis_safe' or 'full_range'.")


def resolve_initialization_checkpoint(args: argparse.Namespace, settings: dict[str, Any]) -> Path | None:
    """Resolve one explicit CLI or YAML warm-start checkpoint.

    Keeping the path in the YAML makes a retained-BC construction replayable;
    accepting the CLI form preserves existing one-off experiment commands.
    """
    configured = settings.get("initialize_from")
    if args.initialize_from is not None and configured is not None:
        raise ValueError("Use either --initialize-from or imitation.initialize_from, not both.")
    candidate = args.initialize_from if args.initialize_from is not None else configured
    if candidate is None:
        return None
    path = Path(candidate)
    if not path.is_absolute():
        path = args.config.parent / path
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Initialization checkpoint does not exist: {resolved}")
    return resolved


def load_prediction_model(checkpoint_path: Path, device: torch.device) -> HistoryTargetPredictor:
    checkpoint = torch.load(checkpoint_path.resolve(), map_location="cpu", weights_only=True)
    model_config = checkpoint.get("model")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise ValueError("Prediction checkpoint must contain model and model_state_dict.")
    model = HistoryTargetPredictor(**model_config)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def observe(
    env: CaptureRadiusPursuit3DEnv,
    observer: LearnedPredictionObserver | None,
    observation: dict[str, Any],
    reset: bool = False,
) -> np.ndarray:
    if observer is None:
        return policy_observations(env, observation)
    return observer.reset(observation) if reset else observer.observe(observation)


def expert_episode_quality(
    final_info: dict[str, Any],
    defender_zone_entered: np.ndarray,
    episode_metadata: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Apply the configured expert-data acceptance contract to one rollout.

    Failed expert rollouts are useful diagnostics but poor imitation labels:
    retaining their final collision or timeout actions teaches the actor to
    reproduce behavior that the task explicitly rejects.  This helper keeps
    the policy contract explicit and records every rejected attempt.
    """
    entered = np.asarray(defender_zone_entered, dtype=bool)
    if entered.ndim != 1:
        raise ValueError("defender_zone_entered must be a one-dimensional boolean vector.")
    configured_entries = episode_metadata.get("required_defender_zone_entries")
    if configured_entries is None:
        configured_entries = settings.get("training_required_defender_zone_entries", 1)
    required_entries = int(configured_entries)
    if not 1 <= required_entries <= entered.size:
        raise ValueError("Expert rollout required_defender_zone_entries is invalid.")
    safe_capture = bool(final_info.get("safe_capture_success", False))
    entry_count = int(np.sum(entered))
    cooperative_requirement_met = bool(entry_count >= required_entries)
    require_safe_capture = bool(settings.get("expert_require_safe_capture", False))
    require_cooperative_capture = bool(settings.get("expert_require_cooperative_safe_capture", False))
    accepted = bool(
        (not require_safe_capture or safe_capture)
        and (not require_cooperative_capture or (safe_capture and cooperative_requirement_met))
    )
    return {
        "accepted": accepted,
        "safe_capture_success": safe_capture,
        "defender_zone_entry_count": entry_count,
        "required_defender_zone_entries": required_entries,
        "cooperative_requirement_met": cooperative_requirement_met,
    }


def collection_checkpoint_paths(output: Path) -> tuple[Path, Path]:
    """Return the data and metadata paths for an interrupted expert collection."""
    return output / "expert_collection_checkpoint.npz", output / "expert_collection_progress.json"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def write_collection_checkpoint(
    output: Path,
    *,
    local_frames: np.ndarray,
    action_frames: np.ndarray,
    reset_frames: np.ndarray,
    accepted_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    total_attempts: int,
    centralized_state_dim: int,
    rng_state: dict[str, Any],
) -> None:
    """Atomically persist only the raw expert collection needed for resumption."""
    data_path, metadata_path = collection_checkpoint_paths(output)
    temporary_data_path = data_path.with_name(f"{data_path.stem}.tmp.npz")
    temporary_metadata_path = metadata_path.with_name(f"{metadata_path.stem}.tmp.json")
    np.savez_compressed(
        temporary_data_path,
        local_observations=local_frames,
        actions=action_frames,
        reset_masks=reset_frames,
    )
    temporary_data_path.replace(data_path)
    temporary_metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "accepted_episodes": len(accepted_rows),
                "rejected_episodes": len(rejected_rows),
                "collection_attempts": int(total_attempts),
                "centralized_state_dim": int(centralized_state_dim),
                "rng_state": rng_state,
                "episodes": accepted_rows,
                "rejected_expert_episodes": rejected_rows,
            },
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    temporary_metadata_path.replace(metadata_path)


def load_collection_checkpoint(output: Path) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
    int,
    dict[str, Any],
]:
    """Load one compatible interrupted collection without changing episode order."""
    data_path, metadata_path = collection_checkpoint_paths(output)
    if not data_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Expert collection checkpoint and progress JSON must both exist to resume.")
    with np.load(data_path) as archive:
        required = {"local_observations", "actions", "reset_masks"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"Expert collection checkpoint is missing: {', '.join(missing)}")
        local_frames = np.asarray(archive["local_observations"], dtype=np.float32)
        action_frames = np.asarray(archive["actions"], dtype=np.float32)
        reset_frames = np.asarray(archive["reset_masks"], dtype=np.float32)
    if local_frames.ndim != 3 or action_frames.shape != (*local_frames.shape[:2], 3):
        raise ValueError("Expert collection checkpoint has incompatible local-observation or action shapes.")
    if reset_frames.shape != (local_frames.shape[0],):
        raise ValueError("Expert collection checkpoint has incompatible reset-mask shape.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or int(metadata.get("schema_version", -1)) != 1:
        raise ValueError("Expert collection progress JSON has an unsupported schema.")
    accepted_rows = metadata.get("episodes")
    rejected_rows = metadata.get("rejected_expert_episodes")
    rng_state = metadata.get("rng_state")
    if not isinstance(accepted_rows, list) or not isinstance(rejected_rows, list) or not isinstance(rng_state, dict):
        raise ValueError("Expert collection progress JSON is missing resumable episode data.")
    if int(metadata.get("accepted_episodes", -1)) != len(accepted_rows):
        raise ValueError("Expert collection accepted episode count is inconsistent.")
    if int(metadata.get("rejected_episodes", -1)) != len(rejected_rows):
        raise ValueError("Expert collection rejected episode count is inconsistent.")
    total_attempts = int(metadata.get("collection_attempts", -1))
    centralized_state_dim = int(metadata.get("centralized_state_dim", -1))
    if total_attempts < len(accepted_rows) or centralized_state_dim <= 0:
        raise ValueError("Expert collection progress JSON has invalid counters.")
    return (
        local_frames,
        action_frames,
        reset_frames,
        accepted_rows,
        rejected_rows,
        total_attempts,
        centralized_state_dim,
        rng_state,
    )


def collect_expert_dataset(
    config: dict[str, Any],
    settings: dict[str, Any],
    device: torch.device,
    prediction_model: HistoryTargetPredictor | None,
    prediction_history_length: int,
    prediction_horizon_index: int,
    *,
    output: Path | None = None,
    resume: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], int]:
    seed = int(settings["seed"])
    rng = np.random.default_rng(seed)
    local_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    reset_chunks: list[np.ndarray] = []
    episode_rows: list[dict[str, Any]] = []
    centralized_state_dim: int | None = None
    requested_episodes = int(settings["episodes"])
    maximum_attempts_per_episode = int(settings.get("expert_max_attempts_per_episode", 1))
    if maximum_attempts_per_episode <= 0:
        raise ValueError("expert_max_attempts_per_episode must be positive.")
    accepted_episodes = 0
    total_attempts = 0
    rejected_rows: list[dict[str, Any]] = []
    checkpoint_interval = int(settings.get("expert_collection_checkpoint_every_accepted_episodes", 0))
    if checkpoint_interval < 0:
        raise ValueError("expert_collection_checkpoint_every_accepted_episodes must be non-negative.")
    if resume:
        if output is None:
            raise ValueError("Resuming expert collection requires an output directory.")
        (
            previous_local,
            previous_actions,
            previous_resets,
            episode_rows,
            rejected_rows,
            total_attempts,
            previous_state_dim,
            rng_state,
        ) = load_collection_checkpoint(output)
        rng.bit_generator.state = rng_state
        local_chunks.append(previous_local)
        action_chunks.append(previous_actions)
        reset_chunks.append(previous_resets)
        accepted_episodes = len(episode_rows)
        centralized_state_dim = previous_state_dim
        if accepted_episodes >= requested_episodes:
            raise ValueError("Resumable expert collection already has the requested number of accepted episodes.")
    attempt_committed = True
    rng_state_before_attempt: dict[str, Any] | None = None
    try:
        while accepted_episodes < requested_episodes:
            if total_attempts >= requested_episodes * maximum_attempts_per_episode:
                raise RuntimeError(
                    "Unable to collect the requested number of accepted expert episodes within "
                    "expert_max_attempts_per_episode. Inspect rejected_expert_episodes in the manifest."
                )
            attempt_committed = False
            rng_state_before_attempt = rng.bit_generator.state
            rollout_seed = seed + total_attempts
            env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.55)
            observation, episode_metadata = sample_training_episode(
                env,
                settings,
                rng,
                seed=rollout_seed,
                progress=accepted_episodes / max(requested_episodes - 1, 1),
            )
            controller = SafetyFilteredPursuitController(DynamicEncirclementController(env))
            observer = (
                LearnedPredictionObserver(
                    env,
                    prediction_model,
                    device,
                    history_length=prediction_history_length,
                    horizon_index=prediction_horizon_index,
                )
                if prediction_model is not None
                else None
            )
            local = observe(env, observer, observation, reset=True)
            if centralized_state_dim is None:
                centralized_state_dim = int(env.centralized_state().shape[-1])
            rollout_local_frames: list[np.ndarray] = []
            rollout_action_frames: list[np.ndarray] = []
            rollout_reset_frames: list[bool] = []
            obstacle_zone_x = episode_metadata.get("obstacle_zone_x")
            defender_zone_entered = np.zeros(env.n_defenders, dtype=bool)
            if obstacle_zone_x is not None:
                low, high = (float(value) for value in obstacle_zone_x)
                defender_zone_entered |= (env.defender_positions[:, 0] >= low) & (env.defender_positions[:, 0] <= high)
            while True:
                rollout_local_frames.append(local)
                rollout_reset_frames.append(len(rollout_local_frames) == 1)
                action = np.asarray(controller.act(observation), dtype=np.float32)
                rollout_action_frames.append(action)
                observation, _reward, terminated, truncated, info = env.step(action)
                if obstacle_zone_x is not None:
                    defender_zone_entered |= (env.defender_positions[:, 0] >= low) & (env.defender_positions[:, 0] <= high)
                if terminated or truncated:
                    quality = expert_episode_quality(info, defender_zone_entered, episode_metadata, settings)
                    row = {
                        "episode": accepted_episodes if bool(quality["accepted"]) else None,
                        "attempt": total_attempts,
                        "seed": rollout_seed,
                        "obstacle_count": int(env.obstacle_count),
                        "target_speed_scale": float(env.target_speed_scale),
                        "collision": bool(info["collision"]),
                        "steps": int(env.step_count),
                        "termination_reason": str(info["termination_reason"]),
                        **episode_metadata,
                        **quality,
                    }
                    if bool(quality["accepted"]):
                        local_chunks.append(np.asarray(rollout_local_frames, dtype=np.float32))
                        action_chunks.append(np.asarray(rollout_action_frames, dtype=np.float32))
                        reset_chunks.append(np.asarray(rollout_reset_frames, dtype=np.float32))
                        episode_rows.append(row)
                        accepted_episodes += 1
                    else:
                        rejected_rows.append(row)
                    break
                local = observe(env, observer, observation)
            accepted_this_attempt = bool(quality["accepted"])
            total_attempts += 1
            attempt_committed = True
            if (
                output is not None
                and checkpoint_interval
                and accepted_this_attempt
                and accepted_episodes % checkpoint_interval == 0
            ):
                write_collection_checkpoint(
                    output,
                    local_frames=np.concatenate(local_chunks, axis=0),
                    action_frames=np.concatenate(action_chunks, axis=0),
                    reset_frames=np.concatenate(reset_chunks, axis=0),
                    accepted_rows=episode_rows,
                    rejected_rows=rejected_rows,
                    total_attempts=total_attempts,
                    centralized_state_dim=int(centralized_state_dim),
                    rng_state=rng.bit_generator.state,
                )
    except BaseException:
        # An interrupted in-flight rollout has consumed curriculum RNG but has
        # not created a label. Replay it on resume with the same attempt seed.
        if rng_state_before_attempt is not None and not attempt_committed:
            rng.bit_generator.state = rng_state_before_attempt
        if output is not None and centralized_state_dim is not None and episode_rows:
            write_collection_checkpoint(
                output,
                local_frames=np.concatenate(local_chunks, axis=0),
                action_frames=np.concatenate(action_chunks, axis=0),
                reset_frames=np.concatenate(reset_chunks, axis=0),
                accepted_rows=episode_rows,
                rejected_rows=rejected_rows,
                total_attempts=total_attempts,
                centralized_state_dim=int(centralized_state_dim),
                rng_state=rng.bit_generator.state,
            )
        raise
    local_values = np.concatenate(local_chunks, axis=0).astype(np.float32, copy=False)
    action_values = np.concatenate(action_chunks, axis=0).astype(np.float32, copy=False)
    reset_values = np.concatenate(reset_chunks, axis=0).astype(np.float32, copy=False)
    manifest = {
        "episodes": episode_rows,
        "rejected_expert_episodes": rejected_rows,
        "frame_count": int(local_values.shape[0]),
        "expert_safe_capture_rate": float(np.mean([row["safe_capture_success"] for row in episode_rows])),
        "expert_collision_rate": float(np.mean([row["collision"] for row in episode_rows])),
        "expert_cooperative_requirement_rate": float(
            np.mean([row["cooperative_requirement_met"] for row in episode_rows])
        ),
        "requested_episodes": requested_episodes,
        "accepted_episodes": len(episode_rows),
        "rejected_episodes": len(rejected_rows),
        "collection_attempts": total_attempts,
        "expert_rejection_rate": float(len(rejected_rows) / max(total_attempts, 1)),
    }
    return local_values, action_values, reset_values, manifest, int(centralized_state_dim)


def load_reused_expert_dataset(
    dataset_path: Path,
    config: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], int]:
    """Load a previously audited recurrent BC dataset without silently changing it."""
    resolved = dataset_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Expert dataset does not exist: {resolved}")
    with np.load(resolved) as archive:
        required = {"local_observations", "actions", "reset_masks"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"Expert dataset is missing: {', '.join(missing)}")
        local_sequences = np.asarray(archive["local_observations"], dtype=np.float32)
        action_sequences = np.asarray(archive["actions"], dtype=np.float32)
        reset_sequences = np.asarray(archive["reset_masks"], dtype=np.float32)
    if local_sequences.ndim != 4 or action_sequences.shape[:3] != local_sequences.shape[:3]:
        raise ValueError("Reused expert dataset has incompatible recurrent sequence shapes.")
    if action_sequences.shape[-1] != 3 or reset_sequences.shape != local_sequences.shape[:2]:
        raise ValueError("Reused expert dataset has incompatible action or reset-mask shapes.")
    manifest_path = resolved.with_name("expert_dataset_manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Expert dataset manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Expert dataset manifest must be a JSON object.")
    prototype = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=int(settings["training_obstacle_counts"][0]),
        target_speed_scale=float(settings["training_target_speed_scales"][0]),
    )
    prototype.reset(seed=int(settings["seed"]))
    centralized_state_dim = int(prototype.centralized_state().shape[-1])
    expected_local_dim = int(policy_observations(prototype).shape[-1])
    if local_sequences.shape[-1] != expected_local_dim:
        raise ValueError(
            "Reused expert dataset observation dimension does not match the selected environment configuration."
        )
    manifest = {
        **manifest,
        "reused_expert_dataset": str(resolved),
        "reused_expert_dataset_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "reused_expert_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    return local_sequences, action_sequences, reset_sequences, manifest, centralized_state_dim


def resolve_configured_dataset_paths(configured_paths: Any, config_path: Path) -> list[Path]:
    """Resolve the explicit expert archive list recorded in a BC YAML."""
    if not isinstance(configured_paths, list) or not configured_paths:
        raise ValueError("expert_datasets must be a non-empty list when provided.")
    paths: list[Path] = []
    for value in configured_paths:
        path = Path(str(value))
        paths.append(path if path.is_absolute() else config_path.parent / path)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("expert_datasets must not contain the same archive more than once.")
    return paths


def load_reused_expert_datasets(
    dataset_paths: list[Path],
    config: dict[str, Any],
    settings: dict[str, Any],
    *,
    source_balance: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], int]:
    """Load audited archives and explicitly balance their source contributions."""
    if source_balance not in {"proportional", "equal_sequences"}:
        raise ValueError("expert_dataset_source_balance must be 'proportional' or 'equal_sequences'.")
    archives = [load_reused_expert_dataset(path, config, settings) for path in dataset_paths]
    state_dimensions = {state_dim for *_values, state_dim in archives}
    if len(state_dimensions) != 1:
        raise ValueError("Reused expert datasets disagree on centralized-state dimension.")
    sequence_counts = [int(local.shape[0]) for local, *_values in archives]
    if any(count <= 0 for count in sequence_counts):
        raise ValueError("Reused expert datasets must each contain at least one sequence.")
    target_count = max(sequence_counts) if source_balance == "equal_sequences" else None
    rng = np.random.default_rng(seed)
    local_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    reset_parts: list[np.ndarray] = []
    source_metadata: list[dict[str, Any]] = []
    for source_index, (local, actions, resets, manifest, _state_dim) in enumerate(archives):
        original_count = int(local.shape[0])
        selected_indices = (
            np.arange(original_count, dtype=np.int64)
            if target_count is None
            else rng.choice(original_count, size=target_count, replace=original_count < target_count)
        )
        local_parts.append(local[selected_indices])
        action_parts.append(actions[selected_indices])
        reset_parts.append(resets[selected_indices])
        source_metadata.append(
            {
                "source_index": source_index,
                "original_sequences": original_count,
                "selected_sequences": int(selected_indices.size),
                "manifest": manifest,
            }
        )
    local_sequences = np.concatenate(local_parts, axis=0)
    action_sequences = np.concatenate(action_parts, axis=0)
    reset_sequences = np.concatenate(reset_parts, axis=0)
    manifest = {
        "reused_expert_datasets": source_metadata,
        "source_balance": source_balance,
        "source_balance_seed": int(seed),
        "sequence_count": int(local_sequences.shape[0]),
        "frame_count": int(local_sequences.shape[0] * local_sequences.shape[1]),
    }
    return local_sequences, action_sequences, reset_sequences, manifest, state_dimensions.pop()


def initialize_recurrent_actor(
    policy: RecurrentCentralizedSharedActorCritic,
    checkpoint_path: Path | None,
    *,
    local_observation_dim: int,
    centralized_state_dim: int,
    action_scale: float,
    device: torch.device,
) -> dict[str, Any] | None:
    """Load a full compatible recurrent BC checkpoint before fine-tuning."""
    if checkpoint_path is None:
        return None
    resolved = checkpoint_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Initialization checkpoint does not exist: {resolved}")
    checkpoint = torch.load(resolved, map_location=device, weights_only=True)
    if not bool(checkpoint.get("actor_recurrent", False)):
        raise ValueError("Initialization checkpoint must contain a recurrent actor.")
    if int(checkpoint.get("local_observation_dim", -1)) != local_observation_dim:
        raise ValueError("Initialization checkpoint observation dimension differs from this BC configuration.")
    if int(checkpoint.get("centralized_state_dim", -1)) != centralized_state_dim:
        raise ValueError("Initialization checkpoint critic-state dimension differs from this BC configuration.")
    if not np.isclose(float(checkpoint.get("action_scale", np.nan)), action_scale):
        raise ValueError("Initialization checkpoint action scale differs from this BC configuration.")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Initialization checkpoint has no state_dict.")
    policy.load_state_dict(state_dict, strict=True)
    return {
        "checkpoint": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "source_algorithm": checkpoint.get("algorithm"),
        "source_seed": checkpoint.get("seed"),
    }


def evaluate_actor(
    policy: RecurrentCentralizedSharedActorCritic,
    config: dict[str, Any],
    episode_count: int,
    seed_offset: int,
    device: torch.device,
    action_scale: float,
    prediction_model: HistoryTargetPredictor | None,
    prediction_history_length: int,
    prediction_horizon_index: int,
) -> dict[str, float]:
    outcomes: list[dict[str, float | bool]] = []
    policy.eval()
    with torch.no_grad():
        for scenario_index, experiment in enumerate(config["experiments"]):
            for episode_index in range(episode_count):
                env = CaptureRadiusPursuit3DEnv(
                    config,
                    obstacle_count=int(experiment["obstacle_count"]),
                    target_speed_scale=float(experiment["target_speed_scale"]),
                )
                observation = env.reset(seed=seed_offset + scenario_index * 10_000 + episode_index)
                observer = (
                    LearnedPredictionObserver(
                        env,
                        prediction_model,
                        device,
                        history_length=prediction_history_length,
                        horizon_index=prediction_horizon_index,
                    )
                    if prediction_model is not None
                    else None
                )
                local = observe(env, observer, observation, reset=True)
                hidden = policy.initial_actor_hidden(env.n_defenders, device=device)
                while True:
                    distribution, hidden = policy.distribution_step(
                        torch.as_tensor(local, device=device),
                        hidden,
                    )
                    action = torch.tanh(distribution.mean).cpu().numpy() * action_scale
                    observation, _reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        outcomes.append(
                            {
                                "safe_capture": bool(info["safe_capture_success"]),
                                "capture": bool(info["capture_event"]),
                                "collision": bool(info["collision_steps"]),
                                "capture_time": float(info["capture_time_seconds"])
                                if info["capture_time_seconds"] is not None
                                else float(config["world"]["max_steps"]) * float(config["world"]["dt"]),
                            }
                        )
                        break
                    local = observe(env, observer, observation)
    return {
        "safe_capture_rate": float(np.mean([row["safe_capture"] for row in outcomes])),
        "capture_rate": float(np.mean([row["capture"] for row in outcomes])),
        "collision_rate": float(np.mean([row["collision"] for row in outcomes])),
        "mean_capture_time_seconds": float(np.mean([row["capture_time"] for row in outcomes])),
    }


def write_artifacts(
    output: Path,
    document: dict[str, Any],
    environment: dict[str, Any],
    settings: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    prediction_checkpoint: Path | None,
) -> None:
    recurrent_arguments = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    output.joinpath("config.yaml").write_text(
        yaml.safe_dump(
            {
                "imitation_document": document,
                "effective_imitation": settings,
                "recurrent": recurrent_arguments,
                "environment": environment,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output.joinpath("environment.txt").write_text(
        "\n".join(
            [
                f"python={sys.version.replace(chr(10), ' ')}",
                f"platform={platform.platform()}",
                f"numpy={version('numpy')}",
                f"torch={version('torch')}",
                f"tensorboard={version('tensorboard')}",
                f"device={device}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source_paths = [
        PROJECT_ROOT / "scripts" / "train_capture_radius_recurrent_behavior_cloning.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "learning.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "observation_encoding.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "prediction.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_env.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_controllers.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "showcase.py",
    ]
    output.joinpath("source_hashes.json").write_text(
        json.dumps({str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}, indent=2),
        encoding="utf-8",
    )
    if prediction_checkpoint is not None:
        output.joinpath("prediction_protocol.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(prediction_checkpoint.resolve()),
                    "checkpoint_sha256": hashlib.sha256(prediction_checkpoint.read_bytes()).hexdigest(),
                    "history_length": args.prediction_history_length,
                    "horizon_index": args.prediction_horizon_index,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    if args.sequence_length <= 0 or args.sequence_batch_size <= 0:
        raise ValueError("sequence lengths and batch size must be positive.")
    if args.prediction_history_length <= 0 or args.prediction_horizon_index < 0:
        raise ValueError("Invalid prediction history arguments.")
    document, config, settings = load_configuration(args)
    configured_datasets = settings.get("expert_datasets")
    if args.expert_dataset is not None and configured_datasets is not None:
        raise ValueError("Use either --expert-dataset or imitation.expert_datasets, not both.")
    initialization_checkpoint = resolve_initialization_checkpoint(args, settings)
    output = args.output
    if args.resume_expert_collection:
        if args.expert_dataset is not None or configured_datasets is not None:
            raise ValueError("--resume-expert-collection only supports locally collected expert data.")
        if output.joinpath("checkpoint.pt").is_file() or output.joinpath("expert_sequence_dataset.npz").is_file():
            raise ValueError("Cannot resume expert collection after a completed training artifact was written.")
        data_path, metadata_path = collection_checkpoint_paths(output)
        if not data_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("--resume-expert-collection requires a saved collection checkpoint in --output.")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(str(settings["device"]))
    prediction_checkpoint = args.prediction_checkpoint.resolve() if args.prediction_checkpoint is not None else None
    if prediction_checkpoint is not None and not prediction_checkpoint.is_file():
        raise FileNotFoundError(f"Prediction checkpoint does not exist: {prediction_checkpoint}")
    prediction_model = load_prediction_model(prediction_checkpoint, device) if prediction_checkpoint is not None else None
    np.random.seed(int(settings["seed"]))
    torch.manual_seed(int(settings["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(settings["seed"]))
    write_artifacts(output, document, config, settings, device, args, prediction_checkpoint)

    if args.expert_dataset is None and configured_datasets is None:
        local_data, expert_actions, reset_masks, manifest, centralized_state_dim = collect_expert_dataset(
            config,
            settings,
            device,
            prediction_model,
            args.prediction_history_length,
            args.prediction_horizon_index,
            output=output,
            resume=bool(args.resume_expert_collection),
        )
        manifest["resumed_expert_collection"] = bool(args.resume_expert_collection)
        maximum_rejection_rate = float(settings.get("expert_max_rejection_rate", 1.0))
        if not 0.0 <= maximum_rejection_rate <= 1.0:
            raise ValueError("expert_max_rejection_rate must lie in [0, 1].")
        if float(manifest["expert_rejection_rate"]) > maximum_rejection_rate:
            output.joinpath("expert_dataset_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            raise RuntimeError(
                "Expert rejection rate exceeds expert_max_rejection_rate; training is stopped before "
                "writing a dataset or checkpoint. See expert_dataset_manifest.json for rejected rollouts."
            )
        usable_frames = (local_data.shape[0] // args.sequence_length) * args.sequence_length
        if usable_frames == 0:
            raise RuntimeError("Expert trajectory dataset is shorter than one recurrent sequence.")
        local_data = local_data[:usable_frames]
        expert_actions = expert_actions[:usable_frames]
        reset_masks = reset_masks[:usable_frames]
        local_sequences = local_data.reshape(-1, args.sequence_length, *local_data.shape[1:])
        action_sequences = expert_actions.reshape(-1, args.sequence_length, *expert_actions.shape[1:])
        reset_sequences = reset_masks.reshape(-1, args.sequence_length)
        # Expert episodes have arbitrary lengths. Each BC sequence starts from
        # a zero hidden state at the truncated-BPTT chunk boundary.
        reset_sequences[:, 0] = 1.0
        manifest.update(
            {
                "sequence_length": args.sequence_length,
                "sequence_count": int(local_sequences.shape[0]),
                "discarded_frames": int(manifest["frame_count"] - usable_frames),
                "chunk_boundaries_reset_hidden": True,
            }
        )
    elif args.expert_dataset is not None:
        local_sequences, action_sequences, reset_sequences, manifest, centralized_state_dim = load_reused_expert_dataset(
            args.expert_dataset,
            config,
            settings,
        )
        if local_sequences.shape[1] != args.sequence_length:
            raise ValueError("Reused expert dataset sequence length does not match --sequence-length.")
        local_data = local_sequences.reshape(-1, *local_sequences.shape[2:])
    else:
        dataset_paths = resolve_configured_dataset_paths(configured_datasets, args.config)
        local_sequences, action_sequences, reset_sequences, manifest, centralized_state_dim = load_reused_expert_datasets(
            dataset_paths,
            config,
            settings,
            source_balance=str(settings.get("expert_dataset_source_balance", "proportional")),
            seed=int(settings["seed"]),
        )
        if local_sequences.shape[1] != args.sequence_length:
            raise ValueError("Configured expert dataset sequence length does not match --sequence-length.")
        local_data = local_sequences.reshape(-1, *local_sequences.shape[2:])
    np.savez_compressed(
        output / "expert_sequence_dataset.npz",
        local_observations=local_sequences,
        actions=action_sequences,
        reset_masks=reset_sequences,
    )
    output.joinpath("expert_dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    action_scale = action_scale_for_settings(config, settings)
    policy = RecurrentCentralizedSharedActorCritic(
        local_observation_dim=int(local_data.shape[-1]),
        centralized_state_dim=centralized_state_dim,
        hidden_dim=int(settings["hidden_dim"]),
    ).to(device)
    initialization = initialize_recurrent_actor(
        policy,
        initialization_checkpoint,
        local_observation_dim=int(local_data.shape[-1]),
        centralized_state_dim=centralized_state_dim,
        action_scale=action_scale,
        device=device,
    )
    if initialization is not None:
        output.joinpath("initialization.json").write_text(json.dumps(initialization, indent=2), encoding="utf-8")
    optimizer = torch.optim.Adam(policy.actor_parameters(), lr=float(settings["learning_rate"]))
    local_tensor = torch.as_tensor(local_sequences, device=device)
    action_tensor = torch.as_tensor(action_sequences, device=device)
    reset_tensor = torch.as_tensor(reset_sequences, device=device)
    writer = SummaryWriter(log_dir=str(output / "tensorboard"), flush_secs=10)
    writer.add_text("Config/effective_imitation", yaml.safe_dump(settings, sort_keys=False), 0)
    history: list[dict[str, float | int]] = []
    rng = np.random.default_rng(int(settings["seed"]))
    try:
        for epoch in range(int(settings["epochs"])):
            policy.train()
            losses: list[float] = []
            permutation = torch.as_tensor(rng.permutation(local_tensor.shape[0]), device=device)
            for start in range(0, local_tensor.shape[0], args.sequence_batch_size):
                indices = permutation[start : start + args.sequence_batch_size]
                selected_local = local_tensor[indices]
                selected_actions = action_tensor[indices]
                selected_resets = reset_tensor[indices]
                hidden = policy.initial_actor_hidden(
                    selected_local.shape[2], batch_size=selected_local.shape[0], device=device
                )
                _log_probabilities, _entropy, means = policy.evaluate_actions_sequence(
                    selected_local,
                    hidden,
                    selected_resets,
                    selected_actions,
                    action_scale,
                )
                loss = torch.nn.functional.mse_loss(torch.tanh(means) * action_scale, selected_actions)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.actor_parameters(), 0.5)
                optimizer.step()
                losses.append(float(loss.detach()))
            evaluation = evaluate_actor(
                policy,
                config,
                int(settings["validation_episodes"]),
                880_000,
                device,
                action_scale,
                prediction_model,
                args.prediction_history_length,
                args.prediction_horizon_index,
            )
            record = {"epoch": epoch + 1, "action_mse": float(np.mean(losses)), **evaluation}
            history.append(record)
            for key, value in record.items():
                if key != "epoch":
                    writer.add_scalar(f"Imitation/{key}", float(value), epoch + 1)
            writer.flush()
    finally:
        writer.close()

    with output.joinpath("training.csv").open("w", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=list(history[0])).writeheader()
        csv.DictWriter(stream, fieldnames=list(history[0])).writerows(history)
    final_evaluation = evaluate_actor(
        policy,
        config,
        int(settings["validation_episodes"]),
        990_000,
        device,
        action_scale,
        prediction_model,
        args.prediction_history_length,
        args.prediction_horizon_index,
    )
    output.joinpath("evaluation.json").write_text(json.dumps(final_evaluation, indent=2), encoding="utf-8")
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "local_observation_dim": int(local_data.shape[-1]),
            "centralized_state_dim": centralized_state_dim,
            "action_scale": float(action_scale),
            "action_scale_mode": str(settings.get("action_scale_mode", "per_axis_safe")),
            "seed": int(settings["seed"]),
            "algorithm": "behavior_cloning_recurrent_local_rule_expert",
            "actor_recurrent": True,
            "recurrent_hidden_dim": int(settings["hidden_dim"]),
            "prediction_checkpoint": str(prediction_checkpoint) if prediction_checkpoint is not None else None,
            "prediction_history_length": args.prediction_history_length if prediction_checkpoint is not None else None,
            "prediction_horizon_index": args.prediction_horizon_index if prediction_checkpoint is not None else None,
        },
        output / "checkpoint.pt",
    )
    print(json.dumps(final_evaluation, indent=2))


if __name__ == "__main__":
    main()
