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
        return env.policy_observations(observation)
    return observer.reset(observation) if reset else observer.observe(observation)


def collect_expert_dataset(
    config: dict[str, Any],
    settings: dict[str, Any],
    device: torch.device,
    prediction_model: HistoryTargetPredictor | None,
    prediction_history_length: int,
    prediction_horizon_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], int]:
    seed = int(settings["seed"])
    rng = np.random.default_rng(seed)
    local_frames: list[np.ndarray] = []
    action_frames: list[np.ndarray] = []
    reset_frames: list[bool] = []
    episode_rows: list[dict[str, Any]] = []
    centralized_state_dim: int | None = None
    for episode_index in range(int(settings["episodes"])):
        env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.55)
        observation, episode_metadata = sample_training_episode(
            env,
            settings,
            rng,
            seed=seed + episode_index,
            progress=episode_index / max(int(settings["episodes"]) - 1, 1),
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
        episode_start = len(local_frames)
        while True:
            local_frames.append(local)
            reset_frames.append(len(local_frames) - 1 == episode_start)
            action = np.asarray(controller.act(observation), dtype=np.float32)
            action_frames.append(action)
            observation, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                episode_rows.append(
                    {
                        "episode": episode_index,
                        "seed": seed + episode_index,
                        "obstacle_count": int(env.obstacle_count),
                        "target_speed_scale": float(env.target_speed_scale),
                        "safe_capture_success": bool(info["safe_capture_success"]),
                        "collision": bool(info["collision"]),
                        "steps": int(env.step_count),
                        **episode_metadata,
                    }
                )
                break
            local = observe(env, observer, observation)
    local_values = np.asarray(local_frames, dtype=np.float32)
    action_values = np.asarray(action_frames, dtype=np.float32)
    reset_values = np.asarray(reset_frames, dtype=np.float32)
    manifest = {
        "episodes": episode_rows,
        "frame_count": int(local_values.shape[0]),
        "expert_safe_capture_rate": float(np.mean([row["safe_capture_success"] for row in episode_rows])),
        "expert_collision_rate": float(np.mean([row["collision"] for row in episode_rows])),
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
    manifest = {
        **manifest,
        "reused_expert_dataset": str(resolved),
        "reused_expert_dataset_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "reused_expert_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    return local_sequences, action_sequences, reset_sequences, manifest, centralized_state_dim


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
    output = args.output
    if output.exists() and any(output.iterdir()):
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

    if args.expert_dataset is None:
        local_data, expert_actions, reset_masks, manifest, centralized_state_dim = collect_expert_dataset(
            config,
            settings,
            device,
            prediction_model,
            args.prediction_history_length,
            args.prediction_horizon_index,
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
    else:
        local_sequences, action_sequences, reset_sequences, manifest, centralized_state_dim = load_reused_expert_dataset(
            args.expert_dataset,
            config,
            settings,
        )
        if local_sequences.shape[1] != args.sequence_length:
            raise ValueError("Reused expert dataset sequence length does not match --sequence-length.")
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
