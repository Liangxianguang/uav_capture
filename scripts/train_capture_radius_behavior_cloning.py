"""Initialize a decentralized capture-radius actor from a local-observation rule expert."""

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

from encirclement3d.learning import CentralizedSharedActorCritic
from encirclement3d.prediction import HistoryTargetPredictor, LearnedPredictionObserver
from encirclement3d.pursuit_controllers import DynamicEncirclementController, SafetyFilteredPursuitController
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--episodes", type=int, help="Optional smoke-run override for imitation episodes.")
    parser.add_argument("--epochs", type=int, help="Optional smoke-run override for optimization epochs.")
    parser.add_argument("--prediction-checkpoint", type=Path, help="Optional frozen GRU predictor checkpoint.")
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
    for name in ("seed", "device", "episodes", "epochs"):
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
        "batch_size",
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


def load_prediction_model(checkpoint_path: Path, device: torch.device) -> HistoryTargetPredictor:
    resolved = checkpoint_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Prediction checkpoint does not exist: {resolved}")
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=True)
    model_config = checkpoint.get("model")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise ValueError("Prediction checkpoint must contain model and model_state_dict.")
    model = HistoryTargetPredictor(**model_config)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def collect_expert_dataset(
    config: dict[str, Any],
    settings: dict[str, Any],
    device: torch.device,
    prediction_model: HistoryTargetPredictor | None = None,
    prediction_history_length: int = 8,
    prediction_horizon_index: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], int]:
    seed = int(settings["seed"])
    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_rows: list[dict[str, Any]] = []
    central_state_dim: int | None = None
    for episode_index in range(int(settings["episodes"])):
        obstacle_count = int(rng.choice(np.asarray(settings["training_obstacle_counts"], dtype=np.int64)))
        target_speed_scale = float(rng.choice(np.asarray(settings["training_target_speed_scales"], dtype=np.float64)))
        env = CaptureRadiusPursuit3DEnv(config, obstacle_count=obstacle_count, target_speed_scale=target_speed_scale)
        controller = SafetyFilteredPursuitController(DynamicEncirclementController(env))
        observation = env.reset(seed=seed + episode_index)
        prediction_observer = (
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
        local_observation = (
            prediction_observer.reset(observation)
            if prediction_observer is not None
            else env.policy_observations(observation)
        )
        if central_state_dim is None:
            central_state_dim = int(env.centralized_state().shape[-1])
        while True:
            observations.append(local_observation)
            action = controller.act(observation)
            actions.append(np.asarray(action, dtype=np.float32))
            observation, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                episode_rows.append(
                    {
                        "episode": episode_index,
                        "seed": seed + episode_index,
                        "obstacle_count": obstacle_count,
                        "target_speed_scale": target_speed_scale,
                        "safe_capture_success": bool(info["safe_capture_success"]),
                        "collision": bool(info["collision"]),
                        "steps": int(env.step_count),
                    }
                )
                break
            local_observation = (
                prediction_observer.observe(observation)
                if prediction_observer is not None
                else env.policy_observations(observation)
            )
    local = np.concatenate(observations, axis=0).astype(np.float32)
    action_values = np.concatenate(actions, axis=0).astype(np.float32)
    manifest = {
        "episodes": episode_rows,
        "sample_count": int(local.shape[0]),
        "expert_safe_capture_rate": float(np.mean([row["safe_capture_success"] for row in episode_rows])),
        "expert_collision_rate": float(np.mean([row["collision"] for row in episode_rows])),
    }
    return local, action_values, manifest, int(central_state_dim)


def evaluate_actor(
    policy: CentralizedSharedActorCritic,
    config: dict[str, Any],
    episode_count: int,
    seed_offset: int,
    device: torch.device,
    action_scale: float,
    prediction_model: HistoryTargetPredictor | None = None,
    prediction_history_length: int = 8,
    prediction_horizon_index: int = 2,
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
                prediction_observer = (
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
                local = (
                    prediction_observer.reset(observation)
                    if prediction_observer is not None
                    else env.policy_observations(observation)
                )
                while True:
                    local_tensor = torch.as_tensor(local, device=device)
                    action = torch.tanh(policy.distribution(local_tensor).mean).cpu().numpy() * action_scale
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
                    local = (
                        prediction_observer.observe(observation)
                        if prediction_observer is not None
                        else env.policy_observations(observation)
                    )
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
    prediction_checkpoint: Path | None = None,
    prediction_history_length: int | None = None,
    prediction_horizon_index: int | None = None,
) -> None:
    output.joinpath("config.yaml").write_text(
        yaml.safe_dump({"imitation_document": document, "effective_imitation": settings, "environment": environment}),
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
    if prediction_checkpoint is not None:
        output.joinpath("prediction_protocol.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(prediction_checkpoint.resolve()),
                    "checkpoint_sha256": hashlib.sha256(prediction_checkpoint.read_bytes()).hexdigest(),
                    "history_length": int(prediction_history_length) if prediction_history_length is not None else None,
                    "horizon_index": int(prediction_horizon_index) if prediction_horizon_index is not None else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    source_paths = [
        PROJECT_ROOT / "scripts" / "train_capture_radius_behavior_cloning.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "learning.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "prediction.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_env.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_controllers.py",
    ]
    output.joinpath("source_hashes.json").write_text(
        json.dumps(
            {
                str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_paths
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def train(args: argparse.Namespace) -> None:
    document, config, settings = load_configuration(args)
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(str(settings["device"]))
    prediction_checkpoint = args.prediction_checkpoint.resolve() if args.prediction_checkpoint is not None else None
    if args.prediction_history_length <= 0 or args.prediction_horizon_index < 0:
        raise ValueError("Prediction history length must be positive and horizon index must be non-negative.")
    prediction_model = load_prediction_model(prediction_checkpoint, device) if prediction_checkpoint is not None else None
    np.random.seed(int(settings["seed"]))
    torch.manual_seed(int(settings["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(settings["seed"]))
    write_artifacts(
        output,
        document,
        config,
        settings,
        device,
        prediction_checkpoint=prediction_checkpoint,
        prediction_history_length=args.prediction_history_length if prediction_checkpoint is not None else None,
        prediction_horizon_index=args.prediction_horizon_index if prediction_checkpoint is not None else None,
    )

    local_data, expert_actions, manifest, centralized_state_dim = collect_expert_dataset(
        config,
        settings,
        device,
        prediction_model=prediction_model,
        prediction_history_length=args.prediction_history_length,
        prediction_horizon_index=args.prediction_horizon_index,
    )
    np.savez_compressed(output / "expert_dataset.npz", local_observations=local_data, actions=expert_actions)
    output.joinpath("expert_dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    action_scale = float(config["agents"]["defender_max_speed"]) / np.sqrt(3.0)
    policy = CentralizedSharedActorCritic(
        local_observation_dim=int(local_data.shape[-1]),
        centralized_state_dim=centralized_state_dim,
        hidden_dim=int(settings["hidden_dim"]),
    ).to(device)
    optimizer = torch.optim.Adam(policy.actor_parameters(), lr=float(settings["learning_rate"]))
    observations = torch.as_tensor(local_data, device=device)
    targets = torch.as_tensor(expert_actions, device=device)
    writer = SummaryWriter(log_dir=str(output / "tensorboard"), flush_secs=10)
    writer.add_text("Config/effective_imitation", yaml.safe_dump(settings, sort_keys=False), 0)
    rng = np.random.default_rng(int(settings["seed"]))
    history: list[dict[str, float | int]] = []
    try:
        for epoch in range(int(settings["epochs"])):
            policy.train()
            permutation = torch.as_tensor(rng.permutation(observations.shape[0]), device=device)
            epoch_losses: list[float] = []
            for start in range(0, observations.shape[0], int(settings["batch_size"])):
                indices = permutation[start : start + int(settings["batch_size"])]
                predicted = torch.tanh(policy.distribution(observations[indices]).mean) * action_scale
                loss = torch.nn.functional.mse_loss(predicted, targets[indices])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.actor_parameters(), 0.5)
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
            evaluation = evaluate_actor(
                policy,
                config,
                episode_count=int(settings["validation_episodes"]),
                seed_offset=880_000,
                device=device,
                action_scale=action_scale,
                prediction_model=prediction_model,
                prediction_history_length=args.prediction_history_length,
                prediction_horizon_index=args.prediction_horizon_index,
            )
            record = {"epoch": epoch + 1, "action_mse": float(np.mean(epoch_losses)), **evaluation}
            history.append(record)
            for key, value in record.items():
                if key != "epoch":
                    writer.add_scalar(f"Imitation/{key}", value, epoch + 1)
            writer.flush()
    finally:
        writer.close()

    with output.joinpath("training.csv").open("w", encoding="utf-8", newline="") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        csv_writer.writeheader()
        csv_writer.writerows(history)
    final_evaluation = evaluate_actor(
        policy,
        config,
        episode_count=int(settings["validation_episodes"]),
        seed_offset=990_000,
        device=device,
        action_scale=action_scale,
        prediction_model=prediction_model,
        prediction_history_length=args.prediction_history_length,
        prediction_horizon_index=args.prediction_horizon_index,
    )
    output.joinpath("evaluation.json").write_text(json.dumps(final_evaluation, indent=2), encoding="utf-8")
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "local_observation_dim": int(local_data.shape[-1]),
            "centralized_state_dim": centralized_state_dim,
            "action_scale": float(action_scale),
            "seed": int(settings["seed"]),
            "algorithm": "behavior_cloning_local_rule_expert",
            "prediction_checkpoint": str(prediction_checkpoint) if prediction_checkpoint is not None else None,
            "prediction_history_length": args.prediction_history_length if prediction_checkpoint is not None else None,
            "prediction_horizon_index": args.prediction_horizon_index if prediction_checkpoint is not None else None,
        },
        output / "checkpoint.pt",
    )
    print(json.dumps(final_evaluation, indent=2))


if __name__ == "__main__":
    train(parse_args())
