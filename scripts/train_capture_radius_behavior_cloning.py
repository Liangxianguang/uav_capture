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
from encirclement3d.pursuit_controllers import DynamicEncirclementController, SafetyFilteredPursuitController
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
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


def collect_expert_dataset(
    config: dict[str, Any],
    settings: dict[str, Any],
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
        if central_state_dim is None:
            central_state_dim = int(env.centralized_state().shape[-1])
        while True:
            observations.append(env.policy_observations(observation))
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
                while True:
                    local = torch.as_tensor(env.policy_observations(observation), device=device)
                    action = torch.tanh(policy.distribution(local).mean).cpu().numpy() * action_scale
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
    source_paths = [
        PROJECT_ROOT / "scripts" / "train_capture_radius_behavior_cloning.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "learning.py",
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
    np.random.seed(int(settings["seed"]))
    torch.manual_seed(int(settings["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(settings["seed"]))
    write_artifacts(output, document, config, settings, device)

    local_data, expert_actions, manifest, centralized_state_dim = collect_expert_dataset(config, settings)
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
        },
        output / "checkpoint.pt",
    )
    print(json.dumps(final_evaluation, indent=2))


if __name__ == "__main__":
    train(parse_args())
