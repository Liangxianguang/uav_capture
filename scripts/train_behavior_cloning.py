"""Train a supervised policy to imitate the verified 3D slot controller.

This is an expert-imitation baseline, not a replacement for MARL. It tests
whether the observation encoder and shared actor can represent a successful
kinematic policy before PPO fine-tuning is attempted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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

from encirclement3d.controllers import CBFSafetyFilteredSlotController, TetrahedralSlotController
from encirclement3d.dynamics import InertialEncirclement3DEnv
from encirclement3d.environment import Encirclement3DEnv
from encirclement3d.learning import SharedActorCritic, defender_observations
from encirclement3d.pybullet_env import PyBulletEncirclement3DEnv


def scenario_by_name(config: dict[str, Any], name: str) -> dict[str, Any]:
    scenario = next((item for item in config["experiments"] if item["name"] == name), None)
    if scenario is None:
        available = ", ".join(str(item["name"]) for item in config["experiments"])
        raise ValueError(f"Unknown scenario {name!r}; available: {available}")
    return scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    training_document = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    environment_path = Path(training_document["environment_config"])
    if not environment_path.is_absolute():
        environment_path = args.config.parent / environment_path
    environment_config = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    settings = dict(training_document["training"])
    if args.device is not None:
        settings["device"] = args.device
    if args.seed is not None:
        settings["seed"] = args.seed
    required = {
        "seed",
        "device",
        "dataset_episodes",
        "epochs",
        "minibatch_size",
        "learning_rate",
        "evaluation_episodes",
        "histogram_interval",
        "torch_num_threads",
        "deterministic_algorithms",
        "include_agent_id",
    }
    missing = sorted(required.difference(settings))
    if missing:
        raise ValueError(f"Missing training settings: {', '.join(missing)}")
    _deep_update(environment_config, training_document.get("environment_overrides", {}))
    environment_config.setdefault("dynamics", {})["backend"] = str(
        settings.get("environment_backend", environment_config["dynamics"].get("backend", "kinematic"))
    )
    return training_document, environment_config, settings


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def make_environment(config: dict[str, Any], obstacle_count: int, target_speed_scale: float):
    backend = str(config.get("dynamics", {}).get("backend", "kinematic"))
    if backend == "pybullet":
        return PyBulletEncirclement3DEnv(config, obstacle_count, target_speed_scale)
    if backend == "inertial":
        return InertialEncirclement3DEnv(config, obstacle_count, target_speed_scale)
    return Encirclement3DEnv(config, obstacle_count, target_speed_scale)


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def device_metadata(device: torch.device) -> dict[str, Any]:
    details: dict[str, Any] = {
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        details.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "device_capability": list(torch.cuda.get_device_capability(device)),
                "total_memory_bytes": properties.total_memory,
            }
        )
    return details


def write_artifacts(
    output: Path,
    training_document: dict[str, Any],
    environment_config: dict[str, Any],
    settings: dict[str, Any],
    device: torch.device,
) -> None:
    output.joinpath("config.yaml").write_text(
        yaml.safe_dump(
            {
                "training_document": training_document,
                "effective_training": settings,
                "environment": environment_config,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True
    ).stdout
    lines = [
        f"python={sys.version.replace(chr(10), ' ')}",
        f"platform={platform.platform()}",
        f"numpy={version('numpy')}",
        f"torch={version('torch')}",
        f"PyYAML={version('PyYAML')}",
        f"tensorboard={version('tensorboard')}",
        f"device={device}",
        *[f"{key}={value}" for key, value in device_metadata(device).items()],
        "",
        "pip_freeze:",
        freeze.rstrip(),
    ]
    output.joinpath("environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    source_paths = [
        PROJECT_ROOT / "scripts" / "train_behavior_cloning.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "learning.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "controllers.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "safety.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "environment.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "dynamics.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pybullet_env.py",
        PROJECT_ROOT / "third_party" / "gym-pybullet-drones-7688e7208a1572b1680736a3c0c9b93c379db3fe" / "gym_pybullet_drones" / "envs" / "BaseAviary.py",
        PROJECT_ROOT / "third_party" / "gym-pybullet-drones-7688e7208a1572b1680736a3c0c9b93c379db3fe" / "gym_pybullet_drones" / "envs" / "CtrlAviary.py",
        PROJECT_ROOT / "third_party" / "gym-pybullet-drones-7688e7208a1572b1680736a3c0c9b93c379db3fe" / "gym_pybullet_drones" / "control" / "DSLPIDControl.py",
    ]
    hashes = {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    output.joinpath("source_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")


def collect_expert_dataset(
    config: dict[str, Any],
    episode_count: int,
    seed: int,
    position_scale: float,
    defender_speed_scale: float,
    target_speed_scale: float,
    include_agent_id: bool,
    action_scale: float,
    scenario_name: str,
    expert_controller: str,
    obstacle_feature_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    scenario = scenario_by_name(config, scenario_name)
    if expert_controller not in {"rule", "cbf"}:
        raise ValueError("expert_controller must be 'rule' or 'cbf'.")
    observations: list[np.ndarray] = []
    raw_targets: list[np.ndarray] = []
    episode_rows: list[dict[str, float | int | bool]] = []
    for episode in range(episode_count):
        env = make_environment(
            config,
            obstacle_count=int(scenario["obstacle_count"]),
            target_speed_scale=float(scenario["target_speed_scale"]),
        )
        try:
            observation = env.reset(seed=seed + episode)
            controller = TetrahedralSlotController(env) if expert_controller == "rule" else CBFSafetyFilteredSlotController(env)
            while True:
                local = defender_observations(
                    observation,
                    env.n_defenders,
                    position_scale=position_scale,
                    defender_speed_scale=defender_speed_scale,
                    target_speed_scale=target_speed_scale,
                    include_agent_id=include_agent_id,
                    obstacle_feature_count=obstacle_feature_count,
                )
                expert_action = controller.act(observation)
                # The actor uses independent per-axis squashing. Match the
                # representable action range, then invert tanh for the regression
                # target in the actor's unsquashed output space.
                clipped_action = np.clip(expert_action, -action_scale, action_scale)
                target = np.arctanh(np.clip(clipped_action / action_scale, -0.999999, 0.999999))
                observations.append(local)
                raw_targets.append(target.astype(np.float32))
                observation, _reward, terminated, truncated, info = env.step(expert_action)
                if terminated or truncated:
                    episode_rows.append(
                        {
                            "seed": seed + episode,
                            "success": bool(info["success"]),
                            "collision": bool(info["collision_steps"]),
                            "steps": int(env.step_count),
                            "final_slot_error": float(info["mean_slot_error"]),
                        }
                    )
                    break
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()
    return (
        np.concatenate(observations, axis=0).astype(np.float32),
        np.concatenate(raw_targets, axis=0).astype(np.float32),
        {
            "scenario": scenario_name,
            "expert_controller": expert_controller,
            "episodes": episode_rows,
            "episode_success_rate": float(np.mean([row["success"] for row in episode_rows])),
            "episode_collision_rate": float(np.mean([row["collision"] for row in episode_rows])),
        },
    )


def evaluate(
    policy: SharedActorCritic,
    config: dict[str, Any],
    episodes: int,
    seed: int,
    position_scale: float,
    defender_speed_scale: float,
    target_speed_scale: float,
    include_agent_id: bool,
    action_scale: float,
    device: torch.device,
    scenario_name: str,
    obstacle_feature_count: int,
) -> dict[str, float]:
    policy.eval()
    scenario = scenario_by_name(config, scenario_name)
    outcomes: list[dict[str, float | bool]] = []
    with torch.no_grad():
        for episode in range(episodes):
            env = make_environment(
                config,
                obstacle_count=int(scenario["obstacle_count"]),
                target_speed_scale=float(scenario["target_speed_scale"]),
            )
            try:
                observation = env.reset(seed=seed + episode)
                while True:
                    local = defender_observations(
                        observation,
                        env.n_defenders,
                        position_scale=position_scale,
                        defender_speed_scale=defender_speed_scale,
                        target_speed_scale=target_speed_scale,
                        include_agent_id=include_agent_id,
                        obstacle_feature_count=obstacle_feature_count,
                    )
                    distribution, _value = policy.distribution_and_value(torch.as_tensor(local, device=device))
                    action = (torch.tanh(distribution.mean) * action_scale).cpu().numpy()
                    observation, _reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        outcomes.append(
                            {
                                "success": bool(info["success"]),
                                "collision": bool(info["collision_steps"]),
                                "slot_error": float(info["mean_slot_error"]),
                            }
                        )
                        break
            finally:
                close = getattr(env, "close", None)
                if close is not None:
                    close()
    return {
        "success_rate": float(np.mean([bool(item["success"]) for item in outcomes])),
        "collision_rate": float(np.mean([bool(item["collision"]) for item in outcomes])),
        "mean_final_slot_error": float(np.mean([float(item["slot_error"]) for item in outcomes])),
    }


def checkpoint_payload(
    state_dict: dict[str, torch.Tensor],
    observation_dim: int,
    settings: dict[str, Any],
    action_scale: float,
    training_scenario: str,
    evaluation_scenario: str,
    expert_controller: str,
    obstacle_feature_count: int,
    sample_count: int,
) -> dict[str, Any]:
    return {
        "state_dict": state_dict,
        "observation_dim": observation_dim,
        "hidden_dim": int(settings.get("hidden_dim", 128)),
        "action_scale": action_scale,
        "include_agent_id": bool(settings["include_agent_id"]),
        "action_scale_mode": str(settings.get("action_scale_mode", "per_axis_safe")),
        "training_scenario": training_scenario,
        "evaluation_scenario": evaluation_scenario,
        "expert_controller": expert_controller,
        "obstacle_feature_count": obstacle_feature_count,
        "seed": int(settings["seed"]),
        "algorithm": "behavior_cloning_from_tetrahedral_slot_controller",
        "dataset_samples": sample_count,
        "device": str(settings["device"]),
    }


def train(args: argparse.Namespace) -> None:
    training_document, config, settings = load_config(args)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    device = select_device(str(settings["device"]))
    seed = int(settings["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(settings["torch_num_threads"]))
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(bool(settings["deterministic_algorithms"]), warn_only=True)
    write_artifacts(args.output, training_document, config, settings, device)

    training_scenario = str(settings.get("training_scenario", "easy"))
    evaluation_scenario = str(settings.get("evaluation_scenario", training_scenario))
    expert_controller = str(settings.get("expert_controller", "rule"))
    obstacle_feature_count = int(settings.get("obstacle_feature_count", 0))
    scenario = scenario_by_name(config, training_scenario)
    env = make_environment(
        config,
        obstacle_count=int(scenario["obstacle_count"]),
        target_speed_scale=float(scenario["target_speed_scale"]),
    )
    observation = env.reset(seed=seed)
    position_scale = float(config["world"]["half_extent_xy"])
    defender_speed_scale = float(env.agents["defender_max_speed"])
    target_speed_scale = float(env.agents["target_max_speed"])
    action_scale = (
        float(env.agents["defender_max_speed"])
        if str(settings.get("action_scale_mode", "per_axis_safe")) == "full_range"
        else float(env.agents["defender_max_speed"]) / np.sqrt(3.0)
    )
    close = getattr(env, "close", None)
    if close is not None:
        close()
    include_agent_id = bool(settings["include_agent_id"])
    observation_dim = defender_observations(
        observation,
        env.n_defenders,
        position_scale=position_scale,
        defender_speed_scale=defender_speed_scale,
        target_speed_scale=target_speed_scale,
        include_agent_id=include_agent_id,
        obstacle_feature_count=obstacle_feature_count,
    ).shape[-1]
    policy = SharedActorCritic(observation_dim, hidden_dim=int(settings.get("hidden_dim", 128))).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(settings["learning_rate"]), eps=1e-5)
    data_observations, data_targets, dataset_summary = collect_expert_dataset(
        config,
        int(settings["dataset_episodes"]),
        seed,
        position_scale,
        defender_speed_scale,
        target_speed_scale,
        include_agent_id,
        action_scale,
        training_scenario,
        expert_controller,
        obstacle_feature_count,
    )
    dataset_summary["samples"] = int(data_observations.shape[0])
    args.output.joinpath("expert_dataset.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    observations_tensor = torch.as_tensor(data_observations, device=device)
    targets_tensor = torch.as_tensor(data_targets, device=device)
    sample_count = observations_tensor.shape[0]
    writer = SummaryWriter(log_dir=str(args.output / "tensorboard"), flush_secs=10)
    writer.add_text("Config/effective_training", f"```yaml\n{yaml.safe_dump(settings, sort_keys=False)}```", 0)
    writer.add_text("Dataset/summary", f"```json\n{json.dumps(dataset_summary, indent=2)}\n```", 0)
    writer.add_scalar("Dataset/expert_success_rate", float(dataset_summary["episode_success_rate"]), 0)
    writer.add_scalar("Dataset/expert_collision_rate", float(dataset_summary["episode_collision_rate"]), 0)
    history: list[dict[str, float | int]] = []
    selection_enabled = bool(settings.get("select_best_checkpoint", False))
    fixed_evaluation_seed = settings.get("evaluation_seed")
    best_key: tuple[float, float, float] | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_selection: dict[str, float | int] | None = None
    started = time.perf_counter()
    try:
        for epoch in range(int(settings["epochs"])):
            policy.train()
            permutation = torch.randperm(sample_count, device=device)
            epoch_losses: list[float] = []
            for start in range(0, sample_count, int(settings["minibatch_size"])):
                indices = permutation[start : start + int(settings["minibatch_size"])]
                distribution, _value = policy.distribution_and_value(observations_tensor[indices])
                loss = torch.nn.functional.mse_loss(distribution.loc, targets_tensor[indices])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
            mse = float(np.mean(epoch_losses))
            record: dict[str, float | int] = {
                "epoch": epoch,
                "samples": sample_count,
                "expert_mse": mse,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            if epoch == 0 or (epoch + 1) % int(settings["histogram_interval"]) == 0 or epoch + 1 == int(settings["epochs"]):
                evaluation_seed = (
                    int(fixed_evaluation_seed)
                    if fixed_evaluation_seed is not None
                    else 90_000 + epoch * 1000
                )
                metrics = evaluate(
                    policy,
                    config,
                    int(settings["evaluation_episodes"]),
                    evaluation_seed,
                    position_scale,
                    defender_speed_scale,
                    target_speed_scale,
                    include_agent_id,
                    action_scale,
                    device,
                    evaluation_scenario,
                    obstacle_feature_count,
                )
                record.update(metrics)
                writer.add_scalar("Evaluation/success_rate", metrics["success_rate"], epoch)
                writer.add_scalar("Evaluation/collision_rate", metrics["collision_rate"], epoch)
                writer.add_scalar("Evaluation/final_slot_error", metrics["mean_final_slot_error"], epoch)
                candidate_key = (
                    float(metrics["success_rate"]),
                    -float(metrics["collision_rate"]),
                    -float(metrics["mean_final_slot_error"]),
                )
                if selection_enabled and (best_key is None or candidate_key > best_key):
                    best_key = candidate_key
                    best_state_dict = {
                        name: parameter.detach().cpu().clone()
                        for name, parameter in policy.state_dict().items()
                    }
                    best_selection = {
                        "epoch": epoch,
                        "evaluation_seed": evaluation_seed,
                        **metrics,
                    }
                    writer.add_scalar("Selection/best_epoch", epoch, epoch)
            writer.add_scalar("Loss/expert_mse", mse, epoch)
            writer.add_scalar("Training/learning_rate", record["learning_rate"], epoch)
            if epoch % int(settings["histogram_interval"]) == 0:
                for name, parameter in policy.named_parameters():
                    writer.add_histogram(f"Parameters/{name}", parameter.detach().cpu(), epoch)
            writer.flush()
            history.append(record)
    finally:
        writer.close()

    with (args.output / "training.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(history[0].keys())
        writer_csv = csv.DictWriter(handle, fieldnames=fields)
        writer_csv.writeheader()
        writer_csv.writerows(history)
    final_metrics = evaluate(
        policy,
        config,
        int(settings["evaluation_episodes"]),
        99_000,
        position_scale,
        defender_speed_scale,
        target_speed_scale,
        include_agent_id,
        action_scale,
        device,
        evaluation_scenario,
        obstacle_feature_count,
    )
    (args.output / "evaluation.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    last_payload = checkpoint_payload(
        policy.state_dict(),
        observation_dim,
        settings,
        action_scale,
        training_scenario,
        evaluation_scenario,
        expert_controller,
        obstacle_feature_count,
        sample_count,
    )
    if selection_enabled and best_state_dict is not None:
        selected_payload = checkpoint_payload(
            best_state_dict,
            observation_dim,
            settings,
            action_scale,
            training_scenario,
            evaluation_scenario,
            expert_controller,
            obstacle_feature_count,
            sample_count,
        )
        torch.save(selected_payload, args.output / "checkpoint.pt")
        torch.save(selected_payload, args.output / "best_checkpoint.pt")
        torch.save(last_payload, args.output / "last_checkpoint.pt")
    else:
        torch.save(last_payload, args.output / "checkpoint.pt")
    args.output.joinpath("selection_metadata.json").write_text(
        json.dumps(
            {
                "enabled": selection_enabled,
                "evaluation_seed": fixed_evaluation_seed,
                "selection_key": "success_rate, then lower collision_rate, then lower mean_final_slot_error",
                "selected": best_selection,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output / "run_metadata.json").write_text(
        json.dumps(
            {
                "algorithm": "behavior_cloning_from_tetrahedral_slot_controller",
                "seed": seed,
                "dataset_samples": sample_count,
                "elapsed_seconds": time.perf_counter() - started,
                "device": str(device),
                "cuda": device_metadata(device),
                "checkpoint_selection": best_selection,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    train(parse_args())
