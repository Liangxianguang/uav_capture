"""Train MAPPO for partially observable 3D capture-radius pursuit."""

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

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.learning import CentralizedSharedActorCritic
from encirclement3d.prediction import HistoryTargetPredictor, LearnedPredictionObserver
from encirclement3d.pursuit_controllers import PursuitCBFSafetyFilter
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--initialize-from", type=Path, help="Optional behavior-cloning checkpoint for actor warm start.")
    parser.add_argument("--prediction-checkpoint", type=Path, help="Optional frozen GRU predictor checkpoint.")
    parser.add_argument("--prediction-history-length", type=int, default=8)
    parser.add_argument("--prediction-horizon-index", type=int, default=2)
    return parser.parse_args()


def load_configuration(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    document = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "training" not in document:
        raise ValueError("MAPPO YAML must contain training.")
    environment_path = Path(document["environment_config"])
    if not environment_path.is_absolute():
        environment_path = args.config.parent / environment_path
    environment = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    settings = dict(document["training"])
    for name in ("seed", "total_steps", "device"):
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
    required = {
        "seed",
        "device",
        "total_steps",
        "rollout_steps",
        "epochs",
        "minibatch_size",
        "learning_rate",
        "evaluation_episodes",
        "evaluation_interval",
        "gamma",
        "gae_lambda",
        "clip_ratio",
        "value_coefficient",
        "entropy_coefficient",
        "max_gradient_norm",
        "hidden_dim",
        "torch_num_threads",
        "deterministic_algorithms",
        "training_obstacle_counts",
        "training_target_speed_scales",
    }
    missing = sorted(required.difference(settings))
    if missing:
        raise ValueError(f"Missing MAPPO settings: {', '.join(missing)}")
    if int(settings["rollout_steps"]) <= 0 or int(settings["minibatch_size"]) <= 0:
        raise ValueError("rollout_steps and minibatch_size must be positive.")
    return document, environment, settings


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else requested)


def cuda_details(device: torch.device) -> dict[str, Any]:
    details: dict[str, Any] = {"torch_cuda_available": torch.cuda.is_available(), "torch_cuda_runtime": torch.version.cuda}
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


def reset_training_episode(
    env: CaptureRadiusPursuit3DEnv,
    settings: dict[str, Any],
    rng: np.random.Generator,
    seed: int,
) -> dict[str, Any]:
    env.obstacle_count = int(rng.choice(np.asarray(settings["training_obstacle_counts"], dtype=np.int64)))
    env.target_speed_scale = float(rng.choice(np.asarray(settings["training_target_speed_scales"], dtype=np.float64)))
    return env.reset(seed=seed)


def evaluate(
    policy: CentralizedSharedActorCritic,
    config: dict[str, Any],
    episode_count: int,
    seed_offset: int,
    device: torch.device,
    action_scale: float,
    use_safety_filter: bool = False,
    prediction_model: HistoryTargetPredictor | None = None,
    prediction_history_length: int = 8,
    prediction_horizon_index: int = 2,
) -> dict[str, float]:
    policy.eval()
    outcomes: list[dict[str, float | bool]] = []
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
                safety_filter = PursuitCBFSafetyFilter(env) if use_safety_filter else None
                while True:
                    local_tensor = torch.as_tensor(local, device=device)
                    actions = torch.tanh(policy.distribution(local_tensor).mean).cpu().numpy() * action_scale
                    if safety_filter is not None:
                        actions, _diagnostics = safety_filter.filter(actions, observation)
                    next_observation, _reward, terminated, truncated, info = env.step(actions)
                    if terminated or truncated:
                        outcomes.append(
                            {
                                "success": bool(info["safe_capture_success"]),
                                "capture": bool(info["capture_event"]),
                                "collision": bool(info["collision_steps"]),
                                "capture_time": float(info["capture_time_seconds"])
                                if info["capture_time_seconds"] is not None
                                else float(config["world"]["max_steps"]) * float(config["world"]["dt"]),
                            }
                        )
                        break
                    observation = next_observation
                    local = (
                        prediction_observer.observe(observation)
                        if prediction_observer is not None
                        else env.policy_observations(observation)
                    )
    return {
        "safe_capture_rate": sum(bool(row["success"]) for row in outcomes) / len(outcomes),
        "capture_rate": sum(bool(row["capture"]) for row in outcomes) / len(outcomes),
        "collision_rate": sum(bool(row["collision"]) for row in outcomes) / len(outcomes),
        "mean_capture_time_seconds": float(np.mean([float(row["capture_time"]) for row in outcomes])),
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
        yaml.safe_dump(
            {"training_document": document, "effective_training": settings, "environment": environment},
            sort_keys=False,
        ),
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
    pip_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    output.joinpath("environment.txt").write_text(
        "\n".join(
            [
                f"python={sys.version.replace(chr(10), ' ')}",
                f"platform={platform.platform()}",
                f"numpy={version('numpy')}",
                f"torch={version('torch')}",
                f"tensorboard={version('tensorboard')}",
                f"device={device}",
                *[f"{key}={value}" for key, value in cuda_details(device).items()],
                "",
                "pip_freeze:",
                pip_freeze.rstrip(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source_paths = [
        PROJECT_ROOT / "scripts" / "train_capture_radius_mappo.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "learning.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_env.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "prediction.py",
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


def write_training_history(output: Path, history: list[dict[str, float | int]]) -> None:
    if not history:
        return
    with output.joinpath("training.csv").open("w", encoding="utf-8", newline="") as stream:
        writer_csv = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(history)


def initialize_actor(
    policy: CentralizedSharedActorCritic,
    checkpoint_path: Path | None,
    device: torch.device,
    expected_observation_dim: int,
    expected_action_scale: float,
    warm_start_log_std: float | None,
) -> dict[str, Any] | None:
    if checkpoint_path is None:
        return None
    resolved = checkpoint_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Initialization checkpoint does not exist: {resolved}")
    checkpoint = torch.load(resolved, map_location=device, weights_only=True)
    if int(checkpoint.get("local_observation_dim", -1)) != expected_observation_dim:
        raise ValueError("Initialization checkpoint observation dimension differs from this pursuit task.")
    if not np.isclose(float(checkpoint.get("action_scale", np.nan)), expected_action_scale):
        raise ValueError("Initialization checkpoint action scale differs from this pursuit task.")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Initialization checkpoint has no state_dict.")
    actor_keys = ("actor_body.", "actor_mean.", "log_std")
    actor_state = {name: value for name, value in state_dict.items() if name.startswith(actor_keys)}
    expected_actor_keys = {f"actor_body.{name}" for name in policy.actor_body.state_dict()} | {
        f"actor_mean.{name}" for name in policy.actor_mean.state_dict()
    } | {"log_std"}
    if set(actor_state) != expected_actor_keys:
        raise ValueError("Initialization checkpoint does not contain a compatible shared actor.")
    policy.load_state_dict(actor_state, strict=False)
    if warm_start_log_std is not None:
        with torch.no_grad():
            policy.log_std.fill_(float(warm_start_log_std))
    return {
        "checkpoint": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "source_algorithm": checkpoint.get("algorithm"),
        "source_seed": checkpoint.get("seed"),
        "warm_start_log_std": warm_start_log_std,
    }
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

    seed = int(settings["seed"])
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(settings["torch_num_threads"]))
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(bool(settings["deterministic_algorithms"]), warn_only=True)

    env = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=int(settings["training_obstacle_counts"][0]),
        target_speed_scale=float(settings["training_target_speed_scales"][0]),
    )
    observation = reset_training_episode(env, settings, rng, seed)
    prediction_model = load_prediction_model(prediction_checkpoint, device) if prediction_checkpoint is not None else None
    prediction_observer = (
        LearnedPredictionObserver(
            env,
            prediction_model,
            device,
            history_length=args.prediction_history_length,
            horizon_index=args.prediction_horizon_index,
        )
        if prediction_model is not None
        else None
    )
    local_observation = (
        prediction_observer.reset(observation)
        if prediction_observer is not None
        else env.policy_observations(observation)
    )
    local_observation_dim = int(local_observation.shape[-1])
    centralized_state_dim = int(env.centralized_state().shape[-1])
    action_scale = float(config["agents"]["defender_max_speed"]) / np.sqrt(3.0)
    policy = CentralizedSharedActorCritic(
        local_observation_dim=local_observation_dim,
        centralized_state_dim=centralized_state_dim,
        hidden_dim=int(settings["hidden_dim"]),
    ).to(device)
    initialization = initialize_actor(
        policy,
        args.initialize_from,
        device,
        local_observation_dim,
        action_scale,
        (float(settings["warm_start_log_std"]) if "warm_start_log_std" in settings else None),
    )
    if initialization is not None:
        output.joinpath("initialization.json").write_text(json.dumps(initialization, indent=2), encoding="utf-8")
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(settings["learning_rate"]), eps=1e-5)
    writer = SummaryWriter(log_dir=str(output / "tensorboard"), flush_secs=10)
    writer.add_text("Config/effective_training", yaml.safe_dump(settings, sort_keys=False), 0)
    writer.add_text("Config/environment", yaml.safe_dump(config, sort_keys=False), 0)

    total_steps = int(settings["total_steps"])
    steps_completed = 0
    update_index = 0
    episode_seed = seed
    history: list[dict[str, float | int]] = []
    last_evaluation = {
        "safe_capture_rate": 0.0,
        "capture_rate": 0.0,
        "collision_rate": 0.0,
        "mean_capture_time_seconds": float(config["world"]["max_steps"]) * float(config["world"]["dt"]),
    }
    started = time.perf_counter()
    try:
        while steps_completed < total_steps:
            local_rollout: list[np.ndarray] = []
            state_rollout: list[np.ndarray] = []
            action_rollout: list[np.ndarray] = []
            log_probability_rollout: list[float] = []
            value_rollout: list[float] = []
            reward_rollout: list[float] = []
            terminal_rollout: list[bool] = []
            episode_returns: list[float] = []
            current_episode_return = 0.0
            policy.train()

            for _ in range(min(int(settings["rollout_steps"]), total_steps - steps_completed)):
                local = local_observation
                centralized_state = env.centralized_state()
                with torch.no_grad():
                    local_tensor = torch.as_tensor(local, device=device)
                    state_tensor = torch.as_tensor(centralized_state[None, :], device=device)
                    action_tensor, local_log_probabilities = policy.sample_actions(local_tensor, action_scale)
                    value_tensor = policy.value(state_tensor)
                action = action_tensor.cpu().numpy()
                next_observation, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)
                local_rollout.append(local)
                state_rollout.append(centralized_state)
                action_rollout.append(action)
                log_probability_rollout.append(float(local_log_probabilities.sum().cpu()))
                value_rollout.append(float(value_tensor.cpu().item()))
                reward_rollout.append(float(reward))
                terminal_rollout.append(done)
                current_episode_return += float(reward)
                steps_completed += 1
                observation = next_observation
                if done:
                    episode_returns.append(current_episode_return)
                    current_episode_return = 0.0
                    episode_seed += 1
                    observation = reset_training_episode(env, settings, rng, episode_seed)
                    local_observation = (
                        prediction_observer.reset(observation)
                        if prediction_observer is not None
                        else env.policy_observations(observation)
                    )
                else:
                    local_observation = (
                        prediction_observer.observe(observation)
                        if prediction_observer is not None
                        else env.policy_observations(observation)
                    )

            with torch.no_grad():
                next_value = float(
                    policy.value(torch.as_tensor(env.centralized_state()[None, :], device=device)).cpu().item()
                )
            values = np.asarray(value_rollout, dtype=np.float32)
            advantages = np.zeros_like(values)
            running_advantage = 0.0
            gamma = float(settings["gamma"])
            gae_lambda = float(settings["gae_lambda"])
            for index in reversed(range(len(reward_rollout))):
                continuation = 1.0 - float(terminal_rollout[index])
                following_value = next_value if index == len(reward_rollout) - 1 else values[index + 1]
                delta = reward_rollout[index] + gamma * following_value * continuation - values[index]
                running_advantage = delta + gamma * gae_lambda * continuation * running_advantage
                advantages[index] = running_advantage
            returns = advantages + values
            normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            local_batch = torch.as_tensor(np.asarray(local_rollout), device=device)
            state_batch = torch.as_tensor(np.asarray(state_rollout), device=device)
            action_batch = torch.as_tensor(np.asarray(action_rollout), device=device)
            old_log_probability_batch = torch.as_tensor(np.asarray(log_probability_rollout), device=device)
            advantage_batch = torch.as_tensor(normalized_advantages, device=device)
            return_batch = torch.as_tensor(returns, device=device)
            sample_count = local_batch.shape[0]
            losses: list[dict[str, float]] = []
            for _epoch in range(int(settings["epochs"])):
                permutation = torch.randperm(sample_count, device=device)
                for start in range(0, sample_count, int(settings["minibatch_size"])):
                    indices = permutation[start : start + int(settings["minibatch_size"])]
                    selected_local = local_batch[indices]
                    selected_actions = action_batch[indices]
                    selected_log_probabilities, selected_entropy = policy.evaluate_actions(
                        selected_local.reshape(-1, local_observation_dim),
                        selected_actions.reshape(-1, 3),
                        action_scale,
                    )
                    selected_log_probabilities = selected_log_probabilities.reshape(-1, env.n_defenders).sum(dim=1)
                    selected_entropy = selected_entropy.reshape(-1, env.n_defenders).sum(dim=1)
                    predicted_values = policy.value(state_batch[indices])
                    ratio = (selected_log_probabilities - old_log_probability_batch[indices]).exp()
                    surrogate_a = ratio * advantage_batch[indices]
                    surrogate_b = torch.clamp(
                        ratio,
                        1.0 - float(settings["clip_ratio"]),
                        1.0 + float(settings["clip_ratio"]),
                    ) * advantage_batch[indices]
                    policy_loss = -torch.minimum(surrogate_a, surrogate_b).mean()
                    value_loss = 0.5 * (return_batch[indices] - predicted_values).pow(2).mean()
                    entropy = selected_entropy.mean()
                    loss = (
                        policy_loss
                        + float(settings["value_coefficient"]) * value_loss
                        - float(settings["entropy_coefficient"]) * entropy
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), float(settings["max_gradient_norm"]))
                    optimizer.step()
                    losses.append(
                        {
                            "total": float(loss.detach()),
                            "policy": float(policy_loss.detach()),
                            "value": float(value_loss.detach()),
                            "entropy": float(entropy.detach()),
                        }
                    )

            evaluation_ran = update_index % int(settings["evaluation_interval"]) == 0 or steps_completed >= total_steps
            if evaluation_ran:
                last_evaluation = evaluate(
                    policy,
                    config,
                    int(settings["evaluation_episodes"]),
                    seed_offset=800_000 + update_index * 10_000,
                    device=device,
                    action_scale=action_scale,
                    use_safety_filter=bool(settings.get("evaluation_use_cbf", False)),
                    prediction_model=prediction_model,
                    prediction_history_length=args.prediction_history_length,
                    prediction_horizon_index=args.prediction_horizon_index,
                )
            mean_losses = {name: float(np.mean([item[name] for item in losses])) for name in losses[0]}
            record = {
                "update": update_index,
                "environment_steps": steps_completed,
                "mean_rollout_return": float(np.mean(episode_returns)) if episode_returns else current_episode_return,
                "action_std": float(policy.log_std.detach().exp().mean().cpu()),
                "evaluation_ran": int(evaluation_ran),
                **{f"{name}_loss": value for name, value in mean_losses.items()},
                **last_evaluation,
            }
            history.append(record)
            write_training_history(output, history)
            for key, value in mean_losses.items():
                writer.add_scalar(f"Loss/{key}", value, steps_completed)
            writer.add_scalar("Training/rollout_return", record["mean_rollout_return"], steps_completed)
            writer.add_scalar("Policy/action_std", record["action_std"], steps_completed)
            if evaluation_ran:
                for key, value in last_evaluation.items():
                    writer.add_scalar(f"Evaluation/{key}", value, steps_completed)
            writer.flush()
            update_index += 1
    finally:
        writer.close()

    final_evaluation = evaluate(
        policy,
        config,
        int(settings["evaluation_episodes"]),
        seed_offset=990_000,
        device=device,
        action_scale=action_scale,
        use_safety_filter=bool(settings.get("evaluation_use_cbf", False)),
        prediction_model=prediction_model,
        prediction_history_length=args.prediction_history_length,
        prediction_horizon_index=args.prediction_horizon_index,
    )
    output.joinpath("evaluation.json").write_text(json.dumps(final_evaluation, indent=2), encoding="utf-8")
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "local_observation_dim": local_observation_dim,
            "centralized_state_dim": centralized_state_dim,
            "action_scale": float(action_scale),
            "seed": seed,
            "algorithm": "mappo_ctde",
            "prediction_checkpoint": str(prediction_checkpoint) if prediction_checkpoint is not None else None,
            "prediction_horizon_index": args.prediction_horizon_index if prediction_checkpoint is not None else None,
            "prediction_history_length": args.prediction_history_length if prediction_checkpoint is not None else None,
        },
        output / "checkpoint.pt",
    )
    output.joinpath("run_metadata.json").write_text(
        json.dumps(
            {
                "algorithm": "mappo_ctde",
                "seed": seed,
                "total_environment_steps": steps_completed,
                "elapsed_seconds": time.perf_counter() - started,
                "device": str(device),
                "cuda": cuda_details(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(final_evaluation, indent=2))


if __name__ == "__main__":
    train(parse_args())
