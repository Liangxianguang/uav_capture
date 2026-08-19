"""Evaluate one frozen MAPPO capture-radius policy on a fixed test block."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.learning import CentralizedSharedActorCritic, RecurrentCentralizedSharedActorCritic
from encirclement3d.prediction import HistoryTargetPredictor, LearnedPredictionObserver
from encirclement3d.pursuit_controllers import PursuitCBFSafetyFilter
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Frozen environment YAML.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="First seed in the locked test block.")
    parser.add_argument("--episodes", type=int, required=True, help="Episodes per configured scenario.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--use-cbf", action="store_true", help="Apply the local-information CBF execution filter.")
    parser.add_argument("--prediction-checkpoint", type=Path, help="Optional frozen GRU predictor checkpoint.")
    parser.add_argument("--prediction-history-length", type=int, default=8)
    parser.add_argument("--prediction-horizon-index", type=int, default=2)
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else requested)


def safe_load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load native checkpoints, with a narrow allow-list for the early v1 artifact."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception as error:
        if not isinstance(error, (RuntimeError, ValueError, pickle.UnpicklingError)):
            raise
        safe_types = [np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType]
        with torch.serialization.safe_globals(safe_types):
            return torch.load(path, map_location=device, weights_only=True)


def load_policy(
    checkpoint_path: Path,
    env: CaptureRadiusPursuit3DEnv,
    observation: dict[str, Any],
    device: torch.device,
) -> tuple[CentralizedSharedActorCritic | RecurrentCentralizedSharedActorCritic, float, dict[str, Any]]:
    checkpoint = safe_load_checkpoint(checkpoint_path, device)
    local_dim = int(env.policy_observations(observation).shape[-1])
    state_dim = int(env.centralized_state().shape[-1])
    if int(checkpoint.get("local_observation_dim", -1)) != local_dim:
        raise ValueError("Checkpoint observation dimension does not match the selected environment YAML.")
    if int(checkpoint.get("centralized_state_dim", -1)) != state_dim:
        raise ValueError("Checkpoint critic-state dimension does not match the selected environment YAML.")
    if bool(checkpoint.get("actor_recurrent", False)):
        hidden_dim = int(checkpoint.get("recurrent_hidden_dim", 128))
        policy: CentralizedSharedActorCritic | RecurrentCentralizedSharedActorCritic = RecurrentCentralizedSharedActorCritic(
            local_dim,
            state_dim,
            hidden_dim=hidden_dim,
        ).to(device)
    else:
        policy = CentralizedSharedActorCritic(local_dim, state_dim).to(device)
    policy.load_state_dict(checkpoint["state_dict"], strict=True)
    return policy.eval(), float(checkpoint["action_scale"]), checkpoint


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


def rollout_episode(
    policy: CentralizedSharedActorCritic | RecurrentCentralizedSharedActorCritic,
    config: dict[str, Any],
    obstacle_count: int,
    target_speed_scale: float,
    seed: int,
    device: torch.device,
    action_scale: float,
    use_cbf: bool,
    record_history: bool,
    prediction_model: HistoryTargetPredictor | None = None,
    prediction_history_length: int = 8,
    prediction_horizon_index: int = 2,
) -> tuple[dict[str, Any], CaptureRadiusPursuit3DEnv]:
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=obstacle_count, target_speed_scale=target_speed_scale)
    observation = env.reset(seed=seed, record_history=record_history)
    safety_filter = PursuitCBFSafetyFilter(env) if use_cbf else None
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
    actor_hidden = (
        policy.initial_actor_hidden(env.n_defenders, device=device)
        if isinstance(policy, RecurrentCentralizedSharedActorCritic)
        else None
    )
    visible_fractions: list[float] = []
    message_ages: list[float] = []
    final_info: dict[str, Any] = {}
    with torch.no_grad():
        while True:
            local = torch.as_tensor(local_observation, device=device)
            if isinstance(policy, RecurrentCentralizedSharedActorCritic):
                distribution, actor_hidden = policy.distribution_step(local, actor_hidden)
            else:
                distribution = policy.distribution(local)
            action = torch.tanh(distribution.mean).cpu().numpy() * action_scale
            if safety_filter is not None:
                action, _diagnostics = safety_filter.filter(action, observation)
            observation, _reward, terminated, truncated, final_info = env.step(action, record_history=record_history)
            visible_fractions.append(float(final_info["target_visible_fraction"]))
            message_ages.append(float(final_info["mean_message_age_steps"]))
            if terminated or truncated:
                break
            local_observation = (
                prediction_observer.observe(observation)
                if prediction_observer is not None
                else env.policy_observations(observation)
            )
    return (
        {
            "seed": seed,
            "safe_capture_success": bool(final_info["safe_capture_success"]),
            "capture_event": bool(final_info["capture_event"]),
            "collision": bool(final_info["collision"]),
            "capture_time_seconds": final_info["capture_time_seconds"],
            "capturing_defender_id": final_info["capturing_defender_id"],
            "steps": int(env.step_count),
            "termination_reason": str(final_info["termination_reason"]),
            "world_violation_steps": int(final_info["world_violation_steps"]),
            "min_clearance_m": float(final_info["min_clearance_so_far"]),
            "mean_visible_fraction": float(np.mean(visible_fractions)),
            "mean_message_age_steps": float(np.mean(message_ages)),
        },
        env,
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    summary: dict[str, dict[str, float | int | None]] = {}
    for scenario in sorted({str(row["scenario"]) for row in rows}):
        subset = [row for row in rows if row["scenario"] == scenario]
        times = [float(row["capture_time_seconds"]) for row in subset if row["capture_time_seconds"] is not None]
        summary[scenario] = {
            "episodes": len(subset),
            "safe_capture_rate": float(np.mean([bool(row["safe_capture_success"]) for row in subset])),
            "capture_rate": float(np.mean([bool(row["capture_event"]) for row in subset])),
            "collision_rate": float(np.mean([bool(row["collision"]) for row in subset])),
            "world_violation_rate": float(np.mean([int(row["world_violation_steps"]) > 0 for row in subset])),
            "mean_capture_time_seconds": float(np.mean(times)) if times else None,
            "mean_min_clearance_m": float(np.mean([float(row["min_clearance_m"]) for row in subset])),
            "worst_min_clearance_m": float(min(float(row["min_clearance_m"]) for row in subset)),
            "mean_visible_fraction": float(np.mean([float(row["mean_visible_fraction"]) for row in subset])),
            "mean_message_age_steps": float(np.mean([float(row["mean_message_age_steps"]) for row in subset])),
        }
    combined = list(rows)
    times = [float(row["capture_time_seconds"]) for row in combined if row["capture_time_seconds"] is not None]
    summary["overall"] = {
        "episodes": len(combined),
        "safe_capture_rate": float(np.mean([bool(row["safe_capture_success"]) for row in combined])),
        "capture_rate": float(np.mean([bool(row["capture_event"]) for row in combined])),
        "collision_rate": float(np.mean([bool(row["collision"]) for row in combined])),
        "world_violation_rate": float(np.mean([int(row["world_violation_steps"]) > 0 for row in combined])),
        "mean_capture_time_seconds": float(np.mean(times)) if times else None,
        "mean_min_clearance_m": float(np.mean([float(row["min_clearance_m"]) for row in combined])),
        "worst_min_clearance_m": float(min(float(row["min_clearance_m"]) for row in combined)),
        "mean_visible_fraction": float(np.mean([float(row["mean_visible_fraction"]) for row in combined])),
        "mean_message_age_steps": float(np.mean([float(row["mean_message_age_steps"]) for row in combined])),
    }
    return summary


def save_trajectory(env: CaptureRadiusPursuit3DEnv, path: Path) -> None:
    if not env.history:
        raise ValueError("Cannot export an empty pursuit trajectory.")
    np.savez_compressed(
        path,
        defender_positions=np.asarray([frame["defender_positions"] for frame in env.history], dtype=np.float64),
        target_positions=np.asarray([frame["target_position"] for frame in env.history], dtype=np.float64),
        obstacle_centers_xy=np.asarray([item.center_xy for item in env.obstacles], dtype=np.float64),
        obstacle_radii=np.asarray([item.radius for item in env.obstacles], dtype=np.float64),
        obstacle_heights=np.asarray([item.height for item in env.obstacles], dtype=np.float64),
        capture_radius=float(env.pursuit["capture_radius"]),
        capturing_defender_id=(-1 if env.capturing_defender_id is None else int(env.capturing_defender_id)),
        world_half_extent=float(env.world["half_extent_xy"]),
        world_height=float(env.world["height"]),
    )


def write_artifacts(
    output: Path,
    config: dict[str, Any],
    checkpoint: Path,
    args: argparse.Namespace,
    prediction_checkpoint: Path | None = None,
    actor_recurrent: bool = False,
) -> None:
    output.joinpath("config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output.joinpath("evaluation_protocol.json").write_text(
        json.dumps(
            {
                "task": "partial_observable_3d_capture_radius_pursuit",
                "success_definition": "Any defender enters r_capture before timeout without a safety failure.",
                "actor_information": "decentralized local observations only; centralized state is not queried at rollout.",
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "seed": args.seed,
                "episodes_per_scenario": args.episodes,
                "use_local_cbf_execution_filter": bool(args.use_cbf),
                "actor_recurrent": bool(actor_recurrent),
                "prediction_checkpoint": str(prediction_checkpoint.resolve()) if prediction_checkpoint is not None else None,
                "prediction_checkpoint_sha256": (
                    hashlib.sha256(prediction_checkpoint.read_bytes()).hexdigest()
                    if prediction_checkpoint is not None
                    else None
                ),
                "prediction_history_length": args.prediction_history_length if prediction_checkpoint is not None else None,
                "prediction_horizon_index": args.prediction_horizon_index if prediction_checkpoint is not None else None,
            },
            indent=2,
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
                f"device={args.device}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source_paths = [
        PROJECT_ROOT / "scripts" / "evaluate_capture_radius_mappo.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "learning.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "prediction.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_env.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_controllers.py",
        PROJECT_ROOT / "scripts" / "render_capture_radius_trajectory.py",
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


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.prediction_history_length <= 0 or args.prediction_horizon_index < 0:
        raise ValueError("Prediction history length must be positive and horizon index must be non-negative.")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    prediction_checkpoint = args.prediction_checkpoint.resolve() if args.prediction_checkpoint is not None else None
    if prediction_checkpoint is not None and not prediction_checkpoint.is_file():
        raise FileNotFoundError(f"Prediction checkpoint does not exist: {prediction_checkpoint}")
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    prediction_model = load_prediction_model(prediction_checkpoint, device) if prediction_checkpoint is not None else None
    first_experiment = config["experiments"][0]
    prototype = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=int(first_experiment["obstacle_count"]),
        target_speed_scale=float(first_experiment["target_speed_scale"]),
    )
    policy, action_scale, checkpoint_metadata = load_policy(
        checkpoint,
        prototype,
        prototype.reset(seed=args.seed),
        device,
    )
    write_artifacts(
        output,
        config,
        checkpoint,
        args,
        prediction_checkpoint=prediction_checkpoint,
        actor_recurrent=bool(checkpoint_metadata.get("actor_recurrent", False)),
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    renderer_prefix = [os.environ["CAPTURE_RADIUS_RENDER_PYTHON"]] if os.environ.get("CAPTURE_RADIUS_RENDER_PYTHON") else ["py", "-3.13"]
    for scenario_index, experiment in enumerate(config["experiments"]):
        for episode_index in range(args.episodes):
            seed = args.seed + scenario_index * 100_000 + episode_index
            row, env = rollout_episode(
                policy,
                config,
                obstacle_count=int(experiment["obstacle_count"]),
                target_speed_scale=float(experiment["target_speed_scale"]),
                seed=seed,
                device=device,
                action_scale=action_scale,
                use_cbf=bool(args.use_cbf),
                record_history=episode_index == 0,
                prediction_model=prediction_model,
                prediction_history_length=args.prediction_history_length,
                prediction_horizon_index=args.prediction_horizon_index,
            )
            row["scenario"] = str(experiment["name"])
            rows.append(row)
            if episode_index == 0:
                trajectory_path = output / f"trajectory_{experiment['name']}.npz"
                save_trajectory(env, trajectory_path)
                subprocess.run(
                    [
                        *renderer_prefix,
                        str(PROJECT_ROOT / "scripts" / "render_capture_radius_trajectory.py"),
                        "--trajectory",
                        str(trajectory_path),
                        "--output",
                        str(output / f"trajectory_{experiment['name']}.png"),
                        "--title",
                        f"{experiment['name']} MAPPO seed {seed}",
                    ],
                    check=True,
                )
    with output.joinpath("episodes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    output.joinpath("summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output.joinpath("run_metadata.json").write_text(
        json.dumps(
            {
                "algorithm": checkpoint_metadata.get("algorithm"),
                "checkpoint_seed": checkpoint_metadata.get("seed"),
                "device": str(device),
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
