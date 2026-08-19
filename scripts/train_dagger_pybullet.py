"""DAgger fine-tuning of a shared policy under PyBullet dynamics.

The student controls the physical simulator. At every visited state the
configured rule expert supplies a label, so states reached after student
mistakes are added to the supervised set. This addresses covariate shift in
plain behavior cloning without changing the frozen simulator or task rules.
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.controllers import HoldAwareTetrahedralSlotController, TetrahedralSlotController
from encirclement3d.learning import SharedActorCritic, defender_observations
from encirclement3d.pybullet_env import PYBULLET_DRONES_ROOT
from encirclement3d.residual import compute_policy_residual
from encirclement3d.safety import PyBulletResponseCBFSafetyFilter
from train_behavior_cloning import (
    _deep_update,
    device_metadata,
    make_environment,
    scenario_by_name,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def load_dagger_config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    document = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    environment_path = Path(document["environment_config"])
    if not environment_path.is_absolute():
        environment_path = args.config.parent / environment_path
    environment = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    settings = dict(document["training"])
    if args.device is not None:
        settings["device"] = args.device
    if args.seed is not None:
        settings["seed"] = args.seed
    required = {
        "seed",
        "device",
        "initial_checkpoint",
        "training_scenario",
        "evaluation_scenario",
        "obstacle_feature_count",
        "dagger_iterations",
        "episodes_per_iteration",
        "epochs_per_iteration",
        "minibatch_size",
        "learning_rate",
        "evaluation_episodes",
        "torch_num_threads",
        "deterministic_algorithms",
        "include_agent_id",
    }
    missing = sorted(required.difference(settings))
    if missing:
        raise ValueError(f"Missing training settings: {', '.join(missing)}")
    _deep_update(environment, document.get("environment_overrides", {}))
    settings["policy_safety_filter"] = dict(document.get("policy_safety_filter", {}))
    settings["policy_residual"] = dict(document.get("policy_residual", {}))
    label_mode = str(settings["policy_residual"].get("label_mode", "deployment_inverse"))
    if label_mode not in {"deployment_inverse", "expert_safe"}:
        raise ValueError("policy_residual.label_mode must be 'deployment_inverse' or 'expert_safe'.")
    settings["expert_controller"] = str(document.get("expert_controller", "rule"))
    if settings["expert_controller"] not in {"rule", "hold_aware"}:
        raise ValueError("expert_controller must be 'rule' or 'hold_aware'.")
    environment.setdefault("dynamics", {})["backend"] = str(
        settings.get("environment_backend", environment["dynamics"].get("backend", "pybullet"))
    )
    if environment["dynamics"]["backend"] != "pybullet":
        raise ValueError("DAgger PyBullet training requires dynamics.backend=pybullet.")
    if int(settings["dagger_iterations"]) <= 0 or int(settings["episodes_per_iteration"]) <= 0:
        raise ValueError("dagger_iterations and episodes_per_iteration must be positive.")
    if int(settings["epochs_per_iteration"]) <= 0 or int(settings["evaluation_episodes"]) <= 0:
        raise ValueError("epochs_per_iteration and evaluation_episodes must be positive.")
    if int(settings["obstacle_feature_count"]) <= 0:
        raise ValueError("obstacle_feature_count must be positive for obstacle DAgger training.")
    evaluation_seeds = settings.get("evaluation_seeds")
    if evaluation_seeds is None:
        evaluation_seeds = [int(settings["evaluation_seed"])] if settings.get("evaluation_seed") is not None else []
    if not evaluation_seeds:
        raise ValueError("Set evaluation_seed or a non-empty evaluation_seeds list for DAgger checkpoint selection.")
    settings["evaluation_seeds"] = [int(value) for value in evaluation_seeds]
    return document, environment, settings


def make_policy_safety_filter(settings: dict[str, Any], env: Any) -> PyBulletResponseCBFSafetyFilter | None:
    """Construct the deployed response-aware shield for DAgger rollouts.

    Safety filtering must be identical when collecting student states,
    generating expert labels, and evaluating a checkpoint. Otherwise DAgger
    learns an action distribution that is not the one seen at deployment.
    """
    document = dict(settings.get("policy_safety_filter", {}))
    if not bool(document.get("enabled", False)):
        return None
    filter_type = str(document.get("type", "pybullet_response_cbf"))
    if filter_type != "pybullet_response_cbf":
        raise ValueError("DAgger supports only policy_safety_filter.type=pybullet_response_cbf.")
    return PyBulletResponseCBFSafetyFilter(env)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_artifacts(
    output: Path,
    training_document: dict[str, Any],
    environment: dict[str, Any],
    settings: dict[str, Any],
    checkpoint: Path,
    device: torch.device,
) -> None:
    output.joinpath("config.yaml").write_text(
        yaml.safe_dump(
            {
                "training_document": training_document,
                "effective_training": settings,
                "environment": environment,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
    output.joinpath("environment.txt").write_text(
        "\n".join(
            [
                f"python={sys.version.replace(chr(10), ' ')}",
                f"platform={platform.platform()}",
                f"numpy={version('numpy')}",
                f"torch={version('torch')}",
                f"tensorboard={version('tensorboard')}",
                f"device={device}",
                *[f"{key}={value}" for key, value in device_metadata(device).items()],
                "",
                "pip_freeze:",
                freeze.rstrip(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source_paths = [
        PROJECT_ROOT / "scripts" / "train_dagger_pybullet.py",
        PROJECT_ROOT / "scripts" / "train_behavior_cloning.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "learning.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "controllers.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "environment.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pybullet_env.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "residual.py",
        PROJECT_ROOT / "third_party" / "gym-pybullet-drones-7688e7208a1572b1680736a3c0c9b93c379db3fe" / "gym_pybullet_drones" / "envs" / "BaseAviary.py",
        PROJECT_ROOT / "third_party" / "gym-pybullet-drones-7688e7208a1572b1680736a3c0c9b93c379db3fe" / "gym_pybullet_drones" / "envs" / "CtrlAviary.py",
        PROJECT_ROOT / "third_party" / "gym-pybullet-drones-7688e7208a1572b1680736a3c0c9b93c379db3fe" / "gym_pybullet_drones" / "control" / "DSLPIDControl.py",
        checkpoint,
    ]
    hashes = {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(path)
        for path in source_paths
    }
    output.joinpath("source_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")


def encode(
    observation: dict[str, Any],
    env: Any,
    include_agent_id: bool,
    obstacle_feature_count: int,
) -> np.ndarray:
    return defender_observations(
        observation,
        env.n_defenders,
        position_scale=float(env.world["half_extent_xy"]),
        defender_speed_scale=float(env.agents["defender_max_speed"]),
        target_speed_scale=float(env.agents["target_max_speed"]),
        include_agent_id=include_agent_id,
        obstacle_feature_count=obstacle_feature_count,
    )


def expert_target(action: np.ndarray, action_scale: float) -> np.ndarray:
    clipped = np.clip(np.asarray(action, dtype=np.float64), -action_scale, action_scale)
    return np.arctanh(np.clip(clipped / action_scale, -0.999999, 0.999999)).astype(np.float32)


def make_expert_controller(env: Any, settings: dict[str, Any]) -> Any:
    controller_name = str(settings.get("expert_controller", "rule"))
    if controller_name == "rule":
        return TetrahedralSlotController(env)
    if controller_name == "hold_aware":
        return HoldAwareTetrahedralSlotController(env)
    raise ValueError(f"Unsupported expert_controller={controller_name!r}.")


def student_action(
    policy: SharedActorCritic,
    observation: dict[str, Any],
    env: Any,
    include_agent_id: bool,
    obstacle_feature_count: int,
    action_scale: float,
    device: torch.device,
) -> np.ndarray:
    local = encode(observation, env, include_agent_id, obstacle_feature_count)
    with torch.no_grad():
        distribution, _value = policy.distribution_and_value(torch.as_tensor(local, device=device))
        return (torch.tanh(distribution.mean) * action_scale).cpu().numpy()


def apply_deployment_residual(
    action: np.ndarray,
    observation: dict[str, Any],
    env: Any,
    settings: dict[str, Any],
):
    """Apply the exact residual stage used between policy and CBF."""

    return compute_policy_residual(action, observation, env, dict(settings.get("policy_residual", {})))


def expert_policy_target(
    expert_action: np.ndarray,
    observation: dict[str, Any],
    env: Any,
    safety_filter: PyBulletResponseCBFSafetyFilter | None,
    settings: dict[str, Any],
) -> np.ndarray:
    """Return the raw policy label for the residual-then-CBF action chain.

    The expert is first passed through the deployment residual and safety
    stages. Its resulting safe action is then de-residualized so that adding
    the same residual at policy execution reconstructs the safe label.
    """

    residual = apply_deployment_residual(expert_action, observation, env, settings)
    label_mode = str(settings.get("policy_residual", {}).get("label_mode", "deployment_inverse"))
    safe_action = residual.action if label_mode == "deployment_inverse" else np.asarray(expert_action, dtype=np.float64)
    if safety_filter is not None:
        safe_action, _diagnostics = safety_filter.filter(safe_action, observation)
    raw_label = safe_action - residual.residual if label_mode == "deployment_inverse" else safe_action
    return env._clip_rows(raw_label, float(env.agents["defender_max_speed"]))


def collect_dagger_rollouts(
    policy: SharedActorCritic,
    config: dict[str, Any],
    scenario_name: str,
    episodes: int,
    seed: int,
    include_agent_id: bool,
    obstacle_feature_count: int,
    action_scale: float,
    device: torch.device,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    scenario = scenario_by_name(config, scenario_name)
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    rows: list[dict[str, float | int | bool]] = []
    student_safety_corrections: list[float] = []
    student_solver_nonconvergence_steps = 0
    student_safety_repair_steps = 0
    residual_norms: list[float] = []
    residual_scales: list[float] = []
    residual_active_steps = 0
    policy.eval()
    for episode in range(episodes):
        env = make_environment(
            config,
            obstacle_count=int(scenario["obstacle_count"]),
            target_speed_scale=float(scenario["target_speed_scale"]),
        )
        try:
            observation = env.reset(seed=seed + episode, record_history=False)
            expert = make_expert_controller(env, settings)
            safety_filter = make_policy_safety_filter(settings, env)
            while True:
                local = encode(observation, env, include_agent_id, obstacle_feature_count)
                expert_action = expert.act(observation)
                expert_action = expert_policy_target(expert_action, observation, env, safety_filter, settings)
                observations.append(local)
                targets.append(expert_target(expert_action, action_scale))
                raw_action = student_action(
                    policy,
                    observation,
                    env,
                    include_agent_id,
                    obstacle_feature_count,
                    action_scale,
                    device,
                )
                residual = apply_deployment_residual(raw_action, observation, env, settings)
                action = residual.action
                residual_norms.append(residual.norm)
                residual_scales.append(residual.mean_clearance_scale)
                residual_active_steps += int(residual.active)
                if safety_filter is not None:
                    action, diagnostics = safety_filter.filter(action, observation)
                    student_safety_corrections.append(float(diagnostics.action_correction_norm))
                    student_solver_nonconvergence_steps += int(not diagnostics.solver_success)
                    student_safety_repair_steps += int(diagnostics.used_fallback)
                observation, _reward, terminated, truncated, info = env.step(action, record_history=False)
                if terminated or truncated:
                    rows.append(
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
            env.close()
    return (
        np.concatenate(observations, axis=0).astype(np.float32),
        np.concatenate(targets, axis=0).astype(np.float32),
        {
            "episodes": rows,
            "episode_success_rate": float(np.mean([row["success"] for row in rows])),
            "episode_collision_rate": float(np.mean([row["collision"] for row in rows])),
            "samples": int(sum(item.shape[0] for item in observations)),
            "mean_student_policy_safety_filter_correction": float(np.mean(student_safety_corrections))
            if student_safety_corrections
            else 0.0,
            "student_policy_safety_solver_nonconvergence_steps": student_solver_nonconvergence_steps,
            "student_policy_safety_repair_steps": student_safety_repair_steps,
            "mean_student_policy_residual_norm": float(np.mean(residual_norms)) if residual_norms else 0.0,
            "mean_student_policy_residual_clearance_scale": float(np.mean(residual_scales))
            if residual_scales
            else 1.0,
            "student_policy_residual_active_steps": residual_active_steps,
        },
    )


def evaluate(
    policy: SharedActorCritic,
    config: dict[str, Any],
    scenario_name: str,
    episodes: int,
    seed: int,
    include_agent_id: bool,
    obstacle_feature_count: int,
    action_scale: float,
    device: torch.device,
    settings: dict[str, Any],
) -> dict[str, float]:
    scenario = scenario_by_name(config, scenario_name)
    outcomes: list[dict[str, float | bool]] = []
    safety_corrections: list[float] = []
    solver_nonconvergence_steps = 0
    repair_steps = 0
    residual_norms: list[float] = []
    residual_scales: list[float] = []
    residual_clearances: list[float] = []
    residual_active_steps = 0
    hold_run = 0
    max_hold_runs: list[int] = []
    constraint_minima: list[float] = []
    policy.eval()
    for episode in range(episodes):
        env = make_environment(
            config,
            obstacle_count=int(scenario["obstacle_count"]),
            target_speed_scale=float(scenario["target_speed_scale"]),
        )
        try:
            observation = env.reset(seed=seed + episode, record_history=False)
            safety_filter = make_policy_safety_filter(settings, env)
            while True:
                raw_action = student_action(
                    policy,
                    observation,
                    env,
                    include_agent_id,
                    obstacle_feature_count,
                    action_scale,
                    device,
                )
                residual = apply_deployment_residual(raw_action, observation, env, settings)
                action = residual.action
                residual_norms.append(residual.norm)
                residual_scales.append(residual.mean_clearance_scale)
                residual_clearances.append(residual.minimum_clearance)
                residual_active_steps += int(residual.active)
                if safety_filter is not None:
                    action, diagnostics = safety_filter.filter(action, observation)
                    safety_corrections.append(float(diagnostics.action_correction_norm))
                    constraint_minima.append(float(diagnostics.minimum_constraint_value))
                    solver_nonconvergence_steps += int(not diagnostics.solver_success)
                    repair_steps += int(diagnostics.used_fallback)
                observation, _reward, terminated, truncated, info = env.step(action, record_history=False)
                covered = bool(
                    np.all(np.asarray(info["slot_error"], dtype=np.float64) <= float(env.task["slot_tolerance"]))
                )
                hold_run = hold_run + 1 if covered else 0
                if terminated or truncated:
                    max_hold_runs.append(max(hold_run, int(info.get("hold_steps", 0))))
                    outcomes.append(
                        {
                            "success": bool(info["success"]),
                            "collision": bool(info["collision_steps"]),
                            "physical_collision": bool(info.get("physical_collision_steps", 0)),
                            "world_violation": bool(info.get("world_violation_steps", 0)),
                            "slot_error": float(info["mean_slot_error"]),
                            "steps": float(env.step_count),
                            "min_clearance": float(info["min_clearance_so_far"]),
                            "final_hold_steps": float(info.get("hold_steps", 0)),
                        }
                    )
                    hold_run = 0
                    break
        finally:
            env.close()
    return {
        "success_rate": float(np.mean([bool(item["success"]) for item in outcomes])),
        "collision_rate": float(np.mean([bool(item["collision"]) for item in outcomes])),
        "physical_collision_rate": float(np.mean([bool(item["physical_collision"]) for item in outcomes])),
        "world_violation_rate": float(np.mean([bool(item["world_violation"]) for item in outcomes])),
        "mean_final_slot_error": float(np.mean([float(item["slot_error"]) for item in outcomes])),
        "mean_steps": float(np.mean([float(item["steps"]) for item in outcomes])),
        "mean_min_clearance": float(np.mean([float(item["min_clearance"]) for item in outcomes])),
        "worst_min_clearance": float(np.min([float(item["min_clearance"]) for item in outcomes])),
        "mean_final_hold_steps": float(np.mean([float(item["final_hold_steps"]) for item in outcomes])),
        "mean_max_hold_run": float(np.mean(max_hold_runs)) if max_hold_runs else 0.0,
        "mean_policy_safety_filter_correction": float(np.mean(safety_corrections)) if safety_corrections else 0.0,
        "worst_policy_safety_constraint": float(np.min(constraint_minima)) if constraint_minima else float("inf"),
        "policy_safety_solver_nonconvergence_steps": float(solver_nonconvergence_steps),
        "policy_safety_repair_steps": float(repair_steps),
        "mean_policy_residual_norm": float(np.mean(residual_norms)) if residual_norms else 0.0,
        "mean_policy_residual_clearance_scale": float(np.mean(residual_scales)) if residual_scales else 1.0,
        "worst_policy_residual_min_obstacle_clearance": float(np.min(residual_clearances))
        if residual_clearances
        else float("inf"),
        "policy_residual_active_steps": float(residual_active_steps),
    }


def aggregate_evaluations(evaluations: list[dict[str, float]]) -> dict[str, float]:
    """Combine equal-sized development-suite evaluations conservatively.

    Means are averaged across suites; safety minima use the worst suite and
    solver/repair counts are summed. This prevents checkpoint selection from
    optimizing one deterministic seed while hiding a failure on another.
    """

    if not evaluations:
        raise ValueError("At least one development evaluation is required.")
    result: dict[str, float] = {}
    sum_keys = {"policy_safety_solver_nonconvergence_steps", "policy_safety_repair_steps"}
    min_keys = {
        "worst_min_clearance",
        "worst_policy_safety_constraint",
        "worst_policy_residual_min_obstacle_clearance",
    }
    for key in evaluations[0]:
        values = [float(item[key]) for item in evaluations]
        if key in sum_keys:
            result[key] = float(sum(values))
        elif key in min_keys:
            result[key] = float(min(values))
        else:
            result[key] = float(np.mean(values))
    return result


def selection_key(record: dict[str, float]) -> tuple[float, float, float, float, float, float, float]:
    """Order checkpoints by task success first, then safety and convergence."""

    return (
        float(record["success_rate"]),
        -float(record["physical_collision_rate"]),
        -float(record["collision_rate"]),
        -float(record["policy_safety_repair_steps"]),
        float(record["worst_min_clearance"]),
        float(record["mean_max_hold_run"]),
        -float(record["mean_final_slot_error"]),
    )


def checkpoint_payload(
    state_dict: dict[str, torch.Tensor],
    loaded: dict[str, Any],
    settings: dict[str, Any],
    action_scale: float,
    include_agent_id: bool,
    obstacle_feature_count: int,
    scenario_name: str,
    evaluation_scenario: str,
    initial_checkpoint: Path,
) -> dict[str, Any]:
    return {
        "state_dict": state_dict,
        "observation_dim": int(loaded["observation_dim"]),
        "hidden_dim": int(loaded.get("hidden_dim", 128)),
        "action_scale": action_scale,
        "include_agent_id": include_agent_id,
        "obstacle_feature_count": obstacle_feature_count,
        "training_scenario": scenario_name,
        "evaluation_scenario": evaluation_scenario,
        "algorithm": "dagger_pybullet_rule_expert",
        "policy_residual": dict(settings.get("policy_residual", {})),
        "initial_checkpoint": str(initial_checkpoint),
        "seed": int(settings["seed"]),
        "device": str(settings["device"]),
    }


def train(args: argparse.Namespace) -> None:
    training_document, config, settings = load_dagger_config(args)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output}")
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

    scenario_name = str(settings.get("training_scenario", "moderate"))
    evaluation_scenario = str(settings.get("evaluation_scenario", scenario_name))
    include_agent_id = bool(settings.get("include_agent_id", False))
    obstacle_feature_count = int(settings.get("obstacle_feature_count", 3))
    initial_checkpoint = Path(settings["initial_checkpoint"])
    if not initial_checkpoint.is_absolute():
        initial_checkpoint = (args.config.parent / initial_checkpoint).resolve()
    if not initial_checkpoint.is_file():
        raise FileNotFoundError(f"Initial checkpoint not found: {initial_checkpoint}")
    loaded = torch.load(initial_checkpoint, map_location=device, weights_only=False)
    policy = SharedActorCritic(
        int(loaded["observation_dim"]),
        hidden_dim=int(loaded.get("hidden_dim", 128)),
    ).to(device)
    if int(loaded.get("obstacle_feature_count", obstacle_feature_count)) != obstacle_feature_count:
        raise ValueError("Initial checkpoint obstacle_feature_count does not match DAgger configuration.")
    policy.load_state_dict(loaded["state_dict"])
    policy.eval()
    action_scale = float(loaded["action_scale"])
    write_artifacts(args.output, training_document, config, settings, initial_checkpoint, device)
    writer = SummaryWriter(log_dir=str(args.output / "tensorboard"), flush_secs=10)
    writer.add_text("Config/effective_training", f"```yaml\n{yaml.safe_dump(settings, sort_keys=False)}```", 0)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(settings["learning_rate"]), eps=1e-5)
    aggregate_observations: list[np.ndarray] = []
    aggregate_targets: list[np.ndarray] = []
    history: list[dict[str, float | int]] = []
    selection_enabled = bool(settings.get("select_best_checkpoint", False))
    evaluation_seeds = [int(value) for value in settings["evaluation_seeds"]]
    save_iteration_checkpoints = bool(settings.get("save_iteration_checkpoints", False))
    best_key: tuple[float, float, float, float] | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_selection: dict[str, float | int] | None = None
    started = time.perf_counter()
    try:
        if selection_enabled and bool(settings.get("evaluate_initial_checkpoint", False)):
            initial_runs = [
                evaluate(
                    policy,
                    config,
                    evaluation_scenario,
                    int(settings["evaluation_episodes"]),
                    evaluation_seed,
                    include_agent_id,
                    obstacle_feature_count,
                    action_scale,
                    device,
                    settings,
                )
                for evaluation_seed in evaluation_seeds
            ]
            initial_evaluation = aggregate_evaluations(initial_runs)
            best_key = selection_key(initial_evaluation)
            best_state_dict = {
                name: parameter.detach().cpu().clone()
                for name, parameter in policy.state_dict().items()
            }
            best_selection = {
                "iteration": -1,
                "source": "initial_checkpoint",
                "evaluation_seeds": evaluation_seeds,
                "evaluation_runs": initial_runs,
                **initial_evaluation,
            }
            writer.add_scalar("Selection/best_iteration", -1, -1)
        for iteration in range(int(settings["dagger_iterations"])):
            iteration_observations, iteration_targets, rollout_summary = collect_dagger_rollouts(
                policy,
                config,
                scenario_name,
                int(settings["episodes_per_iteration"]),
                seed + iteration * 10_000,
                include_agent_id,
                obstacle_feature_count,
                action_scale,
                device,
                settings,
            )
            aggregate_observations.append(iteration_observations)
            aggregate_targets.append(iteration_targets)
            observations_tensor = torch.as_tensor(np.concatenate(aggregate_observations), device=device)
            targets_tensor = torch.as_tensor(np.concatenate(aggregate_targets), device=device)
            sample_count = int(observations_tensor.shape[0])
            policy.train()
            losses: list[float] = []
            for _epoch in range(int(settings["epochs_per_iteration"])):
                permutation = torch.randperm(sample_count, device=device)
                for start in range(0, sample_count, int(settings["minibatch_size"])):
                    indices = permutation[start : start + int(settings["minibatch_size"])]
                    distribution, _value = policy.distribution_and_value(observations_tensor[indices])
                    loss = torch.nn.functional.mse_loss(distribution.loc, targets_tensor[indices])
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))
            evaluation_runs = [
                evaluate(
                    policy,
                    config,
                    evaluation_scenario,
                    int(settings["evaluation_episodes"]),
                    evaluation_seed,
                    include_agent_id,
                    obstacle_feature_count,
                    action_scale,
                    device,
                    settings,
                )
                for evaluation_seed in evaluation_seeds
            ]
            evaluation = aggregate_evaluations(evaluation_runs)
            record = {
                "iteration": iteration,
                "samples": sample_count,
                "evaluation_seed_count": len(evaluation_runs),
                "dagger_expert_success_rate": float(rollout_summary["episode_success_rate"]),
                "dagger_student_collision_rate": float(rollout_summary["episode_collision_rate"]),
                "expert_mse": float(np.mean(losses)),
                **evaluation,
            }
            history.append(record)
            candidate_key = selection_key(record)
            if selection_enabled and (best_key is None or candidate_key > best_key):
                best_key = candidate_key
                best_state_dict = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in policy.state_dict().items()
                }
                best_selection = {
                    "iteration": iteration,
                    "evaluation_seeds": evaluation_seeds,
                    "evaluation_runs": evaluation_runs,
                    **evaluation,
                }
                writer.add_scalar("Selection/best_iteration", iteration, iteration)
            writer.add_scalar("DAgger/expert_success_rate", record["dagger_expert_success_rate"], iteration)
            writer.add_scalar("DAgger/student_collision_rate", record["dagger_student_collision_rate"], iteration)
            writer.add_scalar(
                "DAgger/student_policy_safety_filter_correction",
                float(rollout_summary["mean_student_policy_safety_filter_correction"]),
                iteration,
            )
            writer.add_scalar(
                "DAgger/student_policy_safety_solver_nonconvergence_steps",
                float(rollout_summary["student_policy_safety_solver_nonconvergence_steps"]),
                iteration,
            )
            writer.add_scalar(
                "DAgger/student_policy_safety_repair_steps",
                float(rollout_summary["student_policy_safety_repair_steps"]),
                iteration,
            )
            writer.add_scalar(
                "DAgger/student_policy_residual_norm",
                float(rollout_summary["mean_student_policy_residual_norm"]),
                iteration,
            )
            writer.add_scalar(
                "DAgger/student_policy_residual_clearance_scale",
                float(rollout_summary["mean_student_policy_residual_clearance_scale"]),
                iteration,
            )
            writer.add_scalar(
                "DAgger/student_policy_residual_active_steps",
                float(rollout_summary["student_policy_residual_active_steps"]),
                iteration,
            )
            writer.add_scalar("Loss/expert_mse", record["expert_mse"], iteration)
            for name in (
                "success_rate",
                "collision_rate",
                "physical_collision_rate",
                "world_violation_rate",
                "mean_final_slot_error",
                "mean_steps",
                "mean_min_clearance",
                "worst_min_clearance",
                "mean_final_hold_steps",
                "mean_max_hold_run",
                "mean_policy_safety_filter_correction",
                "worst_policy_safety_constraint",
                "policy_safety_solver_nonconvergence_steps",
                "policy_safety_repair_steps",
                "mean_policy_residual_norm",
                "mean_policy_residual_clearance_scale",
                "worst_policy_residual_min_obstacle_clearance",
                "policy_residual_active_steps",
            ):
                writer.add_scalar(f"Evaluation/{name}", record[name], iteration)
            writer.flush()
            args.output.joinpath(f"rollout_iteration{iteration:02d}.json").write_text(
                json.dumps(rollout_summary, indent=2), encoding="utf-8"
            )
            args.output.joinpath(f"evaluation_iteration{iteration:02d}.json").write_text(
                json.dumps(
                    {
                        "evaluation_seeds": evaluation_seeds,
                        "evaluation_episodes_per_seed": int(settings["evaluation_episodes"]),
                        "aggregate": evaluation,
                        "per_seed": evaluation_runs,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if save_iteration_checkpoints:
                torch.save(
                    checkpoint_payload(
                        policy.state_dict(),
                        loaded,
                        settings,
                        action_scale,
                        include_agent_id,
                        obstacle_feature_count,
                        scenario_name,
                        evaluation_scenario,
                        initial_checkpoint,
                    ),
                    args.output / f"checkpoint_iteration{iteration:02d}.pt",
                )
    finally:
        writer.close()
    with (args.output / "training.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(history[0].keys())
        csv_writer = csv.DictWriter(handle, fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(history)
    last_payload = checkpoint_payload(
        policy.state_dict(),
        loaded,
        settings,
        action_scale,
        include_agent_id,
        obstacle_feature_count,
        scenario_name,
        evaluation_scenario,
        initial_checkpoint,
    )
    if selection_enabled and best_state_dict is not None:
        selected_payload = checkpoint_payload(
            best_state_dict,
            loaded,
            settings,
            action_scale,
            include_agent_id,
            obstacle_feature_count,
            scenario_name,
            evaluation_scenario,
            initial_checkpoint,
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
                "evaluation_seeds": evaluation_seeds,
                "selection_key": "success_rate, lower physical/collision/CBF-repair rates, higher worst clearance, higher hold run, lower slot error",
                "selected": best_selection,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    args.output.joinpath("run_metadata.json").write_text(
        json.dumps(
            {
                "algorithm": "dagger_pybullet_rule_expert",
                "initial_checkpoint": str(initial_checkpoint),
                "dagger_iterations": int(settings["dagger_iterations"]),
                "episodes_per_iteration": int(settings["episodes_per_iteration"]),
                "elapsed_seconds": time.perf_counter() - started,
                "device": str(device),
                "cuda": device_metadata(device),
                "checkpoint_selection": best_selection,
                "cross_domain_statement": "PyBullet dynamics training result; not a real-flight result.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    train(parse_args())
