"""Evaluate a frozen policy under the pinned PyBullet quadrotor dynamics.

The command is deliberately evaluation-only. It does not fine-tune, overwrite
the source checkpoint, or report cross-domain results as kinematic success.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.learning import SharedActorCritic, defender_observations
from encirclement3d.pybullet_env import PYBULLET_DRONES_ROOT, PyBulletEncirclement3DEnv
from encirclement3d.reporting import plot_trajectory
from encirclement3d.residual import apply_policy_residual as _shared_apply_policy_residual
from encirclement3d.safety import DiscreteTimeCBFSafetyFilter, PyBulletResponseCBFSafetyFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Override the policy checkpoint recorded in the YAML; intended for frozen-model replication sweeps.",
    )
    parser.add_argument("--target-speed-scale", type=float)
    parser.add_argument("--trace-csv", type=Path, help="Write per-control-step state/action trace; requires one episode.")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_document(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    document = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.checkpoint is not None:
        document["checkpoint"] = str(args.checkpoint.resolve())
    environment_path = Path(document["environment_config"])
    if not environment_path.is_absolute():
        environment_path = (args.config.parent / environment_path).resolve()
    environment = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    settings = dict(document["evaluation"])
    if args.device is not None:
        settings["device"] = args.device
    if args.episodes is not None:
        settings["episodes"] = args.episodes
    if args.seed is not None:
        settings["seed"] = args.seed
    if args.target_speed_scale is not None:
        if args.target_speed_scale <= 0.0:
            raise ValueError("target speed scale must be positive.")
        settings["target_speed_scale"] = args.target_speed_scale
    required = {"seed", "device", "episodes", "scenario", "record_trajectory", "deterministic_algorithms"}
    missing = sorted(required.difference(settings))
    if missing:
        raise ValueError(f"Missing evaluation settings: {', '.join(missing)}")
    if int(settings["episodes"]) <= 0:
        raise ValueError("episodes must be positive.")
    environment.setdefault("dynamics", {}).update(dict(document["pybullet"]))
    task_overrides = dict(document.get("task_overrides", {}))
    unknown_task_keys = sorted(set(task_overrides).difference(environment.get("task", {})))
    if unknown_task_keys:
        raise ValueError(f"Unknown task_overrides keys: {', '.join(unknown_task_keys)}")
    environment.setdefault("task", {}).update(task_overrides)
    environment["dynamics"]["backend"] = "pybullet"
    return document, environment, settings, environment_path


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


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def write_artifacts(
    output: Path,
    document: dict[str, Any],
    environment: dict[str, Any],
    settings: dict[str, Any],
    checkpoint: Path,
    device: torch.device,
) -> None:
    output.joinpath("config.yaml").write_text(
        yaml.safe_dump(
            {"evaluation_document": document, "effective_evaluation": settings, "environment": environment},
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
                f"numpy={package_version('numpy')}",
                f"torch={package_version('torch')}",
                f"tensorboard={package_version('tensorboard')}",
                f"gym={package_version('gym')}",
                f"pybullet_conda_package={package_version('pybullet')}",
                f"device={device}",
                *[f"{key}={value}" for key, value in device_metadata(device).items()],
                "",
                "pip_freeze:",
                freeze.rstrip(),
                "",
            ]
        ),
        encoding="utf-8",
    )
    source_paths = [
        PROJECT_ROOT / "scripts" / "evaluate_pybullet.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pybullet_env.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "environment.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "capture.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "flexible_net.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "learning.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "safety.py",
        PYBULLET_DRONES_ROOT / "gym_pybullet_drones" / "envs" / "BaseAviary.py",
        PYBULLET_DRONES_ROOT / "gym_pybullet_drones" / "envs" / "VelocityAviary.py",
        PYBULLET_DRONES_ROOT / "gym_pybullet_drones" / "envs" / "CtrlAviary.py",
        PYBULLET_DRONES_ROOT / "gym_pybullet_drones" / "control" / "DSLPIDControl.py",
        checkpoint,
    ]
    hashes = {str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(path) for path in source_paths}
    output.joinpath("source_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    output.joinpath("policy_reference.json").write_text(
        json.dumps({"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint)}, indent=2), encoding="utf-8"
    )


def policy_action(
    policy: SharedActorCritic,
    observation: dict[str, Any],
    env: PyBulletEncirclement3DEnv,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    local = defender_observations(
        observation,
        env.n_defenders,
        position_scale=float(env.world["half_extent_xy"]),
        defender_speed_scale=float(env.agents["defender_max_speed"]),
        target_speed_scale=float(env.agents["target_max_speed"]),
        include_agent_id=bool(checkpoint["include_agent_id"]),
        obstacle_feature_count=int(checkpoint.get("obstacle_feature_count", 0)),
    )
    with torch.no_grad():
        distribution, _value = policy.distribution_and_value(torch.as_tensor(local, device=device))
        return (torch.tanh(distribution.mean) * float(checkpoint["action_scale"])).cpu().numpy()


def apply_policy_residual(
    action: np.ndarray,
    observation: dict[str, Any],
    env: PyBulletEncirclement3DEnv,
    document: dict[str, Any],
) -> tuple[np.ndarray, float, float, float]:
    return _shared_apply_policy_residual(action, observation, env, document)


def clearance_components(env: PyBulletEncirclement3DEnv) -> dict[str, float]:
    """Break a trajectory's closest approach into auditable safety components."""
    radius = float(env.agents["drone_radius"])
    obstacle_clearances = [
        float(env._cylinder_clearance_and_normal(position, obstacle)[0] - radius)
        for position in env.defender_positions
        for obstacle in env.obstacles
    ]
    inter_agent_clearances = [
        float(np.linalg.norm(env.defender_positions[first] - env.defender_positions[second]) - 2.0 * radius)
        for first in range(env.n_defenders)
        for second in range(first + 1, env.n_defenders)
    ]
    target_clearances = [
        float(np.linalg.norm(position - env.target_position) - 2.0 * radius)
        for position in env.defender_positions
    ]
    return {
        "min_obstacle_clearance": min(obstacle_clearances, default=float("inf")),
        "min_inter_agent_clearance": min(inter_agent_clearances, default=float("inf")),
        "min_target_clearance": min(target_clearances, default=float("inf")),
        "min_boundary_clearance": float(env._boundary_clearance()),
    }


def run_episode(
    env: PyBulletEncirclement3DEnv,
    policy: SharedActorCritic,
    checkpoint: dict[str, Any],
    seed: int,
    record_history: bool,
    device: torch.device,
    policy_safety_filter_type: str | None,
    policy_residual_document: dict[str, Any],
    capture_closure_strategy: str,
    trace_rows: list[dict[str, float | int | bool]] | None = None,
) -> dict[str, Any]:
    observation = env.reset(seed=seed, record_history=record_history)
    info: dict[str, Any] = {}
    if policy_safety_filter_type == "kinematic_cbf":
        safety_filter = DiscreteTimeCBFSafetyFilter(env)
    elif policy_safety_filter_type == "pybullet_response_cbf":
        safety_filter = PyBulletResponseCBFSafetyFilter(env)
    elif policy_safety_filter_type is None:
        safety_filter = None
    else:
        raise ValueError(f"Unsupported policy safety filter type: {policy_safety_filter_type!r}")
    action_corrections: list[float] = []
    constraint_minima: list[float] = []
    solver_nonconvergence_steps = 0
    repair_steps = 0
    current_hold_run = 0
    max_hold_run = 0
    residual_norms: list[float] = []
    residual_clearances: list[float] = []
    residual_scales: list[float] = []
    while True:
        raw_action = policy_action(policy, observation, env, checkpoint, device)
        residual_action, residual_norm, residual_clearance, residual_scale = apply_policy_residual(
            raw_action, observation, env, policy_residual_document
        )
        residual_norms.append(float(residual_norm))
        residual_clearances.append(float(residual_clearance))
        residual_scales.append(float(residual_scale))
        action = residual_action
        if safety_filter is not None:
            action, diagnostics = safety_filter.filter(residual_action, observation)
            action_corrections.append(float(diagnostics.action_correction_norm))
            constraint_minima.append(float(diagnostics.minimum_constraint_value))
            solver_nonconvergence_steps += int(not diagnostics.solver_success)
            repair_steps += int(diagnostics.used_fallback)
        close_cage = capture_closure_strategy == "rule_guard" and env.capture_close_feasible()
        observation, _reward, terminated, truncated, info = env.step(
            action,
            record_history=record_history,
            close_cage=close_cage,
        )
        covered = bool(np.all(np.asarray(info["slot_error"], dtype=np.float64) <= float(env.task["slot_tolerance"])))
        current_hold_run = current_hold_run + 1 if covered else 0
        max_hold_run = max(max_hold_run, current_hold_run)
        if trace_rows is not None:
            clearance_trace = clearance_components(env)
            trace_row: dict[str, float | int | bool] = {
                "step": int(env.step_count),
                "world_violation": bool(info["world_violation"]),
                "physical_contact": bool(info["physical_contact"]),
                **clearance_trace,
                "mean_slot_error": float(info["mean_slot_error"]),
                "max_slot_error": float(np.max(np.asarray(info["slot_error"], dtype=np.float64))),
                "hold_steps": int(info["hold_steps"]),
                "current_hold_run": int(current_hold_run),
                "max_hold_run": int(max_hold_run),
                "encirclement_success": bool(info["encirclement_success"]),
                "capture_enabled": bool(info["capture_enabled"]),
                "capture_success": bool(info["capture_success"]),
                "capture_closed": bool(info["capture_closed"]),
                "capture_close_requested": bool(info["capture_close_requested"]),
                "capture_close_ready": bool(info["capture_close_ready"]),
                "capture_close_accepted": bool(info["capture_close_accepted"]),
                "capture_close_attempts": int(info["capture_close_attempts"]),
                "capture_closure_events": int(info["capture_closure_events"]),
                "capture_close_rejected_steps": int(info["capture_close_rejected_steps"]),
                "capture_escaped": bool(info["capture_escaped"]),
                "capture_structural_failure": bool(info.get("capture_structural_failure", False)),
                "capture_escape_event": bool(info["capture_escape_event"]),
                "capture_escape_events": int(info["capture_escape_events"]),
                "capture_compression_events": int(info.get("capture_compression_events", 0)),
                "capture_net_contact": bool(info.get("capture_net_contact", False)),
                "capture_net_contact_steps": int(info.get("capture_net_contact_steps", 0)),
                "capture_hold_steps": int(info["capture_hold_steps"]),
                "capture_time_seconds": (
                    float(info["capture_time_seconds"]) if info["capture_time_seconds"] is not None else float("nan")
                ),
                "capture_relative_speed_at_closure": (
                    float(info["capture_relative_speed_at_closure"])
                    if info["capture_relative_speed_at_closure"] is not None
                    else float("nan")
                ),
                "cage_min_face_clearance": float(info["cage_min_face_clearance"]),
                "cage_target_radius": float(info.get("cage_target_radius", 0.0)),
                "cage_net_margin": float(info.get("cage_net_margin", info["cage_min_face_clearance"])),
                "cage_sphere_contained": bool(info.get("cage_sphere_contained", True)),
                "cage_min_edge_length": float(info["cage_min_edge_length"]),
                "cage_max_edge_length": float(info["cage_max_edge_length"]),
                "capture_max_relative_speed": float(info["capture_max_relative_speed"]),
                "reward_encirclement": float(info["reward_components"]["encirclement"]),
                "reward_safety": float(info["reward_components"]["safety"]),
                "reward_capture_feasibility": float(info["reward_components"]["capture_feasibility"]),
                "reward_capture_closure": float(info["reward_components"]["capture_closure"]),
                "reward_capture_escape": float(info["reward_components"]["capture_escape"]),
                "reward_capture_structural_failure": float(
                    info["reward_components"].get("capture_structural_failure", 0.0)
                ),
                "reward_capture_success": float(info["reward_components"]["capture_success"]),
                "policy_residual_norm": float(residual_norm),
                "policy_residual_min_obstacle_clearance": float(residual_clearance),
                "policy_residual_mean_clearance_scale": float(residual_scale),
                "policy_safety_filter_enabled": safety_filter is not None,
                "policy_safety_filter_correction": action_corrections[-1] if action_corrections else 0.0,
                "policy_safety_filter_minimum_constraint": constraint_minima[-1] if constraint_minima else float("inf"),
            }
            for axis, name in enumerate(("x", "y", "z")):
                trace_row[f"target_pos_{name}"] = float(observation["target_position"][axis])
                trace_row[f"target_vel_{name}"] = float(observation["target_velocity"][axis])
            for index in range(env.n_defenders):
                trace_row[f"d{index}_slot_error"] = float(np.asarray(info["slot_error"], dtype=np.float64)[index])
                for axis, name in enumerate(("x", "y", "z")):
                    trace_row[f"d{index}_pos_{name}"] = float(env.defender_positions[index, axis])
                    trace_row[f"d{index}_vel_{name}"] = float(env.defender_velocities[index, axis])
                    trace_row[f"d{index}_raw_{name}"] = float(raw_action[index, axis])
                    trace_row[f"d{index}_cmd_{name}"] = float(action[index, axis])
                    trace_row[f"d{index}_exec_{name}"] = float(env.last_executed_defender_actions[index, axis])
                    trace_row[f"d{index}_ref_{name}"] = float(env.last_pid_target_positions[index, axis])
                if env.aviary is not None:
                    trace_row[f"d{index}_roll"] = float(env.aviary.rpy[index, 0])
                    trace_row[f"d{index}_pitch"] = float(env.aviary.rpy[index, 1])
                    trace_row[f"d{index}_yaw"] = float(env.aviary.rpy[index, 2])
                    trace_row[f"d{index}_ang_vel_x"] = float(env.aviary.ang_v[index, 0])
                    trace_row[f"d{index}_ang_vel_y"] = float(env.aviary.ang_v[index, 1])
                    trace_row[f"d{index}_ang_vel_z"] = float(env.aviary.ang_v[index, 2])
                    for motor in range(4):
                        trace_row[f"d{index}_rpm_{motor}"] = float(env.aviary.last_clipped_action[index, motor])
                trace_row[f"d{index}_vertical_recovery"] = bool(info["vertical_recovery_active_agents"][index])
                trace_row[f"d{index}_vertical_emergency"] = bool(info["vertical_emergency_active_agents"][index])
                trace_row[f"d{index}_vertical_emergency_required_distance"] = float(
                    info["vertical_emergency_required_distance"][index]
                )
                trace_row[f"d{index}_attitude_recovery"] = bool(info["attitude_recovery_active_agents"][index])
                trace_row[f"d{index}_attitude_tilt"] = float(info["attitude_tilt"][index])
            trace_rows.append(trace_row)
        if terminated or truncated:
            break
    return {
        "seed": seed,
        "success": bool(info["success"]),
        "encirclement_success": bool(info["encirclement_success"]),
        "capture_enabled": bool(info["capture_enabled"]),
        "capture_success": bool(info["capture_success"]),
        "capture_closed": bool(info["capture_closed"]),
        "capture_escaped": bool(info["capture_escaped"]),
        "capture_structural_failure": bool(info.get("capture_structural_failure", False)),
        "capture_close_attempts": int(info["capture_close_attempts"]),
        "capture_closure_events": int(info["capture_closure_events"]),
        "capture_close_rejected_steps": int(info["capture_close_rejected_steps"]),
        "capture_escape_events": int(info["capture_escape_events"]),
        "capture_compression_events": int(info.get("capture_compression_events", 0)),
        "capture_net_contact_steps": int(info.get("capture_net_contact_steps", 0)),
        "capture_peak_net_tension": float(info.get("capture_peak_net_tension", 0.0)),
        "capture_peak_net_strain": float(info.get("capture_peak_net_strain", 0.0)),
        "capture_peak_contact_impulse": float(info.get("capture_peak_contact_impulse", 0.0)),
        "capture_time_seconds": info["capture_time_seconds"],
        "capture_relative_speed_at_closure": info["capture_relative_speed_at_closure"],
        "capture_min_face_clearance_while_closed": info["capture_min_face_clearance_while_closed"],
        "capture_min_net_margin_while_closed": info.get("capture_min_net_margin_while_closed"),
        "steps": int(env.step_count),
        "simulated_seconds": float(env.step_count * env.dt),
        "collision_steps": int(info["collision_steps"]),
        "physical_collision_steps": int(info["physical_collision_steps"]),
        "world_violation_steps": int(info["world_violation_steps"]),
        "min_clearance": float(info["min_clearance_so_far"]),
        "mean_slot_error": float(info["mean_slot_error"]),
        "mean_command_speed": float(info["mean_command_speed"]),
        "mean_filtered_command_speed": float(info["mean_filtered_command_speed"]),
        "mean_command_filter_correction": float(info["mean_command_filter_correction"]),
        "mean_realized_speed": float(info["mean_realized_speed"]),
        "mean_policy_residual_norm": float(np.mean(residual_norms)) if residual_norms else 0.0,
        "mean_policy_residual_min_obstacle_clearance": float(np.mean(residual_clearances))
        if residual_clearances
        else float("inf"),
        "worst_policy_residual_min_obstacle_clearance": float(np.min(residual_clearances))
        if residual_clearances
        else float("inf"),
        "mean_policy_residual_clearance_scale": float(np.mean(residual_scales)) if residual_scales else 1.0,
        "final_hold_steps": int(info["hold_steps"]),
        "max_hold_run": int(max_hold_run),
        "ever_hold": bool(max_hold_run > 0),
        "boundary_governor_active_steps": int(info["boundary_governor_active_steps"]),
        "mean_boundary_governor_correction": float(info["mean_boundary_governor_correction"]),
        "vertical_recovery_active_steps": int(info["vertical_recovery_active_steps"]),
        "vertical_recovery_agent_steps": int(info["vertical_recovery_agent_steps"]),
        "vertical_emergency_active_steps": int(info["vertical_emergency_active_steps"]),
        "vertical_emergency_agent_steps": int(info["vertical_emergency_agent_steps"]),
        "attitude_recovery_active_steps": int(info["attitude_recovery_active_steps"]),
        "attitude_recovery_agent_steps": int(info["attitude_recovery_agent_steps"]),
        "mean_policy_safety_filter_correction": float(np.mean(action_corrections)) if action_corrections else 0.0,
        "worst_policy_safety_constraint": float(min(constraint_minima)) if constraint_minima else None,
        "policy_safety_solver_nonconvergence_steps": solver_nonconvergence_steps,
        "policy_safety_repair_steps": repair_steps,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    capture_times = [float(row["capture_time_seconds"]) for row in rows if row["capture_time_seconds"] is not None]
    closure_speeds = [
        float(row["capture_relative_speed_at_closure"])
        for row in rows
        if row["capture_relative_speed_at_closure"] is not None
    ]
    closed_face_clearances = [
        float(row["capture_min_face_clearance_while_closed"])
        for row in rows
        if row["capture_min_face_clearance_while_closed"] is not None
    ]
    closed_net_margins = [
        float(row["capture_min_net_margin_while_closed"])
        for row in rows
        if row.get("capture_min_net_margin_while_closed") is not None
    ]
    closure_episodes = [row for row in rows if int(row["capture_closure_events"]) > 0]
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "encirclement_success_rate": float(np.mean([row["encirclement_success"] for row in rows])),
        "capture_success_rate": float(np.mean([row["capture_success"] for row in rows])),
        "capture_closure_episode_rate": float(len(closure_episodes) / len(rows)),
        "capture_escape_episode_rate": float(np.mean([row["capture_escaped"] for row in rows])),
        "capture_structural_failure_episode_rate": float(
            np.mean([row.get("capture_structural_failure", False) for row in rows])
        ),
        "capture_escape_after_closure_rate": (
            float(np.mean([row["capture_escaped"] for row in closure_episodes])) if closure_episodes else None
        ),
        "mean_capture_time_seconds": float(np.mean(capture_times)) if capture_times else None,
        "mean_capture_relative_speed_at_closure": float(np.mean(closure_speeds)) if closure_speeds else None,
        "mean_capture_min_face_clearance_while_closed": (
            float(np.mean(closed_face_clearances)) if closed_face_clearances else None
        ),
        "mean_capture_min_net_margin_while_closed": (
            float(np.mean(closed_net_margins)) if closed_net_margins else None
        ),
        "mean_capture_net_contact_steps": float(
            np.mean([row.get("capture_net_contact_steps", 0) for row in rows])
        ),
        "max_capture_peak_net_tension": float(
            np.max([row.get("capture_peak_net_tension", 0.0) for row in rows])
        ),
        "max_capture_peak_net_strain": float(
            np.max([row.get("capture_peak_net_strain", 0.0) for row in rows])
        ),
        "max_capture_peak_contact_impulse": float(
            np.max([row.get("capture_peak_contact_impulse", 0.0) for row in rows])
        ),
        "capture_compression_events": int(sum(row.get("capture_compression_events", 0) for row in rows)),
        "collision_episode_rate": float(np.mean([row["collision_steps"] > 0 for row in rows])),
        "physical_collision_episode_rate": float(np.mean([row["physical_collision_steps"] > 0 for row in rows])),
        "world_violation_episode_rate": float(np.mean([row["world_violation_steps"] > 0 for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_final_slot_error": float(np.mean([row["mean_slot_error"] for row in rows])),
        "mean_final_hold_steps": float(np.mean([row["final_hold_steps"] for row in rows])),
        "mean_max_hold_run": float(np.mean([row["max_hold_run"] for row in rows])),
        "episodes_ever_in_hold": int(sum(bool(row["ever_hold"]) for row in rows)),
        "mean_min_clearance": float(np.mean([row["min_clearance"] for row in rows])),
        "worst_min_clearance": float(np.min([row["min_clearance"] for row in rows])),
        "mean_command_speed": float(np.mean([row["mean_command_speed"] for row in rows])),
        "mean_filtered_command_speed": float(np.mean([row["mean_filtered_command_speed"] for row in rows])),
        "mean_command_filter_correction": float(
            np.mean([row["mean_command_filter_correction"] for row in rows])
        ),
        "mean_realized_speed": float(np.mean([row["mean_realized_speed"] for row in rows])),
        "mean_policy_residual_norm": float(np.mean([row["mean_policy_residual_norm"] for row in rows])),
        "mean_policy_residual_min_obstacle_clearance": float(
            np.mean([row["mean_policy_residual_min_obstacle_clearance"] for row in rows])
        ),
        "worst_policy_residual_min_obstacle_clearance": float(
            np.min([row["worst_policy_residual_min_obstacle_clearance"] for row in rows])
        ),
        "mean_policy_residual_clearance_scale": float(
            np.mean([row["mean_policy_residual_clearance_scale"] for row in rows])
        ),
        "mean_vertical_recovery_active_steps": float(
            np.mean([row["vertical_recovery_active_steps"] for row in rows])
        ),
        "mean_vertical_recovery_agent_steps": float(
            np.mean([row["vertical_recovery_agent_steps"] for row in rows])
        ),
        "mean_vertical_emergency_active_steps": float(
            np.mean([row["vertical_emergency_active_steps"] for row in rows])
        ),
        "mean_vertical_emergency_agent_steps": float(
            np.mean([row["vertical_emergency_agent_steps"] for row in rows])
        ),
        "mean_attitude_recovery_active_steps": float(
            np.mean([row["attitude_recovery_active_steps"] for row in rows])
        ),
        "mean_attitude_recovery_agent_steps": float(
            np.mean([row["attitude_recovery_agent_steps"] for row in rows])
        ),
        "mean_policy_safety_filter_correction": float(
            np.mean([row["mean_policy_safety_filter_correction"] for row in rows])
        ),
        "worst_policy_safety_constraint": min(
            float(row["worst_policy_safety_constraint"])
            for row in rows
            if row["worst_policy_safety_constraint"] is not None
        ) if any(row["worst_policy_safety_constraint"] is not None for row in rows) else None,
        "policy_safety_solver_nonconvergence_steps": int(
            sum(row["policy_safety_solver_nonconvergence_steps"] for row in rows)
        ),
        "policy_safety_repair_steps": int(sum(row["policy_safety_repair_steps"] for row in rows)),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    document, environment, settings, _environment_path = load_document(args)
    if args.trace_csv is not None and int(settings["episodes"]) != 1:
        raise ValueError("--trace-csv requires exactly one episode.")
    checkpoint_path = Path(document["checkpoint"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = (args.config.parent / checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    args.output.mkdir(parents=True, exist_ok=True)

    device = select_device(str(settings["device"]))
    seed = int(settings["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(bool(settings["deterministic_algorithms"]), warn_only=True)

    loaded_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy = SharedActorCritic(
        int(loaded_checkpoint["observation_dim"]),
        hidden_dim=int(loaded_checkpoint["hidden_dim"]),
    ).to(device)
    policy.load_state_dict(loaded_checkpoint["state_dict"])
    policy.eval()
    write_artifacts(args.output, document, environment, settings, checkpoint_path, device)
    writer = SummaryWriter(log_dir=str(args.output / "tensorboard"), flush_secs=10)
    writer.add_text("Config/effective_evaluation", f"```yaml\n{yaml.safe_dump(settings, sort_keys=False)}```", 0)
    policy_safety_document = dict(document.get("policy_safety_filter", {}))
    policy_residual_document = dict(document.get("policy_residual", {}))
    capture_closure_strategy = str(document.get("capture_closure_strategy", "disabled")).lower()
    if capture_closure_strategy not in {"disabled", "rule_guard"}:
        raise ValueError("capture_closure_strategy must be 'disabled' or 'rule_guard'.")
    enable_policy_safety_filter = bool(policy_safety_document.get("enabled", False))
    policy_safety_filter_type: str | None = None
    if enable_policy_safety_filter:
        policy_safety_filter_type = str(policy_safety_document.get("type", "kinematic_cbf"))
        allowed_filter_types = {"kinematic_cbf", "pybullet_response_cbf"}
        if policy_safety_filter_type not in allowed_filter_types:
            allowed = ", ".join(sorted(allowed_filter_types))
            raise ValueError(f"Unsupported policy_safety_filter.type={policy_safety_filter_type!r}; choose {allowed}.")

    scenario = next((item for item in environment["experiments"] if item["name"] == settings["scenario"]), None)
    if scenario is None:
        available = ", ".join(str(item["name"]) for item in environment["experiments"])
        raise ValueError(f"Unknown scenario {settings['scenario']!r}; available: {available}")
    target_speed_scale = float(
        settings.get("target_speed_scale", document.get("scenario_target_speed_scale", scenario["target_speed_scale"]))
    )
    env = PyBulletEncirclement3DEnv(
        environment,
        obstacle_count=int(scenario["obstacle_count"]),
        target_speed_scale=target_speed_scale,
    )
    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, float | int | bool]] | None = [] if args.trace_csv is not None else None
    started = time.perf_counter()
    try:
        for episode_index in range(int(settings["episodes"])):
            episode_seed = seed + episode_index
            record_history = bool(settings["record_trajectory"]) and episode_index == 0
            row = run_episode(
                env,
                policy,
                loaded_checkpoint,
                seed=episode_seed,
                record_history=record_history,
                device=device,
                policy_safety_filter_type=policy_safety_filter_type,
                policy_residual_document=policy_residual_document,
                capture_closure_strategy=capture_closure_strategy,
                trace_rows=trace_rows,
            )
            if record_history:
                plot_trajectory(env, args.output / f"trajectory_seed{episode_seed}.png", f"PyBullet policy seed {episode_seed}")
            rows.append(row)
            for metric, value in row.items():
                if metric != "seed" and isinstance(value, (int, float, bool)):
                    writer.add_scalar(f"Episode/{metric}", value, episode_index)
            writer.flush()
    finally:
        env.close()
        writer.close()

    fieldnames = list(rows[0].keys())
    with (args.output / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    if trace_rows is not None:
        trace_path = args.trace_csv
        if not trace_path.is_absolute():
            trace_path = args.output / trace_path
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", encoding="utf-8", newline="") as handle:
            trace_writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0].keys()))
            trace_writer.writeheader()
            trace_writer.writerows(trace_rows)
    summary = summarize(rows)
    summary.update(
        {
            "simulator": "gym-pybullet-drones CF2X with CtrlAviary/DSLPIDControl",
            "simulator_source_commit": "7688e7208a1572b1680736a3c0c9b93c379db3fe",
            "physics": environment["dynamics"]["pybullet_physics"],
            "control_mode": environment["dynamics"]["pybullet_control_mode"],
            "control_dt": env.control_dt,
            "defender_velocity_limit": environment["dynamics"]["pybullet_speed_limit"],
            "position_reference_horizon": environment["dynamics"]["pybullet_position_horizon"],
            "position_velocity_feedforward": environment["dynamics"].get(
                "pybullet_position_velocity_feedforward", 0.0
            ),
            "command_max_acceleration": environment["dynamics"].get("pybullet_command_max_acceleration", 0.0),
            "vertical_recovery_enabled": environment["dynamics"].get("pybullet_vertical_recovery_enabled", False),
            "vertical_recovery_altitude": environment["dynamics"].get("pybullet_vertical_recovery_altitude"),
            "vertical_recovery_descend_speed": environment["dynamics"].get(
                "pybullet_vertical_recovery_descend_speed"
            ),
            "vertical_recovery_climb_height": environment["dynamics"].get(
                "pybullet_vertical_recovery_climb_height"
            ),
            "vertical_emergency_enabled": environment["dynamics"].get("pybullet_vertical_emergency_enabled", False),
            "vertical_emergency_braking_deceleration": environment["dynamics"].get(
                "pybullet_vertical_emergency_braking_deceleration"
            ),
            "vertical_emergency_reaction_time": environment["dynamics"].get(
                "pybullet_vertical_emergency_reaction_time"
            ),
            "vertical_emergency_margin": environment["dynamics"].get("pybullet_vertical_emergency_margin"),
            "vertical_emergency_climb_height": environment["dynamics"].get(
                "pybullet_vertical_emergency_climb_height"
            ),
            "attitude_recovery_enabled": environment["dynamics"].get("pybullet_attitude_recovery_enabled", False),
            "attitude_recovery_max_tilt": environment["dynamics"].get("pybullet_attitude_recovery_max_tilt"),
            "attitude_recovery_climb_height": environment["dynamics"].get(
                "pybullet_attitude_recovery_climb_height"
            ),
            "boundary_reference_margin": environment["dynamics"].get("pybullet_boundary_reference_margin", 0.0),
            "policy_safety_filter": policy_safety_document,
            "policy_residual": policy_residual_document,
            "capture_closure_strategy": capture_closure_strategy,
            "target_speed_scale": target_speed_scale,
            "policy_algorithm": loaded_checkpoint["algorithm"],
        }
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "run_metadata.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": time.perf_counter() - started,
                "device": str(device),
                "cuda": device_metadata(device),
                "evaluation_only": True,
                "cross_domain_statement": "PyBullet dynamics result; not a real-flight result.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
