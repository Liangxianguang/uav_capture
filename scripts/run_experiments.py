"""Run deterministic 3D encirclement baseline experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.controllers import (
    CBFSafetyFilteredSlotController,
    CaptureAwareTetrahedralSlotController,
    HoldAwareTetrahedralSlotController,
    SpatialContainmentTetrahedralSlotController,
    TetrahedralSlotController,
)
from encirclement3d.dynamics import InertialEncirclement3DEnv
from encirclement3d.environment import Encirclement3DEnv
from encirclement3d.pybullet_env import PYBULLET_DRONES_ROOT, PyBulletEncirclement3DEnv
from encirclement3d.reporting import plot_summary, plot_trajectory


def run_episode(
    env: Encirclement3DEnv,
    seed: int,
    record_history: bool,
    controller_name: str,
) -> dict[str, float | int | bool | None]:
    observation = env.reset(seed=seed, record_history=record_history)
    if controller_name == "rule":
        controller = TetrahedralSlotController(env)
    elif controller_name == "hold_aware":
        controller = HoldAwareTetrahedralSlotController(env)
    elif controller_name == "cbf":
        controller = CBFSafetyFilteredSlotController(env)
    elif controller_name == "capture_rule":
        controller = CaptureAwareTetrahedralSlotController(env)
    elif controller_name == "spatial_containment_rule":
        controller = SpatialContainmentTetrahedralSlotController(env)
    else:
        raise ValueError(f"Unknown controller: {controller_name}")
    info: dict[str, float | int | bool] = {}
    correction_norms: list[float] = []
    constraint_minima: list[float] = []
    fallback_steps = 0
    constraint_violation_steps = 0
    solver_nonconvergence_steps = 0
    net_contact_steps = 0
    while True:
        action = controller.act(observation)
        close_cage = bool(getattr(controller, "should_close", lambda _observation: False)(observation))
        diagnostics = getattr(controller, "last_diagnostics", None)
        if diagnostics is not None:
            correction_norms.append(float(diagnostics.action_correction_norm))
            constraint_minima.append(float(diagnostics.minimum_constraint_value))
            fallback_steps += int(diagnostics.used_fallback)
            constraint_violation_steps += int(diagnostics.minimum_constraint_value < -1e-5)
            solver_nonconvergence_steps += int(not diagnostics.solver_success)
        observation, _reward, terminated, truncated, info = env.step(
            action,
            record_history=record_history,
            close_cage=close_cage,
        )
        net_contact_steps += int(bool(info.get("capture_net_contact", False)))
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
        "capture_net_contact_steps": net_contact_steps,
        "capture_peak_net_tension": float(info.get("capture_peak_net_tension", 0.0)),
        "capture_peak_net_strain": float(info.get("capture_peak_net_strain", 0.0)),
        "capture_peak_contact_impulse": float(info.get("capture_peak_contact_impulse", 0.0)),
        "capture_time_seconds": info["capture_time_seconds"],
        "capture_relative_speed_at_closure": info["capture_relative_speed_at_closure"],
        "capture_min_face_clearance_while_closed": info["capture_min_face_clearance_while_closed"],
        "capture_min_net_margin_while_closed": info.get("capture_min_net_margin_while_closed"),
        "steps": env.step_count,
        "collision_steps": int(info["collision_steps"]),
        "physical_collision_steps": int(info.get("physical_collision_steps", 0)),
        "world_violation_steps": int(info.get("world_violation_steps", 0)),
        "min_clearance": float(info["min_clearance_so_far"]),
        "mean_slot_error": float(info["mean_slot_error"]),
        "mean_action_correction": float(sum(correction_norms) / len(correction_norms)) if correction_norms else 0.0,
        "max_action_correction": float(max(correction_norms)) if correction_norms else 0.0,
        "safety_filter_fallback_steps": fallback_steps,
        "worst_safety_constraint": float(min(constraint_minima)) if constraint_minima else None,
        "safety_constraint_violation_steps": constraint_violation_steps,
        "safety_solver_nonconvergence_steps": solver_nonconvergence_steps,
        "mean_defender_mass": (
            float(np.mean(env.defender_masses)) if hasattr(env, "defender_masses") else None
        ),
        "mean_defender_drag": (
            float(np.mean(env.defender_drag_coefficients)) if hasattr(env, "defender_drag_coefficients") else None
        ),
    }


def make_environment(config: dict, obstacle_count: int, target_speed_scale: float):
    backend = str(config.get("dynamics", {}).get("backend", "kinematic"))
    if backend == "inertial":
        environment_class = InertialEncirclement3DEnv
    elif backend == "pybullet":
        environment_class = PyBulletEncirclement3DEnv
    else:
        environment_class = Encirclement3DEnv
    return environment_class(
        config,
        obstacle_count=obstacle_count,
        target_speed_scale=target_speed_scale,
    )


def validate_spatial_containment_configuration(config: dict) -> None:
    """Reject physical-capture options from the primary containment task."""
    capture = config.get("task", {}).get("capture", {})
    if not isinstance(capture, dict) or not bool(capture.get("enabled", False)):
        raise ValueError("spatial_containment_rule requires task.capture.enabled=true for the virtual closure gate.")
    if str(capture.get("model", "analytical")) not in {"analytical", "rigid_contact"}:
        raise ValueError(
            "spatial_containment_rule accepts only analytical or rigid_contact virtual-cage models, not flexible nets."
        )
    dynamics = config.get("dynamics", {})
    if bool(dynamics.get("pybullet_softbody_net_enabled", False)):
        raise ValueError("spatial_containment_rule cannot enable the PyBullet soft-body net diagnostic.")


def write_spatial_containment_summary(
    output: Path,
    summary: dict[str, dict[str, float | int | None]],
    config: dict,
) -> None:
    """Write primary-task terminology without changing legacy result fields."""
    research_task = config.get("research_task", {})
    scenarios: dict[str, dict[str, float | int | None]] = {}
    for scenario, values in summary.items():
        scenarios[scenario] = {
            "episodes": values["episodes"],
            "spatial_containment_success_rate": values["capture_success_rate"],
            "virtual_cage_closure_rate": values["capture_closure_episode_rate"],
            "post_closure_escape_rate": values["capture_escape_after_closure_rate"],
            "collision_episode_rate": values["collision_episode_rate"],
            "mean_minimum_virtual_face_margin_m": values["mean_capture_min_net_margin_while_closed"],
            "mean_containment_time_seconds": values["mean_capture_time_seconds"],
            "mean_relative_speed_at_closure_m_s": values["mean_capture_relative_speed_at_closure"],
            "source_metric_names": {
                "success": "capture_success_rate",
                "closure": "capture_closure_episode_rate",
                "escape": "capture_escape_after_closure_rate",
                "face_margin": "mean_capture_min_net_margin_while_closed",
                "time": "mean_capture_time_seconds",
                "relative_speed": "mean_capture_relative_speed_at_closure",
            },
        }
    payload = {
        "task_name": research_task.get("name", "obstacle_aware_3d_spatial_containment"),
        "target_model": research_task.get("target_model", "kinematic_evasive_target"),
        "success_definition": (
            "A radius-aware virtual tetrahedral cage closes and remains collision-free for the configured hold time; "
            "the kinematic target does not escape."
        ),
        "not_claimed": [
            "physical net deployment",
            "target vehicle contact dynamics",
            "mesh loading or rupture",
            "real-flight capture",
        ],
        "scenarios": scenarios,
    }
    output.joinpath("spatial_containment_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def write_reproducibility_artifacts(output: Path, config_path: Path, config: dict) -> dict:
    """Store the exact inputs and runtime identity used for this experiment."""
    output.joinpath("config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    environment = "\n".join(
        [
            f"python={sys.version.replace(chr(10), ' ')}",
            f"platform={platform.platform()}",
            f"numpy={version('numpy')}",
            f"matplotlib={version('matplotlib')}",
            f"PyYAML={version('PyYAML')}",
            f"pytest={version('pytest')}",
            f"scipy={version('scipy')}",
        ]
    )
    output.joinpath("environment.txt").write_text(environment + "\n", encoding="utf-8")

    source_paths = [
        PROJECT_ROOT / "scripts" / "run_experiments.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "environment.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "capture.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "flexible_net.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "dynamics.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "controllers.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "safety.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "reporting.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pybullet_env.py",
    ]
    if str(config.get("dynamics", {}).get("backend", "kinematic")) == "pybullet":
        source_paths.extend(
            [
                PYBULLET_DRONES_ROOT / "gym_pybullet_drones" / "envs" / "BaseAviary.py",
                PYBULLET_DRONES_ROOT / "gym_pybullet_drones" / "envs" / "VelocityAviary.py",
                PYBULLET_DRONES_ROOT / "gym_pybullet_drones" / "envs" / "CtrlAviary.py",
                PYBULLET_DRONES_ROOT / "gym_pybullet_drones" / "control" / "DSLPIDControl.py",
            ]
        )
    hashes = {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    output.joinpath("source_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--controller",
        choices=["rule", "hold_aware", "cbf", "capture_rule", "spatial_containment_rule"],
        default="rule",
    )
    parser.add_argument("--backend", choices=["kinematic", "inertial", "pybullet"], default=None)
    parser.add_argument("--episodes", type=int, default=None, help="Override every scenario's episode count for a smoke test.")
    parser.add_argument("--scenario", type=str, default=None, help="Run one named scenario only.")
    parser.add_argument("--pybullet-physics", type=str, default=None, help="Override the named PyBullet physics backend.")
    parser.add_argument("--pybullet-speed-limit", type=float, default=None, help="Override VelocityAviary's m/s command limit.")
    parser.add_argument("--pybullet-position-horizon", type=float, default=None, help="Override the position-PID reference horizon in seconds.")
    parser.add_argument("--target-speed-scale", type=float, default=None, help="Override the target speed scale for the selected scenario(s).")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if "dynamics" not in config:
        config["dynamics"] = {}
    if args.backend is not None:
        config["dynamics"]["backend"] = args.backend
    if args.episodes is not None:
        if args.episodes <= 0:
            raise ValueError("--episodes must be positive.")
        for experiment in config["experiments"]:
            experiment["episodes"] = args.episodes
    if args.scenario is not None:
        selected = [item for item in config["experiments"] if item["name"] == args.scenario]
        if not selected:
            available = ", ".join(str(item["name"]) for item in config["experiments"])
            raise ValueError(f"Unknown scenario {args.scenario!r}; available: {available}")
        config["experiments"] = selected
    if args.pybullet_physics is not None:
        config["dynamics"]["pybullet_physics"] = args.pybullet_physics
    if args.pybullet_speed_limit is not None:
        if args.pybullet_speed_limit <= 0.0:
            raise ValueError("--pybullet-speed-limit must be positive.")
        config["dynamics"]["pybullet_speed_limit"] = args.pybullet_speed_limit
    if args.pybullet_position_horizon is not None:
        if args.pybullet_position_horizon <= 0.0:
            raise ValueError("--pybullet-position-horizon must be positive.")
        config["dynamics"]["pybullet_position_horizon"] = args.pybullet_position_horizon
    if args.target_speed_scale is not None:
        if args.target_speed_scale <= 0.0:
            raise ValueError("--target-speed-scale must be positive.")
        for experiment in config["experiments"]:
            experiment["target_speed_scale"] = args.target_speed_scale
    if args.controller == "spatial_containment_rule":
        validate_spatial_containment_configuration(config)
    backend = str(config["dynamics"].get("backend", "kinematic"))
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_reproducibility_artifacts(output, args.config, config)
    started_at = time.perf_counter()
    rows: list[dict[str, float | int | bool | str | None]] = []

    for experiment_index, experiment in enumerate(config["experiments"]):
        for episode_index in range(int(experiment["episodes"])):
            # PyBullet creates a native physics client. Reusing one wrapper
            # across reset/close cycles can terminate the Windows process
            # without a Python exception, so every episode owns exactly one
            # environment and closes it in the same iteration. This leaves
            # seeds, controller logic, and reported task semantics unchanged.
            env = make_environment(
                config,
                obstacle_count=int(experiment["obstacle_count"]),
                target_speed_scale=float(experiment["target_speed_scale"]),
            )
            record_history = episode_index == 0
            seed = int(config["seed"]) + experiment_index * 10_000 + episode_index
            try:
                row = run_episode(
                    env,
                    seed=seed,
                    record_history=record_history,
                    controller_name=args.controller,
                )
                row["scenario"] = str(experiment["name"])
                row["controller"] = args.controller
                rows.append(row)
                if record_history and args.controller != "spatial_containment_rule":
                    plot_trajectory(env, output / f"trajectory_{experiment['name']}.png", f"{experiment['name']} seed {seed}")
            finally:
                close = getattr(env, "close", None)
                if close is not None:
                    close()

    fieldnames = [
        "controller",
        "scenario",
        "seed",
        "success",
        "encirclement_success",
        "capture_enabled",
        "capture_success",
        "capture_closed",
        "capture_escaped",
        "capture_structural_failure",
        "capture_close_attempts",
        "capture_closure_events",
        "capture_close_rejected_steps",
        "capture_escape_events",
        "capture_compression_events",
        "capture_net_contact_steps",
        "capture_peak_net_tension",
        "capture_peak_net_strain",
        "capture_peak_contact_impulse",
        "capture_time_seconds",
        "capture_relative_speed_at_closure",
        "capture_min_face_clearance_while_closed",
        "capture_min_net_margin_while_closed",
        "steps",
        "collision_steps",
        "physical_collision_steps",
        "world_violation_steps",
        "min_clearance",
        "mean_slot_error",
        "mean_action_correction",
        "max_action_correction",
        "safety_filter_fallback_steps",
        "worst_safety_constraint",
        "safety_constraint_violation_steps",
        "safety_solver_nonconvergence_steps",
        "mean_defender_mass",
        "mean_defender_drag",
    ]
    with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, dict[str, float | int | None]] = {}
    for name in sorted({str(row["scenario"]) for row in rows}):
        subset = [row for row in rows if row["scenario"] == name]
        safety_minima = [
            float(row["worst_safety_constraint"])
            for row in subset
            if row["worst_safety_constraint"] is not None
        ]
        mass_values = [float(row["mean_defender_mass"]) for row in subset if row["mean_defender_mass"] is not None]
        drag_values = [float(row["mean_defender_drag"]) for row in subset if row["mean_defender_drag"] is not None]
        capture_times = [float(row["capture_time_seconds"]) for row in subset if row["capture_time_seconds"] is not None]
        closure_speeds = [
            float(row["capture_relative_speed_at_closure"])
            for row in subset
            if row["capture_relative_speed_at_closure"] is not None
        ]
        closed_face_clearances = [
            float(row["capture_min_face_clearance_while_closed"])
            for row in subset
            if row["capture_min_face_clearance_while_closed"] is not None
        ]
        closed_net_margins = [
            float(row["capture_min_net_margin_while_closed"])
            for row in subset
            if row["capture_min_net_margin_while_closed"] is not None
        ]
        closure_episodes = [row for row in subset if int(row["capture_closure_events"]) > 0]
        summary[name] = {
            "episodes": len(subset),
            "success_rate": sum(bool(row["success"]) for row in subset) / len(subset),
            "encirclement_success_rate": sum(bool(row["encirclement_success"]) for row in subset) / len(subset),
            "capture_success_rate": sum(bool(row["capture_success"]) for row in subset) / len(subset),
            "capture_closure_episode_rate": len(closure_episodes) / len(subset),
            "capture_escape_episode_rate": sum(bool(row["capture_escaped"]) for row in subset) / len(subset),
            "capture_structural_failure_episode_rate": (
                sum(bool(row["capture_structural_failure"]) for row in subset) / len(subset)
            ),
            "capture_escape_after_closure_rate": (
                sum(bool(row["capture_escaped"]) for row in closure_episodes) / len(closure_episodes)
                if closure_episodes
                else None
            ),
            "mean_capture_time_seconds": sum(capture_times) / len(capture_times) if capture_times else None,
            "mean_capture_relative_speed_at_closure": (
                sum(closure_speeds) / len(closure_speeds) if closure_speeds else None
            ),
            "mean_capture_min_face_clearance_while_closed": (
                sum(closed_face_clearances) / len(closed_face_clearances) if closed_face_clearances else None
            ),
            "mean_capture_min_net_margin_while_closed": (
                sum(closed_net_margins) / len(closed_net_margins) if closed_net_margins else None
            ),
            "mean_capture_net_contact_steps": sum(int(row["capture_net_contact_steps"]) for row in subset) / len(subset),
            "max_capture_peak_net_tension": max(float(row["capture_peak_net_tension"]) for row in subset),
            "max_capture_peak_net_strain": max(float(row["capture_peak_net_strain"]) for row in subset),
            "max_capture_peak_contact_impulse": max(
                float(row["capture_peak_contact_impulse"]) for row in subset
            ),
            "capture_compression_events": sum(int(row["capture_compression_events"]) for row in subset),
            "mean_steps": sum(int(row["steps"]) for row in subset) / len(subset),
            "mean_collision_steps": sum(int(row["collision_steps"]) for row in subset) / len(subset),
            "collision_episode_rate": sum(int(row["collision_steps"]) > 0 for row in subset) / len(subset),
            "physical_collision_episode_rate": sum(
                int(row["physical_collision_steps"]) > 0 for row in subset
            ) / len(subset),
            "world_violation_episode_rate": sum(
                int(row["world_violation_steps"]) > 0 for row in subset
            ) / len(subset),
            "mean_min_clearance": sum(float(row["min_clearance"]) for row in subset) / len(subset),
            "worst_min_clearance": min(float(row["min_clearance"]) for row in subset),
            "mean_final_slot_error": sum(float(row["mean_slot_error"]) for row in subset) / len(subset),
            "mean_action_correction": sum(float(row["mean_action_correction"]) for row in subset) / len(subset),
            "mean_max_action_correction": sum(float(row["max_action_correction"]) for row in subset) / len(subset),
            "safety_filter_fallback_steps": sum(int(row["safety_filter_fallback_steps"]) for row in subset),
            "worst_safety_constraint": min(safety_minima) if safety_minima else None,
            "safety_constraint_violation_steps": sum(
                int(row["safety_constraint_violation_steps"]) for row in subset
            ),
            "safety_solver_nonconvergence_steps": sum(
                int(row["safety_solver_nonconvergence_steps"]) for row in subset
            ),
            "mean_defender_mass": sum(mass_values) / len(mass_values) if mass_values else None,
            "mean_defender_drag": sum(drag_values) / len(drag_values) if drag_values else None,
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.controller == "spatial_containment_rule":
        write_spatial_containment_summary(output, summary, config)
    (output / "run_metadata.json").write_text(
        json.dumps(
            {
                "controller": args.controller,
                "backend": backend,
                "episode_override": args.episodes,
                "scenario": args.scenario,
                "pybullet_physics_override": args.pybullet_physics,
                "pybullet_speed_limit_override": args.pybullet_speed_limit,
                "pybullet_position_horizon_override": args.pybullet_position_horizon,
                "target_speed_scale_override": args.target_speed_scale,
                "elapsed_seconds": time.perf_counter() - started_at,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.controller == "spatial_containment_rule":
        success_label = "spatial containment success rate"
    else:
        success_label = "capture success rate" if bool(config["task"].get("capture", {}).get("enabled", False)) else "containment success rate"
    if args.controller != "spatial_containment_rule":
        plot_summary(rows, output / "summary.png", success_label=success_label)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
