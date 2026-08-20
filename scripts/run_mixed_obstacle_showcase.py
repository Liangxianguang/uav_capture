"""Run a frozen checkpoint on the controlled central mixed-obstacle showcase."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.pursuit_controllers import (  # noqa: E402
    DynamicEncirclementController,
    PursuitCBFSafetyFilter,
    SafetyFilteredPursuitController,
)
from encirclement3d.showcase import (  # noqa: E402
    capture_contract_metrics,
    central_mixed_obstacle_scenario,
    crossing_metrics,
    prepare_showcase_episode,
    scenario_metadata,
    target_crossing_pursuit_overrides,
    transit_execution_metrics,
    target_min_clearance,
    transit_route_metrics,
)
from evaluate_capture_radius_mappo import load_policy, save_trajectory, select_device  # noqa: E402
from replay_capture_radius_checkpoint import METHOD_CONFIGS, render_animation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHOD_CONFIGS), default="f2")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=642002)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-side-distance", type=float, default=5.0)
    parser.add_argument("--scenario", choices=("s1", "s1_cross", "s2", "s2_cross"), default="s1_cross")
    parser.add_argument(
        "--detection-range",
        type=float,
        default=14.0,
        help="Showcase-only sensor range; preserves partial observations while making opposite-side starts observable.",
    )
    parser.add_argument("--target-speed-scale", type=float, default=0.55)
    parser.add_argument("--use-cbf", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frame-stride", type=int, default=1)
    return parser.parse_args()


def build_config(
    method: str,
    detection_range: float,
    target_speed_scale: float,
    *,
    target_crossing_required: bool = False,
) -> dict[str, Any]:
    if detection_range <= 0.0 or target_speed_scale <= 0.0:
        raise ValueError("detection-range and target-speed-scale must be positive.")
    config = yaml.safe_load(METHOD_CONFIGS[method].read_text(encoding="utf-8"))
    config["task"]["pursuit"].setdefault("include_uncertainty_features", method == "f2")
    config["task"]["pursuit"]["obstacle_profile"] = "mixed"
    config["task"]["pursuit"]["detection_range"] = float(detection_range)
    if target_crossing_required:
        # A crossing target still avoids obstacles, but its deliberate transit
        # intent must outweigh the normal "flee away from pursuers" bias.
        # This is a controlled capture task, not a claim that an adversary
        # voluntarily drives toward a defender in the open world.
        config["task"]["pursuit"].update(target_crossing_pursuit_overrides())
    config["experiments"] = [
        {
            "name": "central_mixed_obstacles",
            "episodes": 1,
            "obstacle_count": 3,
            "target_speed_scale": float(target_speed_scale),
        }
    ]
    return config


def build_showcase_scenario(scenario_kind: str, initial_side_distance: float) -> Any:
    """Return a fixed scenario with an explicit crossing contract.

    ``s1_cross`` is the V3 main task: defenders start on the left, the target
    starts on the right, and both approach the central obstacle zone from
    opposite directions.  ``s2``/``s2_cross`` retain the reverse-side
    regression tasks.
    """
    if scenario_kind not in {"s1", "s1_cross", "s2", "s2_cross"}:
        raise ValueError(f"Unknown showcase scenario: {scenario_kind}")
    defender_side = "left" if scenario_kind in {"s1", "s1_cross"} else "right"
    return central_mixed_obstacle_scenario(
        initial_side_distance=initial_side_distance,
        target_crossing_required=scenario_kind in {"s1_cross", "s2_cross"},
        defender_side=defender_side,
    )


def _finalize_showcase_row(
    row: dict[str, Any],
    env: CaptureRadiusPursuit3DEnv,
    scenario: Any,
    final_info: dict[str, Any],
    *,
    target_collision: bool,
) -> dict[str, Any]:
    crossing = crossing_metrics(env, scenario.obstacle_zone_x)
    contract = capture_contract_metrics(
        final_info,
        crossing,
        target_collision=target_collision,
        target_crossing_required=bool(scenario.target_crossing_required),
    )
    transit = transit_route_metrics(env, scenario)
    transit_execution = transit_execution_metrics(env, scenario)
    row.update(crossing)
    row.update(contract)
    row.update(transit)
    row.update(transit_execution)
    # Kept as a diagnostic. It refers only to completed crossings in this
    # capture rollout; it is not the V3 capture success criterion.
    row["rollout_all_defenders_crossed"] = bool(crossing["all_defenders_crossed"])
    row["rollout_target_crossed"] = bool(crossing["target_crossed"])
    row["showcase_success"] = bool(contract["safe_capture_in_pursuit"])
    return row


def rollout_showcase(
    policy: Any,
    config: dict[str, Any],
    scenario: Any,
    seed: int,
    device: torch.device,
    action_scale: float,
    use_cbf: bool,
) -> tuple[dict[str, Any], CaptureRadiusPursuit3DEnv]:
    env = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=len(scenario.obstacles),
        target_speed_scale=float(config["experiments"][0]["target_speed_scale"]),
    )
    observation = prepare_showcase_episode(env, scenario, seed=seed, record_history=True)
    safety_filter = PursuitCBFSafetyFilter(env) if use_cbf else None
    local_observation = env.policy_observations(observation)
    actor_hidden = (
        policy.initial_actor_hidden(env.n_defenders, device=device)
        if hasattr(policy, "initial_actor_hidden")
        else None
    )
    visible_fractions: list[float] = []
    message_ages: list[float] = []
    observation_ages: list[float] = []
    final_info: dict[str, Any] = {}
    target_collision = False
    with torch.no_grad():
        while True:
            local = torch.as_tensor(local_observation, device=device)
            if actor_hidden is not None:
                distribution, actor_hidden = policy.distribution_step(local, actor_hidden)
            else:
                distribution = policy.distribution(local)
            action = torch.tanh(distribution.mean).cpu().numpy() * action_scale
            if safety_filter is not None:
                action, _diagnostics = safety_filter.filter(action, observation)
            observation, _reward, terminated, truncated, final_info = env.step(action, record_history=True)
            visible_fractions.append(float(final_info["target_visible_fraction"]))
            message_ages.append(float(final_info["mean_message_age_steps"]))
            observation_ages.append(float(final_info["mean_observation_age_steps"]))
            target_clearance = min(
                float(env._obstacle_clearance(env.target_position, obstacle)) for obstacle in env.obstacles
            )
            if target_clearance < 0.0:
                target_collision = True
                break
            if terminated or truncated:
                break
            local_observation = env.policy_observations(observation)
    target_clearance = target_min_clearance(env)
    safe_capture = bool(final_info.get("safe_capture_success", False)) and not target_collision
    row = {
        "seed": int(seed),
        "scenario": scenario.name,
        "safe_capture_success": safe_capture,
        "capture_event": bool(final_info.get("capture_event", False)),
        "collision": bool(final_info.get("collision", False) or target_collision),
        "target_obstacle_collision": target_collision,
        "capture_time_seconds": final_info.get("capture_time_seconds"),
        "capturing_defender_id": final_info.get("capturing_defender_id"),
        "steps": int(env.step_count),
        "termination_reason": "target_safety_failure" if target_collision else str(final_info.get("termination_reason", "running")),
        "world_violation_steps": int(final_info.get("world_violation_steps", 0)),
        "min_clearance_m": float(final_info.get("min_clearance_so_far", float("inf"))),
        "target_min_obstacle_clearance_m": target_clearance,
        "mean_visible_fraction": float(np.mean(visible_fractions)) if visible_fractions else 0.0,
        "mean_message_age_steps": float(np.mean(message_ages)) if message_ages else 0.0,
        "mean_observation_age_steps": float(np.mean(observation_ages)) if observation_ages else 0.0,
        "use_cbf": bool(use_cbf),
    }
    return _finalize_showcase_row(row, env, scenario, final_info, target_collision=target_collision), env


def rollout_showcase_expert(
    config: dict[str, Any],
    scenario: Any,
    seed: int,
    use_cbf: bool,
) -> tuple[dict[str, Any], CaptureRadiusPursuit3DEnv]:
    """Replay the local-information rule expert on the same showcase task."""
    env = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=len(scenario.obstacles),
        target_speed_scale=float(config["experiments"][0]["target_speed_scale"]),
    )
    observation = prepare_showcase_episode(env, scenario, seed=seed, record_history=True)
    base_controller = DynamicEncirclementController(env)
    controller: Any = SafetyFilteredPursuitController(base_controller) if use_cbf else base_controller
    visible_fractions: list[float] = []
    message_ages: list[float] = []
    observation_ages: list[float] = []
    final_info: dict[str, Any] = {}
    target_collision = False
    while True:
        action = controller.act(observation)
        observation, _reward, terminated, truncated, final_info = env.step(action, record_history=True)
        visible_fractions.append(float(final_info["target_visible_fraction"]))
        message_ages.append(float(final_info["mean_message_age_steps"]))
        observation_ages.append(float(final_info["mean_observation_age_steps"]))
        target_clearance = min(
            float(env._obstacle_clearance(env.target_position, obstacle)) for obstacle in env.obstacles
        )
        if target_clearance < 0.0:
            target_collision = True
            break
        if terminated or truncated:
            break
    row = {
        "seed": int(seed),
        "scenario": scenario.name,
        "safe_capture_success": bool(final_info.get("safe_capture_success", False)) and not target_collision,
        "capture_event": bool(final_info.get("capture_event", False)),
        "collision": bool(final_info.get("collision", False) or target_collision),
        "target_obstacle_collision": target_collision,
        "capture_time_seconds": final_info.get("capture_time_seconds"),
        "capturing_defender_id": final_info.get("capturing_defender_id"),
        "steps": int(env.step_count),
        "termination_reason": "target_safety_failure" if target_collision else str(final_info.get("termination_reason", "running")),
        "world_violation_steps": int(final_info.get("world_violation_steps", 0)),
        "min_clearance_m": float(final_info.get("min_clearance_so_far", float("inf"))),
        "target_min_obstacle_clearance_m": target_min_clearance(env),
        "mean_visible_fraction": float(np.mean(visible_fractions)) if visible_fractions else 0.0,
        "mean_message_age_steps": float(np.mean(message_ages)) if message_ages else 0.0,
        "mean_observation_age_steps": float(np.mean(observation_ages)) if observation_ages else 0.0,
        "use_cbf": bool(use_cbf),
    }
    return _finalize_showcase_row(row, env, scenario, final_info, target_collision=target_collision), env


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = build_showcase_scenario(args.scenario, args.initial_side_distance)
    config = build_config(
        args.method,
        args.detection_range,
        args.target_speed_scale,
        target_crossing_required=bool(scenario.target_crossing_required),
    )
    device = select_device(args.device)
    prototype = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=3,
        target_speed_scale=float(args.target_speed_scale),
    )
    policy, action_scale, _metadata = load_policy(
        checkpoint,
        prototype,
        prototype.reset(seed=args.seed),
        device,
    )
    row, env = rollout_showcase(
        policy,
        config,
        scenario,
        seed=args.seed,
        device=device,
        action_scale=action_scale,
        use_cbf=args.use_cbf,
    )
    row.update({"method": args.method, "checkpoint": str(checkpoint), "device": str(device)})
    trajectory_path = output_dir / "trajectory.npz"
    save_trajectory(env, trajectory_path)
    (output_dir / "episode.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    (output_dir / "scenario.json").write_text(
        json.dumps(
            {"scenario_kind": args.scenario, **scenario_metadata(scenario)},
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    media = render_animation(
        trajectory_path,
        output_dir,
        title=f"{args.method} / central mixed obstacles / {'CBF' if args.use_cbf else 'raw'}",
        fps=args.fps,
        frame_stride=args.frame_stride,
        result=row,
    )
    row["media"] = media
    (output_dir / "episode.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps({"episode": row, "media": media}, indent=2), flush=True)


if __name__ == "__main__":
    main()
