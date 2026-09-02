"""Run a frozen checkpoint on the controlled central mixed-obstacle showcase."""

from __future__ import annotations

import argparse
import ast
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
from encirclement3d.observation_encoding import policy_observations  # noqa: E402
from encirclement3d.prediction import (  # noqa: E402
    ActionConditionedCandidateHistory,
    ActionConditionedCandidateReranker,
    ActionConditionedJEPAPredictor,
    make_action_candidates,
)
from encirclement3d.pursuit_controllers import (  # noqa: E402
    DynamicEncirclementController,
    PursuitCBFSafetyFilter,
    SafetyFilteredPursuitController,
)
from encirclement3d.showcase import (  # noqa: E402
    CentralCaptureProtocol,
    capture_contract_metrics,
    central_capture_protocol_metadata,
    central_capture_v4_scenario,
    central_mixed_obstacle_scenario,
    crossing_metrics,
    load_central_capture_protocol,
    prepare_showcase_episode,
    scenario_metadata,
    target_crossing_pursuit_overrides,
    transit_execution_metrics,
    target_min_clearance,
    transit_route_metrics,
    validate_central_capture_protocol_environment,
)
from evaluate_capture_radius_mappo import load_policy, save_trajectory, select_device  # noqa: E402
from replay_capture_radius_checkpoint import METHOD_CONFIGS, render_animation  # noqa: E402


TRANSIT_LIST_FIELDS = (
    "defender_transit_route_feasible",
    "defender_transit_route_length_m",
    "defender_transit_min_clearance_m",
    "defender_transit_goals",
    "target_transit_goal",
    "defender_transit_success",
    "defender_transit_steps",
    "defender_transit_reasons",
    "defender_transit_execution_min_clearance_m",
)
TRANSIT_BOOL_FIELDS = (
    "all_defenders_transit_route_feasible",
    "target_transit_route_feasible",
    "transit_route_feasible",
    "all_defenders_transit_success",
    "target_transit_success",
    "transit_success",
)
TRANSIT_FLOAT_FIELDS = (
    "transit_grid_step_m",
    "target_transit_route_length_m",
    "target_transit_min_clearance_m",
    "target_transit_execution_min_clearance_m",
)


def transit_metrics_from_episode_row(row: dict[str, Any]) -> dict[str, Any]:
    """Restore typed, policy-independent Transit evidence from an episode CSV row."""

    transit: dict[str, Any] = {}
    for field in TRANSIT_LIST_FIELDS:
        transit[field] = ast.literal_eval(str(row[field]))
    for field in TRANSIT_BOOL_FIELDS:
        transit[field] = str(row[field]).strip().lower() in {"1", "true", "yes"}
    for field in TRANSIT_FLOAT_FIELDS:
        transit[field] = float(row[field])
    transit["target_transit_steps"] = int(row["target_transit_steps"])
    transit["target_transit_reason"] = str(row["target_transit_reason"])
    return transit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHOD_CONFIGS), default="f2")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=642002)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-side-distance", type=float, default=5.0)
    parser.add_argument("--scenario", choices=("s1", "s1_cross", "s2", "s2_cross", "v4_s2"), default="s1_cross")
    parser.add_argument("--layout", choices=("open", "cylinder", "box", "wall", "cylinder_box", "mixed"), default="mixed")
    parser.add_argument(
        "--protocol-config",
        type=Path,
        help="Frozen V4 protocol YAML. Required for --scenario v4_s2.",
    )
    parser.add_argument(
        "--detection-range",
        type=float,
        default=14.0,
        help="Showcase-only sensor range; preserves partial observations while making opposite-side starts observable.",
    )
    parser.add_argument("--target-speed-scale", type=float, default=0.55)
    parser.add_argument("--use-cbf", action="store_true")
    parser.add_argument(
        "--recurrent-reset-interval",
        type=int,
        help="Reset recurrent actor state at this many control steps; defaults to checkpoint metadata.",
    )
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
    protocol: CentralCaptureProtocol | None = None,
    obstacle_count: int = 3,
) -> dict[str, Any]:
    if detection_range <= 0.0 or target_speed_scale <= 0.0:
        raise ValueError("detection-range and target-speed-scale must be positive.")
    config = yaml.safe_load(METHOD_CONFIGS[method].read_text(encoding="utf-8"))
    config["task"]["pursuit"].setdefault("include_uncertainty_features", method == "f2")
    config["task"]["pursuit"]["obstacle_profile"] = "mixed"
    config["task"]["pursuit"]["detection_range"] = float(
        protocol.detection_range if protocol is not None else detection_range
    )
    if protocol is not None:
        config["world"].update(
            {
                "half_extent_xy": protocol.half_extent_xy,
                "height": protocol.height,
                "minimum_altitude": protocol.minimum_altitude,
            }
        )
        config["task"]["pursuit"].update(
            {
                "capture_radius": protocol.capture_radius,
                "safety_margin": protocol.safety_margin,
                "target_motion_mode": protocol.target_motion_mode,
            }
        )
        config["task"]["policy_obstacle_geometry"] = protocol.policy_obstacle_geometry
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
            "obstacle_count": int(obstacle_count),
            "target_speed_scale": float(protocol.target_speed_scale if protocol is not None else target_speed_scale),
        }
    ]
    return config


def build_showcase_scenario(
    scenario_kind: str,
    initial_side_distance: float,
    protocol: CentralCaptureProtocol | None = None,
    layout: str = "mixed",
) -> Any:
    """Return a fixed scenario with an explicit crossing contract.

    ``s1_cross`` is the V3 main task: defenders start on the left, the target
    starts on the right, and both approach the central obstacle zone from
    opposite directions.  ``s2``/``s2_cross`` retain the reverse-side
    regression tasks.
    """
    if scenario_kind == "v4_s2":
        if protocol is None:
            raise ValueError("--scenario v4_s2 requires --protocol-config.")
        return central_capture_v4_scenario(protocol)
    if scenario_kind not in {"s1", "s1_cross", "s2", "s2_cross"}:
        raise ValueError(f"Unknown showcase scenario: {scenario_kind}")
    defender_side = "left" if scenario_kind in {"s1", "s1_cross"} else "right"
    target_crossing_required = (
        protocol.target_crossing_required
        if protocol is not None
        else scenario_kind in {"s1_cross", "s2_cross"}
    )
    return central_mixed_obstacle_scenario(
        initial_side_distance=initial_side_distance,
        target_crossing_required=target_crossing_required,
        defender_side=defender_side,
        layout=layout,
        required_defender_zone_entries=(protocol.required_defender_zone_entries if protocol is not None else 1),
        require_target_zone_entry=(protocol.require_target_zone_entry if protocol is not None else None),
    )


def _finalize_showcase_row(
    row: dict[str, Any],
    env: CaptureRadiusPursuit3DEnv,
    scenario: Any,
    final_info: dict[str, Any],
    *,
    target_collision: bool,
    transit_override: dict[str, Any] | None = None,
    validate_scenario: bool = True,
) -> dict[str, Any]:
    crossing = crossing_metrics(env, scenario.obstacle_zone_x)
    contract = capture_contract_metrics(
        final_info,
        crossing,
        target_collision=target_collision,
        target_crossing_required=bool(scenario.target_crossing_required),
        required_defender_zone_entries=int(scenario.required_defender_zone_entries),
        require_target_zone_entry=scenario.require_target_zone_entry,
    )
    if transit_override is None:
        transit = transit_route_metrics(env, scenario)
        transit_execution = transit_execution_metrics(env, scenario)
    else:
        transit = transit_override
        transit_execution = {}
    row.update(crossing)
    row.update(contract)
    row.update(transit)
    row.update(transit_execution)
    # Kept as a diagnostic. It refers only to completed crossings in this
    # capture rollout; it is not the V3 capture success criterion.
    row["rollout_all_defenders_crossed"] = bool(crossing["all_defenders_crossed"])
    row["rollout_target_crossed"] = bool(crossing["target_crossed"])
    row["showcase_success"] = bool(contract["cooperative_safe_capture"])
    return row


def rollout_showcase(
    policy: Any,
    config: dict[str, Any],
    scenario: Any,
    seed: int,
    device: torch.device,
    action_scale: float,
    use_cbf: bool,
    transit_override: dict[str, Any] | None = None,
    validate_scenario: bool = True,
    recurrent_reset_interval: int | None = None,
    jepa_predictor: ActionConditionedJEPAPredictor | None = None,
    jepa_history_length: int = 8,
    jepa_candidate_count: int = 5,
    jepa_perturbation_mps: float = 0.60,
    jepa_uncertainty_weight: float = 0.10,
    jepa_action_change_weight: float = 0.02,
) -> tuple[dict[str, Any], CaptureRadiusPursuit3DEnv]:
    if recurrent_reset_interval is not None and recurrent_reset_interval <= 0:
        raise ValueError("recurrent_reset_interval must be positive when provided.")
    env = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=len(scenario.obstacles),
        target_speed_scale=float(config["experiments"][0]["target_speed_scale"]),
    )
    observation = prepare_showcase_episode(
        env, scenario, seed=seed, record_history=True, validate_scenario=validate_scenario
    )
    safety_filter = PursuitCBFSafetyFilter(env) if use_cbf else None
    local_observation = policy_observations(env, observation)
    jepa_history = (
        ActionConditionedCandidateHistory(
            env,
            jepa_predictor,
            device,
            history_length=jepa_history_length,
            action_scale=action_scale,
        )
        if jepa_predictor is not None
        else None
    )
    jepa_reranker = (
        ActionConditionedCandidateReranker(
            jepa_history,
            horizon_seconds=float(config["task"]["pursuit"]["prediction_horizon_seconds"]),
            position_extent=float(config["world"]["half_extent_xy"]),
            uncertainty_weight=jepa_uncertainty_weight,
            action_change_weight=jepa_action_change_weight,
        )
        if jepa_history is not None
        else None
    )
    if jepa_history is not None:
        jepa_history.reset(local_observation)
    actor_hidden = (
        policy.initial_actor_hidden(env.n_defenders, device=device)
        if hasattr(policy, "initial_actor_hidden")
        else None
    )
    visible_fractions: list[float] = []
    message_ages: list[float] = []
    observation_ages: list[float] = []
    cbf_corrections: list[float] = []
    jepa_selection_indices: list[int] = []
    jepa_selection_scores: list[float] = []
    recurrent_hidden_resets = 0
    path_lengths = np.zeros(env.n_defenders, dtype=np.float64)
    previous_positions = env.defender_positions.copy()
    final_info: dict[str, Any] = {}
    target_collision = False
    with torch.no_grad():
        while True:
            if (
                actor_hidden is not None
                and recurrent_reset_interval is not None
                and env.step_count > 0
                and env.step_count % recurrent_reset_interval == 0
            ):
                actor_hidden = policy.initial_actor_hidden(env.n_defenders, device=device)
                recurrent_hidden_resets += 1
            local = torch.as_tensor(local_observation, device=device)
            if actor_hidden is not None:
                distribution, actor_hidden = policy.distribution_step(local, actor_hidden)
            else:
                distribution = policy.distribution(local)
            desired_action = torch.tanh(distribution.mean).cpu().numpy() * action_scale
            action = desired_action
            if jepa_reranker is not None:
                candidates = make_action_candidates(
                    desired_action,
                    perturbation_mps=jepa_perturbation_mps,
                    candidate_count=jepa_candidate_count,
                )
                action, selection = jepa_reranker.select(observation, candidates)
                jepa_selection_indices.append(int(selection.selected_index))
                jepa_selection_scores.append(float(selection.scores[selection.selected_index]))
            if safety_filter is not None:
                action, diagnostics = safety_filter.filter(action, observation)
                cbf_corrections.append(float(diagnostics.action_correction_norm))
            else:
                cbf_corrections.append(0.0)
            observation, _reward, terminated, truncated, final_info = env.step(action, record_history=True)
            path_lengths += np.linalg.norm(env.defender_positions - previous_positions, axis=1)
            previous_positions = env.defender_positions.copy()
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
            local_observation = policy_observations(env, observation)
            if jepa_history is not None:
                jepa_history.observe_after_action(local_observation, action)
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
        "defender_path_length_m": path_lengths.tolist(),
        "mean_defender_path_length_m": float(np.mean(path_lengths)),
        "total_defender_path_length_m": float(np.sum(path_lengths)),
        "mean_cbf_action_correction_norm": float(np.mean(cbf_corrections)) if cbf_corrections else 0.0,
        "max_cbf_action_correction_norm": float(max(cbf_corrections)) if cbf_corrections else 0.0,
        "use_cbf": bool(use_cbf),
        "recurrent_reset_interval_steps": recurrent_reset_interval,
        "recurrent_hidden_resets": recurrent_hidden_resets,
        "jepa_enabled": jepa_reranker is not None,
        "jepa_candidate_count": jepa_candidate_count if jepa_reranker is not None else None,
        "jepa_perturbation_mps": jepa_perturbation_mps if jepa_reranker is not None else None,
        "jepa_mean_selected_index": float(np.mean(jepa_selection_indices)) if jepa_selection_indices else None,
        "jepa_mean_selected_score": float(np.mean(jepa_selection_scores)) if jepa_selection_scores else None,
    }
    return _finalize_showcase_row(
        row,
        env,
        scenario,
        final_info,
        target_collision=target_collision,
        transit_override=transit_override,
        validate_scenario=validate_scenario,
    ), env


def rollout_showcase_expert(
    config: dict[str, Any],
    scenario: Any,
    seed: int,
    use_cbf: bool,
    transit_override: dict[str, Any] | None = None,
    validate_scenario: bool = True,
) -> tuple[dict[str, Any], CaptureRadiusPursuit3DEnv]:
    """Replay the local-information rule expert on the same showcase task."""
    env = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=len(scenario.obstacles),
        target_speed_scale=float(config["experiments"][0]["target_speed_scale"]),
    )
    observation = prepare_showcase_episode(
        env, scenario, seed=seed, record_history=True, validate_scenario=validate_scenario
    )
    base_controller = DynamicEncirclementController(env)
    controller: Any = SafetyFilteredPursuitController(base_controller) if use_cbf else base_controller
    visible_fractions: list[float] = []
    message_ages: list[float] = []
    observation_ages: list[float] = []
    cbf_corrections: list[float] = []
    path_lengths = np.zeros(env.n_defenders, dtype=np.float64)
    previous_positions = env.defender_positions.copy()
    final_info: dict[str, Any] = {}
    target_collision = False
    while True:
        action = controller.act(observation)
        diagnostics = getattr(controller, "last_diagnostics", None)
        cbf_corrections.append(float(diagnostics.action_correction_norm) if diagnostics is not None else 0.0)
        observation, _reward, terminated, truncated, final_info = env.step(action, record_history=True)
        path_lengths += np.linalg.norm(env.defender_positions - previous_positions, axis=1)
        previous_positions = env.defender_positions.copy()
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
        "defender_path_length_m": path_lengths.tolist(),
        "mean_defender_path_length_m": float(np.mean(path_lengths)),
        "total_defender_path_length_m": float(np.sum(path_lengths)),
        "mean_cbf_action_correction_norm": float(np.mean(cbf_corrections)) if cbf_corrections else 0.0,
        "max_cbf_action_correction_norm": float(max(cbf_corrections)) if cbf_corrections else 0.0,
        "use_cbf": bool(use_cbf),
    }
    return _finalize_showcase_row(
        row,
        env,
        scenario,
        final_info,
        target_collision=target_collision,
        transit_override=transit_override,
        validate_scenario=validate_scenario,
    ), env


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_central_capture_protocol(args.protocol_config) if args.protocol_config is not None else None
    scenario = build_showcase_scenario(
        args.scenario,
        args.initial_side_distance,
        protocol=protocol,
        layout=args.layout,
    )
    config = build_config(
        args.method,
        args.detection_range,
        args.target_speed_scale,
        target_crossing_required=bool(scenario.target_crossing_required),
        protocol=protocol,
        obstacle_count=len(scenario.obstacles),
    )
    device = select_device(args.device)
    prototype = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=len(scenario.obstacles),
        target_speed_scale=float(config["experiments"][0]["target_speed_scale"]),
    )
    if protocol is not None:
        validate_central_capture_protocol_environment(prototype, protocol)
    policy, action_scale, checkpoint_metadata = load_policy(
        checkpoint,
        prototype,
        prototype.reset(seed=args.seed),
        device,
    )
    metadata_reset_interval = checkpoint_metadata.get("recurrent_reset_interval_steps")
    recurrent_reset_interval = (
        int(args.recurrent_reset_interval)
        if args.recurrent_reset_interval is not None
        else int(metadata_reset_interval)
        if metadata_reset_interval is not None
        else None
    )
    row, env = rollout_showcase(
        policy,
        config,
        scenario,
        seed=args.seed,
        device=device,
        action_scale=action_scale,
        use_cbf=args.use_cbf,
        recurrent_reset_interval=recurrent_reset_interval,
    )
    row.update({"method": args.method, "checkpoint": str(checkpoint), "device": str(device)})
    trajectory_path = output_dir / "trajectory.npz"
    save_trajectory(env, trajectory_path)
    (output_dir / "episode.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    (output_dir / "scenario.json").write_text(
        json.dumps(
            {
                "scenario_kind": args.scenario,
                "layout": args.layout,
                **scenario_metadata(scenario),
                "central_capture_protocol": (
                    central_capture_protocol_metadata(protocol) if protocol is not None else None
                ),
            },
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
