"""Shared rollout implementation for E1 rule-expert and frozen-policy runs."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch

from .execution_dynamics import ExecutionDynamicsConfig
from .execution_env import ExecutionDynamicsPursuitWrapper
from .execution_safety import ExecutionAwarePursuitCBFSafetyFilter, make_kinematic_cbf
from .observation_encoding import policy_observations
from .pursuit_controllers import DynamicEncirclementController
from .pursuit_env import CaptureRadiusPursuit3DEnv
from .showcase import (
    ShowcaseScenario,
    capture_contract_metrics,
    crossing_metrics,
    prepare_showcase_episode,
    target_min_clearance,
    transit_execution_metrics,
    transit_route_metrics,
)


ExecutionMode = Literal["raw", "kinematic_cbf", "execution_aware_cbf"]


def rollout_execution_expert(
    config: dict[str, Any],
    scenario: ShowcaseScenario,
    *,
    seed: int,
    execution_config: ExecutionDynamicsConfig,
    execution_mode: ExecutionMode,
    validate_scenario: bool = True,
) -> tuple[dict[str, Any], CaptureRadiusPursuit3DEnv, ExecutionDynamicsPursuitWrapper]:
    """Evaluate the rule expert under exactly the E1 action path."""
    env, wrapper, observation = _make_wrapped_episode(
        config,
        scenario,
        seed=seed,
        execution_config=execution_config,
        validate_scenario=validate_scenario,
    )
    controller = DynamicEncirclementController(env)
    return _rollout(
        env,
        wrapper,
        scenario,
        observation,
        action_provider=lambda _observation, _hidden: (controller.act(_observation), None),
        execution_mode=execution_mode,
        seed=seed,
        actor_hidden=None,
    )


def rollout_execution_policy(
    policy: Any,
    config: dict[str, Any],
    scenario: ShowcaseScenario,
    *,
    seed: int,
    device: torch.device,
    action_scale: float,
    execution_config: ExecutionDynamicsConfig,
    execution_mode: ExecutionMode,
    validate_scenario: bool = True,
) -> tuple[dict[str, Any], CaptureRadiusPursuit3DEnv, ExecutionDynamicsPursuitWrapper]:
    """Evaluate a frozen feed-forward or recurrent policy under E1."""
    env, wrapper, observation = _make_wrapped_episode(
        config,
        scenario,
        seed=seed,
        execution_config=execution_config,
        validate_scenario=validate_scenario,
    )
    hidden = policy.initial_actor_hidden(env.n_defenders, device=device) if hasattr(policy, "initial_actor_hidden") else None

    def action_provider(current: dict[str, Any], actor_hidden: Any) -> tuple[np.ndarray, Any]:
        local = torch.as_tensor(policy_observations(env, current), device=device)
        with torch.no_grad():
            if actor_hidden is not None:
                distribution, next_hidden = policy.distribution_step(local, actor_hidden)
            else:
                distribution = policy.distribution(local)
                next_hidden = None
            action = torch.tanh(distribution.mean).cpu().numpy() * float(action_scale)
        return action, next_hidden

    return _rollout(
        env,
        wrapper,
        scenario,
        observation,
        action_provider=action_provider,
        execution_mode=execution_mode,
        seed=seed,
        actor_hidden=hidden,
    )


def _make_wrapped_episode(
    config: dict[str, Any],
    scenario: ShowcaseScenario,
    *,
    seed: int,
    execution_config: ExecutionDynamicsConfig,
    validate_scenario: bool,
) -> tuple[CaptureRadiusPursuit3DEnv, ExecutionDynamicsPursuitWrapper, dict[str, Any]]:
    env = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=len(scenario.obstacles),
        target_speed_scale=float(config["experiments"][0]["target_speed_scale"]),
    )
    wrapper = ExecutionDynamicsPursuitWrapper(env, execution_config)
    # wrapper.reset owns an RNG independent of the environment.  The showcase
    # helper repeats env.reset with the same seed before installing the fixed
    # map; defender velocities remain zero, so the execution state is still
    # exactly the reset state for this case.
    wrapper.reset(seed=int(seed), record_history=False)
    observation = prepare_showcase_episode(
        env,
        scenario,
        seed=int(seed),
        record_history=True,
        validate_scenario=validate_scenario,
    )
    wrapper.execution.reset(seed=_execution_seed(seed), initial_velocity=env.defender_velocities)
    wrapper.execution_history = []
    return env, wrapper, observation


def _rollout(
    env: CaptureRadiusPursuit3DEnv,
    wrapper: ExecutionDynamicsPursuitWrapper,
    scenario: ShowcaseScenario,
    observation: dict[str, Any],
    *,
    action_provider: Any,
    execution_mode: ExecutionMode,
    seed: int,
    actor_hidden: Any,
) -> tuple[dict[str, Any], CaptureRadiusPursuit3DEnv, ExecutionDynamicsPursuitWrapper]:
    safety_filter: Any
    if execution_mode == "raw":
        safety_filter = None
    elif execution_mode == "kinematic_cbf":
        safety_filter = make_kinematic_cbf(env)
    elif execution_mode == "execution_aware_cbf":
        safety_filter = ExecutionAwarePursuitCBFSafetyFilter(env, wrapper.execution)
    else:
        raise ValueError(f"Unsupported E1 execution mode: {execution_mode}")

    visible_fractions: list[float] = []
    message_ages: list[float] = []
    observation_ages: list[float] = []
    path_lengths = np.zeros(env.n_defenders, dtype=np.float64)
    previous_positions = env.defender_positions.copy()
    cbf_corrections: list[float] = []
    barrier_values: list[float] = []
    execution_margins: list[float] = []
    final_info: dict[str, Any] = {}
    target_collision = False

    while True:
        action, actor_hidden = action_provider(observation, actor_hidden)
        if safety_filter is None:
            cbf_corrections.append(0.0)
        else:
            action, diagnostics = safety_filter.filter(action, observation)
            cbf_corrections.append(float(diagnostics.action_correction_norm))
            barrier_values.append(float(diagnostics.minimum_barrier_value))
            if hasattr(diagnostics, "mean_execution_margin_m"):
                execution_margins.append(float(diagnostics.mean_execution_margin_m))
        observation, _reward, terminated, truncated, final_info = wrapper.step(action, record_history=True)
        path_lengths += np.linalg.norm(env.defender_positions - previous_positions, axis=1)
        previous_positions = env.defender_positions.copy()
        visible_fractions.append(float(final_info["target_visible_fraction"]))
        message_ages.append(float(final_info["mean_message_age_steps"]))
        observation_ages.append(float(final_info["mean_observation_age_steps"]))
        target_clearance = min(float(env._obstacle_clearance(env.target_position, obstacle)) for obstacle in env.obstacles)
        if target_clearance < 0.0:
            target_collision = True
            break
        if terminated or truncated:
            break

    crossing = crossing_metrics(env, scenario.obstacle_zone_x)
    contract = capture_contract_metrics(
        final_info,
        crossing,
        target_collision=target_collision,
        target_crossing_required=bool(scenario.target_crossing_required),
        required_defender_zone_entries=int(scenario.required_defender_zone_entries),
        require_target_zone_entry=scenario.require_target_zone_entry,
    )
    row = {
        "seed": int(seed),
        "scenario": scenario.name,
        "execution_mode": execution_mode,
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
        "minimum_cbf_barrier_value": float(min(barrier_values)) if barrier_values else None,
        "mean_execution_cbf_margin_m": float(np.mean(execution_margins)) if execution_margins else 0.0,
        "max_execution_cbf_margin_m": float(max(execution_margins)) if execution_margins else 0.0,
        **_execution_metrics(wrapper),
        **crossing,
        **contract,
        **transit_route_metrics(env, scenario),
        **transit_execution_metrics(env, scenario),
    }
    row["rollout_all_defenders_crossed"] = bool(crossing["all_defenders_crossed"])
    row["rollout_target_crossed"] = bool(crossing["target_crossed"])
    row["showcase_success"] = bool(contract["cooperative_safe_capture"])
    return row, env, wrapper


def _execution_metrics(wrapper: ExecutionDynamicsPursuitWrapper) -> dict[str, Any]:
    history = wrapper.execution_history
    if not history:
        return {
            "mean_command_execution_error_mps": 0.0,
            "p95_command_execution_error_mps": 0.0,
            "max_command_execution_error_mps": 0.0,
            "mean_command_age_steps": 0.0,
            "acceleration_saturation_rate": 0.0,
            "speed_saturation_rate": 0.0,
        }
    errors = np.asarray([float(item["mean_command_execution_error_mps"]) for item in history], dtype=np.float64)
    maximum_errors = np.asarray([float(item["max_command_execution_error_mps"]) for item in history], dtype=np.float64)
    acceleration = np.asarray([int(item["acceleration_saturated_defenders"]) for item in history], dtype=np.float64)
    speed = np.asarray([int(item["speed_saturated_defenders"]) for item in history], dtype=np.float64)
    ages = np.asarray([int(item["command_age_steps"]) for item in history], dtype=np.float64)
    denominator = float(len(history) * wrapper.env.n_defenders)
    return {
        "mean_command_execution_error_mps": float(np.mean(errors)),
        "p95_command_execution_error_mps": float(np.quantile(errors, 0.95)),
        "max_command_execution_error_mps": float(np.max(maximum_errors)),
        "mean_command_age_steps": float(np.mean(ages)),
        "acceleration_saturation_rate": float(np.sum(acceleration) / denominator),
        "speed_saturation_rate": float(np.sum(speed) / denominator),
    }


def _execution_seed(seed: int) -> int:
    sequence = np.random.SeedSequence([int(seed), 0xE1_2026])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])
