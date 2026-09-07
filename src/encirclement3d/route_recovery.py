"""Deterministic recovery guards for geometry-conditioned route execution.

This module contains no learned policy and does not relax any CBF row.  It
only detects when the current velocity cannot be stopped inside the remaining
directional clearance and chooses among routes that have already passed the
read-only CBF probe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class StoppingGuardResult:
    """Small, JSON-friendly description of the most urgent stopping hazard."""

    triggered: bool
    risk_type: str
    stopping_distance_m: float
    available_clearance_m: float
    closing_speed_mps: float
    agent_indices: tuple[int, ...]
    reason_code: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "triggered": bool(self.triggered),
            "risk_type": self.risk_type,
            "stopping_distance_m": float(self.stopping_distance_m),
            "available_clearance_m": float(self.available_clearance_m),
            "closing_speed_mps": float(self.closing_speed_mps),
            "agent_indices": list(self.agent_indices),
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class BarrierImminenceResult:
    """One-step CBF look-ahead result used to trigger route recovery.

    This is a diagnostic gate only.  It never changes a CBF row or treats a
    future probe as an executed action.
    """

    triggered: bool
    risk_type: str
    minimum_predicted_slack_m: float
    threshold_m: float
    prediction_steps: int
    solver_accepted: bool
    reason_code: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "triggered": bool(self.triggered),
            "risk_type": self.risk_type,
            "minimum_predicted_slack_m": float(self.minimum_predicted_slack_m),
            "threshold_m": float(self.threshold_m),
            "prediction_steps": int(self.prediction_steps),
            "solver_accepted": bool(self.solver_accepted),
            "reason_code": self.reason_code,
        }


def stopping_distance(speed_mps: float, maximum_deceleration_mps2: float) -> float:
    """Return the constant-deceleration stopping distance in metres."""

    speed = float(speed_mps)
    acceleration = float(maximum_deceleration_mps2)
    if not np.isfinite(speed) or not np.isfinite(acceleration):
        raise ValueError("speed and maximum_deceleration must be finite.")
    if speed < 0.0 or acceleration <= 0.0:
        raise ValueError("speed must be non-negative and deceleration positive.")
    return speed * speed / (2.0 * acceleration)


def _margin(safety_filter: Any, private_name: str, public_name: str) -> float:
    value = getattr(safety_filter, private_name, getattr(safety_filter, public_name, 0.0))
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{private_name} must be finite and non-negative.")
    return result


def compute_stopping_guard(
    observation: Mapping[str, Any],
    env: Any,
    safety_filter: Any,
) -> StoppingGuardResult:
    """Check directional boundary, obstacle, and pairwise stopping feasibility.

    The check is intentionally one-sided: a clearance is hazardous only when
    the measured velocity closes that barrier.  This prevents a large speed in
    a safe tangential direction from unnecessarily forcing a hold command.
    """

    positions = np.asarray(observation.get("defender_positions"), dtype=np.float64)
    velocities = np.asarray(observation.get("defender_velocities"), dtype=np.float64)
    expected = (int(env.n_defenders), 3)
    if positions.shape != expected or velocities.shape != expected:
        raise ValueError("defender positions and velocities must match the environment.")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("defender positions and velocities must be finite.")

    radius = float(env.agents["drone_radius"])
    maximum_deceleration = float(env.agents["defender_max_acceleration"])
    obstacle_margin = _margin(safety_filter, "_obstacle_geometry_margin_m", "obstacle_margin_m")
    boundary_margin = _margin(safety_filter, "_boundary_geometry_margin_m", "boundary_margin_m")
    pairwise_margin = _margin(safety_filter, "_inter_agent_geometry_margin_m", "inter_agent_margin_m")

    hazards: list[tuple[float, float, float, str, tuple[int, ...]]] = []

    def add_hazard(
        clearance: float,
        closing: float,
        deceleration: float,
        risk_type: str,
        agents: tuple[int, ...],
    ) -> None:
        barrier_clearance = max(float(clearance), 0.0)
        closing_speed = max(float(closing), 0.0)
        stop = stopping_distance(closing_speed, deceleration)
        if closing_speed > 1e-6 and stop > barrier_clearance + 1e-9:
            hazards.append((stop - barrier_clearance, stop, barrier_clearance, risk_type, agents))

    lower = np.asarray(env.lower, dtype=np.float64)
    upper = np.asarray(env.upper, dtype=np.float64)
    for index, position in enumerate(positions):
        for axis in range(3):
            lower_normal = np.zeros(3, dtype=np.float64)
            lower_normal[axis] = 1.0
            upper_normal = -lower_normal
            add_hazard(
                position[axis] - lower[axis] - radius - boundary_margin,
                -float(np.dot(velocities[index], lower_normal)),
                maximum_deceleration,
                f"boundary_lower_axis_{axis}",
                (index,),
            )
            add_hazard(
                upper[axis] - position[axis] - radius - boundary_margin,
                -float(np.dot(velocities[index], upper_normal)),
                maximum_deceleration,
                f"boundary_upper_axis_{axis}",
                (index,),
            )

    obstacles = safety_filter._obstacles_from_observation(observation)
    for obstacle_index, obstacle in enumerate(obstacles):
        for index, position in enumerate(positions):
            clearance, normal = env._cylinder_clearance_and_normal(position, obstacle)
            add_hazard(
                clearance - radius - obstacle_margin,
                -float(np.dot(velocities[index], normal)),
                maximum_deceleration,
                f"obstacle_{obstacle_index}",
                (index,),
            )

    for first in range(env.n_defenders):
        for second in range(first + 1, env.n_defenders):
            delta = positions[first] - positions[second]
            distance = float(np.linalg.norm(delta))
            normal = delta / max(distance, 1e-12)
            relative_velocity = velocities[first] - velocities[second]
            add_hazard(
                distance - 2.0 * radius - pairwise_margin,
                -float(np.dot(relative_velocity, normal)),
                2.0 * maximum_deceleration,
                "pairwise",
                (first, second),
            )

    if not hazards:
        return StoppingGuardResult(False, "none", 0.0, float("inf"), 0.0, (), "clear")
    _excess, stop, clearance, risk_type, agents = max(hazards, key=lambda value: (value[0], value[1], value[3]))
    # Recompute the closing speed from the selected stop/deceleration pair so
    # the trace is internally consistent without retaining mutable state.
    deceleration = 2.0 * maximum_deceleration if risk_type == "pairwise" else maximum_deceleration
    closing = float(np.sqrt(max(2.0 * deceleration * stop, 0.0)))
    return StoppingGuardResult(
        True,
        risk_type,
        float(stop),
        float(clearance),
        closing,
        tuple(int(value) for value in agents),
        f"stopping_distance_exceeds_{risk_type}",
    )


def compute_barrier_imminence(
    observation: Mapping[str, Any],
    safety_filter: Any,
    requested_action: np.ndarray,
    *,
    threshold_m: float = 0.05,
    prediction_steps: int = 1,
) -> BarrierImminenceResult:
    """Detect a CBF barrier that will be exhausted on the next step.

    The stopping-distance guard is intentionally directional and can miss a
    curved obstacle normal or a coupled pairwise constraint.  This helper
    instead propagates the *reachable* requested velocity and asks the normal
    read-only CBF verifier for the predicted state.  It is only an
    anticipatory route-recovery trigger; the real action still goes through
    ``filter`` at the current state.
    """

    threshold = float(threshold_m)
    steps = int(prediction_steps)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold_m must be finite and non-negative.")
    if steps <= 0:
        raise ValueError("prediction_steps must be positive.")
    if safety_filter is None or not hasattr(safety_filter, "verify_requested_action"):
        raise ValueError("compute_barrier_imminence requires a CBF verification interface.")
    action = np.asarray(requested_action, dtype=np.float64)
    positions = np.asarray(observation.get("defender_positions"), dtype=np.float64)
    velocities = np.asarray(observation.get("defender_velocities"), dtype=np.float64)
    expected = (int(safety_filter.env.n_defenders), 3)
    if action.shape != expected or positions.shape != expected or velocities.shape != expected:
        raise ValueError("requested action and defender state must match the environment.")
    if not np.isfinite(action).all() or not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("requested action and defender state must be finite.")

    env = safety_filter.env
    current_observation = dict(observation)
    current_positions = positions.copy()
    current_velocities = velocities.copy()
    last_diagnostics: Any | None = None
    for _step in range(steps):
        reachable = safety_filter._reachable_reference(current_velocities, action)
        current_velocities = reachable
        current_positions = current_positions + current_velocities * float(env.dt)
        future = dict(current_observation)
        future["defender_positions"] = current_positions
        future["defender_velocities"] = current_velocities
        last_diagnostics = safety_filter.verify_requested_action(action, future)
        current_observation = future

    if last_diagnostics is None:
        return BarrierImminenceResult(False, "none", float("inf"), threshold, steps, True, "clear")
    slacks = dict(getattr(last_diagnostics, "constraint_slacks", {}))
    geometric = {
        str(name): float(value)
        for name, value in slacks.items()
        if (
            str(name).startswith(("obstacle_", "boundary_", "altitude_", "pairwise_"))
            and np.isfinite(float(value))
        )
    }
    if not geometric:
        return BarrierImminenceResult(
            False,
            "none",
            float("inf"),
            threshold,
            steps,
            bool(getattr(last_diagnostics, "verified_feasible", False)),
            "no_geometric_constraints",
        )
    risk_name, minimum = min(geometric.items(), key=lambda item: (item[1], item[0]))
    if risk_name.startswith("obstacle_"):
        risk_type = "obstacle"
    elif risk_name.startswith(("boundary_", "altitude_")):
        risk_type = "boundary"
    else:
        risk_type = "pairwise"
    triggered = bool(minimum <= threshold)
    accepted = bool(getattr(last_diagnostics, "verified_feasible", False))
    reason = "barrier_imminent" if triggered else "clear"
    if not accepted and minimum < 0.0:
        reason = "predicted_cbf_infeasible"
        triggered = True
    return BarrierImminenceResult(
        triggered,
        risk_type if triggered else "none",
        float(minimum),
        threshold,
        steps,
        accepted,
        reason,
    )


def select_verified_progress_route(
    route_batch: Any,
    counterfactuals: Sequence[Any],
    observation: Mapping[str, Any],
    *,
    require_detour: bool = False,
    preferred_route_id: str | None = None,
    hold_steps_remaining: int = 0,
    preferred_route_side: str | None = None,
    nearest_tangent_route: bool = False,
    tangent_switch_tolerance_m: float = 0.25,
) -> tuple[int | None, str]:
    """Select the most useful route from independently verified candidates.

    ``verified_safe_hold`` is never considered progress.  A stopping guard can
    request a detour; in that case nominal is excluded even when it is valid.
    The deterministic key prefers target-directed first-step progress and then
    shorter geometric routes.
    """

    if int(hold_steps_remaining) < 0:
        raise ValueError("hold_steps_remaining must be non-negative.")
    if not np.isfinite(float(tangent_switch_tolerance_m)) or tangent_switch_tolerance_m < 0.0:
        raise ValueError("tangent_switch_tolerance_m must be finite and non-negative.")
    positions = np.asarray(observation.get("defender_positions"), dtype=np.float64)
    beliefs = np.asarray(observation.get("target_belief_positions"), dtype=np.float64)
    if positions.ndim != 2 or beliefs.shape != positions.shape or not np.isfinite(positions).all() or not np.isfinite(beliefs).all():
        raise ValueError("positions and target beliefs must be finite and shape-compatible.")
    target = beliefs.mean(axis=0)
    directions = target[None, :] - positions
    direction_norm = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(direction_norm, 1e-12)
    progress_labels = {
        "left_detour",
        "right_detour",
        "upper_detour",
        "lower_detour",
        "radial_out",
        "formation_split",
        "safe_intercept",
        "visibility_hold",
    }
    # Recovery used to re-rank independently verified routes on every step,
    # which caused route oscillation even when the current route remained
    # safe.  Keep the current route during its registered hold window.  A
    # route that disappeared or failed the fresh probe cannot be held.
    if preferred_route_id is not None and int(hold_steps_remaining) > 0:
        for index, candidate in enumerate(route_batch.candidates):
            probe = counterfactuals[index] if index < len(counterfactuals) else None
            if (
                str(getattr(candidate, "route_id", "")) == str(preferred_route_id)
                and bool(getattr(candidate, "valid", True))
                and probe is not None
                and bool(getattr(probe, "accepted", False))
                and str(candidate.label) not in {"braking", "verified_safe_hold"}
                and (not require_detour or str(candidate.label) not in {"nominal", "formation_contract"})
            ):
                return int(index), "verified_progress_route_hold"

    if nearest_tangent_route:
        tangent_indices: list[int] = []
        for index, candidate in enumerate(route_batch.candidates):
            probe = counterfactuals[index] if index < len(counterfactuals) else None
            if (
                str(getattr(candidate, "label", "")) in {"left_detour", "right_detour"}
                and bool(getattr(candidate, "valid", True))
                and probe is not None
                and bool(getattr(probe, "accepted", False))
            ):
                tangent_indices.append(index)
        if tangent_indices:
            shortest_length = min(
                float(route_batch.candidates[index].route_length_m) for index in tangent_indices
            )
            shortest = [
                index
                for index in tangent_indices
                if float(route_batch.candidates[index].route_length_m)
                <= shortest_length + float(tangent_switch_tolerance_m)
            ]
            preferred = [
                index
                for index in shortest
                if preferred_route_side is not None
                and str(getattr(route_batch.candidates[index], "side", "")) == str(preferred_route_side)
            ]
            pool = preferred or shortest
            selected = min(
                pool,
                key=lambda index: (
                    float(route_batch.candidates[index].route_length_m),
                    -float(
                        np.mean(
                            np.sum(
                                np.asarray(route_batch.candidates[index].action_chunk[0], dtype=np.float64)
                                * directions,
                                axis=1,
                            )
                        )
                    ),
                    int(index),
                ),
            )
            return int(selected), (
                "nearest_tangent_route_hold"
                if preferred
                else "nearest_tangent_route"
            )
    scored: list[tuple[float, float, int]] = []
    fallback_scored: list[tuple[float, int]] = []
    for index, candidate in enumerate(route_batch.candidates):
        probe = counterfactuals[index] if index < len(counterfactuals) else None
        if (
            probe is None
            or not bool(getattr(candidate, "valid", True))
            or not bool(getattr(probe, "accepted", False))
        ):
            continue
        label = str(candidate.label)
        if label in {"braking", "verified_safe_hold"}:
            fallback_scored.append((float(candidate.route_length_m), index))
            continue
        if require_detour and label in {"nominal", "formation_contract"}:
            continue
        first_action = np.asarray(candidate.action_chunk[0], dtype=np.float64)
        progress = float(np.mean(np.sum(first_action * directions, axis=1)))
        if label in progress_labels or (not require_detour and label == "nominal"):
            scored.append((-progress, float(candidate.route_length_m), index))
    if scored:
        scored.sort(key=lambda value: (value[0], value[1], value[2]))
        return int(scored[0][2]), "verified_progress_route"
    if fallback_scored:
        fallback_scored.sort(key=lambda value: (value[0], value[1]))
        return int(fallback_scored[0][1]), "verified_braking_route"
    return None, "no_verified_progress_route"


__all__ = [
    "BarrierImminenceResult",
    "StoppingGuardResult",
    "stopping_distance",
    "compute_stopping_guard",
    "compute_barrier_imminence",
    "select_verified_progress_route",
]
