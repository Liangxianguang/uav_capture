"""Development-only distributed minimax route planner.

This module is deliberately separate from the historical JEPA/CBF evaluators.
It scores already projected route candidates against a bounded target escape set
and returns a route decision.  It never executes an action and never replaces
the downstream Joint CBF safety boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}.")
    return array.copy()


@dataclass(frozen=True)
class DNMPCConfig:
    """Bounded minimax objective and route-continuity contract."""

    horizon_steps: int = 5
    dt_seconds: float = 0.1
    target_escape_speed_mps: float = 1.0
    target_escape_acceleration_mps2: float = 1.0
    capture_weight: float = 1.0
    worst_case_escape_weight: float = 1.0
    formation_weight: float = 0.15
    smoothness_weight: float = 0.05
    route_switch_penalty_m: float = 0.15
    switch_improvement_m: float = 0.25
    minimum_hold_steps: int = 2
    max_route_age_steps: int = 50
    terminal_progress_weight: float = 0.25
    stopping_distance_weight: float = 0.15
    stopping_acceleration_mps2: float = 6.0

    def __post_init__(self) -> None:
        if int(self.horizon_steps) <= 0:
            raise ValueError("horizon_steps must be positive.")
        for name in (
            "dt_seconds",
            "target_escape_speed_mps",
            "target_escape_acceleration_mps2",
            "capture_weight",
            "worst_case_escape_weight",
            "formation_weight",
            "smoothness_weight",
            "route_switch_penalty_m",
            "switch_improvement_m",
            "terminal_progress_weight",
            "stopping_distance_weight",
            "stopping_acceleration_mps2",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0 or (name == "dt_seconds" and value <= 0.0):
                raise ValueError(f"{name} must be finite and non-negative.")
        if int(self.minimum_hold_steps) < 0 or int(self.max_route_age_steps) <= 0:
            raise ValueError("minimum_hold_steps must be non-negative and max_route_age_steps positive.")
        if float(self.stopping_acceleration_mps2) <= 0.0:
            raise ValueError("stopping_acceleration_mps2 must be positive.")


@dataclass(frozen=True)
class DNMPCDecision:
    """Planner output; the action remains subject to independent CBF checks."""

    selected_index: int | None
    selected_route_id: str | None
    selected_side: str | None
    reason: str
    switched: bool
    route_age_steps: int
    scores: tuple[float, ...]
    worst_case_escape_costs: tuple[float, ...]
    capture_costs: tuple[float, ...]
    formation_costs: tuple[float, ...]
    smoothness_costs: tuple[float, ...]
    local_agent_costs: tuple[tuple[float, ...], ...]
    escape_hypotheses: tuple[tuple[float, float, float], ...]
    route_phase: str = "approach"
    active_obstacle_id: int | None = None
    terminal_progress_costs: tuple[float, ...] = ()
    stopping_distance_costs: tuple[float, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_index": self.selected_index,
            "selected_route_id": self.selected_route_id,
            "selected_side": self.selected_side,
            "reason": self.reason,
            "switched": bool(self.switched),
            "route_age_steps": int(self.route_age_steps),
            "scores": list(self.scores),
            "worst_case_escape_costs": list(self.worst_case_escape_costs),
            "capture_costs": list(self.capture_costs),
            "formation_costs": list(self.formation_costs),
            "smoothness_costs": list(self.smoothness_costs),
            "local_agent_costs": [list(row) for row in self.local_agent_costs],
            "escape_hypotheses": [list(row) for row in self.escape_hypotheses],
            "route_phase": self.route_phase,
            "active_obstacle_id": self.active_obstacle_id,
            "terminal_progress_costs": list(self.terminal_progress_costs),
            "stopping_distance_costs": list(self.stopping_distance_costs),
        }


class DistributedMinimaxMPC:
    """Score route candidates under a finite target escape set.

    The planner is distributed in the sense that it exposes per-agent local
    costs and a shared route score.  The first implementation uses a bounded,
    deterministic candidate set so it can be audited before adding an iterative
    neurodynamic solver.  It consumes public target beliefs only.
    """

    def __init__(self, config: DNMPCConfig | None = None) -> None:
        self.config = config or DNMPCConfig()
        self._route_id: str | None = None
        self._route_side: str | None = None
        self._route_age = 0

    @property
    def route_id(self) -> str | None:
        return self._route_id

    def reset(self) -> None:
        self._route_id = None
        self._route_side = None
        self._route_age = 0

    def _belief(self, observation: Mapping[str, Any], defender_count: int) -> tuple[np.ndarray, np.ndarray]:
        positions = _finite_array(observation.get("defender_positions"), (defender_count, 3), "defender_positions")
        beliefs = _finite_array(observation.get("target_belief_positions"), (defender_count, 3), "target_belief_positions")
        velocities = _finite_array(
            observation.get("target_belief_velocities", np.zeros((defender_count, 3))),
            (defender_count, 3),
            "target_belief_velocities",
        )
        received_value = observation.get("target_observation_received")
        if received_value is None:
            received = np.ones(defender_count, dtype=bool)
        else:
            received = np.asarray(received_value, dtype=bool)
            if received.shape != (defender_count,):
                raise ValueError("target_observation_received must match defender count.")
        if not bool(received.any()):
            return positions.mean(axis=0), np.zeros(3, dtype=np.float64)
        return beliefs[received].mean(axis=0), velocities[received].mean(axis=0)

    def _escape_hypotheses(
        self,
        observation: Mapping[str, Any],
        target: np.ndarray,
        target_velocity: np.ndarray,
        centroid: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        direction = target - centroid
        norm = float(np.linalg.norm(direction))
        radial = direction / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        lateral = np.array([-radial[1], radial[0], 0.0], dtype=np.float64)
        if float(np.linalg.norm(lateral)) <= 1e-9:
            lateral = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        lateral /= float(np.linalg.norm(lateral))
        vertical = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        # Include the public observed direction and bounded adversarial
        # radial/lateral/vertical alternatives.  No simulator future state is
        # consulted here.
        observed = target_velocity.copy()
        observed_norm = float(np.linalg.norm(observed))
        observed_dir = observed / observed_norm if observed_norm > 1e-9 else radial
        directions = (observed_dir, radial, lateral, -lateral, vertical, -vertical)
        hypotheses: list[np.ndarray] = []
        for item in directions:
            unit = item / max(float(np.linalg.norm(item)), 1e-9)
            hypotheses.append(
                target_velocity + unit * float(self.config.target_escape_speed_mps)
            )
        return tuple(hypotheses)

    @staticmethod
    def _pairwise_matrix(positions: np.ndarray) -> np.ndarray:
        delta = positions[:, None, :] - positions[None, :, :]
        return np.linalg.norm(delta, axis=-1)

    def _score_candidate(
        self,
        candidate: Any,
        observation: Mapping[str, Any],
        previous_action: np.ndarray,
        escape_hypotheses: tuple[np.ndarray, ...],
    ) -> tuple[float, float, float, float, float, float, float, tuple[float, ...]]:
        positions = _finite_array(observation.get("defender_positions"), previous_action.shape, "defender_positions")
        actions = np.asarray(candidate.action_chunk, dtype=np.float64)
        if actions.ndim != 3 or actions.shape[1:] != positions.shape or not np.isfinite(actions).all():
            raise ValueError("candidate.action_chunk must be finite with shape [steps, defenders, 3].")
        target, _ = self._belief(observation, positions.shape[0])
        predictions = []
        current = positions.copy()
        for action in actions[: int(self.config.horizon_steps)]:
            current = current + action * float(self.config.dt_seconds)
            predictions.append(current.copy())
        if not predictions:
            raise ValueError("candidate.action_chunk must contain at least one step.")
        defender_rollout = np.stack(predictions, axis=0)
        horizon = defender_rollout.shape[0]
        initial_nearest = float(np.min(np.linalg.norm(positions - target[None, :], axis=-1)))
        escape_distances: list[float] = []
        local_costs_by_hypothesis: list[np.ndarray] = []
        for escape_velocity in escape_hypotheses:
            escape_direction = escape_velocity / max(float(np.linalg.norm(escape_velocity)), 1e-9)
            target_rollout = np.stack(
                [
                    target
                    + (step + 1) * float(self.config.dt_seconds) * escape_velocity
                    + 0.5
                    * ((step + 1) * float(self.config.dt_seconds)) ** 2
                    * float(self.config.target_escape_acceleration_mps2)
                    * escape_direction
                    for step in range(horizon)
                ],
                axis=0,
            )
            distances = np.linalg.norm(defender_rollout - target_rollout[:, None, :], axis=-1)
            nearest = np.min(distances, axis=1)
            escape_distances.append(float(nearest[-1]))
            local_costs_by_hypothesis.append(distances[-1])
        capture_cost = float(np.mean(escape_distances))
        worst_case_escape = float(np.max(escape_distances))
        final_nearest = float(np.min(np.linalg.norm(defender_rollout[-1] - target[None, :], axis=-1)))
        # A negative cost is an explicit reward for closing the terminal gap.
        # Keeping it separate in the trace makes the progress contribution
        # auditable instead of hiding it inside the capture distance.
        terminal_progress_cost = -max(0.0, initial_nearest - final_nearest)
        local = tuple(
            float(value)
            for value in np.max(np.stack(local_costs_by_hypothesis, axis=0), axis=0)
        )
        baseline_pairwise = self._pairwise_matrix(positions)
        formation_errors = []
        for frame in defender_rollout:
            formation_errors.append(float(np.mean(np.abs(self._pairwise_matrix(frame) - baseline_pairwise))))
        formation_cost = float(np.mean(formation_errors))
        action_delta = actions[0] - previous_action
        if actions.shape[0] > 1:
            action_delta = np.concatenate((action_delta.reshape(1, *action_delta.shape), np.diff(actions, axis=0)), axis=0)
        smoothness_cost = float(np.mean(np.linalg.norm(action_delta, axis=-1)))
        speeds = np.linalg.norm(actions[:horizon], axis=-1)
        stopping_distance = np.square(speeds) / (2.0 * float(self.config.stopping_acceleration_mps2))
        route_clearance = float(getattr(candidate, "minimum_geometric_clearance_m", np.inf))
        stopping_distance_cost = float(
            np.mean(np.maximum(0.0, stopping_distance - max(route_clearance, 0.0)))
        )
        switch_cost = (
            float(self.config.route_switch_penalty_m)
            if self._route_id is not None and str(candidate.route_id) != self._route_id
            else 0.0
        )
        score = (
            float(self.config.capture_weight) * capture_cost
            + float(self.config.worst_case_escape_weight) * worst_case_escape
            + float(self.config.formation_weight) * formation_cost
            + float(self.config.smoothness_weight) * smoothness_cost
            + float(self.config.terminal_progress_weight) * terminal_progress_cost
            + float(self.config.stopping_distance_weight) * stopping_distance_cost
            + switch_cost
        )
        return (
            score,
            worst_case_escape,
            capture_cost,
            formation_cost,
            smoothness_cost,
            terminal_progress_cost,
            stopping_distance_cost,
            local,
        )

    @staticmethod
    def _route_phase(candidate: Any) -> str:
        label = str(getattr(candidate, "label", "nominal"))
        if label in {"braking", "radial_out"}:
            return "pre_brake"
        if label in {"left_detour", "right_detour", "upper_detour", "lower_detour"}:
            return "tangent_left" if label == "left_detour" else "tangent_right" if label == "right_detour" else "tangent_vertical"
        if label in {"formation_split", "formation_contract"}:
            return "encircle"
        if label in {"safe_intercept", "visibility_hold"}:
            return "intercept"
        if label == "verified_safe_hold":
            return "safe_hold"
        return "approach"

    def plan(
        self,
        route_batch: Any,
        observation: Mapping[str, Any],
        *,
        previous_action: np.ndarray | None = None,
    ) -> DNMPCDecision:
        candidates = tuple(getattr(route_batch, "candidates", ()))
        if not candidates:
            raise ValueError("route_batch must contain candidates.")
        first_action = np.asarray(candidates[0].action_chunk, dtype=np.float64)[0]
        previous = np.zeros_like(first_action) if previous_action is None else _finite_array(previous_action, first_action.shape, "previous_action")
        positions = _finite_array(observation.get("defender_positions"), first_action.shape, "defender_positions")
        target, target_velocity = self._belief(observation, positions.shape[0])
        hypotheses = self._escape_hypotheses(observation, target, target_velocity, positions.mean(axis=0))
        scores = np.full(len(candidates), np.inf, dtype=np.float64)
        worst = np.full(len(candidates), np.inf, dtype=np.float64)
        capture = np.full(len(candidates), np.inf, dtype=np.float64)
        formation = np.full(len(candidates), np.inf, dtype=np.float64)
        smoothness = np.full(len(candidates), np.inf, dtype=np.float64)
        terminal_progress = np.full(len(candidates), np.inf, dtype=np.float64)
        stopping_distance = np.full(len(candidates), np.inf, dtype=np.float64)
        local_rows: list[tuple[float, ...]] = [tuple() for _ in candidates]
        for index, candidate in enumerate(candidates):
            if not bool(getattr(candidate, "valid", True)):
                continue
            score, worst_value, capture_value, formation_value, smoothness_value, progress_value, stopping_value, local = self._score_candidate(
                candidate, observation, previous, hypotheses
            )
            scores[index] = score
            worst[index] = worst_value
            capture[index] = capture_value
            formation[index] = formation_value
            smoothness[index] = smoothness_value
            terminal_progress[index] = progress_value
            stopping_distance[index] = stopping_value
            local_rows[index] = local
        valid_indices = np.flatnonzero(np.isfinite(scores))
        if valid_indices.size == 0:
            self._route_id = None
            self._route_side = None
            self._route_age = 0
            return DNMPCDecision(
                selected_index=None,
                selected_route_id=None,
                selected_side=None,
                reason="no_valid_candidate",
                switched=False,
                route_age_steps=0,
                scores=tuple(float(value) for value in scores),
                worst_case_escape_costs=tuple(float(value) for value in worst),
                capture_costs=tuple(float(value) for value in capture),
                formation_costs=tuple(float(value) for value in formation),
                smoothness_costs=tuple(float(value) for value in smoothness),
                local_agent_costs=tuple(local_rows),
                escape_hypotheses=tuple(tuple(float(value) for value in row) for row in hypotheses),
                terminal_progress_costs=tuple(float(value) for value in terminal_progress),
                stopping_distance_costs=tuple(float(value) for value in stopping_distance),
            )
        best_index = int(valid_indices[np.argmin(scores[valid_indices])])
        current_index = next(
            (index for index in valid_indices if str(getattr(candidates[index], "route_id", "")) == self._route_id),
            None,
        )
        selected_index = best_index
        reason = "minimax_best_route"
        if current_index is not None and int(self._route_age) < int(self.config.max_route_age_steps):
            improvement = float(scores[current_index] - scores[best_index])
            if int(self._route_age) < int(self.config.minimum_hold_steps):
                selected_index = int(current_index)
                reason = "route_hold_minimum_age"
            elif improvement < float(self.config.switch_improvement_m):
                selected_index = int(current_index)
                reason = "route_hold_hysteresis"
        selected = candidates[selected_index]
        switched = self._route_id is not None and str(selected.route_id) != self._route_id
        if switched:
            self._route_age = 0
        else:
            self._route_age += 1
        self._route_id = str(selected.route_id)
        self._route_side = str(getattr(selected, "side", ""))
        return DNMPCDecision(
            selected_index=int(selected_index),
            selected_route_id=self._route_id,
            selected_side=self._route_side,
            reason=reason,
            switched=bool(switched),
            route_age_steps=int(self._route_age),
            scores=tuple(float(value) for value in scores),
            worst_case_escape_costs=tuple(float(value) for value in worst),
            capture_costs=tuple(float(value) for value in capture),
            formation_costs=tuple(float(value) for value in formation),
            smoothness_costs=tuple(float(value) for value in smoothness),
            local_agent_costs=tuple(local_rows),
            escape_hypotheses=tuple(tuple(float(value) for value in row) for row in hypotheses),
            route_phase=self._route_phase(selected),
            active_obstacle_id=getattr(selected, "obstacle_id", None),
            terminal_progress_costs=tuple(float(value) for value in terminal_progress),
            stopping_distance_costs=tuple(float(value) for value in stopping_distance),
        )


__all__ = ["DNMPCConfig", "DNMPCDecision", "DistributedMinimaxMPC"]
