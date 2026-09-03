"""Deterministic joint multi-agent CBF-QP safety filter.

The optimizer minimizes a quadratic deviation from the requested velocity while
enforcing all defender constraints in one solve.  CBF rows are linearized at
the current observed positions; speed and acceleration are enforced as exact
convex norm constraints.  SciPy SLSQP is used because the RTX 5050 Conda
environment does not ship OSQP/CVXPy.  The API reports solver status and
post-solve residuals explicitly, so a failed solve can never silently execute
the unfiltered request.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize

from .pursuit_env import CaptureRadiusPursuit3DEnv, CylinderObstacle, _unit


@dataclass(frozen=True)
class _LinearBarrier:
    name: str
    row: np.ndarray
    lower_bound: float


@dataclass(frozen=True)
class JointCBFQPDiagnostics:
    """Auditable result of one joint solve and any fallback."""

    solver: str
    solver_status: str
    solver_message: str
    solver_success: bool
    infeasible: bool
    timed_out: bool
    used_fallback: bool
    fallback_mode: str
    verified_feasible: bool
    requested_action_finite: bool
    action_correction_norm: float
    minimum_constraint_value: float
    objective_value: float
    solve_latency_ms: float
    constraint_slacks: dict[str, float]
    active_constraints: tuple[str, ...]
    task_constraint_slacks: dict[str, float]
    state_safety_violation: bool = False
    minimum_state_clearance: float = float("inf")

    @property
    def minimum_barrier_value(self) -> float:
        """Compatibility alias used by existing evaluation summaries."""

        return self.minimum_constraint_value


@dataclass(frozen=True)
class _SolveResult:
    action: np.ndarray | None
    success: bool
    verified_feasible: bool
    timed_out: bool
    status: str
    message: str
    objective_value: float
    latency_ms: float
    minimum_constraint_value: float
    slacks: dict[str, float]
    active_constraints: tuple[str, ...]


class JointCBFQPSafetyFilter:
    """Solve one coupled CBF-QP for all defenders.

    ``filter`` returns a desired velocity that is either post-solve verified or
    an explicitly labelled fallback.  It never returns the original request
    after a failed/non-finite solve.
    """

    SOLVER_NAME = "scipy_slsqp_joint_cbf_qp"

    def __init__(
        self,
        env: CaptureRadiusPursuit3DEnv,
        *,
        gamma: float | None = None,
        obstacle_margin_m: float | None = None,
        inter_agent_margin_m: float | None = None,
        boundary_margin_m: float | None = None,
        max_correction_norm_mps: float = 5.0,
        max_latency_ms: float = 100.0,
        solver_maxiter: int = 80,
        tolerance: float = 1e-5,
        active_tolerance: float = 5e-5,
        anticipatory_horizon_steps: int = 3,
    ) -> None:
        self.env = env
        self.gamma = float(gamma if gamma is not None else env.task.get("cbf_gamma", 0.25))
        self.obstacle_margin_m = float(
            obstacle_margin_m if obstacle_margin_m is not None else env.pursuit.get("safety_margin", 0.35)
        )
        self.inter_agent_margin_m = float(
            inter_agent_margin_m if inter_agent_margin_m is not None else env.pursuit.get("safety_margin", 0.35)
        )
        self.boundary_margin_m = float(boundary_margin_m if boundary_margin_m is not None else self.obstacle_margin_m)
        self.max_correction_norm_mps = float(max_correction_norm_mps)
        self.max_latency_ms = float(max_latency_ms)
        self.solver_maxiter = int(solver_maxiter)
        self.tolerance = float(tolerance)
        self.active_tolerance = float(active_tolerance)
        self.anticipatory_horizon_steps = int(anticipatory_horizon_steps)
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1].")
        if min(self.obstacle_margin_m, self.inter_agent_margin_m, self.boundary_margin_m) < 0.0:
            raise ValueError("CBF margins must be non-negative.")
        if self.max_correction_norm_mps <= 0.0 or self.max_latency_ms <= 0.0:
            raise ValueError("max correction and latency limits must be positive.")
        if self.solver_maxiter <= 0 or self.tolerance <= 0.0 or self.active_tolerance <= 0.0:
            raise ValueError("solver_maxiter and tolerances must be positive.")
        if self.anticipatory_horizon_steps < 0:
            raise ValueError("anticipatory_horizon_steps must be non-negative.")

    @property
    def contract(self) -> dict[str, float | int | str]:
        """Return the immutable solver parameters for provenance logging."""

        return {
            "solver": self.SOLVER_NAME,
            "gamma": float(self.gamma),
            "obstacle_margin_m": float(self.obstacle_margin_m),
            "inter_agent_margin_m": float(self.inter_agent_margin_m),
            "boundary_margin_m": float(self.boundary_margin_m),
            "max_correction_norm_mps": float(self.max_correction_norm_mps),
            "max_latency_ms": float(self.max_latency_ms),
            "solver_maxiter": int(self.solver_maxiter),
            "tolerance": float(self.tolerance),
            "active_tolerance": float(self.active_tolerance),
            "anticipatory_horizon_steps": int(self.anticipatory_horizon_steps),
        }

    def filter(
        self,
        desired_actions: np.ndarray,
        observation: Mapping[str, Any],
        *,
        nominal_actions: np.ndarray | None = None,
        execution_mode: str = "normal",
    ) -> tuple[np.ndarray, JointCBFQPDiagnostics]:
        desired = np.asarray(desired_actions, dtype=np.float64)
        expected = (self.env.n_defenders, 3)
        current = np.asarray(observation["defender_velocities"], dtype=np.float64)
        if current.shape != expected or not np.isfinite(current).all():
            raise ValueError("Observed defender velocities must be finite and match the defender count.")
        if desired.shape != expected:
            raise ValueError(f"desired_actions must have shape {expected}, got {desired.shape}.")
        if execution_mode not in {"normal", "safe_hold"}:
            raise ValueError("execution_mode must be 'normal' or 'safe_hold'.")
        records = self._build_barriers(observation)
        task_slacks = self._task_constraint_slacks(observation, current)
        minimum_state_clearance, state_violations = self._state_safety(observation)
        if state_violations:
            return self._controlled_abort(
                current,
                desired,
                records,
                task_slacks,
                status="state_safety_violation",
                message="; ".join(state_violations),
                requested_action_finite=bool(np.isfinite(desired).all()),
                state_safety_violation=True,
                minimum_state_clearance=minimum_state_clearance,
            )
        if not np.isfinite(desired).all():
            return self._controlled_abort(
                current,
                desired,
                records,
                task_slacks,
                status="nonfinite_request",
                message="desired action contains non-finite values",
                requested_action_finite=False,
                minimum_state_clearance=minimum_state_clearance,
            )

        if execution_mode == "safe_hold":
            requested = current.copy()
        else:
            requested = desired.copy()
        reference = self._reachable_reference(current, requested)
        primary = self._solve(reference, current, records)
        if primary.verified_feasible and not primary.timed_out and self._correction_ok(primary.action, desired):
            return primary.action, self._diagnostics(
                primary,
                desired,
                requested_action_finite=True,
                used_fallback=execution_mode == "safe_hold",
                fallback_mode="safe_hold" if execution_mode == "safe_hold" else "none",
                infeasible=False,
                task_slacks=task_slacks,
                minimum_state_clearance=minimum_state_clearance,
            )

        # First fallback: solve the same constraints around a hold reference.
        hold = self._solve(self._reachable_reference(current, current), current, records)
        if hold.verified_feasible and not hold.timed_out and self._correction_ok(hold.action, current):
            return hold.action, self._diagnostics(
                hold,
                desired,
                requested_action_finite=True,
                used_fallback=True,
                fallback_mode="safe_hold",
                infeasible=True,
                task_slacks=task_slacks,
                primary=primary,
                minimum_state_clearance=minimum_state_clearance,
            )

        # Second fallback: if a separate nominal action was supplied, route it
        # through this same filter rather than executing it directly.
        if nominal_actions is not None:
            nominal = np.asarray(nominal_actions, dtype=np.float64)
            if nominal.shape != expected or not np.isfinite(nominal).all():
                nominal = current.copy()
            nominal_result = self._solve(self._reachable_reference(current, nominal), current, records)
            if nominal_result.verified_feasible and not nominal_result.timed_out and self._correction_ok(nominal_result.action, nominal):
                return nominal_result.action, self._diagnostics(
                    nominal_result,
                    desired,
                    requested_action_finite=True,
                    used_fallback=True,
                    fallback_mode="nominal_cbf",
                    infeasible=True,
                    task_slacks=task_slacks,
                    primary=primary,
                    minimum_state_clearance=minimum_state_clearance,
                )

        # Final fallback is deliberately explicit and never claims a safety
        # proof. The current velocity avoids a discontinuous command while the
        # diagnostic tells the caller to hover/abort and records infeasibility.
        return self._controlled_abort(
            current,
            desired,
            records,
            task_slacks,
            status="controlled_abort",
            message=f"primary={primary.status}; hold={hold.status}",
            requested_action_finite=True,
            primary=primary,
            minimum_state_clearance=minimum_state_clearance,
        )

    def _correction_ok(self, action: np.ndarray | None, requested: np.ndarray) -> bool:
        if action is None or not np.isfinite(action).all():
            return False
        correction = float(np.max(np.linalg.norm(action - requested, axis=1)))
        return correction <= self.max_correction_norm_mps + self.tolerance

    def _emergency_action(self, current: np.ndarray) -> np.ndarray:
        """Return a finite, motion-limited stop request for the abort path."""

        max_delta = float(self.env.agents["defender_max_acceleration"]) * self.env.dt
        action = self.env._move_toward_velocity(current, np.zeros_like(current), max_delta)
        return self.env._clip_rows(action, float(self.env.agents["defender_max_speed"]))

    def _reachable_reference(self, current: np.ndarray, requested: np.ndarray) -> np.ndarray:
        return self.env._move_toward_velocity(
            current,
            self.env._clip_rows(np.asarray(requested, dtype=np.float64), float(self.env.agents["defender_max_speed"])),
            max_delta=float(self.env.agents["defender_max_acceleration"]) * self.env.dt,
        )

    def _build_barriers(self, observation: Mapping[str, Any]) -> list[_LinearBarrier]:
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        expected = (self.env.n_defenders, 3)
        if positions.shape != expected or not np.isfinite(positions).all():
            raise ValueError(f"defender_positions must be finite with shape {expected}.")
        rows: list[_LinearBarrier] = []
        obstacles = self._obstacles_from_observation(observation)

        def add_single(name: str, defender: int, normal: np.ndarray, lower_bound: float) -> None:
            row = np.zeros(self.env.n_defenders * 3, dtype=np.float64)
            row[3 * defender : 3 * defender + 3] = np.asarray(normal, dtype=np.float64)
            rows.append(_LinearBarrier(name, row, float(lower_bound)))

        radius = float(self.env.agents["drone_radius"])
        max_accel = float(self.env.agents["defender_max_acceleration"])
        for defender, position in enumerate(positions):
            for obstacle_index, obstacle in enumerate(obstacles):
                clearance, normal = self.env._cylinder_clearance_and_normal(position, obstacle)
                if not np.isfinite(clearance) or not np.isfinite(normal).all():
                    raise ValueError(f"Obstacle {obstacle_index} produced non-finite geometry.")
                barrier = float(clearance) - radius - self.obstacle_margin_m
                add_single(
                    f"obstacle_{obstacle_index}_defender_{defender}",
                    defender,
                    normal,
                    self._closing_speed_bound(barrier, max_accel),
                )
            for axis in range(3):
                lower_barrier = float(position[axis] - self.env.lower[axis] - radius - self.boundary_margin_m)
                lower_normal = np.zeros(3, dtype=np.float64)
                lower_normal[axis] = 1.0
                upper_barrier = float(self.env.upper[axis] - radius - self.boundary_margin_m - position[axis])
                upper_normal = np.zeros(3, dtype=np.float64)
                upper_normal[axis] = -1.0
                prefix = "altitude" if axis == 2 else "boundary"
                add_single(
                    f"{prefix}_lower_defender_{defender}_axis_{axis}",
                    defender,
                    lower_normal,
                    self._closing_speed_bound(lower_barrier, max_accel),
                )
                add_single(
                    f"{prefix}_upper_defender_{defender}_axis_{axis}",
                    defender,
                    upper_normal,
                    self._closing_speed_bound(upper_barrier, max_accel),
                )

        for first in range(self.env.n_defenders):
            for second in range(first + 1, self.env.n_defenders):
                delta = positions[first] - positions[second]
                distance = float(np.linalg.norm(delta))
                normal = _unit(delta, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64))
                barrier = distance - (2.0 * radius + self.inter_agent_margin_m)
                row = np.zeros(self.env.n_defenders * 3, dtype=np.float64)
                row[3 * first : 3 * first + 3] = normal
                row[3 * second : 3 * second + 3] = -normal
                rows.append(
                    _LinearBarrier(
                        f"pairwise_{first}_{second}",
                        row,
                        self._closing_speed_bound(barrier, 2.0 * max_accel),
                    )
                )
        return rows

    def _obstacles_from_observation(self, observation: Mapping[str, Any]) -> list[CylinderObstacle]:
        """Decode public obstacle geometry without reading target ground truth."""

        raw_obstacles = observation.get("obstacles", self.env.obstacles)
        obstacles: list[CylinderObstacle] = []
        for index, raw in enumerate(raw_obstacles):
            if isinstance(raw, CylinderObstacle):
                obstacle = raw
            elif isinstance(raw, Mapping):
                center = np.asarray(raw.get("center_xy"), dtype=np.float64)
                half_raw = raw.get("half_extents_xy")
                half = None if half_raw is None else np.asarray(half_raw, dtype=np.float64)
                obstacle = CylinderObstacle(
                    center_xy=center,
                    radius=float(raw.get("radius")),
                    height=float(raw.get("height")),
                    shape=str(raw.get("shape", "cylinder")),
                    half_extents_xy=half,
                )
            else:
                raise ValueError(f"Unsupported obstacle record at index {index}.")
            if obstacle.center_xy.shape != (2,) or not np.isfinite(obstacle.center_xy).all():
                raise ValueError(f"Obstacle {index} center must be finite with shape (2,).")
            if not np.isfinite(obstacle.radius) or obstacle.radius <= 0.0:
                raise ValueError(f"Obstacle {index} radius must be positive and finite.")
            if not np.isfinite(obstacle.height) or obstacle.height <= 0.0:
                raise ValueError(f"Obstacle {index} height must be positive and finite.")
            if obstacle.shape != "cylinder":
                if obstacle.half_extents_xy is None or obstacle.half_extents_xy.shape != (2,):
                    raise ValueError(f"Obstacle {index} box/wall requires half_extents_xy shape (2,).")
                if not np.isfinite(obstacle.half_extents_xy).all() or np.any(obstacle.half_extents_xy <= 0.0):
                    raise ValueError(f"Obstacle {index} half_extents_xy must be positive and finite.")
            obstacles.append(obstacle)
        return obstacles

    def _state_safety(self, observation: Mapping[str, Any]) -> tuple[float, tuple[str, ...]]:
        """Check physical state safety separately from derivative barriers."""

        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        radius = float(self.env.agents["drone_radius"])
        slacks: dict[str, float] = {}
        for defender, position in enumerate(positions):
            for obstacle_index, obstacle in enumerate(self._obstacles_from_observation(observation)):
                clearance, _normal = self.env._cylinder_clearance_and_normal(position, obstacle)
                slacks[f"obstacle_{obstacle_index}_defender_{defender}"] = float(clearance - radius)
            for axis in range(3):
                slacks[f"boundary_lower_defender_{defender}_axis_{axis}"] = float(position[axis] - self.env.lower[axis])
                slacks[f"boundary_upper_defender_{defender}_axis_{axis}"] = float(self.env.upper[axis] - position[axis])
        for first in range(self.env.n_defenders):
            for second in range(first + 1, self.env.n_defenders):
                slacks[f"pairwise_{first}_{second}"] = float(
                    np.linalg.norm(positions[first] - positions[second]) - 2.0 * radius
                )
        minimum = min(slacks.values()) if slacks else float("inf")
        violations = tuple(name for name, value in slacks.items() if not np.isfinite(value) or value < -self.tolerance)
        return float(minimum), violations

    def _task_constraint_slacks(self, observation: Mapping[str, Any], action: np.ndarray) -> dict[str, float]:
        """Report target-approach progress without weakening safety rows."""

        beliefs = np.asarray(observation.get("target_belief_positions", np.zeros_like(action)), dtype=np.float64)
        positions = np.asarray(observation.get("defender_positions", np.zeros_like(action)), dtype=np.float64)
        if beliefs.shape != positions.shape or not np.isfinite(beliefs).all():
            return {"target_approach_progress": float("nan")}
        directions = beliefs - positions
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / np.maximum(norms, 1e-12)
        return {"target_approach_progress": float(np.mean(np.sum(directions * action, axis=1)))}

    def _closing_speed_bound(self, clearance: float, maximum_deceleration: float) -> float:
        clearance = max(float(clearance), 0.0)
        first_order = -self.gamma * clearance / self.env.dt
        braking = -float(np.sqrt(2.0 * maximum_deceleration * clearance))
        bound = max(first_order, braking)

        # The ordinary discrete CBF row can become infeasible one step later:
        # a velocity that is admissible at the current clearance may move the
        # agent into a state whose required braking change exceeds the next
        # acceleration ball.  Assume the most favorable admissible braking
        # change at each future step and impose the resulting linear lower
        # bound now.  For step k, v_k = v_0 + k*delta and
        # b_k = b_0 + dt*(k*v_0 + delta*k*(k-1)/2), which yields the bound
        # below.  This is a feasibility-preserving anticipation term, not a
        # relaxation of the physical obstacle, pairwise, or boundary margin.
        delta_velocity = float(maximum_deceleration) * self.env.dt
        for step in range(1, self.anticipatory_horizon_steps + 1):
            future_bound = (
                -self.gamma * clearance / self.env.dt
                - delta_velocity * (step + 0.5 * self.gamma * step * (step - 1))
            ) / (1.0 + self.gamma * step)
            bound = max(bound, float(future_bound))
        return bound

    def _solve(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        records: list[_LinearBarrier],
    ) -> _SolveResult:
        started = perf_counter()
        flat_reference = np.asarray(reference, dtype=np.float64).reshape(-1)
        if not np.isfinite(flat_reference).all():
            return _SolveResult(None, False, False, False, "nonfinite_reference", "reference is non-finite", float("inf"), 0.0, -float("inf"), {}, ())
        matrix = np.vstack([record.row for record in records]) if records else np.zeros((0, flat_reference.size), dtype=np.float64)
        lower = np.asarray([record.lower_bound for record in records], dtype=np.float64)

        def objective(flattened: np.ndarray) -> float:
            delta = flattened - flat_reference
            return 0.5 * float(np.dot(delta, delta))

        def motion_values(flattened: np.ndarray) -> np.ndarray:
            velocity = flattened.reshape(self.env.n_defenders, 3)
            max_speed = float(self.env.agents["defender_max_speed"])
            max_delta = float(self.env.agents["defender_max_acceleration"]) * self.env.dt
            return np.concatenate(
                [
                    max_speed**2 - np.sum(velocity**2, axis=1),
                    max_delta**2 - np.sum((velocity - current) ** 2, axis=1),
                ]
            )

        constraints: list[Any] = [
            NonlinearConstraint(
                motion_values,
                np.zeros(2 * self.env.n_defenders, dtype=np.float64),
                np.full(2 * self.env.n_defenders, np.inf, dtype=np.float64),
            )
        ]
        if records:
            constraints.insert(
                0,
                LinearConstraint(matrix, lower, np.full(len(records), np.inf, dtype=np.float64)),
            )
        result = None
        solver_success = False
        try:
            result = minimize(
                objective,
                x0=flat_reference,
                method="SLSQP",
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": self.solver_maxiter, "disp": False},
            )
            solver_success = bool(result.success)
            candidate = None if result.x is None else np.asarray(result.x, dtype=np.float64).reshape(self.env.n_defenders, 3)
            status = "success" if solver_success else "solver_failure"
            message = str(result.message)
            objective_value = float(result.fun) if np.isfinite(result.fun) else float("inf")
        except (FloatingPointError, TypeError, ValueError, RuntimeError) as error:
            candidate = None
            status = "solver_exception"
            message = repr(error)
            objective_value = float("inf")
        latency_ms = (perf_counter() - started) * 1000.0
        timed_out = latency_ms > self.max_latency_ms
        if timed_out:
            status = "timeout"
        if candidate is None or not np.isfinite(candidate).all():
            return _SolveResult(None, False, False, timed_out, status, message, objective_value, latency_ms, -float("inf"), {}, ())
        slacks = self._constraint_slacks(candidate, current, records)
        minimum = min(slacks.values()) if slacks else float("inf")
        feasible = bool(np.isfinite(minimum) and minimum >= -self.tolerance)
        # Solver success is useful provenance, but safety is determined from
        # measured residuals. A non-converged yet feasible iterate is retained
        # and labelled explicitly instead of being treated as a proof.
        if feasible and not solver_success:
            status = "feasible_nonconverged"
        return _SolveResult(
            candidate,
            solver_success,
            feasible,
            timed_out,
            status,
            message,
            objective_value,
            latency_ms,
            float(minimum),
            slacks,
            tuple(name for name, value in slacks.items() if value <= self.active_tolerance),
        )

    def _constraint_slacks(
        self,
        candidate: np.ndarray,
        current: np.ndarray,
        records: list[_LinearBarrier],
    ) -> dict[str, float]:
        flattened = candidate.reshape(-1)
        slacks = {record.name: float(record.row @ flattened - record.lower_bound) for record in records}
        max_speed = float(self.env.agents["defender_max_speed"])
        max_delta = float(self.env.agents["defender_max_acceleration"]) * self.env.dt
        speed = max_speed**2 - np.sum(candidate**2, axis=1)
        acceleration = max_delta**2 - np.sum((candidate - current) ** 2, axis=1)
        slacks.update({f"speed_defender_{index}": float(value) for index, value in enumerate(speed)})
        slacks.update({f"acceleration_defender_{index}": float(value) for index, value in enumerate(acceleration)})
        return slacks

    def _diagnostics(
        self,
        result: _SolveResult,
        desired: np.ndarray,
        *,
        requested_action_finite: bool,
        used_fallback: bool,
        fallback_mode: str,
        infeasible: bool,
        task_slacks: dict[str, float],
        primary: _SolveResult | None = None,
        minimum_state_clearance: float = float("inf"),
        state_safety_violation: bool = False,
    ) -> JointCBFQPDiagnostics:
        action = result.action if result.action is not None else np.zeros_like(desired)
        correction = float(np.max(np.linalg.norm(action - desired, axis=1))) if np.isfinite(action).all() else float("inf")
        return JointCBFQPDiagnostics(
            solver=self.SOLVER_NAME,
            solver_status=result.status if primary is None else f"{result.status};primary={primary.status}",
            solver_message=result.message,
            solver_success=bool(result.success),
            infeasible=bool(infeasible or not result.verified_feasible),
            timed_out=bool(result.timed_out),
            used_fallback=bool(used_fallback),
            fallback_mode=fallback_mode,
            verified_feasible=bool(result.verified_feasible),
            requested_action_finite=bool(requested_action_finite),
            action_correction_norm=correction,
            minimum_constraint_value=float(result.minimum_constraint_value),
            objective_value=float(result.objective_value),
            solve_latency_ms=float(result.latency_ms),
            constraint_slacks=dict(result.slacks),
            active_constraints=tuple(result.active_constraints),
            task_constraint_slacks=dict(task_slacks),
            state_safety_violation=bool(state_safety_violation),
            minimum_state_clearance=float(minimum_state_clearance),
        )

    def _controlled_abort(
        self,
        current: np.ndarray,
        desired: np.ndarray,
        records: list[_LinearBarrier],
        task_slacks: dict[str, float],
        *,
        status: str,
        message: str,
        requested_action_finite: bool,
        primary: _SolveResult | None = None,
        minimum_state_clearance: float = float("inf"),
        state_safety_violation: bool = False,
    ) -> tuple[np.ndarray, JointCBFQPDiagnostics]:
        action = self._emergency_action(current)
        slacks = self._constraint_slacks(action, current, records)
        minimum = min(slacks.values()) if slacks else float("inf")
        diagnostics = JointCBFQPDiagnostics(
            solver=self.SOLVER_NAME,
            solver_status=status if primary is None else f"{status};primary={primary.status}",
            solver_message=message,
            solver_success=False,
            infeasible=True,
            timed_out=bool(primary.timed_out) if primary is not None else False,
            used_fallback=True,
            fallback_mode="controlled_abort",
            # A controlled abort is an emergency command, not a solver proof.
            # Even when its measured residuals happen to be non-negative, the
            # caller must treat the fallback as unverified and keep it visible.
            verified_feasible=False,
            requested_action_finite=bool(requested_action_finite),
            action_correction_norm=float(np.max(np.linalg.norm(action - desired, axis=1))) if np.isfinite(desired).all() else float("inf"),
            minimum_constraint_value=float(minimum),
            objective_value=float("inf"),
            solve_latency_ms=float(primary.latency_ms) if primary is not None else 0.0,
            constraint_slacks=slacks,
            active_constraints=tuple(name for name, value in slacks.items() if value <= self.active_tolerance),
            task_constraint_slacks=dict(task_slacks),
            state_safety_violation=bool(state_safety_violation),
            minimum_state_clearance=float(minimum_state_clearance),
        )
        return action, diagnostics


def make_joint_cbf_qp(env: CaptureRadiusPursuit3DEnv, **kwargs: Any) -> JointCBFQPSafetyFilter:
    """Factory used by evaluation scripts and tests."""

    return JointCBFQPSafetyFilter(env, **kwargs)
