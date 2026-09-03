from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from encirclement3d.cbf_qp import JointCBFQPSafetyFilter
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv, CylinderObstacle


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    with (ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _env(*, obstacles: list[CylinderObstacle] | None = None) -> CaptureRadiusPursuit3DEnv:
    env = CaptureRadiusPursuit3DEnv(copy.deepcopy(_config()), obstacle_count=0, target_speed_scale=0.45)
    env.reset(20260911)
    if obstacles is not None:
        env.obstacles = obstacles
    return env


def _observation(env: CaptureRadiusPursuit3DEnv, positions: np.ndarray, velocities: np.ndarray | None = None) -> dict:
    env.defender_positions = np.asarray(positions, dtype=np.float64).copy()
    env.defender_velocities = (
        np.zeros_like(env.defender_positions)
        if velocities is None
        else np.asarray(velocities, dtype=np.float64).copy()
    )
    return env.observe()


def test_normal_joint_solve_is_feasible_and_deterministic() -> None:
    env = _env()
    observation = _observation(
        env,
        np.array([[5.0, 0.0, 4.0], [0.0, 5.0, 4.0], [-5.0, 0.0, 4.0], [0.0, -5.0, 4.0]]),
    )
    desired = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [0.0, -2.0, 0.0]])
    filter_ = JointCBFQPSafetyFilter(env)

    first_action, first = filter_.filter(desired, observation)
    second_action, second = filter_.filter(desired, observation)

    assert first.verified_feasible
    assert not first.infeasible
    assert first.fallback_mode == "none"
    assert first.solver_status == "success"
    assert np.allclose(first_action, second_action, atol=1e-8)
    assert first.constraint_slacks == second.constraint_slacks
    assert first.active_constraints == second.active_constraints


def test_obstacle_constraint_reduces_inward_velocity_and_is_reported() -> None:
    env = _env(obstacles=[CylinderObstacle(np.array([0.0, 0.0]), 1.0, 5.0)])
    observation = _observation(
        env,
        np.array([[1.65, 0.0, 2.0], [5.0, 0.0, 4.0], [-5.0, 0.0, 4.0], [0.0, -5.0, 4.0]]),
    )
    desired = np.array([[-5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    action, diagnostics = JointCBFQPSafetyFilter(env).filter(desired, observation)

    assert diagnostics.verified_feasible
    assert action[0, 0] > desired[0, 0]
    assert diagnostics.constraint_slacks["obstacle_0_defender_0"] >= -1e-5
    assert "obstacle_0_defender_0" in diagnostics.active_constraints or action[0, 0] > desired[0, 0]


def test_box_geometry_is_decoded_from_public_observation() -> None:
    env = _env(obstacles=[CylinderObstacle(np.array([0.0, 0.0]), 1.0, 5.0, "box", np.array([1.0, 2.0]))])
    observation = _observation(
        env,
        np.array([[0.0, 2.8, 2.0], [5.0, 0.0, 4.0], [-5.0, 0.0, 4.0], [0.0, -5.0, 4.0]]),
    )
    desired = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    action, diagnostics = JointCBFQPSafetyFilter(env).filter(desired, observation)

    assert diagnostics.verified_feasible
    assert action[0, 1] > desired[0, 1]
    assert diagnostics.constraint_slacks["obstacle_0_defender_0"] >= -1e-5


def test_pairwise_separation_constraint_is_joint() -> None:
    env = _env()
    observation = _observation(
        env,
        np.array([[0.5, 0.0, 4.0], [-0.5, 0.0, 4.0], [5.0, 0.0, 4.0], [0.0, -5.0, 4.0]]),
    )
    desired = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    action, diagnostics = JointCBFQPSafetyFilter(env).filter(desired, observation)

    assert diagnostics.verified_feasible
    assert action[0, 0] - action[1, 0] > desired[0, 0] - desired[1, 0]
    assert diagnostics.constraint_slacks["pairwise_0_1"] >= -1e-5
    assert "pairwise_0_1" in diagnostics.constraint_slacks


@pytest.mark.parametrize(
    ("position", "desired", "constraint_prefix"),
    [
        (np.array([[-9.0, 0.0, 4.0], [4.0, 4.0, 4.0], [5.0, 0.0, 4.0], [0.0, -5.0, 4.0]]), np.array([[-5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), "boundary_lower_defender_0_axis_0"),
        (np.array([[4.0, 4.0, 0.7], [4.0, -4.0, 4.0], [-4.0, 4.0, 4.0], [-4.0, -4.0, 4.0]]), np.array([[0.0, 0.0, -5.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), "altitude_lower_defender_0_axis_2"),
    ],
)
def test_boundary_and_altitude_constraints_are_enforced(
    position: np.ndarray, desired: np.ndarray, constraint_prefix: str
) -> None:
    env = _env()
    observation = _observation(env, position)
    action, diagnostics = JointCBFQPSafetyFilter(env).filter(desired, observation)

    assert diagnostics.verified_feasible
    assert diagnostics.constraint_slacks[constraint_prefix] >= -1e-5
    assert np.isfinite(action).all()


def test_speed_and_acceleration_constraints_are_enforced() -> None:
    env = _env()
    current = np.full((4, 3), 0.4)
    observation = _observation(
        env,
        np.array([[5.0, 0.0, 4.0], [0.0, 5.0, 4.0], [-5.0, 0.0, 4.0], [0.0, -5.0, 4.0]]),
        current,
    )
    desired = np.full((4, 3), 20.0)
    action, diagnostics = JointCBFQPSafetyFilter(env).filter(desired, observation)
    max_delta = float(env.agents["defender_max_acceleration"]) * env.dt

    assert diagnostics.verified_feasible
    assert np.max(np.linalg.norm(action, axis=1)) <= float(env.agents["defender_max_speed"]) + 1e-6
    assert np.max(np.linalg.norm(action - current, axis=1)) <= max_delta + 1e-5
    assert min(value for name, value in diagnostics.constraint_slacks.items() if name.startswith("speed_")) >= -1e-5
    assert min(value for name, value in diagnostics.constraint_slacks.items() if name.startswith("acceleration_")) >= -1e-5


def test_anticipatory_horizon_adds_only_a_conservative_braking_constraint() -> None:
    env = _env(obstacles=[CylinderObstacle(np.array([0.0, 0.0]), 1.0, 5.0)])
    observation = _observation(
        env,
        np.array([[3.45, 0.0, 2.0], [5.0, 0.0, 4.0], [-5.0, 0.0, 4.0], [0.0, -5.0, 4.0]]),
        np.array([[-3.8, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    no_lookahead = JointCBFQPSafetyFilter(env, anticipatory_horizon_steps=0)
    with_lookahead = JointCBFQPSafetyFilter(env, anticipatory_horizon_steps=3)
    base = next(record for record in no_lookahead._build_barriers(observation) if record.name == "obstacle_0_defender_0")
    conservative = next(
        record for record in with_lookahead._build_barriers(observation) if record.name == "obstacle_0_defender_0"
    )

    assert conservative.lower_bound >= base.lower_bound
    assert conservative.lower_bound > base.lower_bound
    action, diagnostics = with_lookahead.filter(observation["defender_velocities"], observation)
    assert diagnostics.verified_feasible
    assert diagnostics.constraint_slacks["obstacle_0_defender_0"] >= -1e-5
    assert np.isfinite(action).all()


def test_current_state_violation_is_not_claimed_feasible() -> None:
    env = _env(obstacles=[CylinderObstacle(np.array([0.0, 0.0]), 1.0, 5.0)])
    observation = _observation(
        env,
        np.array([[0.1, 0.0, 2.0], [5.0, 0.0, 4.0], [-5.0, 0.0, 4.0], [0.0, -5.0, 4.0]]),
    )
    desired = np.ones((4, 3))
    action, diagnostics = JointCBFQPSafetyFilter(env).filter(desired, observation)

    assert diagnostics.state_safety_violation
    assert diagnostics.infeasible
    assert not diagnostics.verified_feasible
    assert diagnostics.fallback_mode == "controlled_abort"
    assert not np.allclose(action, desired)


def test_nonfinite_request_routes_to_controlled_abort() -> None:
    env = _env()
    observation = _observation(
        env,
        np.array([[5.0, 0.0, 4.0], [0.0, 5.0, 4.0], [-5.0, 0.0, 4.0], [0.0, -5.0, 4.0]]),
    )
    desired = np.zeros((4, 3))
    desired[0, 0] = np.nan
    action, diagnostics = JointCBFQPSafetyFilter(env).filter(desired, observation)

    assert diagnostics.solver_status == "nonfinite_request"
    assert diagnostics.fallback_mode == "controlled_abort"
    assert not diagnostics.requested_action_finite
    assert not diagnostics.verified_feasible
    assert np.isfinite(action).all()
    assert np.isinf(diagnostics.action_correction_norm)


def test_motion_infeasibility_does_not_execute_original_action() -> None:
    env = _env()
    current = np.full((4, 3), 20.0)
    observation = _observation(
        env,
        np.array([[5.0, 0.0, 4.0], [0.0, 5.0, 4.0], [-5.0, 0.0, 4.0], [0.0, -5.0, 4.0]]),
        current,
    )
    desired = np.full((4, 3), -20.0)
    action, diagnostics = JointCBFQPSafetyFilter(env).filter(desired, observation)

    assert diagnostics.infeasible
    assert diagnostics.used_fallback
    assert diagnostics.fallback_mode == "controlled_abort"
    assert not np.allclose(action, desired)
    assert np.isfinite(action).all()


def test_nominal_fallback_is_filtered_and_task_diagnostic_is_separate() -> None:
    env = _env()
    observation = _observation(
        env,
        np.array([[5.0, 0.0, 4.0], [0.0, 5.0, 4.0], [-5.0, 0.0, 4.0], [0.0, -5.0, 4.0]]),
    )
    desired = np.full((4, 3), np.nan)
    nominal = np.full((4, 3), 0.25)
    action, diagnostics = JointCBFQPSafetyFilter(env).filter(desired, observation, nominal_actions=nominal)

    assert diagnostics.fallback_mode == "controlled_abort"
    assert diagnostics.task_constraint_slacks.keys() == {"target_approach_progress"}
    assert np.isfinite(action).all()


def test_timeout_is_observable_and_never_returns_unfiltered_request() -> None:
    env = _env()
    observation = _observation(
        env,
        np.array([[5.0, 0.0, 4.0], [0.0, 5.0, 4.0], [-5.0, 0.0, 4.0], [0.0, -5.0, 4.0]]),
    )
    desired = np.full((4, 3), 1.0)
    action, diagnostics = JointCBFQPSafetyFilter(env, max_latency_ms=1e-12).filter(desired, observation)

    assert diagnostics.timed_out
    assert diagnostics.used_fallback
    assert diagnostics.fallback_mode == "controlled_abort"
    assert not np.allclose(action, desired)


def test_public_observation_geometry_is_used() -> None:
    env = _env()
    observation = _observation(
        env,
        np.array([[3.2, 0.0, 2.0], [5.0, 0.0, 4.0], [-5.0, 0.0, 4.0], [0.0, -5.0, 4.0]]),
    )
    observation["obstacles"] = [
        {
            "center_xy": np.array([2.0, 0.0]),
            "radius": 0.8,
            "height": 5.0,
            "shape": "cylinder",
            "half_extents_xy": None,
        }
    ]
    desired = np.array([[-5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    action, diagnostics = JointCBFQPSafetyFilter(env).filter(desired, observation)

    assert diagnostics.verified_feasible
    assert "obstacle_0_defender_0" in diagnostics.constraint_slacks
    assert action[0, 0] > desired[0, 0]
