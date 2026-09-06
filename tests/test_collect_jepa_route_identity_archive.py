from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "collect_jepa_route_identity_archive.py"
SPEC = importlib.util.spec_from_file_location("collect_jepa_route_identity_archive", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)

_append_boundary_shadow_samples = COLLECTOR._append_boundary_shadow_samples
_boundary_shadow_rollout = COLLECTOR._boundary_shadow_rollout
_class_counts = COLLECTOR._class_counts
_empty_samples = COLLECTOR._empty_samples


class _ShadowEnv:
    def __init__(self) -> None:
        self.defender_positions = np.array(
            [
                [-9.0, -1.0, 4.0],
                [-9.0, 1.0, 4.0],
                [-9.0, 0.8, 5.0],
                [-9.0, -0.8, 5.0],
            ],
            dtype=np.float64,
        )
        self.lower = np.array([-10.0, -10.0, 0.5], dtype=np.float64)
        self.upper = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        self.n_defenders = 4
        self.dt = 0.5
        self.agents = {"drone_radius": 0.5, "defender_max_speed": 2.0}
        self.step_calls = 0

    def step(self, _action: np.ndarray) -> None:
        self.step_calls += 1
        raise AssertionError("boundary shadow must never enter environment execution")


def test_boundary_shadow_from_nine_meter_initial_side_distance_has_negative_clearance() -> None:
    env = _ShadowEnv()

    _actions, boundary = _boundary_shadow_rollout(env)

    assert float(boundary.min()) < 0.0
    assert env.step_calls == 0


def test_boundary_shadow_append_is_offline_only_and_does_not_step_environment() -> None:
    env = _ShadowEnv()
    samples = _empty_samples()
    observation_history = [np.zeros((4, 63), dtype=np.float32) for _ in range(8)]
    executed_action_history = [np.zeros((4, 3), dtype=np.float32) for _ in range(7)]

    minimum = _append_boundary_shadow_samples(
        samples,
        observation_history=observation_history,
        executed_action_history=executed_action_history,
        env=env,
        episode_seed=11,
        scenario_index=2,
        time_index=8,
        action_scale=5.0,
    )

    assert minimum < 0.0
    assert env.step_calls == 0
    assert samples["sample_type"] == [1, 1, 1, 1]
    assert samples["route_candidate_index"] == [-1, -1, -1, -1]


def test_class_counts_separate_runtime_rows_from_boundary_shadow_rows() -> None:
    arrays = {
        "sample_type": np.array([0, 0, 1, 1], dtype=np.int64),
        "route_geometry_valid": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "labels_cbf_feasible": np.array(
            [[1.0], [0.0], [1.0], [0.0]], dtype=np.float32
        ),
        "earliest_failure_step": np.array([5, 1, 1, 1], dtype=np.int64),
        "labels_boundary_clearance": np.array(
            [[1.0], [1.0], [-1.0], [-2.0]], dtype=np.float32
        ),
    }

    counts = _class_counts(arrays)

    assert counts == {
        "route_geometry_valid": 1,
        "route_geometry_invalid": 1,
        "cbf_first_step_feasible": 1,
        "cbf_first_step_infeasible": 1,
        "branch_failure_within_horizon": 2,
        "boundary_clearance_negative": 2,
        "boundary_shadow_samples": 2,
    }
