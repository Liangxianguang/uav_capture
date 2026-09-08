from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "collect_jepa_route_identity_archive.py"
SPEC = importlib.util.spec_from_file_location("collect_jepa_route_identity_archive", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)

_append_boundary_shadow_samples = COLLECTOR._append_boundary_shadow_samples
_boundary_shadow_rollout = COLLECTOR._boundary_shadow_rollout
_class_counts = COLLECTOR._class_counts
_empty_samples = COLLECTOR._empty_samples
_copy_cbf_filter = COLLECTOR._copy_cbf_filter
_archive_cbf_contract = COLLECTOR._archive_cbf_contract
_pairwise_ttc_labels = COLLECTOR._pairwise_ttc_labels
_risk_labels = COLLECTOR._risk_labels
_boundary_shadow_ttc = COLLECTOR._boundary_shadow_ttc
_interaction_hard_negative_chunks = COLLECTOR._interaction_hard_negative_chunks
_reachable_interaction_chunk = COLLECTOR._reachable_interaction_chunk
_append_samples = COLLECTOR._append_samples
_match_executed_route = COLLECTOR._match_executed_route


def test_executed_route_match_records_nearest_candidate_and_residual() -> None:
    batch = SimpleNamespace(
        candidates=[
            SimpleNamespace(action_chunk=np.zeros((5, 4, 3), dtype=np.float64)),
            SimpleNamespace(action_chunk=np.ones((5, 4, 3), dtype=np.float64)),
        ]
    )

    index, residual = _match_executed_route(batch, np.ones((4, 3), dtype=np.float64) * 0.2)

    assert index == 0
    assert residual == pytest.approx(0.2 * np.sqrt(3.0))


def test_executed_route_match_marks_large_residual_unknown() -> None:
    batch = SimpleNamespace(
        candidates=[SimpleNamespace(action_chunk=np.zeros((5, 2, 3), dtype=np.float64))]
    )

    index, residual = _match_executed_route(
        batch,
        np.ones((2, 3), dtype=np.float64),
        tolerance_mps=0.5,
    )

    assert index == -1
    assert residual == pytest.approx(np.sqrt(3.0))


def test_unknown_runtime_route_is_an_explicit_switch_abstention() -> None:
    class _Route:
        action_chunk = np.zeros((5, 1, 3), dtype=np.float64)
        side = "nominal"
        obstacle_id = None
        route_length_m = 1.0
        minimum_geometric_clearance_m = 1.0
        label = "nominal"
        valid = True

    samples = _empty_samples()
    labels = {
        name: np.zeros((5, 1), dtype=np.float32)
        for name in (
            "relative", "obstacle_clearance", "inter_agent_clearance",
            "boundary_clearance", "stopping_distance", "obstacle_ttc",
            "boundary_ttc", "pairwise_ttc", "acceleration_slack",
            "target_visible", "cbf_correction", "cbf_intervention",
            "cbf_feasible", "cbf_min_slack", "route_progress", "rollout_valid",
        )
    }
    labels["relative"] = np.zeros((5, 1, 3), dtype=np.float32)
    labels["earliest_failure_step"] = 1
    labels["branch_terminated"] = True
    _append_samples(
        samples,
        observation_history=[np.zeros((1, 63), dtype=np.float32) for _ in range(8)],
        executed_action_history=[np.zeros((1, 3), dtype=np.float32) for _ in range(7)],
        route=_Route(),
        route_index=0,
        labels=labels,
        episode_seed=1,
        scenario_index=2,
        time_index=47,
        action_scale=5.0,
        previous_executed_route_index=3,
        executed_route_index=-1,
        route_match_residual_mps=-1.0,
    )
    assert samples["executed_route_index"] == [-1]
    assert samples["route_switch_outcome"] == [-1]


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
        self.agents = {
            "drone_radius": 0.5,
            "defender_max_speed": 2.0,
            "defender_max_acceleration": 2.0,
        }
        self.step_calls = 0

    def step(self, _action: np.ndarray) -> None:
        self.step_calls += 1
        raise AssertionError("boundary shadow must never enter environment execution")


def test_boundary_shadow_from_nine_meter_initial_side_distance_has_negative_clearance() -> None:
    env = _ShadowEnv()

    actions, boundary = _boundary_shadow_rollout(env, horizon=5)

    assert actions.shape == (5, env.n_defenders, 3)
    assert boundary.shape == (5, env.n_defenders)
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
    assert np.asarray(samples["route_relative_action_chunk"]).shape == (4, 5, 3)


def test_boundary_shadow_ttc_is_nonnegative_and_finite() -> None:
    env = _ShadowEnv()
    actions, boundary = _boundary_shadow_rollout(env)
    ttc = _boundary_shadow_ttc(env, actions)

    assert ttc.shape == boundary.shape
    assert np.all(np.isfinite(ttc))
    assert float(ttc.min()) >= 0.0
    assert float(ttc.max()) <= COLLECTOR.TTC_CLIP_SECONDS


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


def test_copy_cbf_filter_preserves_horizon_and_barrier_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeFilter:
        def __init__(self, _env: object, **kwargs: object) -> None:
            captured.update(kwargs)

    class _Source:
        contract = {
            "gamma": 0.25,
            "obstacle_margin_m": 0.35,
            "inter_agent_margin_m": 0.35,
            "boundary_margin_m": 0.35,
            "max_correction_norm_mps": 5.0,
            "max_latency_ms": 100.0,
            "solver_maxiter": 80,
            "tolerance": 1e-5,
            "active_tolerance": 5e-5,
            "anticipatory_horizon_steps": 5,
            "barrier_mode": "strict_buffer",
        }

    monkeypatch.setattr(COLLECTOR, "JointCBFQPSafetyFilter", _FakeFilter)
    _copy_cbf_filter(_Source(), object())

    assert captured["anticipatory_horizon_steps"] == 5
    assert captured["barrier_mode"] == "strict_buffer"


def test_archive_cbf_contract_is_explicit_and_validated() -> None:
    contract = _archive_cbf_contract(
        {"cbf_contract": {"anticipatory_horizon_steps": 5, "barrier_mode": "strict_buffer"}}
    )
    assert contract == {"anticipatory_horizon_steps": 5, "barrier_mode": "strict_buffer"}

    with pytest.raises(ValueError, match="barrier_mode"):
        _archive_cbf_contract({"cbf_contract": {"anticipatory_horizon_steps": 5, "barrier_mode": "invalid"}})


def test_pairwise_ttc_labels_distinguish_approaching_and_receding_agents() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64)
    approaching = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
    receding = -approaching

    approaching_ttc = _pairwise_ttc_labels(
        positions, approaching, radius=0.5, margin=0.35
    )
    receding_ttc = _pairwise_ttc_labels(
        positions, receding, radius=0.5, margin=0.35
    )

    assert np.all(np.isfinite(approaching_ttc))
    assert np.all(approaching_ttc < 10.0)
    assert np.all(receding_ttc == 10.0)


def test_risk_labels_include_stopping_distance_and_boundary_ttc() -> None:
    class _RiskEnv:
        defender_positions = np.array([[8.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        lower = np.array([-10.0, -10.0, 0.5], dtype=np.float64)
        upper = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        n_defenders = 2
        obstacles: list[object] = []
        agents = {
            "drone_radius": 0.5,
            "defender_max_acceleration": 2.0,
        }

    env = _RiskEnv()
    velocities = np.array([[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    labels = _risk_labels(
        env,
        velocities,
        obstacle_margin_m=0.35,
        boundary_margin_m=0.35,
        inter_agent_margin_m=0.35,
    )

    assert set(labels) == {"stopping_distance", "obstacle_ttc", "boundary_ttc", "pairwise_ttc"}
    assert all(np.all(np.isfinite(value)) for value in labels.values())
    assert labels["stopping_distance"][0] == pytest.approx(4.0)
    assert labels["boundary_ttc"][0] < 10.0
    assert labels["boundary_ttc"][1] == pytest.approx(10.0)


class _InteractionEnv:
    defender_positions = np.array(
        [
            [-2.0, -0.5, 2.0],
            [2.0, 0.5, 2.0],
            [-1.5, 2.0, 2.0],
            [1.5, 2.2, 2.0],
        ],
        dtype=np.float64,
    )
    n_defenders = 4
    dt = 0.1
    agents = {"defender_max_speed": 5.0, "defender_max_acceleration": 6.0}

    @staticmethod
    def _clip_rows(value: np.ndarray, max_norm: float) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64).copy()
        norms = np.linalg.norm(value, axis=1, keepdims=True)
        scale = np.minimum(1.0, float(max_norm) / np.maximum(norms, 1e-12))
        return value * scale

    @staticmethod
    def _move_toward_velocity(previous: np.ndarray, desired: np.ndarray, *, max_delta: float) -> np.ndarray:
        delta = np.asarray(desired) - np.asarray(previous)
        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        return np.asarray(previous) + delta * np.minimum(1.0, float(max_delta) / np.maximum(norms, 1e-12))


def test_interaction_hard_negative_modes_are_projected_and_distinct() -> None:
    env = _InteractionEnv()
    nominal = np.zeros((env.n_defenders, 3), dtype=np.float64)
    previous = np.zeros_like(nominal)
    chunks = _interaction_hard_negative_chunks(env, nominal, previous, 5)

    assert set(chunks) == {"near_pass", "formation_crossing", "split_merge"}
    for chunk in chunks.values():
        assert chunk.shape == (5, env.n_defenders, 3)
        assert np.isfinite(chunk).all()
        assert float(np.linalg.norm(chunk, axis=-1).max()) <= 5.0 + 1e-8
    assert not np.allclose(chunks["near_pass"], chunks["formation_crossing"])
    assert not np.allclose(chunks["formation_crossing"], chunks["split_merge"])


def test_reachable_interaction_projection_respects_acceleration_envelope() -> None:
    env = _InteractionEnv()
    previous = np.zeros((env.n_defenders, 3), dtype=np.float64)
    desired = np.full((5, env.n_defenders, 3), 5.0, dtype=np.float64)
    projected = _reachable_interaction_chunk(env, previous, desired)

    assert projected.shape == desired.shape
    assert np.isfinite(projected).all()
    step_deltas = np.linalg.norm(np.diff(np.concatenate([previous[None], projected], axis=0), axis=2), axis=2)
    assert float(step_deltas.max()) <= env.agents["defender_max_acceleration"] * env.dt + 1e-8


def test_hard_negative_sample_keeps_runtime_side_contract_for_boundary_rescue() -> None:
    class _Route:
        action_chunk = np.zeros((5, 2, 3), dtype=np.float64)
        side = "boundary_rescue"
        obstacle_id = None
        route_length_m = 1.0
        minimum_geometric_clearance_m = 0.5
        label = "boundary_rescue"
        valid = False

    samples = _empty_samples()
    labels = {
        name: np.zeros((5, 2), dtype=np.float32)
        for name in (
            "relative", "obstacle_clearance", "inter_agent_clearance",
            "boundary_clearance", "stopping_distance", "obstacle_ttc",
            "boundary_ttc", "pairwise_ttc", "acceleration_slack",
            "target_visible", "cbf_correction", "cbf_intervention",
            "cbf_feasible", "cbf_min_slack", "route_progress", "rollout_valid",
        )
    }
    labels["relative"] = np.zeros((5, 2, 3), dtype=np.float32)
    labels["earliest_failure_step"] = 1
    labels["branch_terminated"] = False
    _append_samples(
        samples,
        observation_history=[np.zeros((2, 63), dtype=np.float32) for _ in range(8)],
        executed_action_history=[np.zeros((2, 3), dtype=np.float32) for _ in range(7)],
        route=_Route(),
        route_index=-1,
        labels=labels,
        episode_seed=1,
        scenario_index=2,
        time_index=47,
        action_scale=5.0,
        sample_type=COLLECTOR.HARD_NEGATIVE_SAMPLE_TYPES["boundary_rescue"],
    )
    assert samples["sample_type"] == [8, 8]
    assert samples["route_candidate_index"] == [-1, -1]
    assert samples["route_side_index"] == [11, 11]
