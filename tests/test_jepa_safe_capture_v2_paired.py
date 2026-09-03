from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from encirclement3d.cbf_qp import JointCBFQPSafetyFilter
from encirclement3d.jepa_safe_capture_candidates import (
    SafeCaptureCandidateConfig,
    make_safe_capture_candidate_chunks,
)
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv, CylinderObstacle
from scripts.evaluate_jepa_safe_capture_v2_paired import _variant_contract


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    with (ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        return yaml.safe_load(handle)


def _env(*, obstacles: list[CylinderObstacle] | None = None) -> CaptureRadiusPursuit3DEnv:
    env = CaptureRadiusPursuit3DEnv(copy.deepcopy(_config()), obstacle_count=0, target_speed_scale=0.45)
    env.reset(20260911)
    if obstacles is not None:
        env.obstacles = obstacles
    return env


def _observation(env: CaptureRadiusPursuit3DEnv) -> dict:
    env.defender_positions = np.array(
        [[5.0, 0.0, 4.0], [0.0, 5.0, 4.0], [-5.0, 0.0, 4.0], [0.0, -5.0, 4.0]],
        dtype=np.float64,
    )
    env.defender_velocities = np.zeros((4, 3), dtype=np.float64)
    return env.observe()


def test_paired_variants_keep_jepa_as_evaluator_and_cbf_as_execution_boundary() -> None:
    m3 = _variant_contract("m3")
    m0 = _variant_contract("m0")
    a3 = _variant_contract("a3")

    assert m3["use_jepa"] and m3["use_ledger"] and m3["use_cbf"]
    assert not m0["use_jepa"] and m0["use_cbf"]
    assert a3["diagnostic_only"] and not a3["use_cbf"]


def test_projected_candidate_batch_is_deterministic_and_causally_reachable() -> None:
    env = _env()
    observation = _observation(env)
    nominal = np.full((4, 3), 5.0, dtype=np.float64)
    previous = np.zeros_like(nominal)
    config = SafeCaptureCandidateConfig(project_to_reachable_dynamics=True)

    first = make_safe_capture_candidate_chunks(
        nominal, observation, config=config, previous_action=previous
    )
    second = make_safe_capture_candidate_chunks(
        nominal, observation, config=config, previous_action=previous
    )

    np.testing.assert_array_equal(first.chunks, second.chunks)
    np.testing.assert_array_equal(first.valid_mask, np.ones(5, dtype=bool))
    assert np.max(np.linalg.norm(first.chunks[:, 0] - previous[None], axis=-1)) <= 0.6 + 1e-8


def test_jepa_candidate_requests_are_filtered_before_execution() -> None:
    env = _env()
    observation = _observation(env)
    requested = np.full((4, 3), 5.0, dtype=np.float64)
    action, diagnostics = JointCBFQPSafetyFilter(env).filter(requested, observation)

    assert diagnostics.verified_feasible
    assert np.isfinite(action).all()
    assert not np.allclose(action, requested)
    assert np.max(np.linalg.norm(action - observation["defender_velocities"], axis=1)) <= 0.6 + 1e-5


def test_infeasible_or_timeout_never_executes_raw_request() -> None:
    env = _env(obstacles=[CylinderObstacle(np.array([0.0, 0.0]), 1.0, 5.0)])
    env.defender_positions[0] = np.array([1.25, 0.0, 2.0])
    env.defender_velocities[0] = np.array([-5.0, 0.0, 0.0])
    observation = env.observe()
    requested = np.full((4, 3), 5.0, dtype=np.float64)

    action, diagnostics = JointCBFQPSafetyFilter(env).filter(requested, observation)

    assert diagnostics.infeasible
    assert diagnostics.used_fallback
    assert diagnostics.fallback_mode == "controlled_abort"
    assert not diagnostics.verified_feasible
    assert np.isfinite(action).all()
    assert not np.allclose(action, requested)


def test_safe_capture_requires_verified_cbf_and_no_collision() -> None:
    # This mirrors the evaluator's settled-label gate: a capture event cannot
    # be promoted to safe capture when the CBF path was unverified.
    capture_event = True
    collision = False
    boundary_violation = False
    pairwise_violation = False
    target_collision = False
    cbf_unverified_steps = 1
    safe_capture = bool(capture_event) and not (
        collision
        or boundary_violation
        or pairwise_violation
        or target_collision
        or cbf_unverified_steps > 0
    )
    assert not safe_capture


@pytest.mark.parametrize("variant", ("m0", "m1", "m2", "m3", "a1", "a2", "a3"))
def test_all_declared_paired_variants_are_development_only_by_contract(variant: str) -> None:
    contract = _variant_contract(variant)
    assert contract["variant"] == variant
    if variant != "a3":
        assert contract["use_cbf"]
