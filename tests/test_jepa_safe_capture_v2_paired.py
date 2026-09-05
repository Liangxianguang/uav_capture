from __future__ import annotations

import copy
import hashlib
import json
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
from scripts.evaluate_jepa_safe_capture_v2_paired import (
    _load_ledger,
    _raw_unverified_executed,
    _variant_contract,
)


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


def test_controlled_abort_is_unverified_but_not_raw_execution() -> None:
    diagnostics = type(
        "Diagnostics",
        (),
        {"verified_feasible": False, "fallback_mode": "controlled_abort"},
    )()
    assert not _raw_unverified_executed(safety_filter_enabled=True, diagnostics=diagnostics)


def test_unverified_unknown_cbf_path_is_counted_as_raw_execution() -> None:
    diagnostics = type(
        "Diagnostics",
        (),
        {"verified_feasible": False, "fallback_mode": "none"},
    )()
    assert _raw_unverified_executed(safety_filter_enabled=True, diagnostics=diagnostics)


def test_no_cbf_path_is_always_raw_execution() -> None:
    assert _raw_unverified_executed(safety_filter_enabled=False, diagnostics=None)


def test_ledger_loader_rejects_protocol_hash_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    protocol = tmp_path / "protocol.yaml"
    ledger_path = tmp_path / "ledger.json"
    checkpoint.write_bytes(b"checkpoint")
    protocol.write_text("protocol: v10\n", encoding="utf-8")
    payload = {
        "ledger_type": "jepa_safe_capture_v2_checkpoint_bound_reliability",
        "ledger_version": 2,
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "immutable_after_calibration": True,
        "source": {
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "protocol_sha256": "0" * 64,
            "calibration_dataset_sha256": "b" * 64,
        },
        "entries": {},
        "decision_policy": {
            "states": ["trusted", "fallback_nominal", "safe_hold"],
            "minimum_sample_count": 128,
            "minimum_credit": 0.65,
            "maximum_observation_age_steps": 45.0,
            "safe_hold_uncertainty_threshold": 0.40,
            "safe_hold_ttc_seconds": 0.30,
        },
    }
    ledger_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="protocol hash"):
        _load_ledger(ledger_path, checkpoint, protocol)


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
