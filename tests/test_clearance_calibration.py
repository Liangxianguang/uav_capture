import hashlib
import json
import numpy as np
import pytest

from encirclement3d.clearance_calibration import (
    apply_head_offsets,
    build_calibration_transform,
    normalized_to_meters,
    offsets_for_horizon,
    q10_residual_offset,
)
from encirclement3d.reliability import SafeCaptureReliabilityLedger, make_safe_capture_global_key


def test_q10_residual_offset_is_lower_quantile_and_records_ci() -> None:
    raw = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    labels = np.array([0.8, 1.0, 1.1, 1.2, 1.3])
    result = q10_residual_offset(raw, labels)
    assert result["quantile"] == 0.1
    assert result["offset_m"] == pytest.approx(-0.12)
    assert result["sample_count"] == 5
    assert len(result["quantile_ci95_m"]) == 2
    assert result["calibrated_residual_q10_m"] == pytest.approx(0.0)


def test_normalized_conversion_and_head_offsets_are_explicit() -> None:
    np.testing.assert_allclose(normalized_to_meters([0.1, 0.2], 10.0), [1.0, 2.0])
    obstacle, inter = apply_head_offsets([1.0, 2.0], [3.0, 4.0], {"obstacle_clearance": 0.2, "inter_agent_clearance": -0.1})
    np.testing.assert_allclose(obstacle, [1.2, 2.2])
    np.testing.assert_allclose(inter, [2.9, 3.9])


def test_build_transform_is_per_head_and_horizon() -> None:
    raw_obstacle = np.full((8, 2), 0.2, dtype=np.float64)
    raw_inter = np.full((8, 2), 0.5, dtype=np.float64)
    labels_obstacle = np.full((8, 2), 0.3, dtype=np.float64)
    labels_inter = np.full((8, 2), 0.4, dtype=np.float64)
    transform = build_calibration_transform(
        raw_obstacle,
        raw_inter,
        labels_obstacle,
        labels_inter,
        world_extent_m=10.0,
        horizon_seconds=[0.1, 0.2],
    )
    assert transform["residual_definition"] == "label_m_minus_raw_prediction_m"
    assert offsets_for_horizon(transform, 0) == {"obstacle_clearance": pytest.approx(1.0), "inter_agent_clearance": pytest.approx(-1.0)}
    with pytest.raises(ValueError, match="unavailable"):
        offsets_for_horizon(transform, 2)


def test_calibrated_ledger_requires_transform_hash_binding() -> None:
    transform = {
        "version": 1,
        "by_horizon": [
            {
                "horizon_index": 0,
                "obstacle_clearance": {"offset_m": 0.1},
                "inter_agent_clearance": {"offset_m": -0.1},
            }
        ],
    }
    digest = hashlib.sha256(json.dumps(transform, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload = {
        "ledger_type": SafeCaptureReliabilityLedger.LEDGER_TYPE_V3_CALIBRATED,
        "ledger_version": 3,
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "immutable_after_calibration": True,
        "clearance_calibration": transform,
        "clearance_calibration_sha256": digest,
        "source": {
            "checkpoint_sha256": "a" * 64,
            "calibration_dataset_sha256": "b" * 64,
            "clearance_calibration_sha256": digest,
        },
        "entries": {make_safe_capture_global_key(0): {"credit": 0.9, "sample_count": 1000}},
        "decision_policy": {
            "states": ["trusted", "fallback_nominal", "safe_hold"],
            "minimum_sample_count": 128,
            "minimum_credit": 0.65,
            "maximum_observation_age_steps": 45.0,
            "safe_hold_uncertainty_threshold": 0.4,
            "safe_hold_ttc_seconds": 0.3,
        },
    }
    ledger = SafeCaptureReliabilityLedger(payload)
    obstacle, inter = ledger.calibrated_clearance_m(0, np.array([1.0]), np.array([2.0]))
    np.testing.assert_allclose(obstacle, [1.1])
    np.testing.assert_allclose(inter, [1.9])
    payload["source"]["clearance_calibration_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="transform hash"):
        SafeCaptureReliabilityLedger(payload)
