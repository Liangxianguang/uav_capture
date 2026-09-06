from __future__ import annotations

import math

import numpy as np

from scripts.audit_jepa_safe_capture_v5_settled_counterfactual import (
    _best_candidate,
    _candidate_chunks_from_trace,
    _ece,
    _jsonable,
    _kendall,
    _softmax_probabilities,
    _spearman,
)


class _EnvStub:
    n_defenders = 2


def test_route_trace_candidate_chunks_support_twelve_candidates() -> None:
    candidates = [
        {
            "label": f"route_{index}",
            "action_chunk": np.zeros((3, 2, 3)).tolist(),
            "fallback_only": index == 11,
        }
        for index in range(12)
    ]
    record = {
        "route_runtime": {
            "routes": {"candidates": candidates},
            "candidate_batch": {"valid_mask": [True] * 12},
            "cbf_counterfactuals": [{"accepted": True} for _ in range(12)],
        }
    }
    chunks, cbf_eligible, labels, valid = _candidate_chunks_from_trace(
        record, {}, env=_EnvStub(), previous_action=np.zeros((2, 3))
    )
    assert chunks.shape == (12, 3, 2, 3)
    assert cbf_eligible.tolist() == [True] * 11 + [False]
    assert labels[0] == "route_0"
    assert valid.tolist() == [True] * 12


def test_jsonable_converts_numpy_and_nonfinite_values() -> None:
    value = _jsonable({"finite": np.float64(1.25), "nan": float("nan"), "nested": np.array([1, 2])})
    assert value == {"finite": 1.25, "nan": None, "nested": [1, 2]}


def test_rank_correlations_respect_score_direction() -> None:
    # Lower model scores are intended to mean better candidates, while larger
    # settled progress is better, hence the expected negative correlation.
    scores = [1.0, 2.0, 3.0]
    progress = [3.0, 2.0, 1.0]
    assert _spearman(scores, progress) == -1.0
    assert _kendall(scores, progress) == -1.0


def test_rank_correlations_handle_ties_without_nan() -> None:
    assert math.isfinite(float(_spearman([1.0, 1.0, 2.0], [3.0, 2.0, 1.0])))
    assert math.isfinite(float(_kendall([1.0, 1.0, 2.0], [3.0, 2.0, 1.0])))


def test_softmax_and_ece_proxy_are_finite_and_masked() -> None:
    probabilities = _softmax_probabilities([1.0, 2.0, float("inf")], [True, True, False])
    assert np.isclose(float(np.sum(probabilities)), 1.0)
    assert probabilities[2] == 0.0
    assert _ece(probabilities.tolist(), [1.0, 0.0, 0.0]) is not None


def test_best_candidate_prioritizes_safety_then_progress() -> None:
    outcomes = [
        {"settled_safe_capture": False, "settled_safety_ok": True, "settled_progress_m": 10.0, "settled_cbf_correction_norm_mps": 0.0},
        {"settled_safe_capture": True, "settled_safety_ok": True, "settled_progress_m": -10.0, "settled_cbf_correction_norm_mps": 2.0},
        {"settled_safe_capture": False, "settled_safety_ok": False, "settled_progress_m": 100.0, "settled_cbf_correction_norm_mps": 0.0},
        {"settled_safe_capture": False, "settled_safety_ok": True, "settled_progress_m": 5.0, "settled_cbf_correction_norm_mps": 0.0},
        {"settled_safe_capture": False, "settled_safety_ok": True, "settled_progress_m": 4.0, "settled_cbf_correction_norm_mps": 0.0},
    ]
    assert _best_candidate(outcomes, [True, True, True, True, True]) == 1
    assert _best_candidate(outcomes, [True, False, True, True, True]) == 0
