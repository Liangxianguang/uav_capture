from __future__ import annotations

import numpy as np

from scripts.build_route_identity_ledger import _safety_priority_credit


def _inputs(target_error: np.ndarray, *, safe: bool = True) -> dict[str, np.ndarray]:
    size = target_error.shape
    return {
        "target_error_m": target_error,
        "clearance_error_m": np.full(size, 0.1 if safe else 2.0),
        "clearance_overprediction_m": np.full(size, 0.1 if safe else 1.5),
        "visibility_error": np.full(size, 0.1 if safe else 0.8),
        "intervention_error": np.full(size, 0.1 if safe else 0.8),
        "feasibility_brier": np.full(size, 0.02 if safe else 0.8),
        "route_geometry_brier": np.full(size, 0.02 if safe else 0.8),
        "route_termination_brier": np.full(size, 0.02 if safe else 0.8),
        "route_progress_error": np.full(size, 0.05 if safe else 1.0),
    }


def test_safety_priority_credit_is_finite_and_bounded() -> None:
    credit, components = _safety_priority_credit(**_inputs(np.array([0.2, 0.8])))

    assert np.isfinite(credit).all()
    assert np.all((credit >= 0.0) & (credit <= 1.0))
    assert set(components) == {
        "feasibility",
        "clearance",
        "clearance_conservatism",
        "route_geometry",
        "route_termination",
        "intervention",
        "visibility",
        "route_progress",
        "target",
        "credit",
    }


def test_target_error_cannot_override_safety_priority() -> None:
    perfect_target, _ = _safety_priority_credit(**_inputs(np.array([0.0])))
    poor_target, _ = _safety_priority_credit(**_inputs(np.array([100.0])))
    safe_poor_target, _ = _safety_priority_credit(**_inputs(np.array([100.0]), safe=True))
    unsafe_perfect_target, _ = _safety_priority_credit(**_inputs(np.array([0.0]), safe=False))

    assert float((perfect_target - poor_target).item()) <= 0.03 + 1e-9
    assert float(safe_poor_target.item()) > float(unsafe_perfect_target.item())
