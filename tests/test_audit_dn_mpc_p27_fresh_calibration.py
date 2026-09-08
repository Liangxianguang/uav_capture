from __future__ import annotations

import numpy as np
import torch

from scripts.audit_dn_mpc_p27_fresh_calibration import (
    _summary,
    apply_ood_profile,
    fit_ood_profile,
)


def _tensors() -> dict[str, torch.Tensor]:
    return {
        "inputs": torch.zeros((4, 8, 63), dtype=torch.float32),
        "action_history": torch.zeros((4, 8, 3), dtype=torch.float32),
        "route_action_chunk": torch.zeros((4, 5, 3), dtype=torch.float32),
        "route_pairwise_relative_action_chunk": torch.zeros((4, 5, 9), dtype=torch.float32),
        "sample_type": torch.zeros(4, dtype=torch.float32),
    }


def test_ood_profile_is_finite_and_deterministic() -> None:
    tensors = _tensors()
    tensors["inputs"][1, -1, 0] = 2.0
    profile = fit_ood_profile(tensors)
    distances = apply_ood_profile(tensors, profile)
    assert profile["feature_dim"] == 126
    assert np.isfinite(distances).all()
    assert distances[0] < distances[1]


def test_summary_handles_empty_values() -> None:
    assert _summary(np.asarray([], dtype=np.float64))["count"] == 0
    result = _summary(np.asarray([1.0, 2.0, np.nan], dtype=np.float64))
    assert result["count"] == 2
    assert result["p95"] > 1.0
