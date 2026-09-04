from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_jepa_safe_capture_v2_paired import _latency_stats, _observation_queue_age_steps


def test_latency_stats_returns_required_quantiles_and_rejects_invalid_values() -> None:
    stats = _latency_stats([1.0, 2.0, 4.0, 8.0])
    assert stats["count"] == 4
    assert stats["mean_ms"] == pytest.approx(3.75)
    assert stats["p50_ms"] == pytest.approx(3.0)
    assert stats["p95_ms"] >= stats["p50_ms"]
    assert stats["p99_ms"] <= stats["max_ms"]

    with pytest.raises(ValueError, match="finite and non-negative"):
        _latency_stats([1.0, np.nan])
    with pytest.raises(ValueError, match="finite and non-negative"):
        _latency_stats([1.0, -0.1])


def test_observation_queue_age_uses_oldest_online_signal() -> None:
    observation = {
        "message_age_steps": np.array([1.0, 4.0]),
        "target_observation_age_steps": np.array([2.0, 3.0]),
    }
    assert _observation_queue_age_steps(observation) == pytest.approx(4.0)
    assert _observation_queue_age_steps({}) == 0.0

    with pytest.raises(ValueError, match="non-finite"):
        _observation_queue_age_steps({"message_age_steps": [np.inf]})
