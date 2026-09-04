from __future__ import annotations

from scripts.audit_jepa_safe_capture_v5_temporal_ledger import _temporal_signals


def _ranking(*, margin: float = 0.002, uncertainty: float = 0.1, risk: float = 0.2) -> dict[str, object]:
    return {
        "top_two_margin_m": margin,
        "predicted_uncertainty": [uncertainty] * 5,
        "predicted_cbf_risk": [risk] * 5,
        "predicted_min_clearance_m": [0.8] * 5,
        "predicted_min_ttc_s": [1.0] * 5,
    }


def _observation(age: float = 0.0) -> dict[str, object]:
    return {"target_observation_age_steps": [age, age]}


def _kwargs() -> dict[str, float]:
    return {
        "margin_threshold": 0.0015,
        "clearance_floor": 0.35,
        "uncertainty_threshold": 0.40,
        "risk_threshold": 0.60,
        "ttc_threshold": 0.30,
        "stale_age_steps": 45.0,
        "uncertainty_spike_threshold": 0.05,
        "risk_spike_threshold": 0.10,
    }


def test_temporal_signals_detect_low_margin_stale_and_joint_risk() -> None:
    ranking = _ranking(margin=0.001, uncertainty=0.45, risk=0.8)
    ranking["predicted_min_clearance_m"] = [0.2] * 5
    ranking["predicted_min_ttc_s"] = [0.2] * 5
    signals, values = _temporal_signals(ranking, _observation(age=46.0), None, **_kwargs())

    assert all(
        signals[name]
        for name in ("low_margin", "clearance_floor", "uncertainty_high", "risk_ttc_high", "stale_observation")
    )
    assert values["uncertainty"] == 0.45
    assert values["max_observation_age"] == 46.0


def test_temporal_signals_detect_sequential_spikes_without_false_positive() -> None:
    previous = {"uncertainty": 0.10, "risk": 0.20}
    signals, _ = _temporal_signals(
        _ranking(uncertainty=0.16, risk=0.35),
        _observation(),
        previous,
        **_kwargs(),
    )
    assert signals["uncertainty_spike"]
    assert signals["risk_spike"]
    assert not signals["low_margin"]
    assert not signals["stale_observation"]


def test_temporal_signals_are_quiet_for_nominal_context() -> None:
    signals, values = _temporal_signals(_ranking(), _observation(), None, **_kwargs())
    assert not any(signals.values())
    assert values["clearance"] == 0.8
