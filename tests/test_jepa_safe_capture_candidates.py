from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from encirclement3d.jepa_safe_capture_candidates import (
    CANDIDATE_LABELS,
    SafeCaptureCandidateBatch,
    SafeCaptureCandidateConfig,
    SafeCaptureCandidateHistory,
    make_safe_capture_candidate_chunks,
)
from encirclement3d.jepa_safe_capture_ranker import (
    SafeCaptureJEPARanker,
    SafeCaptureRankerConfig,
    _conservative_margin_for_comparison,
    _candidate_specific_separation,
    _fixed_point_score_key,
    _fixed_point_score_keys,
)
from encirclement3d.prediction import InteractionAwareActionConditionedSafeCaptureJEPAPredictor
from encirclement3d.reliability import SafeCaptureReliabilityLedger, make_safe_capture_global_key


def _observation(defender_count: int = 2) -> dict[str, object]:
    positions = np.zeros((defender_count, 3), dtype=np.float64)
    positions[:, 1] = np.arange(defender_count, dtype=np.float64)
    return {
        "defender_positions": positions,
        "target_belief_positions": np.column_stack(
            [np.full(defender_count, 2.0), np.zeros(defender_count), np.full(defender_count, 1.0)]
        ),
        "target_belief_velocities": np.zeros_like(positions),
        "target_visible": np.ones(defender_count, dtype=bool),
        "target_observation_age_steps": np.zeros(defender_count),
        "obstacles": [{"shape": "cylinder"}],
    }


def _predictor() -> InteractionAwareActionConditionedSafeCaptureJEPAPredictor:
    torch.manual_seed(7)
    return InteractionAwareActionConditionedSafeCaptureJEPAPredictor(
        input_dim=63,
        horizon_count=4,
        hidden_dim=16,
        latent_dim=8,
        interaction_group_slices=((0, 15), (15, 33), (33, 48), (48, 63)),
    )


class _FakeHistory:
    defender_count = 2
    predictor = SimpleNamespace(action_dim=3, input_dim=63, horizon_count=4)

    def predict_candidates_multitask(self, actions: np.ndarray, *, horizon_index: int):
        assert horizon_index == 2
        count = actions.shape[0]
        means = np.zeros((count, self.defender_count, 3), dtype=np.float32)
        # The predicted target is fixed at x=2 m. A positive x action therefore
        # makes the future defender closer, proving the scorer follows actions.
        means[..., 0] = 0.20
        std = np.full((count, self.defender_count, 3), 0.01, dtype=np.float32)
        auxiliary = {
            "obstacle_clearance_lower_quantile": np.full((count, self.defender_count), 1.0, dtype=np.float32),
            "inter_agent_clearance_lower_quantile": np.full((count, self.defender_count), 1.0, dtype=np.float32),
            "pairwise_ttc": np.full((count, self.defender_count), 10.0, dtype=np.float32),
            "target_visibility_logit": np.full((count, self.defender_count), 10.0, dtype=np.float32),
            "cbf_intervention_logit": np.full((count, self.defender_count), -10.0, dtype=np.float32),
            "cbf_correction": np.zeros((count, self.defender_count), dtype=np.float32),
            "cbf_qp_feasibility_logit": np.full((count, self.defender_count), 10.0, dtype=np.float32),
            "action_consistency": actions.copy(),
        }
        return means, std, auxiliary


def _batch(actions: np.ndarray, valid: np.ndarray | None = None) -> SafeCaptureCandidateBatch:
    chunks = np.repeat(actions[:, None, :, :], 3, axis=1)
    return SafeCaptureCandidateBatch(
        chunks=chunks,
        labels=CANDIDATE_LABELS,
        valid_mask=np.ones(5, dtype=bool) if valid is None else valid,
        rejection_reasons=tuple(() for _ in range(5)),
    )


def _ledger(credit: float, *, ood: bool = False) -> SafeCaptureReliabilityLedger:
    return SafeCaptureReliabilityLedger(
        {
            "ledger_type": SafeCaptureReliabilityLedger.LEDGER_TYPE,
            "ledger_version": 2,
            "not_a_locked_test": True,
            "immutable_after_calibration": True,
            "source": {"checkpoint_sha256": "a" * 64, "calibration_dataset_sha256": "b" * 64},
            "entries": {make_safe_capture_global_key(2): {"credit": credit, "sample_count": 1000}},
            "decision_policy": {
                "states": ["trusted", "fallback_nominal", "safe_hold"],
                "minimum_sample_count": 128,
                "minimum_credit": 0.65,
                "maximum_observation_age_steps": 45.0,
                "safe_hold_uncertainty_threshold": 0.40,
                "safe_hold_ttc_seconds": 0.30,
            },
        }
    )


def test_candidate_contract_is_fixed_and_nominal_is_exact_anchor() -> None:
    nominal = np.array([[0.5, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=np.float64)
    batch = make_safe_capture_candidate_chunks(nominal, _observation(), config=SafeCaptureCandidateConfig())
    assert batch.chunks.shape == (5, 3, 2, 3)
    assert batch.labels == CANDIDATE_LABELS
    np.testing.assert_array_equal(batch.chunks[0], np.repeat(nominal[None], 3, axis=0))
    assert bool(batch.valid_mask[0])
    assert all(np.allclose(chunk, chunk[:1], rtol=0.0, atol=1e-9) for chunk in batch.chunks)


def test_candidate_generator_rejects_nonfinite_nominal_and_marks_dynamics_failures() -> None:
    nominal = np.array([[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]])
    try:
        make_safe_capture_candidate_chunks(nominal, _observation())
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite nominal action was accepted")

    fast = np.full((2, 3), 5.0, dtype=np.float64)
    batch = make_safe_capture_candidate_chunks(
        fast,
        _observation(),
        config=SafeCaptureCandidateConfig(max_speed_mps=5.0),
        previous_action=np.zeros_like(fast),
    )
    assert not bool(batch.valid_mask[0])
    assert "acceleration_limit" in batch.rejection_reasons[0]
    assert "nominal_infeasible" in batch.rejection_reasons[0]


def test_history_uses_requested_three_step_horizon_and_is_action_conditioned() -> None:
    history = SafeCaptureCandidateHistory(
        _predictor(), defender_count=2, device=torch.device("cpu"), history_length=8, action_scale=5.0
    )
    base = np.zeros((2, 63), dtype=np.float32)
    history.reset(base)
    actions = np.zeros((2, 2, 3), dtype=np.float32)
    actions[1, :, 0] = 1.0
    mean, std, auxiliary = history.predict_candidates_multitask(actions, horizon_index=2)
    assert mean.shape == (2, 2, 3)
    assert std.shape == (2, 2, 3)
    assert auxiliary["pairwise_ttc"].shape == (2, 2)
    assert auxiliary["action_consistency"].shape == (2, 2, 3)
    assert not np.allclose(mean[0], mean[1])


def test_candidate_separation_is_local_to_each_candidate() -> None:
    costs = np.array([0.0, 0.20, 1.00, 1.30, np.inf], dtype=np.float64)
    separation = _candidate_specific_separation(costs)

    np.testing.assert_allclose(separation, [0.20, 0.20, 0.30, 0.30, 0.0])


def test_candidate_separation_is_zero_for_single_finite_candidate() -> None:
    separation = _candidate_specific_separation(np.array([np.inf, 2.0, np.inf]))
    np.testing.assert_array_equal(separation, np.zeros(3, dtype=np.float64))


def test_ranker_follows_action_and_executes_only_first_step() -> None:
    history = _FakeHistory()
    ranker = SafeCaptureJEPARanker(history, config=SafeCaptureRankerConfig())
    actions = np.zeros((5, 2, 3), dtype=np.float64)
    actions[1, :, 0] = 1.0
    result = ranker.rank(_observation(), _batch(actions))
    assert result.execution_mode == "trusted"
    assert result.selected_index == 1
    np.testing.assert_array_equal(result.selected_action, result.selected_chunk[0])
    assert result.selected_chunk.shape == (3, 2, 3)
    assert result.trace.predicted_visibility[1] > 0.99
    assert result.trace.scores[1] < result.trace.scores[0]
    assert result.trace.candidate_rejection_reasons == ((), (), (), (), ())


def test_ranker_nominal_fallback_and_safe_hold_never_use_untrusted_candidate() -> None:
    actions = np.zeros((5, 2, 3), dtype=np.float64)
    actions[1, :, 0] = 1.0
    observation = _observation()
    fallback = SafeCaptureJEPARanker(
        _FakeHistory(),
        reliability_ledger=_ledger(0.20),
        context_defaults={"layout_signature": "scenario_0", "target_motion_mode": "flee_persistence"},
    ).rank(observation, _batch(actions))
    assert fallback.execution_mode == "fallback_nominal"
    assert fallback.selected_index == 0
    assert fallback.fallback_reason == "low_credit"

    safe_hold = SafeCaptureJEPARanker(
        _FakeHistory(),
        reliability_ledger=_ledger(0.90),
        context_defaults={"layout_signature": "scenario_0", "target_motion_mode": "flee_persistence", "ood": True},
    ).rank(observation, _batch(actions))
    assert safe_hold.execution_mode == "safe_hold"
    assert safe_hold.selected_index == 0
    assert safe_hold.fallback_reason == "ood"

    missing_provenance = SafeCaptureJEPARanker(
        _FakeHistory(),
        reliability_ledger=_ledger(0.90),
    ).rank(observation, _batch(actions))
    assert missing_provenance.execution_mode == "safe_hold"
    assert missing_provenance.fallback_reason == "ood"


def test_zero_perturbation_keeps_nominal_anchor() -> None:
    nominal = np.array([[0.5, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=np.float64)
    batch = make_safe_capture_candidate_chunks(
        nominal,
        _observation(),
        config=SafeCaptureCandidateConfig(perturbation_mps=0.0),
    )
    result = SafeCaptureJEPARanker(_FakeHistory()).rank(_observation(), batch)
    assert result.selected_index == 0
    np.testing.assert_array_equal(result.selected_action, nominal)


def test_ranker_uses_candidate_index_to_break_score_ties() -> None:
    history = _FakeHistory()
    ranker = SafeCaptureJEPARanker(
        history,
        config=SafeCaptureRankerConfig(score_tie_tolerance_m=10.0),
    )
    actions = np.zeros((5, 2, 3), dtype=np.float64)
    actions[1, :, 0] = 1.0
    result = ranker.rank(_observation(), _batch(actions))
    assert result.selected_index == 0


def test_ranker_tie_tolerance_covers_cpu_cuda_roundoff_scale() -> None:
    config = SafeCaptureRankerConfig()
    assert config.score_tie_tolerance_m == 5e-4

    # A 2.05e-4 score gap was enough to produce different CPU/CUDA choices in
    # the previous 2e-4 protocol.  The fixed tolerance must classify it as a
    # tie and use the candidate index as the secondary key.
    assert 2.05e-4 <= config.score_tie_tolerance_m


def test_ranker_abstains_to_nominal_when_top_two_margin_is_small() -> None:
    history = _FakeHistory()
    actions = np.zeros((5, 2, 3), dtype=np.float64)
    actions[1, :, 0] = 1.0
    result = SafeCaptureJEPARanker(
        history,
        config=SafeCaptureRankerConfig(top_two_abstention_margin_m=10.0),
    ).rank(_observation(), _batch(actions))

    assert result.selected_index == 0
    assert result.execution_mode == "fallback_nominal"
    assert result.fallback_reason == "top_two_margin_abstention"
    assert result.trace.rank_abstention_reason == "top_two_margin_abstention"


def test_ranker_uses_score_tolerance_around_abstention_boundary() -> None:
    history = _FakeHistory()
    actions = np.zeros((5, 2, 3), dtype=np.float64)
    actions[1, :, 0] = 0.006
    # The fake model produces a margin close to the configured abstention
    # threshold. The tolerance band must make the decision deterministic.
    result = SafeCaptureJEPARanker(
        history,
        config=SafeCaptureRankerConfig(
            top_two_abstention_margin_m=0.0015,
            score_tie_tolerance_m=10.0,
        ),
    ).rank(_observation(), _batch(actions))
    assert result.trace.top_two_margin_m == pytest.approx(0.0015, abs=1e-6)
    assert result.trace.top_two_margin_comparison_m == pytest.approx(0.0015, abs=1e-6)
    assert result.execution_mode == "fallback_nominal"
    assert result.fallback_reason == "nominal_anchor_tie"
    assert result.trace.rank_abstention_reason == "nominal_anchor_tie"


def test_conservative_margin_quantization_routes_cpu_cuda_boundary_to_same_path() -> None:
    quantum = 0.0005
    # These are the observed CUDA/CPU margins from the failed replay.  The
    # quantizer is intentionally paired with a separate 0.0005 m safety band;
    # both values must therefore route to nominal despite opposite rounding.
    cuda_margin = _conservative_margin_for_comparison(0.0025176494516, quantum)
    cpu_margin = _conservative_margin_for_comparison(0.0024942341255, quantum)
    assert cuda_margin == pytest.approx(0.0025, abs=1e-12)
    assert cpu_margin == pytest.approx(0.0020, abs=1e-12)
    abstention_limit = 0.0015 + 0.0005 + 0.0005
    assert cuda_margin <= abstention_limit
    assert cpu_margin <= abstention_limit


def test_ranker_safety_band_covers_observed_margin_drift() -> None:
    config = SafeCaptureRankerConfig(
        top_two_abstention_margin_m=0.0015,
        score_tie_tolerance_m=0.0005,
        score_comparison_quantum_m=0.0005,
        score_comparison_safety_band_m=0.001,
    )
    limit = (
        config.top_two_abstention_margin_m
        + config.score_tie_tolerance_m
        + config.score_comparison_safety_band_m
    )
    assert limit == pytest.approx(0.003, abs=1e-12)
    assert _conservative_margin_for_comparison(0.0030021580729, config.score_comparison_quantum_m) <= limit
    assert _conservative_margin_for_comparison(0.0029707248747, config.score_comparison_quantum_m) <= limit


def test_ranker_records_raw_and_quantized_abstention_margins() -> None:
    history = _FakeHistory()
    actions = np.zeros((5, 2, 3), dtype=np.float64)
    actions[1, :, 0] = 0.006
    result = SafeCaptureJEPARanker(
        history,
        config=SafeCaptureRankerConfig(
            top_two_abstention_margin_m=0.0015,
            score_tie_tolerance_m=0.0005,
            score_comparison_quantum_m=0.0005,
        ),
    ).rank(_observation(), _batch(actions))
    trace = result.trace.as_dict()
    assert "top_two_margin_m" in trace
    assert "top_two_margin_comparison_m" in trace
    assert trace["rank_abstention_reason"] == "top_two_margin_abstention"


def test_fixed_point_score_key_is_round_half_up_and_nonfinite_is_ineligible() -> None:
    assert _fixed_point_score_key(1.00024, 0.0005) == 2000
    assert _fixed_point_score_key(1.00025, 0.0005) == 2001
    assert _fixed_point_score_key(-1.00025, 0.0005) == -2001
    assert _fixed_point_score_key(float("inf"), 0.0005) is None
    assert _fixed_point_score_keys(np.array([1.0, np.nan]), 0.0005) == (2000, None)


def test_robust_fixed_point_quantum_collapses_observed_cpu_cuda_score_drift() -> None:
    cuda_scores = np.array([3.3318943, 3.3052830, 3.3124670, 3.3363080, 3.3086671])
    cpu_scores = np.array([3.3319224, 3.3053500, 3.3125565, 3.3363672, 3.3087504])
    cuda_keys = _fixed_point_score_keys(cuda_scores, 0.0015)
    cpu_keys = _fixed_point_score_keys(cpu_scores, 0.0015)
    assert cuda_keys == cpu_keys == (2221, 2204, 2208, 2224, 2206)
    ordered = sorted(range(5), key=lambda index: (cuda_keys[index], index))
    assert ordered == sorted(range(5), key=lambda index: (cpu_keys[index], index))
    assert (cuda_keys[4] - cuda_keys[1]) * 0.0015 == pytest.approx(0.003)


def test_fixed_point_profile_uses_discrete_candidate_order_and_trace_keys() -> None:
    history = _FakeHistory()
    actions = np.zeros((5, 2, 3), dtype=np.float64)
    actions[1, :, 0] = 1.0
    result = SafeCaptureJEPARanker(
        history,
        config=SafeCaptureRankerConfig(
            fixed_point_score_comparison=True,
            score_comparison_quantum_m=0.0005,
        ),
    ).rank(_observation(), _batch(actions))

    trace = result.trace
    assert trace.fixed_point_score_comparison is True
    assert len(trace.score_comparison_keys) == 5
    assert all(key is not None for key in trace.score_comparison_keys)
    assert trace.candidate_order[0] == result.selected_index
    assert tuple(
        sorted(
            trace.candidate_order,
            key=lambda index: (trace.score_comparison_keys[index], index),
        )
    ) == trace.candidate_order


def test_fixed_point_near_tie_uses_nominal_anchor() -> None:
    actions = np.zeros((5, 2, 3), dtype=np.float64)
    actions[1, :, 0] = 1.0
    result = SafeCaptureJEPARanker(
        _FakeHistory(),
        config=SafeCaptureRankerConfig(
            fixed_point_score_comparison=True,
            score_comparison_quantum_m=1.0,
            score_tie_tolerance_m=0.5,
            score_comparison_safety_band_m=0.5,
        ),
    ).rank(_observation(), _batch(actions))

    assert result.selected_index == 0
    assert result.execution_mode == "trusted"


def test_fixed_point_nominal_tie_is_reported_as_deterministic_fallback() -> None:
    actions = np.zeros((5, 2, 3), dtype=np.float64)
    actions[1, :, 0] = 1.0
    result = SafeCaptureJEPARanker(
        _FakeHistory(),
        config=SafeCaptureRankerConfig(
            fixed_point_score_comparison=True,
            score_comparison_quantum_m=0.004,
            score_tie_tolerance_m=10.0,
            score_comparison_safety_band_m=0.004,
        ),
    ).rank(_observation(), _batch(actions))
    assert result.selected_index == 0
    assert result.execution_mode == "fallback_nominal"
    assert result.fallback_reason == "nominal_anchor_tie"


def test_ranker_exposes_hold_and_hysteresis_inputs_without_changing_cbf_boundary() -> None:
    history = _FakeHistory()
    actions = np.zeros((5, 2, 3), dtype=np.float64)
    actions[1, :, 0] = 1.0
    result = SafeCaptureJEPARanker(
        history,
        config=SafeCaptureRankerConfig(candidate_hysteresis_margin_m=10.0, minimum_hold_steps=2),
    ).rank(
        _observation(),
        _batch(actions),
        previous_selected_index=1,
        hold_steps_remaining=2,
    )

    assert result.selected_index == 1
    assert result.execution_mode == "trusted"
    assert result.trace.hysteresis_applied is False
    assert result.trace.hold_steps_remaining == 1


def test_ranker_exposes_finite_latency_breakdown_without_affecting_selection() -> None:
    history = _FakeHistory()
    actions = np.zeros((5, 2, 3), dtype=np.float64)
    actions[1, :, 0] = 1.0
    result = SafeCaptureJEPARanker(history).rank(_observation(), _batch(actions))

    assert result.selected_index == 1
    timing = result.trace.as_dict()
    for field in (
        "jepa_inference_latency_ms",
        "ledger_route_latency_ms",
        "ranker_compute_latency_ms",
        "rank_total_latency_ms",
    ):
        assert np.isfinite(float(timing[field]))
        assert float(timing[field]) >= 0.0
    assert float(timing["rank_total_latency_ms"]) >= float(timing["jepa_inference_latency_ms"])


def test_v12_calibrated_ranking_profile_accepts_declaration_only_fields() -> None:
    from scripts.evaluate_jepa_safe_capture_v2_paired import _ranker_config

    config = _ranker_config(
        "m3",
        {
            "profile": "p12_calibrated_clearance_v1",
            "clearance_transform": "checkpoint_bound_q10_residual_offset_m",
            "clearance_quantile": 0.10,
            "cbf_margin_changed": False,
            "minimum_predicted_clearance_m": 0.15,
        },
    )
    assert config.minimum_predicted_clearance_m == 0.15
    assert config.score_comparison_quantum_m == 5e-4

    deterministic = _ranker_config(
        "m3",
        {
            "profile": "p12_deterministic_v2",
            "score_comparison_quantum_m": 5e-4,
        },
    )
    assert deterministic.fixed_point_score_comparison is True


def test_projected_candidates_use_the_reachable_first_step_envelope() -> None:
    nominal = np.full((2, 3), 5.0, dtype=np.float64)
    previous = np.zeros_like(nominal)
    batch = make_safe_capture_candidate_chunks(
        nominal,
        _observation(),
        config=SafeCaptureCandidateConfig(project_to_reachable_dynamics=True),
        previous_action=previous,
    )

    assert np.all(batch.valid_mask)
    assert np.max(np.linalg.norm(batch.chunks[:, 0] - previous[None], axis=-1)) <= 0.6 + 1e-8
    assert np.max(np.linalg.norm(batch.chunks[:, 0], axis=-1)) <= 5.0 + 1e-8
