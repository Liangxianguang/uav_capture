from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from encirclement3d.jepa_safe_capture_candidates import (
    CANDIDATE_LABELS,
    SafeCaptureCandidateBatch,
    SafeCaptureCandidateConfig,
    SafeCaptureCandidateHistory,
    make_safe_capture_candidate_chunks,
)
from encirclement3d.jepa_safe_capture_ranker import SafeCaptureJEPARanker, SafeCaptureRankerConfig
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
