from __future__ import annotations

import torch
import numpy as np
import yaml
from pathlib import Path

from encirclement3d.prediction import (
    ActionConditionedCandidateHistory,
    ActionConditionedCandidateReranker,
    ActionConditionedJEPAPredictor,
    InteractionAwareActionConditionedJEPAPredictor,
    HistoryTargetPredictor,
    LearnedPredictionObserver,
    deterministic_mse,
    gaussian_nll,
    make_action_candidates,
)
from encirclement3d.learning import RecurrentCentralizedSharedActorCritic
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_history_target_predictor_shapes_and_finite_losses() -> None:
    model = HistoryTargetPredictor(input_dim=52, horizon_count=4, hidden_dim=16)
    inputs = torch.randn(5, 8, 52)
    target = torch.randn(5, 4, 3)
    mean, log_variance = model(inputs)
    assert mean.shape == (5, 4, 3)
    assert log_variance.shape == (5, 4, 3)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(log_variance).all()
    assert torch.isfinite(gaussian_nll(mean, log_variance, target))
    assert torch.isfinite(deterministic_mse(mean, target))


def test_history_target_predictor_rejects_wrong_feature_dimension() -> None:
    model = HistoryTargetPredictor(input_dim=52, horizon_count=4, hidden_dim=16)
    try:
        model(torch.randn(2, 8, 48))
    except ValueError as error:
        assert "Expected" in str(error)
    else:
        raise AssertionError("Expected wrong feature dimension to raise ValueError.")


def test_learned_prediction_observer_preserves_actor_width_and_replaces_block() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "capture_radius_pursuit_prediction_dev.yaml").read_text(encoding="utf-8")
    )
    config["task"]["pursuit"]["include_uncertainty_features"] = True
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.55)
    predictor = HistoryTargetPredictor(input_dim=52, horizon_count=4, hidden_dim=16)
    adapter = LearnedPredictionObserver(env, predictor, torch.device("cpu"), history_length=8, horizon_index=2)
    observation = env.reset(seed=520201)
    augmented = adapter.reset(observation)
    assert augmented.shape == (4, 52)
    feature_slice = env.prediction_feature_slice()
    assert feature_slice == slice(15, 19)
    assert np.isfinite(augmented).all()
    next_observation, *_ = env.step(np.zeros((4, 3)))
    repeated = adapter.observe(next_observation)
    assert repeated.shape == (4, 52)
    assert adapter.last_prediction_mean is not None
    assert adapter.last_prediction_std is not None


def test_recurrent_actor_sequence_matches_step_rollout_and_resets() -> None:
    torch.manual_seed(520203)
    model = RecurrentCentralizedSharedActorCritic(
        local_observation_dim=6,
        centralized_state_dim=9,
        hidden_dim=16,
    )
    local = torch.randn(1, 4, 4, 6)
    actions = torch.randn(1, 4, 4, 3).clamp(-1.0, 1.0)
    reset_masks = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)
    initial_hidden = model.initial_actor_hidden(4, batch_size=1)
    _log_probabilities, _entropy, sequence_means = model.evaluate_actions_sequence(
        local,
        initial_hidden,
        reset_masks,
        actions,
        action_scale=1.0,
    )
    hidden = initial_hidden[0]
    step_means: list[torch.Tensor] = []
    for index in range(local.shape[1]):
        distribution, hidden = model.distribution_step(local[0, index], hidden, reset_masks[0, index])
        step_means.append(distribution.mean)
    assert torch.allclose(sequence_means[0], torch.stack(step_means), atol=1e-6)


def test_action_conditioned_jepa_shapes_and_target_information_boundary() -> None:
    torch.manual_seed(520204)
    model = ActionConditionedJEPAPredictor(
        input_dim=63,
        horizon_count=4,
        action_dim=3,
        hidden_dim=16,
        latent_dim=12,
    )
    observations = torch.randn(5, 8, 63)
    actions = torch.randn(5, 8, 3)
    labels = torch.randn(5, 4, 3)
    mean, log_variance, latent = model(observations, actions)
    assert mean.shape == labels.shape
    assert log_variance.shape == labels.shape
    assert latent.shape == (5, 4, 12)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(log_variance).all()
    target_latent = model.target_latent(labels)
    assert target_latent.shape == latent.shape
    assert not target_latent.requires_grad


def test_action_conditioned_candidate_history_and_reranker_are_deterministic() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml").read_text(encoding="utf-8")
    )
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=3, target_speed_scale=0.55)
    observation = env.reset(seed=520205)
    base = __import__("encirclement3d.observation_encoding", fromlist=["policy_observations"]).policy_observations(
        env, observation
    )
    predictor = ActionConditionedJEPAPredictor(63, 4, hidden_dim=16, latent_dim=12)
    history = ActionConditionedCandidateHistory(env, predictor, torch.device("cpu"), history_length=8, action_scale=5.0)
    history.reset(base)
    candidates = make_action_candidates(np.zeros((4, 3), dtype=np.float32), perturbation_mps=0.2, candidate_count=5)
    reranker = ActionConditionedCandidateReranker(history, horizon_seconds=0.5, position_extent=10.0)
    selected, diagnostics = reranker.select(observation, candidates)
    assert selected.shape == (4, 3)
    assert 0 <= diagnostics.selected_index < 5
    assert len(diagnostics.scores) == 5
    assert np.isfinite(np.asarray(diagnostics.scores)).all()


def test_interaction_aware_jepa_uses_structured_groups_and_factory() -> None:
    from encirclement3d.prediction import build_action_conditioned_predictor

    config = {
        "input_dim": 63,
        "horizon_count": 4,
        "action_dim": 3,
        "hidden_dim": 16,
        "latent_dim": 12,
        "num_layers": 1,
        "interaction_group_slices": [[0, 15], [15, 33], [33, 48], [48, 63]],
    }
    model = build_action_conditioned_predictor("interaction_aware_action_conditioned_jepa", config)
    assert isinstance(model, InteractionAwareActionConditionedJEPAPredictor)
    mean, log_variance, latent = model(torch.randn(3, 8, 63), torch.randn(3, 8, 3))
    assert mean.shape == (3, 4, 3)
    assert log_variance.shape == mean.shape
    assert latent.shape == (3, 4, 12)
    assert torch.isfinite(mean).all()
