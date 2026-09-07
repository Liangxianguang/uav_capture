import numpy as np
import pytest
import torch

from encirclement3d.jepa_safe_capture_candidates import SafeCaptureCandidateHistory
from encirclement3d.jepa_safe_capture_ranker import ROUTE_SIDE_INDEX_BY_LABEL
from encirclement3d.prediction import (
    InteractionAwareActionConditionedRouteHardNegativeJEPAPredictor,
    InteractionAwareActionConditionedRouteJEPAPredictor,
    build_action_conditioned_predictor,
)


def _model() -> InteractionAwareActionConditionedRouteJEPAPredictor:
    torch.manual_seed(19)
    return InteractionAwareActionConditionedRouteJEPAPredictor(
        input_dim=63,
        horizon_count=5,
        hidden_dim=16,
        latent_dim=8,
        route_chunk_length=3,
        route_candidate_count=12,
        route_side_count=12,
        interaction_group_slices=((0, 15), (15, 33), (33, 48), (48, 63)),
    )


def test_route_model_exposes_route_conditioned_heads():
    model = _model().eval()
    inputs = torch.randn(4, 8, 63)
    actions = torch.randn(4, 8, 3)
    route_chunks = torch.randn(4, 3, 3)
    mean, log_variance, latent, auxiliary = model.forward_multitask(inputs, actions, route_chunks)

    assert mean.shape == (4, 5, 3)
    assert log_variance.shape == (4, 5, 3)
    assert latent.shape == (4, 5, 8)
    assert auxiliary["boundary_clearance"].shape == (4, 5)
    assert auxiliary["cbf_min_slack"].shape == (4, 5)
    assert auxiliary["route_progress"].shape == (4, 5)
    assert auxiliary["cbf_feasibility_logit"].shape == (4, 5)
    assert auxiliary["route_identity_logits"].shape == (4, 12)
    assert auxiliary["route_side_logits"].shape == (4, 12)
    assert auxiliary["route_geometry_logit"].shape == (4,)
    assert auxiliary["route_termination_logit"].shape == (4,)
    assert all(torch.isfinite(value).all() for value in auxiliary.values())


def test_route_chunk_changes_conditioned_prediction():
    model = _model().eval()
    inputs = torch.zeros(1, 8, 63)
    actions = torch.zeros(1, 8, 3)
    first = torch.zeros(1, 3, 3)
    second = first.clone()
    second[:, :, 0] = 0.7
    first_mean = model(inputs, actions, first)[0]
    second_mean = model(inputs, actions, second)[0]
    assert not torch.allclose(first_mean, second_mean)


def test_interaction_conditioned_route_chunk_changes_prediction_and_is_finite():
    torch.manual_seed(23)
    model = InteractionAwareActionConditionedRouteJEPAPredictor(
        input_dim=63,
        horizon_count=5,
        hidden_dim=16,
        latent_dim=8,
        route_chunk_length=3,
        route_interaction_chunk_dim=3,
        route_candidate_count=12,
        route_side_count=12,
        interaction_group_slices=((0, 15), (15, 33), (33, 48), (48, 63)),
    ).eval()
    inputs = torch.zeros(2, 8, 63)
    actions = torch.zeros(2, 8, 3)
    route = torch.zeros(2, 3, 3)
    relative = torch.zeros(2, 3, 3)
    changed = relative.clone()
    changed[:, :, 1] = 0.8
    first = model(inputs, actions, route, route_interaction_chunks=relative)[0]
    second = model(inputs, actions, route, route_interaction_chunks=changed)[0]
    assert not torch.allclose(first, second)
    assert torch.isfinite(second).all()


def test_legacy_route_model_rejects_interaction_feature_without_dimension():
    model = _model().eval()
    with pytest.raises(ValueError, match="route_interaction_chunk_dim"):
        model(
            torch.zeros(1, 8, 63),
            torch.zeros(1, 8, 3),
            torch.zeros(1, 3, 3),
            route_interaction_chunks=torch.zeros(1, 3, 3),
        )


def test_hard_negative_model_exposes_finite_risk_heads_and_factory_contract():
    model = build_action_conditioned_predictor(
        "interaction_aware_action_conditioned_jepa_route_identity_hard_negative_v2",
        {
            "input_dim": 63,
            "horizon_count": 5,
            "hidden_dim": 16,
            "latent_dim": 8,
            "interaction_group_slices": [[0, 15], [15, 33], [33, 48], [48, 63]],
            "route_chunk_length": 3,
            "route_candidate_count": 12,
            "route_side_count": 12,
        },
    )
    assert isinstance(model, InteractionAwareActionConditionedRouteHardNegativeJEPAPredictor)
    inputs = torch.randn(2, 8, 63)
    actions = torch.randn(2, 8, 3)
    chunks = torch.randn(2, 3, 3)
    _mean, _log_variance, _latent, auxiliary = model.forward_multitask(inputs, actions, chunks)
    for name in (
        "stopping_distance",
        "obstacle_ttc",
        "boundary_ttc",
        "pairwise_ttc_risk",
        "acceleration_slack",
        "obstacle_ttc_lower_quantile",
        "boundary_ttc_lower_quantile",
        "pairwise_ttc_lower_quantile",
    ):
        assert auxiliary[name].shape == (2, 5)
        assert torch.isfinite(auxiliary[name]).all()
    assert torch.all(auxiliary["stopping_distance"] >= 0.0)
    assert torch.all((auxiliary["obstacle_ttc"] >= 0.0) & (auxiliary["obstacle_ttc"] <= 10.0))
    for name in ("obstacle_ttc_hazard_logits", "boundary_ttc_hazard_logits", "pairwise_ttc_hazard_logits"):
        assert auxiliary[name].shape == (2, 5, 3)
        assert torch.isfinite(auxiliary[name]).all()


def test_hard_negative_pairwise_pooling_conditions_only_risk_heads():
    model = build_action_conditioned_predictor(
        "interaction_aware_action_conditioned_jepa_route_identity_hard_negative_v2",
        {
            "input_dim": 63,
            "horizon_count": 5,
            "hidden_dim": 16,
            "latent_dim": 8,
            "interaction_group_slices": [[0, 15], [15, 33], [33, 48], [48, 63]],
            "route_chunk_length": 3,
            "route_candidate_count": 12,
            "route_side_count": 12,
            "pairwise_pooling": True,
        },
    ).eval()
    inputs = torch.zeros(2, 8, 63)
    actions = torch.zeros(2, 8, 3)
    chunks = torch.zeros(2, 3, 3)
    first = model.forward_multitask(inputs, actions, chunks)[3]
    inputs[:, -1, 15:18] = 0.8
    second = model.forward_multitask(inputs, actions, chunks)[3]
    assert not torch.allclose(first["pairwise_ttc_hazard_logits"], second["pairwise_ttc_hazard_logits"])
    assert first["route_identity_logits"].shape == (2, 12)
    assert all(torch.isfinite(value).all() for value in second.values())


def test_hard_negative_pairwise_relational_context_is_action_conditioned():
    model = build_action_conditioned_predictor(
        "interaction_aware_action_conditioned_jepa_route_identity_hard_negative_v2",
        {
            "input_dim": 63,
            "horizon_count": 5,
            "hidden_dim": 16,
            "latent_dim": 8,
            "interaction_group_slices": [[0, 15], [15, 33], [33, 48], [48, 63]],
            "route_chunk_length": 3,
            "route_interaction_chunk_dim": 9,
            "route_candidate_count": 12,
            "route_side_count": 12,
            "pairwise_relational": True,
        },
    ).eval()
    inputs = torch.zeros(2, 8, 63)
    actions = torch.zeros(2, 8, 3)
    chunks = torch.zeros(2, 3, 3)
    relative = torch.zeros(2, 3, 9)
    first = model.forward_multitask(inputs, actions, chunks, route_interaction_chunks=relative)[3]
    changed = relative.clone()
    changed[:, :, :3] = 0.8
    second = model.forward_multitask(inputs, actions, chunks, route_interaction_chunks=changed)[3]
    assert not torch.allclose(first["pairwise_ttc_hazard_logits"], second["pairwise_ttc_hazard_logits"])
    assert all(torch.isfinite(value).all() for value in second.values())


def test_factory_and_runtime_history_forward_route_chunks():
    model = build_action_conditioned_predictor(
        "interaction_aware_action_conditioned_jepa_route_identity_v1",
        {
            "input_dim": 63,
            "horizon_count": 5,
            "hidden_dim": 16,
            "latent_dim": 8,
            "interaction_group_slices": [[0, 15], [15, 33], [33, 48], [48, 63]],
            "route_chunk_length": 3,
            "route_candidate_count": 12,
            "route_side_count": 12,
        },
    )
    assert isinstance(model, InteractionAwareActionConditionedRouteJEPAPredictor)
    history = SafeCaptureCandidateHistory(
        model,
        defender_count=2,
        device=torch.device("cpu"),
        history_length=8,
        action_scale=5.0,
    )
    history.reset(np.zeros((2, 63), dtype=np.float32))
    candidates = np.zeros((3, 2, 3), dtype=np.float32)
    candidates[1, :, 0] = 0.3
    candidates[2, :, 1] = -0.2
    chunks = np.repeat(candidates[:, None, :, :], 3, axis=1)
    _mean, _std, auxiliary = history.predict_candidates_multitask(
        candidates,
        horizon_index=2,
        candidate_chunks=chunks,
        candidate_indices=np.asarray([0, 1, 2], dtype=np.int64),
        route_side_indices=np.asarray([0, 1, 2], dtype=np.int64),
    )
    assert auxiliary["route_identity_logits"].shape == (3, 2, 12)
    assert auxiliary["route_side_logits"].shape == (3, 2, 12)
    assert auxiliary["route_geometry_logit"].shape == (3, 2)
    assert all(np.isfinite(value).all() for value in auxiliary.values())


def test_runtime_route_side_indices_match_archive_vocab():
    expected = {
        "nominal": 0,
        "left_detour": 1,
        "right_detour": 2,
        "upper_detour": 3,
        "lower_detour": 4,
        "radial_out": 5,
        "formation_split": 6,
        "formation_contract": 7,
        "braking": 8,
        "safe_intercept": 9,
        "visibility_hold": 10,
        "verified_safe_hold": 11,
    }
    assert ROUTE_SIDE_INDEX_BY_LABEL == expected
