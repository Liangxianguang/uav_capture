from __future__ import annotations
import torch

from encirclement3d.prediction import (
    InteractionAwareActionConditionedMultitaskJEPAPredictor,
    InteractionAwareActionConditionedSafeCaptureJEPAPredictor,
    build_action_conditioned_predictor,
)


def _inputs(batch: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(batch, 8, 63), torch.randn(batch, 8, 3)


def test_safe_capture_model_emits_all_p2_evaluator_heads_without_action_output() -> None:
    model = InteractionAwareActionConditionedSafeCaptureJEPAPredictor(
        input_dim=63,
        horizon_count=4,
        interaction_group_slices=((0, 15), (15, 33), (33, 48), (48, 63)),
    )
    inputs, actions = _inputs()
    mean, log_variance, latent, auxiliary = model.forward_multitask(inputs, actions)
    assert mean.shape == (3, 4, 3)
    assert log_variance.shape == mean.shape
    assert latent.shape == (3, 4, 64)
    for key in (
        "target_velocity",
        "target_acceleration",
    ):
        assert auxiliary[key].shape == (3, 4, 3)
    for key in (
        "obstacle_clearance_lower_quantile",
        "inter_agent_clearance_lower_quantile",
        "pairwise_ttc",
        "observation_age",
        "cbf_qp_feasibility_logit",
    ):
        assert auxiliary[key].shape == (3, 4)
    assert auxiliary["action_consistency"].shape == (3, 3)
    assert torch.isfinite(mean).all()
    assert torch.all(auxiliary["pairwise_ttc"] >= 0.0)
    assert torch.all(auxiliary["observation_age"] >= 0.0)


def test_safe_capture_factory_is_distinct_and_runtime_compatible_with_old_multitask_type() -> None:
    model = build_action_conditioned_predictor(
        "interaction_aware_action_conditioned_jepa_safe_capture_v2",
        {
            "input_dim": 63,
            "horizon_count": 4,
            "interaction_group_slices": ((0, 15), (15, 33), (33, 48), (48, 63)),
        },
    )
    assert isinstance(model, InteractionAwareActionConditionedSafeCaptureJEPAPredictor)
    assert isinstance(model, InteractionAwareActionConditionedMultitaskJEPAPredictor)
