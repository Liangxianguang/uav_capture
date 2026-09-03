from __future__ import annotations

import pytest
import torch

from scripts.train_jepa_safe_capture_v3 import hard_context_weights


def _batch(**overrides: torch.Tensor) -> dict[str, torch.Tensor]:
    value = {
        "labels_target_visible": torch.ones(2, 4),
        "labels_observation_age": torch.zeros(2, 4),
        "labels_obstacle_clearance": torch.full((2, 4), 0.5),
        "labels_inter_agent_clearance": torch.full((2, 4), 0.2),
        "labels_pairwise_ttc": torch.full((2, 4), 10.0),
        "labels_cbf_intervention": torch.zeros(2, 4),
    }
    value.update(overrides)
    return value


def test_easy_context_is_unit_weight_and_hard_context_is_upweighted() -> None:
    easy = hard_context_weights(_batch())
    hard = hard_context_weights(_batch(
        labels_target_visible=torch.zeros(2, 4),
        labels_observation_age=torch.full((2, 4), 6.0),
        labels_obstacle_clearance=torch.full((2, 4), 0.03),
        labels_inter_agent_clearance=torch.full((2, 4), 0.04),
        labels_pairwise_ttc=torch.full((2, 4), 0.2),
        labels_cbf_intervention=torch.ones(2, 4),
    ))
    assert torch.allclose(easy, torch.ones(2))
    assert torch.all(hard > easy)
    assert torch.all(hard <= 8.0)


def test_hard_context_weighting_rejects_missing_or_invalid_contract() -> None:
    with pytest.raises(ValueError):
        hard_context_weights({})
    with pytest.raises(ValueError):
        hard_context_weights(_batch(), cap=0.5)
