from __future__ import annotations

from scripts.evaluate_jepa_v3_multitask import binary_auc
from scripts.train_interaction_aware_jepa_multitask import validate_paired_dataset_contract
import pytest


def test_binary_auc_handles_ties_and_degenerate_labels() -> None:
    assert binary_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert binary_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert binary_auc([1, 1], [0.2, 0.8]) is None


def test_binary_auc_assigns_average_ranks_to_several_tie_groups() -> None:
    # Positives have one tied middle score and the highest score, yielding
    # Mann-Whitney U = 5 over 2 * 3 positive/negative pairs.
    assert binary_auc([0, 0, 1, 0, 1], [0.1, 0.5, 0.5, 0.5, 0.9]) == 5.0 / 6.0


def test_paired_dataset_contract_rejects_mixed_chunk_semantics() -> None:
    train = {
        "history_length": 8,
        "candidate_count": 5,
        "candidate_perturbation_mps": 0.1,
        "chunk_length_steps": 3,
        "candidate_action_semantics": "constant_desired_action_chunk_execute_first_step_then_replan",
        "candidate_chunk_is_constant": True,
        "action_history_normalization": "actions_divided_by_frozen_actor_action_scale",
        "action_scale": 5.0,
    }
    validation = {**train, "chunk_length_steps": 1}
    with pytest.raises(ValueError, match="counterfactual contracts differ"):
        validate_paired_dataset_contract(train, validation)
