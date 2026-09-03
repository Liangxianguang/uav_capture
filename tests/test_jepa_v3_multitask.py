from __future__ import annotations

from scripts.evaluate_jepa_v3_multitask import binary_auc


def test_binary_auc_handles_ties_and_degenerate_labels() -> None:
    assert binary_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert binary_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert binary_auc([1, 1], [0.2, 0.8]) is None


def test_binary_auc_assigns_average_ranks_to_several_tie_groups() -> None:
    # Positives have one tied middle score and the highest score, yielding
    # Mann-Whitney U = 5 over 2 * 3 positive/negative pairs.
    assert binary_auc([0, 0, 1, 0, 1], [0.1, 0.5, 0.5, 0.5, 0.9]) == 5.0 / 6.0
