from __future__ import annotations

from scripts.evaluate_jepa_v3_multitask import binary_auc


def test_binary_auc_handles_ties_and_degenerate_labels() -> None:
    assert binary_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert binary_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert binary_auc([1, 1], [0.2, 0.8]) is None
