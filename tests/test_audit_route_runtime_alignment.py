from __future__ import annotations

from scripts.audit_route_runtime_alignment import (
    _pairwise_sign_agreement,
    _top1_index,
)


def test_pairwise_sign_agreement_detects_reversed_route_progress() -> None:
    assert _pairwise_sign_agreement([3.0, 2.0, 1.0], [1.0, 2.0, 3.0]) == 0.0


def test_pairwise_sign_agreement_ignores_ties() -> None:
    assert _pairwise_sign_agreement([3.0, 3.0, 1.0], [2.0, 1.0, 0.0]) == 1.0


def test_top1_index_is_deterministic_and_empty_safe() -> None:
    assert _top1_index([]) is None
    assert _top1_index([1.0, 4.0, 4.0]) == 1
