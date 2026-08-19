from __future__ import annotations

import pytest

from scripts.aggregate_stage4d_formal import pp_difference, statistics


def test_stage4d_statistics_and_paired_percentage_points() -> None:
    assert statistics([0.8, 0.9, 1.0])["mean"] == pytest.approx(0.9)
    result = pp_difference([0.8, 0.9, 1.0], [0.7, 0.95, 0.9])
    assert result["values"] == pytest.approx([10.0, -5.0, 10.0])
    assert result["positive_seed_count"] == 2


def test_stage4d_uses_same_four_locked_conditions() -> None:
    from scripts.aggregate_stage4d_formal import CONDITIONS, F2_METHOD

    assert F2_METHOD == "f2_uncertainty_features"
    assert len(CONDITIONS) == 4
