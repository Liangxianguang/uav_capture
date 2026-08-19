from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.aggregate_stage4c_formal import paired_difference, statistics


def test_stage4c_statistics_uses_training_seed_as_unit() -> None:
    result = statistics([0.80, 0.90, 1.00])
    assert result["count"] == 3
    assert result["mean"] == pytest.approx(0.90)
    assert result["values"] == [0.80, 0.90, 1.00]


def test_stage4c_paired_difference_reports_percentage_points() -> None:
    result = paired_difference([0.80, 0.90, 1.00], [0.70, 0.95, 0.90])
    assert result["values"] == pytest.approx([10.0, -5.0, 10.0])
    assert result["positive_seed_count"] == 2


def test_stage4c_formal_runner_defines_four_locked_conditions() -> None:
    from scripts.run_stage4c_formal import CONDITIONS, METHOD

    assert METHOD == "f1_time_aligned_belief"
    assert tuple(CONDITIONS) == (
        "nominal_partial_observation",
        "delayed_measurements",
        "burst_occlusion",
        "communication_loss",
    )
