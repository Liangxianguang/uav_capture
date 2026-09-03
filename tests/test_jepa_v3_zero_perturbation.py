from __future__ import annotations

import csv
from pathlib import Path

from scripts.compare_jepa_v3_zero_perturbation import compare


def _write_rows(directory: Path, value: str = "same") -> None:
    directory.mkdir()
    with (directory / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("episode_index", "episode_seed", "layout_seed", "safe_capture_success", "jepa_enabled"),
        )
        writer.writeheader()
        writer.writerow({"episode_index": 0, "episode_seed": 1, "layout_seed": 2, "safe_capture_success": value, "jepa_enabled": "False"})
    (directory / "scenes.jsonl").write_text('{"episode_index": 0}\n', encoding="utf-8")


def test_zero_perturbation_comparison_ignores_jepa_diagnostics_but_requires_paired_fields(tmp_path: Path) -> None:
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    _write_rows(baseline)
    _write_rows(candidate)
    result = compare(baseline, candidate)
    assert result["passed"] is True
    assert result["non_jepa_fields_compared"] == 4


def test_zero_perturbation_comparison_reports_behavior_change(tmp_path: Path) -> None:
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    _write_rows(baseline, value="True")
    _write_rows(candidate, value="False")
    result = compare(baseline, candidate)
    assert result["passed"] is False
    assert result["field_difference_count"] == 1
    assert result["field_differences"][0]["field"] == "safe_capture_success"
