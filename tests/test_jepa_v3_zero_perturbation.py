from __future__ import annotations

import csv
from pathlib import Path

from scripts.compare_jepa_v3_zero_perturbation import compare
from scripts.evaluate_jepa_safe_capture_v2_paired import requires_zero_perturbation_identity_bypass


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


def test_zero_perturbation_comparison_ignores_full_chain_runtime_diagnostics(tmp_path: Path) -> None:
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    _write_rows(baseline)
    _write_rows(candidate)
    for directory, latency in ((baseline, "1.0"), (candidate, "99.0")):
        rows = list(csv.DictReader((directory / "episodes.csv").open(newline="", encoding="utf-8")))
        rows[0]["control_cycle_count"] = latency
        rows[0]["latency_breakdown"] = "{\"cycle_total\": {\"p95_ms\": " + latency + "}}"
        rows[0]["trace_write_latency_ms"] = latency
        with (directory / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    result = compare(baseline, candidate)
    assert result["passed"] is True


def test_zero_perturbation_comparison_reports_behavior_change(tmp_path: Path) -> None:
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    _write_rows(baseline, value="True")
    _write_rows(candidate, value="False")
    result = compare(baseline, candidate)
    assert result["passed"] is False
    assert result["field_difference_count"] == 1
    assert result["field_differences"][0]["field"] == "safe_capture_success"


def test_zero_perturbation_ignores_variant_specific_scene_outcome(tmp_path: Path) -> None:
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    _write_rows(baseline)
    _write_rows(candidate)
    baseline.joinpath("scenes.jsonl").write_text(
        '{"episode_index":0,"scenario":{"obstacles":[]},"outcome":{"variant":"m0"}}\n',
        encoding="utf-8",
    )
    candidate.joinpath("scenes.jsonl").write_text(
        '{"episode_index":0,"scenario":{"obstacles":[]},"outcome":{"variant":"m3"}}\n',
        encoding="utf-8",
    )
    result = compare(baseline, candidate)
    assert result["scenes_byte_identical"] is False
    assert result["scenes_geometry_identical"] is True
    assert result["passed"] is True


def test_zero_perturbation_identity_bypass_is_explicit_and_exact() -> None:
    assert requires_zero_perturbation_identity_bypass(True, 0.0) is True
    assert requires_zero_perturbation_identity_bypass(False, 0.0) is False
    assert requires_zero_perturbation_identity_bypass(True, 1e-12) is False
