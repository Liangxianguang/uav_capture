from __future__ import annotations

import json

from scripts.aggregate_central_v4_validation import collect, render_markdown


def test_aggregate_normalizes_fixed_and_s3_summaries(tmp_path) -> None:
    fixed = tmp_path / "fixed"
    fixed.mkdir()
    (fixed / "summary.json").write_text(
        json.dumps({"summary": {"episodes": 2, "cooperative_safe_capture_rate": 1.0, "capture_rate": 1.0}}),
        encoding="utf-8",
    )
    randomized = tmp_path / "randomized"
    randomized.mkdir()
    (randomized / "summary.json").write_text(
        json.dumps({"overall": {"episodes": 4, "safe_capture_rate": 0.75, "transit_success_rate": 1.0}}),
        encoding="utf-8",
    )

    aggregate = collect(tmp_path, {"S1": ["fixed"], "S3": ["randomized"], "S2": []})
    assert aggregate["S1"]["fixed"]["metrics"]["cooperative_safe_capture_rate"] == 1.0
    assert aggregate["S3"]["randomized"]["metrics"]["safe_capture_rate"] == 0.75
    assert "not a locked-test" in render_markdown({"groups": aggregate})
