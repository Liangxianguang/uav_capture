from __future__ import annotations

import json

from scripts.audit_jepa_safe_capture_v5_rank_guard_smoke import _trace_stats


def _ranking() -> dict[str, object]:
    return {
        "candidate_labels": [
            "nominal",
            "intercept",
            "lateral_clearance",
            "formation_clearance",
            "visibility_hold",
        ],
        "selected_index": 0,
        "scores": [0.0] * 5,
        "top_two_margin_m": 0.0,
        "rank_abstention_reason": "top_two_margin_abstention",
        "hysteresis_applied": False,
        "hold_steps_remaining": 0,
    }


def test_trace_stats_distinguishes_controlled_abort_from_raw_execution(tmp_path) -> None:
    trace_dir = tmp_path / "step_traces"
    trace_dir.mkdir()
    record = {
        "candidate_ranking": _ranking(),
        "raw_unverified_executed": False,
        "cbf": {
            "verified_feasible": False,
            "fallback_mode": "controlled_abort",
        },
    }
    (trace_dir / "episode_0000.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )

    stats = _trace_stats(tmp_path)

    assert stats["cbf_unverified_trace_steps"] == 1
    assert stats["raw_unverified_trace_steps"] == 0
    assert stats["missing_raw_unverified_trace_steps"] == 0


def test_trace_stats_requires_explicit_raw_execution_field(tmp_path) -> None:
    trace_dir = tmp_path / "step_traces"
    trace_dir.mkdir()
    record = {
        "candidate_ranking": _ranking(),
        "cbf": {"verified_feasible": True, "fallback_mode": "none"},
    }
    (trace_dir / "episode_0000.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )

    stats = _trace_stats(tmp_path)

    assert stats["missing_raw_unverified_trace_steps"] == 1
