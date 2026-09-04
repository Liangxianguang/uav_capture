from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_jepa_safe_capture_v2_latency import (
    LATENCY_STAGES,
    _write_tensorboard,
    audit_run,
)


def _make_run(root: Path, *, missing_stage: str | None = None) -> Path:
    run = root / "run"
    trace_dir = run / "step_traces"
    trace_dir.mkdir(parents=True)
    latency = {stage: float(index + 1) for index, stage in enumerate(LATENCY_STAGES)}
    if missing_stage is not None:
        latency.pop(missing_stage)
    records = []
    for step in (1, 2):
        records.append(
            {
                "episode_index": 0,
                "step": step,
                "raw_unverified_executed": False,
                "input_observation": {"queue_age_steps": float(step)},
                "latency_ms": latency,
                "candidate_ranking": {"selected_index": 0},
            }
        )
    (trace_dir / "episode_0000.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    metadata = {
        "development_only": True,
        "locked_test_opened": False,
        "trace_schema_version": 2,
        "variant": {"variant": "m3", "use_jepa": True},
        "git_revision": "test",
        "latency_contract": {
            "per_step_fields": list(LATENCY_STAGES),
            "unit": "milliseconds",
            "queue_age_unit": "control_steps",
        },
    }
    overall = {
        "episodes": 1,
        "control_cycles": 2,
        "raw_unverified_executed_steps": 0,
    }
    (run / "summary.json").write_text(
        json.dumps({"metadata": metadata, "overall": overall}), encoding="utf-8"
    )
    (run / "provenance.json").write_text(
        json.dumps({"development_only": True, "locked_test_opened": False}), encoding="utf-8"
    )
    (run / "episodes.csv").write_text(
        "episode_index,control_cycle_count\n0,2\n", encoding="utf-8"
    )
    return run


def test_latency_audit_validates_trace_and_tensorboard(tmp_path: Path) -> None:
    report = audit_run(_make_run(tmp_path))
    assert all(report["gates"].values())
    assert report["control_cycles"] == 2
    assert report["latency"]["cycle_total"]["p95_ms"] == pytest.approx(10.0)
    assert report["queue_age_steps"]["p95_steps"] == pytest.approx(1.95)

    tb = _write_tensorboard(tmp_path / "tb", report)
    assert tb["event_files"]
    assert tb["scalar_tag_count"] >= len(LATENCY_STAGES)


def test_latency_audit_rejects_missing_per_step_stage(tmp_path: Path) -> None:
    report = audit_run(_make_run(tmp_path, missing_stage="jepa_inference"))
    assert report["gates"]["all_latency_fields_present"] is False
