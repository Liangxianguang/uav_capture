from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.replay_jepa_safe_capture_failures import (
    replay_failure_index,
    _reduce_trace,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trace(episode_index: int = 0) -> list[dict[str, object]]:
    action = [[0.1, 0.0, 0.0]]
    return [
        {
            "episode_index": episode_index,
            "step": 1,
            "desired_action": action,
            "reachable_nominal_action": action,
            "requested_action": action,
            "executed_action": action,
            "observation": {
                "target_visible": [True],
                "target_observation_age_steps": [0],
                "message_age_steps": [0],
            },
            "safety_observables": {
                "minimum_obstacle_clearance_m": 2.0,
                "minimum_pairwise_clearance_m": 1.5,
                "minimum_boundary_clearance_m": 2.0,
            },
            "candidate_ranking": {
                "candidate_labels": ["nominal"],
                "valid_mask": [True],
                "eligible_mask": [True],
                "scores": [1.0],
                "selected_index": 0,
                "execution_mode": "trusted",
                "ledger_states": ["trusted"],
                "ledger_credits": [0.9],
            },
            "cbf": {
                "solver_status": "success",
                "verified_feasible": True,
                "infeasible": False,
                "timed_out": False,
                "fallback_mode": "none",
                "used_fallback": False,
                "active_constraints": [],
                "action_correction_norm": 0.0,
                "solve_latency_ms": 1.0,
            },
        }
    ]


def _make_fixture(tmp_path: Path, *, locked: bool = False, seed_mismatch: bool = False, trace: bool = True) -> Path:
    run = tmp_path / "run"
    (run / "step_traces").mkdir(parents=True)
    manifest = run / "scene_manifest.jsonl"
    manifest.write_text(json.dumps({"episode_index": 0, "episode_seed": 700001, "training_seed": 1, "scene_hash": "a" * 64}) + "\n", encoding="utf-8")
    metadata = {
        "development_only": not locked,
        "locked_test_opened": locked,
        "training_seed": 1,
        "variant": {"variant": "m3"},
        "inputs": {"scene_manifest_sha256": _sha256(manifest)},
    }
    (run / "summary.json").write_text(json.dumps({"metadata": metadata}), encoding="utf-8")
    (run / "provenance.json").write_text(json.dumps({"development_only": not locked, "locked_test_opened": locked}), encoding="utf-8")
    if trace:
        (run / "step_traces" / "episode_0000.jsonl").write_text(
            "\n".join(json.dumps(item) for item in _trace()) + "\n", encoding="utf-8"
        )
    row = {
        "training_seed": 1,
        "variant": "m3",
        "episode_index": 0,
        "episode_seed": 999999 if seed_mismatch else 700001,
        "safe_capture": False,
        "termination_reason": "timeout",
        "defender_boundary_violation": False,
        "target_boundary_violation": True,
        "collision": False,
        "pairwise_violation": False,
        "primary_cause": "timeout",
        "diagnostic_labels": ["timeout"],
    }
    report = {
        "index_type": "jepa_safe_capture_v3_wp1_failure_index",
        "input_format": "v3",
        "development_only": not locked,
        "locked_test_opened": locked,
        "runs": [{
            "training_seed": 1,
            "variant": "m3",
            "path": str(run),
            "summary_sha256": _sha256(run / "summary.json"),
            "provenance_sha256": _sha256(run / "provenance.json"),
            "manifest_sha256": _sha256(manifest),
        }],
        "rows": [row],
    }
    index = tmp_path / "failure_index.json"
    index.write_text(json.dumps(report), encoding="utf-8")
    return index


def test_replay_is_deterministic_and_preserves_target_boundary_diagnostic(tmp_path: Path) -> None:
    index = _make_fixture(tmp_path)
    result = replay_failure_index(index, tmp_path / "out", tmp_path / "tb")
    episode = result["episodes"][0]
    assert episode["repeat_deterministic"] is True
    assert episode["repeat_sha256"][0] == episode["repeat_sha256"][1]
    replay = json.loads((tmp_path / "out" / "replays" / "1_m3_0000" / "replay_1.jsonl").read_text().splitlines()[0])
    assert replay["termination"]["target_boundary_violation"] is True
    assert replay["termination"]["defender_boundary_violation"] is False
    assert result["tensorboard"]["required_provenance"] is True
    assert list((tmp_path / "tb").glob("events.out.tfevents.*"))


def test_replay_rejects_missing_source_trace(tmp_path: Path) -> None:
    index = _make_fixture(tmp_path, trace=False)
    with pytest.raises(FileNotFoundError, match="Missing source trace"):
        replay_failure_index(index, tmp_path / "out", tmp_path / "tb")


def test_replay_rejects_locked_test_index(tmp_path: Path) -> None:
    index = _make_fixture(tmp_path, locked=True)
    with pytest.raises(ValueError, match="locked-test boundary"):
        replay_failure_index(index, tmp_path / "out", tmp_path / "tb")


def test_replay_rejects_episode_seed_mismatch(tmp_path: Path) -> None:
    index = _make_fixture(tmp_path, seed_mismatch=True)
    with pytest.raises(ValueError, match="episode_seed mismatch"):
        replay_failure_index(index, tmp_path / "out", tmp_path / "tb")


def test_reduce_trace_requires_finite_actions() -> None:
    trace = _trace()
    trace[0]["requested_action"] = [[float("nan"), 0.0, 0.0]]
    with pytest.raises(ValueError, match="Non-numeric|Non-finite"):
        _reduce_trace(trace, {"episode_index": 0, "episode_seed": 700001, "training_seed": 1, "variant": "m3", "safe_capture": False, "termination_reason": "timeout", "primary_cause": "timeout"}, identifier="1:m3:0000", categories=["timeout"], scene_hash="a" * 64)
