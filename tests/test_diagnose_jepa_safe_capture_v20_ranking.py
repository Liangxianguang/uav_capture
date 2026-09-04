from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import diagnose_jepa_safe_capture_v20_ranking as diagnosis


def _settled_candidate(*, safe: bool, progress: float, correction: float = 0.0) -> dict[str, object]:
    return {
        "settled_safe_capture": safe,
        "settled_safety_ok": safe,
        "settled_progress_m": progress,
        "settled_cbf_correction_norm_mps": correction,
    }


def _row(*, episode_index: int, step: int, eligible_mask: list[bool]) -> dict[str, object]:
    return {
        "episode_index": episode_index,
        "step": step,
        "eligible_mask": eligible_mask,
        "scores": [0.20, 0.10, 0.30, 0.40, 0.50],
        "selected_index": 1,
        "predicted_min_clearance_m": [0.10, 0.20, 0.30, 0.40, 0.50],
        "settled_candidates": [
            _settled_candidate(safe=False, progress=0.1),
            _settled_candidate(safe=True, progress=0.2),
            _settled_candidate(safe=False, progress=0.3),
            _settled_candidate(safe=False, progress=0.4),
            _settled_candidate(safe=False, progress=0.5),
        ],
    }


def _trace(*, step: int, selected_index: int = 1) -> dict[str, object]:
    values = [0.20, 0.10, 0.30, 0.40, 0.50]
    return {
        "step": step,
        "candidate_ranking": {
            "execution_mode": "jepa_ranked",
            "fallback_reason": None,
            "rank_abstention_reason": None,
            "selected_index": selected_index,
            "target_cost_m": values,
            "uncertainty_cost_m": values,
            "clearance_cost_m": values,
            "ttc_cost": values,
            "visibility_cost": values,
            "cbf_risk_cost": values,
            "action_change_cost_mps": values,
        }
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_rank_correlation_reports_expected_cost_direction() -> None:
    positive = diagnosis.rank_correlation([1.0, 2.0], [2.0, 4.0])
    assert positive["spearman"] == pytest.approx(1.0)
    assert positive["kendall"] == pytest.approx(1.0)
    assert diagnosis.rank_correlation([2.0, 1.0], [2.0, 4.0])["spearman"] == pytest.approx(-1.0)
    assert diagnosis.rank_correlation([1.0, 1.0], [2.0, 4.0]) == {
        "spearman": None,
        "kendall": None,
    }


def test_best_settled_index_prioritizes_safety_then_progress() -> None:
    row = {
        "settled_candidates": [
            _settled_candidate(safe=False, progress=0.9),
            _settled_candidate(safe=True, progress=0.1),
            _settled_candidate(safe=True, progress=0.8),
            _settled_candidate(safe=False, progress=1.0),
            _settled_candidate(safe=False, progress=0.0),
        ]
    }
    assert diagnosis.best_settled_index(row, [0, 1, 2, 3, 4]) == 2
    assert diagnosis.best_settled_index(row, [0, 3, 4]) == 3


def test_diagnose_seed_separates_all_ineligible_from_multi_eligible(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seed = 20260911
    settled_rel = Path("settled.jsonl")
    replay_rel = Path("replay")
    monkeypatch.setitem(
        diagnosis.RUNS,
        seed,
        {"settled": str(settled_rel), "replay": str(replay_rel)},
    )

    _write_jsonl(
        tmp_path / settled_rel,
        [
            _row(episode_index=0, step=0, eligible_mask=[False] * 5),
            _row(episode_index=0, step=1, eligible_mask=[True, True, False, False, False]),
        ],
    )
    _write_jsonl(
        tmp_path / replay_rel / "step_traces" / "episode_0.jsonl",
        [_trace(step=0), _trace(step=1)],
    )

    report = diagnosis.diagnose_seed(tmp_path, seed)

    assert report["decision_rows"] == 2
    assert report["all_ineligible_count"] == 1
    assert report["all_ineligible_rate"] == 0.5
    assert report["multi_eligible_count"] == 1
    assert report["score_argmin_settled_best_rate"] == 1.0
    assert report["selected_settled_best_rate"] == 1.0
    assert report["selected_score_argmin_rate"] == 1.0
    assert report["execution_modes"] == {"jepa_ranked": 2}


def test_diagnose_seed_rejects_incomplete_trace_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seed = 20260912
    monkeypatch.setitem(
        diagnosis.RUNS,
        seed,
        {"settled": "settled.jsonl", "replay": "replay"},
    )
    _write_jsonl(tmp_path / "settled.jsonl", [_row(episode_index=0, step=0, eligible_mask=[True] * 5)])
    _write_jsonl(tmp_path / "replay" / "step_traces" / "episode_0.jsonl", [])

    with pytest.raises(ValueError, match="trace index is incomplete"):
        diagnosis.diagnose_seed(tmp_path, seed)
