from __future__ import annotations

from scripts.diagnose_jepa_safe_capture_v21_abstention_counterfactual import analyze_rows


def _outcome(*, safe: bool, progress: float) -> dict[str, object]:
    return {
        "settled_safe_capture": safe,
        "settled_safety_ok": safe,
        "settled_progress_m": progress,
        "settled_cbf_correction_norm_mps": 0.0,
    }


def _row(*, selected: int, scores: list[float], eligible: list[bool]) -> dict[str, object]:
    return {
        "training_seed": 20260911,
        "episode_index": 0,
        "step": 1,
        "selected_index": selected,
        "scores": scores,
        "eligible_mask": eligible,
        "settled_candidates": [
            _outcome(safe=False, progress=0.1),
            _outcome(safe=True, progress=0.2),
            _outcome(safe=False, progress=0.3),
            _outcome(safe=False, progress=0.4),
            _outcome(safe=False, progress=0.5),
        ],
    }


def test_analyze_rows_isolates_score_argmin_from_recorded_selection() -> None:
    result = analyze_rows(
        [
            _row(
                selected=0,
                scores=[1.0, 0.5, 2.0, 3.0, 4.0],
                eligible=[True, True, False, False, False],
            ),
            _row(
                selected=0,
                scores=[1.0, 2.0, 0.5, 3.0, 4.0],
                eligible=[True, False, True, False, False],
            ),
            _row(
                selected=0,
                scores=[1.0, 2.0, 3.0, 4.0, 5.0],
                eligible=[True, False, False, False, False],
            ),
        ]
    )

    assert result["decision_rows"] == 3
    assert result["multi_eligible_decisions"] == 2
    assert result["recorded_selected"]["selected_not_best_count"] == 2
    assert result["score_argmin"]["selected_not_best_count"] == 0
    assert result["agreement"]["recorded_vs_score_argmin_rate"] == 0.0
    assert result["agreement"]["score_argmin_vs_settled_best_rate"] == 1.0
