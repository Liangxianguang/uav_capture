from __future__ import annotations

import pytest

from scripts.aggregate_jepa_safe_capture_v21_settled import (
    SEEDS,
    VARIANTS,
    aggregate_reports,
    summarize_rows,
)


def _row(seed: int, variant: str, episode: int, step: int, *, selected_not_best: bool, safe: bool, settled: bool, spearman: float | None = 0.2) -> dict:
    return {
        "training_seed": seed,
        "variant": variant,
        "episode_index": episode,
        "step": step,
        "selected_index": 0,
        "selected_not_best": selected_not_best,
        "selected_settled_safe_capture": safe,
        "best_settled_safe_capture": safe,
        "selected_settled_safety_ok": safe,
        "selected_settled_termination_reason": "chunk_complete" if settled else "ineligible",
        "selected_settled_progress_m": 1.0 if settled else None,
        "best_settled_progress_m": 1.0 if settled else None,
        "selected_cbf_correction_norm_mps": 0.05 if settled else None,
        "ledger_credits": [0.9] * 5,
        "ledger_states": ["trusted"] * 5,
        "predicted_rank_spearman": spearman,
        "predicted_rank_kendall": spearman,
        "top_two_margin_m": 0.01 if settled else None,
        "eligible_mask": [True] * 5 if settled else [False] * 5,
        "predicted_visibility": [0.8] * 5,
        "predicted_min_ttc_s": [2.0] * 5,
        "pair_label": "tied",
    }


def _report(seed: int, *, bad: bool = False) -> dict:
    rows = []
    for variant in VARIANTS:
        for index in range(4):
            rows.append(_row(seed, variant, index, 1, selected_not_best=bad and index < 2, safe=index == 0, settled=True))
    return {
        "seed": seed,
        "path": f"run-{seed}",
        "protocol_sha256": "protocol",
        "environment_config_sha256": "environment",
        "scene_manifest_sha256": f"scene-{seed}",
        "rows": rows,
        "report": {"all_gates_pass": True},
        "report_sha256": f"report-{seed}",
        "rows_sha256": f"rows-{seed}",
    }


def test_summarize_rows_reports_settled_and_separation_rates() -> None:
    rows = [_row(SEEDS[0], "m3", 0, 1, selected_not_best=True, safe=False, settled=True)]
    summary = summarize_rows(rows, "m3")
    assert summary["decisions"] == 1
    assert summary["selected_not_best_rate"] == 1.0
    assert summary["candidate_separation_pass_rate"] == 1.0
    assert summary["settled_decisions"] == 1


def test_aggregate_requires_exact_three_seed_matrix() -> None:
    with pytest.raises(ValueError, match="exactly"):
        aggregate_reports([_report(SEEDS[0]), _report(SEEDS[1])])


def test_aggregate_marks_systematic_bad_ranking_unresolved() -> None:
    report = aggregate_reports([_report(seed, bad=True) for seed in SEEDS])
    assert report["decision"]["source_gates_pass"] is True
    assert report["decision"]["ranking_gate"] is False
    assert report["decision"]["classification"] == "ranking_unresolved"
    assert report["decision"]["locked_test_opened"] is False


def test_aggregate_preserves_scene_manifest_per_seed() -> None:
    report = aggregate_reports([_report(seed) for seed in SEEDS])
    assert report["scene_manifest_sha256_by_seed"] == {str(seed): f"scene-{seed}" for seed in SEEDS}
