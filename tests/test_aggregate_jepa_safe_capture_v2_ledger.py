from __future__ import annotations

from pathlib import Path

import pytest

from scripts.aggregate_jepa_safe_capture_v2_ledger import aggregate, render_report, write_tensorboard


def _run(seed: int) -> dict:
    horizons = []
    forecasts = []
    for index, seconds in enumerate((0.1, 0.2)):
        horizons.append(
            {
                "horizon_index": index,
                "horizon_seconds": seconds,
                "global_sample_count": 100,
                "global_credit": 0.8,
                "target_mae_m": 0.2,
                "clearance_mae_m": 0.1,
                "collision_rate": 0.0,
                "boundary_rate": 0.0,
                "qp_label_unique_values": 1,
            }
        )
        forecasts.append({"local_fraction": 0.9, "coarse_fraction": 0.05, "global_fraction": 0.05})
    return {
        "seed": seed,
        "sha256": "a" * 64,
        "path": str(Path(f"ledger_seed{seed}").resolve()),
        "payload": {
            "ledger_type": "jepa_safe_capture_v2_checkpoint_bound_reliability",
            "ledger_version": 2,
            "not_a_locked_test": True,
            "locked_test_opened": False,
            "immutable_after_calibration": True,
            "source": {
                "checkpoint": f"D:/results/checkpoint_seed{seed}.pt",
                "checkpoint_sha256": "b" * 64,
                "calibration_dataset_sha256": "c" * 64,
                "calibration_metadata_sha256": "d" * 64,
            },
            "diagnostics": {"horizon_diagnostics": horizons},
            "forecast": {
                "per_horizon": forecasts,
                "state_counts": {"trusted": 180, "fallback_nominal": 20},
                "fallback_reason_counts": {"low_credit": 20},
                "unsafe_rate_by_state": {"trusted": 0.0, "fallback_nominal": 0.01},
                "high_credit_failure_rate_not_above_low_credit": True,
                "ood_or_hard_contexts_trigger_safe_hold": True,
            },
        },
    }


def test_ledger_aggregate_requires_three_distinct_seeds() -> None:
    report = aggregate([_run(11), _run(12), _run(13)])
    assert report["decision"]["eligible_for_candidate_ranking_development"] is True
    assert report["per_horizon"][0]["global_credit"]["mean"] == pytest.approx(0.8)
    assert "Calibration Summary" in render_report(report)


def test_ledger_aggregate_rejects_duplicate_seed() -> None:
    with pytest.raises(ValueError, match="distinct checkpoint seeds"):
        aggregate([_run(11), _run(11), _run(13)])


def test_ledger_aggregate_tensorboard_records_provenance(tmp_path: Path) -> None:
    report = aggregate([_run(11), _run(12), _run(13)])
    audit = write_tensorboard(report, tmp_path / "tb")
    assert audit["required_text_complete"] is True
    assert audit["scalar_tag_count"] > 0
