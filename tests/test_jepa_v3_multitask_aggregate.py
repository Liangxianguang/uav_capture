from __future__ import annotations

from scripts.aggregate_jepa_v3_multitask import aggregate, render


def _run(seed: int, improvement: float) -> dict:
    horizons = [
        {
            "horizon_seconds": 0.1,
            "target_position_mae_m": 0.2,
            "target_improvement_over_constant_velocity_fraction": improvement,
            "obstacle_clearance_mae_m": 0.4,
            "inter_agent_clearance_mae_m": 0.2,
            "visibility_auc": 0.6,
            "cbf_intervention_auc": 0.9,
            "target_one_std_coverage": 0.8,
        },
        {
            "horizon_seconds": 0.5,
            "target_position_mae_m": 0.3,
            "target_improvement_over_constant_velocity_fraction": 0.5,
            "obstacle_clearance_mae_m": 0.5,
            "inter_agent_clearance_mae_m": 0.25,
            "visibility_auc": 0.65,
            "cbf_intervention_auc": 0.92,
            "target_one_std_coverage": 0.75,
        },
    ]
    return {
        "seed": seed,
        "checkpoint_sha256": "a" * 64,
        "best_epoch": 2,
        "best_validation_loss": -1.0,
        "prediction_gate": {"metrics_by_horizon": horizons},
        "action_following": {
            "axes": [
                {"mean_plus_minus_separation_norm": [0.01, 0.02]},
                {"mean_plus_minus_separation_norm": [0.02, 0.03]},
                {"mean_plus_minus_separation_norm": [0.03, 0.04]},
            ]
        },
        "tensorboard": {"loss_train_epochs": 40, "histogram_tag_count": 10},
    }


def test_aggregate_requires_distinct_seeds_and_renders_provenance() -> None:
    report = aggregate([_run(11, 0.1), _run(12, -0.1), _run(13, 0.2)])
    assert report["metrics_by_horizon"][0]["seeds_better_than_constant_velocity"] == 2
    assert report["metrics_by_horizon"][1]["seeds_better_than_constant_velocity"] == 3
    assert report["decision"]["eligible_for_reliability_ledger_development"] is True
    markdown = render(report)
    assert "Run Provenance" in markdown
    assert "11" in markdown
