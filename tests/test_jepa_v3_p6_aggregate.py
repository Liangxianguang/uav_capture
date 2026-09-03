import numpy as np

from scripts.aggregate_jepa_v3_p6_development import (
    _run_metrics,
    classify,
    exact_mcnemar_pvalue,
    frozen_scene_hash,
    hierarchical_bootstrap_interval,
)


def test_exact_mcnemar_is_two_sided_and_handles_no_discordance():
    assert exact_mcnemar_pvalue(0, 0) == 1.0
    assert exact_mcnemar_pvalue(6, 6) == 1.0
    assert exact_mcnemar_pvalue(5, 0) == 0.0625


def test_hierarchical_bootstrap_is_seeded_and_tracks_mean():
    values = np.array([[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]])
    first = hierarchical_bootstrap_interval(values, replicates=1000, random_seed=9)
    second = hierarchical_bootstrap_interval(values, replicates=1000, random_seed=9)
    assert first == second
    assert first["mean"] == 0.0
    assert first["ci_95_low"] <= 0.0 <= first["ci_95_high"]


def test_safe_capture_first_rejects_nonpositive_three_seed_delta_from_locked_readiness():
    report = {
        "aggregate": {
            "candidate_collision_total": 0,
            "candidate_boundary_total": 0,
            "safe_capture_delta_percentage_points": {"mean": 0.0},
            "safe_capture_delta_nonnegative_seed_count": 2,
        }
    }
    decision = classify(report)
    assert decision["classification"] == "prediction_improvement_no_control_gain"
    assert decision["eligible_to_open_locked_test"] is False


def test_baseline_metrics_do_not_require_jepa_only_csv_columns():
    row = {
        "safe_capture_success": "True",
        "collision": "False",
        "world_violation_steps": "0",
        "transit_success": "True",
        "capture_time_seconds": "1.2",
        "termination_reason": "safe_capture",
        "total_defender_path_length_m": "4.0",
        "min_clearance_m": "0.4",
        "mean_cbf_action_correction_norm": "0.2",
    }
    metrics = _run_metrics([row])
    assert metrics["safe_capture_rate"] == 1.0
    assert metrics["mean_ledger_credit"] is None


def test_frozen_scene_hash_excludes_method_dependent_outcomes(tmp_path):
    base = tmp_path / "base.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    base.write_text('{"episode_index": 0, "spec": {"seed": 1}, "scenario": {"layout": "fixed"}, "outcome": {"capture": true}}\n')
    candidate.write_text('{"episode_index": 0, "spec": {"seed": 1}, "scenario": {"layout": "fixed"}, "outcome": {"capture": false}}\n')
    assert frozen_scene_hash(base) == frozen_scene_hash(candidate)
