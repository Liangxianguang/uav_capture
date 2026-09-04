from scripts.index_jepa_safe_capture_v11_failures import canonical_episode


def test_canonical_episode_maps_v11_csv_contract() -> None:
    row = canonical_episode(
        {
            "safe_capture_success": "True",
            "collision": "False",
            "defender_boundary_violation": "False",
            "target_boundary_violation": "True",
            "pairwise_violation": "False",
            "cbf_infeasible_steps": "2",
            "cbf_timeout_steps": "0",
            "cbf_unverified_steps": "2",
            "cbf_controlled_abort_steps": "1",
            "termination_reason": "cbf_controlled_abort",
        }
    )
    assert row["safe_capture"] is True
    assert row["boundary_violation"] is False
    assert row["target_boundary_violation"] is True
    assert row["cbf_infeasible_steps"] == 2
    assert row["cbf_controlled_abort_steps"] == 1
    assert row["termination_reason"] == "cbf_controlled_abort"
