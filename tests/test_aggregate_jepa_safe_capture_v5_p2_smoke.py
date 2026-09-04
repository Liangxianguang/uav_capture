from __future__ import annotations

from scripts.aggregate_jepa_safe_capture_v5_p2_smoke import (
    VARIANTS,
    SEEDS,
    aggregate,
    paired,
)


def _run(seed: int, variant: str, outcomes: list[bool]) -> dict:
    return {
        "path": f"run-{seed}-{variant}",
        "seed": seed,
        "variant": variant,
        "canonical_manifest_sha256": "manifest",
        "manifest_sha256": f"manifest-{seed}",
        "summary_sha256": f"summary-{seed}-{variant}",
        "provenance_sha256": f"provenance-{seed}-{variant}",
        "provenance": {
            "development_only": True,
            "locked_test_opened": False,
            "episodes": len(outcomes),
            "variant": {"variant": variant, "use_ledger": variant in {"m3", "a2"}},
            "inputs": {"protocol_sha256": "protocol", "environment_config_sha256": "environment"},
        },
        "summary": {
            "overall": {
                "safe_capture_count": sum(outcomes),
                "safe_capture_rate": sum(outcomes) / len(outcomes),
                "collision_count": 0,
                "boundary_violation_count": 0,
                "pairwise_violation_count": 0,
                "cbf_infeasible_steps": 0,
                "cbf_timeout_steps": 0,
                "cbf_controlled_abort_steps": 0,
                "cbf_unverified_steps": 0,
                "raw_unverified_executed_steps": 0,
                "cbf_fallback_steps": 0,
            }
        },
        "rows": [
            {"episode_index": str(index), "episode_seed": str(650001 + index), "cooperative_safe_capture": str(outcome)}
            for index, outcome in enumerate(outcomes)
        ],
    }


def test_paired_reports_episode_level_delta() -> None:
    result = paired(
        _run(20260911, "m0", [False, True, False, True]),
        _run(20260911, "m3", [True, False, False, True]),
    )
    assert result["improved"] == 1
    assert result["degraded"] == 1
    assert result["tied"] == 2
    assert result["delta_rate"] == 0.0


def test_aggregate_requires_complete_three_seed_matrix() -> None:
    runs = [
        _run(seed, variant, [True, False, True, False])
        for seed in SEEDS
        for variant in VARIANTS
    ]
    report = aggregate(runs, expected_episodes=4)
    assert report["safety_gate"] is True
    assert report["decision"]["locked_test_opened"] is False
    assert report["decision"]["m3_seeds_nonnegative"] == 3
