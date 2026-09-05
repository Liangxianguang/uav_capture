from __future__ import annotations

import json

import numpy as np

from scripts.aggregate_jepa_safe_capture_v2_paired import (
    _bootstrap_delta,
    _run_metric,
    canonical_scene_manifest_sha256,
    paired_comparison,
)


def _run(variant: str, seed: int, outcomes: list[bool]) -> dict:
    return {
        "variant": variant,
        "seed": seed,
        "episodes": {
            index: {
                "episode_seed": 646001 + index,
                "safe_capture": outcome,
            }
            for index, outcome in enumerate(outcomes)
        },
    }


def test_paired_comparison_reports_improved_degraded_and_tied_counts() -> None:
    result = paired_comparison(
        _run("m0", 20260911, [False, True, False, True]),
        _run("m3", 20260911, [True, False, False, True]),
    )

    assert result["improved"] == 1
    assert result["degraded"] == 1
    assert result["tied"] == 2
    assert result["delta_rate"] == 0.0
    assert result["mcnemar_exact_two_sided_p"] == 1.0


def test_bootstrap_is_fixed_and_uses_episode_pairs() -> None:
    values = np.asarray([1.0, 0.0, -1.0, 0.0])
    first = _bootstrap_delta(values)
    second = _bootstrap_delta(values)

    assert first == second
    assert first["unit"] == "episode_pair"
    assert first["bootstrap_seed"] == 20260903


def test_canonical_manifest_hash_excludes_only_training_provenance(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    records = [
        {"episode_index": 0, "episode_seed": 646001, "scene_hash": "abc"},
        {"episode_index": 1, "episode_seed": 646002, "scene_hash": "def"},
    ]
    first.write_text(
        "\n".join(json.dumps({**record, "training_seed": 20260911}) for record in records) + "\n",
        encoding="utf-8",
    )
    second.write_text(
        "\n".join(json.dumps({**record, "training_seed": 20260913}) for record in records) + "\n",
        encoding="utf-8",
    )

    assert canonical_scene_manifest_sha256(first) == canonical_scene_manifest_sha256(second)


def test_verified_fallback_counts_are_distinguished_from_unverified_abort() -> None:
    run = {
        "variant": "m2",
        "seed": 20260911,
        "episodes": {0: {"episode_seed": 646001, "safe_capture": True}},
        "overall": {
            "safe_capture_count": 1,
            "safe_capture_rate": 1.0,
            "collision_count": 0,
            "boundary_violation_count": 0,
            "pairwise_violation_count": 0,
            "cbf_infeasible_steps": 3,
            "cbf_timeout_steps": 0,
            "cbf_unverified_steps": 2,
            "cbf_fallback_steps": 3,
            "cbf_controlled_abort_steps": 2,
            "transit_success_rate": 1.0,
            "mean_capture_time_seconds": 1.0,
            "mean_cbf_p95_solve_latency_ms": 1.0,
        },
        "stage": "full",
    }
    metrics = _run_metric(run)
    assert metrics["cbf_fallback_steps"] == 3
    assert metrics["cbf_controlled_abort_steps"] == metrics["cbf_unverified_steps"]
    assert metrics["cbf_fallback_steps"] >= metrics["cbf_infeasible_steps"]
