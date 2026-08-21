from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "aggregate_central_v5_development_replication.py"
SPEC = importlib.util.spec_from_file_location("aggregate_central_v5_development_replication", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AGGREGATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGGREGATOR)


def _summary(seed: int, *, passed: bool = True) -> dict[str, object]:
    fixed_wall = 1.0 if passed else 0.95
    gates = {
        "s3_cooperative_safe_capture_at_least_85_percent": True,
        "s3_collision_at_most_2_percent": True,
        "s3_boundary_at_most_2_percent": True,
        "s3_transit_at_least_99_percent": True,
        "all_fixed_cbf_at_least_98_percent": passed,
    }
    return {
        "evaluation_type": "central_v5_retained_bc_development_validation",
        "not_a_locked_test": True,
        "training": {
            "run_dir": f"F:/runs/wallcoverage_retained_seed{seed}",
            "checkpoint_sha256": f"{seed:064x}",
            "passed": True,
        },
        "candidate_gates": gates,
        "candidate_gate_passed": passed,
        "fixed_regression": {
            scene: {"cbf": {"metrics": {"cooperative_safe_capture_rate": fixed_wall if scene == "s1_wall" else 1.0}}}
            for scene in AGGREGATOR.FIXED_SCENES
        },
        "s3_validation": {
            "cbf": {
                "metrics": {
                    "episodes": 60,
                    "cooperative_safe_capture_rate": 0.95,
                    "collision_rate": 0.0,
                    "boundary_violation_rate": 0.0,
                    "transit_success_rate": 1.0,
                }
            }
        },
        "s3_scene_pairing": {
            "static_scenes_exactly_paired": True,
            "raw_static_scene_sha256": "a" * 64,
        },
    }


def _write_summary(directory: Path, seed: int, *, passed: bool = True) -> Path:
    path = directory / f"seed{seed}.json"
    path.write_text(json.dumps(_summary(seed, passed=passed)), encoding="utf-8")
    return path


def test_collect_rejects_locked_test_when_one_of_three_development_seeds_fails(tmp_path: Path) -> None:
    inputs = [
        _write_summary(tmp_path, 661602),
        _write_summary(tmp_path, 661604),
        _write_summary(tmp_path, 661606, passed=False),
    ]

    aggregate = AGGREGATOR.collect(inputs)
    report = AGGREGATOR.render_markdown(aggregate)

    assert aggregate["development_candidate_pass_count"] == 2
    assert aggregate["locked_test_opened"] is False
    assert aggregate["decision"] == "replication_rejected_do_not_open_locked_test"
    assert "locked block 647201 remains unopened" in report
    assert "seed 661606" in report


def test_collect_allows_locked_test_only_after_three_passing_development_seeds(tmp_path: Path) -> None:
    inputs = [_write_summary(tmp_path, seed) for seed in (661702, 661704, 661706)]

    aggregate = AGGREGATOR.collect(inputs)

    assert aggregate["development_candidate_pass_count"] == 3
    assert aggregate["locked_test_opened"] is True


def test_collect_rejects_duplicate_training_seed(tmp_path: Path) -> None:
    inputs = [
        _write_summary(tmp_path, 661602),
        _write_summary(tmp_path, 661604),
        _write_summary(tmp_path, 661604, passed=False),
    ]

    with pytest.raises(ValueError, match="Training seeds must be distinct"):
        AGGREGATOR.collect(inputs)
