from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "aggregate_central_v5_locked_test.py"
SPEC = importlib.util.spec_from_file_location("aggregate_central_v5_locked_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AGGREGATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGGREGATOR)


def _row(index: int, *, cbf: bool) -> dict[str, str]:
    return {
        "episode_index": str(index),
        "cooperative_safe_capture": "True",
        "capture_event": "True",
        "collision": "False",
        "world_violation_steps": "0",
        "transit_success": "True",
        "termination_reason": "safe_capture",
        "capture_time_seconds": "4.0",
        "min_clearance_m": "0.4",
        "mean_defender_path_length_m": "12.0",
        "mean_cbf_action_correction_norm": "0.2" if cbf else "0.0",
        "max_cbf_action_correction_norm": "0.4" if cbf else "0.0",
        "observation_condition": "nominal",
        "obstacle_count": "3",
        "planned_route_clearance_band": "medium",
        "target_motion_mode": "flee_persistence",
    }


def _write_locked_artifact(root: Path, run_id: str, checkpoint: Path, *, cbf: bool) -> None:
    directory = root / f"locked_s3_{run_id}_{'cbf' if cbf else 'raw'}_100"
    directory.mkdir(parents=True)
    rows = [_row(index, cbf=cbf) for index in range(100)]
    with (directory / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "scenes.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "episode_index": index,
                    "spec": {"episode_seed": 647201 + index},
                    "scenario": {"name": "locked", "obstacles": []},
                    "outcome": {"use_cbf": cbf},
                }
            )
            + "\n"
            for index in range(100)
        ),
        encoding="utf-8",
    )
    (directory / "evaluation_metadata.json").write_text(
        json.dumps(
            {
                "evaluation_type": "randomized_central_mixed_obstacle_s3_locked_test",
                "split": "locked_test",
                "locked_test": True,
                "seed_block": 647201,
                "episodes": 100,
                "use_cbf": cbf,
                "checkpoint": str(checkpoint),
            }
        ),
        encoding="utf-8",
    )


def test_collect_requires_three_passing_development_models_and_paired_locked_scenes(tmp_path: Path) -> None:
    root = tmp_path / "locked"
    models = []
    for training_seed in (661602, 661604, 661606):
        run_id = f"shapeaware_retained_seed{training_seed}"
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        checkpoint = run_dir / "checkpoint.pt"
        checkpoint.write_bytes(str(training_seed).encode("ascii"))
        summary = tmp_path / f"{run_id}_summary.json"
        summary.write_text(
            json.dumps(
                {
                    "candidate_gate_passed": True,
                    "training": {"checkpoint_sha256": AGGREGATOR._sha256(checkpoint)},
                    "s3_scene_pairing": {"static_scenes_exactly_paired": True},
                }
            ),
            encoding="utf-8",
        )
        _write_locked_artifact(root, run_id, checkpoint, cbf=False)
        _write_locked_artifact(root, run_id, checkpoint, cbf=True)
        models.append(
            {
                "run_id": run_id,
                "training_seed": training_seed,
                "run_dir": str(run_dir),
                "validation_summary": str(summary),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "locked_seed_block": 647201,
                "episodes_per_checkpoint": 100,
                "evaluation_root": str(root),
                "models": models,
            }
        ),
        encoding="utf-8",
    )

    aggregate = AGGREGATOR.collect(manifest)

    assert aggregate["locked_gate_passed"] is True
    assert aggregate["raw_cbf_scene_digests_paired"] is True
    assert aggregate["s3"]["cbf"]["across_training_seeds"]["cooperative_safe_capture_rate"]["mean"] == 1.0
    assert "All three pre-registered locked gates pass: True." in AGGREGATOR.render_markdown(aggregate)


def test_collect_rejects_development_summary_that_did_not_pass(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "locked_seed_block": 647201,
                "episodes_per_checkpoint": 100,
                "evaluation_root": str(tmp_path),
                "models": [],
            }
        ),
        encoding="utf-8",
    )

    import pytest

    with pytest.raises(ValueError, match="exactly three"):
        AGGREGATOR.collect(manifest)
