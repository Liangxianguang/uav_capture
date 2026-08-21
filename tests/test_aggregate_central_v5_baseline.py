from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "aggregate_central_v5_baseline.py"
SPEC = importlib.util.spec_from_file_location("aggregate_central_v5_baseline", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AGGREGATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGGREGATOR)


def _row(index: int, *, cbf: bool) -> dict[str, str]:
    return {
        "cooperative_safe_capture": "True",
        "safe_capture_success": "True",
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
        "episode_index": str(index),
    }


def _write_artifact(directory: Path, *, cbf: bool, s3: bool) -> None:
    directory.mkdir(parents=True)
    rows = [_row(index, cbf=cbf) for index in range(60 if s3 else 20)]
    with (directory / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"use_cbf": cbf}
    if s3:
        metadata.update({"split": "validation", "locked_test": False, "seed_block": 646101})
        (directory / "evaluation_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (directory / "failure_index.json").write_text(
            json.dumps(
                {
                    "summary": {
                        "failure_stages": {},
                    },
                    "groups": {
                        field: {"nominal": {"episodes": 60, "cooperative_failure_rate": 0.0, "failure_stages": {}}}
                        for field in (
                            "observation_condition",
                            "obstacle_count",
                            "planned_route_clearance_band",
                            "target_motion_mode",
                        )
                    }
                }
            ),
            encoding="utf-8",
        )
        (directory / "scenes.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "episode_index": index,
                        "spec": {"episode_seed": 646101 + index},
                        "scenario": {"name": "frozen", "obstacles": []},
                        "outcome": {"use_cbf": cbf},
                    }
                )
                + "\n"
                for index in range(60)
            ),
            encoding="utf-8",
        )
    else:
        (directory / "summary.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_collect_validates_v5_artifact_contract_and_reports_gates(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    (run_dir / "expert_sequence_dataset.npz").write_bytes(b"archive")
    (run_dir / "config.yaml").write_text(
        "effective_imitation:\n  episodes: 320\n  expert_max_rejection_rate: 0.25\n",
        encoding="utf-8",
    )
    (run_dir / "expert_dataset_manifest.json").write_text(
        json.dumps(
            {
                "accepted_episodes": 320,
                "rejected_episodes": 0,
                "collection_attempts": 320,
                "expert_rejection_rate": 0.0,
                "expert_safe_capture_rate": 1.0,
                "expert_cooperative_requirement_rate": 1.0,
                "episodes": [{"safe_capture_success": True, "cooperative_requirement_met": True}],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "training.csv").write_text("epoch,action_mse\n1,0.1\n2,0.01\n", encoding="utf-8")

    root = tmp_path / "evaluations"
    run_id = "bc_baseline_seed661401"
    for scene in AGGREGATOR.FIXED_SCENES:
        for mode in AGGREGATOR.MODES:
            _write_artifact(root / f"{run_id}_{scene}_{mode}_20", cbf=mode == "cbf", s3=False)
    for mode in AGGREGATOR.MODES:
        _write_artifact(root / f"{run_id}_s3_validation_{mode}_60", cbf=mode == "cbf", s3=True)

    aggregate = AGGREGATOR.collect(run_dir, root, run_id)
    report = AGGREGATOR.render_markdown(aggregate)

    assert aggregate["candidate_gate_passed"] is True
    assert aggregate["s3_scene_pairing"]["static_scenes_exactly_paired"] is True
    assert aggregate["s3_validation"]["cbf"]["metrics"]["cooperative_safe_capture_wilson_95"][0] > 0.9
    assert "one-training-seed development-validation" in report
    assert "Raw actor and CBF execution are separate artifacts" in report
    assert "fixed-contract recovery" in AGGREGATOR.render_policy_failure_report(aggregate)


def test_collect_rejects_unpaired_s3_scene_inputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    (run_dir / "expert_sequence_dataset.npz").write_bytes(b"archive")
    (run_dir / "config.yaml").write_text("effective_imitation:\n  episodes: 320\n", encoding="utf-8")
    (run_dir / "expert_dataset_manifest.json").write_text(
        json.dumps({"accepted_episodes": 320, "rejected_episodes": 0, "collection_attempts": 320, "expert_rejection_rate": 0.0, "expert_safe_capture_rate": 1.0, "expert_cooperative_requirement_rate": 1.0, "episodes": [{"safe_capture_success": True, "cooperative_requirement_met": True}]}),
        encoding="utf-8",
    )
    (run_dir / "training.csv").write_text("epoch,action_mse\n1,0.1\n", encoding="utf-8")
    root = tmp_path / "evaluations"
    run_id = "bc_baseline_seed661401"
    for scene in AGGREGATOR.FIXED_SCENES:
        for mode in AGGREGATOR.MODES:
            _write_artifact(root / f"{run_id}_{scene}_{mode}_20", cbf=mode == "cbf", s3=False)
    for mode in AGGREGATOR.MODES:
        _write_artifact(root / f"{run_id}_s3_validation_{mode}_60", cbf=mode == "cbf", s3=True)
    scene_path = root / f"{run_id}_s3_validation_cbf_60" / "scenes.jsonl"
    records = scene_path.read_text(encoding="utf-8").splitlines()
    record = json.loads(records[0])
    record["scenario"]["name"] = "different"
    records[0] = json.dumps(record)
    scene_path.write_text("\n".join(records) + "\n", encoding="utf-8")

    import pytest
    with pytest.raises(ValueError, match="identical static scenes"):
        AGGREGATOR.collect(run_dir, root, run_id)


def test_training_quality_accepts_balanced_reused_archives(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    (run_dir / "expert_sequence_dataset.npz").write_bytes(b"archive")
    (run_dir / "config.yaml").write_text("effective_imitation:\n  episodes: 320\n", encoding="utf-8")
    nested = {"episodes": [{"safe_capture_success": True, "cooperative_requirement_met": True}]}
    (run_dir / "expert_dataset_manifest.json").write_text(
        json.dumps(
            {
                "source_balance": "equal_sequences",
                "reused_expert_datasets": [
                    {"original_sequences": 3, "selected_sequences": 5, "manifest": nested},
                    {"original_sequences": 5, "selected_sequences": 5, "manifest": nested},
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "training.csv").write_text("epoch,action_mse\n1,0.1\n", encoding="utf-8")

    quality = AGGREGATOR._training_quality(run_dir)

    assert quality["data_provenance"] == "reused_expert_archives"
    assert quality["source_selection_balanced"] is True
    assert quality["passed"] is True
