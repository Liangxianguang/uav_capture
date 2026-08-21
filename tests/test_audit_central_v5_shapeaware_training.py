from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "audit_central_v5_shapeaware_training.py"
SPEC = importlib.util.spec_from_file_location("audit_central_v5_shapeaware_training", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


def _write_training(path: Path, losses: list[float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "action_mse"])
        writer.writeheader()
        for epoch, loss in enumerate(losses, start=1):
            writer.writerow({"epoch": epoch, "action_mse": loss})


def _episode() -> dict[str, object]:
    return {"accepted": True, "safe_capture_success": True, "cooperative_requirement_met": True, "layout": "wall"}


def test_collect_audits_warm_start_and_balanced_archives(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed"
    retained = tmp_path / "retained"
    fixed.mkdir()
    retained.mkdir()
    (fixed / "checkpoint.pt").write_bytes(b"fixed checkpoint")
    (fixed / "expert_sequence_dataset.npz").write_bytes(b"fixed archive")
    (fixed / "config.yaml").write_text("effective_imitation:\n  episodes: 2\n  expert_max_rejection_rate: 0.25\n", encoding="utf-8")
    (fixed / "expert_dataset_manifest.json").write_text(
        json.dumps(
            {
                "accepted_episodes": 2,
                "rejected_episodes": 0,
                "collection_attempts": 2,
                "expert_rejection_rate": 0.0,
                "sequence_count": 3,
                "frame_count": 96,
                "episodes": [_episode(), _episode()],
            }
        ),
        encoding="utf-8",
    )
    _write_training(fixed / "training.csv", [1.0, 0.1])

    fixed_hash = AUDITOR._sha256(fixed / "checkpoint.pt")
    nested_manifest = {
        "accepted_episodes": 2,
        "expert_rejection_rate": 0.0,
        "episodes": [_episode(), _episode()],
    }
    (retained / "checkpoint.pt").write_bytes(b"retained checkpoint")
    (retained / "expert_sequence_dataset.npz").write_bytes(b"retained archive")
    (retained / "initialization.json").write_text(json.dumps({"checkpoint": str(fixed / "checkpoint.pt"), "sha256": fixed_hash}), encoding="utf-8")
    (retained / "expert_dataset_manifest.json").write_text(
        json.dumps(
            {
                "source_balance": "equal_sequences",
                "sequence_count": 6,
                "frame_count": 192,
                "reused_expert_datasets": [
                    {"original_sequences": 3, "selected_sequences": 3, "manifest": nested_manifest},
                    {"original_sequences": 2, "selected_sequences": 3, "manifest": nested_manifest},
                ],
            }
        ),
        encoding="utf-8",
    )
    (retained / "source_hashes.json").write_text(json.dumps({"src/encirclement3d/pursuit_env.py": "a" * 64}), encoding="utf-8")
    _write_training(retained / "training.csv", [0.5, 0.05])

    audit = AUDITOR.collect(fixed, retained)
    report = AUDITOR.render_markdown(audit)

    assert audit["candidate_training_integrity_passed"] is True
    assert audit["retained_stage"]["warm_start_matches_fixed_checkpoint"] is True
    assert audit["retained_stage"]["source_selection_balanced"] is True
    assert "Training integrity passes" in report


def test_collect_enforces_preregistered_wall_distance_coverage(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed"
    retained = tmp_path / "retained"
    fixed.mkdir()
    retained.mkdir()
    (fixed / "checkpoint.pt").write_bytes(b"fixed checkpoint")
    (fixed / "expert_sequence_dataset.npz").write_bytes(b"fixed archive")
    (fixed / "config.yaml").write_text("effective_imitation:\n  episodes: 2\n", encoding="utf-8")
    episode = {
        "accepted": True,
        "safe_capture_success": True,
        "cooperative_requirement_met": True,
        "layout": "wall",
        "initial_side_distance": 5.0,
    }
    (fixed / "expert_dataset_manifest.json").write_text(
        json.dumps(
            {
                "accepted_episodes": 2,
                "rejected_episodes": 0,
                "collection_attempts": 2,
                "expert_rejection_rate": 0.0,
                "sequence_count": 3,
                "frame_count": 96,
                "episodes": [episode, episode],
            }
        ),
        encoding="utf-8",
    )
    _write_training(fixed / "training.csv", [1.0, 0.1])
    fixed_hash = AUDITOR._sha256(fixed / "checkpoint.pt")
    nested = {"accepted_episodes": 2, "expert_rejection_rate": 0.0, "episodes": [episode, episode]}
    (retained / "checkpoint.pt").write_bytes(b"retained checkpoint")
    (retained / "expert_sequence_dataset.npz").write_bytes(b"retained archive")
    (retained / "initialization.json").write_text(json.dumps({"checkpoint": str(fixed), "sha256": fixed_hash}), encoding="utf-8")
    (retained / "expert_dataset_manifest.json").write_text(
        json.dumps(
            {
                "source_balance": "equal_sequences",
                "sequence_count": 6,
                "frame_count": 192,
                "reused_expert_datasets": [
                    {"original_sequences": 3, "selected_sequences": 3, "manifest": nested},
                    {"original_sequences": 2, "selected_sequences": 3, "manifest": nested},
                ],
            }
        ),
        encoding="utf-8",
    )
    (retained / "source_hashes.json").write_text(json.dumps({}), encoding="utf-8")
    _write_training(retained / "training.csv", [0.5, 0.05])
    preregistration = tmp_path / "pre-registration.json"
    preregistration.write_text(
        json.dumps(
            {
                "fixed_stage_quality_gate": {
                    "minimum_accepted_wall_examples_per_initial_distance": {"5.0": 1, "5.5": 1}
                }
            }
        ),
        encoding="utf-8",
    )

    audit = AUDITOR.collect(fixed, retained, preregistration)

    assert audit["fixed_stage"]["wall_coverage_passed"] is False
    assert audit["candidate_training_integrity_passed"] is False
