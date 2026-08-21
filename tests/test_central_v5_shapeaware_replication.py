from __future__ import annotations

import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(name: str) -> dict:
    with (PROJECT_ROOT / "configs" / name).open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    assert isinstance(document, dict)
    return document


def test_frozen_replication_manifest_preserves_the_gate_and_locked_holdout() -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "FROZEN_V5_SHAPEAWARE_REPLICATION_MANIFEST.json").read_text(encoding="utf-8")
    )

    assert manifest["development_checkpoint"]["candidate_gate_passed"] is True
    assert manifest["evaluation_contract"]["s3_development"]["seed_block"] == 646101
    assert manifest["evaluation_contract"]["s3_locked"]["seed_block"] == 647201
    assert manifest["evaluation_contract"]["s3_locked"]["not_opened"] is True
    assert manifest["development_gate"] == {
        "s3_cooperative_safe_capture_at_least": 0.85,
        "s3_collision_at_most": 0.02,
        "s3_boundary_at_most": 0.02,
        "s3_transit_at_least": 0.99,
        "all_fixed_cbf_cooperative_safe_capture_at_least": 0.98,
    }


def test_replication_configs_match_the_frozen_shapeaware_contract() -> None:
    fixed_1 = _load_yaml(
        "capture_radius_recurrent_behavior_cloning_central_v5_fixed_shapeaware_seed661603.yaml"
    )
    fixed_2 = _load_yaml(
        "capture_radius_recurrent_behavior_cloning_central_v5_fixed_shapeaware_seed661605.yaml"
    )
    retained_1 = _load_yaml(
        "capture_radius_recurrent_behavior_cloning_central_v5_shapeaware_retained_seed661604.yaml"
    )
    retained_2 = _load_yaml(
        "capture_radius_recurrent_behavior_cloning_central_v5_shapeaware_retained_seed661606.yaml"
    )

    for fixed, expected_seed in ((fixed_1, 661603), (fixed_2, 661605)):
        imitation = fixed["imitation"]
        assert imitation["seed"] == expected_seed
        assert imitation["episodes"] == 640
        assert imitation["epochs"] == 96
        assert imitation["learning_rate"] == 3e-4
        assert imitation["hidden_dim"] == 128
        assert imitation["action_scale_mode"] == "full_range"
        assert imitation["expert_require_safe_capture"] is True
        assert imitation["expert_require_cooperative_safe_capture"] is True
        assert imitation["training_required_defender_zone_entries"] == 2

    for retained, fixed_seed, expected_seed in (
        (retained_1, 661603, 661604),
        (retained_2, 661605, 661606),
    ):
        imitation = retained["imitation"]
        assert imitation["seed"] == expected_seed
        assert f"fixed_shapeaware_seed{fixed_seed}/checkpoint.pt" in imitation["initialize_from"]
        assert f"fixed_shapeaware_seed{fixed_seed}/expert_sequence_dataset.npz" in imitation["expert_datasets"][0]
        assert "bc_baseline_seed661401/expert_sequence_dataset.npz" in imitation["expert_datasets"][1]
        assert imitation["expert_dataset_source_balance"] == "equal_sequences"
        assert imitation["epochs"] == 64
        assert imitation["learning_rate"] == 5e-5
        assert imitation["hidden_dim"] == 128
        assert imitation["action_scale_mode"] == "full_range"
        assert imitation["training_required_defender_zone_entries"] == 2


def test_replication_launcher_refuses_existing_outputs() -> None:
    launcher = (PROJECT_ROOT / "scripts" / "run_central_v5_shapeaware_replication.ps1").read_text(
        encoding="utf-8"
    )

    assert "Refusing to overwrite existing output" in launcher
    assert "shapeaware_retained_seed661604" in launcher
    assert "shapeaware_retained_seed661606" in launcher
