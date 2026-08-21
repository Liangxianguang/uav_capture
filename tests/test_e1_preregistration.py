from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_e1_preregistration import verify


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_e1_preregistration_matches_the_frozen_protocol_and_source_files() -> None:
    result = verify(PROJECT_ROOT)
    assert result["verified"] is True
    assert result["locked_seed_block"] == 681201
    assert result["checkpoint_sources"] == 3


def test_e1_source_manifest_records_missing_artifacts_without_fabricating_results() -> None:
    manifest = json.loads((PROJECT_ROOT / "E1_SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["availability_at_manifest_creation"]["all_checkpoint_files_present"] is False
    assert manifest["historical_evidence"]["historical_locked_seed_block"] == 647001
    assert [item["training_seed"] for item in manifest["required_frozen_checkpoints"]] == [661201, 661202, 661203]
