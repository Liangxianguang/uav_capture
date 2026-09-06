from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_jepa_safe_capture_l0_l3_artifacts import (
    RUN_RE,
    canonical_manifest_sha256,
    expected_run_names,
)


def test_expected_matrix_has_21_unique_runs() -> None:
    names = expected_run_names()
    assert len(names) == 21
    assert len(set(names)) == 21
    assert all(RUN_RE.fullmatch(name) for name in names)


def test_canonical_manifest_ignores_training_seed(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    rows = [{"episode_index": 0, "scene_hash": "abc"}, {"episode_index": 1, "scene_hash": "def"}]
    first.write_text("".join(json.dumps({**row, "training_seed": 20260911}) + "\n" for row in rows), encoding="utf-8")
    second.write_text("".join(json.dumps({**row, "training_seed": 20260913}) + "\n" for row in rows), encoding="utf-8")
    assert canonical_manifest_sha256(first) == canonical_manifest_sha256(second)


def test_run_name_rejects_non_matrix_stage() -> None:
    assert RUN_RE.fullmatch("jepa_safe_capture_l0_l3_paired_full_seed20260911_m3")
    assert RUN_RE.fullmatch("jepa_safe_capture_l0_l3_paired_smoke_seed20260911_m3") is None


@pytest.mark.parametrize("seed", (20260911, 20260912, 20260913))
def test_each_seed_has_seven_variants(seed: int) -> None:
    names = [name for name in expected_run_names() if f"seed{seed}_" in name]
    assert len(names) == 7
