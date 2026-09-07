import json

import pytest

from scripts.summarize_dn_mpc_jepa_chunk_contract_audit import (
    _assert_same_manifest,
    _summary_from_current,
)


def test_contract_audit_requires_one_shared_scene_manifest() -> None:
    records = [
        {"inputs": {"scene_manifest_sha256": "same"}},
        {"inputs": {"scene_manifest_sha256": "same"}},
    ]
    assert _assert_same_manifest(records) == "same"
    with pytest.raises(ValueError, match="scene manifest hash"):
        _assert_same_manifest([records[0], {"inputs": {"scene_manifest_sha256": "different"}}])


def test_contract_audit_reads_one_paired_variant_and_contract(tmp_path) -> None:
    summary = {
        "metadata": {
            "contract": {"route_chunk_length_steps": 3},
            "inputs": {"scene_manifest_sha256": "manifest"},
            "git_revision": "revision",
        },
        "summaries": {
            "baseline": {
                "episodes": 4,
                "safe_capture_count": 2,
                "safe_capture_rate": 0.5,
            }
        },
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    record = _summary_from_current(path, "baseline")

    assert record["variant"] == "baseline"
    assert record["summary"]["safe_capture_count"] == 2
    assert record["contract"]["route_chunk_length_steps"] == 3
    assert record["inputs"]["scene_manifest_sha256"] == "manifest"
