"""Verify the immutable E1 pre-registration against its frozen implementation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.e1_protocol import E1_PROFILES, E1_SPLITS, load_e1_protocol  # noqa: E402


PREREGISTRATION = PROJECT_ROOT / "E1_EXECUTION_DYNAMICS_PREREGISTRATION.json"
SOURCE_MANIFEST = PROJECT_ROOT / "E1_SOURCE_MANIFEST.json"


def verify(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    preregistration_path = project_root / PREREGISTRATION.name
    source_manifest_path = project_root / SOURCE_MANIFEST.name
    document = json.loads(preregistration_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    protocol_path = project_root / str(document["protocol"]["path"])
    if _sha256(protocol_path) != str(document["protocol"]["sha256"]):
        raise ValueError("E1 protocol SHA-256 differs from the pre-registration.")
    protocol = load_e1_protocol(protocol_path)
    if tuple(document["protocol"]["profiles"]) != E1_PROFILES:
        raise ValueError("E1 pre-registration profile set differs from E0-E6.")
    if set(document["protocol"]["seed_blocks"]) != set(E1_SPLITS):
        raise ValueError("E1 pre-registration seed blocks are incomplete.")
    if document["protocol"]["seed_blocks"] != protocol["seed_blocks"]:
        raise ValueError("E1 protocol seed blocks differ from the pre-registration.")
    for relative, expected in dict(document["source_files"]).items():
        actual = _sha256(project_root / relative)
        if actual != expected:
            raise ValueError(f"E1 source hash changed: {relative}")
    checkpoints = source_manifest.get("required_frozen_checkpoints")
    if not isinstance(checkpoints, list) or [item.get("training_seed") for item in checkpoints] != [661201, 661202, 661203]:
        raise ValueError("E1 source manifest must retain all three V4 checkpoint seeds.")
    return {
        "verified": True,
        "protocol_sha256": _sha256(protocol_path),
        "source_file_count": len(document["source_files"]),
        "checkpoint_sources": len(checkpoints),
        "locked_seed_block": int(protocol["seed_blocks"]["locked_test"]),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2), flush=True)
