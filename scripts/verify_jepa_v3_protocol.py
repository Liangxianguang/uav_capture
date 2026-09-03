"""Verify the frozen local inputs required by the JEPA-v3 development plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("JEPA-v3 protocol must be a mapping.")
    if protocol.get("phase") != "development_only" or protocol.get("locked_test_opened") is not False:
        raise ValueError("JEPA-v3 protocol must remain a closed development protocol.")
    if protocol.get("model_contract", {}).get("cbf_is_final_safety_filter") is not True:
        raise ValueError("JEPA-v3 requires CBF as the final safety filter.")
    if protocol.get("training", {}).get("tensorboard", {}).get("required") is not True:
        raise ValueError("JEPA-v3 requires TensorBoard training records.")
    return protocol


def verify_manifest(manifest_path: Path, root: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("locked_test_opened") is not False:
        raise ValueError("Manifest must declare locked_test_opened: false.")
    entries = manifest.get("artifacts")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Manifest must contain frozen artifacts.")
    verification: dict[str, Any] = {}
    for name, expected in entries.items():
        path = root / str(expected["path"])
        if not path.is_file():
            verification[name] = {"path": str(path), "exists": False, "matches": False}
            continue
        actual_hash = sha256(path)
        actual_size = path.stat().st_size
        verification[name] = {
            "path": str(path),
            "exists": True,
            "expected_sha256": expected["sha256"],
            "actual_sha256": actual_hash,
            "expected_bytes": int(expected["bytes"]),
            "actual_bytes": actual_size,
            "matches": actual_hash == expected["sha256"] and actual_size == int(expected["bytes"]),
        }
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs/jepa_v3_development_protocol.yaml")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "docs/JEPA_V3_INPUT_MANIFEST_20260903.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol.resolve())
    verification = verify_manifest(args.manifest.resolve(), PROJECT_ROOT)
    result = {
        "protocol": str(args.protocol.resolve()),
        "protocol_name": protocol["protocol_name"],
        "locked_test_opened": protocol["locked_test_opened"],
        "all_artifacts_match": all(item["matches"] for item in verification.values()),
        "artifacts": verification,
    }
    if args.output is not None:
        args.output.resolve().write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_artifacts_match"]:
        raise SystemExit("Frozen JEPA-v3 artifact verification failed.")


if __name__ == "__main__":
    main()
