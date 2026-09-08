"""Merge P25 traceable archives with offline P28 hard-negative rows.

The merger is deterministic and provenance-bound.  It appends only P28 rows
whose sample type is 5-8; it never changes the P25 runtime rows, validation
block, CBF margins, or route candidate contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter

SAMPLE_TYPES = {"braking": 5, "left_detour": 6, "right_detour": 7, "boundary_rescue": 8}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--base-metadata", type=Path, required=True)
    parser.add_argument("--hard-dataset", type=Path, required=True)
    parser.add_argument("--hard-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def merge(base_dataset: Path, base_metadata: Path, hard_dataset: Path, hard_metadata: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    base_meta = json.loads(base_metadata.read_text(encoding="utf-8"))
    hard_meta = json.loads(hard_metadata.read_text(encoding="utf-8"))
    if base_meta.get("split") != hard_meta.get("split"):
        raise ValueError("Base and P28 archives must have the same split")
    if hard_meta.get("anticipatory_hard_negative_replay", {}).get("enabled") is not True:
        raise ValueError("Hard archive is not a declared P28 replay")
    if hard_meta.get("anticipatory_hard_negative_replay", {}).get("sample_type_mapping") != SAMPLE_TYPES:
        raise ValueError("P28 sample-type mapping is not stable")
    with np.load(base_dataset, allow_pickle=False) as base_archive, np.load(hard_dataset, allow_pickle=False) as hard_archive:
        if set(base_archive.files) != set(hard_archive.files):
            missing = sorted(set(base_archive.files) ^ set(hard_archive.files))
            raise ValueError(f"Archive arrays differ: {missing}")
        base = {name: np.asarray(base_archive[name]) for name in base_archive.files}
        hard = {name: np.asarray(hard_archive[name]) for name in hard_archive.files}
    hard_mask = np.isin(hard["sample_type"], list(SAMPLE_TYPES.values()))
    if not bool(hard_mask.any()):
        raise ValueError("P28 archive has no sample_type 5-8 rows")
    merged = {name: np.concatenate((base[name], hard[name][hard_mask]), axis=0) for name in base}
    if not np.isfinite(merged["inputs"]).all() or not np.isfinite(merged["route_action_chunk"]).all():
        raise ValueError("Merged archive has non-finite inputs or actions")
    metadata = dict(base_meta)
    # Keep the structural route-contract version unchanged so the augmented
    # train/calibration archives remain paired with the frozen validation
    # contract.  Augmentation provenance is explicit below rather than
    # encoded as a new tensor schema version.
    metadata["augmentation_version"] = "p28_anticipatory_hard_negative_v1"
    metadata["sample_count_per_defender"] = int(merged["inputs"].shape[0])
    metadata["array_shapes"] = {name: list(value.shape) for name, value in merged.items()}
    metadata["source"] = {
        "base_dataset": str(base_dataset.resolve()),
        "base_dataset_sha256": sha256(base_dataset),
        "base_metadata": str(base_metadata.resolve()),
        "base_metadata_sha256": sha256(base_metadata),
        "hard_negative_dataset": str(hard_dataset.resolve()),
        "hard_negative_dataset_sha256": sha256(hard_dataset),
        "hard_negative_metadata": str(hard_metadata.resolve()),
        "hard_negative_metadata_sha256": sha256(hard_metadata),
    }
    metadata["anticipatory_hard_negative_replay"] = dict(hard_meta["anticipatory_hard_negative_replay"])
    metadata["anticipatory_hard_negative_replay"]["merged_rows"] = int(hard_mask.sum())
    metadata["anticipatory_hard_negative_replay"]["base_rows"] = int(base["inputs"].shape[0])
    metadata["class_counts"] = {
        "runtime_rows": int(np.sum(merged["sample_type"] == 0)),
        "boundary_shadow_rows": int(np.sum(merged["sample_type"] == 1)),
        "interaction_rows": int(np.sum(np.isin(merged["sample_type"], [2, 3, 4]))),
        "anticipatory_hard_negative_rows": int(np.sum(merged["sample_type"] >= 5)),
    }
    provenance = {
        "development_only": True,
        "locked_test_opened": False,
        "raw_unverified_action_executed": False,
        "cbf_margin_changed": False,
        "merged_dataset_sha256": None,
        "base_dataset_sha256": sha256(base_dataset),
        "hard_negative_dataset_sha256": sha256(hard_dataset),
    }
    return merged, metadata, provenance


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    tensorboard = args.tensorboard_logdir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    if tensorboard.exists() and any(tensorboard.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard directory: {tensorboard}")
    merged, metadata, provenance = merge(
        args.base_dataset.resolve(), args.base_metadata.resolve(),
        args.hard_dataset.resolve(), args.hard_metadata.resolve(),
    )
    output.mkdir(parents=True, exist_ok=True)
    tensorboard.mkdir(parents=True, exist_ok=True)
    dataset = output / "route_identity_counterfactual.npz"
    np.savez_compressed(dataset, **merged)
    provenance["merged_dataset_sha256"] = sha256(dataset)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=_json) + "\n", encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True, default=_json) + "\n", encoding="utf-8")
    with SummaryWriter(log_dir=str(tensorboard), flush_secs=1) as writer:
        writer.add_text("P28Augment/Metadata", json.dumps(metadata, sort_keys=True), 0)
        writer.add_text("P28Augment/Provenance", json.dumps(provenance, sort_keys=True), 0)
        writer.add_scalar("P28Augment/TotalRows", merged["inputs"].shape[0], 0)
        writer.add_scalar("P28Augment/BaseRows", metadata["anticipatory_hard_negative_replay"]["base_rows"], 0)
        writer.add_scalar("P28Augment/HardNegativeRows", metadata["anticipatory_hard_negative_replay"]["merged_rows"], 0)
    print(json.dumps({"dataset": str(dataset), "metadata": str(output / 'metadata.json'), "rows": int(merged['inputs'].shape[0]), "hard_negative_rows": int((merged['sample_type'] >= 5).sum()), "tensorboard": str(tensorboard)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
