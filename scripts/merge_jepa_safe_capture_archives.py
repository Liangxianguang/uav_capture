"""Merge disjoint JEPA safe-capture archive shards into one train archive.

The merger is deliberately strict: all shards must share the frozen input
contract and collection/protocol hashes, while episode seeds must be disjoint.
The merged archive remains development-only and records every input hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
from importlib.metadata import version

from torch.utils.tensorboard import SummaryWriter


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    return parser.parse_args()


def load_shard(directory: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], Path]:
    directory = directory.resolve()
    dataset = directory / "counterfactual_safe_capture_v2.npz"
    metadata_path = directory / "metadata.json"
    if not dataset.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete archive shard: {directory}")
    metadata = load_json(metadata_path)
    with np.load(dataset, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if not arrays or any(not np.isfinite(value).all() for value in arrays.values() if np.issubdtype(value.dtype, np.number)):
        raise ValueError(f"Shard contains no arrays or non-finite values: {directory}")
    return arrays, metadata, dataset


def main() -> None:
    args = parse_args()
    input_dirs = [path.resolve() for path in args.input_dir]
    if len(set(input_dirs)) != len(input_dirs):
        raise ValueError("Duplicate input shard directory.")
    output = args.output.resolve()
    tensorboard = args.tensorboard_logdir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    if tensorboard.exists() and any(tensorboard.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard directory: {tensorboard}")

    loaded = [load_shard(path) for path in input_dirs]
    first_arrays, first_metadata, first_dataset = loaded[0]
    required_keys = tuple(first_arrays.keys())
    reference = {
        "dataset_version": first_metadata.get("dataset_version"),
        "split": first_metadata.get("split"),
        "collection_config_sha256": first_metadata.get("collection_config_sha256"),
        "protocol_sha256": first_metadata.get("protocol_sha256"),
        "input_shapes": {name: tuple(value.shape[1:]) for name, value in first_arrays.items()},
    }
    if reference["split"] != "train":
        raise ValueError(f"Only train shards may be merged, got {reference['split']!r}.")

    all_arrays: dict[str, list[np.ndarray]] = {name: [] for name in required_keys}
    all_seeds: list[int] = []
    scenario_records: list[dict[str, Any]] = []
    source_archives: list[dict[str, Any]] = []
    for arrays, metadata, dataset in loaded:
        if tuple(arrays.keys()) != required_keys:
            raise ValueError("All shards must contain the same arrays in the same order.")
        for name, value in arrays.items():
            if tuple(value.shape[1:]) != reference["input_shapes"][name]:
                raise ValueError(f"Array shape mismatch for {name}.")
            all_arrays[name].append(value)
        for field in ("dataset_version", "split", "collection_config_sha256", "protocol_sha256"):
            if metadata.get(field) != reference[field]:
                raise ValueError(f"Shard {dataset.parent} disagrees on {field}.")
        seeds = [int(seed) for seed in metadata.get("episode_seeds", [])]
        if len(seeds) != int(metadata.get("episode_seed_count", -1)) or len(set(seeds)) != len(seeds):
            raise ValueError(f"Invalid episode seed metadata in {dataset.parent}.")
        if set(all_seeds).intersection(seeds):
            raise ValueError(f"Episode seeds overlap in {dataset.parent}.")
        all_seeds.extend(seeds)
        scenario_records.extend(metadata.get("scenario_records", []))
        source_archives.append(
            {
                "directory": str(dataset.parent),
                "dataset": str(dataset),
                "dataset_sha256": sha256(dataset),
                "metadata": str(dataset.parent / "metadata.json"),
                "metadata_sha256": sha256(dataset.parent / "metadata.json"),
                "episode_seed_count": len(seeds),
            }
        )

    merged = {name: np.concatenate(parts, axis=0) for name, parts in all_arrays.items()}
    output.mkdir(parents=True, exist_ok=True)
    dataset = output / "counterfactual_safe_capture_v2.npz"
    np.savez_compressed(dataset, **merged)

    metadata = dict(first_metadata)
    metadata.update(
        {
            "episode_seeds": sorted(all_seeds),
            "episode_seed_count": len(all_seeds),
            "scenario_records": scenario_records,
            "merged_shard_count": len(input_dirs),
            "merged_source_archives": source_archives,
            "merged_dataset_sha256": sha256(dataset),
            "merged_from_disjoint_shards": True,
            "locked_test_opened": False,
        }
    )
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    scenario_manifest = output / "scenario_manifest.json"
    scenario_manifest.write_text(
        json.dumps({"split": "train", "scenarios": scenario_records, "episode_seeds": sorted(all_seeds)}, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset_version": metadata["dataset_version"],
        "split": "train",
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "metadata": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
        "scenario_manifest": str(scenario_manifest),
        "scenario_manifest_sha256": sha256(scenario_manifest),
        "tensorboard_logdir": str(tensorboard),
        "episode_seeds": sorted(all_seeds),
        "seed_count": len(all_seeds),
        "merged_shards": source_archives,
        "locked_test_opened": False,
    }
    (output / "archive_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    writer = SummaryWriter(log_dir=str(tensorboard), flush_secs=10)
    writer.add_text("Merge/input_shards", json.dumps(source_archives, indent=2), 0)
    writer.add_text("Merge/metadata", json.dumps(metadata, indent=2), 0)
    writer.add_text("Merge/provenance", json.dumps({"dataset_sha256": manifest["dataset_sha256"], "locked_test_opened": False}, indent=2), 0)
    writer.add_scalar("Data/shard_count", len(input_dirs), 0)
    writer.add_scalar("Data/episode_count", len(all_seeds), 0)
    writer.add_scalar("Data/sample_count", int(merged["inputs"].shape[0]), 0)
    writer.add_scalar("Data/nominal_fraction", float(np.mean(merged["candidate_is_nominal"])), 0)
    writer.add_scalar("Data/all_finite", 1.0, 0)
    writer.flush()
    writer.close()
    print(
        json.dumps(
            {
                "split": "train",
                "shards": len(input_dirs),
                "episodes": len(all_seeds),
                "samples": int(merged["inputs"].shape[0]),
                "dataset_sha256": manifest["dataset_sha256"],
                "locked_test_opened": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
