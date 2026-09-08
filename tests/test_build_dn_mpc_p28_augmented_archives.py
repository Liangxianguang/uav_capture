from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.build_dn_mpc_p28_augmented_archives import SAMPLE_TYPES, merge


def _write_archive(tmp_path: Path, name: str, offset: int, sample_types: list[int]) -> tuple[Path, Path]:
    dataset = tmp_path / f"{name}.npz"
    metadata = tmp_path / f"{name}.json"
    np.savez_compressed(
        dataset,
        inputs=np.full((len(sample_types), 1), float(offset), dtype=np.float32),
        route_action_chunk=np.zeros((len(sample_types), 5, 1), dtype=np.float32),
        sample_type=np.asarray(sample_types, dtype=np.int64),
    )
    payload = {
        "split": "train",
        "anticipatory_hard_negative_replay": {
            "enabled": name == "hard",
            "offline_only": True,
            "sample_type_mapping": SAMPLE_TYPES,
        },
    }
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    return dataset, metadata


def test_merge_appends_only_p28_rows(tmp_path: Path) -> None:
    base, base_meta = _write_archive(tmp_path, "base", 1, [0, 1])
    hard, hard_meta = _write_archive(tmp_path, "hard", 2, [5, 6, 0])
    merged, metadata, _provenance = merge(base, base_meta, hard, hard_meta)
    assert merged["inputs"].shape[0] == 4
    assert merged["sample_type"].tolist() == [0, 1, 5, 6]
    assert metadata["anticipatory_hard_negative_replay"]["merged_rows"] == 2
