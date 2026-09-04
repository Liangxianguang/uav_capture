from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.replay_jepa_safe_capture_wp8_tie3 import _load_pairs


def _aggregate_fixture(path: Path, *, valid: bool = True) -> Path:
    comparisons = []
    for seed in (20260911, 20260912, 20260913):
        pairs = []
        for index in range(40):
            # 10 improved, 30 degraded and 80 tied over the three seeds.
            if seed == 20260911 and index < 4:
                delta = 1
            elif seed == 20260912 and index < 4:
                delta = 1
            elif seed == 20260913 and index < 2:
                delta = 1
            elif seed == 20260911 and index < 12:
                delta = -1
            elif seed == 20260912 and index < 15:
                delta = -1
            elif seed == 20260913 and index < 13:
                delta = -1
            else:
                delta = 0
            pairs.append(
                {
                    "episode_index": index,
                    "episode_seed": 649001 + index,
                    "base_safe_capture": delta == -1,
                    "candidate_safe_capture": delta == 1,
                    "delta": delta,
                }
            )
        comparisons.append({"training_seed": seed, "base_variant": "m0", "candidate_variant": "m3", "pairs": pairs})
    if not valid:
        comparisons[0]["pairs"] = comparisons[0]["pairs"][:-1]
    path.write_text(
        json.dumps({"stage": "full", "not_a_locked_test": True, "locked_test_opened": False, "paired_comparisons": comparisons}),
        encoding="utf-8",
    )
    return path


def test_load_pairs_requires_the_complete_tie3_block(tmp_path: Path) -> None:
    pairs = _load_pairs(_aggregate_fixture(tmp_path / "summary.json"))
    assert len(pairs) == 120
    assert {item["pair_label"] for item in pairs.values()} == {"degraded", "improved", "tied"}
    assert sum(item["pair_label"] == "degraded" for item in pairs.values()) == 30
    assert sum(item["pair_label"] == "improved" for item in pairs.values()) == 10
    assert sum(item["pair_label"] == "tied" for item in pairs.values()) == 80


def test_load_pairs_rejects_incomplete_seed_block(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Expected 40 M3 pairs"):
        _load_pairs(_aggregate_fixture(tmp_path / "summary.json", valid=False))
