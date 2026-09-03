"""Verify that a zero-perturbation JEPA run preserves paired V5 + CBF behaviour."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


JEPA_ONLY_FIELDS = {
    "jepa_enabled",
    "jepa_candidate_count",
    "jepa_perturbation_mps",
    "jepa_action_chunk_length_steps",
    "jepa_action_chunk_semantics",
    "jepa_mean_selected_index",
    "jepa_mean_selected_score",
    "jepa_reliability_ledger_enabled",
    "jepa_ledger_mean_credit",
    "jepa_ledger_nominal_fallback_fraction",
    "jepa_ledger_global_fallback_fraction",
}
PAIR_FIELDS = ("episode_index", "episode_seed", "layout_seed")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(directory: Path) -> list[dict[str, str]]:
    with (directory / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compare(baseline_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    baseline_rows = load_rows(baseline_dir)
    candidate_rows = load_rows(candidate_dir)
    if len(baseline_rows) != len(candidate_rows):
        raise ValueError("Paired zero-perturbation runs have different episode counts.")
    scene_baseline = baseline_dir / "scenes.jsonl"
    scene_candidate = candidate_dir / "scenes.jsonl"
    scene_hashes = {"baseline": sha256(scene_baseline), "candidate": sha256(scene_candidate)}
    scene_byte_identical = scene_baseline.read_bytes() == scene_candidate.read_bytes()
    fields = sorted(
        (set(baseline_rows[0]) & set(candidate_rows[0])).difference(JEPA_ONLY_FIELDS)
        if baseline_rows
        else set()
    )
    differences: list[dict[str, Any]] = []
    for index, (baseline, candidate) in enumerate(zip(baseline_rows, candidate_rows)):
        pair = {field: baseline.get(field) for field in PAIR_FIELDS}
        if any(baseline.get(field) != candidate.get(field) for field in PAIR_FIELDS):
            differences.append({"episode_row": index, "field": "pairing", "baseline": pair, "candidate": {field: candidate.get(field) for field in PAIR_FIELDS}})
            continue
        for field in fields:
            if baseline.get(field) != candidate.get(field):
                differences.append(
                    {
                        "episode_row": index,
                        "episode": pair,
                        "field": field,
                        "baseline": baseline.get(field),
                        "candidate": candidate.get(field),
                    }
                )
    return {
        "comparison_type": "jepa_v3_zero_perturbation_regression",
        "not_a_locked_test": True,
        "baseline_dir": str(baseline_dir.resolve()),
        "candidate_dir": str(candidate_dir.resolve()),
        "episodes": len(baseline_rows),
        "non_jepa_fields_compared": len(fields),
        "scene_sha256": scene_hashes,
        "scenes_byte_identical": scene_byte_identical,
        "field_difference_count": len(differences),
        "field_differences": differences,
        "passed": bool(scene_byte_identical and not differences),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite zero-perturbation comparison output.")
    result = compare(args.baseline_dir.resolve(), args.candidate_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Zero-perturbation regression failed.")


if __name__ == "__main__":
    main()
