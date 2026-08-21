"""Summarize the wall-specific expert coverage of a fixed V5 training archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


STRATUM_FIELDS = (
    "defender_side",
    "initial_side_distance",
    "target_speed_scale",
    "target_motion_mode",
    "target_crossing_required",
    "required_defender_zone_entries",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return document


def collect(fixed_run_dir: Path, layout: str = "wall") -> dict[str, Any]:
    manifest_path = fixed_run_dir / "expert_dataset_manifest.json"
    manifest = _read_json(manifest_path)
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("Fixed expert manifest must contain an episodes list")
    accepted = [episode for episode in episodes if episode.get("accepted", True)]
    selected = [episode for episode in accepted if str(episode.get("layout")) == layout]
    if not selected:
        raise ValueError(f"Fixed archive has no accepted {layout!r} examples")
    counts = Counter(tuple(episode.get(field) for field in STRATUM_FIELDS) for episode in selected)
    strata = [
        {**dict(zip(STRATUM_FIELDS, values, strict=True)), "accepted_episodes": count}
        for values, count in sorted(counts.items(), key=lambda item: tuple(str(value) for value in item[0]))
    ]
    layout_counts = Counter(str(episode.get("layout", "unknown")) for episode in accepted)
    return {
        "analysis_type": "central_v5_fixed_archive_layout_coverage",
        "fixed_run_dir": str(fixed_run_dir.resolve()),
        "fixed_manifest_sha256": _sha256(manifest_path),
        "fixed_archive_sha256": _sha256(fixed_run_dir / "expert_sequence_dataset.npz"),
        "layout": layout,
        "accepted_expert_episodes": len(accepted),
        "accepted_layout_episodes": len(selected),
        "accepted_episode_count_by_layout": dict(sorted(layout_counts.items())),
        "stratum_fields": list(STRATUM_FIELDS),
        "strata": strata,
    }


def render_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Central V5 Fixed Archive Wall Coverage Audit",
        "",
        "This audit describes accepted expert demonstrations only. It does not assess a policy or change an evaluation split.",
        "",
        f"- Fixed archive SHA-256: `{analysis['fixed_archive_sha256']}`",
        f"- Manifest SHA-256: `{analysis['fixed_manifest_sha256']}`",
        f"- Accepted expert episodes / selected {analysis['layout']} episodes: `{analysis['accepted_expert_episodes']}` / `{analysis['accepted_layout_episodes']}`",
        "",
        "## Accepted Layout Counts",
        "",
        "| Layout | Accepted episodes |",
        "| --- | ---: |",
    ]
    for layout, count in analysis["accepted_episode_count_by_layout"].items():
        lines.append(f"| {layout} | {count} |")
    lines.extend(
        [
            "",
            f"## {analysis['layout'].title()} Strata",
            "",
            "| Defender side | Initial distance | Target speed | Target motion | Target crossing | Required zone entries | Accepted episodes |",
            "| --- | ---: | ---: | --- | --- | ---: | ---: |",
        ]
    )
    for row in analysis["strata"]:
        lines.append(
            f"| {row['defender_side']} | {row['initial_side_distance']} | {row['target_speed_scale']} | "
            f"{row['target_motion_mode']} | {row['target_crossing_required']} | "
            f"{row['required_defender_zone_entries']} | {row['accepted_episodes']} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-run-dir", type=Path, required=True)
    parser.add_argument("--layout", default="wall")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = collect(args.fixed_run_dir.resolve(), args.layout)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(analysis), encoding="utf-8")


if __name__ == "__main__":
    main()
