"""Build an auditable S3 failure index from an episode-level CSV.

The index is deliberately episode based.  It refuses to infer failures from a
grouped summary, because grouped counts cannot identify the trajectory stage or
reproduce a hard example.  The same tool can be used for the expert feasibility
run and later for policy + CBF validation artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


GROUP_FIELDS = (
    "observation_condition",
    "obstacle_count",
    "layout_signature",
    "defender_side",
    "target_speed_scale",
    "target_motion_mode",
)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any, default: float | None = None) -> float | None:
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_failure(row: dict[str, str]) -> str:
    """Assign one primary stage while retaining raw termination fields."""
    if _bool(row.get("cooperative_safe_capture")):
        return "cooperative_safe_capture"
    if (
        _bool(row.get("collision"))
        or _bool(row.get("target_obstacle_collision"))
        or float(row.get("world_violation_steps", 0) or 0) > 0
    ):
        return "safety_failure"
    if row.get("termination_reason") == "timeout" or row.get("task_termination_reason") == "timeout":
        return "timeout"
    if not _bool(row.get("capture_event")):
        return "no_capture"
    if not _bool(row.get("safe_capture_success")):
        return "unsafe_capture"
    return "cooperation_failure"


def _hard_flags(row: dict[str, str], *, clearance_threshold: float, correction_threshold: float) -> list[str]:
    flags: list[str] = []
    if classify_failure(row) != "cooperative_safe_capture":
        flags.append("task_failure")
    min_clearance = _float(row.get("min_clearance_m"))
    if min_clearance is not None and min_clearance < clearance_threshold:
        flags.append("low_clearance")
    correction = _float(row.get("max_cbf_action_correction_norm"))
    if correction is not None and correction > correction_threshold:
        flags.append("large_cbf_correction")
    return flags


def _group(rows: Iterable[dict[str, Any]], field: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(field, "unknown"))].append(row)
    output: dict[str, Any] = {}
    for key, subset in sorted(buckets.items()):
        failures = [row for row in subset if row["failure_stage"] != "cooperative_safe_capture"]
        output[key] = {
            "episodes": len(subset),
            "cooperative_safe_captures": len(subset) - len(failures),
            "cooperative_failures": len(failures),
            "cooperative_failure_rate": len(failures) / len(subset),
            "failure_stages": dict(sorted(Counter(row["failure_stage"] for row in failures).items())),
        }
    return output


def build_index(
    episodes_csv: Path,
    *,
    clearance_threshold: float = 0.30,
    correction_threshold: float = 0.25,
) -> dict[str, Any]:
    episodes_csv = episodes_csv.resolve()
    if not episodes_csv.is_file():
        raise FileNotFoundError(f"Episode CSV does not exist: {episodes_csv}")
    with episodes_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Episode CSV is empty: {episodes_csv}")
    required = {"episode_index", "episode_seed", "layout_seed", "cooperative_safe_capture", "termination_reason"}
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"Episode CSV is missing required fields: {', '.join(missing)}")

    indexed: list[dict[str, Any]] = []
    for row in rows:
        failure_stage = classify_failure(row)
        hard_flags = _hard_flags(
            row,
            clearance_threshold=clearance_threshold,
            correction_threshold=correction_threshold,
        )
        indexed.append(
            {
                "episode_index": int(row["episode_index"]),
                "episode_seed": int(row["episode_seed"]),
                "layout_seed": int(row["layout_seed"]),
                "method": row.get("method"),
                "use_cbf": _bool(row.get("use_cbf")),
                "observation_condition": row.get("observation_condition"),
                "obstacle_count": int(row["obstacle_count"]) if row.get("obstacle_count") else None,
                "layout_signature": row.get("layout_signature"),
                "defender_side": row.get("defender_side"),
                "target_speed_scale": _float(row.get("target_speed_scale")),
                "target_motion_mode": row.get("target_motion_mode"),
                "failure_stage": failure_stage,
                "hard_example_flags": hard_flags,
                "capture_event": _bool(row.get("capture_event")),
                "safe_capture_success": _bool(row.get("safe_capture_success")),
                "cooperative_safe_capture": _bool(row.get("cooperative_safe_capture")),
                "collision": _bool(row.get("collision")),
                "boundary_violation": float(row.get("world_violation_steps", 0) or 0) > 0,
                "termination_reason": row.get("termination_reason"),
                "task_termination_reason": row.get("task_termination_reason"),
                "min_clearance_m": _float(row.get("min_clearance_m")),
                "capture_time_seconds": _float(row.get("capture_time_seconds")),
                "max_cbf_action_correction_norm": _float(row.get("max_cbf_action_correction_norm")),
                "transit_success": _bool(row.get("transit_success")),
                "defender_zone_entry_count": int(row["defender_zone_entry_count"])
                if row.get("defender_zone_entry_count")
                else None,
            }
        )

    failures = [row for row in indexed if row["failure_stage"] != "cooperative_safe_capture"]
    hard_examples = [row for row in indexed if row["hard_example_flags"]]
    return {
        "index_type": "central_v5_s3_episode_failure_index",
        "episode_level": True,
        "source_csv": str(episodes_csv),
        "source_csv_sha256": _sha256(episodes_csv),
        "thresholds": {
            "min_clearance_below_m": clearance_threshold,
            "max_cbf_action_correction_above": correction_threshold,
        },
        "summary": {
            "episodes": len(indexed),
            "cooperative_safe_captures": len(indexed) - len(failures),
            "cooperative_failures": len(failures),
            "cooperative_failure_rate": len(failures) / len(indexed),
            "hard_examples": len(hard_examples),
            "failure_stages": dict(sorted(Counter(row["failure_stage"] for row in failures).items())),
            "transit_failures": sum(not row["transit_success"] for row in indexed),
        },
        "groups": {field: _group(indexed, field) for field in GROUP_FIELDS},
        "failures": failures,
        "hard_examples": hard_examples,
    }


def render_report(index: dict[str, Any]) -> str:
    summary = index["summary"]
    lines = [
        "# V5 S3 Failure Analysis (Expert Feasibility Validation)",
        "",
        "This report audits a fresh V5 validation artifact generated with `rule expert + CBF`.",
        "It is a map/dynamics feasibility check, not a learned-policy improvement claim.",
        "",
        f"- Episodes: `{summary['episodes']}`",
        f"- Cooperative Safe Capture: `{summary['cooperative_safe_captures']}/{summary['episodes']}`",
        f"- Cooperative failures: `{summary['cooperative_failures']}` ({100.0 * summary['cooperative_failure_rate']:.2f}%)",
        f"- Transit failures: `{summary['transit_failures']}`",
        f"- Hard-example rows: `{summary['hard_examples']}`",
        f"- Source CSV SHA-256: `{index['source_csv_sha256']}`",
        "",
        "## Failure stages",
        "",
        "| Stage | Count |",
        "| --- | ---: |",
    ]
    for stage, count in summary["failure_stages"].items():
        lines.append(f"| {stage} | {count} |")
    if not summary["failure_stages"]:
        lines.append("| none | 0 |")

    lines.extend(
        [
            "",
            "## Episode-level failures",
            "",
            "| Episode | Episode seed | Layout seed | Condition | Obstacles | Stage | Termination | Min clearance (m) |",
            "| ---: | ---: | ---: | --- | ---: | --- | --- | ---: |",
        ]
    )
    for row in index["failures"]:
        clearance = "n/a" if row["min_clearance_m"] is None else f"{row['min_clearance_m']:.3f}"
        lines.append(
            f"| {row['episode_index']} | {row['episode_seed']} | {row['layout_seed']} | "
            f"{row['observation_condition']} | {row['obstacle_count']} | {row['failure_stage']} | "
            f"{row['termination_reason']} | {clearance} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The expert result establishes that the V5 validation maps have a safe route "
            "under the current kinematic contract. It does not explain retained-BC failures, "
            "because no policy checkpoint was evaluated in this run. The same tool must be "
            "run on policy + CBF `episodes.csv` before selecting hard examples for training.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--clearance-threshold", type=float, default=0.30)
    parser.add_argument("--correction-threshold", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = build_index(
        args.episodes_csv,
        clearance_threshold=args.clearance_threshold,
        correction_threshold=args.correction_threshold,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(index, indent=2), encoding="utf-8")
    args.output_md.write_text(render_report(index), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md)}, indent=2))


if __name__ == "__main__":
    main()
