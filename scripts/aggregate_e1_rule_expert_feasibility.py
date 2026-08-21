"""Aggregate the pre-registered E1 rule-expert feasibility gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.e1_protocol import E1_PROFILES, episode_count, load_e1_protocol  # noqa: E402


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "e1_execution_dynamics_protocol.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def evaluate_profile(rows: list[dict[str, str]], *, expected_episodes: int, profile: str) -> dict[str, Any]:
    if len(rows) != expected_episodes:
        raise ValueError(f"{profile} expected {expected_episodes} episodes, found {len(rows)}.")
    if any(row.get("profile") != profile for row in rows):
        raise ValueError(f"{profile} artifact contains a different execution profile.")
    if any(row.get("execution_mode") != "execution_aware_cbf" for row in rows):
        raise ValueError(f"{profile} feasibility must use execution_aware_cbf.")
    if any(row.get("method") != "dynamic_encirclement" for row in rows):
        raise ValueError(f"{profile} feasibility must use the rule expert.")
    cooperative = _rate(rows, "cooperative_safe_capture")
    collision = _rate(rows, "collision")
    boundary = _rate(rows, "world_violation_steps", predicate=lambda value: int(value) > 0)
    passed = cooperative >= 0.95 and collision <= 0.05 and boundary <= 0.05
    return {
        "episodes": len(rows),
        "cooperative_safe_capture_rate": cooperative,
        "collision_rate": collision,
        "boundary_violation_rate": boundary,
        "gate": {
            "cooperative_safe_capture_at_least": 0.95,
            "collision_at_most": 0.05,
            "boundary_at_most": 0.05,
            "passed": passed,
        },
        "termination_reasons": _counts(rows, "termination_reason"),
        "case_sha256_unique": len({str(row["case_sha256"]) for row in rows}),
    }


def aggregate(runs_root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    expected = episode_count(protocol, "development")
    profiles: dict[str, Any] = {}
    for profile in E1_PROFILES:
        artifact = runs_root / f"development_{profile.lower()}_expert_execution_aware" / "episodes.csv"
        if not artifact.is_file():
            raise FileNotFoundError(f"Missing E1 feasibility artifact: {artifact}")
        with artifact.open(newline="", encoding="utf-8") as handle:
            profiles[profile] = evaluate_profile(list(csv.DictReader(handle)), expected_episodes=expected, profile=profile)
    feasible = all(bool(value["gate"]["passed"]) for value in profiles.values())
    return {
        "experiment": "E1_execution_dynamics_rule_expert_feasibility",
        "split": "development",
        "policy_checkpoint_used": False,
        "profiles": profiles,
        "all_profiles_feasible": feasible,
        "policy_development_authorized": feasible,
        "decision": (
            "all E0-E6 profiles have a rule-expert + E-CBF feasible path; frozen-policy development may proceed"
            if feasible
            else "one or more profiles fail the pre-registered feasibility gate; do not run frozen-policy development or locked-test"
        ),
    }


def report(summary: dict[str, Any]) -> str:
    lines = [
        "# E1 Rule-Expert Feasibility Report",
        "",
        "This is a development-only map feasibility check using rule expert + execution-aware CBF. It is not a learning-policy result.",
        "",
        "| Profile | Episodes | Cooperative Safe Capture | Collision | Boundary | Gate |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for profile, value in summary["profiles"].items():
        gate = value["gate"]
        lines.append(
            f"| {profile} | {value['episodes']} | {value['cooperative_safe_capture_rate']:.1%} | "
            f"{value['collision_rate']:.1%} | {value['boundary_violation_rate']:.1%} | "
            f"{'PASS' if gate['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            str(summary["decision"]),
            "",
            "A failed profile remains an explicit negative result. Any weaker replacement requires a new pre-registration and new development/locked seed blocks.",
        ]
    )
    return "\n".join(lines) + "\n"


def _rate(rows: list[dict[str, str]], field: str, predicate: Any = None) -> float:
    if predicate is None:
        predicate = lambda value: str(value).strip().lower() == "true"
    return sum(bool(predicate(row[field])) for row in rows) / len(rows)


def _counts(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for row in rows:
        key = str(row[field])
        values[key] = values.get(key, 0) + 1
    return dict(sorted(values.items()))


def main() -> None:
    args = parse_args()
    protocol = load_e1_protocol(args.protocol.resolve())
    summary = aggregate(args.runs_root.resolve(), protocol)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.output_md.write_text(report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
