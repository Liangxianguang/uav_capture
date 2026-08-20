"""Aggregate fixed S1/S2 and randomized S3 Central V4 validation outputs.

The evaluator writes two closely related summary schemas: fixed showcase runs
store metrics directly under ``summary`` while randomized S3 runs store them
under ``overall``. This utility normalizes both without treating episodes as
independent training seeds or claiming a locked-test result.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "safe_capture_rate",
    "cooperative_safe_capture_rate",
    "capture_rate",
    "collision_rate",
    "boundary_violation_rate",
    "transit_success_rate",
    "mean_capture_time_seconds",
    "mean_min_clearance_m",
    "worst_min_clearance_m",
)


def _load_metrics(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    metrics = document.get("summary")
    if isinstance(metrics, dict):
        return metrics
    metrics = document.get("overall")
    if isinstance(metrics, dict):
        return metrics
    raise ValueError(f"{path} has neither a fixed-run summary nor an S3 overall summary.")


def _normalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(metrics)
    if "cooperative_safe_capture_rate" not in normalized:
        cooperative = normalized.get("showcase_success_rate")
        if cooperative is not None:
            normalized["cooperative_safe_capture_rate"] = cooperative
    return normalized


def _find_summary(directory: Path) -> Path:
    for name in ("summary.json", "evaluation.json"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No summary.json or evaluation.json in {directory}")


def _episode_boolean_rate(path: Path, field: str) -> float | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or field not in rows[0]:
        return None
    truthy = {"1", "true", "yes"}
    return sum(str(row[field]).strip().lower() in truthy for row in rows) / len(rows)


def collect(root: Path, entries: dict[str, list[str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, directories in entries.items():
        methods: dict[str, Any] = {}
        for directory in directories:
            path = root / directory
            summary_path = _find_summary(path)
            metrics = _normalize_metrics(_load_metrics(summary_path))
            if "capture_rate" not in metrics:
                capture_rate = _episode_boolean_rate(path / "episodes.csv", "capture_event")
                if capture_rate is not None:
                    metrics["capture_rate"] = capture_rate
            methods[directory] = {
                "summary_path": str(summary_path.relative_to(root)),
                "episodes": metrics.get("episodes"),
                "metrics": {key: metrics.get(key) for key in METRIC_KEYS if key in metrics},
            }
        result[label] = methods
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/central_v4"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--s1", nargs="*", default=[])
    parser.add_argument("--s2", nargs="*", default=[])
    parser.add_argument("--s3", nargs="*", default=[])
    return parser.parse_args()


def render_markdown(aggregate: dict[str, Any]) -> str:
    lines = [
        "# Central V4 D1 Validation Aggregate",
        "",
        "This report combines fixed S1/S2 regression and S3 validation artifacts.",
        "It is validation evidence only: no locked-test or multi-seed claim is made.",
        "",
        "| Group | Artifact | Episodes | Cooperative safe capture | Capture | Collision | Boundary | Transit |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, artifacts in aggregate["groups"].items():
        for artifact, payload in artifacts.items():
            metrics = payload["metrics"]
            def fmt(key: str) -> str:
                value = metrics.get(key)
                if value is None:
                    return "n/a"
                if key == "mean_capture_time_seconds":
                    return f"{float(value):.2f} s"
                if key == "worst_min_clearance_m":
                    return f"{float(value):.3f} m"
                return f"{100.0 * float(value):.1f}%"
            lines.append(
                f"| {group} | `{artifact}` | {payload.get('episodes', 'n/a')} | "
                f"{fmt('cooperative_safe_capture_rate')} | {fmt('capture_rate')} | "
                f"{fmt('collision_rate')} | {fmt('boundary_violation_rate')} | {fmt('transit_success_rate')} |"
            )
    lines.extend(
        [
            "",
            "Raw actor and CBF execution are listed as separate artifacts. CBF improvements are not attributed to the policy network.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = args.results_root.resolve()
    aggregate = {
        "evaluation_type": "central_v4_d1_validation_aggregate",
        "not_a_locked_test": True,
        "statistical_unit": "evaluation_artifact",
        "groups": collect(root, {"S1": args.s1, "S2": args.s2, "S3": args.s3}),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(aggregate), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
