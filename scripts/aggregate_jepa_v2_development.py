"""Aggregate the paired v2 JEPA development evaluations.

This report is intentionally a development report: it consumes the frozen V5
baseline scenes and episode rows, but it never opens or modifies a locked test.
The independent unit for cross-seed summaries is the JEPA training seed; paired
comparisons are computed at the episode level within each seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any) -> float | None:
    if value in (None, "", "null", "None"):
        return None
    return float(value)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metric(rows: list[dict[str, str]], field: str, boolean: bool = False) -> float:
    values = [float(_bool(row[field])) if boolean else float(row[field]) for row in rows]
    return float(np.mean(values))


def _run_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    captures = [_bool(row["safe_capture_success"]) for row in rows]
    collisions = [_bool(row["collision"]) for row in rows]
    boundaries = [int(float(row["world_violation_steps"])) > 0 for row in rows]
    time_values = [float(row["capture_time_seconds"]) for row in rows if row["capture_time_seconds"] not in ("", "None")]
    return {
        "episodes": len(rows),
        "safe_capture_count": int(sum(captures)),
        "safe_capture_rate": float(np.mean(captures)),
        "collision_count": int(sum(collisions)),
        "collision_rate": float(np.mean(collisions)),
        "boundary_count": int(sum(boundaries)),
        "boundary_violation_rate": float(np.mean(boundaries)),
        "timeout_count": int(sum(row["termination_reason"] == "timeout" for row in rows)),
        "safety_failure_count": int(sum(row["termination_reason"] == "safety_failure" for row in rows)),
        "transit_success_rate": _metric(rows, "transit_success", boolean=True),
        "mean_capture_time_seconds": float(np.mean(time_values)) if time_values else None,
        "mean_total_defender_path_length_m": _metric(rows, "total_defender_path_length_m"),
        "mean_min_clearance_m": _metric(rows, "min_clearance_m"),
        "mean_cbf_action_correction_norm": _metric(rows, "mean_cbf_action_correction_norm"),
        "max_cbf_action_correction_norm": float(max(float(row["max_cbf_action_correction_norm"]) for row in rows)),
    }


def _paired(method_rows: list[dict[str, str]], baseline_rows: list[dict[str, str]]) -> dict[str, Any]:
    baseline = {int(row["episode_index"]): row for row in baseline_rows}
    method = {int(row["episode_index"]): row for row in method_rows}
    if set(method) != set(baseline):
        raise ValueError("Method and baseline episode indices are not exactly paired.")
    ordered = sorted(method)
    method_capture = [_bool(method[i]["safe_capture_success"]) for i in ordered]
    base_capture = [_bool(baseline[i]["safe_capture_success"]) for i in ordered]
    deltas = [int(a) - int(b) for a, b in zip(method_capture, base_capture)]
    method_times = [_float(method[i]["capture_time_seconds"]) for i in ordered]
    base_times = [_float(baseline[i]["capture_time_seconds"]) for i in ordered]
    paired_times = [float(a) - float(b) for a, b in zip(method_times, base_times) if a is not None and b is not None]
    method_path = [float(method[i]["total_defender_path_length_m"]) for i in ordered]
    base_path = [float(baseline[i]["total_defender_path_length_m"]) for i in ordered]
    return {
        "episodes": len(ordered),
        "capture_improved_count": int(sum(delta > 0 for delta in deltas)),
        "capture_degraded_count": int(sum(delta < 0 for delta in deltas)),
        "capture_tied_count": int(sum(delta == 0 for delta in deltas)),
        "capture_rate_delta_percentage_points": float(100.0 * (np.mean(method_capture) - np.mean(base_capture))),
        "capture_outcome_agreement_rate": float(np.mean([a == b for a, b in zip(method_capture, base_capture)])),
        "capture_time_delta_seconds_on_joint_success": float(np.mean(paired_times)) if paired_times else None,
        "path_delta_m": float(np.mean(np.asarray(method_path) - np.asarray(base_path))),
        "method_only_safety_failures": int(sum(_bool(method[i]["collision"]) or int(float(method[i]["world_violation_steps"])) > 0 for i in ordered)),
        "baseline_only_safety_failures": int(sum(_bool(baseline[i]["collision"]) or int(float(baseline[i]["world_violation_steps"])) > 0 for i in ordered)),
    }


def _summary_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std_sample": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def collect(root: Path) -> dict[str, Any]:
    baseline_path = root / "jepa_v2_control_baseline60" / "episodes.csv"
    baseline_rows = _rows(baseline_path)
    run_dirs = {
        "interaction_aware_seed20260911": root / "jepa_v2_control_seed11_p0005_60",
        "interaction_aware_seed20260912": root / "jepa_v2_control_seed12_p0005_60",
        "interaction_aware_seed20260913": root / "jepa_v2_control_seed13_p0005_60",
        "plain_seed20260911": root / "jepa_v2_control_plain_seed11_p0005_60",
        "plain_seed20260912": root / "jepa_v2_control_plain_seed12_p0005_60",
        "plain_seed20260913": root / "jepa_v2_control_plain_seed13_p0005_60",
    }
    runs: list[dict[str, Any]] = []
    for name, run_dir in run_dirs.items():
        path = run_dir / "episodes.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = _rows(path)
        metrics = _run_metrics(rows)
        paired = _paired(rows, baseline_rows)
        family = "interaction_aware" if name.startswith("interaction") else "plain"
        seed = int(name[-8:])
        runs.append({"name": name, "family": family, "training_seed": seed, "directory": str(run_dir.resolve()), "metrics": metrics, "paired_to_baseline": paired})

    families: dict[str, Any] = {}
    for family in ("interaction_aware", "plain"):
        selected = [run for run in runs if run["family"] == family]
        fields = ("safe_capture_rate", "collision_rate", "boundary_violation_rate", "transit_success_rate", "mean_capture_time_seconds", "mean_total_defender_path_length_m", "mean_min_clearance_m", "mean_cbf_action_correction_norm")
        aggregate: dict[str, Any] = {"training_seeds": [run["training_seed"] for run in selected]}
        for field in fields:
            values = [float(run["metrics"][field]) for run in selected if run["metrics"][field] is not None]
            aggregate[field] = _summary_stats(values)
        for field in ("capture_improved_count", "capture_degraded_count", "capture_tied_count"):
            aggregate[f"paired_{field}"] = {"mean": float(np.mean([run["paired_to_baseline"][field] for run in selected])), "total": int(sum(run["paired_to_baseline"][field] for run in selected))}
        aggregate["paired_capture_rate_delta_percentage_points"] = _summary_stats([run["paired_to_baseline"]["capture_rate_delta_percentage_points"] for run in selected])
        aggregate["paired_path_delta_m"] = _summary_stats([run["paired_to_baseline"]["path_delta_m"] for run in selected])
        families[family] = aggregate

    return {
        "report_type": "jepa_v2_paired_development_aggregate",
        "not_a_locked_test": True,
        "baseline": {"directory": str((root / "jepa_v2_control_baseline60").resolve()), "metrics": _run_metrics(baseline_rows)},
        "runs": runs,
        "families": families,
        "decision": "development_evidence_only_no_locked_test_opened",
    }


def _pct(stat: dict[str, float]) -> str:
    return f"{100.0 * stat['mean']:.2f}% +/- {100.0 * stat['std_sample']:.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    base = report["baseline"]["metrics"]
    lines = [
        "# JEPA v2 Paired Development Aggregate",
        "",
        "> This is development evidence on the frozen V5 development scenes. It is not a V4/V5 locked test and does not change any formal conclusion.",
        "",
        "## Baseline",
        "",
        f"- Frozen V5 baseline: `{base['safe_capture_count']}/{base['episodes']} = {100*base['safe_capture_rate']:.2f}%` safe capture.",
        f"- Collision/boundary: `{100*base['collision_rate']:.2f}%` / `{100*base['boundary_violation_rate']:.2f}%`; transit `{100*base['transit_success_rate']:.2f}%`.",
        "",
        "## Three-seed family summary",
        "",
        "| Family | Safe capture | Collision | Boundary | Mean capture time (s) | Path (m) | Paired capture delta (pp) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, item in report["families"].items():
        lines.append(f"| {family} | {_pct(item['safe_capture_rate'])} | {_pct(item['collision_rate'])} | {_pct(item['boundary_violation_rate'])} | {item['mean_capture_time_seconds']['mean']:.3f} +/- {item['mean_capture_time_seconds']['std_sample']:.3f} | {item['mean_total_defender_path_length_m']['mean']:.3f} +/- {item['mean_total_defender_path_length_m']['std_sample']:.3f} | {item['paired_capture_rate_delta_percentage_points']['mean']:.2f} +/- {item['paired_capture_rate_delta_percentage_points']['std_sample']:.2f} |")
    lines += ["", "## Per-seed paired outcomes", "", "| Run | Safe capture | Collision | Boundary | Improved / degraded / tied vs baseline | Path delta (m) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for run in report["runs"]:
        m, p = run["metrics"], run["paired_to_baseline"]
        lines.append(f"| {run['name']} | {m['safe_capture_count']}/{m['episodes']} ({100*m['safe_capture_rate']:.2f}%) | {100*m['collision_rate']:.2f}% | {100*m['boundary_violation_rate']:.2f}% | {p['capture_improved_count']} / {p['capture_degraded_count']} / {p['capture_tied_count']} | {p['path_delta_m']:.3f} |")
    lines += ["", "## Interpretation", "", "- Interaction-aware JEPA is the primary candidate because it is the only family whose prediction gate passed all four horizons for all three seeds.", "- Capture deltas are episode-paired development evidence, not a claim of a statistically significant or locked improvement.", "- CBF remains enabled in every control run; predictor uncertainty is a ranking feature, not a safety proof.", "- The next required audit is action-following sensitivity: candidate actions must change predicted futures in a way that tracks the corresponding rollout differences.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.results_root.resolve())
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["families"], indent=2))


if __name__ == "__main__":
    main()
