"""Diagnose route-utility label scales on an independent calibration archive.

This is a label-only, offline diagnostic.  It measures the normalized utility
components used by the DN-MPC route audit, their ranges and correlations, and
the route-switch coverage.  It executes no actions and does not fit or select
any model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter


RANKING_HORIZON_INDEX = 2
ROUTE_COUNT = 12
TERMS = ("progress", "route_length", "escape", "cbf", "switch")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, metadata_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("split") != "calibration":
        raise ValueError("P44 requires the independent calibration split")
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError("P44 requires a closed development archive")
    required = {
        "labels_route_progress",
        "labels_target_escape_cost",
        "labels_cbf_feasible",
        "route_length_m",
        "route_geometry_valid",
        "route_candidate_index",
        "sample_type",
        "scenario_index",
        "time_index",
        "previous_selected_candidate_index",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"Calibration archive is missing {missing}")
        arrays = {name: np.asarray(archive[name]) for name in required}
    samples = int(arrays["sample_type"].shape[0])
    for name, value in arrays.items():
        if value.shape[0] != samples or not np.isfinite(value).all():
            raise ValueError(f"Non-finite or inconsistent array: {name}")
    return arrays, metadata


def _candidate_records(arrays: dict[str, np.ndarray]) -> list[dict[str, float]]:
    runtime = arrays["sample_type"] == 0
    scenarios = arrays["scenario_index"]
    times = arrays["time_index"]
    candidates = arrays["route_candidate_index"]
    records: list[dict[str, float]] = []
    keys = sorted({(int(scenarios[i]), int(times[i])) for i in np.flatnonzero(runtime)})
    for scenario, time_index in keys:
        group = runtime & (scenarios == scenario) & (times == time_index)
        previous_values = arrays["previous_selected_candidate_index"][group]
        if previous_values.size == 0 or not np.all(previous_values == previous_values[0]):
            raise ValueError(f"Previous planner route is not state-consistent at {(scenario, time_index)}")
        previous = int(previous_values[0])
        for candidate in range(ROUTE_COUNT):
            rows = group & (candidates == candidate)
            if not np.any(rows):
                continue
            eligible = bool(
                np.all(arrays["route_geometry_valid"][rows] >= 0.5)
                and np.all(arrays["labels_cbf_feasible"][rows, 0] >= 0.5)
            )
            if not eligible:
                continue
            records.append(
                {
                    "scenario_index": float(scenario),
                    "time_index": float(time_index),
                    "candidate": float(candidate),
                    "progress": float(np.mean(arrays["labels_route_progress"][rows, RANKING_HORIZON_INDEX]) / 0.3),
                    "route_length": float(np.mean(arrays["route_length_m"][rows]) / 10.0),
                    "escape": float(np.mean(arrays["labels_target_escape_cost"][rows, RANKING_HORIZON_INDEX]) / 2.0),
                    "cbf": float(np.mean(arrays["labels_cbf_feasible"][rows, : RANKING_HORIZON_INDEX + 1])),
                    "switch": float(candidate != previous) if previous >= 0 else float("nan"),
                }
            )
    return records


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def _summarize(records: list[dict[str, float]]) -> dict[str, Any]:
    if not records:
        raise ValueError("No eligible runtime candidates were found")
    values = {term: np.asarray([row[term] for row in records], dtype=np.float64) for term in TERMS}
    stats: dict[str, Any] = {}
    for term, value in values.items():
        value = value[np.isfinite(value)]
        if value.size == 0:
            stats[term] = None
            continue
        stats[term] = {
            "count": int(value.size),
            "min": float(value.min()),
            "p05": float(np.quantile(value, 0.05)),
            "median": float(np.median(value)),
            "p95": float(np.quantile(value, 0.95)),
            "max": float(value.max()),
            "mean": float(value.mean()),
            "std": float(value.std()),
        }
    correlations: dict[str, float | None] = {}
    for left_index, left_name in enumerate(TERMS):
        for right_name in TERMS[left_index + 1 :]:
            correlations[f"{left_name}__{right_name}"] = _pearson(values[left_name], values[right_name])
    known_switch = np.isfinite(values["switch"])
    stats["switch_known_fraction"] = float(known_switch.mean())
    known_switch_values = values["switch"][known_switch]
    stats["switch_rate"] = float(known_switch_values.mean()) if bool(known_switch.any()) else None
    return {
        "eligible_candidate_rows": len(records),
        "terms": stats,
        "pearson_correlation": correlations,
    }


def _markdown(report: dict[str, Any], dataset: Path, metadata: Path) -> str:
    lines = [
        "# DN-MPC P44 Utility Label and Scale Diagnosis",
        "",
        "**Phase:** development-only, offline-only",
        "",
        "This label-only diagnostic does not execute actions, train a model, select online weights, or modify CBF gates.",
        "",
        f"- Calibration dataset: `{dataset}`",
        f"- Metadata: `{metadata}`",
        f"- Eligible candidate rows: `{report['eligible_candidate_rows']}`",
        "",
        "## Normalized component distributions",
        "",
        "| Component | Count | Min | P05 | Median | P95 | Max | Mean | Std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for term in TERMS:
        values = report["terms"][term]
        lines.append(
            f"| {term} | {values['count']} | {values['min']:.4f} | {values['p05']:.4f} | "
            f"{values['median']:.4f} | {values['p95']:.4f} | {values['max']:.4f} | "
            f"{values['mean']:.4f} | {values['std']:.4f} |"
        )
    lines += [
        "",
        f"- Planner previous-route known fraction: `{report['terms']['switch_known_fraction']:.2%}`",
        f"- Planner switch rate among audited candidates: `{report['terms']['switch_rate']:.2%}`",
        "",
        "## Pearson correlations",
        "",
        "| Pair | Correlation |",
        "|---|---:|",
    ]
    for pair, value in report["pearson_correlation"].items():
        lines.append(f"| {pair.replace('__', ' vs ')} | {'n/a' if value is None else f'{value:.4f}'} |")
    lines += [
        "",
        "Interpret correlations together with the component ranges. A high correlation between progress and route length indicates possible double counting; a large route-length range relative to progress indicates a unit-normalization risk. This report does not prescribe new weights by itself.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    args = parser.parse_args()
    outputs = [args.output.resolve(), args.markdown_output.resolve()]
    if any(path.exists() for path in outputs) or (args.tensorboard_logdir.exists() and any(args.tensorboard_logdir.iterdir())):
        raise FileExistsError("Refusing to overwrite P44 outputs")
    dataset = args.dataset.resolve()
    metadata_path = args.metadata.resolve()
    arrays, metadata = _load(dataset, metadata_path)
    records = _candidate_records(arrays)
    report = _summarize(records)
    report.update(
        {
            "audit_type": "dn_mpc_p44_utility_label_scale_diagnosis",
            "development_only": True,
            "locked_test_opened": False,
            "dataset": str(dataset),
            "dataset_sha256": _sha256(dataset),
            "metadata": str(metadata_path),
            "metadata_sha256": _sha256(metadata_path),
            "dataset_version": metadata.get("dataset_version"),
            "normalization": {
                "progress": "labels_route_progress[:,2] / 0.3",
                "route_length": "route_length_m / 10.0",
                "escape": "labels_target_escape_cost[:,2] / 2.0",
                "cbf": "mean(labels_cbf_feasible[:,:3])",
                "switch": "candidate != previous_selected_candidate_index",
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_markdown(report, dataset, metadata_path), encoding="utf-8")
    with SummaryWriter(log_dir=str(args.tensorboard_logdir.resolve()), flush_secs=1) as writer:
        writer.add_text("Config/normalization", json.dumps(report["normalization"], sort_keys=True), 0)
        writer.add_text("Provenance/archive", json.dumps({"dataset": str(dataset), "dataset_sha256": report["dataset_sha256"], "metadata_sha256": report["metadata_sha256"]}, sort_keys=True), 0)
        writer.add_scalar("Archive/eligible_candidate_rows", float(report["eligible_candidate_rows"]), 0)
        writer.add_scalar("Planner/previous_route_known_fraction", report["terms"]["switch_known_fraction"], 0)
        writer.add_scalar("Planner/switch_rate", report["terms"]["switch_rate"] or 0.0, 0)
        for term in TERMS:
            for statistic in ("min", "p05", "median", "p95", "max", "mean", "std"):
                writer.add_scalar(f"UtilityScale/{term}/{statistic}", report["terms"][term][statistic], 0)
        for pair, value in report["pearson_correlation"].items():
            writer.add_scalar(f"UtilityCorrelation/{pair}", value or 0.0, 0)
        writer.add_text("Audit/report", _markdown(report, dataset, metadata_path), 0)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
