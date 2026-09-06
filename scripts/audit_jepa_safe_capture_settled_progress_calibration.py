"""Audit offline JEPA heads against settled one-step route outcomes.

This script is deliberately read-only with respect to the online controller. It
joins the existing one-step route-regret audits with their source traces and
measures whether each predicted head orders candidates like the settled
counterfactual outcome. A failed calibration gate blocks online score changes,
training, and scene expansion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from torch.utils.tensorboard import SummaryWriter


COST_FEATURES = (
    "ranker_score",
    "target_cost_m",
    "uncertainty_cost_m",
    "clearance_cost_m",
    "ttc_cost",
    "visibility_cost",
    "cbf_risk_cost",
    "candidate_separation_m",
)
PROGRESS_FEATURES = (
    "predicted_route_progress_m",
    "target_escape_progress_m",
)
FEATURES = COST_FEATURES + PROGRESS_FEATURES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"Trace is empty: {path}")
    return rows


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _rankdata(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.shape[0], dtype=np.float64)
    start = 0
    while start < order.size:
        stop = start + 1
        while stop < order.size and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    x_array = np.asarray(list(x), dtype=np.float64)
    y_array = np.asarray(list(y), dtype=np.float64)
    if x_array.size < 2 or x_array.size != y_array.size:
        return None
    x_rank = _rankdata(x_array)
    y_rank = _rankdata(y_array)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator <= 1e-12:
        return None
    return float(np.dot(x_centered, y_centered) / denominator)


def _top1_agreement(
    rows: list[dict[str, Any]],
    feature: str,
    *,
    minimize: bool,
    eligible_only: bool = False,
) -> float | None:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["episode_seed"]), int(row["step"])), []).append(row)
    agreements: list[bool] = []
    for candidates in grouped.values():
        usable = [item for item in candidates if item.get(feature) is not None]
        predicted_pool = [item for item in usable if not eligible_only or bool(item.get("online_eligible"))]
        if not usable or not predicted_pool:
            continue
        predicted = (min if minimize else max)(
            predicted_pool, key=lambda item: (float(item[feature]), int(item["candidate_index"]))
        )
        settled = min(
            usable, key=lambda item: (float(item["settled_after_distance_m"]), int(item["candidate_index"]))
        )
        agreements.append(int(predicted["candidate_index"]) == int(settled["candidate_index"]))
    return float(np.mean(agreements)) if agreements else None


def _feature_metric(rows: list[dict[str, Any]], feature: str, *, minimize: bool) -> dict[str, Any]:
    usable = [row for row in rows if row.get(feature) is not None]
    eligible_only = feature == "ranker_score"
    metric: dict[str, Any] = {
        "feature": feature,
        "direction": "cost_minimize" if minimize else "progress_maximize",
        "expected_after_distance_correlation": "positive" if minimize else "negative",
        "candidate_rows": len(usable),
        "step_top1_agreement": _top1_agreement(usable, feature, minimize=minimize, eligible_only=eligible_only),
        "spearman_vs_settled_after_distance_m": _spearman(
            [float(row[feature]) for row in usable],
            [float(row["settled_after_distance_m"]) for row in usable],
        ),
        "spearman_vs_settled_progress_m": _spearman(
            [float(row[feature]) for row in usable],
            [float(row["settled_progress_m"]) for row in usable],
        ),
        "episodes": {},
    }
    for episode_seed in sorted({int(row["episode_seed"]) for row in usable}):
        episode_rows = [row for row in usable if int(row["episode_seed"]) == episode_seed]
        metric["episodes"][str(episode_seed)] = {
            "candidate_rows": len(episode_rows),
            "step_top1_agreement": _top1_agreement(episode_rows, feature, minimize=minimize, eligible_only=eligible_only),
            "spearman_vs_settled_after_distance_m": _spearman(
                [float(row[feature]) for row in episode_rows],
                [float(row["settled_after_distance_m"]) for row in episode_rows],
            ),
            "spearman_vs_settled_progress_m": _spearman(
                [float(row[feature]) for row in episode_rows],
                [float(row["settled_progress_m"]) for row in episode_rows],
            ),
        }
    return metric


def _collect_rows(audit_paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audits = [_read_json(path) for path in audit_paths]
    if not audits:
        raise ValueError("At least one route-regret audit is required")
    if any(
        audit.get("audit_type") != "jepa_safe_capture_one_step_settled_route_regret"
        or audit.get("development_only") is not True
        or audit.get("locked_test_opened") is not False
        or audit.get("online_contract_modified") is not False
        for audit in audits
    ):
        raise ValueError("All inputs must be development-only and online-contract unchanged")
    manifests = {str(audit.get("source_manifest_sha256")) for audit in audits}
    runs = {str(audit.get("source_run")) for audit in audits}
    if len(manifests) != 1 or len(runs) != 1:
        raise ValueError("Audits must share one source run and scene manifest")
    for audit, audit_path in zip(audits, audit_paths):
        trace_path = Path(str(audit["source_run"])) / "step_traces" / f"episode_{int(audit['episode_index']):04d}.jsonl"
        if not trace_path.is_file():
            raise FileNotFoundError(trace_path)
        traces = {int(row.get("step", -1)): row for row in _read_jsonl(trace_path)}
        for step in audit.get("steps", []):
            if not bool(step.get("online_trusted")) or step.get("settled_best_index") is None:
                continue
            trace = traces.get(int(step["step"]))
            if trace is None:
                raise ValueError(f"Missing source trace step {step['step']} in {trace_path}")
            ranking = trace.get("candidate_ranking") or {}
            labels = list(ranking.get("candidate_labels") or [])
            arrays = {feature: list(ranking.get(feature) or []) for feature in FEATURES}
            arrays["ranker_score"] = list(ranking.get("scores") or [])
            for candidate in step.get("candidates", []):
                if not isinstance(candidate, dict) or not bool(candidate.get("physical_safe")):
                    continue
                candidate_index = int(candidate["candidate_index"])
                row: dict[str, Any] = {
                    "episode_seed": int(audit["episode_seed"]),
                    "step": int(step["step"]),
                    "candidate_index": candidate_index,
                    "candidate_label": labels[candidate_index] if candidate_index < len(labels) else str(candidate.get("label", "unknown")),
                    "settled_after_distance_m": _finite(candidate.get("after_distance_m")),
                    "settled_progress_m": _finite(candidate.get("distance_improvement_m")),
                    "online_eligible": bool(candidate.get("ranker_eligible", False)),
                }
                if row["settled_after_distance_m"] is None or row["settled_progress_m"] is None:
                    continue
                for feature, values in arrays.items():
                    row[feature] = _finite(values[candidate_index]) if candidate_index < len(values) else None
                rows.append(row)
    metadata = {
        "source_run": next(iter(runs)),
        "source_manifest_sha256": next(iter(manifests)),
        "audit_json_sha256": [_sha256(path) for path in audit_paths],
        "trusted_settled_candidate_rows": len(rows),
        "trusted_settled_episode_seeds": sorted({int(row["episode_seed"]) for row in rows}),
    }
    return rows, metadata


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    audit_paths = [path.resolve() for path in args.audit_json]
    rows, metadata = _collect_rows(audit_paths)
    if len(metadata["trusted_settled_episode_seeds"]) < 3:
        raise ValueError("Calibration requires at least three independent episode seeds")
    metrics = {
        feature: _feature_metric(rows, feature, minimize=feature in COST_FEATURES)
        for feature in FEATURES
    }
    episode_signs = {
        feature: [
            value["spearman_vs_settled_after_distance_m"]
            for value in metrics[feature]["episodes"].values()
            if value["spearman_vs_settled_after_distance_m"] is not None
        ]
        for feature in FEATURES
    }
    stable_features = []
    for feature, signs in episode_signs.items():
        if len(signs) != len(metadata["trusted_settled_episode_seeds"]):
            continue
        expected_sign = 1.0 if feature in COST_FEATURES else -1.0
        if all(expected_sign * sign >= 0.0 for sign in signs):
            stable_features.append(feature)
    ranker_metric = metrics["ranker_score"]
    ranker_episode_top1 = {
        seed: value["step_top1_agreement"]
        for seed, value in ranker_metric["episodes"].items()
    }
    ranker_top1_threshold = 0.50
    ranker_gate_passed = bool(
        ranker_metric["step_top1_agreement"] is not None
        and float(ranker_metric["step_top1_agreement"]) >= ranker_top1_threshold
        and all(
            value is not None and float(value) >= ranker_top1_threshold
            for value in ranker_episode_top1.values()
        )
        and all(sign >= 0.0 for sign in episode_signs["ranker_score"])
    )
    result: dict[str, Any] = {
        "audit_type": "jepa_safe_capture_settled_progress_calibration",
        "development_only": True,
        "locked_test_opened": False,
        "online_contract_modified": False,
        "settled_label_scope": "one_step_counterfactual_only",
        **metadata,
        "features": metrics,
        "calibration_gate": {
            "stable_nonnegative_cost_or_progress_features": stable_features,
            "ranker_top1_threshold": ranker_top1_threshold,
            "ranker_top1_observed": ranker_metric["step_top1_agreement"],
            "ranker_episode_top1": ranker_episode_top1,
            "ranker_gate_passed": ranker_gate_passed,
            "stable_signal": ranker_gate_passed,
            "training_authorized": False,
            "scene_expansion_authorized": False,
            "online_score_change_authorized": False,
            "reason": (
                "Ranker top-1 agreement and per-episode direction passed the calibration gate."
                if ranker_gate_passed
                else "Existing ranker top-1 agreement is below 0.50 overall or in at least one episode."
            ),
        },
        "interpretation": {
            "target_truth_used": "offline_branch_labels_only",
            "online_controller_replayed": False,
            "next_gate": "require_new_calibration_archive_and_hash_bound_ledger_before_online_change",
        },
    }
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    if args.tensorboard_dir.exists() and any(args.tensorboard_dir.iterdir()):
        raise FileExistsError(args.tensorboard_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "calibration.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Settled Progress Calibration Audit",
        "",
        "Development-only offline calibration of existing JEPA heads against one-step settled CBF counterfactuals.",
        "",
        f"- Trusted settled candidate rows: `{len(rows)}`",
        f"- Episode seeds: `{', '.join(str(value) for value in metadata['trusted_settled_episode_seeds'])}`",
        f"- Stable nonnegative features: `{', '.join(stable_features) if stable_features else 'none'}`",
        f"- Existing ranker top-1 gate: `{ranker_metric['step_top1_agreement']:.4f}` (threshold `{ranker_top1_threshold:.2f}`)",
        f"- Calibration gate passed: `{str(ranker_gate_passed).lower()}`",
        "",
        "| Feature | Direction | Candidate rows | Top-1 agreement | Spearman vs settled after-distance |",
        "|---|---|---:|---:|---:|",
    ]
    for feature in FEATURES:
        item = metrics[feature]
        top1 = "n/a" if item["step_top1_agreement"] is None else f"{item['step_top1_agreement']:.4f}"
        rho = "n/a" if item["spearman_vs_settled_after_distance_m"] is None else f"{item['spearman_vs_settled_after_distance_m']:.4f}"
        lines.append(f"| `{feature}` | `{item['direction']}` | {item['candidate_rows']} | {top1} | {rho} |")
    lines.extend([
        "",
        "This audit does not authorize online score changes, CBF margin changes, retraining, scene expansion, or a locked test.",
        "A stable feature is only a prerequisite for a fresh calibration archive and hash-bound Ledger; it is not a safe-capture improvement claim.",
    ])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with SummaryWriter(log_dir=str(args.tensorboard_dir), flush_secs=1) as writer:
        writer.add_text("Config/interpretation", json.dumps(result["interpretation"], indent=2), 0)
        writer.add_text("Provenance/source", json.dumps(metadata, indent=2), 0)
        writer.add_text("Gates/calibration", json.dumps(result["calibration_gate"], indent=2), 0)
        writer.add_scalar("Gates/stable_signal", float(result["calibration_gate"]["stable_signal"]), 0)
        writer.add_scalar("Gates/online_score_change_authorized", 0.0, 0)
        writer.add_scalar("Gates/training_authorized", 0.0, 0)
        writer.add_scalar("Aggregate/trusted_settled_candidate_rows", float(len(rows)), 0)
        for feature, item in metrics.items():
            prefix = f"Features/{feature}"
            if item["step_top1_agreement"] is not None:
                writer.add_scalar(f"{prefix}/step_top1_agreement", float(item["step_top1_agreement"]), 0)
            if item["spearman_vs_settled_after_distance_m"] is not None:
                writer.add_scalar(f"{prefix}/spearman_vs_settled_after_distance_m", float(item["spearman_vs_settled_after_distance_m"]), 0)
            for episode_seed, episode in item["episodes"].items():
                if episode["spearman_vs_settled_after_distance_m"] is not None:
                    writer.add_scalar(f"{prefix}/episode_{episode_seed}_spearman", float(episode["spearman_vs_settled_after_distance_m"]), 0)
    result["tensorboard"] = {
        "logdir": str(args.tensorboard_dir.resolve()),
        "event_files": sorted(path.name for path in args.tensorboard_dir.glob("events.out.tfevents.*")),
    }
    (args.output_dir / "calibration.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    args = parser.parse_args()
    if not args.development_only:
        raise ValueError("This calibration requires --development-only.")
    result = calibrate(args)
    print(json.dumps({
        "trusted_settled_candidate_rows": result["trusted_settled_candidate_rows"],
        "stable_nonnegative_cost_or_progress_features": result["calibration_gate"]["stable_nonnegative_cost_or_progress_features"],
        "stable_signal": result["calibration_gate"]["stable_signal"],
        "tensorboard": result["tensorboard"],
    }, indent=2))


if __name__ == "__main__":
    main()
