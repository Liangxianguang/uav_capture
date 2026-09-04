"""Aggregate settled ranking diagnostics for the P2 safe-capture audit.

This audit joins the online step trace with the offline local-chunk settled
counterfactual rows.  It is intentionally diagnostic: it does not modify the
policy, re-score candidates, or open a locked split.  In particular, a
positive settled ranking statistic is not treated as a full-episode
safe-capture improvement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_TYPE = "jepa_safe_capture_v5_p2_settled_ranking"
PAIR_LABELS = ("degraded", "improved", "tied")
CREDIT_BUCKETS = ("high", "low_or_missing")
MIN_CREDIT_DEFAULT = 0.65
TIE_TOLERANCE_DEFAULT = 5e-4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _p95(values: Sequence[float]) -> float | None:
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.95)) if values else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.resolve().read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"No JSONL rows: {path}")
    return rows


def _trace_rows(run_dir: Path) -> dict[tuple[int, int], dict[str, Any]]:
    trace_dir = run_dir.resolve() / "step_traces"
    files = sorted(trace_dir.glob("episode_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No step traces: {trace_dir}")
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for path in files:
        for row in _load_jsonl(path):
            episode = int(row.get("episode_index", -1))
            step = int(row.get("step", -1))
            key = (episode, step)
            if key in indexed:
                raise ValueError(f"Duplicate trace key {key} in {run_dir}")
            indexed[key] = row
    return indexed


def _selected_value(row: Mapping[str, Any], name: str, default: Any = None) -> Any:
    value = row.get(name)
    if value is not None:
        return value
    ranking = row.get("candidate_ranking")
    if isinstance(ranking, Mapping):
        return ranking.get(name, default)
    return default


def _deterministic_best(scores: Sequence[Any], eligible: Sequence[Any], tolerance: float) -> int:
    finite = [
        index
        for index in range(min(len(scores), len(eligible)))
        if bool(eligible[index]) and _finite(scores[index]) is not None
    ]
    if not finite:
        return 0
    numeric = np.asarray([float(scores[index]) for index in finite], dtype=np.float64)
    best_score = float(np.min(numeric))
    tied = [index for index in finite if float(scores[index]) <= best_score + tolerance]
    return int(min(tied))


def _top_two_margin(scores: Sequence[Any], eligible: Sequence[Any], tolerance: float) -> float | None:
    del tolerance  # The margin itself is reported before any abstention rule.
    values = sorted(
        float(scores[index])
        for index in range(min(len(scores), len(eligible)))
        if bool(eligible[index]) and _finite(scores[index]) is not None
    )
    if len(values) < 2:
        return None
    return max(values[1] - values[0], 0.0)


def _nominal_displacement(trace: Mapping[str, Any]) -> float | None:
    selected = trace.get("requested_action", trace.get("executed_action"))
    nominal = trace.get("reachable_nominal_action")
    try:
        selected_array = np.asarray(selected, dtype=np.float64)
        nominal_array = np.asarray(nominal, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if selected_array.shape != nominal_array.shape or selected_array.ndim != 2:
        return None
    if not np.isfinite(selected_array).all() or not np.isfinite(nominal_array).all():
        return None
    return float(np.mean(np.linalg.norm(selected_array - nominal_array, axis=1)))


def _selected_credit(row: Mapping[str, Any]) -> float | None:
    selected = int(row.get("selected_index", 0))
    values = row.get("ledger_credits")
    if not isinstance(values, list) or selected < 0 or selected >= len(values):
        return None
    return _finite(values[selected])


def _credit_bucket(credit: float | None, minimum_credit: float) -> str:
    return "high" if credit is not None and credit >= minimum_credit else "low_or_missing"


def _cbf_abort(trace: Mapping[str, Any]) -> tuple[bool, str | None]:
    cbf = trace.get("cbf")
    if not isinstance(cbf, Mapping):
        return False, None
    fallback_mode = str(cbf.get("fallback_mode", ""))
    if fallback_mode == "controlled_abort":
        return True, "controlled_abort"
    if bool(cbf.get("timed_out")) and not bool(cbf.get("verified_feasible")):
        return True, "timeout_unverified"
    if bool(cbf.get("infeasible")) and not bool(cbf.get("verified_feasible")):
        return True, "infeasible_unverified"
    if bool(cbf.get("unverified")) and not bool(cbf.get("verified_feasible")):
        return True, "unverified"
    return False, None


def _validate_pair(trace_run: Path, settled_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    settled = _json(settled_run / "settled_counterfactual.json")
    if settled.get("development_only") is not True or settled.get("locked_test_opened") is not False:
        raise ValueError(f"Settled audit crossed development boundary: {settled_run}")
    if settled.get("all_gates_pass") is not True:
        raise ValueError(f"Settled audit gates did not pass: {settled_run}")
    decision_rows = _load_jsonl(settled_run / "decision_rows.jsonl")
    trace_index = _trace_rows(trace_run)
    inputs = settled.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError(f"Settled audit has no inputs: {settled_run}")
    expected_manifest = str(inputs.get("scene_manifest_sha256", ""))
    if len(expected_manifest) != 64:
        raise ValueError(f"Settled audit has no manifest hash: {settled_run}")
    summary = _json(trace_run / "summary.json")
    metadata = summary.get("metadata")
    overall = summary.get("overall")
    if not isinstance(metadata, Mapping) or not isinstance(overall, Mapping):
        raise ValueError(f"Invalid trace run summary: {trace_run}")
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError(f"Trace run crossed development boundary: {trace_run}")
    trace_manifest = str(metadata.get("inputs", {}).get("scene_manifest_sha256", ""))
    if trace_manifest != expected_manifest:
        raise ValueError(f"Trace/settled scene manifest mismatch: {trace_run}")
    if sha256(trace_run / "scene_manifest.jsonl") != trace_manifest:
        raise ValueError(f"Trace scene manifest hash mismatch: {trace_run}")
    seed = int(metadata.get("training_seed", -1))
    variant = str(metadata.get("variant", {}).get("variant", ""))
    if not decision_rows:
        raise ValueError(f"Empty settled decision rows: {settled_run}")
    pair_keys: set[tuple[int, int]] = set()
    for row in decision_rows:
        key = (int(row.get("episode_index", -1)), int(row.get("step", -1)))
        if key in pair_keys:
            raise ValueError(f"Duplicate settled decision key {key}: {settled_run}")
        pair_keys.add(key)
        if int(row.get("training_seed", seed)) != seed or str(row.get("variant", variant)) != variant:
            raise ValueError(f"Settled row metadata mismatch: {settled_run} {key}")
        if key not in trace_index:
            raise ValueError(f"Settled decision has no matching trace row: {trace_run} {key}")
    if pair_keys != set(trace_index):
        raise ValueError(f"Trace/settled decision keys differ: {trace_run}")
    run_meta = {
        "trace_run": str(trace_run.resolve()),
        "settled_run": str(settled_run.resolve()),
        "training_seed": seed,
        "variant": variant,
        "episodes": int(overall.get("episodes", -1)),
        "control_cycles": len(trace_index),
        "scene_manifest_sha256": trace_manifest,
        "protocol_sha256": str(metadata.get("inputs", {}).get("protocol_sha256", "")),
        "trace_summary_sha256": sha256(trace_run / "summary.json"),
        "settled_summary_sha256": sha256(settled_run / "settled_counterfactual.json"),
        "source_raw_unverified_executed_steps": int(overall.get("raw_unverified_executed_steps", -1)),
        "source_cbf_unverified_steps": int(overall.get("cbf_unverified_steps", -1)),
    }
    return run_meta, decision_rows, trace_index


def _decision_metrics(
    row: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    previous_selected_index: int | None,
    minimum_credit: float,
    tie_tolerance: float,
) -> dict[str, Any]:
    ranking = trace.get("candidate_ranking")
    if not isinstance(ranking, Mapping):
        raise ValueError("Trace row has no candidate_ranking")
    scores = ranking.get("scores")
    eligible = ranking.get("eligible_mask")
    if not isinstance(scores, list) or not isinstance(eligible, list) or len(scores) != 5 or len(eligible) != 5:
        raise ValueError("candidate_ranking must contain five scores and eligibility flags")
    eligible_finite = [bool(eligible[index]) and _finite(scores[index]) is not None for index in range(5)]
    selected_index = int(row.get("selected_index", ranking.get("selected_index", -1)))
    if selected_index < 0 or selected_index >= 5:
        raise ValueError(f"Invalid selected index: {selected_index}")
    predicted_best = _deterministic_best(scores, eligible_finite, tie_tolerance)
    margin = _finite(ranking.get("top_two_margin_m"))
    if margin is None:
        margin = _top_two_margin(scores, eligible_finite, tie_tolerance)
    credit = _selected_credit(row)
    cbf_abort, cbf_abort_reason = _cbf_abort(trace)
    settled_best = int(row.get("best_settled_index", -1))
    if settled_best < 0 or settled_best >= 5:
        raise ValueError(f"Invalid settled best index: {settled_best}")
    return {
        "training_seed": int(row.get("training_seed", -1)),
        "variant": str(row.get("variant", "")),
        "episode_index": int(row.get("episode_index", -1)),
        "step": int(row.get("step", -1)),
        "pair_label": str(row.get("pair_label", "unknown")),
        "selected_index": selected_index,
        "predicted_best_index": predicted_best,
        "selected_matches_predicted_best": bool(selected_index == predicted_best),
        "settled_best_index": settled_best,
        "selected_not_settled_best": bool(row.get("selected_not_best", selected_index != settled_best)),
        "selected_settled_safe_capture": bool(row.get("selected_settled_safe_capture", False)),
        "selected_settled_safety_ok": bool(row.get("selected_settled_safety_ok", False)),
        "selected_settled_progress_m": _finite(row.get("selected_settled_progress_m")),
        "best_settled_progress_m": _finite(row.get("best_settled_progress_m")),
        "top_two_margin_m": margin,
        "nominal_displacement_mps": _nominal_displacement(trace),
        "selected_ledger_credit": credit,
        "credit_bucket": _credit_bucket(credit, minimum_credit),
        "ledger_state": (
            str(row.get("ledger_states")[selected_index])
            if isinstance(row.get("ledger_states"), list) and selected_index < len(row.get("ledger_states"))
            else "missing"
        ),
        "cbf_abort": cbf_abort,
        "cbf_abort_reason": cbf_abort_reason,
        "execution_mode": str(ranking.get("execution_mode", "unknown")),
        "rank_abstention_reason": ranking.get("rank_abstention_reason"),
        "hysteresis_applied": bool(ranking.get("hysteresis_applied", False)),
        "previous_selected_index": previous_selected_index,
        "candidate_switched": bool(previous_selected_index is not None and selected_index != previous_selected_index),
        "eligible_count": int(sum(eligible_finite)),
        "scores": [float(value) if _finite(value) is not None else None for value in scores],
        "eligible_mask": eligible_finite,
    }


def _group_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def values(name: str) -> list[float]:
        return [number for row in rows if (number := _finite(row.get(name))) is not None]

    safety_failures = [not bool(row.get("selected_settled_safety_ok")) for row in rows]
    switches = [bool(row.get("candidate_switched")) for row in rows]
    return {
        "decisions": len(rows),
        "episodes": len({(row.get("training_seed"), row.get("variant"), row.get("episode_index")) for row in rows}),
        "selected_not_settled_best_count": int(sum(bool(row.get("selected_not_settled_best")) for row in rows)),
        "selected_not_settled_best_rate": float(np.mean([bool(row.get("selected_not_settled_best")) for row in rows])) if rows else None,
        "selected_matches_predicted_best_rate": float(np.mean([bool(row.get("selected_matches_predicted_best")) for row in rows])) if rows else None,
        "safety_failure_count": int(sum(safety_failures)),
        "safety_failure_rate": float(np.mean(safety_failures)) if rows else None,
        "safe_capture_rate": float(np.mean([bool(row.get("selected_settled_safe_capture")) for row in rows])) if rows else None,
        "cbf_abort_count": int(sum(bool(row.get("cbf_abort")) for row in rows)),
        "cbf_abort_rate": float(np.mean([bool(row.get("cbf_abort")) for row in rows])) if rows else None,
        "switch_count": int(sum(bool(row.get("candidate_switched")) for row in rows)),
        "switch_rate": float(np.mean(switches)) if rows else None,
        "mean_top_two_margin_m": _mean(values("top_two_margin_m")),
        "p95_top_two_margin_m": _p95(values("top_two_margin_m")),
        "mean_nominal_displacement_mps": _mean(values("nominal_displacement_mps")),
        "p95_nominal_displacement_mps": _p95(values("nominal_displacement_mps")),
        "mean_selected_credit": _mean(values("selected_ledger_credit")),
        "hysteresis_count": int(sum(bool(row.get("hysteresis_applied")) for row in rows)),
        "abstention_count": int(sum(row.get("rank_abstention_reason") is not None for row in rows)),
    }


def _episode_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["training_seed"]), str(row["variant"]), int(row["episode_index"]))].append(row)
    result: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda item: int(item["step"]))
        margins = [number for item in ordered if (number := _finite(item.get("top_two_margin_m"))) is not None]
        nominal = [number for item in ordered if (number := _finite(item.get("nominal_displacement_mps"))) is not None]
        aborts = [item for item in ordered if bool(item.get("cbf_abort"))]
        credits = [number for item in ordered if (number := _finite(item.get("selected_ledger_credit"))) is not None]
        high = [item for item in ordered if item.get("credit_bucket") == "high"]
        low = [item for item in ordered if item.get("credit_bucket") == "low_or_missing"]
        result.append(
            {
                "training_seed": key[0],
                "variant": key[1],
                "episode_index": key[2],
                "decision_count": len(ordered),
                "pair_label": str(ordered[0].get("pair_label", "unknown")),
                "selected_indices": [int(item["selected_index"]) for item in ordered],
                "switch_count": int(sum(bool(item.get("candidate_switched")) for item in ordered)),
                "switch_rate": float(sum(bool(item.get("candidate_switched")) for item in ordered) / max(len(ordered) - 1, 1)),
                "selected_not_settled_best_count": int(sum(bool(item.get("selected_not_settled_best")) for item in ordered)),
                "safety_failure_count": int(sum(not bool(item.get("selected_settled_safety_ok")) for item in ordered)),
                "cbf_abort_count": len(aborts),
                "first_cbf_abort_step": int(aborts[0]["step"]) if aborts else None,
                "cbf_abort_pre_state": {
                    "selected_index": int(aborts[0]["selected_index"]),
                    "predicted_best_index": int(aborts[0]["predicted_best_index"]),
                    "settled_best_index": int(aborts[0]["settled_best_index"]),
                    "top_two_margin_m": _finite(aborts[0].get("top_two_margin_m")),
                    "nominal_displacement_mps": _finite(aborts[0].get("nominal_displacement_mps")),
                    "credit_bucket": str(aborts[0].get("credit_bucket")),
                    "selected_ledger_credit": _finite(aborts[0].get("selected_ledger_credit")),
                    "execution_mode": str(aborts[0].get("execution_mode")),
                    "rank_abstention_reason": aborts[0].get("rank_abstention_reason"),
                    "cbf_abort_reason": aborts[0].get("cbf_abort_reason"),
                }
                if aborts
                else None,
                "high_credit_decisions": len(high),
                "low_or_missing_credit_decisions": len(low),
                "high_credit_safety_failures": sum(not bool(item.get("selected_settled_safety_ok")) for item in high),
                "low_or_missing_safety_failures": sum(not bool(item.get("selected_settled_safety_ok")) for item in low),
                "mean_top_two_margin_m": _mean(margins),
                "mean_nominal_displacement_mps": _mean(nominal),
                "mean_selected_credit": _mean(credits),
            }
        )
    return result


def _credit_stats(rows: Sequence[Mapping[str, Any]], min_decisions: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for bucket in CREDIT_BUCKETS:
        values = [row for row in rows if row.get("credit_bucket") == bucket]
        stats = _group_stats(values)
        stats["coverage_sufficient"] = len(values) >= min_decisions
        result[bucket] = stats
    high = result["high"]
    low = result["low_or_missing"]
    result["both_buckets_present"] = bool(high["decisions"] and low["decisions"])
    result["high_credit_failure_not_above_low_credit"] = (
        bool(high["safety_failure_rate"] <= low["safety_failure_rate"] + 1e-12)
        if result["both_buckets_present"]
        else None
    )
    return result


def _confusion_matrix(rows: Sequence[Mapping[str, Any]], actual_key: str, predicted_key: str) -> list[list[int]]:
    matrix = [[0 for _ in range(5)] for _ in range(5)]
    for row in rows:
        actual = int(row[actual_key])
        predicted = int(row[predicted_key])
        if not 0 <= actual < 5 or not 0 <= predicted < 5:
            raise ValueError(f"Confusion-matrix index outside candidate range: {actual}, {predicted}")
        matrix[actual][predicted] += 1
    return matrix


def _write_tensorboard(logdir: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite TensorBoard directory: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/p2_ranking", json.dumps(report["policy"], indent=2, sort_keys=True), 0)
        writer.add_text("Provenance/inputs", json.dumps(report["inputs"], indent=2, sort_keys=True), 0)
        writer.add_text("Gates/status", json.dumps(report["gates"], indent=2, sort_keys=True), 0)
        writer.add_text("Diagnostics/interpretation", json.dumps(report["interpretation"], indent=2, sort_keys=True), 0)
        writer.add_text("Ranking/confusion_matrices", json.dumps(report["confusion_matrices"], indent=2, sort_keys=True), 0)
        for bucket, stats in report["by_credit_bucket"].items():
            if bucket not in CREDIT_BUCKETS:
                continue
            writer.add_scalar(f"Ranking/Credit/{bucket}/selected_not_settled_best_rate", float(stats["selected_not_settled_best_rate"] or 0.0), 0)
            writer.add_scalar(f"Ranking/Credit/{bucket}/safety_failure_rate", float(stats["safety_failure_rate"] or 0.0), 0)
            writer.add_scalar(f"Ranking/Credit/{bucket}/cbf_abort_rate", float(stats["cbf_abort_rate"] or 0.0), 0)
            writer.add_scalar(f"Ranking/Credit/{bucket}/decisions", float(stats["decisions"]), 0)
        for label, stats in report["by_pair_label"].items():
            writer.add_scalar(f"Ranking/Pair/{label}/selected_not_settled_best_rate", float(stats["selected_not_settled_best_rate"] or 0.0), 0)
            writer.add_scalar(f"Ranking/Pair/{label}/switch_rate", float(stats["switch_rate"] or 0.0), 0)
            writer.add_scalar(f"Ranking/Pair/{label}/safety_failure_rate", float(stats["safety_failure_rate"] or 0.0), 0)
        writer.add_scalar("Audit/decision_count", float(report["decision_count"]), 0)
        writer.add_scalar("Audit/episode_count", float(report["episode_count"]), 0)
        writer.add_scalar("Audit/cbf_abort_count", float(report["cbf_abort_count"]), 0)
        writer.flush()
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required_text = {
        "Config/p2_ranking/text_summary",
        "Provenance/inputs/text_summary",
        "Gates/status/text_summary",
        "Diagnostics/interpretation/text_summary",
        "Ranking/confusion_matrices/text_summary",
    }
    required_scalars = {
        "Audit/decision_count",
        "Audit/episode_count",
        "Ranking/Credit/high/selected_not_settled_best_rate",
        "Ranking/Credit/low_or_missing/selected_not_settled_best_rate",
    }
    events = sorted(path.name for path in logdir.glob("events.out.tfevents.*"))
    missing = sorted(required_text.difference(tags.get("tensors", [])))
    missing.extend(sorted(required_scalars.difference(tags.get("scalars", []))))
    if missing or not events:
        raise ValueError(f"P2 TensorBoard validation failed: missing={missing}, events={events}")
    return {
        "logdir": str(logdir),
        "event_files": events,
        "scalar_tag_count": len(tags.get("scalars", [])),
        "text_tag_count": len(tags.get("tensors", [])),
    }


def audit_runs(
    pairs: Sequence[tuple[Path, Path]],
    *,
    ledger_path: Path | None,
    output_dir: Path,
    tensorboard_logdir: Path,
    min_decisions: int,
    tie_tolerance: float,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    minimum_credit = MIN_CREDIT_DEFAULT
    ledger_sha256 = None
    if ledger_path is not None:
        ledger = _json(ledger_path)
        policy = ledger.get("decision_policy", {})
        if isinstance(policy, Mapping) and _finite(policy.get("minimum_credit")) is not None:
            minimum_credit = float(policy["minimum_credit"])
        ledger_sha256 = sha256(ledger_path.resolve())
    all_rows: list[dict[str, Any]] = []
    run_metadata: list[dict[str, Any]] = []
    for trace_run, settled_run in pairs:
        metadata, settled_rows, trace_index = _validate_pair(trace_run.resolve(), settled_run.resolve())
        previous_by_episode: dict[int, int | None] = defaultdict(lambda: None)
        for row in sorted(settled_rows, key=lambda item: (int(item["episode_index"]), int(item["step"]))):
            episode = int(row["episode_index"])
            key = (episode, int(row["step"]))
            metrics = _decision_metrics(
                row,
                trace_index[key],
                previous_selected_index=previous_by_episode[episode],
                minimum_credit=minimum_credit,
                tie_tolerance=tie_tolerance,
            )
            previous_by_episode[episode] = int(metrics["selected_index"])
            all_rows.append(metrics)
        run_metadata.append(metadata)
    manifests = {str(item["scene_manifest_sha256"]) for item in run_metadata}
    variants = {str(item["variant"]) for item in run_metadata}
    seeds = {int(item["training_seed"]) for item in run_metadata}
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_pair[str(row["pair_label"])].append(row)
    by_credit = _credit_stats(all_rows, min_decisions)
    episodes = _episode_rows(all_rows)
    abort_rows = [row for row in all_rows if bool(row.get("cbf_abort"))]
    confusion = {
        "settled_best_by_predicted_best": _confusion_matrix(all_rows, "settled_best_index", "predicted_best_index"),
        "settled_best_by_selected": _confusion_matrix(all_rows, "settled_best_index", "selected_index"),
        "axis": "rows=settled_best_index, columns=predicted/selected index",
    }
    gates = {
        "development_only": True,
        "locked_test_not_opened": True,
        "runs_have_one_scene_manifest": len(manifests) == 1,
        "runs_have_one_variant": len(variants) == 1,
        "run_seeds_unique": len(seeds) == len(run_metadata),
        "source_raw_unverified_zero": all(int(item["source_raw_unverified_executed_steps"]) == 0 for item in run_metadata),
        "trace_rows_joined": len(all_rows) == sum(int(item["control_cycles"]) for item in run_metadata),
        "scores_finite_for_eligible": all(
            all(_finite(score) is not None for score, eligible in zip(row["scores"], row["eligible_mask"]) if eligible)
            for row in all_rows
        ),
        "selected_indices_valid": all(0 <= int(row["selected_index"]) < 5 for row in all_rows),
        "cbf_abort_state_observable": all("cbf_abort_reason" in row for row in abort_rows),
        "credit_buckets_covered": bool(by_credit["both_buckets_present"]),
    }
    high_low = by_credit["high_credit_failure_not_above_low_credit"]
    if high_low is not None:
        gates["high_credit_failure_gate_evaluable"] = True
    report: dict[str, Any] = {
        "audit_type": AUDIT_TYPE,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "development_only": True,
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "policy": {
            "minimum_credit": minimum_credit,
            "min_bucket_decisions": min_decisions,
            "score_tie_tolerance_m": tie_tolerance,
            "selected_safety_failure_definition": "not selected_settled_safety_ok",
            "selected_not_best_definition": "selected index differs from offline settled best index",
            "nominal_displacement_definition": "mean defender action norm(requested_action - reachable_nominal_action)",
            "cbf_abort_definition": "unverified infeasible/timeout or fallback_mode=controlled_abort in source trace",
            "safe_capture_is_not_inferred_from_local_settlement": True,
        },
        "inputs": {
            "trace_runs": [str(path.resolve()) for path, _ in pairs],
            "settled_runs": [str(path.resolve()) for _, path in pairs],
            "ledger": str(ledger_path.resolve()) if ledger_path else None,
            "ledger_sha256": ledger_sha256,
            "scene_manifest_sha256": next(iter(manifests)) if len(manifests) == 1 else None,
            "protocol_sha256": sorted({str(item["protocol_sha256"]) for item in run_metadata}),
        },
        "run_metadata": run_metadata,
        "decision_count": len(all_rows),
        "episode_count": len(episodes),
        "cbf_abort_count": len(abort_rows),
        "confusion_matrices": confusion,
        "by_credit_bucket": by_credit,
        "by_pair_label": {label: _group_stats(by_pair.get(label, [])) for label in PAIR_LABELS},
        "episodes": episodes,
        "cbf_abort_rows": abort_rows,
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "interpretation": {
            "status": "no_control_gain" if any(bool(row["selected_not_settled_best"]) for row in all_rows) else "ranking_alignment_unresolved",
            "task_metric": "safe_capture",
            "mean_capture_time_is_diagnostic_only": True,
            "high_credit_ordering": high_low,
            "locked_test_opened": False,
            "next_action": "freeze or revise ranking protocol only after reviewing abstention, nominal anchor, and CBF-abort states",
        },
        "provenance": {
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
            "source_hashes": {"scripts/audit_jepa_safe_capture_v5_p2_ranking.py": sha256(Path(__file__).resolve())},
        },
    }
    report["tensorboard"] = _write_tensorboard(tensorboard_logdir, report)
    (output_dir / "p2_ranking_audit.json").write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "p2_ranking_decisions.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    with (output_dir / "p2_ranking_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "training_seed", "variant", "episode_index", "decision_count", "pair_label", "switch_count", "switch_rate",
            "selected_not_settled_best_count", "safety_failure_count", "cbf_abort_count", "first_cbf_abort_step",
            "high_credit_decisions", "low_or_missing_credit_decisions", "high_credit_safety_failures",
            "low_or_missing_safety_failures", "mean_top_two_margin_m", "mean_nominal_displacement_mps", "mean_selected_credit",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in episodes:
            writer.writerow({field: row.get(field) for field in fields})
    lines = [
        "# P2 Settled Ranking Audit",
        "",
        "> Development-only diagnostic. Offline local settlement is not a full-episode policy outcome.",
        "",
        f"All structural gates pass: `{report['all_gates_pass']}`.",
        f"Interpretation: `{report['interpretation']['status']}`.",
        "",
        "| Credit bucket | Decisions | Selected-not-settled-best | Safety failure | CBF abort | Mean margin (m) | Mean nominal displacement (m/s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for bucket in CREDIT_BUCKETS:
        stats = by_credit[bucket]
        lines.append(
            f"| {bucket} | {stats['decisions']} | {stats['selected_not_settled_best_rate']:.3f} | "
            f"{stats['safety_failure_rate']:.3f} | {stats['cbf_abort_rate']:.3f} | "
            f"{stats['mean_top_two_margin_m'] if stats['mean_top_two_margin_m'] is not None else float('nan'):.6f} | "
            f"{stats['mean_nominal_displacement_mps'] if stats['mean_nominal_displacement_mps'] is not None else float('nan'):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Settled-best confusion matrices",
            "",
            "Rows are `settled_best_index`; columns are the online predicted-best or selected index.",
            "",
            "### predicted-best vs settled-best",
            "",
            "| Row \\ Col | 0 | 1 | 2 | 3 | 4 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for index, values in enumerate(confusion["settled_best_by_predicted_best"]):
        lines.append("| settled " + str(index) + " | " + " | ".join(str(value) for value in values) + " |")
    lines.extend(["", "### selected vs settled-best", "", "| Row \\ Col | 0 | 1 | 2 | 3 | 4 |", "|---:|---:|---:|---:|---:|---:|"])
    for index, values in enumerate(confusion["settled_best_by_selected"]):
        lines.append("| settled " + str(index) + " | " + " | ".join(str(value) for value in values) + " |")
    lines.extend(
        [
            "",
            f"CBF-abort pre-state rows: `{len(abort_rows)}`; each row retains selected/predicted-best/settled-best, margin, credit, nominal displacement and abort reason.",
            "",
            "`safe_capture` remains the primary task metric; `mean_capture_time` is diagnostic only; `locked_test_opened=false`.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-run", type=Path, action="append", required=True)
    parser.add_argument("--settled-run", type=Path, action="append", required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    parser.add_argument("--min-bucket-decisions", type=int, default=20)
    parser.add_argument("--score-tie-tolerance-m", type=float, default=TIE_TOLERANCE_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.development_only:
        raise ValueError("P2 ranking audit requires --development-only.")
    if args.min_bucket_decisions <= 0 or args.score_tie_tolerance_m < 0.0:
        raise ValueError("min-bucket-decisions must be positive and score-tie-tolerance-m non-negative.")
    if len(args.trace_run) != len(args.settled_run):
        raise ValueError("Each --trace-run must have one corresponding --settled-run.")
    if args.ledger is not None and not args.ledger.resolve().is_file():
        raise FileNotFoundError(args.ledger)
    report = audit_runs(
        list(zip(args.trace_run, args.settled_run)),
        ledger_path=args.ledger,
        output_dir=args.output_dir.resolve(),
        tensorboard_logdir=args.tensorboard_logdir.resolve(),
        min_decisions=args.min_bucket_decisions,
        tie_tolerance=args.score_tie_tolerance_m,
    )
    print(json.dumps({"all_gates_pass": report["all_gates_pass"], "interpretation": report["interpretation"], "decision_count": report["decision_count"], "episode_count": report["episode_count"], "cbf_abort_count": report["cbf_abort_count"], "by_credit_bucket": report["by_credit_bucket"], "tensorboard": report["tensorboard"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
