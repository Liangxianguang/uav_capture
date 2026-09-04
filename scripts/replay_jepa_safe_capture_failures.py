"""Deterministically replay auditable records from a frozen V3 failure index.

This is a read-only trace audit, not an environment re-simulation.  It never
regenerates model predictions or settles new trajectories.  Instead it proves
the source identity and safety boundary, derives a compact per-step evidence
record from the original trace, and writes it twice with canonical JSON.  A
hash mismatch between the two copies is a hard failure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from aggregate_jepa_safe_capture_v2_paired import canonical_scene_manifest_sha256, sha256  # noqa: E402
from index_jepa_safe_capture_failures import read_trace  # noqa: E402


REPLAY_TYPE = "jepa_safe_capture_v3_wp1_deterministic_failure_replay"
INDEX_TYPE = "jepa_safe_capture_v3_wp1_failure_index"
CATEGORY_ORDER = (
    "candidate_capture_regression",
    "high_credit_failure",
    "fallback_nominal",
    "candidate_oscillation",
    "stale_or_noisy",
    "timeout",
)
ACTION_FIELDS = (
    "desired_action",
    "reachable_nominal_action",
    "requested_action",
    "executed_action",
)


@dataclass(frozen=True)
class ValidatedRun:
    key: tuple[int, str]
    path: Path
    manifest_path: Path
    manifest_rows: dict[int, dict[str, Any]]
    source_hashes: dict[str, str]
    canonical_manifest_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--development-only", action="store_true", required=True)
    return parser.parse_args()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid integer {label}: {value!r}") from error


def _as_finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Non-numeric {label}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}: {value!r}")
    return result


def _json_finite(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _json_finite(nested, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _json_finite(nested, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite JSON value at {label}")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    _json_finite(value, "replay")
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"JSONL file is empty: {path}")
    return rows


def _hash_or_raise(path: Path, expected: Any, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual = sha256(path)
    if actual != str(expected):
        raise ValueError(f"{label} SHA-256 mismatch for {path}: expected={expected}, actual={actual}")
    return actual


def _failure_index_json_path(path: Path) -> Path:
    if path.suffix.lower() == ".json":
        return path
    if path.suffix.lower() == ".csv":
        candidate = path.with_suffix(".json")
        if not candidate.is_file():
            raise FileNotFoundError(f"CSV replay requires its sibling failure_index.json: {candidate}")
        return candidate
    raise ValueError(f"Failure index must be JSON or CSV: {path}")


def _row_key(row: Mapping[str, Any]) -> tuple[int, str, int]:
    return (
        _as_int(row.get("training_seed"), "training_seed"),
        str(row.get("variant", "")),
        _as_int(row.get("episode_index"), "episode_index"),
    )


def _parse_labels(row: Mapping[str, Any]) -> list[str]:
    values = row.get("diagnostic_labels")
    if isinstance(values, list):
        return [str(value) for value in values]
    encoded = row.get("diagnostic_labels_json", "[]")
    try:
        parsed = json.loads(str(encoded))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid diagnostic_labels_json for {_row_key(row)}") from error
    if not isinstance(parsed, list):
        raise ValueError(f"diagnostic_labels must be a list for {_row_key(row)}")
    return [str(value) for value in parsed]


def _validate_csv_against_index(csv_path: Path, report: Mapping[str, Any]) -> None:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    json_rows = report.get("rows")
    if not isinstance(json_rows, list):
        raise ValueError("Failure-index JSON has no rows list")
    csv_by_key = {_row_key(row): row for row in csv_rows}
    json_by_key = {_row_key(row): row for row in json_rows if isinstance(row, Mapping)}
    if len(csv_by_key) != len(csv_rows) or len(json_by_key) != len(json_rows):
        raise ValueError("Failure-index has duplicate episode identities")
    if set(csv_by_key) != set(json_by_key):
        raise ValueError("Failure-index CSV and JSON episode identities differ")
    fields = (
        "episode_seed",
        "safe_capture",
        "termination_reason",
        "defender_boundary_violation",
        "target_boundary_violation",
    )
    for key in sorted(json_by_key):
        for field in fields:
            expected = json_by_key[key].get(field)
            actual = csv_by_key[key].get(field)
            if field.endswith("_violation") or field == "safe_capture":
                matches = _as_bool(actual) == _as_bool(expected)
            else:
                matches = str(actual) == str(expected)
            if not matches:
                raise ValueError(f"Failure-index CSV mismatch for {key} field {field}")


def load_failure_index(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Load a V3 index and verify its development boundary and CSV consistency."""

    path = path.resolve()
    json_path = _failure_index_json_path(path)
    report = _read_json(json_path)
    if report.get("index_type") != INDEX_TYPE:
        raise ValueError(f"Unexpected failure-index type: {report.get('index_type')!r}")
    if report.get("input_format") != "v3":
        raise ValueError("WP-B2 requires a V3 failure index")
    if report.get("development_only") is not True or report.get("locked_test_opened") is not False:
        raise ValueError("Failure index crossed the locked-test boundary")
    if not isinstance(report.get("runs"), list) or not isinstance(report.get("rows"), list):
        raise ValueError("Failure index must contain runs and rows lists")
    hashes = {"failure_index_json": sha256(json_path)}
    if path.suffix.lower() == ".csv":
        _validate_csv_against_index(path, report)
        hashes["failure_index_csv"] = sha256(path)
    return report, hashes


def _validate_run(record: Mapping[str, Any]) -> ValidatedRun:
    seed = _as_int(record.get("training_seed"), "run.training_seed")
    variant = str(record.get("variant", ""))
    if not variant:
        raise ValueError("Failure-index run has no variant")
    path = Path(str(record.get("path", ""))).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Source run does not exist: {path}")
    summary_path = path / "summary.json"
    provenance_path = path / "provenance.json"
    manifest_path = path / "scene_manifest.jsonl"
    source_hashes = {
        "summary_sha256": _hash_or_raise(summary_path, record.get("summary_sha256"), "source summary"),
        "provenance_sha256": _hash_or_raise(provenance_path, record.get("provenance_sha256"), "source provenance"),
        "manifest_sha256": _hash_or_raise(manifest_path, record.get("manifest_sha256"), "source scene manifest"),
    }
    summary = _read_json(summary_path)
    provenance = _read_json(provenance_path)
    metadata = summary.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Source summary has no metadata: {summary_path}")
    for name, source in (("source summary", metadata), ("source provenance", provenance)):
        if source.get("development_only") is not True or source.get("locked_test_opened") is not False:
            raise ValueError(f"{name} crossed the locked-test boundary: {path}")
    if _as_int(metadata.get("training_seed"), "summary training_seed") != seed:
        raise ValueError(f"Source summary seed mismatch: {path}")
    variant_metadata = metadata.get("variant")
    if not isinstance(variant_metadata, Mapping) or str(variant_metadata.get("variant")) != variant:
        raise ValueError(f"Source summary variant mismatch: {path}")
    manifest_rows = _read_jsonl(manifest_path)
    indexed_manifest: dict[int, dict[str, Any]] = {}
    for manifest_row in manifest_rows:
        index = _as_int(manifest_row.get("episode_index"), "scene manifest episode_index")
        if index in indexed_manifest:
            raise ValueError(f"Duplicate episode in source scene manifest: {manifest_path}")
        if _as_int(manifest_row.get("training_seed"), "scene manifest training_seed") != seed:
            raise ValueError(f"Scene manifest training seed mismatch: {manifest_path}")
        scene_hash = str(manifest_row.get("scene_hash", ""))
        if len(scene_hash) != 64:
            raise ValueError(f"Scene manifest has invalid scene_hash: {manifest_path}")
        indexed_manifest[index] = manifest_row
    return ValidatedRun(
        key=(seed, variant),
        path=path,
        manifest_path=manifest_path,
        manifest_rows=indexed_manifest,
        source_hashes=source_hashes,
        canonical_manifest_sha256=canonical_scene_manifest_sha256(manifest_path),
    )


def validate_source_runs(report: Mapping[str, Any]) -> dict[tuple[int, str], ValidatedRun]:
    validated: dict[tuple[int, str], ValidatedRun] = {}
    for raw in report["runs"]:
        if not isinstance(raw, Mapping):
            raise ValueError("Failure-index run record is not an object")
        run = _validate_run(raw)
        if run.key in validated:
            raise ValueError(f"Duplicate source run identity: {run.key}")
        validated[run.key] = run
    return validated


def _category_matches(row: Mapping[str, Any], category: str) -> bool:
    if _as_bool(row.get("safe_capture")):
        return False
    labels = set(_parse_labels(row))
    if category == "candidate_capture_regression":
        return category in labels
    if category == "high_credit_failure":
        return category in labels
    if category == "fallback_nominal":
        return "low_credit_or_nominal_fallback" in labels or "fallback_nominal" in str(row.get("ranking_modes_json", ""))
    if category == "candidate_oscillation":
        return category in labels
    if category == "stale_or_noisy":
        return "stale_observation" in labels or str(row.get("observation_condition")) == "delayed_noisy"
    if category == "timeout":
        return "timeout" in labels or str(row.get("termination_reason")) in {"timeout", "truncated"}
    raise ValueError(f"Unknown replay category: {category}")


def _row_sort_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
    variant_order = {"m3": 0, "a1": 1, "a2": 2, "m0": 3}
    return (
        variant_order.get(str(row.get("variant")), 99),
        _as_int(row.get("training_seed"), "training_seed"),
        _as_int(row.get("episode_index"), "episode_index"),
        str(row.get("termination_reason", "")),
    )


def select_representatives(rows: Sequence[Mapping[str, Any]], *, per_category: int = 3) -> dict[str, Any]:
    """Select traceable failures deterministically, preferring non-duplicate episodes."""

    if per_category < 1:
        raise ValueError("per_category must be positive")
    eligible_rows = [row for row in rows if not _as_bool(row.get("safe_capture"))]
    selected_by_key: dict[tuple[int, str, int], set[str]] = {}
    category_summary: dict[str, dict[str, Any]] = {}
    selected_keys: set[tuple[int, str, int]] = set()
    for category in CATEGORY_ORDER:
        candidates = sorted((row for row in eligible_rows if _category_matches(row, category)), key=_row_sort_key)
        unique = [row for row in candidates if _row_key(row) not in selected_keys]
        chosen = unique[:per_category]
        if len(chosen) < per_category:
            chosen.extend(candidates[: per_category - len(chosen)])
        for row in chosen:
            key = _row_key(row)
            selected_keys.add(key)
            selected_by_key.setdefault(key, set()).add(category)
        category_summary[category] = {
            "target": per_category,
            "available": len(candidates),
            "selected": len(chosen),
            "shortage": max(per_category - len(chosen), 0),
            "selected_episode_ids": ["%s:%s:%04d" % _row_key(row) for row in chosen],
        }
    lookup = {_row_key(row): dict(row) for row in eligible_rows}
    episodes = []
    for key in sorted(selected_by_key, key=lambda value: _row_sort_key(lookup[value])):
        row = lookup[key]
        episodes.append(
            {
                "identifier": "%s:%s:%04d" % key,
                "training_seed": key[0],
                "variant": key[1],
                "episode_index": key[2],
                "episode_seed": _as_int(row.get("episode_seed"), "episode_seed"),
                "primary_cause": str(row.get("primary_cause", "")),
                "termination_reason": str(row.get("termination_reason", "")),
                "categories": [category for category in CATEGORY_ORDER if category in selected_by_key[key]],
            }
        )
    return {
        "selection_policy": "deterministic category order; prefer an episode not already selected; duplicate only when coverage would otherwise be short",
        "per_category_target": per_category,
        "categories": category_summary,
        "selected_episode_count": len(episodes),
        "episodes": episodes,
    }


def _finite_action(value: Any, label: str) -> Any:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"Action array is missing or empty: {label}")
        result = []
        for index, item in enumerate(value):
            if isinstance(item, (list, tuple)):
                result.append(_finite_action(item, f"{label}[{index}]"))
            else:
                result.append(_as_finite_float(item, f"{label}[{index}]"))
        return result
    return _as_finite_float(value, label)


def _finite_number_list(value: Any, label: str) -> list[float | None]:
    if not isinstance(value, list):
        return []
    result: list[float | None] = []
    for index, item in enumerate(value):
        if item is None:
            result.append(None)
        else:
            result.append(_as_finite_float(item, f"{label}[{index}]"))
    return result


def _copy_bool_list(value: Any) -> list[bool]:
    return [_as_bool(item) for item in value] if isinstance(value, list) else []


def _min_slack(cbf: Mapping[str, Any]) -> float | None:
    values: list[float] = []
    direct = cbf.get("minimum_constraint_value")
    if direct is not None:
        values.append(_as_finite_float(direct, "cbf.minimum_constraint_value"))
    slacks = cbf.get("constraint_slacks")
    if isinstance(slacks, Mapping):
        for key, value in slacks.items():
            if value is not None:
                values.append(_as_finite_float(value, f"cbf.constraint_slacks.{key}"))
    return min(values) if values else None


def _candidate_record(ranking: Mapping[str, Any]) -> dict[str, Any]:
    labels = ranking.get("candidate_labels")
    candidate_labels = [str(value) for value in labels] if isinstance(labels, list) else []
    scores = _finite_number_list(ranking.get("scores"), "candidate_ranking.scores")
    finite_scores = [(index, value) for index, value in enumerate(scores) if value is not None]
    ranked = sorted(finite_scores, key=lambda item: (item[1], item[0]))
    top_two_margin = None
    if len(ranked) >= 2:
        top_two_margin = float(ranked[1][1] - ranked[0][1])
    selected_index = ranking.get("selected_index")
    if selected_index is not None:
        selected_index = _as_int(selected_index, "candidate_ranking.selected_index")
    return {
        "labels": candidate_labels,
        "valid_mask": _copy_bool_list(ranking.get("valid_mask")),
        "eligible_mask": _copy_bool_list(ranking.get("eligible_mask")),
        "scores": scores,
        "best_score_index": ranked[0][0] if ranked else None,
        "top_two_score_margin": top_two_margin,
        "selected_index": selected_index,
        "execution_mode": str(ranking.get("execution_mode", "")),
        "fallback_reason": ranking.get("fallback_reason"),
        "ledger_states": [str(value) for value in ranking.get("ledger_states", [])] if isinstance(ranking.get("ledger_states"), list) else [],
        "ledger_credits": _finite_number_list(ranking.get("ledger_credits"), "candidate_ranking.ledger_credits"),
        "ledger_fallback_reasons": [str(value) if value is not None else None for value in ranking.get("ledger_fallback_reasons", [])] if isinstance(ranking.get("ledger_fallback_reasons"), list) else [],
    }


def _reduce_trace(
    trace: Sequence[Mapping[str, Any]],
    row: Mapping[str, Any],
    *,
    identifier: str,
    categories: Sequence[str],
    scene_hash: str,
) -> list[dict[str, Any]]:
    reduced: list[dict[str, Any]] = []
    episode_index = _as_int(row.get("episode_index"), "episode_index")
    for index, source in enumerate(trace):
        if _as_int(source.get("episode_index"), "trace episode_index") != episode_index:
            raise ValueError(f"Trace episode mismatch for {identifier}")
        ranking = source.get("candidate_ranking")
        cbf = source.get("cbf")
        observation = source.get("observation")
        safety = source.get("safety_observables")
        if not all(isinstance(value, Mapping) for value in (ranking, cbf, observation, safety)):
            raise ValueError(f"Missing required trace section for {identifier} step {index + 1}")
        actions = {field: _finite_action(source.get(field), f"{identifier}.{field}") for field in ACTION_FIELDS}
        target_age = _finite_number_list(observation.get("target_observation_age_steps"), "observation.target_observation_age_steps")
        message_age = _finite_number_list(observation.get("message_age_steps"), "observation.message_age_steps")
        target_visible = _copy_bool_list(observation.get("target_visible"))
        active = cbf.get("active_constraints")
        record: dict[str, Any] = {
            "replay_type": REPLAY_TYPE,
            "identifier": identifier,
            "training_seed": _as_int(row.get("training_seed"), "training_seed"),
            "variant": str(row.get("variant")),
            "episode_index": episode_index,
            "episode_seed": _as_int(row.get("episode_seed"), "episode_seed"),
            "scene_hash": scene_hash,
            "step": _as_int(source.get("step", index + 1), "trace step"),
            "observation": {
                "target_visible": target_visible,
                "target_observation_age_steps": target_age,
                "message_age_steps": message_age,
            },
            "safety_observables": {
                "minimum_obstacle_clearance_m": _as_finite_float(safety.get("minimum_obstacle_clearance_m"), "minimum_obstacle_clearance_m"),
                "minimum_pairwise_clearance_m": _as_finite_float(safety.get("minimum_pairwise_clearance_m"), "minimum_pairwise_clearance_m"),
                "minimum_boundary_clearance_m": _as_finite_float(safety.get("minimum_boundary_clearance_m"), "minimum_boundary_clearance_m"),
            },
            "candidate_ranking": _candidate_record(ranking),
            "actions": actions,
            "cbf": {
                "solver_status": str(cbf.get("solver_status", "")),
                "verified_feasible": _as_bool(cbf.get("verified_feasible", False)),
                "infeasible": _as_bool(cbf.get("infeasible", False)),
                "timed_out": _as_bool(cbf.get("timed_out", False)),
                "unverified": _as_bool(cbf.get("unverified", False)) or not _as_bool(cbf.get("verified_feasible", False)),
                "fallback_mode": str(cbf.get("fallback_mode", "")),
                "used_fallback": _as_bool(cbf.get("used_fallback", False)),
                "active_constraints": [str(value) for value in active] if isinstance(active, list) else [],
                "minimum_slack": _min_slack(cbf),
                "action_correction_norm": _as_finite_float(cbf.get("action_correction_norm"), "cbf.action_correction_norm"),
                "solve_latency_ms": _as_finite_float(cbf.get("solve_latency_ms"), "cbf.solve_latency_ms"),
            },
        }
        if index == len(trace) - 1:
            record["termination"] = {
                "is_final_step": True,
                "safe_capture": _as_bool(row.get("safe_capture")),
                "termination_reason": str(row.get("termination_reason", "")),
                "primary_cause": str(row.get("primary_cause", "")),
                "selected_categories": list(categories),
                # Do not conflate target diagnostic termination with defender safety.
                "defender_boundary_violation": _as_bool(row.get("defender_boundary_violation", row.get("boundary_violation", False))),
                "target_boundary_violation": _as_bool(row.get("target_boundary_violation", False)),
                "collision": _as_bool(row.get("collision", False)),
                "pairwise_violation": _as_bool(row.get("pairwise_violation", False)),
            }
        reduced.append(record)
    if not reduced:
        raise ValueError(f"Trace is empty for {identifier}")
    return reduced


def _validate_episode_source(row: Mapping[str, Any], run: ValidatedRun) -> tuple[Path, str]:
    episode_index = _as_int(row.get("episode_index"), "episode_index")
    episode_seed = _as_int(row.get("episode_seed"), "episode_seed")
    manifest_row = run.manifest_rows.get(episode_index)
    if manifest_row is None:
        raise ValueError(f"Scene manifest does not contain episode {episode_index}: {run.manifest_path}")
    if _as_int(manifest_row.get("episode_seed"), "scene manifest episode_seed") != episode_seed:
        raise ValueError(f"Scene manifest episode_seed mismatch for {(run.key, episode_index)}")
    trace_path = run.path / "step_traces" / f"episode_{episode_index:04d}.jsonl"
    if not trace_path.is_file():
        raise FileNotFoundError(f"Missing source trace: {trace_path}")
    return trace_path, str(manifest_row.get("scene_hash"))


def _write_canonical_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    payload = b"".join(_canonical_bytes(row) for row in rows)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _report_markdown(result: Mapping[str, Any]) -> str:
    selection = result["selection"]
    lines = [
        "# WP-B2 Deterministic Failure Replay",
        "",
        "**Status:** development-only; `locked_test_opened=false`  ",
        "**Method:** read-only canonical derivation from frozen source traces; this is not a new environment rollout.  ",
        f"**Selected episodes:** {selection['selected_episode_count']}  ",
        f"**Repeat count:** {result['repeats']}",
        "",
        "## Category Coverage",
        "",
        "| Category | Available failures | Target | Selected | Shortage |",
        "|---|---:|---:|---:|---:|",
    ]
    for category in CATEGORY_ORDER:
        values = selection["categories"][category]
        lines.append(f"| `{category}` | {values['available']} | {values['target']} | {values['selected']} | {values['shortage']} |")
    lines.extend([
        "",
        "## Replayed Episodes",
        "",
        "| Identifier | Categories | Primary source cause | Termination | Steps | Deterministic |",
        "|---|---|---|---|---:|---:|",
    ])
    for row in result["episodes"]:
        lines.append(
            f"| `{row['identifier']}` | {', '.join('`' + item + '`' for item in row['categories'])} | "
            f"`{row['primary_cause']}` | `{row['termination_reason']}` | {row['trace_steps']} | {str(row['repeat_deterministic']).lower()} |"
        )
    lines.extend([
        "",
        "## Evidence Boundary",
        "",
        "- Source summary/provenance/scene-manifest hashes were rechecked before replay.",
        "- Episode seed, training seed, scene hash, trace episode identity, and finite action arrays were validated.",
        "- The replay records observation age, candidate masks/scores/margin, ledger state/credit, CBF diagnostics, requested/executed action, and source termination.",
        "- No target future ground truth is inferred. A trace-only replay cannot establish an unrecorded causal variable such as future target drift.",
        "- A target boundary diagnostic remains distinct from defender boundary safety failure.",
    ])
    return "\n".join(lines) + "\n"


def _write_tensorboard(result: Mapping[str, Any], logdir: Path) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard logdir: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    selection = result["selection"]
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text(
            "Config/wp1_failure_replay",
            json.dumps({"replay_type": REPLAY_TYPE, "development_only": True, "locked_test_opened": False, "repeats": result["repeats"]}, indent=2),
            0,
        )
        writer.add_text("Provenance/failure_index", json.dumps(result["input_index"], indent=2), 0)
        writer.add_text("Provenance/source_runs", json.dumps(result["source_runs"], indent=2), 0)
        writer.add_text("Selection/policy", str(selection["selection_policy"]), 0)
        writer.add_scalar("Replay/selected_episode_count", float(selection["selected_episode_count"]), 0)
        writer.add_scalar("Replay/repeats", float(result["repeats"]), 0)
        for category, values in selection["categories"].items():
            writer.add_scalar(f"Selection/{category}/available", float(values["available"]), 0)
            writer.add_scalar(f"Selection/{category}/selected", float(values["selected"]), 0)
            writer.add_scalar(f"Selection/{category}/shortage", float(values["shortage"]), 0)
        for index, episode in enumerate(result["episodes"]):
            writer.add_scalar(f"Replay/episode_{index:02d}/trace_steps", float(episode["trace_steps"]), 0)
            writer.add_scalar(f"Replay/episode_{index:02d}/repeat_deterministic", float(episode["repeat_deterministic"]), 0)
            writer.add_scalar(f"Replay/episode_{index:02d}/cbf_unverified_steps", float(episode["cbf_unverified_steps"]), 0)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required_text = {
        "Config/wp1_failure_replay/text_summary",
        "Provenance/failure_index/text_summary",
        "Provenance/source_runs/text_summary",
        "Selection/policy/text_summary",
    }
    missing = sorted(required_text.difference(tags.get("tensors", [])))
    events = sorted(path.name for path in logdir.glob("events.out.tfevents.*"))
    if missing or not events:
        raise ValueError(f"WP-B2 TensorBoard audit failed: missing_text={missing}, event_files={events}")
    return {
        "logdir": str(logdir),
        "event_files": events,
        "scalar_tag_count": len(tags.get("scalars", [])),
        "text_tag_count": len(tags.get("tensors", [])),
        "required_provenance": True,
    }


def _hash_manifest(output_dir: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "hash_manifest.json":
            manifest[str(path.relative_to(output_dir)).replace("\\", "/")] = sha256(path)
    return manifest


def replay_failure_index(
    failure_index_path: Path,
    output_dir: Path,
    tensorboard_logdir: Path,
    *,
    repeats: int = 2,
    per_category: int = 3,
) -> dict[str, Any]:
    """Validate frozen inputs, write deterministic replay artifacts, and return the summary."""

    if repeats < 2:
        raise ValueError("WP-B2 requires at least two deterministic replays")
    report, index_hashes = load_failure_index(failure_index_path)
    runs = validate_source_runs(report)
    rows = [row for row in report["rows"] if isinstance(row, Mapping)]
    if len(rows) != len(report["rows"]):
        raise ValueError("Failure-index contains a non-object episode row")
    selection = select_representatives(rows, per_category=per_category)
    rows_by_key = {_row_key(row): row for row in rows}
    episode_payloads: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    source_runs: list[dict[str, Any]] = []
    used_run_keys: set[tuple[int, str]] = set()
    for selected in selection["episodes"]:
        key = (int(selected["training_seed"]), str(selected["variant"]), int(selected["episode_index"]))
        row = rows_by_key[key]
        run_key = (key[0], key[1])
        if run_key not in runs:
            raise ValueError(f"Failure-index episode has no validated source run: {key}")
        run = runs[run_key]
        trace_path, scene_hash = _validate_episode_source(row, run)
        trace = read_trace(trace_path, key[2])
        payload = _reduce_trace(
            trace,
            row,
            identifier=str(selected["identifier"]),
            categories=selected["categories"],
            scene_hash=scene_hash,
        )
        episode_payloads.append((selected, payload))
        used_run_keys.add(run_key)
    for run_key in sorted(used_run_keys):
        run = runs[run_key]
        source_runs.append(
            {
                "training_seed": run_key[0],
                "variant": run_key[1],
                "path": str(run.path),
                **run.source_hashes,
                "canonical_manifest_sha256": run.canonical_manifest_sha256,
            }
        )
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty replay output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_results: list[dict[str, Any]] = []
    for selected, payload in episode_payloads:
        replay_dir = output_dir / "replays" / str(selected["identifier"]).replace(":", "_")
        replay_dir.mkdir(parents=True, exist_ok=False)
        hashes = []
        for repeat in range(1, repeats + 1):
            hashes.append(_write_canonical_jsonl(replay_dir / f"replay_{repeat}.jsonl", payload))
        if len(set(hashes)) != 1:
            raise ValueError(f"Non-deterministic replay hash for {selected['identifier']}: {hashes}")
        cbf_unverified = sum(int(record["cbf"]["unverified"]) for record in payload)
        episode_results.append(
            {
                **selected,
                "trace_steps": len(payload),
                "repeat_sha256": hashes,
                "repeat_deterministic": True,
                "cbf_unverified_steps": cbf_unverified,
                "scene_hash": payload[0]["scene_hash"],
            }
        )
    result: dict[str, Any] = {
        "replay_type": REPLAY_TYPE,
        "development_only": True,
        "locked_test_opened": False,
        "repeats": repeats,
        "input_index": {
            "path": str(failure_index_path.resolve()),
            **index_hashes,
            "index_type": report["index_type"],
            "input_format": report["input_format"],
        },
        "selection": selection,
        "source_runs": source_runs,
        "episodes": episode_results,
        "outcome_counts": dict(Counter(item["termination_reason"] for item in episode_results)),
        "provenance": {
            "git_revision": git_revision(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tensorboard": version("tensorboard"),
            "source_hashes": {
                "scripts/replay_jepa_safe_capture_failures.py": sha256(Path(__file__).resolve()),
                "scripts/index_jepa_safe_capture_failures.py": sha256(PROJECT_ROOT / "scripts/index_jepa_safe_capture_failures.py"),
            },
        },
    }
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output_dir / "replay_manifest.json").write_text(json.dumps({key: result[key] for key in ("replay_type", "development_only", "locked_test_opened", "input_index", "source_runs", "provenance")}, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output_dir / "replay_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    with (output_dir / "replay_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["identifier", "training_seed", "variant", "episode_index", "episode_seed", "categories", "primary_cause", "termination_reason", "trace_steps", "repeat_deterministic", "cbf_unverified_steps", "scene_hash", "repeat_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in episode_results:
            writer.writerow({**item, "categories": json.dumps(item["categories"], separators=(",", ":")), "repeat_sha256": json.dumps(item["repeat_sha256"], separators=(",", ":"))})
    (output_dir / "report.md").write_text(_report_markdown(result), encoding="utf-8")
    result["tensorboard"] = _write_tensorboard(result, tensorboard_logdir)
    (output_dir / "replay_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    hashes = _hash_manifest(output_dir)
    (output_dir / "hash_manifest.json").write_text(json.dumps(hashes, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    if not args.development_only:
        raise ValueError("WP-B2 requires --development-only")
    result = replay_failure_index(
        args.failure_index,
        args.output_dir,
        args.tensorboard_logdir,
        repeats=args.repeats,
        per_category=args.per_category,
    )
    print(json.dumps({"selected_episodes": len(result["episodes"]), "outcomes": result["outcome_counts"], "tensorboard": result["tensorboard"]}, indent=2))


if __name__ == "__main__":
    main()
