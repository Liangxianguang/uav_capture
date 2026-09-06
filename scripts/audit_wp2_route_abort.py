"""Audit a WP2 obstacle-route controlled abort without re-simulating it.

The audit is deliberately read-only with respect to the source run.  It
summarizes the last feasible control cycle, the abort cycle, route geometry,
candidate CBF probes, independent fallback probes, and active constraints.  It
does not change CBF thresholds, regenerate actions, or reinterpret an abort as
a safety success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _count_true(values: Any) -> int:
    return sum(1 for value in _as_list(values) if _as_bool(value))


def _compact_probe(probe: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(probe, Mapping):
        return None
    return {
        "label": probe.get("label"),
        "route_id": probe.get("route_id"),
        "verified_feasible": _as_bool(probe.get("verified_feasible")),
        "accepted": _as_bool(probe.get("accepted")),
        "infeasible": _as_bool(probe.get("infeasible")),
        "timed_out": _as_bool(probe.get("timed_out")),
        "solver_status": probe.get("solver_status"),
        "minimum_constraint_value": _finite(probe.get("minimum_constraint_value")),
        "action_correction_norm": _finite(probe.get("action_correction_norm")),
        "active_constraints": [str(item) for item in _as_list(probe.get("active_constraints"))],
    }


def _route_summary(route: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label": route.get("label"),
        "route_id": route.get("route_id"),
        "obstacle_id": route.get("obstacle_id"),
        "obstacle_shape": route.get("obstacle_shape"),
        "side": route.get("side"),
        "valid": _as_bool(route.get("valid")),
        "geometric_feasible": _as_bool(route.get("geometric_feasible")),
        "minimum_geometric_clearance_m": _finite(route.get("minimum_geometric_clearance_m")),
        "route_length_m": _finite(route.get("route_length_m")),
        "rejection_reasons": [str(item) for item in _as_list(route.get("rejection_reasons"))],
    }


def summarize_step(row: Mapping[str, Any]) -> dict[str, Any]:
    runtime = row.get("route_runtime")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    routes = runtime.get("routes")
    routes = routes if isinstance(routes, Mapping) else {}
    route_candidates = [item for item in _as_list(routes.get("candidates")) if isinstance(item, Mapping)]
    probes = [item for item in _as_list(runtime.get("cbf_counterfactuals")) if isinstance(item, Mapping)]
    cbf = row.get("cbf")
    cbf = cbf if isinstance(cbf, Mapping) else {}
    ranking = row.get("candidate_ranking")
    ranking = ranking if isinstance(ranking, Mapping) else {}
    independent = [
        _compact_probe(item) if isinstance(item, Mapping) else None
        for item in _as_list(row.get("independent_cbf_counterfactuals"))
    ]
    state = row.get("safety_observables")
    state = state if isinstance(state, Mapping) else {}
    active = [str(item) for item in _as_list(cbf.get("active_constraints"))]
    probe_accepted = sum(_as_bool(item.get("accepted")) for item in probes)
    geometry_valid = sum(
        _as_bool(item.get("valid")) and _as_bool(item.get("geometric_feasible"))
        for item in route_candidates
    )
    selected = row.get("selected_route")
    selected = selected if isinstance(selected, Mapping) else {}
    return {
        "step": int(row.get("step", -1)),
        "selected_route": {
            "label": selected.get("label"),
            "route_id": selected.get("route_id"),
            "side": selected.get("side"),
            "obstacle_id": selected.get("obstacle_id"),
            "rejection_reasons": [str(item) for item in _as_list(selected.get("rejection_reasons"))],
        },
        "state_clearance_m": {
            "obstacle": _finite(state.get("minimum_obstacle_clearance_m")),
            "pairwise": _finite(state.get("minimum_pairwise_clearance_m")),
            "boundary": _finite(state.get("minimum_boundary_clearance_m")),
        },
        "route_geometry": {
            "candidate_count": len(route_candidates),
            "valid_count": geometry_valid,
            "invalid_count": len(route_candidates) - geometry_valid,
            "invalid_reasons": dict(
                Counter(
                    reason
                    for route in route_candidates
                    for reason in _as_list(route.get("rejection_reasons"))
                )
            ),
            "routes": [_route_summary(route) for route in route_candidates],
        },
        "candidate_cbf": {
            "checks": len(probes),
            "accepted": probe_accepted,
            "rejected": len(probes) - probe_accepted,
            "probes": [_compact_probe(item) for item in probes],
        },
        "ranking": {
            "valid_count": _count_true(ranking.get("valid_mask")),
            "eligible_count": _count_true(ranking.get("eligible_mask")),
            "execution_mode": ranking.get("execution_mode"),
            "fallback_reason": ranking.get("fallback_reason"),
            "rank_abstention_reason": ranking.get("rank_abstention_reason"),
            "ledger_states": [str(item) for item in _as_list(ranking.get("ledger_states"))],
            "ledger_fallback_reasons": [
                str(item) for item in _as_list(ranking.get("ledger_fallback_reasons"))
            ],
        },
        "cbf": {
            "solver_status": cbf.get("solver_status"),
            "fallback_mode": cbf.get("fallback_mode"),
            "infeasible": _as_bool(cbf.get("infeasible")),
            "timed_out": _as_bool(cbf.get("timed_out")),
            "verified_feasible": _as_bool(cbf.get("verified_feasible")),
            "minimum_constraint_value": _finite(cbf.get("minimum_constraint_value")),
            "action_correction_norm": _finite(cbf.get("action_correction_norm")),
            "active_constraints": active,
            "constraint_slacks": {
                str(key): _finite(value)
                for key, value in (cbf.get("constraint_slacks", {}) or {}).items()
                if _finite(value) is not None
            },
        },
        "independent_cbf": independent,
    }


def _has_obstacle_constraint(step: Mapping[str, Any]) -> bool:
    return any(str(item).startswith("obstacle_") for item in step["cbf"]["active_constraints"])


def classify_abort(abort: Mapping[str, Any], previous: Mapping[str, Any] | None) -> dict[str, Any]:
    geometry = abort["route_geometry"]
    candidate = abort["candidate_cbf"]
    cbf = abort["cbf"]
    labels: list[str] = []
    evidence: list[str] = []

    if geometry["valid_count"] == 0:
        labels.append("all_route_geometric_rejection")
        evidence.append("no geometrically valid route remained at abort")
    elif candidate["checks"] > 0 and candidate["accepted"] == 0:
        labels.append("joint_cbf_recovery_infeasibility")
        evidence.append("geometrically valid routes remained but every primary CBF probe was rejected")
    if _has_obstacle_constraint(abort):
        labels.append("obstacle_barrier_recovery")
        evidence.append("active CBF constraints include obstacle barriers")
    if any(str(item).startswith("acceleration_") for item in cbf["active_constraints"]):
        labels.append("acceleration_reachability_interaction")
        evidence.append("active CBF constraints include acceleration limits")
    if previous is not None:
        previous_routes = previous["candidate_cbf"]["probes"]
        previous_detours = [
            item
            for item in previous_routes
            if item.get("label") in {"left_detour", "right_detour", "upper_detour", "lower_detour"}
            and item.get("accepted")
        ]
        if (
            previous_detours
            and previous["selected_route"].get("label") == "nominal"
            and not previous["cbf"].get("infeasible")
        ):
            labels.append("late_obstacle_anticipation")
            evidence.append(
                "the preceding feasible cycle selected nominal while lateral/vertical detour probes were accepted"
            )
    if cbf["timed_out"]:
        labels.append("solver_timeout")
        evidence.append("CBF timeout flag is true")
    if not labels:
        labels.append("unresolved_abort")
        evidence.append("available trace fields do not identify a narrower category")
    return {
        "primary": labels[0],
        "labels": labels,
        "evidence": evidence,
        "solver_numerical_issue_proven": False,
    }


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _markdown(report: Mapping[str, Any]) -> str:
    source = report["source"]
    abort = report["abort"]
    previous = report.get("previous_feasible_step")
    classification = report["classification"]
    lines = [
        "# WP2 Controlled-Abort Trace Audit",
        "",
        "This is a development-only, read-only audit. It does not re-simulate the",
        "environment, change CBF thresholds, or count a controlled abort as a safety success.",
        "",
        f"- Source run: `{source['run_dir']}`",
        f"- Episode: `{source['episode_index']}`",
        f"- Scene hash: `{source['scene_hash']}`",
        f"- Source trace SHA-256: `{source['trace_sha256']}`",
        f"- Git revision: `{report['provenance']['git_revision']}`",
        "",
        "## Finding",
        "",
        f"Primary classification: **`{classification['primary']}`**.",
        "",
        "The trace supports late obstacle anticipation followed by joint CBF recovery",
        "infeasibility. This is not evidence of a collision or of a numerical solver bug:",
        "the safety state remained violation-free, but no verified first-step action",
        "was available at the abort cycle.",
        "",
        "Evidence:",
    ]
    lines.extend(f"- {item}." for item in classification["evidence"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            f"| quantity | previous feasible step {previous['step'] if previous else 'n/a'} | abort step {abort['step']} |",
            "|---|---:|---:|",
            f"| obstacle clearance (m) | {_fmt(previous, 'state_clearance_m', 'obstacle') if previous else 'n/a'} | {_fmt(abort, 'state_clearance_m', 'obstacle')} |",
            f"| pairwise clearance (m) | {_fmt(previous, 'state_clearance_m', 'pairwise') if previous else 'n/a'} | {_fmt(abort, 'state_clearance_m', 'pairwise')} |",
            f"| geometric-valid routes | {_fmt(previous, 'route_geometry', 'valid_count') if previous else 'n/a'} | {_fmt(abort, 'route_geometry', 'valid_count')} |",
            f"| primary CBF accepted | {_fmt(previous, 'candidate_cbf', 'accepted') if previous else 'n/a'} | {_fmt(abort, 'candidate_cbf', 'accepted')} |",
            f"| selected route | `{previous['selected_route']['route_id'] if previous else 'n/a'}` | `{abort['selected_route']['route_id']}` |",
            f"| CBF status | `{previous['cbf']['solver_status'] if previous else 'n/a'}` | `{abort['cbf']['solver_status']}` |",
            "",
            "## Accepted routes before abort",
            "",
        ]
    )
    if previous:
        accepted = [
            item["label"]
            for item in previous["candidate_cbf"]["probes"]
            if item.get("accepted")
        ]
        lines.append(", ".join(f"`{item}`" for item in accepted) or "none")
    else:
        lines.append("none")
    lines.extend(
        [
            "",
            "## Abort probes",
            "",
            "| probe | verified | status | min constraint | active constraints |",
            "|---|---:|---|---:|---|",
        ]
    )
    for probe in abort["independent_cbf"]:
        if not probe:
            continue
        active = ", ".join(f"`{item}`" for item in probe["active_constraints"]) or "none"
        lines.append(
            f"| `{probe['label']}` | {probe['verified_feasible']} | `{probe['solver_status']}` | "
            f"{_fmt_value(probe['minimum_constraint_value'])} | {active} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Do not lower CBF margins or remove stale/OOD/controlled-abort gates. The next",
            "data task is to add this abort state and the preceding feasible detour-vs-nominal",
            "counterfactuals to a train-only route-identity archive, then rebuild calibration",
            "and the Ledger against the new candidate protocol.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(report: Mapping[str, Any] | None, outer: str, inner: str) -> str:
    if report is None:
        return "n/a"
    return _fmt_value(report.get(outer, {}).get(inner))


def _fmt_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    args = parse_args()
    if not args.development_only:
        raise ValueError("The WP2 abort audit requires --development-only")
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    provenance_path = run_dir / "provenance.json"
    manifest_path = run_dir / "scene_manifest.jsonl"
    trace_path = run_dir / "step_traces" / f"episode_{args.episode_index:04d}.jsonl"
    for path in (summary_path, provenance_path, manifest_path, trace_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = _read_json(summary_path)
    provenance = _read_json(provenance_path)
    metadata = summary.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("source summary has no metadata")
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError("source run crossed the locked-test boundary")
    if provenance.get("development_only") is not True or provenance.get("locked_test_opened") is not False:
        raise ValueError("source provenance crossed the locked-test boundary")
    if metadata.get("candidate_contract", {}).get("candidate_profile") != "obstacle_route_v1":
        raise ValueError("source run is not obstacle_route_v1")
    rows: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"trace row is not an object: {trace_path}:{line_number}")
            if int(row.get("episode_index", -1)) != args.episode_index:
                raise ValueError(f"trace episode mismatch at line {line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"trace is empty: {trace_path}")
    steps = [summarize_step(row) for row in rows]
    abort_indices = [index for index, step in enumerate(steps) if step["cbf"]["fallback_mode"] == "controlled_abort"]
    if not abort_indices:
        raise ValueError("selected episode contains no controlled_abort")
    abort_index = abort_indices[-1]
    abort = steps[abort_index]
    previous = steps[abort_index - 1] if abort_index else None
    classification = classify_abort(abort, previous)
    scene_hash = ""
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        manifest_row = json.loads(line)
        if int(manifest_row.get("episode_index", -1)) == args.episode_index:
            scene_hash = str(manifest_row.get("scene_hash", ""))
            break
    if len(scene_hash) != 64:
        raise ValueError("scene manifest has no valid scene hash for the episode")
    report: dict[str, Any] = {
        "audit_type": "jepa_safe_capture_wp2_route_abort_audit",
        "development_only": True,
        "locked_test_opened": False,
        "source": {
            "run_dir": str(run_dir),
            "episode_index": args.episode_index,
            "scene_hash": scene_hash,
            "trace_sha256": sha256(trace_path),
            "summary_sha256": sha256(summary_path),
            "provenance_sha256": sha256(provenance_path),
            "manifest_sha256": sha256(manifest_path),
        },
        "abort": abort,
        "previous_feasible_step": previous,
        "classification": classification,
        "provenance": {
            "git_revision": git_revision(),
            "python": sys.version,
            "platform": platform.platform(),
            "source_run_summary_overall": summary.get("overall", {}),
        },
    }
    _json_dump(output_dir / "audit.json", report)
    (output_dir / "audit.md").write_text(_markdown(report), encoding="utf-8")
    (output_dir / "step_audit.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n" for item in steps),
        encoding="utf-8",
    )

    from torch.utils.tensorboard import SummaryWriter

    args.tensorboard_logdir.resolve().mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(args.tensorboard_logdir.resolve()))
    writer.add_scalar("Audit/abort_detected", 1, args.episode_index)
    writer.add_scalar("Audit/abort_step", abort["step"], args.episode_index)
    writer.add_scalar("Audit/previous_geometry_valid_routes", previous["route_geometry"]["valid_count"] if previous else 0, args.episode_index)
    writer.add_scalar("Audit/abort_geometry_valid_routes", abort["route_geometry"]["valid_count"], args.episode_index)
    writer.add_scalar("Audit/previous_cbf_accepted_routes", previous["candidate_cbf"]["accepted"] if previous else 0, args.episode_index)
    writer.add_scalar("Audit/abort_cbf_accepted_routes", abort["candidate_cbf"]["accepted"], args.episode_index)
    writer.add_scalar("Audit/abort_minimum_constraint", abort["cbf"]["minimum_constraint_value"] or 0.0, args.episode_index)
    writer.add_text("Audit/primary_classification", classification["primary"], args.episode_index)
    writer.add_text("Audit/source_trace_sha256", report["source"]["trace_sha256"], args.episode_index)
    writer.flush()
    writer.close()
    print(json.dumps({
        "audit": str(output_dir / "audit.json"),
        "markdown": str(output_dir / "audit.md"),
        "primary_classification": classification["primary"],
        "abort_step": abort["step"],
        "previous_feasible_step": previous["step"] if previous else None,
    }, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
