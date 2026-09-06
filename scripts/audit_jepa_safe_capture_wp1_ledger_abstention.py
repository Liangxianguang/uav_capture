"""Read-only audit of Ledger abstention on a frozen runtime trace.

The audit quantifies whether the Ledger routed a step to ``safe_hold`` while
the route generator and independent CBF probes still exposed an executable,
verified candidate.  It never changes an online decision, disables a gate, or
uses simulator target truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from torch.utils.tensorboard import SummaryWriter


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"Empty trace: {path}")
    return rows


def _count_true(values: Any) -> int:
    return sum(_bool(value) for value in values) if isinstance(values, list) else 0


def _probe_status(probes: Any) -> tuple[bool, int, int]:
    if not isinstance(probes, list):
        return False, 0, 0
    labels = {str(item.get("label")): item for item in probes if isinstance(item, Mapping)}
    required = {"selected", "nominal", "safe_hold"}
    if set(labels) != required:
        return False, len(labels), 0
    accepted = sum(_bool(labels[label].get("accepted")) for label in required)
    return True, len(labels), accepted


def analyze(run_dir: Path, episode_index: int) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    summary = _read_json(run_dir / "summary.json")
    provenance = _read_json(run_dir / "provenance.json")
    trace_path = run_dir / "step_traces" / f"episode_{episode_index:04d}.jsonl"
    rows = _read_trace(trace_path)
    reasons: Counter[str] = Counter()
    execution_modes: Counter[str] = Counter()
    ledger_states: Counter[str] = Counter()
    message_age_values: list[float] = []
    observation_age_values: list[float] = []
    visible_fractions: list[float] = []
    candidate_verified_counts: list[int] = []
    eligible_counts: list[int] = []
    valid_counts: list[int] = []
    actionable_safe_hold_steps = 0
    safe_hold_steps = 0
    independent_probe_complete_steps = 0
    independent_probe_all_accepted_steps = 0
    raw_unverified_steps = 0
    invalid_trace_rows = 0

    for row in rows:
        ranking = row.get("candidate_ranking")
        if not isinstance(ranking, Mapping):
            invalid_trace_rows += 1
            continue
        valid_count = _count_true(ranking.get("valid_mask"))
        eligible_count = _count_true(ranking.get("eligible_mask"))
        prefilter = row.get("candidate_cbf_prefilter")
        verified_count = sum(
            _bool(item.get("accepted"))
            for item in prefilter
            if isinstance(item, Mapping)
        ) if isinstance(prefilter, list) else 0
        valid_counts.append(valid_count)
        eligible_counts.append(eligible_count)
        candidate_verified_counts.append(verified_count)
        mode = str(ranking.get("execution_mode", "missing"))
        execution_modes[mode] += 1
        reason = ranking.get("fallback_reason")
        reasons[str(reason) if reason is not None else "none"] += 1
        states = ranking.get("ledger_states")
        if isinstance(states, list):
            ledger_states.update(str(state) for state in states)
        if mode == "safe_hold":
            safe_hold_steps += 1
            if eligible_count == 0 and verified_count > 0:
                actionable_safe_hold_steps += 1
        complete, _, accepted = _probe_status(row.get("independent_cbf_counterfactuals"))
        independent_probe_complete_steps += int(complete)
        independent_probe_all_accepted_steps += int(complete and accepted == 3)
        raw_unverified_steps += int(_bool(row.get("raw_unverified_executed")))
        input_observation = row.get("input_observation")
        if isinstance(input_observation, Mapping):
            for key, target in (
                ("message_age_steps", message_age_values),
                ("target_observation_age_steps", observation_age_values),
            ):
                values = input_observation.get(key)
                if isinstance(values, list):
                    target.extend(float(value) for value in values if np.isfinite(float(value)))
            visible = input_observation.get("target_visible")
            if isinstance(visible, list) and visible:
                visible_fractions.append(float(np.mean([_bool(value) for value in visible])))

    total_steps = len(rows)
    overall = summary.get("overall", {})
    result: dict[str, Any] = {
        "audit_type": "jepa_safe_capture_wp1_ledger_abstention",
        "development_only": True,
        "locked_test_opened": False,
        "run_dir": str(run_dir),
        "episode_index": int(episode_index),
        "trace_steps": total_steps,
        "metrics": {
            "safe_hold_steps": safe_hold_steps,
            "safe_hold_rate": safe_hold_steps / max(total_steps, 1),
            "actionable_safe_hold_steps": actionable_safe_hold_steps,
            "actionable_safe_hold_rate": actionable_safe_hold_steps / max(total_steps, 1),
            "mean_candidate_verified_count": float(np.mean(candidate_verified_counts)) if candidate_verified_counts else 0.0,
            "mean_valid_candidate_count": float(np.mean(valid_counts)) if valid_counts else 0.0,
            "mean_eligible_candidate_count": float(np.mean(eligible_counts)) if eligible_counts else 0.0,
            "zero_eligible_steps": sum(count == 0 for count in eligible_counts),
            "independent_probe_complete_steps": independent_probe_complete_steps,
            "independent_probe_all_accepted_steps": independent_probe_all_accepted_steps,
            "raw_unverified_steps": raw_unverified_steps,
            "mean_message_age_steps": float(np.mean(message_age_values)) if message_age_values else None,
            "max_message_age_steps": float(np.max(message_age_values)) if message_age_values else None,
            "mean_observation_age_steps": float(np.mean(observation_age_values)) if observation_age_values else None,
            "max_observation_age_steps": float(np.max(observation_age_values)) if observation_age_values else None,
            "mean_visible_fraction": float(np.mean(visible_fractions)) if visible_fractions else None,
        },
        "counts": {
            "fallback_reasons": dict(sorted(reasons.items())),
            "execution_modes": dict(sorted(execution_modes.items())),
            "ledger_states": dict(sorted(ledger_states.items())),
        },
        "safety_summary": {
            "collision_count": int(overall.get("collision_count", 0)),
            "boundary_violation_count": int(overall.get("boundary_violation_count", 0)),
            "pairwise_violation_count": int(overall.get("pairwise_violation_count", 0)),
            "raw_unverified_executed_steps": int(overall.get("raw_unverified_executed_steps", 0)),
        },
        "gates": {
            "trace_rows_valid": invalid_trace_rows == 0,
            "independent_cbf_counterfactuals_complete": independent_probe_complete_steps == total_steps,
            "independent_cbf_all_accepted": independent_probe_all_accepted_steps == total_steps,
            "raw_unverified_zero": raw_unverified_steps == 0,
            "safety_hard_gates_zero": all(
                int(overall.get(field, 0)) == 0
                for field in ("collision_count", "boundary_violation_count", "pairwise_violation_count", "raw_unverified_executed_steps")
            ),
        },
        "provenance": {
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
            "python": platform.python_version(),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "run_provenance_sha256": _sha256(run_dir / "provenance.json"),
            "summary_sha256": _sha256(run_dir / "summary.json"),
            "trace_sha256": _sha256(trace_path),
            "source_git_revision": provenance.get("git_revision"),
        },
    }
    result["gates"]["all_pass"] = bool(all(result["gates"].values()))
    return result


def write_outputs(result: Mapping[str, Any], output_dir: Path, tensorboard_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    if tensorboard_dir.exists() and any(tensorboard_dir.iterdir()):
        raise FileExistsError(tensorboard_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ledger_abstention.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metrics = result["metrics"]
    counts = result["counts"]
    lines = [
        "# WP1 Ledger Abstention Audit",
        "",
        "`development_only=true`; `locked_test_opened=false`. This is a read-only trace audit.",
        "",
        f"- Episode: `{result['episode_index']}`; trace steps: `{result['trace_steps']}`",
        f"- Safe-hold steps: `{metrics['safe_hold_steps']}`; actionable safe-hold steps: `{metrics['actionable_safe_hold_steps']}`",
        f"- Mean valid / eligible / CBF-verified candidates: `{metrics['mean_valid_candidate_count']:.2f}` / `{metrics['mean_eligible_candidate_count']:.2f}` / `{metrics['mean_candidate_verified_count']:.2f}`",
        f"- Independent selected/nominal/safe-hold probes all accepted: `{metrics['independent_probe_all_accepted_steps']}/{result['trace_steps']}`",
        "",
        "## Fallback Reasons",
        "",
        "| Reason | Count |",
        "| --- | ---: |",
    ]
    for reason, count in counts["fallback_reasons"].items():
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "An actionable safe-hold is a step where the Ledger reports safe-hold and zero eligible candidates while at least one route candidate passed the independent CBF prefilter.",
            "This does not authorize bypassing stale/OOD gates. It identifies a candidate for a future cautious state that would still require independent CBF verification and residual checks.",
            "",
            f"All audit gates: **{'PASS' if result['gates']['all_pass'] else 'FAIL'}**",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with SummaryWriter(log_dir=str(tensorboard_dir), flush_secs=1) as writer:
        writer.add_text("Config/provenance", json.dumps(result["provenance"], sort_keys=True), 0)
        writer.add_text("Gates/status", json.dumps(result["gates"], sort_keys=True), 0)
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                writer.add_scalar(f"Ledger/{key}", float(value), 0)
        for group, values in counts.items():
            for key, value in values.items():
                writer.add_scalar(f"Counts/{group}/{key}", float(value), 0)
        writer.flush()
    result_with_tb = dict(result)
    result_with_tb["tensorboard"] = {
        "logdir": str(tensorboard_dir.resolve()),
        "event_files": sorted(path.name for path in tensorboard_dir.glob("events.out.tfevents.*")),
        "required_provenance": True,
    }
    (output_dir / "ledger_abstention.json").write_text(
        json.dumps(result_with_tb, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.episode_index < 0:
        raise ValueError("--episode-index must be non-negative")
    result = analyze(args.run_dir, args.episode_index)
    write_outputs(result, args.output_dir.resolve(), args.tensorboard_dir.resolve())
    print(json.dumps({"gates": result["gates"], "metrics": result["metrics"], "counts": result["counts"]}, indent=2))


if __name__ == "__main__":
    main()
