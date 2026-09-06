"""Audit causal observability and reacquisition behavior on one frozen trace.

This is an offline, development-only diagnostic.  It reads the controller's
public pre-action observation snapshot and may read the scene's target truth
only as an explicitly offline label.  It never changes runtime decisions and
does not certify safety.
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
        raise ValueError(f"Empty JSONL file: {path}")
    return rows


def _array(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _bools(value: Any, *, name: str, count: int) -> np.ndarray:
    result = np.asarray(value, dtype=bool)
    if result.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},), got {result.shape}")
    return result


def analyze(run_dir: Path, scene_manifest: Path, episode_index: int) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    trace_path = run_dir / "step_traces" / f"episode_{episode_index:04d}.jsonl"
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)
    rows = _read_jsonl(trace_path)
    manifest_rows = _read_jsonl(scene_manifest.resolve())
    matches = [row for row in manifest_rows if int(row.get("episode_index", -1)) == int(episode_index)]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest row for episode {episode_index}, got {len(matches)}")
    scene = matches[0]
    scenario = scene.get("scenario", {})
    target_truth = _array(scenario.get("target_position"), name="offline target_position")

    public_rows: list[dict[str, Any]] = []
    positions: list[np.ndarray] = []
    beliefs: list[np.ndarray] = []
    belief_velocities: list[np.ndarray] = []
    never_received: list[bool] = []
    visible_fractions: list[float] = []
    received_fractions: list[float] = []
    reacquisition_indices: list[int] = []
    reacquisition_action_norms: list[float] = []
    reacquisition_route_labels: list[str] = []
    for index, row in enumerate(rows):
        public = row.get("public_observation")
        if not isinstance(public, Mapping):
            raise ValueError(f"Trace row {index} has no public_observation snapshot")
        defender_positions = _array(public.get("defender_positions"), name="defender_positions")
        belief_positions = _array(public.get("target_belief_positions"), name="target_belief_positions")
        belief_velocity = _array(public.get("target_belief_velocities"), name="target_belief_velocities")
        if defender_positions.ndim != 2 or defender_positions.shape[1] != 3:
            raise ValueError("defender_positions must have shape [defenders, 3]")
        count = int(defender_positions.shape[0])
        if belief_positions.shape != defender_positions.shape or belief_velocity.shape != defender_positions.shape:
            raise ValueError("belief arrays must match defender_positions")
        received = _bools(public.get("target_observation_received"), name="target_observation_received", count=count)
        visible = _bools(public.get("target_visible"), name="target_visible", count=count)
        public_rows.append(dict(public))
        positions.append(defender_positions)
        beliefs.append(belief_positions)
        belief_velocities.append(belief_velocity)
        never_received.append(not bool(np.any(received)))
        visible_fractions.append(float(np.mean(visible)))
        received_fractions.append(float(np.mean(received)))
        ranking = row.get("candidate_ranking") or {}
        if str(ranking.get("execution_mode")) == "cautious_reacquisition":
            reacquisition_indices.append(index)
            action = _array(row.get("requested_action"), name="requested_action")
            reacquisition_action_norms.append(float(np.mean(np.linalg.norm(action, axis=1))))
            route = row.get("selected_route") or {}
            reacquisition_route_labels.append(str(route.get("label", "missing")))

    zero_belief_steps = [
        bool(np.max(np.linalg.norm(belief, axis=1)) <= 1e-9)
        for belief in beliefs
    ]
    never_indices = [index for index, value in enumerate(never_received) if value]
    never_zero = [zero_belief_steps[index] for index in never_indices]
    defender_centroid = np.mean(positions[0], axis=0)
    belief_centroid = np.mean(beliefs[0], axis=0)
    offline_truth_vector = target_truth - defender_centroid
    offline_belief_vector = belief_centroid - defender_centroid
    offline_bearing_error = float(np.linalg.norm(offline_truth_vector - offline_belief_vector))
    moved_after_reacquisition: list[float] = []
    for index in reacquisition_indices:
        if index + 1 < len(positions):
            moved_after_reacquisition.append(float(np.mean(np.linalg.norm(positions[index + 1] - positions[index], axis=1))))
    first_received = next((index for index, value in enumerate(received_fractions) if value >= 1.0), None)
    first_visible = next((index for index, value in enumerate(visible_fractions) if value > 0.0), None)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    overall = summary.get("overall", {}) if isinstance(summary, Mapping) else {}
    metrics = {
        "trace_steps": len(rows),
        "never_received_steps": len(never_indices),
        "never_received_zero_belief_steps": int(sum(never_zero)),
        "never_received_zero_belief_rate": float(np.mean(never_zero)) if never_zero else None,
        "mean_visible_fraction": float(np.mean(visible_fractions)),
        "mean_received_fraction": float(np.mean(received_fractions)),
        "cautious_reacquisition_steps": len(reacquisition_indices),
        "cautious_reacquisition_nonzero_action_steps": int(sum(value > 1e-9 for value in reacquisition_action_norms)),
        "cautious_reacquisition_mean_action_norm_mps": float(np.mean(reacquisition_action_norms)) if reacquisition_action_norms else 0.0,
        "cautious_reacquisition_mean_followup_motion_m": float(np.mean(moved_after_reacquisition)) if moved_after_reacquisition else 0.0,
        "first_all_received_step": first_received,
        "first_visible_step": first_visible,
        "offline_initial_target_bearing_error_m": offline_bearing_error,
    }
    gates = {
        "public_snapshots_complete": len(public_rows) == len(rows),
        "never_received_belief_zero_rate_is_reported": metrics["never_received_zero_belief_rate"] is not None,
        "reacquisition_is_active_search": bool(
            metrics["cautious_reacquisition_steps"] == 0
            or metrics["cautious_reacquisition_nonzero_action_steps"] > 0
            or metrics["cautious_reacquisition_mean_followup_motion_m"] > 1e-9
        ),
        "safety_hard_gates_zero": all(
            int(overall.get(field, 0)) == 0
            for field in (
                "collision_count",
                "boundary_violation_count",
                "pairwise_violation_count",
                "raw_unverified_executed_steps",
            )
        ),
    }
    result = {
        "audit_type": "jepa_safe_capture_observation_contract",
        "development_only": True,
        "locked_test_opened": False,
        "target_truth_used_only_for_offline_label": True,
        "run_dir": str(run_dir),
        "scene_manifest": str(scene_manifest.resolve()),
        "episode_index": int(episode_index),
        "metrics": metrics,
        "route_labels": dict(Counter(reacquisition_route_labels)),
        "gates": {**gates, "all_pass": bool(all(gates.values()))},
        "interpretation": {
            "causal_observation": (
                "Before first receipt, all target beliefs are zero and no target visibility is present; "
                "the controller has no measured target bearing in the public input."
                if metrics["never_received_zero_belief_rate"] == 1.0
                else "The public input contains non-zero belief information before first receipt."
            ),
            "reacquisition": (
                "The configured visibility_hold is stationary and therefore cannot actively search."
                if not gates["reacquisition_is_active_search"]
                else "The configured reacquisition produces motion; inspect its bearing and capture effect next."
            ),
        },
        "provenance": {
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
            "python": platform.python_version(),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "trace_sha256": _sha256(trace_path),
            "summary_sha256": _sha256(summary_path) if summary_path.is_file() else None,
            "scene_manifest_sha256": _sha256(scene_manifest.resolve()),
        },
    }
    return result


def write_outputs(result: Mapping[str, Any], output_dir: Path, tensorboard_dir: Path) -> None:
    output_dir = output_dir.resolve()
    tensorboard_dir = tensorboard_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    if tensorboard_dir.exists() and any(tensorboard_dir.iterdir()):
        raise FileExistsError(tensorboard_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "observation_contract.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    metrics = result["metrics"]
    gates = result["gates"]
    lines = [
        "# Observation Contract Audit",
        "",
        "`development_only=true`; `locked_test_opened=false`; target truth is used only as an offline label.",
        "",
        f"- Episode: `{result['episode_index']}`; trace steps: `{metrics['trace_steps']}`",
        f"- Never-received steps: `{metrics['never_received_steps']}`; zero-belief rate: `{metrics['never_received_zero_belief_rate']}`",
        f"- Cautious reacquisition steps: `{metrics['cautious_reacquisition_steps']}`; non-zero action steps: `{metrics['cautious_reacquisition_nonzero_action_steps']}`",
        f"- Mean follow-up motion after reacquisition: `{metrics['cautious_reacquisition_mean_followup_motion_m']:.6f} m`",
        f"- First all-received step: `{metrics['first_all_received_step']}`; first visible step: `{metrics['first_visible_step']}`",
        "",
        "## Interpretation",
        "",
        result["interpretation"]["causal_observation"],
        " ",
        result["interpretation"]["reacquisition"],
        "",
        f"All audit gates: **{'PASS' if gates['all_pass'] else 'FAIL'}**",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with SummaryWriter(log_dir=str(tensorboard_dir), flush_secs=1) as writer:
        writer.add_text("Config/provenance", json.dumps(result["provenance"], sort_keys=True), 0)
        writer.add_text("Audit/interpretation", json.dumps(result["interpretation"], sort_keys=True), 0)
        writer.add_text("Gates/status", json.dumps(gates, sort_keys=True), 0)
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                writer.add_scalar(f"Observation/{key}", float(value), 0)
        writer.flush()
    output = dict(result)
    output["tensorboard"] = {
        "logdir": str(tensorboard_dir),
        "event_files": sorted(path.name for path in tensorboard_dir.glob("events.out.tfevents.*")),
    }
    (output_dir / "observation_contract.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_dir, args.scene_manifest, args.episode_index)
    write_outputs(result, args.output_dir, args.tensorboard_dir)
    print(json.dumps({"metrics": result["metrics"], "gates": result["gates"]}, indent=2))


if __name__ == "__main__":
    main()
