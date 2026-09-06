"""Audit route-ranking predictions against offline branch rollouts.

This tool consumes an already-produced development trace and replays the
recorded control cycle from the same scene manifest.  It also rolls out each
valid route chunk from the pre-action state in an isolated environment clone.
Target truth is used only inside this offline audit to produce a diagnostic
label; it is never passed to the online planner, JEPA, Ledger, or CBF path.

The audit is intentionally separate from the evaluator so a route-ranking
failure can be diagnosed without changing the execution contract or opening a
locked test.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.cbf_qp import JointCBFQPSafetyFilter  # noqa: E402
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import prepare_showcase_episode, scenario_from_metadata  # noqa: E402
from evaluate_random_central_mixed_obstacles import config_for_spec  # noqa: E402


EXTENT_M = 10.0
DEFAULT_MAX_STEPS = 50
REPLAY_TOLERANCE_M = 1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _mean(values: list[float]) -> float | None:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else None


def _pairwise_sign_agreement(predicted: list[float], actual: list[float]) -> float | None:
    if len(predicted) != len(actual) or len(predicted) < 2:
        return None
    correct = 0
    informative = 0
    for left in range(len(predicted)):
        for right in range(left + 1, len(predicted)):
            predicted_delta = float(predicted[left] - predicted[right])
            actual_delta = float(actual[left] - actual[right])
            if abs(predicted_delta) <= 1e-9 or abs(actual_delta) <= 1e-9:
                continue
            informative += 1
            correct += int(np.sign(predicted_delta) == np.sign(actual_delta))
    return float(correct / informative) if informative else None


def _top1_index(values: list[float]) -> int | None:
    if not values:
        return None
    return int(np.argmax(np.asarray(values, dtype=np.float64)))


def _read_trace(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    run_dir = run_dir.resolve()
    summary = _json(run_dir / "summary.json")
    provenance = _json(run_dir / "provenance.json")
    summary_metadata = summary.get("metadata", summary)
    if not isinstance(summary_metadata, Mapping):
        raise ValueError(f"Summary metadata is not an object: {run_dir}")
    for source, name in ((summary_metadata, "summary"), (provenance, "provenance")):
        if source.get("development_only") is not True or source.get("locked_test_opened") is not False:
            raise ValueError(f"{name} crosses the development boundary: {run_dir}")
    paths = sorted((run_dir / "step_traces").glob("episode_*.jsonl"))
    if not paths:
        raise ValueError(f"No step traces found: {run_dir}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Trace row is not an object: {path}")
            rows.append(value)
    rows.sort(key=lambda row: (int(row.get("episode_index", -1)), int(row.get("step", -1))))
    if not rows:
        raise ValueError(f"Empty step traces: {run_dir}")
    return summary, provenance, rows


def _load_manifest(run_dir: Path, trace_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    path = run_dir.resolve() / "scene_manifest.jsonl"
    records: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"Scene manifest row is not an object: {path}")
        index = int(item["episode_index"])
        records[index] = item
    expected = {int(row["episode_index"]) for row in trace_rows}
    if set(records) != expected:
        raise ValueError(f"Scene manifest episodes {sorted(records)} do not match traces {sorted(expected)}")
    return records


def _target_clearance(env: CaptureRadiusPursuit3DEnv) -> float:
    if not env.obstacles:
        return float("inf")
    return float(min(float(env._obstacle_clearance(env.target_position, obstacle)) for obstacle in env.obstacles))


def _branch_rollout(
    env: CaptureRadiusPursuit3DEnv,
    observation: Mapping[str, Any],
    route: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Roll out one route chunk in a private clone for offline diagnostics."""

    raw_chunk = np.asarray(route.get("action_chunk"), dtype=np.float64)
    if raw_chunk.ndim != 3 or raw_chunk.shape != (3, env.n_defenders, 3) or not np.isfinite(raw_chunk).all():
        return None
    clone = copy.deepcopy(env)
    clone_observation = copy.deepcopy(dict(observation))
    safety_filter = JointCBFQPSafetyFilter(clone)
    initial_distances = np.linalg.norm(clone.target_position[None, :] - clone.defender_positions, axis=1)
    corrections: list[float] = []
    for requested in raw_chunk:
        executed, diagnostics = safety_filter.filter(requested, clone_observation)
        if not diagnostics.verified_feasible or diagnostics.fallback_mode != "none":
            return {
                "feasible": False,
                "cbf_fallback": str(diagnostics.fallback_mode),
                "cbf_infeasible": bool(diagnostics.infeasible),
                "cbf_timeout": bool(diagnostics.timed_out),
            }
        corrections.append(float(diagnostics.action_correction_norm))
        clone_observation, _reward, terminated, truncated, _info = clone.step(executed, record_history=True)
        if terminated or truncated:
            break
    distances = np.linalg.norm(clone.target_position[None, :] - clone.defender_positions, axis=1)
    return {
        "feasible": True,
        "steps": int(len(corrections)),
        "progress_m": float(np.mean(initial_distances - distances)),
        "target_distance_m": float(np.mean(distances)),
        "target_clearance_m": _target_clearance(clone),
        "cbf_correction_mean_mps": float(np.mean(corrections)) if corrections else 0.0,
        "cbf_correction_max_mps": float(np.max(corrections)) if corrections else 0.0,
    }


def _episode_rows(
    *,
    trace_rows: list[dict[str, Any]],
    manifest_item: Mapping[str, Any],
    environment_config: Path,
    max_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = dict(manifest_item["spec"])
    scenario = scenario_from_metadata(dict(manifest_item["scenario"]))
    config = config_for_spec("f2", spec, environment_config)
    env = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=len(scenario.obstacles),
        target_speed_scale=float(spec["target_speed_scale"]),
    )
    observation = prepare_showcase_episode(
        env,
        scenario,
        seed=int(spec["episode_seed"]),
        record_history=True,
        validate_scenario=False,
    )
    selected_rows: list[dict[str, Any]] = []
    replay_errors: list[float] = []
    source_rows = [row for row in trace_rows if int(row["episode_index"]) == int(manifest_item["episode_index"])]
    for source in source_rows[:max_steps]:
        ranking = source.get("candidate_ranking")
        runtime = source.get("route_runtime")
        route_payload = runtime.get("routes", {}) if isinstance(runtime, Mapping) else {}
        routes = route_payload.get("candidates", []) if isinstance(route_payload, Mapping) else []
        if not isinstance(ranking, Mapping) or not isinstance(routes, list):
            branch_results: list[dict[str, Any] | None] = []
        else:
            branch_results = [
                _branch_rollout(env, observation, route) if isinstance(route, Mapping) and bool(route.get("valid", False)) else None
                for route in routes
            ]
        labels = list(ranking.get("candidate_labels", [])) if isinstance(ranking, Mapping) else []
        predicted_raw = ranking.get("predicted_route_progress_m", []) if isinstance(ranking, Mapping) else []
        valid_raw = ranking.get("valid_mask", []) if isinstance(ranking, Mapping) else []
        predicted: list[float] = []
        actual: list[float] = []
        compared_labels: list[str] = []
        for index, branch in enumerate(branch_results):
            if branch is None or not branch.get("feasible"):
                continue
            if index >= len(predicted_raw) or index >= len(valid_raw) or not bool(valid_raw[index]):
                continue
            value = _finite(predicted_raw[index])
            if value is None or not np.isfinite(float(branch.get("progress_m", np.nan))):
                continue
            predicted.append(value)
            actual.append(float(branch["progress_m"]))
            compared_labels.append(str(labels[index]) if index < len(labels) else str(index))
        predicted_top = _top1_index(predicted)
        actual_top = _top1_index(actual)
        selected_index = int(ranking.get("selected_index", 0)) if isinstance(ranking, Mapping) else 0
        selected_label = str(labels[selected_index]) if 0 <= selected_index < len(labels) else None
        selected_actual = None
        selected_predicted = None
        if 0 <= selected_index < len(branch_results) and branch_results[selected_index] is not None:
            branch = branch_results[selected_index]
            if branch is not None and branch.get("feasible"):
                selected_actual = float(branch["progress_m"])
            if selected_index < len(predicted_raw):
                selected_predicted = _finite(predicted_raw[selected_index])
        replay_observation, _reward, terminated, truncated, _info = env.step(
            np.asarray(source.get("executed_action"), dtype=np.float64),
            record_history=True,
        )
        observed_clearance = _finite(source.get("target_clearance_m"))
        replay_clearance = _target_clearance(env)
        if observed_clearance is not None and np.isfinite(replay_clearance):
            replay_errors.append(abs(observed_clearance - replay_clearance))
        selected_rows.append(
            {
                "episode_index": int(source["episode_index"]),
                "step": int(source["step"]),
                "selected_index": selected_index,
                "selected_label": selected_label,
                "execution_mode": ranking.get("execution_mode") if isinstance(ranking, Mapping) else None,
                "ledger_state": (ranking.get("ledger_states") or [None])[0] if isinstance(ranking, Mapping) else None,
                "predicted_top_label": compared_labels[predicted_top] if predicted_top is not None else None,
                "actual_top_label": compared_labels[actual_top] if actual_top is not None else None,
                "top1_agreement": bool(predicted_top is not None and predicted_top == actual_top),
                "pairwise_sign_agreement": _pairwise_sign_agreement(predicted, actual),
                "candidate_count_compared": len(predicted),
                "selected_predicted_progress_m": selected_predicted,
                "selected_actual_progress_m": selected_actual,
                "selected_progress_gap_m": (
                    float(selected_predicted - selected_actual)
                    if selected_predicted is not None and selected_actual is not None
                    else None
                ),
                "predicted_progress_by_label": dict(zip(compared_labels, predicted)),
                "actual_progress_by_label": dict(zip(compared_labels, actual)),
                "replay_target_clearance_m": replay_clearance,
                "source_target_clearance_m": observed_clearance,
                "replay_target_clearance_abs_error_m": (
                    abs(observed_clearance - replay_clearance)
                    if observed_clearance is not None and np.isfinite(replay_clearance)
                    else None
                ),
                "offline_only": True,
            }
        )
        observation = replay_observation
        if terminated or truncated:
            break
    summary = {
        "episode_index": int(manifest_item["episode_index"]),
        "episode_seed": int(spec["episode_seed"]),
        "steps_audited": len(selected_rows),
        "route_groups_compared": sum(int(row["candidate_count_compared"] >= 2) for row in selected_rows),
        "top1_agreement_rate": _mean([float(row["top1_agreement"]) for row in selected_rows if row["candidate_count_compared"] >= 2]),
        "pairwise_sign_agreement": _mean(
            [float(value) for row in selected_rows if (value := _finite(row.get("pairwise_sign_agreement"))) is not None]
        ),
        "selected_predicted_progress_m": _mean(
            [float(value) for row in selected_rows if (value := _finite(row.get("selected_predicted_progress_m"))) is not None]
        ),
        "selected_actual_progress_m": _mean(
            [float(value) for row in selected_rows if (value := _finite(row.get("selected_actual_progress_m"))) is not None]
        ),
        "selected_progress_gap_m": _mean(
            [float(value) for row in selected_rows if (value := _finite(row.get("selected_progress_gap_m"))) is not None]
        ),
        "max_replay_target_clearance_abs_error_m": max(replay_errors) if replay_errors else None,
    }
    return selected_rows, summary


def _write_tensorboard(logdir: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard directory: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/route_runtime_alignment", json.dumps(result["policy"], indent=2), 0)
        writer.add_text("Provenance/inputs", json.dumps(result["inputs"], indent=2), 0)
        writer.add_text("Gates/status", json.dumps(result["gates"], indent=2), 0)
        aggregate = result["aggregate"]
        for name in (
            "top1_agreement_rate",
            "pairwise_sign_agreement",
            "selected_predicted_progress_m",
            "selected_actual_progress_m",
            "selected_progress_gap_m",
        ):
            value = _finite(aggregate.get(name))
            writer.add_scalar(f"Alignment/{name}", value if value is not None else 0.0, 0)
        writer.add_scalar("Alignment/route_groups_compared", float(aggregate["route_groups_compared"]), 0)
        writer.add_scalar("Replay/max_target_clearance_abs_error_m", float(aggregate.get("max_replay_target_clearance_abs_error_m") or 0.0), 0)
        for index, row in enumerate(result["steps"]):
            writer.add_scalar(f"Steps/{index:04d}/candidate_count", float(row["candidate_count_compared"]), 0)
            writer.add_scalar(f"Steps/{index:04d}/top1_agreement", float(row["top1_agreement"]), 0)
            value = _finite(row.get("pairwise_sign_agreement"))
            writer.add_scalar(f"Steps/{index:04d}/pairwise_sign_agreement", value if value is not None else 0.0, 0)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required_text = {
        "Config/route_runtime_alignment/text_summary",
        "Provenance/inputs/text_summary",
        "Gates/status/text_summary",
    }
    required_scalars = {"Alignment/route_groups_compared", "Replay/max_target_clearance_abs_error_m"}
    missing = sorted(required_text.difference(tags.get("tensors", []))) + sorted(required_scalars.difference(tags.get("scalars", [])))
    events = sorted(path.name for path in logdir.glob("events.out.tfevents.*"))
    if missing or not events:
        raise ValueError(f"Route alignment TensorBoard validation failed: missing={missing}, events={events}")
    return {"logdir": str(logdir), "event_files": events, "required_provenance": True}


def audit_route_runtime_alignment(
    run: Path,
    protocol: Path,
    environment_config: Path,
    output_dir: Path,
    tensorboard_logdir: Path,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict[str, Any]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    summary, provenance, traces = _read_trace(run)
    manifest = _load_manifest(run, traces)
    metadata = summary.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Run summary metadata is missing")
    source_protocol_hash = metadata.get("inputs", {}).get("protocol_sha256")
    if source_protocol_hash != _sha256(protocol.resolve()):
        raise ValueError("Protocol hash does not match the run provenance")
    source_env_hash = metadata.get("inputs", {}).get("environment_config_sha256")
    if source_env_hash != _sha256(environment_config.resolve()):
        raise ValueError("Environment-config hash does not match the run provenance")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    step_rows: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    for episode_index in sorted(manifest):
        rows, episode_summary = _episode_rows(
            trace_rows=traces,
            manifest_item=manifest[episode_index],
            environment_config=environment_config.resolve(),
            max_steps=max_steps,
        )
        step_rows.extend(rows)
        episode_summaries.append(episode_summary)
    compared = [row for row in step_rows if int(row["candidate_count_compared"]) >= 2]
    aggregate = {
        "episodes": len(episode_summaries),
        "steps_audited": len(step_rows),
        "route_groups_compared": len(compared),
        "top1_agreement_rate": _mean([float(row["top1_agreement"]) for row in compared]),
        "pairwise_sign_agreement": _mean(
            [float(value) for row in compared if (value := _finite(row.get("pairwise_sign_agreement"))) is not None]
        ),
        "selected_predicted_progress_m": _mean(
            [float(value) for row in step_rows if (value := _finite(row.get("selected_predicted_progress_m"))) is not None]
        ),
        "selected_actual_progress_m": _mean(
            [float(value) for row in step_rows if (value := _finite(row.get("selected_actual_progress_m"))) is not None]
        ),
        "selected_progress_gap_m": _mean(
            [float(value) for row in step_rows if (value := _finite(row.get("selected_progress_gap_m"))) is not None]
        ),
        "max_replay_target_clearance_abs_error_m": max(
            (float(value) for row in step_rows if (value := _finite(row.get("replay_target_clearance_abs_error_m"))) is not None),
            default=None,
        ),
    }
    gates = {
        "development_only": True,
        "locked_test_not_opened": True,
        "offline_only_ground_truth": True,
        "trace_rows_audited": len(step_rows) > 0,
        "route_groups_compared": len(compared) > 0,
        "replay_target_clearance_within_tolerance": (
            aggregate["max_replay_target_clearance_abs_error_m"] is not None
            and aggregate["max_replay_target_clearance_abs_error_m"] <= REPLAY_TOLERANCE_M
        ),
    }
    result: dict[str, Any] = {
        "audit_type": "route_runtime_alignment_offline_branch_replay_v1",
        "development_only": True,
        "locked_test_opened": False,
        "offline_only": True,
        "policy": {
            "max_steps_per_episode": max_steps,
            "target_truth_used_only_for_offline_branch_metrics": True,
            "online_planner_unchanged": True,
            "jepa_role": "trajectory_evaluator_only",
            "cbf_role": "final_filter_for_each_offline_branch",
            "safe_capture_is_not_claimed": True,
        },
        "inputs": {
            "run": str(run.resolve()),
            "summary_sha256": _sha256(run.resolve() / "summary.json"),
            "provenance_sha256": _sha256(run.resolve() / "provenance.json"),
            "scene_manifest_sha256": _sha256(run.resolve() / "scene_manifest.jsonl"),
            "protocol": str(protocol.resolve()),
            "protocol_sha256": _sha256(protocol.resolve()),
            "environment_config": str(environment_config.resolve()),
            "environment_config_sha256": _sha256(environment_config.resolve()),
        },
        "source_run_provenance": provenance,
        "aggregate": aggregate,
        "episodes": episode_summaries,
        "steps": step_rows,
        "gates": gates,
        "provenance": {
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
            ).strip(),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    result["tensorboard"] = _write_tensorboard(tensorboard_logdir, result)
    (output_dir / "route_alignment_audit.json").write_text(
        json.dumps(_jsonable(result), indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "route_alignment_steps.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "episode_index", "step", "selected_index", "selected_label", "execution_mode",
            "ledger_state", "predicted_top_label", "actual_top_label", "top1_agreement",
            "pairwise_sign_agreement", "candidate_count_compared", "selected_predicted_progress_m",
            "selected_actual_progress_m", "selected_progress_gap_m", "replay_target_clearance_m",
            "source_target_clearance_m", "replay_target_clearance_abs_error_m", "offline_only",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in step_rows:
            writer.writerow({field: row.get(field) for field in fields})
    lines = [
        "# Route Runtime Alignment Audit",
        "",
        "Development-only offline branch replay; `locked_test_opened=false`.",
        "Target truth is used only for offline branch metrics and is not available to the online stack.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Episodes | {aggregate['episodes']} |",
        f"| Steps audited | {aggregate['steps_audited']} |",
        f"| Route groups compared | {aggregate['route_groups_compared']} |",
        f"| Predicted/actual top-1 agreement | {aggregate['top1_agreement_rate'] if aggregate['top1_agreement_rate'] is not None else 'n/a'} |",
        f"| Pairwise sign agreement | {aggregate['pairwise_sign_agreement'] if aggregate['pairwise_sign_agreement'] is not None else 'n/a'} |",
        f"| Selected predicted progress (m) | {aggregate['selected_predicted_progress_m'] if aggregate['selected_predicted_progress_m'] is not None else 'n/a'} |",
        f"| Selected actual progress (m) | {aggregate['selected_actual_progress_m'] if aggregate['selected_actual_progress_m'] is not None else 'n/a'} |",
        f"| Selected progress gap (m) | {aggregate['selected_progress_gap_m'] if aggregate['selected_progress_gap_m'] is not None else 'n/a'} |",
        f"| Max replay clearance error (m) | {aggregate['max_replay_target_clearance_abs_error_m'] if aggregate['max_replay_target_clearance_abs_error_m'] is not None else 'n/a'} |",
        "",
        "This audit diagnoses route-evaluator alignment; it is not evidence of a safe-capture improvement.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hashes = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "hash_manifest.json":
            hashes[path.name] = _sha256(path)
    (output_dir / "hash_manifest.json").write_text(json.dumps(hashes, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--environment-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--development-only", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.development_only:
        raise ValueError("Route runtime alignment requires --development-only")
    result = audit_route_runtime_alignment(
        args.run,
        args.protocol,
        args.environment_config,
        args.output_dir,
        args.tensorboard_logdir,
        max_steps=args.max_steps,
    )
    print(
        json.dumps(
            {
                "aggregate": result["aggregate"],
                "gates": result["gates"],
                "tensorboard": result["tensorboard"],
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
