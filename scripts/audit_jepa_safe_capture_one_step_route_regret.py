"""Audit one-step settled route regret from a frozen development trace.

The source trace contains public candidate requests and the route selected by
the online ranker.  This audit replays the environment to each decision state
and branches every geometrically valid, primary-CBF-accepted candidate for one
environment step.  The simulator's hidden target state is used only for this
offline label.  It is never fed back into the online controller.

This is a one-step settled counterfactual, not a claim that one action is the
best full-episode route.  Its purpose is to separate route-ranking error from
CBF infeasibility and late pairwise/boundary failure before another checkpoint
or training run is considered.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.cbf_qp import JointCBFQPSafetyFilter  # noqa: E402
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import prepare_showcase_episode, scenario_from_metadata  # noqa: E402
from evaluate_random_central_mixed_obstacles import config_for_spec  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"Trace is empty: {path}")
    return rows


def _as_action(value: Any, label: str) -> np.ndarray:
    action = np.asarray(value, dtype=np.float64)
    if action.ndim != 2 or action.shape[1:] != (3,) or not np.isfinite(action).all():
        raise ValueError(f"{label} must be finite with shape [defenders,3]")
    return action


def _target_clearance(env: CaptureRadiusPursuit3DEnv) -> float:
    if not env.obstacles:
        return float("inf")
    return float(min(float(env._obstacle_clearance(env.target_position, obstacle)) for obstacle in env.obstacles))


def _make_env(
    *,
    manifest_item: Mapping[str, Any],
    environment_config: Path,
) -> tuple[CaptureRadiusPursuit3DEnv, dict[str, Any]]:
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
    return env, observation


def _branch_candidate(
    *,
    manifest_item: Mapping[str, Any],
    environment_config: Path,
    prefix_actions: list[np.ndarray],
    requested_action: np.ndarray,
    nominal_action: np.ndarray,
    cbf_horizon: int,
) -> dict[str, Any]:
    env, observation = _make_env(manifest_item=manifest_item, environment_config=environment_config)
    for index, action in enumerate(prefix_actions):
        observation, _reward, terminated, truncated, _info = env.step(action, record_history=False)
        if terminated or truncated:
            raise RuntimeError(f"Replay terminated before audit state at prefix index {index}.")
    before_distance = float(np.min(env._target_distances()))
    before_target_clearance = _target_clearance(env)
    safety_filter = JointCBFQPSafetyFilter(
        env,
        anticipatory_horizon_steps=int(cbf_horizon),
        barrier_mode="strict_buffer",
    )
    executed, diagnostics = safety_filter.filter(
        requested_action,
        observation,
        nominal_actions=nominal_action,
        execution_mode="normal",
    )
    next_observation, _reward, terminated, truncated, info = env.step(executed, record_history=False)
    after_distance = float(info["nearest_target_distance"])
    after_target_clearance = _target_clearance(env)
    physical_safe = bool(
        diagnostics.verified_feasible
        and not env.defender_boundary_violation
        and not bool(info.get("collision", False))
        and not bool(env.defender_boundary_violation)
    )
    return {
        "before_distance_m": before_distance,
        "after_distance_m": after_distance,
        "distance_improvement_m": before_distance - after_distance,
        "before_target_clearance_m": before_target_clearance,
        "after_target_clearance_m": after_target_clearance,
        "capture_event": bool(info.get("capture_event", False)),
        "safe_capture": bool(info.get("safe_capture_success", False)) and physical_safe,
        "physical_safe": physical_safe,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "cbf_verified_feasible": bool(diagnostics.verified_feasible),
        "cbf_infeasible": bool(diagnostics.infeasible),
        "cbf_timed_out": bool(diagnostics.timed_out),
        "cbf_fallback_mode": str(diagnostics.fallback_mode),
        "cbf_minimum_constraint_value": float(diagnostics.minimum_constraint_value),
        "cbf_action_correction_norm": float(diagnostics.action_correction_norm),
        "cbf_active_constraints": [str(value) for value in diagnostics.active_constraints],
        "executed_action": np.asarray(executed, dtype=np.float64).tolist(),
        "next_observation_step": int(next_observation.get("step", env.step_count)),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    run = args.source_run.resolve()
    manifest_path = run / "scene_manifest.jsonl"
    trace_path = run / "step_traces" / f"episode_{int(args.episode_index):04d}.jsonl"
    for path in (manifest_path, trace_path, args.environment_config.resolve()):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest_rows = _read_jsonl(manifest_path)
    manifest_item = next(
        (item for item in manifest_rows if int(item.get("episode_index", -1)) == int(args.episode_index)),
        None,
    )
    if manifest_item is None:
        raise ValueError(f"Episode {args.episode_index} is missing from {manifest_path}")
    traces = _read_jsonl(trace_path)
    prefix_actions: list[np.ndarray] = []
    step_rows: list[dict[str, Any]] = []
    for step_index, trace in enumerate(traces):
        if int(trace.get("step", step_index + 1)) != step_index + 1:
            raise ValueError(f"Trace step sequence is not contiguous at index {step_index}")
        ranking = trace.get("candidate_ranking") or {}
        runtime = trace.get("route_runtime") or {}
        routes = ((runtime.get("routes") or {}).get("candidates") or [])
        labels = list((runtime.get("routes") or {}).get("labels") or ranking.get("candidate_labels") or [])
        valid_mask = list((runtime.get("candidate_batch") or {}).get("valid_mask") or [])
        probes = list(runtime.get("cbf_counterfactuals") or [])
        if not routes or len(routes) != len(labels):
            prefix_actions.append(_as_action(trace["executed_action"], "executed_action"))
            continue
        nominal_action = _as_action(trace["reachable_nominal_action"], "reachable_nominal_action")
        selected_index = int(ranking.get("selected_index", 0))
        execution_mode = str(ranking.get("execution_mode", "missing"))
        online_trusted = execution_mode == "trusted"
        score_values = list(ranking.get("scores") or [])
        eligible = list(ranking.get("eligible_mask") or [])
        candidate_outcomes: list[dict[str, Any] | None] = []
        for candidate_index, route in enumerate(routes):
            probe = probes[candidate_index] if candidate_index < len(probes) else None
            primary_accepted = bool(
                valid_mask[candidate_index]
                and isinstance(probe, Mapping)
                and bool(probe.get("accepted", False))
            )
            if not primary_accepted:
                candidate_outcomes.append(None)
                continue
            chunk = route.get("action_chunk") if isinstance(route, Mapping) else None
            requested = _as_action(chunk[0], f"candidate[{candidate_index}].action_chunk[0]")
            outcome = _branch_candidate(
                manifest_item=manifest_item,
                environment_config=args.environment_config.resolve(),
                prefix_actions=prefix_actions,
                requested_action=requested,
                nominal_action=nominal_action,
                cbf_horizon=int(args.cbf_horizon),
            )
            outcome["label"] = str(labels[candidate_index])
            outcome["candidate_index"] = int(candidate_index)
            outcome["primary_probe_accepted"] = True
            outcome["ranker_score"] = _finite(score_values[candidate_index]) if candidate_index < len(score_values) else None
            outcome["ranker_eligible"] = bool(eligible[candidate_index]) if candidate_index < len(eligible) else False
            candidate_outcomes.append(outcome)
        settled_candidates = [
            item for item in candidate_outcomes
            if item is not None and bool(item["physical_safe"]) and bool(item["cbf_verified_feasible"])
        ]
        settled_best = min(
            settled_candidates,
            key=lambda item: (float(item["after_distance_m"]), int(item["candidate_index"])),
        ) if settled_candidates else None
        selected_outcome = next(
            (item for item in candidate_outcomes if item is not None and int(item["candidate_index"]) == selected_index),
            None,
        )
        score_argmin = None
        finite_eligible = [
            index for index, value in enumerate(score_values)
            if index < len(eligible) and bool(eligible[index]) and _finite(value) is not None
        ]
        if finite_eligible:
            score_argmin = min(finite_eligible, key=lambda index: (float(score_values[index]), index))
        step_rows.append(
            {
                "step": int(step_index + 1),
                "selected_index": selected_index,
                "selected_label": str(labels[selected_index]) if selected_index < len(labels) else "unknown",
                "online_execution_mode": execution_mode,
                "online_trusted": online_trusted,
                "score_argmin_index": score_argmin,
                "score_argmin_matches_selected": score_argmin == selected_index if score_argmin is not None else None,
                "candidate_count": len(routes),
                "primary_accepted_count": len(settled_candidates),
                "settled_best_index": None if settled_best is None else int(settled_best["candidate_index"]),
                "settled_best_label": None if settled_best is None else str(settled_best["label"]),
                "selected_matches_settled_best": (
                    None
                    if settled_best is None or not online_trusted
                    else selected_index == int(settled_best["candidate_index"])
                ),
                "selected_regret_m": (
                    None
                    if selected_outcome is None or settled_best is None or not online_trusted
                    else float(selected_outcome["after_distance_m"] - settled_best["after_distance_m"])
                ),
                "selected_outcome": selected_outcome,
                "settled_best_outcome": settled_best,
                "candidates": candidate_outcomes,
            }
        )
        prefix_actions.append(_as_action(trace["executed_action"], "executed_action"))

    comparable = [
        row for row in step_rows
        if row["online_trusted"] and row["settled_best_index"] is not None and row["selected_outcome"] is not None
    ]
    selected_matches = [bool(row["selected_matches_settled_best"]) for row in comparable]
    regrets = [float(row["selected_regret_m"]) for row in comparable]
    score_matches = [bool(row["score_argmin_matches_selected"]) for row in comparable if row["score_argmin_matches_selected"] is not None]
    result: dict[str, Any] = {
        "audit_type": "jepa_safe_capture_one_step_settled_route_regret",
        "development_only": True,
        "locked_test_opened": False,
        "online_contract_modified": False,
        "settled_best_claim_scope": "one_step_counterfactual_only",
        "source_run": str(run),
        "source_run_summary_sha256": _sha256(run / "summary.json"),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_trace_sha256": _sha256(trace_path),
        "environment_config": str(args.environment_config.resolve()),
        "environment_config_sha256": _sha256(args.environment_config.resolve()),
        "episode_index": int(args.episode_index),
        "episode_seed": int(manifest_item["episode_seed"]),
        "cbf_contract": {"barrier_mode": "strict_buffer", "anticipatory_horizon_steps": int(args.cbf_horizon)},
        "steps_with_route_trace": len(step_rows),
        "comparable_steps": len(comparable),
        "selected_matches_settled_best_rate": float(np.mean(selected_matches)) if selected_matches else None,
        "mean_selected_regret_m": float(np.mean(regrets)) if regrets else None,
        "max_selected_regret_m": float(np.max(regrets)) if regrets else None,
        "score_argmin_matches_selected_rate": float(np.mean(score_matches)) if score_matches else None,
        "primary_accepted_candidate_rate": float(
            np.mean([row["primary_accepted_count"] / max(row["candidate_count"], 1) for row in step_rows])
        ) if step_rows else None,
        "steps_with_selected_cbf_failure": int(sum(
            row["selected_outcome"] is not None and not bool(row["selected_outcome"]["cbf_verified_feasible"])
            for row in step_rows
        )),
        "steps": step_rows,
        "interpretation": {
            "target_truth_used": "offline_branch_labels_only",
            "online_controller_replayed": True,
            "next_gate": "do_not_enable_online_score_term_until_route_regret_is_calibrated",
        },
    }
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    if args.tensorboard_dir.exists() and any(args.tensorboard_dir.iterdir()):
        raise FileExistsError(args.tensorboard_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "route_regret.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# One-Step Settled Route-Regret Audit",
        "",
        "This is a development-only one-step counterfactual audit. It does not claim a full-episode settled-best route.",
        "",
        f"- Episode: `{result['episode_index']}` / seed `{result['episode_seed']}`",
        f"- Comparable steps: `{result['comparable_steps']}`",
        f"- Selected route matches one-step settled best: `{result['selected_matches_settled_best_rate']}`",
        f"- Mean selected regret: `{result['mean_selected_regret_m']}` m",
        f"- Maximum selected regret: `{result['max_selected_regret_m']}` m",
        f"- Score argmin matches selected: `{result['score_argmin_matches_selected_rate']}`",
        f"- Selected CBF-failure steps: `{result['steps_with_selected_cbf_failure']}`",
        "",
        "The labels use simulator target state only after branching offline. They are not available to the online ranker.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with SummaryWriter(log_dir=str(args.tensorboard_dir), flush_secs=1) as writer:
        writer.add_text("Config/audit", json.dumps(result["interpretation"], indent=2), 0)
        writer.add_text("Provenance/source", json.dumps({key: result[key] for key in (
            "source_manifest_sha256", "source_trace_sha256", "episode_seed", "cbf_contract"
        )}, indent=2), 0)
        writer.add_scalar("Gates/settled_best_claim_available", 1.0, 0)
        writer.add_scalar("Gates/online_contract_modified", 0.0, 0)
        writer.add_scalar("RouteRegret/comparable_steps", float(result["comparable_steps"]), 0)
        if result["selected_matches_settled_best_rate"] is not None:
            writer.add_scalar("RouteRegret/selected_matches_settled_best_rate", float(result["selected_matches_settled_best_rate"]), 0)
            writer.add_scalar("RouteRegret/mean_selected_regret_m", float(result["mean_selected_regret_m"]), 0)
            writer.add_scalar("RouteRegret/max_selected_regret_m", float(result["max_selected_regret_m"]), 0)
        if result["score_argmin_matches_selected_rate"] is not None:
            writer.add_scalar("RouteRegret/score_argmin_matches_selected_rate", float(result["score_argmin_matches_selected_rate"]), 0)
        for row in step_rows:
            step = int(row["step"])
            if row["selected_regret_m"] is not None:
                writer.add_scalar("RouteRegret/selected_regret_m", float(row["selected_regret_m"]), step)
            writer.add_scalar("RouteRegret/primary_accepted_count", float(row["primary_accepted_count"]), step)
            writer.add_scalar("RouteRegret/selected_matches_settled_best", float(bool(row["selected_matches_settled_best"])), step)
    result["tensorboard"] = {
        "logdir": str(args.tensorboard_dir.resolve()),
        "event_files": sorted(path.name for path in args.tensorboard_dir.glob("events.out.tfevents.*")),
    }
    (args.output_dir / "route_regret.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--environment-config", type=Path, required=True)
    parser.add_argument("--cbf-horizon", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    args = parser.parse_args()
    if not args.development_only:
        raise ValueError("This audit requires --development-only.")
    if args.episode_index < 0 or args.cbf_horizon <= 0:
        raise ValueError("episode-index must be non-negative and cbf-horizon must be positive.")
    result = audit(args)
    print(json.dumps({key: result[key] for key in (
        "audit_type", "episode_seed", "comparable_steps", "selected_matches_settled_best_rate",
        "mean_selected_regret_m", "max_selected_regret_m", "score_argmin_matches_selected_rate",
        "steps_with_selected_cbf_failure", "tensorboard"
    )}, indent=2))


if __name__ == "__main__":
    main()
