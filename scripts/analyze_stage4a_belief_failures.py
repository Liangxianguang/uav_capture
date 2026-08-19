"""Diagnose reproducible Stage 3C-P1 delay and occlusion failure trajectories.

The evaluator deliberately replays only seeds that failed in the frozen P1
raw-action CSV files. Ground-truth target states are used only after rollout
to quantify local-belief error; they are never supplied to an actor, predictor,
or safety filter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_capture_radius_mappo import (  # noqa: E402
    load_policy,
    load_prediction_model,
    select_device,
)
from encirclement3d.prediction import HistoryTargetPredictor, LearnedPredictionObserver  # noqa: E402
from encirclement3d.pursuit_controllers import PursuitCBFSafetyFilter  # noqa: E402
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.learning import RecurrentCentralizedSharedActorCritic  # noqa: E402
from run_stage3c_p1_stress import (  # noqa: E402
    ACTIONS,
    BASE_CONFIGS,
    CONDITIONS,
    METHODS,
    checkpoint_path,
    make_config,
    prediction_args_for,
)


DEFAULT_CONDITIONS = ("delayed_measurements", "burst_occlusion")
_REPRODUCTION_FIELDS = (
    "safe_capture_success",
    "capture_event",
    "collision",
    "steps",
    "termination_reason",
    "world_violation_steps",
)
_AGE_BINS = (("fresh_0_1", 0, 1), ("moderate_2_4", 2, 4), ("stale_5_plus", 5, None))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "stage3c_p1_stress",
        help="Completed P1 result root containing episodes_raw.csv files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "stage4a_belief_diagnostics",
        help="New local directory for replay trajectories and diagnostic tables.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=PROJECT_ROOT / "results" / "STAGE4A_BELIEF_DIAGNOSTIC_REPORT.md",
        help="Tracked Markdown summary for this diagnostic stage.",
    )
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS), default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--action", choices=ACTIONS, default="raw")
    parser.add_argument("--max-failures-per-condition", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    return parser.parse_args()


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected a boolean CSV value, got {value!r}.")


def read_failed_rows(
    source_root: Path,
    conditions: Iterable[str],
    methods: Iterable[str],
    action: str,
) -> dict[str, list[dict[str, str]]]:
    """Load raw-action failures from the immutable P1 per-seed CSV artifacts."""
    by_condition: dict[str, list[dict[str, str]]] = {condition: [] for condition in conditions}
    for condition in conditions:
        for method in methods:
            method_root = source_root / method
            if not method_root.is_dir():
                raise FileNotFoundError(f"Missing P1 method root: {method_root}")
            for csv_path in sorted(method_root.glob(f"seed*/{condition}/episodes_{action}.csv")):
                with csv_path.open("r", encoding="utf-8", newline="") as stream:
                    for row in csv.DictReader(stream):
                        if str(row.get("condition")) != condition or str(row.get("method")) != method:
                            raise ValueError(f"Unexpected P1 row provenance in {csv_path}: {row}")
                        if str(row.get("action")) != action:
                            raise ValueError(f"Unexpected P1 action provenance in {csv_path}: {row}")
                        if not parse_bool(str(row["safe_capture_success"])):
                            by_condition[condition].append(dict(row))
        if not by_condition[condition]:
            raise ValueError(f"No failed {action}-action P1 rows found for condition {condition!r}.")
    return by_condition


def select_stratified_failures(rows: Iterable[dict[str, str]], maximum: int) -> list[dict[str, str]]:
    """Round-robin selection across method and training seed without cherry-picking."""
    if maximum <= 0:
        raise ValueError("maximum must be positive.")
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["training_seed"]))].append(dict(row))
    ordered_groups = [(key, sorted(values, key=lambda item: int(item["seed"]))) for key, values in sorted(grouped.items())]
    selected: list[dict[str, str]] = []
    cursors = {key: 0 for key, _values in ordered_groups}
    while len(selected) < maximum:
        made_progress = False
        for key, values in ordered_groups:
            cursor = cursors[key]
            if cursor >= len(values):
                continue
            selected.append(values[cursor])
            cursors[key] = cursor + 1
            made_progress = True
            if len(selected) == maximum:
                break
        if not made_progress:
            break
    return selected


def _append_frame(frames: dict[str, list[np.ndarray]], observation: dict[str, Any], env: CaptureRadiusPursuit3DEnv) -> None:
    """Save local inputs plus simulator labels for offline diagnostic computation."""
    frames["defender_positions"].append(np.asarray(observation["defender_positions"], dtype=np.float64).copy())
    frames["target_positions"].append(env.target_position.copy())
    frames["target_velocities"].append(env.target_velocity.copy())
    frames["belief_positions"].append(np.asarray(observation["target_belief_positions"], dtype=np.float64).copy())
    frames["belief_velocities"].append(np.asarray(observation["target_belief_velocities"], dtype=np.float64).copy())
    frames["observation_ages"].append(
        np.asarray(observation["target_observation_age_steps"], dtype=np.int64).copy()
    )
    frames["observation_timestamps"].append(
        np.asarray(observation["target_observation_timestamps"], dtype=np.int64).copy()
    )
    frames["message_ages"].append(np.asarray(observation["message_age_steps"], dtype=np.int64).copy())
    frames["visible"].append(np.asarray(observation["target_visible"], dtype=bool).copy())
    covariance = np.asarray(observation["target_observation_covariance"], dtype=np.float64)
    frames["covariance_traces"].append(np.trace(covariance, axis1=1, axis2=2).copy())


def replay_with_diagnostics(
    *,
    policy: Any,
    config: dict[str, Any],
    obstacle_count: int,
    target_speed_scale: float,
    episode_seed: int,
    device: torch.device,
    action_scale: float,
    action: str,
    prediction_model: HistoryTargetPredictor | None,
    prediction_history_length: int,
    prediction_horizon_index: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Replay one frozen policy and retain diagnostic-only frame arrays."""
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=obstacle_count, target_speed_scale=target_speed_scale)
    observation = env.reset(seed=episode_seed)
    safety_filter = PursuitCBFSafetyFilter(env) if action == "cbf" else None
    prediction_observer = (
        LearnedPredictionObserver(
            env,
            prediction_model,
            device,
            history_length=prediction_history_length,
            horizon_index=prediction_horizon_index,
        )
        if prediction_model is not None
        else None
    )
    local_observation = (
        prediction_observer.reset(observation)
        if prediction_observer is not None
        else env.policy_observations(observation)
    )
    actor_hidden = (
        policy.initial_actor_hidden(env.n_defenders, device=device)
        if isinstance(policy, RecurrentCentralizedSharedActorCritic)
        else None
    )
    frames: dict[str, list[np.ndarray]] = defaultdict(list)
    visible_fractions: list[float] = []
    message_ages: list[float] = []
    observation_ages: list[float] = []
    final_info: dict[str, Any] = {}
    with torch.no_grad():
        while True:
            _append_frame(frames, observation, env)
            local = torch.as_tensor(local_observation, device=device)
            if isinstance(policy, RecurrentCentralizedSharedActorCritic):
                distribution, actor_hidden = policy.distribution_step(local, actor_hidden)
            else:
                distribution = policy.distribution(local)
            defender_action = torch.tanh(distribution.mean).cpu().numpy() * action_scale
            if safety_filter is not None:
                defender_action, _ = safety_filter.filter(defender_action, observation)
            observation, _reward, terminated, truncated, final_info = env.step(defender_action)
            visible_fractions.append(float(final_info["target_visible_fraction"]))
            message_ages.append(float(final_info["mean_message_age_steps"]))
            observation_ages.append(float(final_info["mean_observation_age_steps"]))
            if terminated or truncated:
                break
            local_observation = (
                prediction_observer.observe(observation)
                if prediction_observer is not None
                else env.policy_observations(observation)
            )
    row = {
        "seed": int(episode_seed),
        "safe_capture_success": bool(final_info["safe_capture_success"]),
        "capture_event": bool(final_info["capture_event"]),
        "collision": bool(final_info["collision"]),
        "steps": int(env.step_count),
        "termination_reason": str(final_info["termination_reason"]),
        "world_violation_steps": int(final_info["world_violation_steps"]),
        "mean_visible_fraction": float(np.mean(visible_fractions)),
        "mean_message_age_steps": float(np.mean(message_ages)),
        "mean_observation_age_steps": float(np.mean(observation_ages)),
    }
    return row, {name: np.asarray(values) for name, values in frames.items()}


def verify_reproduction(source: dict[str, str], replayed: dict[str, Any]) -> None:
    """Reject a diagnostic run if a frozen P1 failure cannot be reproduced exactly."""
    for field in _REPRODUCTION_FIELDS:
        expected: Any = source[field]
        actual: Any = replayed[field]
        if isinstance(actual, bool):
            expected = parse_bool(str(expected))
        elif isinstance(actual, int):
            expected = int(expected)
        else:
            expected = str(expected)
        if actual != expected:
            raise RuntimeError(
                f"Frozen P1 replay mismatch for seed {source['seed']}, field {field}: "
                f"expected {expected!r}, got {actual!r}."
            )


def _safe_mean(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.mean(values))


def diagnostic_metrics(frames: dict[str, np.ndarray]) -> dict[str, float | int | None]:
    """Compute truth-labeled descriptive metrics from local-belief frame arrays."""
    target_positions = frames["target_positions"][:, None, :]
    target_velocities = frames["target_velocities"][:, None, :]
    position_errors = np.linalg.norm(frames["belief_positions"] - target_positions, axis=-1)
    velocity_errors = np.linalg.norm(frames["belief_velocities"] - target_velocities, axis=-1)
    observation_ages = frames["observation_ages"]
    visible = frames["visible"]
    timestamps = frames["observation_timestamps"]
    initialized = timestamps >= 0
    metrics: dict[str, float | int | None] = {
        "decision_steps": int(position_errors.shape[0]),
        "mean_belief_position_error_m": float(np.mean(position_errors)),
        "p95_belief_position_error_m": float(np.quantile(position_errors, 0.95)),
        "max_belief_position_error_m": float(np.max(position_errors)),
        "mean_belief_velocity_error_mps": float(np.mean(velocity_errors)),
        "mean_observation_age_steps_from_frames": float(np.mean(observation_ages)),
        "max_observation_age_steps": int(np.max(observation_ages)),
        "mean_message_age_steps_from_frames": float(np.mean(frames["message_ages"])),
        "mean_covariance_trace": float(np.mean(frames["covariance_traces"])),
        "uninitialized_belief_frames": int(np.count_nonzero(~initialized)),
        "uninitialized_belief_position_error_m": _safe_mean(position_errors[~initialized]),
        "visibility_reacquisition_events": int(
            np.count_nonzero(visible[1:] & ~visible[:-1]) if visible.shape[0] > 1 else 0
        ),
        "delivered_belief_updates": int(
            np.count_nonzero(timestamps[1:] > timestamps[:-1]) if timestamps.shape[0] > 1 else 0
        ),
    }
    for name, low, high in _AGE_BINS:
        mask = initialized & (observation_ages >= low)
        if high is not None:
            mask &= observation_ages <= high
        metrics[f"{name}_frames"] = int(np.count_nonzero(mask))
        metrics[f"{name}_position_error_m"] = _safe_mean(position_errors[mask])
    return metrics


def summarize_diagnostics(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate episode-weighted and frame-weighted diagnostics by condition."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)
    summaries: dict[str, dict[str, Any]] = {}
    for condition, subset in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "trajectories": len(subset),
            "collision_failures": int(sum(bool(row["collision"]) for row in subset)),
            "timeout_failures": int(sum(str(row["termination_reason"]) == "timeout" for row in subset)),
            "mean_belief_position_error_m": float(np.mean([row["mean_belief_position_error_m"] for row in subset])),
            "mean_p95_belief_position_error_m": float(
                np.mean([row["p95_belief_position_error_m"] for row in subset])
            ),
            "mean_belief_velocity_error_mps": float(
                np.mean([row["mean_belief_velocity_error_mps"] for row in subset])
            ),
            "mean_observation_age_steps": float(
                np.mean([row["mean_observation_age_steps_from_frames"] for row in subset])
            ),
            "mean_max_observation_age_steps": float(np.mean([row["max_observation_age_steps"] for row in subset])),
            "mean_visibility_reacquisition_events": float(
                np.mean([row["visibility_reacquisition_events"] for row in subset])
            ),
        "mean_delivered_belief_updates": float(np.mean([row["delivered_belief_updates"] for row in subset])),
            "uninitialized_belief_frames": int(sum(int(row["uninitialized_belief_frames"]) for row in subset)),
        }
        for name, _low, _high in _AGE_BINS:
            count_key = f"{name}_frames"
            error_key = f"{name}_position_error_m"
            frame_count = int(sum(int(row[count_key]) for row in subset))
            numerator = sum(
                int(row[count_key]) * float(row[error_key])
                for row in subset
                if row[error_key] is not None
            )
            summary[count_key] = frame_count
            summary[error_key] = None if frame_count == 0 else float(numerator / frame_count)
        uninitialized_count = summary["uninitialized_belief_frames"]
        uninitialized_numerator = sum(
            int(row["uninitialized_belief_frames"]) * float(row["uninitialized_belief_position_error_m"])
            for row in subset
            if row["uninitialized_belief_position_error_m"] is not None
        )
        summary["uninitialized_belief_position_error_m"] = (
            None if uninitialized_count == 0 else float(uninitialized_numerator / uninitialized_count)
        )
        summaries[condition] = summary
    return summaries


def format_float(value: float | None, precision: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{precision}f}"


def render_report(
    *,
    conditions: list[str],
    methods: list[str],
    action: str,
    selected: list[dict[str, str]],
    summaries: dict[str, dict[str, Any]],
    output_root: Path,
) -> str:
    lines = [
        "# 阶段 4A 结果：时延与遮挡失败轨迹诊断",
        "",
        "任务：部分可观测三维障碍环境下多无人机协同捕获半径追逃。",
        "",
        "## 1. 目的与边界",
        "",
        "本阶段不训练新策略。它对已冻结的 Stage 3C-P1 `raw` 动作失败回合进行分层选择和确定性重放，",
        "以定位后续时间对齐 belief 应解决的观测问题。目标真值只在重放结束后用于计算评估误差，",
        "从不进入 actor、预测器或 CBF 的输入。",
        "",
        "## 2. 诊断协议",
        "",
        f"- 条件：{', '.join(conditions)}。",
        f"- 冻结方法：{', '.join(methods)}；执行方式：{action}。",
        f"- 每个条件至多选取 20 个 Safe Capture 失败回合，按 `(method, training_seed)` 轮询，",
        "  不按最坏最典型轨迹进行人工挑选。",
        "- 每个候选均使用其原始训练 checkpoint、评估 seed 和 P1 场景配置重放；",
        "  Safe Capture、碰撞、步数、终止原因和越界计数必须与原始 CSV 完全一致。",
        f"- 逐回合表、NPZ 轨迹和协议保存在本地 `{output_root.as_posix()}`。",
        "",
        "## 3. 失败轨迹统计",
        "",
        "| 条件 | 轨迹数 | 碰撞失败 | 超时失败 | belief 位置误差 (m) | p95 位置误差 (m) | 观测年龄 (steps) | 最大年龄 (steps) | 重获观测事件/轨迹 | belief 更新/轨迹 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        summary = summaries[condition]
        lines.append(
            "| {condition} | {trajectories} | {collision_failures} | {timeout_failures} | {error} | {p95} | {age} | {max_age} | {reacquisition} | {updates} |".format(
                condition=condition,
                trajectories=summary["trajectories"],
                collision_failures=summary["collision_failures"],
                timeout_failures=summary["timeout_failures"],
                error=format_float(summary["mean_belief_position_error_m"]),
                p95=format_float(summary["mean_p95_belief_position_error_m"]),
                age=format_float(summary["mean_observation_age_steps"]),
                max_age=format_float(summary["mean_max_observation_age_steps"]),
                reacquisition=format_float(summary["mean_visibility_reacquisition_events"]),
                updates=format_float(summary["mean_delivered_belief_updates"]),
            )
        )
    lines.extend(
        [
            "",
            "## 4. belief 误差与观测年龄",
            "",
            "| 条件 | 未初始化帧 / 误差 (m) | Fresh (0-1) 帧 / 误差 (m) | Moderate (2-4) 帧 / 误差 (m) | Stale (>=5) 帧 / 误差 (m) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition in conditions:
        summary = summaries[condition]
        cells = [
            f"{summary['uninitialized_belief_frames']} / "
            f"{format_float(summary['uninitialized_belief_position_error_m'])}"
        ]
        for name, _low, _high in _AGE_BINS:
            cells.append(f"{summary[f'{name}_frames']} / {format_float(summary[f'{name}_position_error_m'])}")
        lines.append(f"| {condition} | {' | '.join(cells)} |")
    lines.extend(
        [
            "",
            "## 5. 可解释结论",
            "",
            "- 所有入选回合均是原始 P1 中已记录的失败，并且重放一致；因此本报告可用于后续方法设计，",
            "  但不以单条视频代替统计结论。",
            "- 时延条件的 belief 仍携带陈旧时间戳；突发遮挡条件存在持续不可见与再次可见事件。",
            "  尚未收到任何观测的初始 belief 单独统计，已初始化 belief 的位置误差再按观测年龄分桶，",
            "  从而避免将空 belief 误解为低年龄观测。",
            "- 这些指标是描述性证据，不将碰撞单独归因于某一模块。后续 F1 应首先检验时间对齐 belief",
            "  是否降低时延/遮挡域的估计误差和重获观测后的恢复时间，再评估 Safe Capture。",
            "",
            "## 6. 复现",
            "",
            "```powershell",
            "conda run --no-capture-output -n uav-encirclement-gpu python scripts/analyze_stage4a_belief_failures.py --device cpu",
            "```",
            "",
            f"本次共重放 {len(selected)} 条失败轨迹。",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_context(
    method: str,
    training_seed: int,
    condition: str,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], Any, float, HistoryTargetPredictor | None, dict[str, Any]]:
    config, condition_settings = make_config(method, condition)
    experiment = config["experiments"][0]
    prototype = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=int(experiment["obstacle_count"]),
        target_speed_scale=float(experiment["target_speed_scale"]),
    )
    checkpoint = checkpoint_path(method, training_seed).resolve()
    policy, action_scale, checkpoint_metadata = load_policy(
        checkpoint,
        prototype,
        prototype.reset(seed=642001),
        device,
    )
    prediction_args = prediction_args_for(method)
    prediction_checkpoint = prediction_args["prediction_checkpoint"]
    prediction_model = load_prediction_model(prediction_checkpoint, device) if prediction_checkpoint is not None else None
    return config, condition_settings, policy, action_scale, prediction_model, {
        "checkpoint": checkpoint,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "prediction_history_length": int(prediction_args["prediction_history_length"]),
        "prediction_horizon_index": int(prediction_args["prediction_horizon_index"]),
        "actor_recurrent": bool(checkpoint_metadata.get("actor_recurrent", False)),
    }


def main() -> None:
    args = parse_args()
    if args.max_failures_per_condition <= 0:
        raise ValueError("--max-failures-per-condition must be positive.")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {args.output_root}")
    if args.output_report.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {args.output_report}")
    source_rows = read_failed_rows(args.source_root, args.conditions, args.methods, args.action)
    selected_by_condition = {
        condition: select_stratified_failures(rows, args.max_failures_per_condition)
        for condition, rows in source_rows.items()
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    protocol = {
        "stage": "4A_delay_and_occlusion_belief_failure_diagnostics",
        "source_root": str(args.source_root.resolve()),
        "conditions": list(args.conditions),
        "methods": list(args.methods),
        "action": args.action,
        "max_failures_per_condition": int(args.max_failures_per_condition),
        "selection": "round_robin_by_method_and_training_seed_from_failed_P1_rows",
        "reproduction_fields": list(_REPRODUCTION_FIELDS),
        "device": str(device),
        "ground_truth_usage": "offline_diagnostic_labels_only",
    }
    args.output_root.joinpath("protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    contexts: dict[tuple[str, int, str], tuple[Any, ...]] = {}
    diagnostic_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, str]] = []
    for condition in args.conditions:
        for source in selected_by_condition[condition]:
            method = str(source["method"])
            training_seed = int(source["training_seed"])
            context_key = (method, training_seed, condition)
            if context_key not in contexts:
                contexts[context_key] = _load_context(method, training_seed, condition, device)
            config, condition_settings, policy, action_scale, prediction_model, metadata = contexts[context_key]
            replayed, frames = replay_with_diagnostics(
                policy=policy,
                config=config,
                obstacle_count=int(condition_settings["obstacle_count"]),
                target_speed_scale=float(condition_settings["target_speed_scale"]),
                episode_seed=int(source["seed"]),
                device=device,
                action_scale=float(action_scale),
                action=args.action,
                prediction_model=prediction_model,
                prediction_history_length=int(metadata["prediction_history_length"]),
                prediction_horizon_index=int(metadata["prediction_horizon_index"]),
            )
            verify_reproduction(source, replayed)
            metrics = diagnostic_metrics(frames)
            trajectory_name = f"{condition}_{method}_train{training_seed}_episode{source['seed']}.npz"
            np.savez_compressed(args.output_root / trajectory_name, **frames)
            diagnostic_row = {
                "condition": condition,
                "method": method,
                "training_seed": training_seed,
                "episode_seed": int(source["seed"]),
                "trajectory": trajectory_name,
                "checkpoint_sha256": str(metadata["checkpoint_sha256"]),
                **replayed,
                **metrics,
            }
            diagnostic_rows.append(diagnostic_row)
            selected_rows.append(dict(source))

    selected_path = args.output_root / "selected_failures.csv"
    with selected_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    diagnostics_path = args.output_root / "diagnostics.csv"
    with diagnostics_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostic_rows[0]))
        writer.writeheader()
        writer.writerows(diagnostic_rows)
    summaries = summarize_diagnostics(diagnostic_rows)
    args.output_root.joinpath("summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        render_report(
            conditions=list(args.conditions),
            methods=list(args.methods),
            action=args.action,
            selected=selected_rows,
            summaries=summaries,
            output_root=args.output_root.relative_to(PROJECT_ROOT),
        ),
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
