"""Compare local target-belief update rules on frozen P1 observation domains.

This is an estimator-only diagnostic. Each mode sees the same target motion,
obstacles, local detections, timestamped packets, and episode seeds. Defender
actions are fixed to zero, so policy behavior cannot alter the comparison.
Simulator target truth is read only after a local observation has been emitted
to calculate offline estimator labels.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402


BASE_CONFIG = PROJECT_ROOT / "configs" / "capture_radius_pursuit_dev.yaml"
CONDITIONS: dict[str, dict[str, Any]] = {
    "nominal_partial_observation": {
        "obstacle_count": 3,
        "target_speed_scale": 0.75,
        "pursuit": {
            "target_motion_mode": "random_turn",
            "obstacle_profile": "mixed",
            "map_seed_offset": 500000,
        },
    },
    "delayed_measurements": {
        "obstacle_count": 3,
        "target_speed_scale": 0.75,
        "pursuit": {
            "target_motion_mode": "random_turn",
            "obstacle_profile": "mixed",
            "observation_delay_steps": 3,
            "message_delay_steps": 5,
            "message_dropout_probability": 0.10,
            "map_seed_offset": 510000,
        },
    },
    "burst_occlusion": {
        "obstacle_count": 5,
        "target_speed_scale": 1.00,
        "pursuit": {
            "target_motion_mode": "s_curve",
            "obstacle_profile": "boxes",
            "detection_dropout_probability": 0.25,
            "detection_loss_burst_probability": 0.20,
            "detection_loss_burst_duration_steps": 5,
            "observation_delay_steps": 2,
            "map_seed_offset": 520000,
        },
    },
    "communication_loss": {
        "obstacle_count": 5,
        "target_speed_scale": 1.00,
        "pursuit": {
            "target_motion_mode": "burst",
            "obstacle_profile": "narrow_channels",
            "target_burst_period_steps": 30,
            "target_burst_duration_steps": 8,
            "message_delay_steps": 6,
            "message_dropout_probability": 0.20,
            "communication_link_dropout_probability": 0.15,
            "map_seed_offset": 530000,
        },
    },
}


def make_config(condition_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    condition = CONDITIONS[condition_name]
    config = copy.deepcopy(config)
    config["task"]["pursuit"].update(copy.deepcopy(condition["pursuit"]))
    return config, condition


MODES = ("legacy", "zero_velocity", "constant_velocity", "time_aligned")
DEFAULT_CONDITIONS = ("delayed_measurements", "burst_occlusion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS), default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--test-seed", type=int, default=642001)
    parser.add_argument("--time-aligned-velocity-decay", type=float, default=1.0)
    parser.add_argument("--time-aligned-decay-start-age-steps", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "stage4b_belief_baselines",
        help="New local directory for per-episode estimator diagnostics.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "results" / "STAGE4B_BELIEF_BASELINE_SUMMARY.json",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=PROJECT_ROOT / "results" / "STAGE4B_BELIEF_BASELINE_REPORT.md",
    )
    return parser.parse_args()


def safe_mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return None if not materialized else float(np.mean(materialized))


def age_binned_error(errors: np.ndarray, ages: np.ndarray, timestamps: np.ndarray) -> dict[str, float | int | None]:
    """Summarize error only where a local belief has received a measurement."""
    if errors.shape != ages.shape or errors.shape != timestamps.shape:
        raise ValueError("belief errors, ages, and timestamps must have identical shape.")
    initialized = timestamps >= 0
    result: dict[str, float | int | None] = {
        "initialized_frames": int(np.count_nonzero(initialized)),
        "uninitialized_frames": int(np.count_nonzero(~initialized)),
        "initialized_position_error_m": safe_mean(errors[initialized].tolist()),
    }
    for name, low, high in (("fresh_0_1", 0, 1), ("moderate_2_4", 2, 4), ("stale_5_plus", 5, None)):
        mask = initialized & (ages >= low)
        if high is not None:
            mask &= ages <= high
        result[f"{name}_frames"] = int(np.count_nonzero(mask))
        result[f"{name}_position_error_m"] = safe_mean(errors[mask].tolist())
    return result


def rollout_estimator_episode(
    config: dict[str, Any],
    obstacle_count: int,
    target_speed_scale: float,
    seed: int,
) -> dict[str, float | int | None]:
    """Collect offline truth labels while all executed actions are fixed to zero."""
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=obstacle_count, target_speed_scale=target_speed_scale)
    observation = env.reset(seed=seed)
    position_errors: list[np.ndarray] = []
    velocity_errors: list[np.ndarray] = []
    ages: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    update_errors: list[float] = []
    recovery_lags: list[int] = []
    previous_visible: np.ndarray | None = None
    previous_timestamps: np.ndarray | None = None
    pending_reacquisition = np.full(env.n_defenders, -1, dtype=np.int64)
    while True:
        belief_positions = np.asarray(observation["target_belief_positions"], dtype=np.float64)
        belief_velocities = np.asarray(observation["target_belief_velocities"], dtype=np.float64)
        visible = np.asarray(observation["target_visible"], dtype=bool)
        current_ages = np.asarray(observation["target_observation_age_steps"], dtype=np.int64)
        current_timestamps = np.asarray(observation["target_observation_timestamps"], dtype=np.int64)
        position_error = np.linalg.norm(belief_positions - env.target_position[None, :], axis=1)
        velocity_error = np.linalg.norm(belief_velocities - env.target_velocity[None, :], axis=1)
        position_errors.append(position_error)
        velocity_errors.append(velocity_error)
        ages.append(current_ages)
        timestamps.append(current_timestamps)
        if previous_visible is not None and previous_timestamps is not None:
            newly_visible = visible & ~previous_visible
            pending_reacquisition[newly_visible] = env.step_count
            updated = current_timestamps > previous_timestamps
            update_errors.extend(float(value) for value in position_error[updated])
            recovered = updated & (pending_reacquisition >= 0)
            recovery_lags.extend(int(env.step_count - started) for started in pending_reacquisition[recovered])
            pending_reacquisition[recovered] = -1
        previous_visible = visible
        previous_timestamps = current_timestamps
        observation, _reward, terminated, truncated, _info = env.step(np.zeros((env.n_defenders, 3), dtype=np.float64))
        if terminated or truncated:
            break
    position_error_array = np.asarray(position_errors)
    velocity_error_array = np.asarray(velocity_errors)
    age_array = np.asarray(ages)
    timestamp_array = np.asarray(timestamps)
    binned = age_binned_error(position_error_array, age_array, timestamp_array)
    initialized = timestamp_array >= 0
    return {
        "seed": int(seed),
        "steps": int(env.step_count),
        "mean_position_error_m_all": float(np.mean(position_error_array)),
        "mean_velocity_error_mps_all": float(np.mean(velocity_error_array)),
        "mean_position_error_m_initialized": safe_mean(position_error_array[initialized].tolist()),
        "mean_velocity_error_mps_initialized": safe_mean(velocity_error_array[initialized].tolist()),
        "mean_observation_age_steps": float(np.mean(age_array)),
        "mean_new_timestamp_position_error_m": safe_mean(update_errors),
        "mean_reacquisition_to_update_steps": safe_mean(recovery_lags),
        "reacquisition_events_recovered": len(recovery_lags),
        **binned,
    }


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), str(row["mode"]))].append(row)
    summary: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for (condition, mode), subset in sorted(grouped.items()):
        row: dict[str, float | int | None] = {"episodes": len(subset)}
        for metric in (
            "mean_position_error_m_all",
            "mean_velocity_error_mps_all",
            "mean_position_error_m_initialized",
            "mean_velocity_error_mps_initialized",
            "mean_observation_age_steps",
            "mean_new_timestamp_position_error_m",
            "mean_reacquisition_to_update_steps",
            "initialized_frames",
            "uninitialized_frames",
            "fresh_0_1_frames",
            "fresh_0_1_position_error_m",
            "moderate_2_4_frames",
            "moderate_2_4_position_error_m",
            "stale_5_plus_frames",
            "stale_5_plus_position_error_m",
        ):
            values = [value for item in subset if (value := item[metric]) is not None]
            row[metric] = safe_mean(float(value) for value in values)
        row["reacquisition_events_recovered"] = int(sum(int(item["reacquisition_events_recovered"]) for item in subset))
        summary.setdefault(condition, {})[mode] = row
    return summary


def format_float(value: float | int | None, precision: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{precision}f}"


def render_report(
    conditions: list[str],
    modes: list[str],
    episodes: int,
    summary: dict[str, dict[str, dict[str, float | int | None]]],
    time_aligned_velocity_decay: float,
    time_aligned_decay_start_age_steps: int,
) -> str:
    lines = [
        "# 阶段 4B 结果：目标 belief 更新规则基线比较",
        "",
        "任务：部分可观测三维障碍环境下多无人机协同捕获半径追逃。",
        "",
        "## 1. 协议",
        "",
        "- 本实验只比较 estimator，不训练或执行学习策略；4 架防守机的动作固定为零。",
        "- 每种模式在每个条件上使用完全相同的障碍、目标轨迹、检测/消息随机数和 episode seed。",
        "- `legacy` 保持既有 P1 语义；`zero_velocity` 和 `constant_velocity` 是可解释基线；",
        "  `time_aligned` 在测量时间戳处融合保存的本地 belief，再传播到当前时间。",
        f"- 本次 `time_aligned` 的陈旧速度衰减设置：系数 {time_aligned_velocity_decay:.2f}，",
        f"  起始观测年龄 {time_aligned_decay_start_age_steps} 步。",
        "- 目标真值只用于 rollout 后的误差标签，不是任何模式的输入。",
        f"- 条件：{', '.join(conditions)}；每个条件每种模式：{episodes} 个锁定 episode。",
        "",
        "## 2. 已初始化 belief 的主要误差",
        "",
        "| 条件 | 模式 | 位置误差 (m) | 速度误差 (m/s) | 新时间戳更新误差 (m) | 重获至更新 (steps) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        for mode in modes:
            row = summary[condition][mode]
            lines.append(
                f"| {condition} | {mode} | {format_float(row['mean_position_error_m_initialized'])} | "
                f"{format_float(row['mean_velocity_error_mps_initialized'])} | "
                f"{format_float(row['mean_new_timestamp_position_error_m'])} | "
                f"{format_float(row['mean_reacquisition_to_update_steps'])} |"
            )
    lines.extend(
        [
            "",
            "## 3. 观测年龄分桶",
            "",
            "| 条件 | 模式 | Fresh (0-1) 误差 (m) | Moderate (2-4) 误差 (m) | Stale (>=5) 误差 (m) | 未初始化帧 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for condition in conditions:
        for mode in modes:
            row = summary[condition][mode]
            lines.append(
                f"| {condition} | {mode} | {format_float(row['fresh_0_1_position_error_m'])} | "
                f"{format_float(row['moderate_2_4_position_error_m'])} | "
                f"{format_float(row['stale_5_plus_position_error_m'])} | "
                f"{int(float(row['uninitialized_frames']))} |"
            )
    lines.extend(
        [
            "",
            "## 4. 解释边界",
            "",
            "本结果只验证 belief 更新规则在冻结仿真观测过程中的估计误差，不是 Safe Capture 改进声明。",
            "包到达时的误差和长陈旧观测误差必须同时评估；若时间对齐只改善前者，",
            "不能据此启动 F1/F2 训练。后续候选应先在独立 validation seed 上选择，",
            "再以新的 locked-test seed 复验估计器收益。",
            "",
            "## 5. 复现",
            "",
            "```powershell",
            "conda run --no-capture-output -n uav-encirclement-gpu python scripts/evaluate_stage4b_belief_baselines.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if not 0.0 <= args.time_aligned_velocity_decay <= 1.0:
        raise ValueError("--time-aligned-velocity-decay must be in [0, 1].")
    if args.time_aligned_decay_start_age_steps < 0:
        raise ValueError("--time-aligned-decay-start-age-steps must be non-negative.")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {args.output_root}")
    if args.output_json.exists() or args.output_report.exists():
        raise FileExistsError("Refusing to overwrite an existing Stage 4B summary or report.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for condition in args.conditions:
        for mode in args.modes:
            config, settings = make_config(condition)
            config["task"]["pursuit"]["belief_update_mode"] = mode
            if mode == "time_aligned":
                config["task"]["pursuit"].update(
                    {
                        "belief_stale_velocity_decay": float(args.time_aligned_velocity_decay),
                        "belief_velocity_decay_start_age_steps": int(args.time_aligned_decay_start_age_steps),
                    }
                )
            for episode_index in range(args.episodes):
                row = rollout_estimator_episode(
                    config,
                    obstacle_count=int(settings["obstacle_count"]),
                    target_speed_scale=float(settings["target_speed_scale"]),
                    seed=int(args.test_seed) + episode_index,
                )
                row.update({"condition": condition, "mode": mode})
                rows.append(row)
    with (args.output_root / "episodes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    args.output_root.joinpath("protocol.json").write_text(
        json.dumps(
            {
                "stage": "4B_belief_update_baseline_comparison",
                "conditions": list(args.conditions),
                "modes": list(args.modes),
                "episodes_per_condition": int(args.episodes),
                "test_seed": int(args.test_seed),
                "defender_actions": "fixed_zero",
                "truth_usage": "offline_estimator_error_labels_only",
                "time_aligned_velocity_decay": float(args.time_aligned_velocity_decay),
                "time_aligned_decay_start_age_steps": int(args.time_aligned_decay_start_age_steps),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.output_report.write_text(
        render_report(
            list(args.conditions),
            list(args.modes),
            int(args.episodes),
            summary,
            float(args.time_aligned_velocity_decay),
            int(args.time_aligned_decay_start_age_steps),
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
