"""Select an age-gated belief-velocity decay on a validation seed block."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_stage4b_belief_baselines import (  # noqa: E402
    CONDITIONS,
    DEFAULT_CONDITIONS,
    make_config,
    rollout_estimator_episode,
    safe_mean,
)


CANDIDATES = {
    "aligned_decay_0_00": 0.00,
    "aligned_decay_0_50": 0.50,
    "aligned_decay_0_80": 0.80,
    "aligned_decay_1_00": 1.00,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS), default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--validation-seed", type=int, default=641001)
    parser.add_argument("--decay-start-age-steps", type=int, default=3)
    parser.add_argument("--maximum-update-error-regression", type=float, default=0.10)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "stage4b_velocity_gate_validation")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "results" / "STAGE4B_VELOCITY_GATE_VALIDATION_SUMMARY.json",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=PROJECT_ROOT / "results" / "STAGE4B_VELOCITY_GATE_VALIDATION_REPORT.md",
    )
    return parser.parse_args()


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), str(row["candidate"]))].append(row)
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for (condition, candidate), subset in sorted(grouped.items()):
        summary.setdefault(condition, {})[candidate] = {
            "episodes": float(len(subset)),
            "mean_position_error_m_initialized": float(
                np.mean([float(item["mean_position_error_m_initialized"]) for item in subset])
            ),
            "mean_new_timestamp_position_error_m": float(
                np.mean([float(item["mean_new_timestamp_position_error_m"]) for item in subset])
            ),
            "mean_velocity_error_mps_initialized": float(
                np.mean([float(item["mean_velocity_error_mps_initialized"]) for item in subset])
            ),
        }
    return summary


def select_candidate(
    summary: dict[str, dict[str, dict[str, float]]],
    conditions: list[str],
    maximum_update_error_regression: float,
) -> tuple[str, dict[str, Any]]:
    """Choose the lowest validation error that retains packet-arrival accuracy."""
    if not 0.0 <= maximum_update_error_regression < 1.0:
        raise ValueError("maximum_update_error_regression must be in [0, 1).")
    reference = "aligned_decay_1_00"
    candidates: list[dict[str, Any]] = []
    for candidate, decay in CANDIDATES.items():
        condition_rows = [summary[condition][candidate] for condition in conditions]
        eligible = all(
            row["mean_new_timestamp_position_error_m"]
            <= summary[condition][reference]["mean_new_timestamp_position_error_m"] * (1.0 + maximum_update_error_regression)
            for condition, row in zip(conditions, condition_rows, strict=True)
        )
        candidates.append(
            {
                "candidate": candidate,
                "decay": decay,
                "eligible": eligible,
                "validation_mean_initialized_position_error_m": float(
                    np.mean([row["mean_position_error_m_initialized"] for row in condition_rows])
                ),
            }
        )
    eligible = [row for row in candidates if bool(row["eligible"])]
    if not eligible:
        raise RuntimeError("No velocity-gate candidate satisfied the packet-arrival error constraint.")
    selected = min(eligible, key=lambda row: (float(row["validation_mean_initialized_position_error_m"]), str(row["candidate"])))
    return str(selected["candidate"]), {"candidates": candidates, "selected": selected}


def report(
    conditions: list[str],
    episodes: int,
    validation_seed: int,
    start_age: int,
    selection: dict[str, Any],
    summary: dict[str, dict[str, dict[str, float]]],
) -> str:
    lines = [
        "# 阶段 4B 结果：年龄门控速度衰减 Validation 选择",
        "",
        f"- Validation seed block：{validation_seed} 起，共 {episodes} 个 episode/条件/候选。",
        f"- 所有候选均使用 `time_aligned`，当观测年龄达到 {start_age} 步后对速度乘以候选衰减系数。",
        "- 选择规则：每个条件的新时间戳更新误差不比 `aligned_decay_1_00` 高 10%，",
        "  再选择平均已初始化位置误差最低的候选。该选择不读取 P1 locked-test seed。",
        "",
        "| 条件 | 候选 | 已初始化位置误差 (m) | 新时间戳更新误差 (m) |",
        "|---|---|---:|---:|",
    ]
    for condition in conditions:
        for candidate in CANDIDATES:
            row = summary[condition][candidate]
            lines.append(
                f"| {condition} | {candidate} | {row['mean_position_error_m_initialized']:.3f} | "
                f"{row['mean_new_timestamp_position_error_m']:.3f} |"
            )
    lines.extend(
        [
            "",
            f"选择：`{selection['selected']['candidate']}`，衰减系数 {selection['selected']['decay']:.2f}。",
            "此结果仅允许进入新的 locked-test estimator 复验；尚不构成策略训练或 Safe Capture 改进结论。",
            "",
            "```powershell",
            "conda run --no-capture-output -n uav-encirclement-gpu python scripts/select_stage4b_velocity_gate.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.decay_start_age_steps < 0:
        raise ValueError("episodes must be positive and decay_start_age_steps must be non-negative.")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {args.output_root}")
    if args.output_json.exists() or args.output_report.exists():
        raise FileExistsError("Refusing to overwrite an existing validation report.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for condition in args.conditions:
        for candidate, decay in CANDIDATES.items():
            config, settings = make_config(condition)
            config = copy.deepcopy(config)
            config["task"]["pursuit"].update(
                {
                    "belief_update_mode": "time_aligned",
                    "belief_stale_velocity_decay": decay,
                    "belief_velocity_decay_start_age_steps": int(args.decay_start_age_steps),
                }
            )
            for episode_index in range(args.episodes):
                row = rollout_estimator_episode(
                    config,
                    obstacle_count=int(settings["obstacle_count"]),
                    target_speed_scale=float(settings["target_speed_scale"]),
                    seed=int(args.validation_seed) + episode_index,
                )
                row.update({"condition": condition, "candidate": candidate, "decay": decay})
                rows.append(row)
    with (args.output_root / "episodes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    selected, selection = select_candidate(summary, list(args.conditions), float(args.maximum_update_error_regression))
    result = {
        "stage": "4B_velocity_gate_validation_selection",
        "conditions": list(args.conditions),
        "episodes_per_condition": int(args.episodes),
        "validation_seed": int(args.validation_seed),
        "decay_start_age_steps": int(args.decay_start_age_steps),
        "maximum_update_error_regression": float(args.maximum_update_error_regression),
        "candidates": CANDIDATES,
        "selected_candidate": selected,
        "selection": selection,
        "results": summary,
    }
    args.output_root.joinpath("protocol.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.output_report.write_text(
        report(list(args.conditions), int(args.episodes), int(args.validation_seed), int(args.decay_start_age_steps), selection, summary),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
