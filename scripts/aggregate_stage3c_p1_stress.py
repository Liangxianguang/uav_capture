"""Aggregate Stage 3C representative partial-observation stress tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("recurrent_no_prediction", "recurrent_gru_prediction")
ACTIONS = ("raw", "cbf")
METRICS = (
    "safe_capture_rate",
    "capture_rate",
    "collision_rate",
    "world_violation_rate",
    "mean_capture_time_seconds",
    "mean_min_clearance_m",
    "mean_visible_fraction",
    "mean_message_age_steps",
    "mean_observation_age_steps",
)
T_CRITICAL_95 = {2: 12.7062047364, 3: 4.30265272975, 4: 3.18244630528, 5: 2.7764451052}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "results" / "stage3c_p1_stress")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "results" / "RECURRENT_POLICY_STAGE3C_P1_STRESS_SUMMARY.json",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=PROJECT_ROOT / "results" / "RECURRENT_POLICY_STAGE3C_P1_STRESS_REPORT.md",
    )
    return parser.parse_args()


def t_critical_95(count: int) -> float:
    return T_CRITICAL_95.get(count, 1.95996398454)


def statistics(values: list[float]) -> dict[str, float | int | list[float]]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.size == 0:
        raise ValueError("Cannot summarize an empty vector.")
    sample_std = float(np.std(vector, ddof=1)) if vector.size > 1 else 0.0
    half_width = (
        float(t_critical_95(int(vector.size)) * sample_std / math.sqrt(vector.size))
        if vector.size > 1
        else 0.0
    )
    return {
        "count": int(vector.size),
        "mean": float(np.mean(vector)),
        "sample_std": sample_std,
        "ci95_half_width": half_width,
        "minimum": float(np.min(vector)),
        "maximum": float(np.max(vector)),
        "values": [float(value) for value in vector],
    }


def format_percent(summary: dict[str, Any]) -> str:
    return f"{100.0 * float(summary['mean']):.2f}% +/- {100.0 * float(summary['ci95_half_width']):.2f}%"


def format_float(summary: dict[str, Any], digits: int = 3) -> str:
    return f"{float(summary['mean']):.{digits}f} +/- {float(summary['ci95_half_width']):.{digits}f}"


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return resolved.as_posix()


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing stress summary: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "overall" not in document:
        raise ValueError(f"Malformed stress summary: {path}")
    row = document["overall"]
    if any(metric not in row for metric in METRICS):
        raise ValueError(f"Missing required metric in {path}")
    return document


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol_path = root / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing stress protocol: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    methods = tuple(protocol.get("methods", []))
    actions = tuple(protocol.get("actions", []))
    seeds = [int(seed) for seed in protocol.get("training_seeds", [])]
    conditions = list(protocol.get("conditions", {}))
    if methods != METHODS:
        raise ValueError(f"Expected methods {METHODS}, found {methods}")
    if actions != ACTIONS:
        raise ValueError(f"Expected actions {ACTIONS}, found {actions}")
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("Stress aggregation requires at least three distinct training seeds.")
    if not conditions:
        raise ValueError("Stress protocol contains no conditions.")

    source_hashes: dict[str, str] = {}
    raw: dict[str, dict[str, dict[str, dict[int, dict[str, Any]]]]] = {
        method: {action: {condition: {} for condition in conditions} for action in ACTIONS} for method in METHODS
    }
    for method in METHODS:
        for action in ACTIONS:
            for condition in conditions:
                for seed in seeds:
                    path = root / method / f"seed{seed}" / condition / f"summary_{action}.json"
                    raw[method][action][condition][seed] = load_summary(path)
                    source_hashes[relative_path(path)] = source_hash(path)

    aggregate: dict[str, Any] = {
        "stage": "3C_P1_representative_partial_observation_stress",
        "statistical_unit": "training_seed",
        "confidence_interval": "two-sided 95 percent Student-t interval across independent training seeds",
        "protocol": protocol,
        "source_summary_sha256": source_hashes,
        "results": {},
    }
    for method in METHODS:
        aggregate["results"][method] = {}
        for action in ACTIONS:
            aggregate["results"][method][action] = {}
            for condition in conditions:
                rows = [raw[method][action][condition][seed]["overall"] for seed in seeds]
                counts = {int(row["episodes"]) for row in rows}
                if len(counts) != 1:
                    raise ValueError(f"Episode count differs by seed: {method}/{action}/{condition}")
                aggregate["results"][method][action][condition] = {
                    "episodes_per_seed": counts.pop(),
                    "episodes_total": int(sum(int(row["episodes"]) for row in rows)),
                    "metrics": {metric: statistics([float(row[metric]) for row in rows]) for metric in METRICS},
                }

    paired: dict[str, Any] = {}
    for condition in conditions:
        for action in ACTIONS:
            values = [
                100.0
                * (
                    float(raw["recurrent_gru_prediction"][action][condition][seed]["overall"]["safe_capture_rate"])
                    - float(raw["recurrent_no_prediction"][action][condition][seed]["overall"]["safe_capture_rate"])
                )
                for seed in seeds
            ]
            key = f"gru_minus_no_prediction_{action}_safe_capture_percentage_points__{condition}"
            paired[key] = {
                **statistics(values),
                "positive_seed_count": int(sum(value > 0.0 for value in values)),
            }
    aggregate["paired_comparisons"] = paired

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    lines = [
        "# 阶段 3C-P1 结果：部分可观测压力测试",
        "",
        "任务：部分可观测障碍环境下多无人机三维捕获半径追逃",
        "后端：kinematic 3D simulation",
        "统计单位：独立训练 seed；压力测试只改变已实现的仿真观测、通信、目标运动和障碍参数。",
        "",
        "## 1. 协议",
        "",
        f"- 训练 seed：{', '.join(str(seed) for seed in seeds)}；locked-test seed：{int(protocol['test_seed'])}。",
        f"- 每个条件每个 seed：{int(protocol['episodes_per_condition'])} 回合；每个方法/执行方式/条件共 {int(protocol['episodes_per_condition']) * len(seeds)} 回合。",
        "- 方法：D Recurrent-MAPPO（无学习式预测）；E Recurrent-MAPPO + 冻结 GRU 预测。",
        "- raw 与 +CBF 分开报告；CI 在训练 seed 间计算，不把 episode 计作独立训练重复。",
        "",
        "## 2. raw action 总体结果",
        "",
        "| 条件 | D Safe Capture | E Safe Capture | D Collision | E Collision | D Obs Age | E Obs Age | E-D Safe Capture (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        d = aggregate["results"]["recurrent_no_prediction"]["raw"][condition]["metrics"]
        e = aggregate["results"]["recurrent_gru_prediction"]["raw"][condition]["metrics"]
        delta = aggregate["paired_comparisons"][f"gru_minus_no_prediction_raw_safe_capture_percentage_points__{condition}"]
        lines.append(
            f"| {condition} | {format_percent(d['safe_capture_rate'])} | {format_percent(e['safe_capture_rate'])} | "
            f"{format_percent(d['collision_rate'])} | {format_percent(e['collision_rate'])} | "
            f"{format_float(d['mean_observation_age_steps'])} | {format_float(e['mean_observation_age_steps'])} | "
            f"{format_float(delta, 2)} |"
        )
    lines.extend(["", "## 3. CBF action 结果", "", "| 条件 | D Safe Capture | E Safe Capture | D Collision | E Collision |", "|---|---:|---:|---:|---:|"])
    for condition in conditions:
        d = aggregate["results"]["recurrent_no_prediction"]["cbf"][condition]["metrics"]
        e = aggregate["results"]["recurrent_gru_prediction"]["cbf"][condition]["metrics"]
        lines.append(
            f"| {condition} | {format_percent(d['safe_capture_rate'])} | {format_percent(e['safe_capture_rate'])} | "
            f"{format_percent(d['collision_rate'])} | {format_percent(e['collision_rate'])} |"
        )

    lines.extend(["", "## 4. 解释边界", ""])
    for condition in conditions:
        delta = aggregate["paired_comparisons"][f"gru_minus_no_prediction_raw_safe_capture_percentage_points__{condition}"]
        lines.append(
            f"- `{condition}`：GRU 相对 D 的 raw Safe Capture 差值为 {format_float(delta, 2)} 个百分点，"
            f"{int(delta['positive_seed_count'])}/{len(seeds)} 个 seed 为正。"
        )
    lines.extend(
        [
            "",
            "这些压力测试用于刻画收益和失败边界，不构成真实传感器数据验证。若 CI 跨 0，结论只能写成方向性趋势；",
            "若 CBF 结果饱和，则安全收益仍归因于独立安全层，不能归因于预测或循环记忆。",
            "",
            "## 5. 复现证据",
            "",
            "- 运行器：`scripts/run_stage3c_p1_stress.py`",
            "- 聚合器：`scripts/aggregate_stage3c_p1_stress.py`",
            "- 结构化结果：`results/RECURRENT_POLICY_STAGE3C_P1_STRESS_SUMMARY.json`",
            "- 本地逐回合结果、配置和 checkpoint 元数据：`results/stage3c_p1_stress/`。",
            "",
            "结论仅适用于当前运动学三维仿真，不等同于实体捕获、真实感知闭环、SITL 或实飞验证。",
        ]
    )
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_report": str(args.output_report)}, indent=2))


if __name__ == "__main__":
    main()
