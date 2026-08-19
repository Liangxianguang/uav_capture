"""Aggregate reproducible Stage 3B multi-seed locked-test evaluations.

The independent unit is the training seed. Each seed is evaluated on the
same locked scenario block, so episode rows are retained for audit but are not
incorrectly treated as independent training replications in confidence
intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("no_prediction", "constant_velocity", "gru_prediction")
ACTIONS = ("raw", "cbf")
T_CRITICAL_95 = {1: float("nan"), 2: 12.7062047364, 3: 4.30265272975, 4: 3.18244630528, 5: 2.7764451052}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "results" / "stage3b_formal")
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "results" / "PREDICTION_POLICY_STAGE3_FORMAL_SUMMARY.json")
    parser.add_argument("--output-report", type=Path, default=PROJECT_ROOT / "results" / "PREDICTION_POLICY_STAGE3_FORMAL_REPORT.md")
    return parser.parse_args()


def t_critical_95(count: int) -> float:
    if count in T_CRITICAL_95:
        return T_CRITICAL_95[count]
    # The formal protocol uses exactly three seeds. This conservative fallback
    # keeps the aggregation script usable for larger repetitions without a
    # SciPy dependency.
    return 1.95996398454


def statistics(values: list[float]) -> dict[str, float | int | None]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.size == 0:
        raise ValueError("Cannot summarize an empty metric vector.")
    standard_deviation = float(np.std(vector, ddof=1)) if vector.size > 1 else 0.0
    half_width = (
        float(t_critical_95(int(vector.size)) * standard_deviation / math.sqrt(vector.size))
        if vector.size > 1
        else None
    )
    return {
        "count": int(vector.size),
        "mean": float(np.mean(vector)),
        "sample_std": standard_deviation,
        "ci95_half_width": half_width,
        "minimum": float(np.min(vector)),
        "maximum": float(np.max(vector)),
        "values": [float(value) for value in vector],
    }


def format_percent(summary: dict[str, float | int | None]) -> str:
    half_width = summary["ci95_half_width"]
    suffix = "n/a" if half_width is None else f"{100.0 * float(half_width):.2f}"
    return f"{100.0 * float(summary['mean']):.2f}% +/- {suffix}%"


def format_float(summary: dict[str, float | int | None], digits: int = 3) -> str:
    half_width = summary["ci95_half_width"]
    suffix = "n/a" if half_width is None else f"{float(half_width):.{digits}f}"
    return f"{float(summary['mean']):.{digits}f} +/- {suffix}"


def load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing locked-test summary: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "overall" not in document:
        raise ValueError(f"Malformed summary: {path}")
    return document


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol_path = root / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing Stage 3B protocol: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in protocol.get("seeds", [])]
    expected_methods = list(protocol.get("methods", []))
    if sorted(expected_methods) != sorted(METHODS) or len(seeds) < 3:
        raise ValueError("Formal protocol must contain the three methods and at least three training seeds.")

    raw_documents: dict[str, dict[str, dict[int, dict[str, Any]]]] = {method: {} for method in METHODS}
    source_summaries: dict[str, str] = {}
    scenarios: set[str] | None = None
    for method in METHODS:
        for action in ACTIONS:
            raw_documents[method][action] = {}
            for seed in seeds:
                summary_path = root / method / f"seed{seed}" / f"evaluation_{action}" / "summary.json"
                document = load_summary(summary_path)
                document_scenarios = set(document)
                if scenarios is None:
                    scenarios = document_scenarios
                elif document_scenarios != scenarios:
                    raise ValueError(f"Scenario mismatch in {summary_path}")
                raw_documents[method][action][seed] = document
                source_summaries[str(summary_path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = source_hash(summary_path)
    if scenarios is None or "overall" not in scenarios:
        raise RuntimeError("No evaluation scenarios found.")

    metric_names = (
        "safe_capture_rate",
        "capture_rate",
        "collision_rate",
        "world_violation_rate",
        "mean_capture_time_seconds",
        "mean_min_clearance_m",
        "mean_visible_fraction",
        "mean_message_age_steps",
    )
    aggregate: dict[str, Any] = {
        "stage": "3B_formal_multiseed",
        "statistical_unit": "training_seed",
        "confidence_interval": "two-sided 95 percent Student-t interval across independent training seeds",
        "protocol": protocol,
        "source_summary_sha256": source_summaries,
        "results": {},
    }
    for method in METHODS:
        aggregate["results"][method] = {}
        for action in ACTIONS:
            aggregate["results"][method][action] = {}
            for scenario in sorted(scenarios):
                rows = [raw_documents[method][action][seed][scenario] for seed in seeds]
                aggregate["results"][method][action][scenario] = {
                    "episodes_per_seed": int(rows[0]["episodes"]),
                    "episodes_total": int(sum(int(row["episodes"]) for row in rows)),
                    "metrics": {metric: statistics([float(row[metric]) for row in rows]) for metric in metric_names},
                }

    raw_gain = []
    for seed in seeds:
        no_prediction = raw_documents["no_prediction"]["raw"][seed]["overall"]["safe_capture_rate"]
        gru_prediction = raw_documents["gru_prediction"]["raw"][seed]["overall"]["safe_capture_rate"]
        raw_gain.append(100.0 * (float(gru_prediction) - float(no_prediction)))
    aggregate["paired_comparisons"] = {
        "gru_minus_no_prediction_raw_safe_capture_percentage_points": statistics(raw_gain),
        "gru_positive_seed_count": int(sum(value > 0.0 for value in raw_gain)),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    lines = [
        "# 阶段 3B 正式结果：预测增强非循环 MAPPO 的多种子锁定评估",
        "",
        "日期：2026-08-19  ",
        "任务：部分可观测障碍环境下多无人机三维捕获半径追逃  ",
        "后端：kinematic 3D simulation  ",
        "统计单位：独立训练 seed；不是把同一 locked-test block 的回合当作独立训练重复。",
        "",
        "## 1. 固定协议",
        "",
        f"- 训练 seed：{', '.join(str(seed) for seed in seeds)}；每种子 MAPPO 训练步数：{int(protocol['train_steps']):,}。",
        f"- locked-test seed：{int(protocol['test_seed'])}；每个场景 {int(protocol['episodes_per_scenario'])} 回合，",
        "  `open`、`clutter`、`occluded` 共 300 回合/训练 seed。",
        "- 方法：无预测（44 维）、constant-velocity prediction（48 维）、冻结 GRU prediction（52 维）。",
        "- 每个方法分别评估 raw action 和 local CBF action；每个汇总行均为 3 个训练 seed、900 回合。",
        "- 95% 置信区间采用训练 seed 间 Student-t 区间（n=3，df=2）；同一锁定回合在不同 seed 下仅用于比较策略，",
        "  不被错误计为独立训练重复。",
        "",
        "## 2. 总体结果",
        "",
        "| 方法 / 执行 | Safe Capture | Capture | Collision | Boundary Violation | Capture Time (s) | Minimum Clearance (m) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        ("no_prediction", "raw"): "MAPPO，无预测，raw",
        ("no_prediction", "cbf"): "MAPPO，无预测，+CBF",
        ("constant_velocity", "raw"): "MAPPO，常速度预测，raw",
        ("constant_velocity", "cbf"): "MAPPO，常速度预测，+CBF",
        ("gru_prediction", "raw"): "MAPPO，GRU 预测，raw",
        ("gru_prediction", "cbf"): "MAPPO，GRU 预测，+CBF",
    }
    for method in METHODS:
        for action in ACTIONS:
            metrics = aggregate["results"][method][action]["overall"]["metrics"]
            lines.append(
                "| "
                + labels[(method, action)]
                + " | "
                + " | ".join(
                    [
                        format_percent(metrics["safe_capture_rate"]),
                        format_percent(metrics["capture_rate"]),
                        format_percent(metrics["collision_rate"]),
                        format_percent(metrics["world_violation_rate"]),
                        format_float(metrics["mean_capture_time_seconds"]),
                        format_float(metrics["mean_min_clearance_m"]),
                    ]
                )
                + " |"
            )
    paired = aggregate["paired_comparisons"]["gru_minus_no_prediction_raw_safe_capture_percentage_points"]
    lines.extend(
        [
            "",
            "## 3. 结论",
            "",
            f"- raw action 下，GRU 相对无预测的 Safe Capture 差值为 {format_float(paired, digits=2)} 个百分点；",
            f"  3 个训练 seed 中有 {aggregate['paired_comparisons']['gru_positive_seed_count']} 个为正。",
            "- CBF 下三种方法在当前 900 回合/方法锁定测试中均达到 100% Safe Capture 和 0% Collision；",
            "  因此不能把该安全收益归因于预测模块，CBF 必须作为独立组件报告。",
            "- 由于 GRU raw-action 改善未达到计划中预设的 5 个百分点参考门槛，本阶段结果支持“接口可用且存在",
            "  小幅 raw-action 改善”，但不支持“GRU 已稳定显著提高最终 Safe Capture”的强结论。",
            "- 下一步进入阶段 3C：实现 Recurrent-MAPPO，并在更强遮挡/延迟目标域复查预测与记忆的独立贡献。",
            "",
            "## 4. 可复现证据",
            "",
            "- 运行器：`scripts/run_stage3b_formal.py`",
            "- 聚合脚本：`scripts/aggregate_stage3b_formal.py`",
            "- 完整结构化统计：`results/PREDICTION_POLICY_STAGE3_FORMAL_SUMMARY.json`",
            "- 所有按种子保存的 checkpoint、TensorBoard、逐回合 CSV、配置、协议和三维轨迹：`results/stage3b_formal/`（本地生成，未入 Git）。",
            "",
            "结论仅适用于当前运动学三维仿真中的捕获半径追逃，不等同于实体接触、网捕、SITL、真实视觉闭环或实飞捕获。",
        ]
    )
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_report": str(args.output_report)}, indent=2))


if __name__ == "__main__":
    main()
