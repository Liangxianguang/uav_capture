"""Aggregate the locked-test Stage 3C recurrent-policy multi-seed ablation.

The independent experimental unit is a training seed. The locked test episodes
are identical across methods within a seed and are retained in local result
directories for audit, but are not treated as independent training repeats.
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
METHODS = ("recurrent_no_prediction", "recurrent_gru_prediction")
ACTIONS = ("raw", "cbf")
SCENARIOS = ("clutter", "occluded", "open", "overall")
METRICS = (
    "safe_capture_rate",
    "capture_rate",
    "collision_rate",
    "world_violation_rate",
    "mean_capture_time_seconds",
    "mean_min_clearance_m",
    "mean_visible_fraction",
    "mean_message_age_steps",
)
# The formal protocol has three independent training seeds. The normal
# approximation keeps the tool usable for later, larger experiments without
# adding a SciPy dependency.
T_CRITICAL_95 = {2: 12.7062047364, 3: 4.30265272975, 4: 3.18244630528, 5: 2.7764451052}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "results" / "stage3c_formal")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "results" / "RECURRENT_POLICY_STAGE3C_FORMAL_SUMMARY.json",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=PROJECT_ROOT / "results" / "RECURRENT_POLICY_STAGE3C_FORMAL_REPORT.md",
    )
    parser.add_argument(
        "--stage3b-summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "PREDICTION_POLICY_STAGE3_FORMAL_SUMMARY.json",
        help="Optional Stage 3B aggregation used for an explicitly non-paired contextual comparison.",
    )
    return parser.parse_args()


def t_critical_95(count: int) -> float:
    return T_CRITICAL_95.get(count, 1.95996398454)


def statistics(values: list[float]) -> dict[str, float | int | list[float]]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.size == 0:
        raise ValueError("Cannot summarize an empty vector.")
    sample_std = float(np.std(vector, ddof=1)) if vector.size > 1 else 0.0
    ci95_half_width = (
        float(t_critical_95(int(vector.size)) * sample_std / math.sqrt(vector.size))
        if vector.size > 1
        else 0.0
    )
    return {
        "count": int(vector.size),
        "mean": float(np.mean(vector)),
        "sample_std": sample_std,
        "ci95_half_width": ci95_half_width,
        "minimum": float(np.min(vector)),
        "maximum": float(np.max(vector)),
        "values": [float(value) for value in vector],
    }


def format_percent(summary: dict[str, float | int | list[float]]) -> str:
    return f"{100.0 * float(summary['mean']):.2f}% +/- {100.0 * float(summary['ci95_half_width']):.2f}%"


def format_float(summary: dict[str, float | int | list[float]], digits: int = 3) -> str:
    return f"{float(summary['mean']):.{digits}f} +/- {float(summary['ci95_half_width']):.{digits}f}"


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        # ``--root`` may point to an externally archived experiment directory.
        # Retain a portable representation rather than rejecting valid evidence.
        return resolved.as_posix()


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing locked-test summary: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != set(SCENARIOS):
        raise ValueError(f"Expected scenarios {SCENARIOS}, found malformed summary: {path}")
    for scenario, row in document.items():
        if not isinstance(row, dict) or any(metric not in row for metric in METRICS):
            raise ValueError(f"Missing required metric in {path} / {scenario}")
    return document


def load_stage3b_context(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    try:
        return {
            "no_prediction_raw": document["results"]["no_prediction"]["raw"]["overall"]["metrics"],
            "gru_prediction_raw": document["results"]["gru_prediction"]["raw"]["overall"]["metrics"],
            "source": relative_path(path),
            "sha256": source_hash(path),
        }
    except (KeyError, TypeError) as error:
        raise ValueError(f"Malformed Stage 3B aggregation: {path}") from error


def append_overall_table(lines: list[str], aggregate: dict[str, Any]) -> None:
    lines.extend(
        [
            "| 方法 / 执行 | Safe Capture | Capture | Collision | Boundary Violation | Time-to-Capture (s) | Minimum Clearance (m) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    labels = {
        ("recurrent_no_prediction", "raw"): "Recurrent-MAPPO，无学习式预测，raw",
        ("recurrent_no_prediction", "cbf"): "Recurrent-MAPPO，无学习式预测，+CBF",
        ("recurrent_gru_prediction", "raw"): "Recurrent-MAPPO，GRU 预测，raw",
        ("recurrent_gru_prediction", "cbf"): "Recurrent-MAPPO，GRU 预测，+CBF",
    }
    for method in METHODS:
        for action in ACTIONS:
            metrics = aggregate["results"][method][action]["overall"]["metrics"]
            lines.append(
                "| "
                + labels[(method, action)]
                + " | "
                + " | ".join(
                    (
                        format_percent(metrics["safe_capture_rate"]),
                        format_percent(metrics["capture_rate"]),
                        format_percent(metrics["collision_rate"]),
                        format_percent(metrics["world_violation_rate"]),
                        format_float(metrics["mean_capture_time_seconds"]),
                        format_float(metrics["mean_min_clearance_m"]),
                    )
                )
                + " |"
            )


def append_scenario_table(lines: list[str], aggregate: dict[str, Any]) -> None:
    lines.extend(
        [
            "| 场景 | D raw Safe Capture | E raw Safe Capture | D raw Collision | E raw Collision |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for scenario in ("open", "clutter", "occluded"):
        d_metrics = aggregate["results"]["recurrent_no_prediction"]["raw"][scenario]["metrics"]
        e_metrics = aggregate["results"]["recurrent_gru_prediction"]["raw"][scenario]["metrics"]
        lines.append(
            "| "
            + scenario
            + " | "
            + " | ".join(
                (
                    format_percent(d_metrics["safe_capture_rate"]),
                    format_percent(e_metrics["safe_capture_rate"]),
                    format_percent(d_metrics["collision_rate"]),
                    format_percent(e_metrics["collision_rate"]),
                )
            )
            + " |"
        )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol_path = root / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing formal protocol: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in protocol.get("seeds", [])]
    methods = tuple(protocol.get("methods", []))
    if methods != METHODS:
        raise ValueError(f"Expected methods {METHODS}, found {methods}")
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("Formal Stage 3C aggregation requires at least three distinct training seeds.")

    documents: dict[str, dict[str, dict[int, dict[str, Any]]]] = {method: {} for method in METHODS}
    source_summaries: dict[str, str] = {}
    for method in METHODS:
        for action in ACTIONS:
            documents[method][action] = {}
            for seed in seeds:
                summary_path = root / method / f"seed{seed}" / f"evaluation_{action}" / "summary.json"
                documents[method][action][seed] = load_summary(summary_path)
                source_summaries[relative_path(summary_path)] = source_hash(summary_path)

    aggregate: dict[str, Any] = {
        "stage": "3C_recurrent_policy_formal_multiseed",
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
            for scenario in SCENARIOS:
                rows = [documents[method][action][seed][scenario] for seed in seeds]
                episodes_per_seed = {int(row["episodes"]) for row in rows}
                if len(episodes_per_seed) != 1:
                    raise ValueError(f"Episode count differs by seed: {method} / {action} / {scenario}")
                aggregate["results"][method][action][scenario] = {
                    "episodes_per_seed": episodes_per_seed.pop(),
                    "episodes_total": int(sum(int(row["episodes"]) for row in rows)),
                    "metrics": {metric: statistics([float(row[metric]) for row in rows]) for metric in METRICS},
                }

    raw_safe_capture_difference = [
        100.0
        * (
            float(documents["recurrent_gru_prediction"]["raw"][seed]["overall"]["safe_capture_rate"])
            - float(documents["recurrent_no_prediction"]["raw"][seed]["overall"]["safe_capture_rate"])
        )
        for seed in seeds
    ]
    aggregate["paired_comparisons"] = {
        "recurrent_gru_minus_recurrent_no_prediction_raw_safe_capture_percentage_points": statistics(
            raw_safe_capture_difference
        ),
        "positive_seed_count": int(sum(value > 0.0 for value in raw_safe_capture_difference)),
    }
    stage3b_context = load_stage3b_context(args.stage3b_summary.resolve())
    if stage3b_context is not None:
        aggregate["stage3b_context"] = stage3b_context

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    paired = aggregate["paired_comparisons"][
        "recurrent_gru_minus_recurrent_no_prediction_raw_safe_capture_percentage_points"
    ]
    lines = [
        "# 阶段 3C 正式结果：Recurrent-MAPPO 多种子锁定评估",
        "",
        "日期：2026-08-19  ",
        "任务：部分可观测障碍环境下多无人机三维捕获半径追逃  ",
        "后端：kinematic 3D simulation  ",
        "统计单位：独立训练 seed；不将同一 locked-test block 的逐回合结果错误视作独立训练重复。",
        "",
        "## 1. 固定协议",
        "",
        f"- 训练 seed：{', '.join(str(seed) for seed in seeds)}；每种子循环 MAPPO 训练 {int(protocol['train_steps']):,} 环境步。",
        f"- locked-test seed：{int(protocol['test_seed'])}；`open`、`clutter`、`occluded` 各 {int(protocol['episodes_per_scenario'])} 回合，",
        "  共 300 回合/训练 seed，900 回合/方法/执行方式。",
        "- D：无学习式预测的 Recurrent-MAPPO；E：Recurrent-MAPPO + 冻结 GRU 预测特征。",
        "- 两个循环 actor 均使用 MLP behavior-cloning prior + 零初始化 GRU residual；",
        "  因此循环模块起点等价于其对应无记忆 prior，随后只学习历史带来的残差修正。",
        "- 每种方法分别评估 raw action 与 local CBF action；95% CI 为训练 seed 间 Student-t 区间（n=3，df=2）。",
        "",
        "## 2. 总体结果",
        "",
    ]
    append_overall_table(lines, aggregate)
    lines.extend(["", "## 3. 场景分桶：raw action", ""])
    append_scenario_table(lines, aggregate)
    lines.extend(
        [
            "",
            "## 4. 配对比较与结论边界",
            "",
            f"- E 相对 D 的 raw-action Safe Capture 配对差值为 {format_float(paired, digits=2)} 个百分点；",
            f"  {aggregate['paired_comparisons']['positive_seed_count']}/{len(seeds)} 个训练 seed 为正。",
            "- 该置信区间描述独立训练 seed 间的不确定性。若区间跨越 0，报告为方向性趋势而非稳定显著提升。",
            "- raw 与 +CBF 的差异用于分离循环策略本身和安全过滤器的贡献。若 +CBF 饱和为 100% Safe Capture，",
            "  不应将该增益归因于预测或循环记忆。",
        ]
    )
    if stage3b_context is not None:
        stage3b_no_prediction = stage3b_context["no_prediction_raw"]["safe_capture_rate"]
        stage3b_gru_prediction = stage3b_context["gru_prediction_raw"]["safe_capture_rate"]
        stage3c_d = aggregate["results"]["recurrent_no_prediction"]["raw"]["overall"]["metrics"]["safe_capture_rate"]
        stage3c_e = aggregate["results"]["recurrent_gru_prediction"]["raw"]["overall"]["metrics"]["safe_capture_rate"]
        lines.extend(
            [
                "",
                "## 5. 与阶段 3B 非循环策略的上下文比较",
                "",
                "| 设置 | raw Safe Capture |",
                "|---|---:|",
                f"| Stage 3B MAPPO，无预测 | {format_percent(stage3b_no_prediction)} |",
                f"| Stage 3B MAPPO，GRU 预测 | {format_percent(stage3b_gru_prediction)} |",
                f"| Stage 3C D，循环无预测 | {format_percent(stage3c_d)} |",
                f"| Stage 3C E，循环 + GRU 预测 | {format_percent(stage3c_e)} |",
                "",
                "阶段 3B 与 3C 都遵循相同 locked-test 协议，但不是对同一训练 seed 的配对随机试验；",
                "该表仅提供跨架构的描述性上下文，不用于声称循环结构相对非循环结构的因果增益。",
            ]
        )
    lines.extend(
        [
            "",
            "## 6. 可复现证据",
            "",
            "- 运行器：`scripts/run_stage3c_formal.py`",
            "- 聚合脚本：`scripts/aggregate_stage3c_formal.py`",
            "- 本报告的结构化统计：`results/RECURRENT_POLICY_STAGE3C_FORMAL_SUMMARY.json`",
            "- 本地完整证据（checkpoint、TensorBoard、逐回合 CSV、配置、协议和轨迹）：`results/stage3c_formal/`。",
            "",
            "结论只适用于当前运动学三维仿真中的捕获半径任务，不等同于实体接触、网捕、SITL、真实视觉闭环或实飞捕获。",
        ]
    )
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_report": str(args.output_report)}, indent=2))


if __name__ == "__main__":
    main()
