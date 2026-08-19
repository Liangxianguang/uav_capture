"""Aggregate F2 and compare it with F1 and the locked D/E baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD = "f2_uncertainty_features"
ACTIONS = ("raw", "cbf")
CONDITIONS = (
    "nominal_partial_observation",
    "delayed_measurements",
    "burst_occlusion",
    "communication_loss",
)
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
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "results" / "stage4d_formal")
    parser.add_argument("--f1-summary", type=Path, default=PROJECT_ROOT / "results" / "STAGE4C_F1_FORMAL_SUMMARY.json")
    parser.add_argument("--p1-summary", type=Path, default=PROJECT_ROOT / "results" / "RECURRENT_POLICY_STAGE3C_P1_STRESS_SUMMARY.json")
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "results" / "STAGE4D_F2_FORMAL_SUMMARY.json")
    parser.add_argument("--output-report", type=Path, default=PROJECT_ROOT / "results" / "STAGE4D_F2_FORMAL_REPORT.md")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return resolved.as_posix()


def statistics(values: list[float]) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.size == 0:
        raise ValueError("Cannot summarize an empty vector.")
    sample_std = float(np.std(vector, ddof=1)) if vector.size > 1 else 0.0
    half_width = float(T_CRITICAL_95.get(int(vector.size), 1.95996398454) * sample_std / math.sqrt(vector.size)) if vector.size > 1 else 0.0
    return {"count": int(vector.size), "mean": float(np.mean(vector)), "sample_std": sample_std, "ci95_half_width": half_width, "minimum": float(np.min(vector)), "maximum": float(np.max(vector)), "values": [float(value) for value in vector]}


def format_percent(row: dict[str, Any]) -> str:
    return f"{100.0 * float(row['mean']):.2f}% +/- {100.0 * float(row['ci95_half_width']):.2f}%"


def format_float(row: dict[str, Any], digits: int = 3) -> str:
    return f"{float(row['mean']):.{digits}f} +/- {float(row['ci95_half_width']):.{digits}f}"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required result: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return document


def load_f2(root: Path, seeds: list[int]) -> tuple[dict[str, Any], dict[str, str]]:
    results: dict[str, dict[str, Any]] = {action: {} for action in ACTIONS}
    hashes: dict[str, str] = {}
    for action in ACTIONS:
        for condition in CONDITIONS:
            rows: list[dict[str, Any]] = []
            for seed in seeds:
                path = root / METHOD / f"seed{seed}" / condition / f"summary_{action}.json"
                document = load_json(path)
                row = document.get(condition)
                if not isinstance(row, dict) or int(row.get("episodes", -1)) <= 0:
                    raise ValueError(f"Malformed summary: {path}")
                for metric in METRICS:
                    if metric not in row:
                        raise ValueError(f"Missing {metric} in {path}")
                rows.append(row)
                hashes[relative_path(path)] = sha256(path)
            results[action][condition] = {
                "episodes_per_seed": int(rows[0]["episodes"]),
                "episodes_total": int(sum(int(row["episodes"]) for row in rows)),
                "metrics": {metric: statistics([float(row[metric]) for row in rows]) for metric in METRICS},
            }
    return results, hashes


def pp_difference(left: list[float], right: list[float]) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("Paired vectors have different lengths.")
    result = statistics([100.0 * (float(a) - float(b)) for a, b in zip(left, right)])
    result["positive_seed_count"] = int(sum(value > 0.0 for value in result["values"]))
    return result


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol = load_json(root / "protocol.json")
    seeds = [int(seed) for seed in protocol.get("training_seeds", [])]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("F2 aggregation requires at least three distinct training seeds.")
    f2_results, source_hashes = load_f2(root, seeds)
    f1 = load_json(args.f1_summary.resolve())
    p1 = load_json(args.p1_summary.resolve())
    f1_results = f1["f1_results"]
    p1_results = p1["results"]
    comparisons: dict[str, Any] = {}
    for baseline_name, baseline_results in (("f1", f1_results), ("d", p1_results["recurrent_no_prediction"]), ("e", p1_results["recurrent_gru_prediction"])):
        for action in ACTIONS:
            for condition in CONDITIONS:
                left = f2_results[action][condition]["metrics"]["safe_capture_rate"]["values"]
                if baseline_name == "f1":
                    right = baseline_results[action][condition]["metrics"]["safe_capture_rate"]["values"]
                else:
                    right = baseline_results[action][condition]["metrics"]["safe_capture_rate"]["values"]
                comparisons[f"f2_minus_{baseline_name}_{action}_safe_capture_pp__{condition}"] = pp_difference(left, right)
    aggregate = {
        "stage": "4D_F2_uncertainty_aware_recurrent_mappo_formal",
        "statistical_unit": "training_seed",
        "confidence_interval": "two-sided 95 percent Student-t interval across independent training seeds",
        "protocol": protocol,
        "f2_results": f2_results,
        "baseline_f1_summary": relative_path(args.f1_summary.resolve()),
        "baseline_p1_summary": relative_path(args.p1_summary.resolve()),
        "source_summary_sha256": {**source_hashes, relative_path(args.f1_summary.resolve()): sha256(args.f1_summary.resolve()), relative_path(args.p1_summary.resolve()): sha256(args.p1_summary.resolve())},
        "paired_comparisons": comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    lines = [
        "# 阶段 4D F2 正式结果：显式 belief 不确定度特征",
        "",
        "任务：部分可观测障碍环境下多无人机三维捕获半径追逃",
        "后端：kinematic 3D simulation",
        "统计单位：独立训练 seed；F2 与 F1/D/E 使用同一组四域困难测试协议。",
        "",
        "## 1. 方法与协议",
        "",
        f"- F2 训练 seed：{', '.join(str(seed) for seed in seeds)}；每个 seed 训练 {int(protocol['train_steps']):,} 环境步。",
        f"- locked-test seed：{int(protocol['test_seed'])}；每个条件 {int(protocol['episodes_per_condition'])} 回合。",
        "- F2 = F1 + include_uncertainty_features：actor 额外接收 belief confidence、协方差对角项和已有消息年龄。",
        "- 奖励、门控时间对齐规则、网络规模、训练场景和 CBF 保持不变。",
        "",
        "## 2. F2 结果",
        "",
        "| 条件 / 执行 | Safe Capture | Capture | Collision | Boundary Violation | Time-to-Capture (s) | Min Clearance (m) | Observation Age | Message Age |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        for action in ACTIONS:
            metrics = f2_results[action][condition]["metrics"]
            lines.append(f"| {condition} / {action} | {format_percent(metrics['safe_capture_rate'])} | {format_percent(metrics['capture_rate'])} | {format_percent(metrics['collision_rate'])} | {format_percent(metrics['world_violation_rate'])} | {format_float(metrics['mean_capture_time_seconds'])} | {format_float(metrics['mean_min_clearance_m'])} | {format_float(metrics['mean_observation_age_steps'])} | {format_float(metrics['mean_message_age_steps'])} |")
    lines.extend(["", "## 3. F2 相对 F1 的 raw Safe Capture 配对差值", "", "| 条件 | F2 - F1 (pp) | 3/3 同向? |", "|---|---:|:---:|"])
    for condition in CONDITIONS:
        row = comparisons[f"f2_minus_f1_raw_safe_capture_pp__{condition}"]
        lines.append(f"| {condition} | {format_float(row, 2)} | {'是' if row['positive_seed_count'] in (0, 3) else '否'} |")
    lines.extend(["", "## 4. 结论边界", "", "- 只有至少 2/3 个训练 seed 同方向，且困难域的 belief/安全指标同时支持时，才表述为改善。", "- CBF 结果用于分离安全过滤器贡献，不能把 CBF 的收益归因于 F2 特征。", "- 本结果仍是冻结运动学仿真下进入捕获半径的证据，不是实体捕获、真实视觉闭环或实飞成功。"])
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
