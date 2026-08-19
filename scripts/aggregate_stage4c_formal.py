"""Aggregate the formal F1 results with the locked P1 D/E baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD = "f1_time_aligned_belief"
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
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "results" / "stage4c_formal")
    parser.add_argument(
        "--p1-summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "RECURRENT_POLICY_STAGE3C_P1_STRESS_SUMMARY.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "results" / "STAGE4C_F1_FORMAL_SUMMARY.json",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=PROJECT_ROOT / "results" / "STAGE4C_F1_FORMAL_REPORT.md",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return resolved.as_posix()


def t_critical_95(count: int) -> float:
    return T_CRITICAL_95.get(count, 1.95996398454)


def statistics(values: list[float]) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.size == 0:
        raise ValueError("Cannot summarize an empty vector.")
    sample_std = float(np.std(vector, ddof=1)) if vector.size > 1 else 0.0
    half_width = float(t_critical_95(int(vector.size)) * sample_std / math.sqrt(vector.size)) if vector.size > 1 else 0.0
    return {
        "count": int(vector.size),
        "mean": float(np.mean(vector)),
        "sample_std": sample_std,
        "ci95_half_width": half_width,
        "minimum": float(np.min(vector)),
        "maximum": float(np.max(vector)),
        "values": [float(value) for value in vector],
    }


def format_percent(row: dict[str, Any]) -> str:
    return f"{100.0 * float(row['mean']):.2f}% +/- {100.0 * float(row['ci95_half_width']):.2f}%"


def format_float(row: dict[str, Any], digits: int = 3) -> str:
    return f"{float(row['mean']):.{digits}f} +/- {float(row['ci95_half_width']):.{digits}f}"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required result: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return document


def load_f1(root: Path, seeds: list[int]) -> tuple[dict[str, Any], dict[str, str]]:
    results: dict[str, dict[str, dict[str, Any]]] = {action: {} for action in ACTIONS}
    source_hashes: dict[str, str] = {}
    for action in ACTIONS:
        for condition in CONDITIONS:
            rows: list[dict[str, Any]] = []
            for seed in seeds:
                path = root / METHOD / f"seed{seed}" / condition / f"summary_{action}.json"
                document = load_json(path)
                if condition not in document or "overall" not in document:
                    raise ValueError(f"Malformed F1 summary (missing condition/overall): {path}")
                row = document[condition]
                if int(row.get("episodes", -1)) <= 0:
                    raise ValueError(f"Invalid episode count in {path}")
                for metric in METRICS:
                    if metric not in row:
                        raise ValueError(f"Missing {metric} in {path}")
                rows.append(row)
                source_hashes[relative_path(path)] = sha256(path)
            results[action][condition] = {
                "episodes_per_seed": int(rows[0]["episodes"]),
                "episodes_total": int(sum(int(row["episodes"]) for row in rows)),
                "metrics": {metric: statistics([float(row[metric]) for row in rows]) for metric in METRICS},
            }
    return results, source_hashes


def load_p1(path: Path, seeds: list[int]) -> tuple[dict[str, Any], dict[str, str]]:
    document = load_json(path)
    protocol = document.get("protocol", {})
    p1_seeds = [int(seed) for seed in protocol.get("training_seeds", [])]
    if p1_seeds != seeds:
        raise ValueError(f"P1 baseline seeds {p1_seeds} do not match F1 seeds {seeds}.")
    source_hashes = {relative_path(path): sha256(path)}
    try:
        return document["results"], source_hashes
    except KeyError as error:
        raise ValueError(f"Malformed P1 summary: {path}") from error


def paired_difference(f1_values: list[float], baseline_values: list[float]) -> dict[str, Any]:
    if len(f1_values) != len(baseline_values):
        raise ValueError("Paired comparison vectors have different lengths.")
    differences = [100.0 * (float(a) - float(b)) for a, b in zip(f1_values, baseline_values)]
    result = statistics(differences)
    result["positive_seed_count"] = int(sum(value > 0.0 for value in differences))
    return result


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol = load_json(root / "protocol.json")
    seeds = [int(seed) for seed in protocol.get("training_seeds", [])]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("F1 aggregation requires at least three distinct training seeds.")
    if tuple(protocol.get("conditions", {})) != CONDITIONS:
        raise ValueError("F1 protocol conditions do not match the locked P1 condition order.")
    f1_results, f1_sources = load_f1(root, seeds)
    p1_results, p1_sources = load_p1(args.p1_summary.resolve(), seeds)
    comparisons: dict[str, Any] = {}
    for baseline in ("recurrent_no_prediction", "recurrent_gru_prediction"):
        for action in ACTIONS:
            for condition in CONDITIONS:
                f1_values = f1_results[action][condition]["metrics"]["safe_capture_rate"]["values"]
                baseline_values = p1_results[baseline][action][condition]["metrics"]["safe_capture_rate"]["values"]
                comparisons[f"f1_minus_{baseline}_{action}_safe_capture_pp__{condition}"] = paired_difference(
                    f1_values, baseline_values
                )

    aggregate: dict[str, Any] = {
        "stage": "4C_F1_belief_aware_recurrent_mappo_formal",
        "statistical_unit": "training_seed",
        "confidence_interval": "two-sided 95 percent Student-t interval across independent training seeds",
        "protocol": protocol,
        "f1_results": f1_results,
        "baseline_p1_summary": relative_path(args.p1_summary.resolve()),
        "source_summary_sha256": {**p1_sources, **f1_sources},
        "paired_comparisons": comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    lines = [
        "# 阶段 4C F1 正式结果：门控时间对齐 belief 的 Recurrent-MAPPO",
        "",
        "任务：部分可观测障碍环境下多无人机三维捕获半径追逃  ",
        "后端：kinematic 3D simulation  ",
        "统计单位：独立训练 seed；P1 的 D/E 与 F1 使用相同四域锁定测试协议。",
        "",
        "## 1. 固定协议",
        "",
        f"- F1 训练 seed：{', '.join(str(seed) for seed in seeds)}；每个 seed 训练 {int(protocol['train_steps']):,} 环境步。",
        f"- locked-test seed：{int(protocol['test_seed'])}；每个条件 {int(protocol['episodes_per_condition'])} 回合。",
        "- F1：Recurrent-MAPPO + time-aligned belief + age-gated stale velocity decay (0.80 after age 3).",
        "- raw action 与 local CBF action 分开统计；P1 D/E 结果来自已冻结的 P1 汇总文件。",
        "",
        "## 2. F1 结果",
        "",
        "| 条件 / 执行 | Safe Capture | Capture | Collision | Boundary Violation | Time-to-Capture (s) | Min Clearance (m) | Observation Age | Message Age |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"raw": "raw", "cbf": "+CBF"}
    for condition in CONDITIONS:
        for action in ACTIONS:
            metrics = f1_results[action][condition]["metrics"]
            lines.append(
                f"| {condition} / {labels[action]} | {format_percent(metrics['safe_capture_rate'])} | "
                f"{format_percent(metrics['capture_rate'])} | {format_percent(metrics['collision_rate'])} | "
                f"{format_percent(metrics['world_violation_rate'])} | {format_float(metrics['mean_capture_time_seconds'])} | "
                f"{format_float(metrics['mean_min_clearance_m'])} | {format_float(metrics['mean_observation_age_steps'])} | "
                f"{format_float(metrics['mean_message_age_steps'])} |"
            )
    lines.extend(["", "## 3. F1 相对 P1 基线的 raw Safe Capture 配对差值", ""])
    lines.extend(
        [
            "| 条件 | F1 - D (pp) | F1 - E (pp) |",
            "|---|---:|---:|",
        ]
    )
    for condition in CONDITIONS:
        d = comparisons[f"f1_minus_recurrent_no_prediction_raw_safe_capture_pp__{condition}"]
        e = comparisons[f"f1_minus_recurrent_gru_prediction_raw_safe_capture_pp__{condition}"]
        lines.append(f"| {condition} | {format_float(d, 2)} | {format_float(e, 2)} |")
    lines.extend(
        [
            "",
            "## 4. 结论边界",
            "",
            "- 只有至少 2/3 个训练 seed 同方向，且 belief 误差/Observation Age 与事件结果一致时，",
            "  才将结果表述为指定困难域的改善。",
            "- CBF 结果用于说明安全过滤器的贡献，不能把 +CBF 的饱和收益归因于 belief 方法。",
            "- 本报告证明的是冻结仿真模型下的协同进入捕获半径，不是实体捕获、真实视觉闭环或实飞成功。",
        ]
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
