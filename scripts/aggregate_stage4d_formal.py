"""Aggregate F1/F2 checkpoint replays on the common P1 locked test block."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
F1_METHOD = "f1_time_aligned_belief"
F2_METHOD = "f2_uncertainty_features"
# Public alias retained for small protocol tests and downstream tooling.
METHOD = F2_METHOD
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
    parser.add_argument("--f1-root", type=Path, required=True)
    parser.add_argument("--f2-root", type=Path, required=True)
    parser.add_argument(
        "--p1-summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "RECURRENT_POLICY_STAGE3C_P1_STRESS_SUMMARY.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "results" / "STAGE4_COMMON_LOCKED_SUMMARY.json",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=PROJECT_ROOT / "results" / "STAGE4_COMMON_LOCKED_REPORT.md",
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


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required result: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return document


def statistics(values: list[float]) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.size == 0:
        raise ValueError("Cannot summarize an empty vector.")
    sample_std = float(np.std(vector, ddof=1)) if vector.size > 1 else 0.0
    half_width = (
        float(T_CRITICAL_95.get(int(vector.size), 1.95996398454) * sample_std / math.sqrt(vector.size))
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


def format_percent(row: dict[str, Any]) -> str:
    return f"{100.0 * float(row['mean']):.2f}% +/- {100.0 * float(row['ci95_half_width']):.2f}%"


def format_float(row: dict[str, Any], digits: int = 3) -> str:
    return f"{float(row['mean']):.{digits}f} +/- {float(row['ci95_half_width']):.{digits}f}"


def load_replay(root: Path, method: str, seeds: list[int]) -> tuple[dict[str, Any], dict[str, str]]:
    results: dict[str, dict[str, Any]] = {action: {} for action in ACTIONS}
    hashes: dict[str, str] = {}
    for action in ACTIONS:
        for condition in CONDITIONS:
            rows: list[dict[str, Any]] = []
            for seed in seeds:
                path = root / method / f"seed{seed}" / condition / f"summary_{action}.json"
                document = load_json(path)
                row = document.get(condition)
                if not isinstance(row, dict) or int(row.get("episodes", -1)) <= 0:
                    raise ValueError(f"Malformed summary: {path}")
                if any(metric not in row for metric in METRICS):
                    raise ValueError(f"Missing metric in {path}")
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


def validate_protocol(root: Path, expected_method: str) -> tuple[dict[str, Any], list[int]]:
    protocol = load_json(root / "protocol.json")
    seeds = [int(seed) for seed in protocol.get("training_seeds", [])]
    if protocol.get("method") != expected_method:
        raise ValueError(f"Expected {expected_method}, found {protocol.get('method')!r} in {root}.")
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("Common locked aggregation requires at least three distinct training seeds.")
    if int(protocol.get("test_seed", -1)) != 642001:
        raise ValueError("All replays must use P1 locked test seed 642001.")
    if tuple(protocol.get("conditions", {})) != CONDITIONS:
        raise ValueError("Replay conditions differ from the frozen P1 conditions.")
    return protocol, seeds


def main() -> None:
    args = parse_args()
    f1_root = args.f1_root.resolve()
    f2_root = args.f2_root.resolve()
    f1_protocol, seeds = validate_protocol(f1_root, F1_METHOD)
    f2_protocol, f2_seeds = validate_protocol(f2_root, F2_METHOD)
    if f2_seeds != seeds:
        raise ValueError("F1 and F2 replay training seeds differ.")
    f1_results, f1_hashes = load_replay(f1_root, F1_METHOD, seeds)
    f2_results, f2_hashes = load_replay(f2_root, F2_METHOD, seeds)
    p1_path = args.p1_summary.resolve()
    p1 = load_json(p1_path)
    if int(p1.get("protocol", {}).get("test_seed", -1)) != 642001:
        raise ValueError("The D/E baseline summary does not use P1 locked test seed 642001.")
    if [int(seed) for seed in p1.get("protocol", {}).get("training_seeds", [])] != seeds:
        raise ValueError("The D/E baseline and F1/F2 replay training seeds differ.")
    p1_results = p1["results"]

    comparisons: dict[str, Any] = {}
    baselines = {
        "f1": f1_results,
        "d": p1_results["recurrent_no_prediction"],
        "e": p1_results["recurrent_gru_prediction"],
    }
    for baseline_name, baseline_results in baselines.items():
        for action in ACTIONS:
            for condition in CONDITIONS:
                left = f2_results[action][condition]["metrics"]["safe_capture_rate"]["values"]
                right = baseline_results[action][condition]["metrics"]["safe_capture_rate"]["values"]
                comparisons[f"f2_minus_{baseline_name}_{action}_safe_capture_pp__{condition}"] = pp_difference(
                    left, right
                )

    aggregate = {
        "stage": "4E_common_locked_checkpoint_replay",
        "statistical_unit": "training_seed",
        "confidence_interval": "two-sided 95 percent Student-t interval across independent training seeds",
        "locked_test_seed": 642001,
        "f1_protocol": f1_protocol,
        "f2_protocol": f2_protocol,
        "f1_results": f1_results,
        "f2_results": f2_results,
        "baseline_p1_summary": relative_path(p1_path),
        "source_summary_sha256": {
            **f1_hashes,
            **f2_hashes,
            relative_path(f1_root / "protocol.json"): sha256(f1_root / "protocol.json"),
            relative_path(f2_root / "protocol.json"): sha256(f2_root / "protocol.json"),
            relative_path(p1_path): sha256(p1_path),
        },
        "paired_comparisons": comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    lines = [
        "# 阶段 4 共同锁定测试：F1/F2 belief 方法消融",
        "",
        "任务：部分可观测障碍环境下多无人机三维捕获半径追逃",
        "后端：kinematic 3D simulation",
        "统计单位：独立训练 seed；F1、F2、D、E 都重放在 P1 的同一 locked-test seed 642001。",
        "",
        "## 1. 协议",
        "",
        f"- 训练 seed：{', '.join(str(seed) for seed in seeds)}；F1/F2 均已完成每 seed 65,536 环境步训练。",
        f"- 共同 locked-test：seed 642001；四个固定条件各 {int(f2_protocol['episodes_per_condition'])} 回合。",
        "- F1：门控时间对齐 belief；F2：F1 + confidence/对角协方差显式特征。",
        "- raw 和 local-CBF action 分开评估；所有比较均以相同训练 seed 和相同测试回合配对。",
        "",
        "## 2. F2 结果",
        "",
        "| 条件 / 执行 | Safe Capture | Collision | Observation Age | Message Age |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        for action in ACTIONS:
            metrics = f2_results[action][condition]["metrics"]
            lines.append(
                f"| {condition} / {action} | {format_percent(metrics['safe_capture_rate'])} | "
                f"{format_percent(metrics['collision_rate'])} | {format_float(metrics['mean_observation_age_steps'])} | "
                f"{format_float(metrics['mean_message_age_steps'])} |"
            )
    lines.extend(["", "## 3. F2 相对基线的 raw Safe Capture 配对差值", "", "| 条件 | F2 - F1 (pp) | F2 - D (pp) | F2 - E (pp) |", "|---|---:|---:|---:|"])
    for condition in CONDITIONS:
        f1_difference = comparisons[f"f2_minus_f1_raw_safe_capture_pp__{condition}"]
        d_difference = comparisons[f"f2_minus_d_raw_safe_capture_pp__{condition}"]
        e_difference = comparisons[f"f2_minus_e_raw_safe_capture_pp__{condition}"]
        lines.append(
            f"| {condition} | {format_float(f1_difference, 2)} | {format_float(d_difference, 2)} | "
            f"{format_float(e_difference, 2)} |"
        )
    lines.extend(
        [
            "",
            "## 4. 结论边界",
            "",
            "- 只有至少 2/3 个训练 seed 同方向、置信区间和失败轨迹一致时，才表述为指定困难域改善。",
            "- CBF 结果用于分离安全过滤器贡献；不能将 CBF 饱和收益归因于 belief 模块。",
            "- 本结果证明的是冻结运动学仿真下的安全进入捕获半径，不是实体捕获、真实视觉闭环或实飞成功。",
        ]
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
