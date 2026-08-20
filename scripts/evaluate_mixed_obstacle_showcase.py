"""Probe a frozen policy on repeated controlled mixed-obstacle S1 episodes.

This is a distribution-shift diagnostic, not a replacement for a locked-test
benchmark. Every episode shares the same validated geometry and opposite-side
spawn layout; only the supplied episode seeds vary perception and motion noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import central_mixed_obstacle_scenario  # noqa: E402
from evaluate_capture_radius_mappo import load_policy, select_device  # noqa: E402
from run_mixed_obstacle_showcase import build_config, rollout_showcase, rollout_showcase_expert  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("f1", "f2"), default="f2")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--baseline", choices=("dynamic_encirclement",))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=643001)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--initial-side-distance", type=float, default=5.0)
    parser.add_argument("--scenario", choices=("s1", "s2"), default="s1")
    parser.add_argument("--detection-range", type=float, default=14.0)
    parser.add_argument("--target-speed-scale", type=float, default=0.55)
    parser.add_argument("--use-cbf", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    return parser.parse_args()


def summarize(rows: list[dict[str, object]]) -> dict[str, float | int | None]:
    capture_times = [float(row["capture_time_seconds"]) for row in rows if row["capture_time_seconds"] is not None]
    return {
        "episodes": len(rows),
        "safe_capture_rate": float(np.mean([bool(row["safe_capture_success"]) for row in rows])),
        "showcase_success_rate": float(np.mean([bool(row["showcase_success"]) for row in rows])),
        "defender_obstacle_crossing_rate": float(np.mean([float(row["defender_crossing_rate"]) for row in rows])),
        "target_obstacle_crossing_rate": float(np.mean([float(row["target_crossing_rate"]) for row in rows])),
        "collision_rate": float(np.mean([bool(row["collision"]) for row in rows])),
        "boundary_violation_rate": float(np.mean([int(row["world_violation_steps"]) > 0 for row in rows])),
        "mean_min_clearance_m": float(np.mean([float(row["min_clearance_m"]) for row in rows])),
        "worst_min_clearance_m": float(min(float(row["min_clearance_m"]) for row in rows)),
        "mean_target_min_obstacle_clearance_m": float(
            np.mean([float(row["target_min_obstacle_clearance_m"]) for row in rows])
        ),
        "mean_capture_time_seconds": float(np.mean(capture_times)) if capture_times else None,
        "mean_visible_fraction": float(np.mean([float(row["mean_visible_fraction"]) for row in rows])),
        "mean_observation_age_steps": float(np.mean([float(row["mean_observation_age_steps"]) for row in rows])),
    }


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive.")
    if (args.checkpoint is None) == (args.baseline is None):
        raise ValueError("Provide exactly one of --checkpoint or --baseline.")
    checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(args.method, args.detection_range, args.target_speed_scale)
    scenario = central_mixed_obstacle_scenario(
        initial_side_distance=args.initial_side_distance,
        target_crossing_required=args.scenario == "s2",
        defender_side="right" if args.scenario == "s2" else "left",
    )
    device = select_device(args.device)
    if checkpoint is not None:
        prototype = CaptureRadiusPursuit3DEnv(
            config,
            obstacle_count=len(scenario.obstacles),
            target_speed_scale=args.target_speed_scale,
        )
        policy, action_scale, _metadata = load_policy(
            checkpoint,
            prototype,
            prototype.reset(seed=args.seed),
            device,
        )
    rows: list[dict[str, object]] = []
    for episode_index in range(args.episodes):
        episode_seed = int(args.seed + episode_index)
        if checkpoint is None:
            row, _env = rollout_showcase_expert(config, scenario, seed=episode_seed, use_cbf=args.use_cbf)
            row.update({"method": str(args.baseline), "checkpoint": None, "device": str(device)})
        else:
            row, _env = rollout_showcase(
                policy,
                config,
                scenario,
                seed=episode_seed,
                device=device,
                action_scale=action_scale,
                use_cbf=args.use_cbf,
            )
            row.update({"method": args.method, "checkpoint": str(checkpoint), "device": str(device)})
        rows.append(row)
    summary = summarize(rows)
    output_dir.joinpath("episodes.csv").write_text("", encoding="utf-8")
    with output_dir.joinpath("episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "evaluation_type": "controlled_showcase_distribution_probe",
                "not_a_locked_test": True,
                "method": args.method if checkpoint is not None else args.baseline,
                "checkpoint": str(checkpoint) if checkpoint is not None else None,
                "use_cbf": bool(args.use_cbf),
                "base_seed": args.seed,
                "episode_seeds": [int(row["seed"]) for row in rows],
                "initial_side_distance_m": args.initial_side_distance,
                "scenario_kind": args.scenario,
                "detection_range_m": args.detection_range,
                "target_speed_scale": args.target_speed_scale,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_dir.joinpath("config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
