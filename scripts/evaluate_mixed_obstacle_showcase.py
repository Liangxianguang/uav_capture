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
from encirclement3d.showcase import (  # noqa: E402
    central_capture_protocol_metadata,
    load_central_capture_protocol,
    validate_central_capture_protocol_environment,
)
from evaluate_capture_radius_mappo import load_policy, select_device  # noqa: E402
from run_mixed_obstacle_showcase import (  # noqa: E402
    build_config,
    build_showcase_scenario,
    rollout_showcase,
    rollout_showcase_expert,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("f1", "f2"), default="f2")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--baseline", choices=("dynamic_encirclement",))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=643001)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--initial-side-distance", type=float, default=5.0)
    parser.add_argument("--scenario", choices=("s1", "s1_cross", "s2", "s2_cross", "v4_s2"), default="s1_cross")
    parser.add_argument("--layout", choices=("open", "cylinder", "box", "wall", "cylinder_box", "mixed"), default="mixed")
    parser.add_argument(
        "--protocol-config",
        type=Path,
        help="Frozen V4 protocol YAML. Required for --scenario v4_s2.",
    )
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
        "safe_capture_in_pursuit_rate": float(np.mean([bool(row["safe_capture_in_pursuit"]) for row in rows])),
        "cooperative_safe_capture_rate": float(
            np.mean([bool(row.get("cooperative_safe_capture", row["safe_capture_in_pursuit"])) for row in rows])
        ),
        "showcase_success_rate": float(np.mean([bool(row["showcase_success"]) for row in rows])),
        "target_zone_entry_rate": float(np.mean([float(row["target_zone_entry_rate"]) for row in rows])),
        "defender_zone_entry_rate": float(np.mean([float(row["defender_zone_entry_rate"]) for row in rows])),
        "mean_defender_zone_entry_count": float(
            np.mean([float(row.get("defender_zone_entry_count", 0.0)) for row in rows])
        ),
        "defender_obstacle_crossing_rate": float(np.mean([float(row["defender_crossing_rate"]) for row in rows])),
        "target_obstacle_crossing_rate": float(np.mean([float(row["target_crossing_rate"]) for row in rows])),
        "transit_route_feasible_rate": float(np.mean([bool(row["transit_route_feasible"]) for row in rows])),
        "transit_success_rate": float(np.mean([bool(row["transit_success"]) for row in rows])),
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
    protocol = load_central_capture_protocol(args.protocol_config) if args.protocol_config is not None else None
    scenario = build_showcase_scenario(
        args.scenario,
        args.initial_side_distance,
        protocol=protocol,
        layout=args.layout,
    )
    config = build_config(
        args.method,
        args.detection_range,
        args.target_speed_scale,
        target_crossing_required=bool(scenario.target_crossing_required),
        protocol=protocol,
        obstacle_count=len(scenario.obstacles),
    )
    device = select_device(args.device)
    protocol_prototype = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=len(scenario.obstacles),
        target_speed_scale=float(config["experiments"][0]["target_speed_scale"]),
    )
    if protocol is not None:
        validate_central_capture_protocol_environment(protocol_prototype, protocol)
    if checkpoint is not None:
        policy, action_scale, _metadata = load_policy(
            checkpoint,
            protocol_prototype,
            protocol_prototype.reset(seed=args.seed),
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
                "initial_side_distance_m": float(abs(scenario.defender_positions[0, 0])),
                "scenario_kind": args.scenario,
                "layout": args.layout,
                "central_capture_protocol": (
                    central_capture_protocol_metadata(protocol) if protocol is not None else None
                ),
                "detection_range_m": float(config["task"]["pursuit"]["detection_range"]),
                "target_speed_scale": float(config["experiments"][0]["target_speed_scale"]),
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
