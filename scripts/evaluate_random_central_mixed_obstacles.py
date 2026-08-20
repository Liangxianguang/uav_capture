"""Evaluate a policy or rule expert on reproducible S3 random central maps.

Each episode has separate motion and layout seeds.  A map is accepted only
after spawn, overlap, boundary, and conservative route checks pass.  This is a
controlled randomized-layout protocol, not a replacement for prior formal
locked benchmarks.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import (  # noqa: E402
    configure_target_crossing_episode,
    random_central_mixed_obstacle_scenario,
    scenario_metadata,
)
from evaluate_capture_radius_mappo import load_policy, select_device  # noqa: E402
from run_mixed_obstacle_showcase import (  # noqa: E402
    build_config,
    rollout_showcase,
    rollout_showcase_expert,
)


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "central_random_mixed_obstacle_s3_protocol.yaml"
REQUIRED_SPLITS = ("train", "validation", "locked_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("f1", "f2"), default="f2")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--baseline", choices=("dynamic_encirclement",))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--environment-config",
        type=Path,
        help="Optional environment YAML. Use the V4 YAML for shape-aware V4 checkpoints.",
    )
    parser.add_argument("--split", choices=REQUIRED_SPLITS, required=True)
    parser.add_argument("--episodes", type=int, help="Optional split-size override for smoke runs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--use-cbf", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    return parser.parse_args()


def load_protocol(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("S3 protocol YAML must be a mapping.")
    seed_blocks = document.get("seed_blocks")
    episodes_per_split = document.get("episodes_per_split")
    settings = document.get("s3")
    if not isinstance(seed_blocks, dict) or not isinstance(episodes_per_split, dict) or not isinstance(settings, dict):
        raise ValueError("S3 protocol requires seed_blocks, episodes_per_split, and s3 mappings.")
    missing = [name for name in REQUIRED_SPLITS if name not in seed_blocks or name not in episodes_per_split]
    if missing:
        raise ValueError(f"S3 protocol is missing split settings: {', '.join(missing)}")
    seeds = [int(seed_blocks[name]) for name in REQUIRED_SPLITS]
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("S3 seed blocks must be distinct non-negative integers.")
    if any(int(episodes_per_split[name]) <= 0 for name in REQUIRED_SPLITS):
        raise ValueError("S3 episodes_per_split values must be positive.")
    for name in (
        "obstacle_count_range",
        "initial_side_distances",
        "defender_sides",
        "target_speed_scales",
        "target_crossing_speed_scales",
        "target_motion_modes",
    ):
        if not isinstance(settings.get(name), list) or not settings[name]:
            raise ValueError(f"S3 protocol s3.{name} must be a non-empty list.")
    observations = settings.get("observation_conditions")
    if not isinstance(observations, list) or not observations or not all(isinstance(item, dict) for item in observations):
        raise ValueError("S3 protocol s3.observation_conditions must be a non-empty list of mappings.")
    if any(not isinstance(item.get("name"), str) or not isinstance(item.get("pursuit_overrides"), dict) for item in observations):
        raise ValueError("Each S3 observation condition requires name and pursuit_overrides.")
    if not isinstance(settings.get("target_crossing_required", False), bool):
        raise ValueError("S3 protocol s3.target_crossing_required must be boolean.")
    return document


def resolved_episode_count(protocol: dict[str, Any], split: str, override: int | None) -> int:
    configured = int(protocol["episodes_per_split"][split])
    if split == "locked_test" and override is not None and int(override) != configured:
        raise ValueError(f"Locked test requires exactly {configured} episodes; got {override}.")
    episodes = int(override) if override is not None else configured
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    return episodes


def episode_spec(protocol: dict[str, Any], split: str, episode_index: int) -> dict[str, Any]:
    settings = protocol["s3"]
    episode_seed = int(protocol["seed_blocks"][split]) + int(episode_index)
    layout_seed = int(protocol["seed_blocks"][split]) + 1_000_000 + int(episode_index)
    minimum_count, maximum_count = (int(value) for value in settings["obstacle_count_range"])
    conditions = list(
        itertools.product(
            range(minimum_count, maximum_count + 1),
            settings["defender_sides"],
            settings["initial_side_distances"],
            settings["target_speed_scales"],
            settings["target_motion_modes"],
            settings["observation_conditions"],
        )
    )
    # The full factorial table is shuffled once per split, then indexed by
    # episode.  Unlike synchronized modulo counters, this prevents the
    # direction, sensing condition, and target behavior from becoming aliases
    # of each other while remaining fully reproducible.
    order = np.random.default_rng(int(protocol["seed_blocks"][split]) + 2_000_000).permutation(len(conditions))
    obstacle_count, defender_side, initial_side_distance, target_speed_scale, target_motion_mode, observation_condition = (
        conditions[int(order[episode_index % len(order)])]
    )
    target_crossing_required = bool(settings.get("target_crossing_required", False))
    if target_crossing_required:
        crossing_speeds = [float(value) for value in settings["target_crossing_speed_scales"]]
        target_speed_scale = crossing_speeds[int(order[episode_index % len(order)]) % len(crossing_speeds)]
    observation_condition = dict(observation_condition)
    return {
        "episode_seed": episode_seed,
        "layout_seed": layout_seed,
        "defender_side": str(defender_side),
        "initial_side_distance": float(initial_side_distance),
        "target_speed_scale": float(target_speed_scale),
        "target_motion_mode": str(target_motion_mode),
        "target_crossing_required": target_crossing_required,
        "observation_condition": str(observation_condition["name"]),
        "pursuit_overrides": dict(observation_condition["pursuit_overrides"]),
        "obstacle_count": int(obstacle_count),
        "condition_index": int(order[episode_index % len(order)]),
        "condition_table_size": len(conditions),
    }


def config_for_spec(
    method: str,
    spec: dict[str, Any],
    environment_config: Path | None = None,
) -> dict[str, Any]:
    """Construct one S3 rollout environment while preserving a checkpoint contract."""
    if environment_config is None:
        config = build_config(
            method,
            float(spec["pursuit_overrides"]["detection_range"]),
            float(spec["target_speed_scale"]),
            target_crossing_required=bool(spec["target_crossing_required"]),
        )
    else:
        config = yaml.safe_load(environment_config.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not isinstance(config.get("task"), dict):
            raise ValueError("--environment-config must contain a valid pursuit environment mapping.")
        config = copy.deepcopy(config)
    pursuit = config["task"]["pursuit"]
    pursuit.update(copy.deepcopy(spec["pursuit_overrides"]))
    pursuit["target_motion_mode"] = str(spec["target_motion_mode"])
    if bool(spec["target_crossing_required"]):
        configure_target_crossing_episode_from_config(config)
    return config


def configure_target_crossing_episode_from_config(config: dict[str, Any]) -> None:
    """Apply the frozen crossing profile before constructing an environment."""
    from encirclement3d.showcase import target_crossing_pursuit_overrides

    config["task"]["pursuit"].update(target_crossing_pursuit_overrides())


def layout_signature(metadata: dict[str, Any]) -> str:
    counts = Counter(str(item["shape"]) for item in metadata["obstacles"])
    return "+".join(f"{shape}{counts[shape]}" for shape in ("cylinder", "box", "wall") if counts[shape])


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty S3 evaluation.")

    def metrics(subset: list[dict[str, Any]]) -> dict[str, Any]:
        capture_times = [float(row["capture_time_seconds"]) for row in subset if row["capture_time_seconds"] is not None]
        return {
            "episodes": len(subset),
            "safe_capture_rate": float(np.mean([bool(row["safe_capture_success"]) for row in subset])),
            "safe_capture_in_pursuit_rate": float(np.mean([bool(row["safe_capture_in_pursuit"]) for row in subset])),
            "capture_rate": float(np.mean([bool(row["capture_event"]) for row in subset])),
            "showcase_success_rate": float(np.mean([bool(row["showcase_success"]) for row in subset])),
            "target_zone_entry_rate": float(np.mean([float(row["target_zone_entry_rate"]) for row in subset])),
            "defender_zone_entry_rate": float(np.mean([float(row["defender_zone_entry_rate"]) for row in subset])),
            "defender_crossing_rate": float(np.mean([float(row["defender_crossing_rate"]) for row in subset])),
            "all_defenders_crossing_rate": float(np.mean([bool(row["all_defenders_crossed"]) for row in subset])),
            "target_crossing_rate": float(np.mean([float(row["target_crossing_rate"]) for row in subset])),
            "transit_route_feasible_rate": float(np.mean([bool(row["transit_route_feasible"]) for row in subset])),
            "transit_success_rate": float(np.mean([bool(row["transit_success"]) for row in subset])),
            "collision_rate": float(np.mean([bool(row["collision"]) for row in subset])),
            "boundary_violation_rate": float(np.mean([int(row["world_violation_steps"]) > 0 for row in subset])),
            "mean_min_clearance_m": float(np.mean([float(row["min_clearance_m"]) for row in subset])),
            "worst_min_clearance_m": float(min(float(row["min_clearance_m"]) for row in subset)),
            "mean_capture_time_seconds": float(np.mean(capture_times)) if capture_times else None,
            "mean_visible_fraction": float(np.mean([float(row["mean_visible_fraction"]) for row in subset])),
            "mean_observation_age_steps": float(np.mean([float(row["mean_observation_age_steps"]) for row in subset])),
            "mean_defender_path_length_m": float(
                np.mean([float(row.get("mean_defender_path_length_m", 0.0)) for row in subset])
            ),
            "mean_total_defender_path_length_m": float(
                np.mean([float(row.get("total_defender_path_length_m", 0.0)) for row in subset])
            ),
            "mean_cbf_action_correction_norm": float(
                np.mean([float(row.get("mean_cbf_action_correction_norm", 0.0)) for row in subset])
            ),
            "max_cbf_action_correction_norm": float(
                max(float(row.get("max_cbf_action_correction_norm", 0.0)) for row in subset)
            ),
            "termination_reasons": dict(sorted(Counter(str(row["termination_reason"]) for row in subset).items())),
        }

    grouped: dict[str, dict[str, Any]] = {}
    for field in ("defender_side", "obstacle_count", "layout_signature", "target_speed_scale", "observation_condition", "target_motion_mode"):
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[str(row[field])].append(row)
        grouped[f"by_{field}"] = {key: metrics(value) for key, value in sorted(buckets.items())}
    return {"overall": metrics(rows), **grouped}


def main() -> None:
    args = parse_args()
    if (args.checkpoint is None) == (args.baseline is None):
        raise ValueError("Provide exactly one of --checkpoint or --baseline.")
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    environment_config = args.environment_config.resolve() if args.environment_config is not None else None
    if environment_config is not None and not environment_config.is_file():
        raise FileNotFoundError(f"Environment config does not exist: {environment_config}")
    episodes = resolved_episode_count(protocol, args.split, args.episodes)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    policy: Any = None
    action_scale: float | None = None
    rows: list[dict[str, Any]] = []
    scenes: list[dict[str, Any]] = []
    for episode_index in range(episodes):
        spec = episode_spec(protocol, args.split, episode_index)
        config = config_for_spec(args.method, spec, environment_config)
        validation_env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=float(spec["target_speed_scale"]))
        scenario = random_central_mixed_obstacle_scenario(
            validation_env,
            layout_seed=int(spec["layout_seed"]),
            initial_side_distance=float(spec["initial_side_distance"]),
            defender_side=str(spec["defender_side"]),
            target_crossing_required=bool(spec["target_crossing_required"]),
            obstacle_count_range=(int(spec["obstacle_count"]), int(spec["obstacle_count"])),
            max_attempts=int(protocol["s3"].get("max_sampling_attempts", 500)),
            required_defender_zone_entries=int(protocol["s3"].get("required_defender_zone_entries", 1)),
        )
        if bool(scenario.target_crossing_required):
            # Keep the rollout environment and the scenario protocol aligned;
            # config_for_spec already applies the same values to its policy
            # prototype, while this object is the actual episode environment.
            configure_target_crossing_episode(validation_env)
        if checkpoint is not None and policy is None:
            policy, action_scale, _metadata = load_policy(
                checkpoint,
                validation_env,
                validation_env.reset(seed=int(spec["episode_seed"])),
                device,
            )
        if checkpoint is None:
            row, _env = rollout_showcase_expert(
                config, scenario, seed=int(spec["episode_seed"]), use_cbf=bool(args.use_cbf)
            )
        else:
            assert action_scale is not None
            row, _env = rollout_showcase(
                policy,
                config,
                scenario,
                seed=int(spec["episode_seed"]),
                device=device,
                action_scale=action_scale,
                use_cbf=bool(args.use_cbf),
            )
        metadata = scenario_metadata(scenario)
        row.update(
            {
                "split": args.split,
                "episode_index": episode_index,
                "episode_seed": int(spec["episode_seed"]),
                "layout_seed": int(spec["layout_seed"]),
                "defender_side": str(spec["defender_side"]),
                "initial_side_distance_m": float(spec["initial_side_distance"]),
                "target_speed_scale": float(spec["target_speed_scale"]),
                "target_motion_mode": str(spec["target_motion_mode"]),
                "target_crossing_required": bool(spec["target_crossing_required"]),
                "observation_condition": str(spec["observation_condition"]),
                "condition_index": int(spec["condition_index"]),
                "obstacle_count": len(scenario.obstacles),
                "layout_signature": layout_signature(metadata),
                "method": args.method if checkpoint is not None else str(args.baseline),
                "checkpoint": str(checkpoint) if checkpoint is not None else None,
                "device": str(device),
            }
        )
        rows.append(row)
        scenes.append(
            {
                "episode_index": episode_index,
                "spec": spec,
                "scenario": metadata,
                "outcome": {
                    key: row[key]
                    for key in (
                        "safe_capture_success",
                        "safe_capture_in_pursuit",
                        "capture_event",
                        "collision",
                        "world_violation_steps",
                        "termination_reason",
                        "task_termination_reason",
                        "target_zone_entered",
                        "defender_zone_entered",
                        "target_crossed",
                        "defender_crossed",
                        "transit_route_feasible",
                        "transit_success",
                    )
                },
            }
        )

    summary = summarize_rows(rows)
    with output_dir.joinpath("episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output_dir.joinpath("scenes.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in scenes), encoding="utf-8"
    )
    output_dir.joinpath("summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_dir.joinpath("protocol.yaml").write_text(protocol_path.read_text(encoding="utf-8"), encoding="utf-8")
    output_dir.joinpath("evaluation_metadata.json").write_text(
        json.dumps(
            {
                "evaluation_type": (
                    "randomized_central_mixed_obstacle_s3_locked_test"
                    if args.split == "locked_test"
                    else "randomized_central_mixed_obstacle_s3_validation"
                ),
                "not_a_locked_test": args.split != "locked_test",
                "locked_test": args.split == "locked_test",
                "seed_block": int(protocol["seed_blocks"][args.split]),
                "protocol": str(protocol_path),
                "environment_config": str(environment_config) if environment_config is not None else None,
                "split": args.split,
                "episodes": episodes,
                "method": args.method if checkpoint is not None else args.baseline,
                "checkpoint": str(checkpoint) if checkpoint is not None else None,
                "use_cbf": bool(args.use_cbf),
                "device": str(device),
                "separate_episode_and_layout_seeds": True,
                "condition_table_size": int(scenes[0]["spec"]["condition_table_size"]),
                "wall_orientation_contract": "axis_aligned_0_or_90_degrees",
                "task_contract": "central_encounter_capture_and_independent_transit",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
