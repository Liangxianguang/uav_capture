"""Replay one recorded random S3 scene and render capture media.

This is a visualization/reproducibility helper.  It restores the exact scene
metadata and episode seed from a validation ``scenes.jsonl`` file, then runs
the selected checkpoint with the same CBF option used by evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import (  # noqa: E402
    scenario_from_metadata,
    scenario_metadata,
    target_crossing_pursuit_overrides,
)
from evaluate_capture_radius_mappo import load_policy, save_trajectory, select_device  # noqa: E402
from replay_capture_radius_checkpoint import render_animation  # noqa: E402
from run_mixed_obstacle_showcase import rollout_showcase  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--environment-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--use-cbf", action="store_true")
    parser.add_argument(
        "--recurrent-reset-interval",
        type=int,
        help="Reset recurrent actor state at this many control steps; defaults to checkpoint metadata.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frame-stride", type=int, default=1)
    return parser.parse_args()


def load_scene(path: Path, episode_index: int) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = [record for record in records if int(record.get("episode_index", -1)) == episode_index]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one scene for episode {episode_index}, found {len(matches)}")
    return matches[0]


def build_config(environment_config: Path, spec: dict[str, Any], obstacle_count: int) -> dict[str, Any]:
    config = yaml.safe_load(environment_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("task"), dict):
        raise ValueError("Environment config must contain a task mapping")
    config = copy.deepcopy(config)
    pursuit = config["task"]["pursuit"]
    pursuit.update(copy.deepcopy(spec.get("pursuit_overrides", {})))
    pursuit["target_motion_mode"] = str(spec["target_motion_mode"])
    if bool(spec.get("target_crossing_required", False)):
        pursuit.update(target_crossing_pursuit_overrides())
    config["experiments"] = [
        {
            "name": "random_scene_replay",
            "episodes": 1,
            "obstacle_count": int(obstacle_count),
            "target_speed_scale": float(spec["target_speed_scale"]),
        }
    ]
    return config


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    scenes_path = args.scenes.resolve()
    environment_config = args.environment_config.resolve()
    output_dir = args.output_dir.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not scenes_path.is_file():
        raise FileNotFoundError(scenes_path)
    if not environment_config.is_file():
        raise FileNotFoundError(environment_config)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    if args.fps <= 0 or args.frame_stride <= 0:
        raise ValueError("fps and frame-stride must be positive")
    if args.recurrent_reset_interval is not None and args.recurrent_reset_interval <= 0:
        raise ValueError("recurrent-reset-interval must be positive when provided.")
    output_dir.mkdir(parents=True, exist_ok=True)

    record = load_scene(scenes_path, args.episode_index)
    spec = record["spec"]
    scenario = scenario_from_metadata(record["scenario"])
    config = build_config(environment_config, spec, len(scenario.obstacles))
    device = select_device(args.device)
    prototype = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=len(scenario.obstacles),
        target_speed_scale=float(spec["target_speed_scale"]),
    )
    policy, action_scale, checkpoint_metadata = load_policy(
        checkpoint,
        prototype,
        prototype.reset(seed=int(spec["episode_seed"])),
        device,
    )
    metadata_reset_interval = checkpoint_metadata.get("recurrent_reset_interval_steps")
    recurrent_reset_interval = (
        int(args.recurrent_reset_interval)
        if args.recurrent_reset_interval is not None
        else int(metadata_reset_interval)
        if metadata_reset_interval is not None
        else None
    )
    row, env = rollout_showcase(
        policy,
        config,
        scenario,
        seed=int(spec["episode_seed"]),
        device=device,
        action_scale=action_scale,
        use_cbf=bool(args.use_cbf),
        validate_scenario=False,
        recurrent_reset_interval=recurrent_reset_interval,
    )
    if isinstance(checkpoint_metadata, dict):
        checkpoint_metadata_summary = {
            key: value for key, value in checkpoint_metadata.items() if key != "state_dict"
        }
    else:
        checkpoint_metadata_summary = str(checkpoint_metadata)
    row.update(
        {
            "episode_index": int(record["episode_index"]),
            "episode_seed": int(spec["episode_seed"]),
            "layout_seed": int(spec["layout_seed"]),
            "split_evidence": "validation",
            "source_scenes": str(scenes_path),
            "checkpoint": str(checkpoint),
            # Checkpoint loaders may expose tensors in auxiliary metadata;
            # retain a readable provenance value without making episode.json
            # depend on PyTorch's serialization types.
            "checkpoint_metadata": checkpoint_metadata_summary,
            "device": str(device),
            "use_cbf": bool(args.use_cbf),
            "scenario_metadata": scenario_metadata(scenario),
            "recorded_outcome": record.get("outcome", {}),
        }
    )
    trajectory_path = output_dir / "trajectory.npz"
    save_trajectory(env, trajectory_path)
    (output_dir / "scenario.json").write_text(
        json.dumps({"spec": spec, "scenario": scenario_metadata(scenario), "recorded_outcome": record.get("outcome", {})}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    media = render_animation(
        trajectory_path,
        output_dir,
        title=f"V5 VALIDATION / EPISODE {args.episode_index} / {'CBF' if args.use_cbf else 'RAW'}",
        fps=args.fps,
        frame_stride=args.frame_stride,
        result=row,
    )
    row["media"] = media
    (output_dir / "episode.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps({"episode": row, "media": media}, indent=2), flush=True)


if __name__ == "__main__":
    main()
