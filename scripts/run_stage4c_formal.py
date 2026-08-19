"""Run the reproducible Stage 4C F1 belief-aware recurrent-MAPPO experiment.

F1 changes only the target-belief update used by the existing 44-dimensional
actor: delayed packets are time-aligned and stale velocity is age-gated. The
training protocol, reward, network size, and scenario distribution remain the
same as the Stage 3C recurrent baseline. This runner trains the three formal
F1 seeds and evaluates raw and local-CBF actions on the four locked stress
conditions used by P1.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from evaluate_capture_radius_mappo import (  # noqa: E402
    load_policy,
    rollout_episode,
    select_device,
    summarize,
)


METHOD = "f1_time_aligned_belief"
ACTIONS = ("raw", "cbf")
CONDITIONS: dict[str, dict[str, Any]] = {
    "nominal_partial_observation": {
        "obstacle_count": 3,
        "target_speed_scale": 0.75,
        "pursuit": {
            "target_motion_mode": "random_turn",
            "obstacle_profile": "mixed",
            "map_seed_offset": 500000,
        },
    },
    "delayed_measurements": {
        "obstacle_count": 3,
        "target_speed_scale": 0.75,
        "pursuit": {
            "target_motion_mode": "random_turn",
            "obstacle_profile": "mixed",
            "observation_delay_steps": 3,
            "message_delay_steps": 5,
            "message_dropout_probability": 0.10,
            "map_seed_offset": 510000,
        },
    },
    "burst_occlusion": {
        "obstacle_count": 5,
        "target_speed_scale": 1.00,
        "pursuit": {
            "target_motion_mode": "s_curve",
            "obstacle_profile": "boxes",
            "detection_dropout_probability": 0.25,
            "detection_loss_burst_probability": 0.20,
            "detection_loss_burst_duration_steps": 5,
            "observation_delay_steps": 2,
            "map_seed_offset": 520000,
        },
    },
    "communication_loss": {
        "obstacle_count": 5,
        "target_speed_scale": 1.00,
        "pursuit": {
            "target_motion_mode": "burst",
            "obstacle_profile": "narrow_channels",
            "target_burst_period_steps": 30,
            "target_burst_duration_steps": 8,
            "message_delay_steps": 6,
            "message_dropout_probability": 0.20,
            "communication_link_dropout_probability": 0.15,
            "map_seed_offset": 530000,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[521001, 521002, 521003])
    parser.add_argument("--train-steps", type=int, default=65_536)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--episodes-per-condition", type=int, default=100)
    parser.add_argument("--test-seed", type=int, default=642001)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--evaluation-device", choices=("cuda", "cpu", "auto"), default="cpu")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "stage4c_formal",
    )
    parser.add_argument("--skip-training", action="store_true", help="Evaluate existing F1 checkpoints only.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return resolved.as_posix()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def f1_environment_path() -> Path:
    return PROJECT_ROOT / "configs" / "capture_radius_pursuit_time_aligned_belief_dev.yaml"


def checkpoint_path(root: Path, seed: int) -> Path:
    return root / METHOD / f"seed{seed}" / "recurrent_mappo" / "checkpoint.pt"


def make_config(condition_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(f1_environment_path().read_text(encoding="utf-8"))
    condition = copy.deepcopy(CONDITIONS[condition_name])
    config["task"]["pursuit"].update(copy.deepcopy(condition["pursuit"]))
    config["experiments"] = [
        {
            "name": condition_name,
            "episodes": 1,
            "obstacle_count": int(condition["obstacle_count"]),
            "target_speed_scale": float(condition["target_speed_scale"]),
        }
    ]
    return config, condition


def train_seed(args: argparse.Namespace, root: Path, seed: int) -> None:
    seed_root = root / METHOD / f"seed{seed}"
    if seed_root.exists() and any(seed_root.iterdir()):
        if args.skip_training:
            return
        raise FileExistsError(f"Refusing to overwrite non-empty output: {seed_root}")
    seed_root.mkdir(parents=True, exist_ok=True)
    bc_output = seed_root / "behavior_cloning_prior"
    mappo_output = seed_root / "recurrent_mappo"
    bc_config = PROJECT_ROOT / "configs" / "capture_radius_behavior_cloning_time_aligned_belief_dev.yaml"
    mappo_config = PROJECT_ROOT / "configs" / "capture_radius_recurrent_mappo_time_aligned_belief.yaml"
    run(
        [
            sys.executable,
            "scripts/train_capture_radius_behavior_cloning.py",
            "--config",
            str(bc_config),
            "--output",
            str(bc_output),
            "--seed",
            str(seed),
            "--device",
            args.device,
        ]
    )
    run(
        [
            sys.executable,
            "scripts/train_capture_radius_recurrent_mappo.py",
            "--config",
            str(mappo_config),
            "--output",
            str(mappo_output),
            "--seed",
            str(seed),
            "--total-steps",
            str(args.train_steps),
            "--device",
            args.device,
            "--sequence-length",
            str(args.sequence_length),
            "--initialize-from",
            str(bc_output / "checkpoint.pt"),
        ]
    )


def evaluate_seed(args: argparse.Namespace, root: Path, seed: int, device: torch.device) -> None:
    checkpoint = checkpoint_path(root, seed).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing F1 checkpoint: {checkpoint}")
    for condition_name in CONDITIONS:
        config, condition = make_config(condition_name)
        condition_root = root / METHOD / f"seed{seed}" / condition_name
        if condition_root.exists() and any(condition_root.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty output: {condition_root}")
        condition_root.mkdir(parents=True, exist_ok=True)
        (condition_root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        experiment = config["experiments"][0]
        prototype = CaptureRadiusPursuit3DEnv(
            config,
            obstacle_count=int(experiment["obstacle_count"]),
            target_speed_scale=float(experiment["target_speed_scale"]),
        )
        policy, action_scale, checkpoint_metadata = load_policy(
            checkpoint,
            prototype,
            prototype.reset(seed=int(args.test_seed)),
            device,
        )
        for action_name in ACTIONS:
            started = time.perf_counter()
            rows: list[dict[str, Any]] = []
            for episode_index in range(args.episodes_per_condition):
                episode_seed = int(args.test_seed) + episode_index
                row, _env = rollout_episode(
                    policy,
                    config,
                    obstacle_count=int(condition["obstacle_count"]),
                    target_speed_scale=float(condition["target_speed_scale"]),
                    seed=episode_seed,
                    device=device,
                    action_scale=action_scale,
                    use_cbf=action_name == "cbf",
                    record_history=episode_index == 0,
                )
                row.update(
                    {
                        "condition": condition_name,
                        "scenario": condition_name,
                        "method": METHOD,
                        "training_seed": int(seed),
                        "action": action_name,
                    }
                )
                rows.append(row)
            summary = summarize(rows)
            with (condition_root / f"episodes_{action_name}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (condition_root / f"summary_{action_name}.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            (condition_root / f"run_{action_name}.json").write_text(
                json.dumps(
                    {
                        "method": METHOD,
                        "training_seed": int(seed),
                        "condition": condition_name,
                        "action": action_name,
                        "checkpoint": relative_path(checkpoint),
                        "checkpoint_sha256": sha256(checkpoint),
                        "device": str(device),
                        "episodes": int(args.episodes_per_condition),
                        "elapsed_seconds": time.perf_counter() - started,
                        "actor_recurrent": bool(checkpoint_metadata.get("actor_recurrent", False)),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "method": METHOD,
                        "seed": int(seed),
                        "condition": condition_name,
                        "action": action_name,
                        "summary": summary["overall"],
                    },
                    indent=2,
                ),
                flush=True,
            )


def write_protocol(args: argparse.Namespace, root: Path) -> None:
    protocol = {
        "stage": "4C_F1_belief_aware_recurrent_mappo_formal",
        "method": METHOD,
        "training_seeds": [int(seed) for seed in args.seeds],
        "test_seed": int(args.test_seed),
        "train_steps": int(args.train_steps),
        "sequence_length": int(args.sequence_length),
        "episodes_per_condition": int(args.episodes_per_condition),
        "actions": list(ACTIONS),
        "conditions": CONDITIONS,
        "statistical_unit": "training_seed",
        "training_device": args.device,
        "evaluation_device": args.evaluation_device,
        "belief_update_mode": "time_aligned",
        "belief_stale_velocity_decay": 0.80,
        "belief_velocity_decay_start_age_steps": 3,
        "source_hashes": {
            relative_path(path): sha256(path)
            for path in (
                PROJECT_ROOT / "scripts" / "run_stage4c_formal.py",
                PROJECT_ROOT / "scripts" / "train_capture_radius_behavior_cloning.py",
                PROJECT_ROOT / "scripts" / "train_capture_radius_recurrent_mappo.py",
                PROJECT_ROOT / "scripts" / "evaluate_capture_radius_mappo.py",
                PROJECT_ROOT / "src" / "encirclement3d" / "learning.py",
                PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_env.py",
                f1_environment_path(),
            )
        },
    }
    root.joinpath("protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must contain distinct training seeds.")
    if args.train_steps <= 0 or args.sequence_length <= 0 or args.episodes_per_condition <= 0:
        raise ValueError("train-steps, sequence-length, and episodes-per-condition must be positive.")
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    write_protocol(args, root)
    for seed in args.seeds:
        train_seed(args, root, int(seed))
    evaluation_device = select_device(args.evaluation_device)
    for seed in args.seeds:
        evaluate_seed(args, root, int(seed), evaluation_device)


if __name__ == "__main__":
    main()
