"""Run the formal F2 ablation: F1 plus explicit belief uncertainty features."""

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

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from evaluate_capture_radius_mappo import load_policy, rollout_episode, select_device, summarize  # noqa: E402
from run_stage4c_formal import CONDITIONS  # noqa: E402

METHOD = "f2_uncertainty_features"
ACTIONS = ("raw", "cbf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[521001, 521002, 521003])
    parser.add_argument("--train-steps", type=int, default=65_536)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--episodes-per-condition", type=int, default=100)
    parser.add_argument("--test-seed", type=int, default=646001)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--evaluation-device", choices=("cuda", "cpu", "auto"), default="cpu")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "stage4d_formal")
    parser.add_argument("--skip-training", action="store_true")
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


def env_config_path() -> Path:
    return PROJECT_ROOT / "configs" / "capture_radius_pursuit_time_aligned_uncertainty_dev.yaml"


def checkpoint_path(root: Path, seed: int) -> Path:
    return root / METHOD / f"seed{seed}" / "recurrent_mappo" / "checkpoint.pt"


def make_config(condition_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(env_config_path().read_text(encoding="utf-8"))
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
    run(
        [
            sys.executable,
            "scripts/train_capture_radius_behavior_cloning.py",
            "--config",
            str(PROJECT_ROOT / "configs" / "capture_radius_behavior_cloning_time_aligned_uncertainty_dev.yaml"),
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
            str(PROJECT_ROOT / "configs" / "capture_radius_recurrent_mappo_time_aligned_uncertainty.yaml"),
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


def evaluate_seed(args: argparse.Namespace, root: Path, seed: int, device: Any) -> None:
    checkpoint = checkpoint_path(root, seed).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing F2 checkpoint: {checkpoint}")
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
            checkpoint, prototype, prototype.reset(seed=int(args.test_seed)), device
        )
        for action_name in ACTIONS:
            started = time.perf_counter()
            rows: list[dict[str, Any]] = []
            for episode_index in range(args.episodes_per_condition):
                row, _env = rollout_episode(
                    policy,
                    config,
                    obstacle_count=int(condition["obstacle_count"]),
                    target_speed_scale=float(condition["target_speed_scale"]),
                    seed=int(args.test_seed) + episode_index,
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
            (condition_root / f"summary_{action_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
            print(json.dumps({"method": METHOD, "seed": seed, "condition": condition_name, "action": action_name, "summary": summary["overall"]}, indent=2), flush=True)


def write_protocol(args: argparse.Namespace, root: Path) -> None:
    source_paths = (
        PROJECT_ROOT / "scripts" / "run_stage4d_formal.py",
        PROJECT_ROOT / "scripts" / "run_stage4c_formal.py",
        PROJECT_ROOT / "scripts" / "train_capture_radius_behavior_cloning.py",
        PROJECT_ROOT / "scripts" / "train_capture_radius_recurrent_mappo.py",
        PROJECT_ROOT / "scripts" / "evaluate_capture_radius_mappo.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "learning.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_env.py",
        env_config_path(),
    )
    root.joinpath("protocol.json").write_text(
        json.dumps(
            {
                "stage": "4D_F2_uncertainty_aware_recurrent_mappo_formal",
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
                "include_uncertainty_features": True,
                "source_hashes": {relative_path(path): sha256(path) for path in source_paths},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


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
    device = select_device(args.evaluation_device)
    for seed in args.seeds:
        evaluate_seed(args, root, int(seed), device)


if __name__ == "__main__":
    main()
