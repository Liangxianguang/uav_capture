"""Re-evaluate frozen Stage 4 F1/F2 checkpoints on the common P1 test block.

F1 and F2 must be compared with D/E on the same locked episodes. This runner
does not train or select models: it replays already frozen checkpoints on the
P1 test seed (642001 by default), preserving the four condition definitions,
the 100 episodes per condition, and raw versus local-CBF action evaluation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
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
from evaluate_capture_radius_mappo import load_policy, rollout_episode, select_device, summarize  # noqa: E402
from run_stage4c_formal import ACTIONS, CONDITIONS  # noqa: E402


METHODS = {
    "f1": {
        "name": "f1_time_aligned_belief",
        "environment_config": "configs/capture_radius_pursuit_time_aligned_belief_dev.yaml",
    },
    "f2": {
        "name": "f2_uncertainty_features",
        "environment_config": "configs/capture_radius_pursuit_time_aligned_uncertainty_dev.yaml",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[521001, 521002, 521003])
    parser.add_argument("--test-seed", type=int, default=642001)
    parser.add_argument("--episodes-per-condition", type=int, default=100)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return resolved.as_posix()


def checkpoint_path(root: Path, method_name: str, seed: int) -> Path:
    return root / method_name / f"seed{seed}" / "recurrent_mappo" / "checkpoint.pt"


def make_config(environment_path: Path, condition_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
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


def write_protocol(args: argparse.Namespace, output_root: Path, checkpoint_root: Path, environment_path: Path) -> None:
    method = METHODS[args.method]
    source_paths = (
        PROJECT_ROOT / "scripts" / "run_stage4_common_locked_evaluation.py",
        PROJECT_ROOT / "scripts" / "evaluate_capture_radius_mappo.py",
        PROJECT_ROOT / "scripts" / "run_stage4c_formal.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "learning.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_env.py",
        environment_path,
    )
    checkpoint_hashes = {
        str(seed): sha256(checkpoint_path(checkpoint_root, method["name"], int(seed))) for seed in args.seeds
    }
    output_root.joinpath("protocol.json").write_text(
        json.dumps(
            {
                "stage": "4E_common_locked_checkpoint_replay",
                "method": method["name"],
                "training_seeds": [int(seed) for seed in args.seeds],
                "test_seed": int(args.test_seed),
                "episodes_per_condition": int(args.episodes_per_condition),
                "actions": list(ACTIONS),
                "conditions": CONDITIONS,
                "statistical_unit": "training_seed",
                "evaluation_device": args.device,
                "checkpoint_root": relative_path(checkpoint_root),
                "checkpoint_sha256_by_training_seed": checkpoint_hashes,
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
    if args.episodes_per_condition <= 0:
        raise ValueError("--episodes-per-condition must be positive.")
    method = METHODS[args.method]
    checkpoint_root = args.checkpoint_root.resolve()
    output_root = args.output_root.resolve()
    environment_path = (PROJECT_ROOT / method["environment_config"]).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_root}")
    for seed in args.seeds:
        checkpoint = checkpoint_path(checkpoint_root, method["name"], int(seed))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing frozen checkpoint: {checkpoint}")
    output_root.mkdir(parents=True, exist_ok=True)
    write_protocol(args, output_root, checkpoint_root, environment_path)
    device = select_device(args.device)
    for seed in args.seeds:
        checkpoint = checkpoint_path(checkpoint_root, method["name"], int(seed)).resolve()
        for condition_name in CONDITIONS:
            config, condition = make_config(environment_path, condition_name)
            condition_root = output_root / method["name"] / f"seed{seed}" / condition_name
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
                            "method": method["name"],
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
                            "method": method["name"],
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
                            "method": method["name"],
                            "seed": int(seed),
                            "condition": condition_name,
                            "action": action_name,
                            "summary": summary["overall"],
                        },
                        indent=2,
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
