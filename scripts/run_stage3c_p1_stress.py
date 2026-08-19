"""Run representative Stage 3C partial-observation stress tests.

This runner evaluates frozen recurrent policies only. It changes the existing
simulator's observation, communication, target-motion, and obstacle-profile
parameters without changing actor dimensions or introducing real sensor data.
Each training checkpoint is evaluated on the same locked episode seed block;
the independent statistical unit is aggregated later as the training seed.
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_capture_radius_mappo import (  # noqa: E402
    load_policy,
    load_prediction_model,
    rollout_episode,
    select_device,
    summarize,
)


METHODS = ("recurrent_no_prediction", "recurrent_gru_prediction")
ACTIONS = ("raw", "cbf")
BASE_CONFIGS = {
    "recurrent_no_prediction": PROJECT_ROOT / "configs" / "capture_radius_pursuit_dev.yaml",
    "recurrent_gru_prediction": PROJECT_ROOT / "configs" / "capture_radius_pursuit_gru_prediction_dev.yaml",
}
CHECKPOINT_ROOT = PROJECT_ROOT / "results" / "stage3c_formal"
PREDICTION_CHECKPOINT = PROJECT_ROOT / "results" / "target_predictor_gru_v1" / "checkpoint.pt"

# The first round deliberately covers different failure mechanisms while
# keeping the number of locked domains small enough for local reproduction.
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
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument("--episodes-per-condition", type=int, default=100)
    parser.add_argument("--test-seed", type=int, default=642001)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cpu")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "stage3c_p1_stress")
    return parser.parse_args()


def checkpoint_path(method: str, seed: int) -> Path:
    return CHECKPOINT_ROOT / method / f"seed{seed}" / "recurrent_mappo" / "checkpoint.pt"


def prediction_args_for(method: str) -> dict[str, Any]:
    if method == "recurrent_gru_prediction":
        return {
            "prediction_checkpoint": PREDICTION_CHECKPOINT,
            "prediction_history_length": 8,
            "prediction_horizon_index": 2,
        }
    return {
        "prediction_checkpoint": None,
        "prediction_history_length": 8,
        "prediction_horizon_index": 2,
    }


def make_config(method: str, condition_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(BASE_CONFIGS[method].read_text(encoding="utf-8"))
    condition = CONDITIONS[condition_name]
    config = copy.deepcopy(config)
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


def relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return resolved.as_posix()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must contain distinct training seeds.")
    if args.episodes_per_condition <= 0:
        raise ValueError("--episodes-per-condition must be positive.")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {args.output_root}")
    if "recurrent_gru_prediction" in args.methods and not PREDICTION_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing frozen predictor checkpoint: {PREDICTION_CHECKPOINT}")
    for method in args.methods:
        for seed in args.seeds:
            checkpoint = checkpoint_path(method, int(seed))
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing recurrent checkpoint: {checkpoint}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "stage": "3C_P1_representative_partial_observation_stress",
        "methods": list(args.methods),
        "training_seeds": [int(seed) for seed in args.seeds],
        "test_seed": int(args.test_seed),
        "episodes_per_condition": int(args.episodes_per_condition),
        "actions": list(ACTIONS),
        "conditions": {name: CONDITIONS[name] for name in args.conditions},
        "statistical_unit": "training_seed",
        "device": args.device,
        "prediction_checkpoint": relative_or_absolute(PREDICTION_CHECKPOINT)
        if "recurrent_gru_prediction" in args.methods
        else None,
    }
    args.output_root.joinpath("protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    device = select_device(args.device)
    for method in args.methods:
        prediction_kwargs = prediction_args_for(method)
        prediction_model = (
            load_prediction_model(prediction_kwargs["prediction_checkpoint"], device)
            if prediction_kwargs["prediction_checkpoint"] is not None
            else None
        )
        for seed in args.seeds:
            checkpoint = checkpoint_path(method, int(seed)).resolve()
            for condition_name in args.conditions:
                config, condition = make_config(method, condition_name)
                condition_root = args.output_root / method / f"seed{seed}" / condition_name
                condition_root.mkdir(parents=True, exist_ok=True)
                config_path = condition_root / "config.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                prototype_experiment = config["experiments"][0]
                from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv

                prototype = CaptureRadiusPursuit3DEnv(
                    config,
                    obstacle_count=int(prototype_experiment["obstacle_count"]),
                    target_speed_scale=float(prototype_experiment["target_speed_scale"]),
                )
                policy, action_scale, checkpoint_metadata = load_policy(
                    checkpoint,
                    prototype,
                    prototype.reset(seed=int(args.test_seed)),
                    device,
                )
                config["_stress_checkpoint_sha256"] = sha256(checkpoint)
                rows_by_action: dict[str, list[dict[str, Any]]] = {action: [] for action in ACTIONS}
                for action_name in ACTIONS:
                    use_cbf = action_name == "cbf"
                    rows: list[dict[str, Any]] = []
                    started = time.perf_counter()
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
                            use_cbf=use_cbf,
                            record_history=False,
                            prediction_model=prediction_model,
                            prediction_history_length=int(prediction_kwargs["prediction_history_length"]),
                            prediction_horizon_index=int(prediction_kwargs["prediction_horizon_index"]),
                        )
                        row["condition"] = condition_name
                        row["scenario"] = condition_name
                        row["method"] = method
                        row["training_seed"] = int(seed)
                        row["action"] = action_name
                        rows.append(row)
                    rows_by_action[action_name] = rows
                    with (condition_root / f"episodes_{action_name}.csv").open(
                        "w", encoding="utf-8", newline=""
                    ) as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    (condition_root / f"summary_{action_name}.json").write_text(
                        json.dumps(summarize(rows), indent=2), encoding="utf-8"
                    )
                    (condition_root / f"run_{action_name}.json").write_text(
                        json.dumps(
                            {
                                "method": method,
                                "training_seed": int(seed),
                                "condition": condition_name,
                                "action": action_name,
                                "checkpoint": relative_or_absolute(checkpoint),
                                "checkpoint_sha256": sha256(checkpoint),
                                "device": str(device),
                                "elapsed_seconds": time.perf_counter() - started,
                                "actor_recurrent": bool(checkpoint_metadata.get("actor_recurrent", False)),
                                "prediction_checkpoint": (
                                    relative_or_absolute(prediction_kwargs["prediction_checkpoint"])
                                    if prediction_kwargs["prediction_checkpoint"] is not None
                                    else None
                                ),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    print(
                        json.dumps(
                            {
                                "method": method,
                                "seed": int(seed),
                                "condition": condition_name,
                                "action": action_name,
                                "summary": summarize(rows)["overall"],
                            },
                            indent=2,
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
