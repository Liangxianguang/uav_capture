"""Run the reproducible Stage 3C recurrent-MAPPO multi-seed ablation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_CHECKPOINT = PROJECT_ROOT / "results" / "target_predictor_gru_v1" / "checkpoint.pt"
METHODS = {
    "recurrent_no_prediction": {
        "bc_config": "configs/capture_radius_behavior_cloning_dev.yaml",
        "mappo_config": "configs/capture_radius_recurrent_mappo_no_prediction_pilot.yaml",
        "eval_config": "configs/capture_radius_pursuit_dev.yaml",
    },
    "recurrent_gru_prediction": {
        "bc_config": "configs/capture_radius_behavior_cloning_gru_prediction_dev.yaml",
        "mappo_config": "configs/capture_radius_recurrent_mappo_gru_prediction_pilot.yaml",
        "eval_config": "configs/capture_radius_pursuit_gru_prediction_dev.yaml",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=list(METHODS))
    parser.add_argument("--train-steps", type=int, default=65_536)
    parser.add_argument("--test-seed", type=int, default=632001)
    parser.add_argument("--episodes-per-scenario", type=int, default=100)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--evaluation-device", choices=("cuda", "cpu"), default="cpu")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def output_dir(method: str, seed: int) -> Path:
    return PROJECT_ROOT / "results" / "stage3c_formal" / method / f"seed{seed}"


def main() -> None:
    args = parse_args()
    if args.train_steps <= 0 or args.episodes_per_scenario <= 0 or args.sequence_length <= 0:
        raise ValueError("train-steps, episodes-per-scenario, and sequence-length must be positive.")
    if "recurrent_gru_prediction" in args.methods and not PREDICTION_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing frozen GRU checkpoint: {PREDICTION_CHECKPOINT}")
    protocol_path = PROJECT_ROOT / "results" / "stage3c_formal" / "protocol.json"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(
        json.dumps(
            {
                "stage": "3C_recurrent_formal_multiseed",
                "seeds": [int(seed) for seed in args.seeds],
                "methods": list(args.methods),
                "train_steps": int(args.train_steps),
                "test_seed": int(args.test_seed),
                "episodes_per_scenario": int(args.episodes_per_scenario),
                "sequence_length": int(args.sequence_length),
                "prediction_checkpoint": str(PREDICTION_CHECKPOINT.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for seed in args.seeds:
        for method in args.methods:
            specification = METHODS[method]
            root = output_dir(method, int(seed))
            if root.exists() and any(root.iterdir()):
                raise FileExistsError(f"Refusing to overwrite non-empty output: {root}")
            root.mkdir(parents=True, exist_ok=True)
            bc_output = root / "behavior_cloning_prior"
            mappo_output = root / "recurrent_mappo"
            prediction_args: list[str] = []
            if method == "recurrent_gru_prediction":
                prediction_args = [
                    "--prediction-checkpoint", str(PREDICTION_CHECKPOINT),
                    "--prediction-history-length", "8", "--prediction-horizon-index", "2",
                ]
            run(
                [
                    sys.executable, "scripts/train_capture_radius_behavior_cloning.py", "--config", specification["bc_config"],
                    "--output", str(bc_output), "--seed", str(seed), "--device", args.device, *prediction_args,
                ]
            )
            run(
                [
                    sys.executable, "scripts/train_capture_radius_recurrent_mappo.py", "--config", specification["mappo_config"],
                    "--output", str(mappo_output), "--seed", str(seed), "--total-steps", str(args.train_steps),
                    "--device", args.device, "--sequence-length", str(args.sequence_length),
                    "--initialize-from", str(bc_output / "checkpoint.pt"), *prediction_args,
                ]
            )
            for action_name, cbf_args in (("raw", []), ("cbf", ["--use-cbf"])):
                run(
                    [
                        sys.executable, "scripts/evaluate_capture_radius_mappo.py", "--config", specification["eval_config"],
                        "--checkpoint", str(mappo_output / "checkpoint.pt"), "--output", str(root / f"evaluation_{action_name}"),
                        "--seed", str(args.test_seed), "--episodes", str(args.episodes_per_scenario),
                        "--device", args.evaluation_device, *cbf_args, *prediction_args,
                    ]
                )


if __name__ == "__main__":
    main()
