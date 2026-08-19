"""Run the reproducible Stage 3B multi-seed policy ablation.

The runner deliberately executes one method/seed at a time so CUDA memory,
random seeds, and output artifacts remain auditable. It does not aggregate
results; ``aggregate_stage3b_formal.py`` consumes the generated summaries.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_CHECKPOINT = PROJECT_ROOT / "results" / "target_predictor_gru_v1" / "checkpoint.pt"

METHODS = {
    "no_prediction": {
        "bc_config": "configs/capture_radius_behavior_cloning_dev.yaml",
        "mappo_config": "configs/capture_radius_mappo_warmstart_dev.yaml",
        "eval_config": "configs/capture_radius_pursuit_dev.yaml",
    },
    "constant_velocity": {
        "bc_config": "configs/capture_radius_behavior_cloning_prediction_dev.yaml",
        "mappo_config": "configs/capture_radius_mappo_prediction_warmstart_dev.yaml",
        "eval_config": "configs/capture_radius_pursuit_prediction_dev.yaml",
    },
    "gru_prediction": {
        "bc_config": "configs/capture_radius_behavior_cloning_gru_prediction_dev.yaml",
        "mappo_config": "configs/capture_radius_mappo_gru_prediction_pilot.yaml",
        "eval_config": "configs/capture_radius_pursuit_gru_prediction_dev.yaml",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHODS),
        default=list(METHODS),
    )
    parser.add_argument("--train-steps", type=int, default=65_536)
    parser.add_argument("--test-seed", type=int, default=632001)
    parser.add_argument("--episodes-per-scenario", type=int, default=100)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--evaluation-device", choices=("cuda", "cpu"), default="cpu")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def output_dir(method: str, seed: int) -> Path:
    return PROJECT_ROOT / "results" / "stage3b_formal" / method / f"seed{seed}"


def ensure_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.train_steps <= 0 or args.episodes_per_scenario <= 0:
        raise ValueError("train-steps and episodes-per-scenario must be positive.")
    if "gru_prediction" in args.methods and not PREDICTION_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing frozen GRU checkpoint: {PREDICTION_CHECKPOINT}")

    manifest = {
        "stage": "3B_formal_multiseed",
        "seeds": [int(seed) for seed in args.seeds],
        "methods": list(args.methods),
        "train_steps": int(args.train_steps),
        "test_seed": int(args.test_seed),
        "episodes_per_scenario": int(args.episodes_per_scenario),
        "prediction_checkpoint": str(PREDICTION_CHECKPOINT.resolve()),
    }
    manifest_path = PROJECT_ROOT / "results" / "stage3b_formal" / "protocol.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for seed in args.seeds:
        for method in args.methods:
            specification = METHODS[method]
            root = output_dir(method, int(seed))
            ensure_empty(root)
            bc_output = root / "behavior_cloning"
            mappo_output = root / "mappo"
            train_prediction_args = []
            if method == "gru_prediction":
                train_prediction_args = [
                    "--prediction-checkpoint",
                    str(PREDICTION_CHECKPOINT),
                    "--prediction-history-length",
                    "8",
                    "--prediction-horizon-index",
                    "2",
                ]
            run(
                [
                    sys.executable,
                    "scripts/train_capture_radius_behavior_cloning.py",
                    "--config",
                    specification["bc_config"],
                    "--output",
                    str(bc_output),
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                    *train_prediction_args,
                ]
            )
            run(
                [
                    sys.executable,
                    "scripts/train_capture_radius_mappo.py",
                    "--config",
                    specification["mappo_config"],
                    "--output",
                    str(mappo_output),
                    "--seed",
                    str(seed),
                    "--total-steps",
                    str(args.train_steps),
                    "--device",
                    args.device,
                    "--initialize-from",
                    str(bc_output / "checkpoint.pt"),
                    *train_prediction_args,
                ]
            )
            for action_name, cbf_args in (("raw", []), ("cbf", ["--use-cbf"])):
                evaluation_output = root / f"evaluation_{action_name}"
                run(
                    [
                        sys.executable,
                        "scripts/evaluate_capture_radius_mappo.py",
                        "--config",
                        specification["eval_config"],
                        "--checkpoint",
                        str(mappo_output / "checkpoint.pt"),
                        "--output",
                        str(evaluation_output),
                        "--seed",
                        str(args.test_seed),
                        "--episodes",
                        str(args.episodes_per_scenario),
                        "--device",
                        args.evaluation_device,
                        *cbf_args,
                        *train_prediction_args,
                    ]
                )


if __name__ == "__main__":
    main()
