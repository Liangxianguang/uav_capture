"""Audit whether action-conditioned JEPA predictions respond to candidate actions.

This is a model-plumbing diagnostic, not a safety certificate.  It perturbs the
final action token of held-out validation samples and reports sensitivity,
antisymmetry, and finite-output rates for every prediction horizon.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.prediction import build_action_conditioned_predictor


def evaluate(checkpoint_path: Path, dataset_path: Path, metadata_path: Path, sample_count: int, perturbation: float, device_name: str) -> dict[str, Any]:
    arrays = np.load(dataset_path)
    inputs = np.asarray(arrays["inputs"], dtype=np.float32)
    actions = np.asarray(arrays["action_history"], dtype=np.float32).copy()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_action_conditioned_predictor(str(checkpoint["model_type"]), checkpoint["model"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda" if device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available()) else "cpu")
    model = model.to(device).eval()
    rng = np.random.default_rng(20260902)
    count = min(int(sample_count), inputs.shape[0])
    indices = rng.choice(inputs.shape[0], size=count, replace=False)
    x = torch.as_tensor(inputs[indices], device=device)
    a = torch.as_tensor(actions[indices], device=device)
    plus = a.clone()
    minus = a.clone()
    # Probe each action axis separately so cancellation cannot hide sensitivity.
    axis_results: list[dict[str, Any]] = []
    with torch.no_grad():
        base_mean, base_logvar, _ = model(x, a)
        base_std = torch.exp(0.5 * base_logvar)
        for axis in range(actions.shape[-1]):
            plus[:, -1, axis] += float(perturbation)
            minus[:, -1, axis] -= float(perturbation)
            plus_mean, plus_logvar, _ = model(x, plus)
            minus_mean, minus_logvar, _ = model(x, minus)
            plus_std = torch.exp(0.5 * plus_logvar)
            minus_std = torch.exp(0.5 * minus_logvar)
            plus_delta = plus_mean - base_mean
            minus_delta = minus_mean - base_mean
            antisym = plus_delta + minus_delta
            separation = plus_mean - minus_mean
            axis_results.append(
                {
                    "action_axis": axis,
                    "horizons_seconds": [float(v) for v in metadata["horizon_seconds"]],
                    "mean_plus_minus_prediction_delta_norm": [float(torch.linalg.vector_norm(plus_delta[:, h], dim=-1).mean().cpu()) for h in range(plus_delta.shape[1])],
                    "mean_plus_minus_separation_norm": [float(torch.linalg.vector_norm(separation[:, h], dim=-1).mean().cpu()) for h in range(separation.shape[1])],
                    "mean_antisymmetry_norm": [float(torch.linalg.vector_norm(antisym[:, h], dim=-1).mean().cpu()) for h in range(antisym.shape[1])],
                    "fraction_nontrivial_plus_delta": [float((torch.linalg.vector_norm(plus_delta[:, h], dim=-1) > 1e-4).float().mean().cpu()) for h in range(plus_delta.shape[1])],
                    "mean_std_change": [float((plus_std[:, h] - minus_std[:, h]).abs().mean().cpu()) for h in range(plus_std.shape[1])],
                }
            )
            plus[:, -1, axis] -= float(perturbation)
            minus[:, -1, axis] += float(perturbation)
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "model_type": checkpoint["model_type"],
        "dataset": str(dataset_path.resolve()),
        "samples": count,
        "perturbation_in_normalized_action_units": float(perturbation),
        "device": str(device),
        "all_finite": bool(torch.isfinite(base_mean).all() and torch.isfinite(base_std).all()),
        "axes": axis_results,
        "interpretation": "Non-zero candidate separation indicates action sensitivity; low antisymmetry relative to separation is a warning for action-following mismatch. This diagnostic is not a formal safety guarantee.",
    }


def render(reports: list[dict[str, Any]]) -> str:
    lines = [
        "# JEPA Action-Following Sensitivity Audit",
        "",
        "> This diagnostic checks whether changing the final candidate action changes the predicted future. It is not a safety proof and does not open a locked test.",
        "",
        "| Model | Axis | Horizon separations (normalized position units) | Non-trivial response |",
        "| --- | ---: | --- | --- |",
    ]
    for report in reports:
        label = Path(report["checkpoint"]).parent.name
        for axis in report["axes"]:
            separation = ", ".join(f"{v:.5f}" for v in axis["mean_plus_minus_separation_norm"])
            response = ", ".join(f"{100*v:.1f}%" for v in axis["fraction_nontrivial_plus_delta"])
            lines.append(f"| {label} | {axis['action_axis']} | {separation} | {response} |")
    lines += ["", "Interpretation: candidate separation should be clearly non-zero. The audit is a model-behavior check only; actual rollout correspondence and CBF safety remain separate evaluations.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("results/jepa_pilot_v2_validation/action_conditioned_prediction_dataset.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("results/jepa_pilot_v2_validation/metadata.json"))
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--perturbation", type=float, default=0.25)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    reports = [evaluate(path.resolve(), args.dataset.resolve(), args.metadata.resolve(), args.sample_count, args.perturbation, args.device) for path in args.checkpoint]
    args.output_json.write_text(json.dumps({"audit_type": "jepa_action_following_sensitivity", "reports": reports}, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render(reports), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
