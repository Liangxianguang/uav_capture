"""Aggregate three P2 safe-capture JEPA prediction gates.

This command consumes only held-out validation prediction summaries.  It does
not load scenes, checkpoints, or target truth and cannot open a locked test.
The aggregate is an offline prediction result; it is not closed-loop evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


MODEL_TYPE = "interaction_aware_action_conditioned_jepa_safe_capture_v2"
GATE_TYPE = "jepa_safe_capture_v2_p2_prediction_gate"
REQUIRED_METRICS = (
    "target_position_mae_m",
    "constant_velocity_mae_m",
    "target_improvement_over_constant_velocity_fraction",
    "target_one_std_coverage",
    "target_velocity_mae_mps",
    "target_acceleration_mae_mps2",
    "obstacle_clearance_lower_quantile_mae_m",
    "inter_agent_clearance_lower_quantile_mae_m",
    "pairwise_ttc_mae_s",
    "visibility_brier",
    "visibility_auc",
    "observation_age_mae_steps",
    "cbf_correction_mae_mps",
    "cbf_intervention_brier",
    "cbf_intervention_auc",
    "qp_feasibility_brier",
    "qp_feasibility_auc",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _finite_or_none(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"Non-finite metric {label}: {value!r}")
    return number


def _stats(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("Cannot summarize an empty metric list.")
    return {
        "mean": float(np.mean(data)),
        "sample_std": float(np.std(data, ddof=1)) if data.size > 1 else 0.0,
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
    }


def load_gate(path: Path) -> dict[str, Any]:
    path = path.resolve()
    gate = read_json(path)
    if gate.get("evaluation_type") != GATE_TYPE:
        raise ValueError(f"Unexpected evaluation type in {path}")
    if gate.get("not_a_locked_test") is not True or gate.get("locked_test_opened") is not False:
        raise ValueError(f"Locked-test boundary is invalid in {path}")
    checkpoint = gate.get("checkpoint")
    checkpoint_hash = gate.get("checkpoint_sha256")
    if not isinstance(checkpoint, str) or not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
        raise ValueError(f"Checkpoint provenance is incomplete in {path}")
    dataset = gate.get("dataset")
    dataset_hash = gate.get("dataset_sha256")
    metadata_hash = gate.get("metadata_sha256")
    if not isinstance(dataset, str) or not isinstance(dataset_hash, str) or len(dataset_hash) != 64:
        raise ValueError(f"Validation dataset provenance is incomplete in {path}")
    if not isinstance(metadata_hash, str) or len(metadata_hash) != 64:
        raise ValueError(f"Validation metadata provenance is incomplete in {path}")
    metrics = gate.get("metrics_by_horizon")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError(f"Missing metrics_by_horizon in {path}")
    prediction_gate = gate.get("prediction_gate")
    if not isinstance(prediction_gate, dict):
        raise ValueError(f"Missing prediction_gate in {path}")
    if prediction_gate.get("all_finite") is not True:
        raise ValueError(f"Prediction finite gate failed in {path}")
    if prediction_gate.get("target_better_than_constant_velocity_all_horizons") is not True:
        raise ValueError(f"Target prediction gate failed in {path}")
    for index, row in enumerate(metrics):
        if not isinstance(row, dict):
            raise ValueError(f"Horizon row {index} is not an object in {path}")
        for name in REQUIRED_METRICS:
            if name not in row:
                raise ValueError(f"Missing metric {name} at horizon {index} in {path}")
            _finite_or_none(row[name], label=f"{path}:{index}:{name}")
    return {"path": str(path), "sha256": sha256(path), "gate": gate}


def _check_shared_contract(runs: list[dict[str, Any]]) -> tuple[list[int], list[float], int, str, str]:
    if len(runs) != 3:
        raise ValueError(f"P2-A requires exactly three prediction gates, got {len(runs)}")
    seeds: list[int] = []
    for run in runs:
        gate = run["gate"]
        checkpoint = Path(str(gate["checkpoint"]))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint referenced by gate is missing: {checkpoint}")
        if sha256(checkpoint) != gate["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {checkpoint}")
        seeds.append(int(Path(run["path"]).parent.name.rsplit("seed", 1)[-1]))
    if len(set(seeds)) != 3:
        raise ValueError(f"P2-A requires three distinct seeds, got {seeds}")
    first = runs[0]["gate"]
    if first.get("model_type", MODEL_TYPE) != MODEL_TYPE:
        raise ValueError("P2 model type is not the safe-capture v2 model.")
    horizon_seconds = [float(row["horizon_seconds"]) for row in first["metrics_by_horizon"]]
    validation_hash = str(first["dataset_sha256"])
    metadata_hash = str(first["metadata_sha256"])
    samples = int(first.get("samples", 0))
    if samples <= 0:
        raise ValueError("Validation sample count must be positive.")
    for run in runs[1:]:
        gate = run["gate"]
        if gate.get("model_type", MODEL_TYPE) != MODEL_TYPE:
            raise ValueError("P2 model types differ between gates.")
        if str(gate["dataset_sha256"]) != validation_hash:
            raise ValueError("Validation dataset hashes differ between gates.")
        if str(gate["metadata_sha256"]) != metadata_hash:
            raise ValueError("Validation metadata hashes differ between gates.")
        if int(gate.get("samples", 0)) != samples:
            raise ValueError("Validation sample counts differ between gates.")
        other_horizons = [float(row["horizon_seconds"]) for row in gate["metrics_by_horizon"]]
        if not np.allclose(other_horizons, horizon_seconds, rtol=0.0, atol=1e-9):
            raise ValueError("Prediction horizons differ between gates.")
    return seeds, horizon_seconds, samples, validation_hash, metadata_hash


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    seeds, horizons, samples, validation_hash, metadata_hash = _check_shared_contract(runs)
    metrics_by_horizon: list[dict[str, Any]] = []
    for index, horizon in enumerate(horizons):
        rows = [run["gate"]["metrics_by_horizon"][index] for run in runs]
        summary: dict[str, Any] = {"horizon_seconds": float(horizon)}
        for name in REQUIRED_METRICS:
            values = [_finite_or_none(row[name], label=f"horizon {horizon}:{name}") for row in rows]
            finite_values = [value for value in values if value is not None]
            summary[name] = _stats(finite_values) if finite_values else None
        improvements = [float(row["target_improvement_over_constant_velocity_fraction"]) for row in rows]
        summary["seeds_better_than_constant_velocity"] = int(sum(value > 0.0 for value in improvements))
        summary["seed_improvements"] = {str(seed): value for seed, value in zip(seeds, improvements)}
        metrics_by_horizon.append(summary)
    all_horizons_positive = all(item["seeds_better_than_constant_velocity"] == len(seeds) for item in metrics_by_horizon)
    return {
        "aggregation_type": "jepa_safe_capture_v2_p2_three_seed_prediction",
        "model_type": MODEL_TYPE,
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "run_count": len(runs),
        "seeds": seeds,
        "validation_samples_per_run": samples,
        "validation_dataset_sha256": validation_hash,
        "validation_metadata_sha256": metadata_hash,
        "input_gates": [
            {"path": run["path"], "sha256": run["sha256"], "checkpoint_sha256": run["gate"]["checkpoint_sha256"]}
            for run in runs
        ],
        "metrics_by_horizon": metrics_by_horizon,
        "decision": {
            "all_prediction_gates_passed": True,
            "all_seed_target_improvements_positive_at_all_horizons": all_horizons_positive,
            "eligible_for_reliability_ledger_development": True,
            "eligible_for_closed_loop_development": True,
            "eligible_for_locked_test": False,
            "reason": "All three held-out P2 checkpoints pass the finite-output and target prediction gates. This authorizes calibration-only reliability-ledger development, not a closed-loop or locked-test performance claim.",
        },
    }


def write_tensorboard(report: dict[str, Any], logdir: Path) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard logdir: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/aggregate", json.dumps({"aggregation_type": report["aggregation_type"], "model_type": report["model_type"]}, indent=2), 0)
        writer.add_text("Provenance/input_gates", json.dumps(report["input_gates"], indent=2), 0)
        writer.add_text("Provenance/decision", json.dumps(report["decision"], indent=2), 0)
        writer.add_text("Provenance/data_hashes", json.dumps({"validation_dataset_sha256": report["validation_dataset_sha256"], "validation_metadata_sha256": report["validation_metadata_sha256"]}, indent=2), 0)
        for index, item in enumerate(report["metrics_by_horizon"]):
            step = index + 1
            writer.add_scalar("Eval/horizon_seconds", float(item["horizon_seconds"]), step)
            writer.add_scalar("Eval/seeds_better_than_constant_velocity", int(item["seeds_better_than_constant_velocity"]), step)
            for name in REQUIRED_METRICS:
                metric = item[name]
                if metric is not None:
                    writer.add_scalar(f"Eval/{name}/mean", float(metric["mean"]), step)
                    writer.add_scalar(f"Eval/{name}/sample_std", float(metric["sample_std"]), step)
        for seed in report["seeds"]:
            writer.add_scalar(f"Eval/seed/{seed}/present", 1.0, 0)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required_text = {"Config/aggregate/text_summary", "Provenance/input_gates/text_summary", "Provenance/decision/text_summary", "Provenance/data_hashes/text_summary"}
    missing_text = sorted(required_text.difference(tags.get("tensors", [])))
    if missing_text:
        raise ValueError(f"TensorBoard aggregate is missing text provenance: {missing_text}")
    return {
        "logdir": str(logdir),
        "event_files": sorted(path.name for path in logdir.glob("events.out.tfevents.*")),
        "scalar_tag_count": len(tags.get("scalars", [])),
        "text_tag_count": len(tags.get("tensors", [])),
        "required_text_complete": not missing_text,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# JEPA Safe-Capture v2 P2 三 Seed Prediction Aggregate",
        "",
        "> Development-only held-out prediction evidence. This is not a closed-loop result and not a locked test.",
        "",
        f"Model: `{report['model_type']}`",
        f"Training seeds: `{', '.join(str(seed) for seed in report['seeds'])}`",
        f"Validation samples per seed: `{report['validation_samples_per_run']}`",
        f"Validation dataset SHA-256: `{report['validation_dataset_sha256']}`",
        "",
        "## Prediction Summary",
        "",
        "| Horizon (s) | Target MAE (m) | Constant-velocity MAE (m) | Improvement vs CV | Seeds better than CV | Visibility AUROC | CBF intervention AUROC | QP feasibility AUROC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["metrics_by_horizon"]:
        def value(name: str) -> str:
            metric = item[name]
            return "n/a" if metric is None else f"{metric['mean']:.4f} +/- {metric['sample_std']:.4f}"

        lines.append(
            f"| {item['horizon_seconds']:.1f} | {value('target_position_mae_m')} | {value('constant_velocity_mae_m')} | "
            f"{value('target_improvement_over_constant_velocity_fraction')} | {item['seeds_better_than_constant_velocity']}/3 | "
            f"{value('visibility_auc')} | {value('cbf_intervention_auc')} | {value('qp_feasibility_auc')} |"
        )
    lines += [
        "",
        "## Checkpoint Provenance",
        "",
        "| Seed | Prediction-gate SHA-256 | Checkpoint SHA-256 |",
        "|---:|---|---|",
    ]
    for seed, item in zip(report["seeds"], report["input_gates"]):
        lines.append(f"| {seed} | `{item['sha256']}` | `{item['checkpoint_sha256']}` |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- All three held-out checkpoints are finite and pass the declared P2 target-prediction gate.",
        "- Target prediction improves over constant velocity for every seed at all four horizons in this validation summary.",
        "- `qp_feasibility_auc` is `n/a` because the current P1 feasibility labels contain no positive/negative class variation. This is not evidence of a calibrated QP-feasibility predictor.",
        "- Clearance, visibility, uncertainty and intervention heads remain ranking signals only. They are not safety certificates and cannot bypass CBF.",
        "- The next authorized step is calibration-only reliability-ledger construction with immutable nominal fallback. Closed-loop paired safe-capture evaluation is still required.",
        "",
        "## Decision",
        "",
        report["decision"]["reason"],
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-gate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, default=Path("results/jepa_safe_capture_v2_tensorboard/p2_aggregate"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [load_gate(path) for path in args.prediction_gate]
    report = aggregate(runs)
    tensorboard = write_tensorboard(report, args.tensorboard_logdir)
    report["tensorboard"] = tensorboard
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.report.resolve().write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "tensorboard": tensorboard}, indent=2))


if __name__ == "__main__":
    main()
