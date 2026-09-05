"""Audit the V21 candidate-separation contract on frozen development traces.

This is an offline audit.  It never executes a policy, changes a checkpoint,
or opens the locked split.  The source traces are the V20 CPU/CUDA replays;
the V21 separation gate is evaluated as a deterministic counterfactual on the
recorded candidate scores and eligibility masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from encirclement3d.jepa_safe_capture_ranker import (
    _candidate_specific_separation,
    _fixed_point_score_keys,
)


SEEDS = (20260911, 20260912, 20260913)
DEFAULT_SETTLED = {
    seed: Path(f"results/jepa_safe_capture_v20_cpu_deterministic_settled_m3_vs_m0_seed{seed}/decision_rows.jsonl")
    for seed in SEEDS
}
DEFAULT_CPU = {
    seed: Path(f"results/jepa_safe_capture_v20_cpu_deterministic_replay_m3_cpu_seed{seed}")
    for seed in SEEDS
}
DEFAULT_CUDA = {
    seed: Path(f"results/jepa_safe_capture_v20_cpu_deterministic_replay_m3_cuda_seed{seed}")
    for seed in SEEDS
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object in {path}")
                rows.append(value)
    if not rows:
        raise ValueError(f"Empty JSONL input: {path}")
    return rows


def _trace_rows(root: Path) -> dict[tuple[int, int], dict[str, Any]]:
    trace_root = root / "step_traces"
    paths = sorted(trace_root.glob("episode_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No step traces under {trace_root}")
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for path in paths:
        for row in _read_jsonl(path):
            key = (int(row["episode_index"]), int(row["step"]))
            if key in rows:
                raise ValueError(f"Duplicate trace key {key} in {root}")
            rows[key] = row
    return rows


def _settled_rows(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    rows = _read_jsonl(path)
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["episode_index"]), int(row["step"]))
        if key in result:
            raise ValueError(f"Duplicate settled key {key} in {path}")
        result[key] = row
    return result


def _score_values(ranking: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(ranking.get("scores", []), dtype=np.float64)
    if values.shape != (5,):
        raise ValueError("Each ranking trace must contain five scores")
    return values


def _candidate_separation(ranking: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(ranking.get("candidate_separation_m", []), dtype=np.float64)
    if values.shape == (5,) and np.isfinite(values).any():
        return values
    # V20 traces predate the explicit trace field.  The ranker defines
    # candidate separation from target-distance cost, so reconstruct it from
    # the immutable diagnostic field for this offline counterfactual.
    target_cost = np.asarray(ranking.get("target_cost_m", []), dtype=np.float64)
    if target_cost.shape != (5,):
        raise ValueError("Ranking trace has neither candidate separation nor target cost")
    return _candidate_specific_separation(target_cost)


def _v21_eligible(ranking: Mapping[str, Any], threshold_m: float) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(ranking.get("eligible_mask", []), dtype=bool)
    valid = np.asarray(ranking.get("valid_mask", []), dtype=bool)
    if base.shape != (5,) or valid.shape != (5,):
        raise ValueError("Ranking masks must contain five candidates")
    separation = _candidate_separation(ranking)
    gate = np.isfinite(separation) & (separation >= float(threshold_m))
    eligible = base & valid
    # Candidate 0 is the nominal anchor.  The separation gate applies only to
    # alternatives; clearance/ledger eligibility remains unchanged.
    eligible[1:] &= gate[1:]
    return eligible, separation


def _settled_best(row: Mapping[str, Any], eligible: Iterable[int]) -> int:
    indices = list(eligible)
    if not indices:
        raise ValueError("Settled best requires at least one eligible candidate")
    return min(
        indices,
        key=lambda index: (
            -int(bool(row["settled_candidates"][index]["settled_safe_capture"])),
            -int(bool(row["settled_candidates"][index]["settled_safety_ok"])),
            -float(row["settled_candidates"][index]["settled_progress_m"]),
            float(row["settled_candidates"][index]["settled_cbf_correction_norm_mps"]),
            index,
        ),
    )


def _fixed_point_argmin(ranking: Mapping[str, Any], eligible: Iterable[int], quantum_m: float) -> int | None:
    indices = list(eligible)
    if not indices:
        return None
    scores = _score_values(ranking)
    keys = _fixed_point_score_keys(scores, quantum_m)
    if any(keys[index] is None for index in indices):
        return None
    return min(indices, key=lambda index: (int(keys[index]), index))


def _fixed_point_order(ranking: Mapping[str, Any], eligible: Iterable[int], quantum_m: float) -> list[int]:
    indices = list(eligible)
    if not indices:
        return []
    scores = _score_values(ranking)
    keys = _fixed_point_score_keys(scores, quantum_m)
    if any(keys[index] is None for index in indices):
        return []
    return sorted(indices, key=lambda index: (int(keys[index]), index))


def _sequence_stats(indices: list[int]) -> dict[str, Any]:
    if not indices:
        return {"count": 0, "switch_count": 0, "switch_rate": 0.0, "mean_run_length": None, "max_alternating_run": 0}
    switches = sum(first != second for first, second in zip(indices, indices[1:]))
    runs: list[int] = []
    current = 1
    for first, second in zip(indices, indices[1:]):
        if first == second:
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)
    alternating = 1
    max_alternating = 1
    for first, second in zip(indices, indices[1:]):
        if first != second:
            alternating += 1
            max_alternating = max(max_alternating, alternating)
        else:
            alternating = 1
    return {
        "count": len(indices),
        "switch_count": int(switches),
        "switch_rate": float(switches / max(len(indices) - 1, 1)),
        "mean_run_length": float(np.mean(runs)),
        "max_alternating_run": int(max_alternating),
    }


def _decision_projection(trace: Mapping[str, Any]) -> dict[str, Any]:
    ranking = trace.get("candidate_ranking", {})
    cbf = trace.get("cbf", {})
    return {
        "episode_index": int(trace["episode_index"]),
        "step": int(trace["step"]),
        "selected_index": int(ranking.get("selected_index", -1)),
        "eligible_mask": list(ranking.get("eligible_mask", [])),
        "candidate_order": list(ranking.get("candidate_order", [])),
        "score_comparison_keys": list(ranking.get("score_comparison_keys", [])),
        "computed_fixed_point_order": _fixed_point_order(
            ranking,
            np.flatnonzero(np.asarray(ranking.get("eligible_mask", []), dtype=bool)),
            0.004,
        ),
        "execution_mode": ranking.get("execution_mode"),
        "fallback_reason": ranking.get("fallback_reason"),
        "rank_abstention_reason": ranking.get("rank_abstention_reason"),
        "cbf_verified_feasible": bool(cbf.get("verified_feasible", False)),
        "cbf_solver_status": cbf.get("solver_status"),
        "cbf_fallback_mode": cbf.get("fallback_mode"),
        "executed_action": _jsonable(np.round(np.asarray(trace.get("executed_action", []), dtype=np.float64), 12)),
        "raw_unverified_executed": bool(trace.get("raw_unverified_executed", False)),
    }


def _device_compare(cpu: Mapping[tuple[int, int], dict[str, Any]], cuda: Mapping[tuple[int, int], dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(set(cpu) | set(cuda))
    differences: list[dict[str, Any]] = []
    for key in keys:
        if key not in cpu or key not in cuda:
            differences.append({"key": list(key), "reason": "missing_trace"})
            continue
        left = _decision_projection(cpu[key])
        right = _decision_projection(cuda[key])
        if left != right:
            differences.append({"key": list(key), "cpu": left, "cuda": right})
    return {
        "cpu_trace_count": len(cpu),
        "cuda_trace_count": len(cuda),
        "decision_difference_count": len(differences),
        "decision_equal": not differences and len(cpu) == len(cuda),
        "differences_sample": differences[:20],
    }


def _seed_audit(
    seed: int,
    settled: Mapping[tuple[int, int], dict[str, Any]],
    cpu: Mapping[tuple[int, int], dict[str, Any]],
    cuda: Mapping[tuple[int, int], dict[str, Any]],
    *,
    separation_threshold_m: float,
    score_quantum_m: float,
) -> dict[str, Any]:
    if set(settled) - set(cpu):
        raise ValueError(f"Settled rows are not covered by CPU traces for seed {seed}")
    separation_values: list[float] = []
    gate_rejected = 0
    base_alternatives = 0
    base_multi = 0
    v21_multi = 0
    abstention = Counter()
    fallback = Counter()
    selected_indices: list[int] = []
    v21_selected_indices: list[int] = []
    selected_not_best = 0
    score_argmin_not_best = 0
    top1_safety: list[float] = []
    top1_capture: list[float] = []
    raw_unverified = 0
    cbf_unverified = 0
    rows_with_settled = 0
    settled_with_v21_eligible = 0
    for key in sorted(cpu):
        trace = cpu[key]
        ranking = trace.get("candidate_ranking", {})
        eligible, separation = _v21_eligible(ranking, separation_threshold_m)
        separation_values.extend(float(value) for value in separation if np.isfinite(value))
        base = np.asarray(ranking.get("eligible_mask", []), dtype=bool)
        base_count = int(np.count_nonzero(base))
        if base_count >= 2:
            base_multi += 1
        gate_rejected += int(np.count_nonzero(base[1:] & ~eligible[1:]))
        base_alternatives += int(np.count_nonzero(base[1:]))
        if int(np.count_nonzero(eligible)) >= 2:
            v21_multi += 1
        reason = ranking.get("rank_abstention_reason")
        if reason:
            abstention[str(reason)] += 1
        reason = ranking.get("fallback_reason")
        if reason:
            fallback[str(reason)] += 1
        selected_indices.append(int(ranking.get("selected_index", -1)))
        score_selected = _fixed_point_argmin(ranking, np.flatnonzero(eligible), score_quantum_m)
        v21_selected_indices.append(0 if score_selected is None else int(score_selected))
        raw_unverified += int(bool(trace.get("raw_unverified_executed", False)))
        cbf_unverified += int(not bool(trace.get("cbf", {}).get("verified_feasible", False)))
        row = settled.get(key)
        if row is None:
            continue
        rows_with_settled += 1
        if not np.any(eligible):
            continue
        settled_with_v21_eligible += 1
        best = _settled_best(row, np.flatnonzero(eligible))
        selected = int(ranking.get("selected_index", 0))
        if selected != best:
            selected_not_best += 1
        if score_selected is not None:
            if score_selected != best:
                score_argmin_not_best += 1
            outcome = row["settled_candidates"][score_selected]
            top1_safety.append(float(bool(outcome["settled_safety_ok"])))
            top1_capture.append(float(bool(outcome["settled_safe_capture"])))
    device = _device_compare(cpu, cuda)
    return {
        "seed": seed,
        "source": "v20_frozen_cpu_cuda_replay_with_v21_separation_counterfactual",
        "trace_count": len(cpu),
        "settled_row_count": len(settled),
        "settled_rows_covered": rows_with_settled,
        "base_multi_eligible_count": base_multi,
        "v21_counterfactual_multi_eligible_count": v21_multi,
        "candidate_separation_threshold_m": float(separation_threshold_m),
        "candidate_separation_summary": {
            "count": len(separation_values),
            "minimum": float(np.min(separation_values)) if separation_values else None,
            "median": float(np.median(separation_values)) if separation_values else None,
            "p10": float(np.percentile(separation_values, 10)) if separation_values else None,
            "p90": float(np.percentile(separation_values, 90)) if separation_values else None,
            "maximum": float(np.max(separation_values)) if separation_values else None,
        },
        "separation_gate_rejected_alternatives": gate_rejected,
        "separation_gate_rejection_rate_among_base_alternatives": float(
            gate_rejected / max(base_alternatives, 1)
        ),
        "source_abstention_rate": float(sum(abstention.values()) / max(len(cpu), 1)),
        "source_abstention_reasons": dict(abstention),
        "source_fallback_reasons": dict(fallback),
        "source_selected_sequence": _sequence_stats(selected_indices),
        "v21_counterfactual_score_sequence": _sequence_stats(v21_selected_indices),
        "selected_not_best_rate_on_v21_eligible_rows": float(
            selected_not_best / max(settled_with_v21_eligible, 1)
        ),
        "score_argmin_not_best_rate_on_v21_eligible_rows": float(
            score_argmin_not_best / max(settled_with_v21_eligible, 1)
        ),
        "settled_rows_with_v21_eligible_candidates": settled_with_v21_eligible,
        "top1_safety_precision": float(np.mean(top1_safety)) if top1_safety else None,
        "top1_safe_capture_precision": float(np.mean(top1_capture)) if top1_capture else None,
        "raw_unverified_executed_count": raw_unverified,
        "cbf_unverified_trace_count": cbf_unverified,
        "device_replay": device,
    }


def _input_manifest(paths: Iterable[Path]) -> dict[str, Any]:
    files: dict[str, str] = {}
    for path in paths:
        if path.is_file():
            files[str(path.resolve())] = _sha256(path)
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    files[str(child.resolve())] = _sha256(child)
        else:
            raise FileNotFoundError(path)
    return {"file_count": len(files), "sha256": files}


def _validate_tensorboard(logdir: Path) -> dict[str, Any]:
    events = sorted(path.name for path in logdir.glob("events.out.tfevents.*"))
    accumulator = EventAccumulator(str(logdir))
    accumulator.Reload()
    tags = accumulator.Tags()
    required = {
        "Config/protocol/text_summary",
        "Provenance/source/text_summary",
        "Gates/raw_unverified_executed/text_summary",
        "Ranking/candidate_separation",
    }
    missing = sorted(required.difference(set(tags.get("tensors", [])) | set(tags.get("scalars", []))))
    if not events or missing:
        raise ValueError(f"TensorBoard validation failed: events={events}, missing={missing}")
    return {"event_files": events, "scalar_tags": sorted(tags.get("scalars", [])), "tensor_tags": sorted(tags.get("tensors", []))}


def write_report(
    project_root: Path,
    output_dir: Path,
    tensorboard_dir: Path,
    *,
    protocol: Path,
    separation_threshold_m: float = 0.002,
    score_quantum_m: float = 0.004,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    if tensorboard_dir.exists() and any(tensorboard_dir.iterdir()):
        raise FileExistsError(tensorboard_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    protocol = protocol.resolve()
    reports: list[dict[str, Any]] = []
    input_paths: list[Path] = [protocol]
    for seed in SEEDS:
        settled_path = project_root / DEFAULT_SETTLED[seed]
        cpu_root = project_root / DEFAULT_CPU[seed]
        cuda_root = project_root / DEFAULT_CUDA[seed]
        settled = _settled_rows(settled_path)
        cpu = _trace_rows(cpu_root)
        cuda = _trace_rows(cuda_root)
        reports.append(
            _seed_audit(
                seed,
                settled,
                cpu,
                cuda,
                separation_threshold_m=separation_threshold_m,
                score_quantum_m=score_quantum_m,
            )
        )
        input_paths.extend([settled_path, cpu_root, cuda_root])
    aggregate = {
        "trace_count": int(sum(item["trace_count"] for item in reports)),
        "separation_gate_rejected_alternatives": int(sum(item["separation_gate_rejected_alternatives"] for item in reports)),
        "raw_unverified_executed_count": int(sum(item["raw_unverified_executed_count"] for item in reports)),
        "device_decision_equal": all(item["device_replay"]["decision_equal"] for item in reports),
        "device_difference_count": int(sum(item["device_replay"]["decision_difference_count"] for item in reports)),
        "top1_safety_precision_mean": float(np.mean([item["top1_safety_precision"] for item in reports if item["top1_safety_precision"] is not None])) if any(item["top1_safety_precision"] is not None for item in reports) else None,
    }
    result: dict[str, Any] = {
        "stage": "WP2_v21_candidate_separation_audit",
        "development_only": True,
        "locked_test_opened": False,
        "online_policy_changed": False,
        "settled_outcomes_used_only_for_offline_audit": True,
        "protocol": {
            "path": str(protocol),
            "sha256": _sha256(protocol),
            "separation_threshold_m": separation_threshold_m,
            "score_comparison_quantum_m": score_quantum_m,
        },
        "seeds": reports,
        "aggregate": aggregate,
        "provenance": {
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }
    result["input_hash_manifest"] = _input_manifest(input_paths)
    (output_dir / "input_hash_manifest.json").write_text(
        json.dumps(_jsonable(result["input_hash_manifest"]), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            _jsonable(
                {
                    "stage": result["stage"],
                    "development_only": True,
                    "locked_test_opened": False,
                    "online_policy_changed": False,
                    "git_revision": result["provenance"]["git_revision"],
                    "protocol_sha256": result["protocol"]["sha256"],
                    "separation_threshold_m": separation_threshold_m,
                    "score_comparison_quantum_m": score_quantum_m,
                    "seeds": list(SEEDS),
                    "source_kind": "frozen_v20_cpu_cuda_replay_and_settled_rows",
                }
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(_jsonable({
        "stage": result["stage"],
        "development_only": True,
        "locked_test_opened": False,
        "aggregate": result["aggregate"],
        "tensorboard_dir": str(tensorboard_dir),
    }), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "candidate_separation_audit.json").write_text(json.dumps(_jsonable(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# V21 candidate-separation audit",
        "",
        "`development_only=true`; `locked_test_opened=false`. This is an offline counterfactual audit over frozen V20 traces; it does not report a new online policy result.",
        "",
        "| Seed | Traces | Base multi-eligible | V21 counterfactual multi-eligible | Separation rejects | Device decision equal | Raw unverified |",
        "|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for item in reports:
        lines.append(
            f"| {item['seed']} | {item['trace_count']} | {item['base_multi_eligible_count']} | "
            f"{item['v21_counterfactual_multi_eligible_count']} | {item['separation_gate_rejected_alternatives']} | "
            f"{item['device_replay']['decision_equal']} | {item['raw_unverified_executed_count']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The separation gate is applied only to non-nominal candidates; nominal remains the anchor. Predicted safety remains an eligibility/ranking signal and does not replace Joint CBF verification.",
        "",
        f"Aggregate device decision equality: `{aggregate['device_decision_equal']}`.",
        f"Aggregate raw-unverified executions: `{aggregate['raw_unverified_executed_count']}`.",
        "This report cannot be used as a safe-capture improvement claim; a new paired smoke is required after WP3 and WP4 safety gates.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with SummaryWriter(log_dir=str(tensorboard_dir), flush_secs=1) as writer:
        writer.add_text("Config/protocol", json.dumps(result["protocol"], indent=2), 0)
        writer.add_text("Provenance/source", json.dumps(result["provenance"], indent=2), 0)
        writer.add_text("Gates/raw_unverified_executed", json.dumps({"count": aggregate["raw_unverified_executed_count"]}), 0)
        writer.add_scalar("Gates/device_decision_equal", float(aggregate["device_decision_equal"]), 0)
        writer.add_scalar("Gates/raw_unverified_executed", float(aggregate["raw_unverified_executed_count"]), 0)
        for index, item in enumerate(reports):
            prefix = f"Seed_{item['seed']}"
            writer.add_scalar(f"{prefix}/Ranking/candidate_separation", item["candidate_separation_summary"]["median"] or 0.0, index)
            writer.add_scalar(f"{prefix}/Ranking/insufficient_candidate_separation", item["separation_gate_rejected_alternatives"], index)
            writer.add_scalar(f"{prefix}/Ranking/selected_not_best", item["selected_not_best_rate_on_v21_eligible_rows"], index)
            writer.add_scalar(f"{prefix}/Ranking/abstention_rate", item["source_abstention_rate"], index)
            writer.add_scalar(f"{prefix}/Ranking/switch_rate", item["source_selected_sequence"]["switch_rate"], index)
            writer.add_scalar(f"{prefix}/Ranking/oscillation_length", item["source_selected_sequence"]["max_alternating_run"], index)
            writer.add_scalar(f"{prefix}/Safety/raw_unverified_executed", item["raw_unverified_executed_count"], index)
        writer.add_scalar("Ranking/candidate_separation", float(np.mean([item["candidate_separation_summary"]["median"] or 0.0 for item in reports])), 0)
        writer.add_scalar("Ranking/selected_not_best", float(np.mean([item["selected_not_best_rate_on_v21_eligible_rows"] for item in reports])), 0)
        writer.add_scalar("Ranking/abstention_rate", float(np.mean([item["source_abstention_rate"] for item in reports])), 0)
        writer.add_scalar("Ranking/switch_rate", float(np.mean([item["source_selected_sequence"]["switch_rate"] for item in reports])), 0)
        writer.add_scalar("Ranking/oscillation_length", float(max(item["source_selected_sequence"]["max_alternating_run"] for item in reports)), 0)
    result["tensorboard"] = _validate_tensorboard(tensorboard_dir)
    (output_dir / "candidate_separation_audit.json").write_text(json.dumps(_jsonable(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--minimum-candidate-separation-m", type=float, default=0.002)
    parser.add_argument("--score-comparison-quantum-m", type=float, default=0.004)
    args = parser.parse_args()
    print(json.dumps(_jsonable(write_report(
        args.project_root.resolve(),
        args.output_dir.resolve(),
        args.tensorboard_dir.resolve(),
        protocol=args.protocol,
        separation_threshold_m=args.minimum_candidate_separation_m,
        score_quantum_m=args.score_comparison_quantum_m,
    )), indent=2))


if __name__ == "__main__":
    main()
