"""Aggregate paired safe-capture-first JEPA-v3 P6 development evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_CANDIDATE_COUNT = 5
EXPECTED_PERTURBATION_MPS = 0.10
EXPECTED_CHUNK_LENGTH = 3
EXPECTED_EPISODES = 60
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260903


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any) -> float | None:
    if value in (None, "", "null", "None"):
        return None
    return float(value)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No episode rows: {path}")
    indices = [int(row["episode_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate episode index: {path}")
    return rows


def frozen_scene_hash(path: Path) -> str:
    """Hash only immutable scene specifications, excluding method outcomes."""

    scenes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        scenes.append({key: row[key] for key in ("episode_index", "spec", "scenario")})
    canonical = json.dumps(scenes, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _optional_column_mean(rows: list[dict[str, str]], field: str) -> float | None:
    values = [_float(row.get(field)) for row in rows]
    return _mean([value for value in values if value is not None])


def _run_metrics(rows: list[dict[str, str]]) -> dict[str, float | int | None]:
    captures = [float(_bool(row["safe_capture_success"])) for row in rows]
    collisions = [float(_bool(row["collision"])) for row in rows]
    boundaries = [float(int(float(row["world_violation_steps"])) > 0) for row in rows]
    transit = [float(_bool(row["transit_success"])) for row in rows]
    capture_times = [_float(row["capture_time_seconds"]) for row in rows]
    return {
        "episodes": len(rows),
        "safe_capture_count": int(sum(captures)),
        "safe_capture_rate": float(np.mean(captures)),
        "collision_count": int(sum(collisions)),
        "collision_rate": float(np.mean(collisions)),
        "boundary_count": int(sum(boundaries)),
        "boundary_rate": float(np.mean(boundaries)),
        "timeout_count": int(sum(row["termination_reason"] == "timeout" for row in rows)),
        "transit_success_rate": float(np.mean(transit)),
        "mean_capture_time_seconds": _mean([value for value in capture_times if value is not None]),
        "mean_total_defender_path_length_m": _mean([float(row["total_defender_path_length_m"]) for row in rows]),
        "mean_min_clearance_m": _mean([float(row["min_clearance_m"]) for row in rows]),
        "worst_min_clearance_m": float(min(float(row["min_clearance_m"]) for row in rows)),
        "mean_cbf_action_correction_norm": _mean([float(row["mean_cbf_action_correction_norm"]) for row in rows]),
        "mean_ledger_credit": _optional_column_mean(rows, "jepa_ledger_mean_credit"),
        "mean_nominal_fallback_fraction": _optional_column_mean(rows, "jepa_ledger_nominal_fallback_fraction"),
        "mean_global_fallback_fraction": _optional_column_mean(rows, "jepa_ledger_global_fallback_fraction"),
        "mean_selected_candidate_index": _optional_column_mean(rows, "jepa_mean_selected_index"),
    }


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def exact_mcnemar_pvalue(improved: int, degraded: int) -> float:
    """Return the two-sided exact McNemar/binomial p-value."""

    discordant = improved + degraded
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(improved, degraded) + 1))
    return float(min(1.0, 2.0 * tail / (2**discordant)))


def hierarchical_bootstrap_interval(
    seed_episode_values: np.ndarray,
    replicates: int = BOOTSTRAP_REPLICATES,
    random_seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Resample training seeds, then paired episodes within each sampled seed."""

    if seed_episode_values.ndim != 2:
        raise ValueError("Expected [training_seed, episode] values.")
    seed_count, episode_count = seed_episode_values.shape
    if seed_count < 2 or episode_count < 1:
        raise ValueError("At least two training seeds and one episode are required.")
    rng = np.random.default_rng(random_seed)
    seed_indices = rng.integers(0, seed_count, size=(replicates, seed_count))
    episode_indices = rng.integers(0, episode_count, size=(replicates, seed_count, episode_count))
    selected = seed_episode_values[seed_indices[:, :, None], episode_indices]
    samples = np.mean(selected, axis=(1, 2))
    return {
        "mean": float(np.mean(seed_episode_values)),
        "ci_95_low": float(np.quantile(samples, 0.025)),
        "ci_95_high": float(np.quantile(samples, 0.975)),
        "replicates": replicates,
        "random_seed": random_seed,
    }


def _validate_candidate_metadata(
    run_dir: Path,
    metadata: dict[str, Any],
    rows: list[dict[str, str]],
    baseline_scene_hash: str,
) -> dict[str, str]:
    if metadata.get("locked_test") is not False or metadata.get("not_a_locked_test") is not True:
        raise ValueError(f"Candidate is not marked development-only: {run_dir}")
    if metadata.get("split") != "validation" or int(metadata.get("episodes", -1)) != EXPECTED_EPISODES:
        raise ValueError(f"Candidate does not use the frozen 60-episode validation block: {run_dir}")
    if len(rows) != EXPECTED_EPISODES or metadata.get("use_cbf") is not True:
        raise ValueError(f"Candidate does not have 60 CBF-filtered episode rows: {run_dir}")
    if metadata.get("action_conditioned_jepa_enabled") is not True:
        raise ValueError(f"JEPA is not enabled: {run_dir}")
    if int(metadata.get("jepa_candidate_count", -1)) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(f"Candidate count changed: {run_dir}")
    if not math.isclose(float(metadata.get("jepa_perturbation_mps", math.nan)), EXPECTED_PERTURBATION_MPS):
        raise ValueError(f"Perturbation changed: {run_dir}")
    if int(metadata.get("jepa_action_chunk_length_steps", -1)) != EXPECTED_CHUNK_LENGTH:
        raise ValueError(f"Action chunk length changed: {run_dir}")
    if metadata.get("jepa_action_chunk_semantics") != "constant_desired_action_chunk_execute_first_step_then_replan":
        raise ValueError(f"Unexpected action chunk semantics: {run_dir}")
    scene_path = run_dir / "scenes.jsonl"
    if not scene_path.is_file() or frozen_scene_hash(scene_path) != baseline_scene_hash:
        raise ValueError(f"Frozen scenes differ from baseline: {run_dir}")
    checkpoint_path = Path(str(metadata["action_conditioned_jepa_checkpoint"])).resolve()
    ledger_path = Path(str(metadata["jepa_reliability_ledger"])).resolve()
    if not checkpoint_path.is_file() or not ledger_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint or ledger for {run_dir}")
    ledger = _read_json(ledger_path)
    checkpoint_hash = _sha256(checkpoint_path)
    if ledger.get("ledger_type") != "jepa_v3_execution_settled_reliability":
        raise ValueError(f"Unexpected reliability ledger: {ledger_path}")
    if ledger.get("source", {}).get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError(f"Ledger is not bound to candidate checkpoint: {run_dir}")
    policy = ledger.get("decision_policy", {})
    if policy.get("minimum_sample_count") != 128 or not math.isclose(float(policy.get("minimum_credit", math.nan)), 0.65):
        raise ValueError(f"Reliability policy changed: {ledger_path}")
    return {
        "candidate_checkpoint": str(checkpoint_path),
        "candidate_checkpoint_sha256": checkpoint_hash,
        "ledger": str(ledger_path),
        "ledger_sha256": _sha256(ledger_path),
        "frozen_scene_sha256": frozen_scene_hash(scene_path),
        "output_scenes_sha256": _sha256(scene_path),
    }


def _paired_run(
    candidate_rows: list[dict[str, str]], baseline_rows: list[dict[str, str]]
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    baseline = {int(row["episode_index"]): row for row in baseline_rows}
    candidate = {int(row["episode_index"]): row for row in candidate_rows}
    if set(candidate) != set(baseline):
        raise ValueError("Candidate and baseline episode indices are not exactly paired.")
    ordered = sorted(candidate)
    if any(candidate[index]["episode_seed"] != baseline[index]["episode_seed"] for index in ordered):
        raise ValueError("Candidate and baseline episode seeds are not exactly paired.")

    capture_delta = np.asarray(
        [int(_bool(candidate[index]["safe_capture_success"])) - int(_bool(baseline[index]["safe_capture_success"])) for index in ordered],
        dtype=np.float64,
    )
    path_delta = np.asarray(
        [float(candidate[index]["total_defender_path_length_m"]) - float(baseline[index]["total_defender_path_length_m"]) for index in ordered],
        dtype=np.float64,
    )
    clearance_delta = np.asarray(
        [float(candidate[index]["min_clearance_m"]) - float(baseline[index]["min_clearance_m"]) for index in ordered],
        dtype=np.float64,
    )
    cbf_delta = np.asarray(
        [float(candidate[index]["mean_cbf_action_correction_norm"]) - float(baseline[index]["mean_cbf_action_correction_norm"]) for index in ordered],
        dtype=np.float64,
    )
    joint_capture_time_delta = np.asarray(
        [
            float(candidate[index]["capture_time_seconds"]) - float(baseline[index]["capture_time_seconds"])
            for index in ordered
            if candidate[index]["capture_time_seconds"] and baseline[index]["capture_time_seconds"]
        ],
        dtype=np.float64,
    )
    fields = (
        "episode_index",
        "episode_seed",
        "termination_reason",
        "min_clearance_m",
        "capture_time_seconds",
        "total_defender_path_length_m",
        "mean_cbf_action_correction_norm",
        "jepa_ledger_mean_credit",
        "jepa_ledger_nominal_fallback_fraction",
        "jepa_ledger_global_fallback_fraction",
        "jepa_mean_selected_index",
        "layout_signature",
        "target_motion_mode",
        "observation_condition",
        "obstacle_count",
    )
    changes: list[dict[str, Any]] = []
    for index, delta in zip(ordered, capture_delta):
        if delta == 0:
            continue
        changes.append(
            {
                "direction": "improved" if delta > 0 else "degraded",
                "baseline": {field: baseline[index].get(field) for field in fields},
                "candidate": {field: candidate[index].get(field) for field in fields},
            }
        )
    improved = int(np.sum(capture_delta > 0))
    degraded = int(np.sum(capture_delta < 0))
    report = {
        "episodes": len(ordered),
        "capture_improved_count": improved,
        "capture_degraded_count": degraded,
        "capture_tied_count": int(np.sum(capture_delta == 0)),
        "safe_capture_delta_percentage_points": float(100.0 * np.mean(capture_delta)),
        "paired_capture_time_delta_seconds_on_joint_success": float(np.mean(joint_capture_time_delta)) if joint_capture_time_delta.size else None,
        "paired_path_delta_m": float(np.mean(path_delta)),
        "paired_min_clearance_delta_m": float(np.mean(clearance_delta)),
        "paired_cbf_correction_delta": float(np.mean(cbf_delta)),
        "exact_mcnemar_two_sided_pvalue": exact_mcnemar_pvalue(improved, degraded),
    }
    arrays = {
        "safe_capture_delta": capture_delta,
        "path_delta": path_delta,
        "min_clearance_delta": clearance_delta,
        "cbf_correction_delta": cbf_delta,
    }
    return report, arrays, changes


def classify(report: dict[str, Any]) -> dict[str, Any]:
    aggregate = report["aggregate"]
    no_new_safety_event = aggregate["candidate_collision_total"] == 0 and aggregate["candidate_boundary_total"] == 0
    positive_mean_delta = aggregate["safe_capture_delta_percentage_points"]["mean"] > 0.0
    nonnegative_seeds = aggregate["safe_capture_delta_nonnegative_seed_count"]
    promising = no_new_safety_event and positive_mean_delta and nonnegative_seeds >= 2
    return {
        "primary_metric": "safe_capture_with_zero_new_collision_or_boundary",
        "classification": "promising_development_candidate" if promising else "prediction_improvement_no_control_gain",
        "locked_test_opened": False,
        "eligible_to_open_locked_test": False,
        "reason": (
            "All P6 candidate runs remain collision-free and in-bounds, but the three-seed mean paired safe-capture delta is not positive. "
            "Do not open a locked block; record offline prediction signal without a reliable closed-loop capture gain."
            if not promising
            else "The development-only candidate meets the prespecified P6 safety and paired-capture consistency screen; a separate preregistration and user authorization are still required before any locked evaluation."
        ),
    }


def collect(baseline_dir: Path, candidate_dirs: list[Path]) -> dict[str, Any]:
    baseline_dir = baseline_dir.resolve()
    baseline_rows = _read_rows(baseline_dir / "episodes.csv")
    baseline_scene_path = baseline_dir / "scenes.jsonl"
    if len(baseline_rows) != EXPECTED_EPISODES or not baseline_scene_path.is_file():
        raise ValueError("Baseline must be the frozen 60-episode block with scenes.jsonl.")
    baseline_metrics = _run_metrics(baseline_rows)
    baseline_scene_hash = frozen_scene_hash(baseline_scene_path)

    runs: list[dict[str, Any]] = []
    arrays_by_metric: dict[str, list[np.ndarray]] = {
        "safe_capture_delta": [],
        "path_delta": [],
        "min_clearance_delta": [],
        "cbf_correction_delta": [],
    }
    for run_dir in candidate_dirs:
        run_dir = run_dir.resolve()
        metadata = _read_json(run_dir / "evaluation_metadata.json")
        rows = _read_rows(run_dir / "episodes.csv")
        provenance = _validate_candidate_metadata(run_dir, metadata, rows, baseline_scene_hash)
        paired, arrays, changes = _paired_run(rows, baseline_rows)
        for name, values in arrays.items():
            arrays_by_metric[name].append(values)
        runs.append(
            {
                "training_seed": int(Path(str(metadata["action_conditioned_jepa_checkpoint"])).parent.name.rsplit("seed", 1)[1]),
                "directory": str(run_dir),
                "provenance": provenance,
                "metrics": _run_metrics(rows),
                "paired_to_baseline": paired,
                "outcome_changes": changes,
            }
        )
    seeds = [run["training_seed"] for run in runs]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("P6 aggregation requires exactly three unique training seeds.")
    if seeds != sorted(seeds):
        raise ValueError("Candidate directories must be supplied in ascending training-seed order.")

    aggregate: dict[str, Any] = {
        "training_seeds": seeds,
        "candidate_safe_capture": _summary([float(run["metrics"]["safe_capture_rate"]) for run in runs]),
        "candidate_collision_rate": _summary([float(run["metrics"]["collision_rate"]) for run in runs]),
        "candidate_boundary_rate": _summary([float(run["metrics"]["boundary_rate"]) for run in runs]),
        "candidate_transit_success_rate": _summary([float(run["metrics"]["transit_success_rate"]) for run in runs]),
        "candidate_mean_path_m": _summary([float(run["metrics"]["mean_total_defender_path_length_m"]) for run in runs]),
        "candidate_mean_min_clearance_m": _summary([float(run["metrics"]["mean_min_clearance_m"]) for run in runs]),
        "candidate_mean_cbf_correction": _summary([float(run["metrics"]["mean_cbf_action_correction_norm"]) for run in runs]),
        "candidate_mean_ledger_credit": _summary([float(run["metrics"]["mean_ledger_credit"]) for run in runs]),
        "candidate_mean_nominal_fallback_fraction": _summary([float(run["metrics"]["mean_nominal_fallback_fraction"]) for run in runs]),
        "candidate_mean_global_fallback_fraction": _summary([float(run["metrics"]["mean_global_fallback_fraction"]) for run in runs]),
        "candidate_collision_total": int(sum(int(run["metrics"]["collision_count"]) for run in runs)),
        "candidate_boundary_total": int(sum(int(run["metrics"]["boundary_count"]) for run in runs)),
        "safe_capture_delta_nonnegative_seed_count": int(sum(float(run["paired_to_baseline"]["safe_capture_delta_percentage_points"]) >= 0.0 for run in runs)),
        "paired_improved_total": int(sum(int(run["paired_to_baseline"]["capture_improved_count"]) for run in runs)),
        "paired_degraded_total": int(sum(int(run["paired_to_baseline"]["capture_degraded_count"]) for run in runs)),
        "paired_tied_total": int(sum(int(run["paired_to_baseline"]["capture_tied_count"]) for run in runs)),
    }
    aggregate["safe_capture_delta_percentage_points"] = _summary(
        [float(run["paired_to_baseline"]["safe_capture_delta_percentage_points"]) for run in runs]
    )
    aggregate["exact_mcnemar_two_sided_pvalue"] = exact_mcnemar_pvalue(
        aggregate["paired_improved_total"], aggregate["paired_degraded_total"]
    )
    for name, value in arrays_by_metric.items():
        aggregate[f"{name}_hierarchical_bootstrap"] = hierarchical_bootstrap_interval(np.stack(value))

    report: dict[str, Any] = {
        "report_type": "jepa_v3_p6_safe_capture_first_paired_development",
        "not_a_locked_test": True,
        "locked_test_opened": False,
        "primary_metric": "safe_capture_with_zero_new_collision_or_boundary",
        "baseline": {
            "directory": str(baseline_dir),
            "frozen_scene_sha256": baseline_scene_hash,
            "output_scenes_sha256": _sha256(baseline_scene_path),
            "metrics": baseline_metrics,
        },
        "runs": runs,
        "aggregate": aggregate,
    }
    report["decision"] = classify(report)
    return report


def _rate(summary: dict[str, float]) -> str:
    return f"{100.0 * summary['mean']:.2f}% +/- {100.0 * summary['sample_std']:.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    baseline = report["baseline"]["metrics"]
    aggregate = report["aggregate"]
    lines = [
        "# JEPA-v3 P6 Safe-Capture-First Paired Development Aggregate",
        "",
        "> Development-only evidence on frozen validation scenes. This run does not open or modify a locked test.",
        "",
        "## Primary Result",
        "",
        f"- Frozen V5 + CBF baseline: `{baseline['safe_capture_count']}/{baseline['episodes']} = {100 * float(baseline['safe_capture_rate']):.1f}%` safe capture, `0` collision, `0` boundary.",
        f"- JEPA + CBF across training seeds `{', '.join(str(seed) for seed in aggregate['training_seeds'])}`: {_rate(aggregate['candidate_safe_capture'])} safe capture; collision {_rate(aggregate['candidate_collision_rate'])}; boundary {_rate(aggregate['candidate_boundary_rate'])}; transit {_rate(aggregate['candidate_transit_success_rate'])}.",
        f"- Paired safe-capture delta: `{aggregate['safe_capture_delta_percentage_points']['mean']:.3f} +/- {aggregate['safe_capture_delta_percentage_points']['sample_std']:.3f}` percentage points; hierarchical 95% CI `{aggregate['safe_capture_delta_hierarchical_bootstrap']['ci_95_low'] * 100:.3f}` to `{aggregate['safe_capture_delta_hierarchical_bootstrap']['ci_95_high'] * 100:.3f}` pp.",
        f"- Paired outcomes: `{aggregate['paired_improved_total']}` improved, `{aggregate['paired_degraded_total']}` degraded, `{aggregate['paired_tied_total']}` tied; exact two-sided McNemar `p={aggregate['exact_mcnemar_two_sided_pvalue']:.6f}`.",
        "",
        "## Per-Seed Paired Outcomes",
        "",
        "| Training seed | Safe capture | Collision / boundary | Transit | Improved / degraded / tied | Delta (pp) | Path delta (m) | Clearance delta (m) | CBF delta | Ledger credit | Nominal fallback |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report["runs"]:
        metrics = run["metrics"]
        paired = run["paired_to_baseline"]
        lines.append(
            f"| {run['training_seed']} | {metrics['safe_capture_count']}/{metrics['episodes']} ({100 * float(metrics['safe_capture_rate']):.1f}%) | "
            f"{metrics['collision_count']} / {metrics['boundary_count']} | {100 * float(metrics['transit_success_rate']):.1f}% | "
            f"{paired['capture_improved_count']} / {paired['capture_degraded_count']} / {paired['capture_tied_count']} | "
            f"{paired['safe_capture_delta_percentage_points']:.3f} | {paired['paired_path_delta_m']:.3f} | "
            f"{paired['paired_min_clearance_delta_m']:.4f} | {paired['paired_cbf_correction_delta']:.5f} | "
            f"{float(metrics['mean_ledger_credit']):.4f} | {100 * float(metrics['mean_nominal_fallback_fraction']):.2f}% |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- Classification: `{report['decision']['classification']}`.",
        f"- {report['decision']['reason']}",
        "- Capture time remains reported in the JSON provenance but is not an automatic rejection condition under the dated safe-capture-first amendment.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.baseline_dir, args.candidate_dir)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
