"""Aggregate Central V4 fixed and randomized locked-test metric replays.

The formal statistical unit is the independently trained checkpoint seed.  The
100 episode rows within a checkpoint are scenarios, not independent training
runs.  Metric replays must use the already-opened locked-test seeds and may add
instrumentation only; they must not be used for model selection or tuning.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


TRAINING_SEEDS = (661201, 661202, 661203)
FIXED_SCENES = ("s1_cylinder", "s1_box", "s1_wall", "s2")
MODES = ("raw", "cbf")
METRIC_REPLAY_SUFFIX = "_metrics_100"
REFERENCE_SUFFIX = "_100"

RATE_FIELDS = {
    "cooperative_safe_capture_rate": "cooperative_safe_capture",
    "capture_rate": "capture_event",
    "target_zone_entry_rate": "target_zone_entered",
    "transit_success_rate": "transit_success",
    "collision_rate": "collision",
}
NUMERIC_FIELDS = {
    "mean_pursuer_zone_entry_count": "defender_zone_entry_count",
    "mean_min_clearance_m": "min_clearance_m",
    "mean_defender_path_length_m": "mean_defender_path_length_m",
    "mean_total_defender_path_length_m": "total_defender_path_length_m",
    "mean_cbf_action_correction_norm": "mean_cbf_action_correction_norm",
}
ACROSS_SEED_METRICS = (
    "cooperative_safe_capture_rate",
    "capture_rate",
    "target_zone_entry_rate",
    "mean_pursuer_zone_entry_count",
    "transit_success_rate",
    "collision_rate",
    "boundary_violation_rate",
    "mean_time_to_capture_seconds",
    "mean_min_clearance_m",
    "worst_min_clearance_m",
    "mean_defender_path_length_m",
    "mean_total_defender_path_length_m",
    "mean_cbf_action_correction_norm",
    "max_cbf_action_correction_norm",
)
OUTCOME_FIELDS = (
    "cooperative_safe_capture",
    "capture_event",
    "collision",
    "world_violation_steps",
    "transit_success",
    "termination_reason",
    "task_termination_reason",
)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _literal_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or str(value).strip() == "":
        return []
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list literal, got {value!r}")
    return parsed


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return document


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no episode rows.")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_rows(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"episodes": len(rows)}
    for metric, field in RATE_FIELDS.items():
        summary[metric] = sum(_bool(row[field]) for row in rows) / len(rows)
    summary["boundary_violation_rate"] = sum(float(row["world_violation_steps"]) > 0 for row in rows) / len(rows)
    for metric, field in NUMERIC_FIELDS.items():
        values = [_float(row.get(field)) for row in rows]
        if any(value is None for value in values):
            raise ValueError(f"Metric replay rows are missing required field {field!r}.")
        summary[metric] = statistics.fmean(float(value) for value in values if value is not None)
    capture_times = [_float(row.get("capture_time_seconds")) for row in rows]
    capture_times = [value for value in capture_times if value is not None]
    summary["mean_time_to_capture_seconds"] = statistics.fmean(capture_times) if capture_times else None
    clearances = [float(row["min_clearance_m"]) for row in rows]
    summary["worst_min_clearance_m"] = min(clearances)
    corrections = [float(row["max_cbf_action_correction_norm"]) for row in rows]
    summary["max_cbf_action_correction_norm"] = max(corrections)
    summary["termination_reasons"] = dict(sorted(Counter(row["termination_reason"] for row in rows).items()))
    return summary


def aggregate_seed_metrics(per_seed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"training_seeds": len(per_seed)}
    for metric in ACROSS_SEED_METRICS:
        values = [payload.get(metric) for payload in per_seed.values()]
        if any(value is None for value in values):
            aggregate[metric] = None
            continue
        numeric = [float(value) for value in values]
        aggregate[metric] = {
            "mean": statistics.fmean(numeric),
            "sample_std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
            "min": min(numeric),
            "max": max(numeric),
        }
    return aggregate


def _checkpoint_details(metadata: dict[str, Any]) -> dict[str, str]:
    raw_path = metadata.get("checkpoint")
    if not raw_path:
        raise ValueError("Locked-test metadata does not name a checkpoint.")
    path = Path(str(raw_path))
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return {"path": str(path), "sha256": _sha256(path)}


def _validate_fixed_metadata(
    document: dict[str, Any], *, scene: str, mode: str, expected_episodes: int
) -> None:
    if document.get("evaluation_type") != "central_v4_fixed_locked_test" or document.get("locked_test") is not True:
        raise ValueError("Fixed artifact is not marked as a Central V4 locked test.")
    if int(document.get("base_seed", -1)) != 660501:
        raise ValueError("Fixed locked-test artifact must use base seed 660501.")
    if int(document.get("summary", {}).get("episodes", -1)) != expected_episodes:
        raise ValueError("Fixed locked-test artifact has the wrong episode count.")
    if bool(document.get("use_cbf")) != (mode == "cbf"):
        raise ValueError("Fixed locked-test CBF metadata does not match its directory mode.")
    expected_scenario = "v4_s2" if scene == "s2" else "s1"
    expected_layout = "mixed" if scene == "s2" else scene.removeprefix("s1_")
    if document.get("scenario_kind") != expected_scenario or document.get("layout") != expected_layout:
        raise ValueError(f"Fixed artifact does not match scene contract {scene!r}.")


def _validate_s3_metadata(document: dict[str, Any], *, mode: str, expected_episodes: int) -> None:
    if document.get("evaluation_type") != "randomized_central_mixed_obstacle_s3_locked_test":
        raise ValueError("S3 artifact has the wrong evaluation type.")
    if document.get("locked_test") is not True or document.get("split") != "locked_test":
        raise ValueError("S3 artifact is not marked as a locked test.")
    if int(document.get("seed_block", -1)) != 647001:
        raise ValueError("S3 locked-test artifact must use seed block 647001.")
    if int(document.get("episodes", -1)) != expected_episodes:
        raise ValueError("S3 locked-test artifact has the wrong episode count.")
    if bool(document.get("use_cbf")) != (mode == "cbf"):
        raise ValueError("S3 locked-test CBF metadata does not match its directory mode.")


def _compare_rows(current: Sequence[dict[str, str]], reference: Sequence[dict[str, str]]) -> dict[str, Any]:
    if len(current) != len(reference):
        return {"identical": False, "mismatch_count": abs(len(current) - len(reference)), "mismatches": []}
    mismatches: list[dict[str, Any]] = []
    for index, (new_row, old_row) in enumerate(zip(current, reference, strict=True)):
        fields = [field for field in OUTCOME_FIELDS if str(new_row.get(field, "")) != str(old_row.get(field, ""))]
        if fields:
            mismatches.append({"episode_index": index, "fields": fields})
    return {"identical": not mismatches, "mismatch_count": len(mismatches), "mismatches": mismatches[:20]}


def _planned_clearance_band(row: dict[str, str]) -> str:
    values = [float(value) for value in _literal_list(row.get("defender_transit_min_clearance_m"))]
    target = _float(row.get("target_transit_min_clearance_m"))
    if target is not None:
        values.append(target)
    if not values:
        return "unknown"
    clearance = min(values)
    if clearance < 0.65:
        return "narrow: planned clearance <0.65 m"
    if clearance < 0.80:
        return "medium: planned clearance 0.65-0.80 m"
    return "wide: planned clearance >=0.80 m"


def _group_failure_rows(rows: Iterable[dict[str, str]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        key = _planned_clearance_band(row) if field == "planned_route_clearance_band" else str(row[field])
        grouped.setdefault(key, []).append(row)
    output: dict[str, Any] = {}
    for key, subset in sorted(grouped.items()):
        total = len(subset)
        cooperative_failures = sum(not _bool(row["cooperative_safe_capture"]) for row in subset)
        output[key] = {
            "scenario_evaluations": total,
            "unique_locked_scenarios": len({int(row["episode_index"]) for row in subset}),
            "cooperative_capture_failures": cooperative_failures,
            "cooperative_capture_failure_rate": cooperative_failures / total,
            "ordinary_capture_failures": sum(not _bool(row["capture_event"]) for row in subset),
            "collisions": sum(_bool(row["collision"]) for row in subset),
            "boundary_violations": sum(float(row["world_violation_steps"]) > 0 for row in subset),
            "timeouts": sum(row["termination_reason"] == "timeout" for row in subset),
            "transit_failures": sum(not _bool(row["transit_success"]) for row in subset),
        }
    return output


def build_failure_analysis(per_mode_rows: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    dimensions = (
        "obstacle_count",
        "layout_signature",
        "planned_route_clearance_band",
        "defender_side",
        "observation_condition",
        "target_speed_scale",
        "target_motion_mode",
    )
    return {
        mode: {field: _group_failure_rows(rows, field) for field in dimensions}
        for mode, rows in per_mode_rows.items()
    }


def build_transit_audit(per_mode_rows: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    evaluations: list[dict[str, Any]] = []
    unique: dict[tuple[int, int], dict[str, Any]] = {}
    for mode, rows in per_mode_rows.items():
        for row in rows:
            if _bool(row["transit_success"]):
                continue
            item = {
                "training_seed": int(row["training_seed"]),
                "mode": mode,
                "episode_index": int(row["episode_index"]),
                "episode_seed": int(row["episode_seed"]),
                "layout_seed": int(row["layout_seed"]),
                "capture_event": _bool(row["capture_event"]),
                "cooperative_safe_capture": _bool(row["cooperative_safe_capture"]),
                "collision": _bool(row["collision"]),
                "boundary_violation": float(row["world_violation_steps"]) > 0,
                "target_transit_success": _bool(row["target_transit_success"]),
                "target_transit_reason": row["target_transit_reason"],
                "target_transit_execution_min_clearance_m": _float(
                    row.get("target_transit_execution_min_clearance_m")
                ),
            }
            evaluations.append(item)
            unique.setdefault(
                (item["episode_seed"], item["layout_seed"]),
                {key: value for key, value in item.items() if key not in {"training_seed", "mode"}},
            )
    return {
        "failed_scenario_evaluations": len(evaluations),
        "unique_failed_locked_scenarios": len(unique),
        "unique_scenarios": list(unique.values()),
        "evaluations": evaluations,
    }


def collect_locked_test(
    root: Path,
    *,
    training_seeds: Sequence[int] = TRAINING_SEEDS,
    fixed_scenes: Sequence[str] = FIXED_SCENES,
    modes: Sequence[str] = MODES,
    suffix: str = METRIC_REPLAY_SUFFIX,
    reference_suffix: str | None = REFERENCE_SUFFIX,
    expected_episodes: int = 100,
) -> dict[str, Any]:
    root = root.resolve()
    checkpoints: dict[str, dict[str, str]] = {}
    consistency: dict[str, Any] = {}
    fixed: dict[str, Any] = {}
    s3: dict[str, Any] = {}
    s3_rows_by_mode = {mode: [] for mode in modes}

    for scene in fixed_scenes:
        fixed[scene] = {}
        for mode in modes:
            per_seed: dict[str, dict[str, Any]] = {}
            for seed in training_seeds:
                name = f"locked_bc_retained_seed{seed}_{scene}_{mode}{suffix}"
                directory = root / name
                metadata = _read_json(directory / "summary.json")
                _validate_fixed_metadata(
                    metadata,
                    scene=scene,
                    mode=mode,
                    expected_episodes=expected_episodes,
                )
                details = _checkpoint_details(metadata)
                previous = checkpoints.setdefault(str(seed), details)
                if previous != details:
                    raise ValueError(f"Training seed {seed} refers to inconsistent checkpoints.")
                rows = _read_rows(directory / "episodes.csv")
                if len(rows) != expected_episodes:
                    raise ValueError(f"{name} has {len(rows)} rows, expected {expected_episodes}.")
                per_seed[str(seed)] = summarize_rows(rows)
                if reference_suffix is not None:
                    reference = root / f"locked_bc_retained_seed{seed}_{scene}_{mode}{reference_suffix}"
                    consistency[name] = _compare_rows(rows, _read_rows(reference / "episodes.csv"))
            fixed[scene][mode] = {
                "per_training_seed": per_seed,
                "across_training_seeds": aggregate_seed_metrics(per_seed),
            }

    for mode in modes:
        per_seed = {}
        for seed in training_seeds:
            name = f"locked_s3_bc_retained_seed{seed}_{mode}{suffix}"
            directory = root / name
            metadata = _read_json(directory / "evaluation_metadata.json")
            _validate_s3_metadata(metadata, mode=mode, expected_episodes=expected_episodes)
            details = _checkpoint_details(metadata)
            previous = checkpoints.setdefault(str(seed), details)
            if previous != details:
                raise ValueError(f"Training seed {seed} refers to inconsistent checkpoints.")
            rows = _read_rows(directory / "episodes.csv")
            if len(rows) != expected_episodes:
                raise ValueError(f"{name} has {len(rows)} rows, expected {expected_episodes}.")
            for row in rows:
                row["training_seed"] = str(seed)
            s3_rows_by_mode[mode].extend(rows)
            per_seed[str(seed)] = summarize_rows(rows)
            if reference_suffix is not None:
                reference = root / f"locked_s3_bc_retained_seed{seed}_{mode}{reference_suffix}"
                consistency[name] = _compare_rows(rows, _read_rows(reference / "episodes.csv"))
        s3[mode] = {
            "per_training_seed": per_seed,
            "across_training_seeds": aggregate_seed_metrics(per_seed),
        }

    replay_identical = all(item["identical"] for item in consistency.values()) if consistency else None
    return {
        "evaluation_type": "central_v4_locked_test_aggregate",
        "locked_test": True,
        "metric_replay_only": True,
        "no_tuning_after_locked_test_opened": True,
        "metric_instrumentation_commit": "6e53efa560b5895da4a83721b8e9bf91b610b77b",
        "metric_replay_commits": [
            "d8ddaf740c120d22599e404e2920a8949a8e42be",
            "5cd53600765ddd85bb7938d91da506f119652169",
        ],
        "statistical_unit": "independently_trained_checkpoint_seed",
        "training_seeds": list(training_seeds),
        "locked_seed_contract": {"fixed_s1_s2": 660501, "random_s3": 647001},
        "episodes_per_artifact": expected_episodes,
        "checkpoints": checkpoints,
        "fixed": fixed,
        "s3": s3,
        "failure_analysis": build_failure_analysis(s3_rows_by_mode),
        "transit_failure_audit": build_transit_audit(s3_rows_by_mode),
        "metric_replay_consistency": {
            "reference_suffix": reference_suffix,
            "all_outcomes_identical": replay_identical,
            "artifacts": consistency,
        },
    }


def _mean(payload: dict[str, Any], metric: str) -> float | None:
    value = payload["across_training_seeds"].get(metric)
    return None if value is None else float(value["mean"])


def _mean_std(payload: dict[str, Any], metric: str, *, percent: bool = False, unit: str = "") -> str:
    value = payload["across_training_seeds"].get(metric)
    if value is None:
        return "n/a"
    scale = 100.0 if percent else 1.0
    decimals = 1 if percent else (3 if "correction" in metric else 2)
    return (
        f"{scale * float(value['mean']):.{decimals}f} +/- "
        f"{scale * float(value['sample_std']):.{decimals}f}{unit}"
    )


def render_markdown(aggregate: dict[str, Any]) -> str:
    lines = [
        "# Central V4 Locked-Test Report (D2-D4)",
        "",
        "## Contract and decision hygiene",
        "",
        "The retained BC checkpoints were frozen before the locked test was opened. The metric replay used the same models, fixed seed `660501`, S3 seed block `647001`, and 100 episodes per artifact. Commit `6e53efa` adds path-length and CBF-correction instrumentation; commit `d8ddaf7` reuses the policy-independent Transit evidence and commit `5cd5360` restores the already-frozen S3 scenes without re-sampling. No model, hyperparameter, scene, or threshold was tuned after opening the locked block.",
        "",
        f"Replay outcomes identical to the original locked artifacts: **{aggregate['metric_replay_consistency']['all_outcomes_identical']}**.",
        "",
        "## Frozen checkpoints",
        "",
        "| Training seed | SHA-256 |",
        "| ---: | --- |",
    ]
    for seed, payload in aggregate["checkpoints"].items():
        lines.append(f"| {seed} | `{payload['sha256']}` |")

    lines.extend(
        [
            "",
            "## Fixed S1/S2 CBF regression",
            "",
            "Values are mean +/- sample standard deviation across the three independently trained checkpoints.",
            "",
            "| Scene | Cooperative capture | Capture | Collision | Boundary | Transit | Mean path / defender | Mean CBF correction |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for scene in FIXED_SCENES:
        if scene not in aggregate["fixed"]:
            continue
        payload = aggregate["fixed"][scene]["cbf"]
        lines.append(
            f"| {scene} | {_mean_std(payload, 'cooperative_safe_capture_rate', percent=True, unit='%')} | "
            f"{_mean_std(payload, 'capture_rate', percent=True, unit='%')} | "
            f"{_mean_std(payload, 'collision_rate', percent=True, unit='%')} | "
            f"{_mean_std(payload, 'boundary_violation_rate', percent=True, unit='%')} | "
            f"{_mean_std(payload, 'transit_success_rate', percent=True, unit='%')} | "
            f"{_mean_std(payload, 'mean_defender_path_length_m', unit=' m')} | "
            f"{_mean_std(payload, 'mean_cbf_action_correction_norm')} |"
        )

    lines.extend(
        [
            "",
            "## Random S3 locked test",
            "",
            "| Execution | Cooperative capture | Capture | Target zone entry | Pursuer zone entries | Transit | Collision | Boundary | Time to capture | Min clearance | Path / defender | CBF correction |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode in MODES:
        if mode not in aggregate["s3"]:
            continue
        payload = aggregate["s3"][mode]
        lines.append(
            f"| {mode.upper()} | {_mean_std(payload, 'cooperative_safe_capture_rate', percent=True, unit='%')} | "
            f"{_mean_std(payload, 'capture_rate', percent=True, unit='%')} | "
            f"{_mean_std(payload, 'target_zone_entry_rate', percent=True, unit='%')} | "
            f"{_mean_std(payload, 'mean_pursuer_zone_entry_count')} | "
            f"{_mean_std(payload, 'transit_success_rate', percent=True, unit='%')} | "
            f"{_mean_std(payload, 'collision_rate', percent=True, unit='%')} | "
            f"{_mean_std(payload, 'boundary_violation_rate', percent=True, unit='%')} | "
            f"{_mean_std(payload, 'mean_time_to_capture_seconds', unit=' s')} | "
            f"{_mean_std(payload, 'mean_min_clearance_m', unit=' m')} | "
            f"{_mean_std(payload, 'mean_defender_path_length_m', unit=' m')} | "
            f"{_mean_std(payload, 'mean_cbf_action_correction_norm')} |"
        )

    lines.extend(
        [
            "",
            "The raw actor is not safely deployable: its S3 collision rate is the complement of its very low capture rate in nearly every episode. CBF is therefore part of the retained execution stack, not evidence attributable to the policy network alone.",
            "",
            "## CBF failure groups on S3",
            "",
            "Counts below pool the 300 checkpoint-scenario evaluations for descriptive failure analysis. Main uncertainty remains across the three training seeds. The planned-route clearance band is a reproducible proxy for channel width, not a direct geometric doorway-width measurement.",
        ]
    )
    labels = {
        "obstacle_count": "Obstacle count",
        "layout_signature": "Obstacle layout",
        "planned_route_clearance_band": "Channel-clearance proxy",
        "defender_side": "Defender birth side",
        "observation_condition": "Observation condition",
        "target_speed_scale": "Target speed scale",
        "target_motion_mode": "Target motion mode",
    }
    for dimension, groups in aggregate["failure_analysis"]["cbf"].items():
        lines.extend(
            [
                "",
                f"### {labels[dimension]}",
                "",
                "| Group | Evaluations | Cooperative failures | Collision | Boundary | Timeout | Transit failure |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for group, payload in groups.items():
            lines.append(
                f"| {group} | {payload['scenario_evaluations']} | {payload['cooperative_capture_failures']} "
                f"({100.0 * payload['cooperative_capture_failure_rate']:.1f}%) | {payload['collisions']} | "
                f"{payload['boundary_violations']} | {payload['timeouts']} | {payload['transit_failures']} |"
            )

    audit = aggregate["transit_failure_audit"]
    lines.extend(
        [
            "",
            "## Shared Transit failure audit",
            "",
            f"There are {audit['unique_failed_locked_scenarios']} unique locked scenario(s) with Transit failure across {audit['failed_scenario_evaluations']} raw/CBF checkpoint evaluations.",
            "",
            "| Episode | Episode seed | Layout seed | Capture | Collision | Boundary | Target Transit reason | Execution clearance |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for item in audit["unique_scenarios"]:
        clearance = item["target_transit_execution_min_clearance_m"]
        clearance_text = "n/a" if clearance is None else f"{float(clearance):.6f} m"
        lines.append(
            f"| {item['episode_index']} | {item['episode_seed']} | {item['layout_seed']} | "
            f"{item['capture_event']} | {item['collision']} | {item['boundary_violation']} | "
            f"{item['target_transit_reason']} | {clearance_text} |"
        )

    cbf = aggregate["s3"]["cbf"]
    lines.extend(
        [
            "",
            "## Locked conclusion",
            "",
            f"Retained BC+CBF achieves {_mean_std(cbf, 'cooperative_safe_capture_rate', percent=True, unit='%')} Cooperative Safe Capture on S3. This is lower than validation and includes non-zero safety failures, so the claim is a reproducible partial success rather than solved robust capture. The locked block remains closed to tuning.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/central_v4"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--suffix", default=METRIC_REPLAY_SUFFIX)
    parser.add_argument("--reference-suffix", default=REFERENCE_SUFFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate = collect_locked_test(
        args.results_root,
        suffix=args.suffix,
        reference_suffix=args.reference_suffix or None,
    )
    if aggregate["metric_replay_consistency"]["all_outcomes_identical"] is False:
        raise RuntimeError("Metric replay changed one or more locked-test outcomes.")
    args.output_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(aggregate), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md)}, indent=2))


if __name__ == "__main__":
    main()
