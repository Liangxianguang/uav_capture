"""Aggregate a frozen three-seed V5 S3 locked test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


LOCKED_SEED_BLOCK = 647201
EPISODES_PER_CHECKPOINT = 100
MODES = ("raw", "cbf")
GROUP_FIELDS = (
    "observation_condition",
    "obstacle_count",
    "planned_route_clearance_band",
    "target_motion_mode",
)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return document


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EPISODES_PER_CHECKPOINT:
        raise ValueError(f"{path} has {len(rows)} rows, expected {EPISODES_PER_CHECKPOINT}.")
    return rows


def _wilson_interval(successes: int, trials: int) -> list[float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Wilson interval requires valid binomial counts.")
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt((rate * (1.0 - rate) + z * z / (4.0 * trials)) / trials) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _static_scene_digest(path: Path) -> str:
    records: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path} contains a non-object scene.")
        static = {key: record[key] for key in ("episode_index", "spec", "scenario")}
        records.append(json.dumps(static, sort_keys=True, separators=(",", ":")))
    if len(records) != EPISODES_PER_CHECKPOINT:
        raise ValueError(f"{path} has {len(records)} static scenes, expected {EPISODES_PER_CHECKPOINT}.")
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def _rate(rows: Sequence[dict[str, str]], field: str) -> tuple[float, int]:
    successes = sum(_bool(row[field]) for row in rows)
    return successes / len(rows), successes


def _mean_numeric(rows: Sequence[dict[str, str]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if str(row.get(field, "")).strip()]
    return statistics.fmean(values) if values else None


def summarize_rows(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    cooperative_rate, cooperative_successes = _rate(rows, "cooperative_safe_capture")
    capture_rate, capture_successes = _rate(rows, "capture_event")
    transit_rate, transit_successes = _rate(rows, "transit_success")
    collision_rate, collision_episodes = _rate(rows, "collision")
    boundary_episodes = sum(float(row["world_violation_steps"]) > 0.0 for row in rows)
    capture_times = [
        float(row["capture_time_seconds"])
        for row in rows
        if str(row.get("capture_time_seconds", "")).strip()
    ]
    corrections = [float(row.get("mean_cbf_action_correction_norm", 0.0)) for row in rows]
    return {
        "episodes": len(rows),
        "cooperative_safe_capture_rate": cooperative_rate,
        "cooperative_safe_capture_successes": cooperative_successes,
        "capture_rate": capture_rate,
        "capture_successes": capture_successes,
        "collision_rate": collision_rate,
        "collision_episodes": collision_episodes,
        "boundary_violation_rate": boundary_episodes / len(rows),
        "boundary_violation_episodes": boundary_episodes,
        "transit_success_rate": transit_rate,
        "transit_successes": transit_successes,
        "timeout_episodes": sum(row.get("termination_reason") == "timeout" for row in rows),
        "mean_capture_time_seconds": statistics.fmean(capture_times) if capture_times else None,
        "mean_min_clearance_m": _mean_numeric(rows, "min_clearance_m"),
        "worst_min_clearance_m": min(float(row["min_clearance_m"]) for row in rows),
        "mean_defender_path_length_m": _mean_numeric(rows, "mean_defender_path_length_m"),
        "cbf_correction": {
            "mean": statistics.fmean(corrections),
            "median": statistics.median(corrections),
            "p95": sorted(corrections)[math.ceil(0.95 * len(corrections)) - 1],
            "max": max(float(row.get("max_cbf_action_correction_norm", 0.0)) for row in rows),
        },
        "termination_reasons": dict(sorted(Counter(row["termination_reason"] for row in rows).items())),
    }


def _aggregate_seed_values(per_seed: dict[str, dict[str, Any]], metric: str) -> dict[str, float] | None:
    values = [entry[metric] for entry in per_seed.values()]
    if any(value is None for value in values):
        return None
    numeric = [float(value) for value in values]
    return {
        "mean": statistics.fmean(numeric),
        "sample_std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
        "min": min(numeric),
        "max": max(numeric),
    }


def _group_failures(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    groups: dict[str, dict[str, list[dict[str, str]]]] = {field: {} for field in GROUP_FIELDS}
    for row in rows:
        for field in GROUP_FIELDS:
            groups[field].setdefault(str(row[field]), []).append(row)
    result: dict[str, Any] = {}
    for field, buckets in groups.items():
        result[field] = {}
        for key, subset in sorted(buckets.items()):
            total = len(subset)
            failures = sum(not _bool(row["cooperative_safe_capture"]) for row in subset)
            result[field][key] = {
                "episodes": total,
                "cooperative_failures": failures,
                "cooperative_failure_rate": failures / total,
                "collision_episodes": sum(_bool(row["collision"]) for row in subset),
                "boundary_episodes": sum(float(row["world_violation_steps"]) > 0.0 for row in subset),
                "timeout_episodes": sum(row["termination_reason"] == "timeout" for row in subset),
                "transit_failures": sum(not _bool(row["transit_success"]) for row in subset),
            }
    return result


def _validate_development_entry(entry: dict[str, Any]) -> dict[str, Any]:
    for key in ("run_id", "training_seed", "run_dir", "validation_summary"):
        if key not in entry:
            raise ValueError(f"Locked manifest model entry is missing {key!r}.")
    run_dir = Path(str(entry["run_dir"])).resolve()
    checkpoint = run_dir / "checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint for {entry['run_id']}: {checkpoint}")
    summary = _read_json(Path(str(entry["validation_summary"])).resolve())
    if summary.get("candidate_gate_passed") is not True:
        raise ValueError(f"Development gate did not pass for {entry['run_id']}.")
    if str(summary.get("training", {}).get("checkpoint_sha256")) != _sha256(checkpoint):
        raise ValueError(f"Development summary checkpoint hash mismatch for {entry['run_id']}.")
    if summary.get("s3_scene_pairing", {}).get("static_scenes_exactly_paired") is not True:
        raise ValueError(f"Development raw/CBF scenes are not paired for {entry['run_id']}.")
    return {
        "run_id": str(entry["run_id"]),
        "training_seed": int(entry["training_seed"]),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "validation_summary": str(Path(str(entry["validation_summary"])).resolve()),
    }


def collect(manifest_path: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if int(manifest.get("locked_seed_block", -1)) != LOCKED_SEED_BLOCK:
        raise ValueError("Locked manifest must declare V5 seed block 647201.")
    if int(manifest.get("episodes_per_checkpoint", -1)) != EPISODES_PER_CHECKPOINT:
        raise ValueError("Locked manifest must require exactly 100 episodes per checkpoint.")
    entries = manifest.get("models")
    if not isinstance(entries, list) or len(entries) != 3:
        raise ValueError("Locked manifest must contain exactly three passing development models.")
    models = [_validate_development_entry(entry) for entry in entries]
    if len({model["run_id"] for model in models}) != 3 or len({model["training_seed"] for model in models}) != 3:
        raise ValueError("Locked manifest model run IDs and training seeds must be distinct.")

    root = Path(str(manifest["evaluation_root"])).resolve()
    per_mode: dict[str, Any] = {}
    all_cbf_rows: list[dict[str, str]] = []
    raw_digests: dict[str, str] = {}
    cbf_digests: dict[str, str] = {}
    for mode in MODES:
        seed_metrics: dict[str, dict[str, Any]] = {}
        for model in models:
            directory = root / f"locked_s3_{model['run_id']}_{mode}_100"
            metadata = _read_json(directory / "evaluation_metadata.json")
            metadata_valid = (
                metadata.get("evaluation_type") == "randomized_central_mixed_obstacle_s3_locked_test"
                and metadata.get("split") == "locked_test"
                and metadata.get("locked_test") is True
                and int(metadata.get("seed_block", -1)) == LOCKED_SEED_BLOCK
                and int(metadata.get("episodes", -1)) == EPISODES_PER_CHECKPOINT
                and bool(metadata.get("use_cbf")) == (mode == "cbf")
            )
            if not metadata_valid:
                raise ValueError(f"{directory} violates the V5 locked-test metadata contract.")
            if str(Path(str(metadata.get("checkpoint"))).resolve()) != model["checkpoint"]:
                raise ValueError(f"{directory} names the wrong checkpoint.")
            rows = _read_rows(directory / "episodes.csv")
            digest = _static_scene_digest(directory / "scenes.jsonl")
            if mode == "raw":
                raw_digests[model["run_id"]] = digest
            else:
                cbf_digests[model["run_id"]] = digest
                all_cbf_rows.extend(rows)
            seed_metrics[str(model["training_seed"])] = {
                **summarize_rows(rows),
                "artifact_directory": str(directory),
                "episodes_csv_sha256": _sha256(directory / "episodes.csv"),
                "static_scene_sha256": digest,
            }
        per_mode[mode] = {
            "per_training_seed": seed_metrics,
            "across_training_seeds": {
                metric: _aggregate_seed_values(seed_metrics, metric)
                for metric in (
                    "cooperative_safe_capture_rate",
                    "capture_rate",
                    "collision_rate",
                    "boundary_violation_rate",
                    "transit_success_rate",
                    "mean_capture_time_seconds",
                    "mean_min_clearance_m",
                    "worst_min_clearance_m",
                    "mean_defender_path_length_m",
                )
            },
        }

    if raw_digests != cbf_digests:
        raise ValueError("Locked raw/CBF scene digests are not paired for every training seed.")
    cbf_per_seed = per_mode["cbf"]["per_training_seed"]
    gates = {
        seed: {
            "cooperative_safe_capture_at_least_85_percent": metrics["cooperative_safe_capture_rate"] >= 0.85,
            "collision_at_most_2_percent": metrics["collision_rate"] <= 0.02,
            "boundary_at_most_2_percent": metrics["boundary_violation_rate"] <= 0.02,
            "transit_at_least_99_percent": metrics["transit_success_rate"] >= 0.99,
        }
        for seed, metrics in cbf_per_seed.items()
    }
    pooled_successes = sum(metrics["cooperative_safe_capture_successes"] for metrics in cbf_per_seed.values())
    pooled_episodes = sum(metrics["episodes"] for metrics in cbf_per_seed.values())
    return {
        "evaluation_type": "central_v5_shapeaware_retained_bc_locked_test",
        "locked_test": True,
        "locked_seed_block": LOCKED_SEED_BLOCK,
        "episodes_per_checkpoint": EPISODES_PER_CHECKPOINT,
        "models": models,
        "raw_cbf_scene_digests_paired": raw_digests == cbf_digests,
        "scene_digest_by_run_id": raw_digests,
        "s3": per_mode,
        "cbf_failure_groups_pooled": _group_failures(all_cbf_rows),
        "cbf_pooled_episode_wilson_95": _wilson_interval(pooled_successes, pooled_episodes),
        "per_seed_gates": gates,
        "locked_gate_passed": all(all(values.values()) for values in gates.values()),
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _mean_std(values: dict[str, Any], metric: str, *, percent: bool = False, unit: str = "") -> str:
    payload = values[metric]
    if payload is None:
        return "n/a"
    scale = 100.0 if percent else 1.0
    decimals = 1 if percent else 2
    return f"{scale * payload['mean']:.{decimals}f} +/- {scale * payload['sample_std']:.{decimals}f}{unit}"


def render_markdown(aggregate: dict[str, Any]) -> str:
    raw = aggregate["s3"]["raw"]["across_training_seeds"]
    cbf = aggregate["s3"]["cbf"]["across_training_seeds"]
    low, high = aggregate["cbf_pooled_episode_wilson_95"]
    lines = [
        "# Central V5 Shape-Aware Retained-BC Locked-Test Report",
        "",
        "All three checkpoints passed the frozen development gate before V5 locked block 647201 was opened.",
        "The independent statistical units are the three trained checkpoints; 100 episodes per checkpoint support scenario-level diagnosis.",
        "",
        "## Frozen Checkpoints",
        "",
        "| Training seed | Run ID | SHA-256 |",
        "| ---: | --- | --- |",
    ]
    for model in aggregate["models"]:
        lines.append(f"| {model['training_seed']} | {model['run_id']} | {model['checkpoint_sha256']} |")
    lines.extend(
        [
            "",
            "## S3 Locked Test",
            "",
            "| Execution | Cooperative Safe Capture | Capture | Collision | Boundary | Transit | Time to Capture | Path / Defender |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| RAW | {_mean_std(raw, 'cooperative_safe_capture_rate', percent=True, unit='%')} | "
            f"{_mean_std(raw, 'capture_rate', percent=True, unit='%')} | "
            f"{_mean_std(raw, 'collision_rate', percent=True, unit='%')} | "
            f"{_mean_std(raw, 'boundary_violation_rate', percent=True, unit='%')} | "
            f"{_mean_std(raw, 'transit_success_rate', percent=True, unit='%')} | "
            f"{_mean_std(raw, 'mean_capture_time_seconds', unit=' s')} | "
            f"{_mean_std(raw, 'mean_defender_path_length_m', unit=' m')} |",
            f"| CBF | {_mean_std(cbf, 'cooperative_safe_capture_rate', percent=True, unit='%')} | "
            f"{_mean_std(cbf, 'capture_rate', percent=True, unit='%')} | "
            f"{_mean_std(cbf, 'collision_rate', percent=True, unit='%')} | "
            f"{_mean_std(cbf, 'boundary_violation_rate', percent=True, unit='%')} | "
            f"{_mean_std(cbf, 'transit_success_rate', percent=True, unit='%')} | "
            f"{_mean_std(cbf, 'mean_capture_time_seconds', unit=' s')} | "
            f"{_mean_std(cbf, 'mean_defender_path_length_m', unit=' m')} |",
            "",
            f"Pooled CBF episode-level Cooperative Safe Capture Wilson 95% CI: {_pct(low)} to {_pct(high)}.",
            f"Raw/CBF static scene pairing verified for every checkpoint: {aggregate['raw_cbf_scene_digests_paired']}.",
            "",
            "## Per-Seed Gates",
            "",
            "| Training seed | Cooperative >= 85% | Collision <= 2% | Boundary <= 2% | Transit >= 99% |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    for seed, gates in aggregate["per_seed_gates"].items():
        lines.append(
            f"| {seed} | {gates['cooperative_safe_capture_at_least_85_percent']} | "
            f"{gates['collision_at_most_2_percent']} | {gates['boundary_at_most_2_percent']} | "
            f"{gates['transit_at_least_99_percent']} |"
        )
    lines.extend(["", "## CBF Failure Groups", ""])
    labels = {
        "observation_condition": "Observation condition",
        "obstacle_count": "Obstacle count",
        "planned_route_clearance_band": "Planned clearance proxy",
        "target_motion_mode": "Target motion",
    }
    for field, label in labels.items():
        lines.extend(
            [
                f"### {label}",
                "",
                "| Group | Episodes | Cooperative failures | Collision | Boundary | Timeout | Transit failure |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for group, values in aggregate["cbf_failure_groups_pooled"][field].items():
            lines.append(
                f"| {group} | {values['episodes']} | {values['cooperative_failures']} "
                f"({_pct(values['cooperative_failure_rate'])}) | {values['collision_episodes']} | "
                f"{values['boundary_episodes']} | {values['timeout_episodes']} | {values['transit_failures']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Locked Decision",
            "",
            f"All three pre-registered locked gates pass: {aggregate['locked_gate_passed']}.",
            "Raw actor and policy + CBF remain separate execution modes; CBF safety correction is not attributed to the raw actor.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate = collect(args.input_manifest.resolve())
    args.output_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(aggregate), encoding="utf-8")
    print(json.dumps({"locked_gate_passed": aggregate["locked_gate_passed"]}, indent=2))


if __name__ == "__main__":
    main()
