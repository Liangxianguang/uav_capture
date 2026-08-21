"""Aggregate one V5 retained-BC baseline training and its P1 validation artifacts.

This is deliberately a development-validation report. It validates artifact
identity and reports episode-level uncertainty, but it never treats a single
training seed as a multi-seed or locked-test conclusion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import yaml


FIXED_SCENES = ("s1_cylinder", "s1_box", "s1_wall", "s2")
MODES = ("raw", "cbf")


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain an object.")
    return document


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no rows.")
    return rows


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    """Return a 95% Wilson interval for an episode-level binomial rate."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Wilson interval requires valid binomial counts.")
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt((rate * (1.0 - rate) + z * z / (4.0 * trials)) / trials) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _rate(rows: list[dict[str, str]], field: str) -> tuple[float, int]:
    successes = sum(_bool(row[field]) for row in rows)
    return successes / len(rows), successes


def _summarize_episode_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    capture_times = [value for row in rows if (value := _float(row.get("capture_time_seconds"))) is not None]
    corrections = [float(row.get("mean_cbf_action_correction_norm", 0.0)) for row in rows]
    path_lengths = [float(row.get("mean_defender_path_length_m", 0.0)) for row in rows]
    clearances = [float(row["min_clearance_m"]) for row in rows]
    cooperative_rate, cooperative_successes = _rate(rows, "cooperative_safe_capture")
    capture_rate, capture_successes = _rate(rows, "capture_event")
    safe_rate, safe_successes = _rate(rows, "safe_capture_success")
    collision_rate, collision_successes = _rate(rows, "collision")
    boundary_successes = sum(float(row["world_violation_steps"]) > 0.0 for row in rows)
    transit_rate, transit_successes = _rate(rows, "transit_success")
    return {
        "episodes": len(rows),
        "cooperative_safe_capture_rate": cooperative_rate,
        "cooperative_safe_capture_successes": cooperative_successes,
        "cooperative_safe_capture_wilson_95": wilson_interval(cooperative_successes, len(rows)),
        "safe_capture_rate": safe_rate,
        "safe_capture_successes": safe_successes,
        "capture_rate": capture_rate,
        "capture_successes": capture_successes,
        "collision_rate": collision_rate,
        "collision_episodes": collision_successes,
        "boundary_violation_rate": boundary_successes / len(rows),
        "boundary_violation_episodes": boundary_successes,
        "transit_success_rate": transit_rate,
        "transit_successes": transit_successes,
        "timeout_episodes": sum(row.get("termination_reason") == "timeout" for row in rows),
        "mean_capture_time_seconds": statistics.fmean(capture_times) if capture_times else None,
        "mean_min_clearance_m": statistics.fmean(clearances),
        "worst_min_clearance_m": min(clearances),
        "mean_defender_path_length_m": statistics.fmean(path_lengths),
        "cbf_correction": {
            "mean": statistics.fmean(corrections),
            "median": statistics.median(corrections),
            "p95": float(sorted(corrections)[math.ceil(0.95 * len(corrections)) - 1]),
            "max": max(float(row.get("max_cbf_action_correction_norm", 0.0)) for row in rows),
        },
        "termination_reasons": {
            reason: sum(row.get("termination_reason") == reason for row in rows)
            for reason in sorted({str(row.get("termination_reason")) for row in rows})
        },
    }


def _artifact_metrics(directory: Path, *, mode: str, s3: bool) -> dict[str, Any]:
    rows = _read_rows(directory / "episodes.csv")
    expected_episodes = 60 if s3 else 20
    if len(rows) != expected_episodes:
        raise ValueError(f"{directory} has {len(rows)} episodes, expected {expected_episodes}.")
    metadata = _read_json(directory / ("evaluation_metadata.json" if s3 else "summary.json"))
    if bool(metadata.get("use_cbf")) != (mode == "cbf"):
        raise ValueError(f"{directory} does not match requested execution mode {mode}.")
    if s3:
        if metadata.get("split") != "validation" or bool(metadata.get("locked_test")):
            raise ValueError(f"{directory} is not an S3 development-validation artifact.")
        if int(metadata.get("seed_block", -1)) != 646101:
            raise ValueError(f"{directory} does not use the frozen V5 validation seed block.")
    return {
        "directory": str(directory),
        "episodes_csv_sha256": _sha256(directory / "episodes.csv"),
        "metrics": _summarize_episode_rows(rows),
    }


def _static_scene_digest(path: Path) -> tuple[str, int]:
    """Hash only scenario inputs, excluding mode-dependent rollout outcomes."""
    records: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path} contains a non-object scene record.")
        static = {key: record[key] for key in ("episode_index", "spec", "scenario")}
        records.append(json.dumps(static, sort_keys=True, separators=(",", ":")))
    if len(records) != 60:
        raise ValueError(f"{path} has {len(records)} scenes, expected 60.")
    digest = hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()
    return digest, len(records)


def _validate_s3_scene_pairing(raw_directory: Path, cbf_directory: Path) -> dict[str, Any]:
    raw_digest, raw_count = _static_scene_digest(raw_directory / "scenes.jsonl")
    cbf_digest, cbf_count = _static_scene_digest(cbf_directory / "scenes.jsonl")
    paired = raw_digest == cbf_digest and raw_count == cbf_count
    if not paired:
        raise ValueError("V5 S3 raw and CBF artifacts do not use identical static scenes.")
    return {
        "static_scenes_exactly_paired": True,
        "episodes": raw_count,
        "raw_static_scene_sha256": raw_digest,
        "cbf_static_scene_sha256": cbf_digest,
    }


def _training_quality(run_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "expert_dataset_manifest.json")
    effective = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(effective, dict) or not isinstance(effective.get("effective_imitation"), dict):
        raise ValueError("Training config artifact has no effective_imitation mapping.")
    settings = effective["effective_imitation"]
    common = {
        "run_dir": str(run_dir),
        "checkpoint_sha256": _sha256(run_dir / "checkpoint.pt"),
        "expert_dataset_sha256": _sha256(run_dir / "expert_sequence_dataset.npz"),
        "training_epochs": len(_read_rows(run_dir / "training.csv")),
        "all_losses_finite": all(math.isfinite(float(row["action_mse"])) for row in _read_rows(run_dir / "training.csv")),
    }
    if "accepted_episodes" in manifest:
        requested = int(settings["episodes"])
        accepted = int(manifest["accepted_episodes"])
        rejection_rate = float(manifest["expert_rejection_rate"])
        allowed_rejection_rate = float(settings.get("expert_max_rejection_rate", 1.0))
        episodes = manifest.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError("Expert manifest episodes must be a list.")
        all_safe = all(bool(row.get("safe_capture_success")) for row in episodes)
        all_cooperative = all(bool(row.get("cooperative_requirement_met")) for row in episodes)
        return {
            **common,
            "data_provenance": "locally_collected_expert_episodes",
            "requested_episodes": requested,
            "accepted_episodes": accepted,
            "rejected_episodes": int(manifest["rejected_episodes"]),
            "collection_attempts": int(manifest["collection_attempts"]),
            "rejection_rate": rejection_rate,
            "maximum_rejection_rate": allowed_rejection_rate,
            "expert_safe_capture_rate": float(manifest["expert_safe_capture_rate"]),
            "expert_cooperative_requirement_rate": float(manifest["expert_cooperative_requirement_rate"]),
            "all_accepted_safe": all_safe,
            "all_accepted_cooperative": all_cooperative,
            "passed": (
            accepted == requested
            and rejection_rate <= allowed_rejection_rate
            and all_safe
            and all_cooperative
            and common["all_losses_finite"]
            ),
        }
    sources = manifest.get("reused_expert_datasets")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Expert manifest must describe local episodes or non-empty reused_expert_datasets.")
    source_rows: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError("Each reused expert dataset source must be a mapping.")
        source_manifest = source.get("manifest")
        if not isinstance(source_manifest, dict):
            raise ValueError("Each reused expert dataset source must include its manifest.")
        episodes = source_manifest.get("episodes")
        all_safe = bool(isinstance(episodes, list) and episodes) and all(
            bool(row.get("safe_capture_success")) for row in episodes
        )
        all_cooperative = bool(isinstance(episodes, list) and episodes) and all(
            bool(row.get("cooperative_requirement_met")) for row in episodes
        )
        source_rows.append(
            {
                "source_index": index,
                "original_sequences": int(source["original_sequences"]),
                "selected_sequences": int(source["selected_sequences"]),
                "accepted_episodes": source_manifest.get("accepted_episodes"),
                "expert_rejection_rate": source_manifest.get("expert_rejection_rate"),
                "all_accepted_safe": all_safe,
                "all_accepted_cooperative": all_cooperative,
            }
        )
    selected = [row["selected_sequences"] for row in source_rows]
    balanced = str(manifest.get("source_balance")) != "equal_sequences" or len(set(selected)) == 1
    sources_safe = all(row["all_accepted_safe"] and row["all_accepted_cooperative"] for row in source_rows)
    return {
        **common,
        "data_provenance": "reused_expert_archives",
        "source_balance": manifest.get("source_balance"),
        "source_rows": source_rows,
        "all_sources_safe_and_cooperative": sources_safe,
        "source_selection_balanced": balanced,
        "passed": bool(sources_safe and balanced and common["all_losses_finite"]),
    }


def collect(run_dir: Path, evaluation_root: Path, run_id: str) -> dict[str, Any]:
    fixed: dict[str, dict[str, Any]] = {}
    for scene in FIXED_SCENES:
        fixed[scene] = {
            mode: _artifact_metrics(evaluation_root / f"{run_id}_{scene}_{mode}_20", mode=mode, s3=False)
            for mode in MODES
        }
    s3 = {
        mode: _artifact_metrics(evaluation_root / f"{run_id}_s3_validation_{mode}_60", mode=mode, s3=True)
        for mode in MODES
    }
    failure_indices = {
        mode: _read_json(evaluation_root / f"{run_id}_s3_validation_{mode}_60" / "failure_index.json")
        for mode in MODES
    }
    scene_pairing = _validate_s3_scene_pairing(
        evaluation_root / f"{run_id}_s3_validation_raw_60",
        evaluation_root / f"{run_id}_s3_validation_cbf_60",
    )
    quality = _training_quality(run_dir)
    cbf = s3["cbf"]["metrics"]
    fixed_cbf_rates = [fixed[scene]["cbf"]["metrics"]["cooperative_safe_capture_rate"] for scene in FIXED_SCENES]
    gates = {
        "s3_cooperative_safe_capture_at_least_85_percent": cbf["cooperative_safe_capture_rate"] >= 0.85,
        "s3_collision_at_most_2_percent": cbf["collision_rate"] <= 0.02,
        "s3_boundary_at_most_2_percent": cbf["boundary_violation_rate"] <= 0.02,
        "s3_transit_at_least_99_percent": cbf["transit_success_rate"] >= 0.99,
        "all_fixed_cbf_at_least_98_percent": min(fixed_cbf_rates) >= 0.98,
    }
    return {
        "evaluation_type": "central_v5_retained_bc_development_validation",
        "not_a_locked_test": True,
        "statistical_unit": "episodes within one development training seed; not independent training seeds",
        "training": quality,
        "fixed_regression": fixed,
        "s3_validation": s3,
        "failure_indices": failure_indices,
        "s3_scene_pairing": scene_pairing,
        "candidate_gates": gates,
        "candidate_gate_passed": all(gates.values()) and quality["passed"],
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def _s(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} s"


def render_markdown(aggregate: dict[str, Any]) -> str:
    training = aggregate["training"]
    lines = [
        "# Central V5 Retained-BC Development Validation",
        "",
        "This is a one-training-seed development-validation report. It is not a locked test and is not evidence of a multi-seed improvement.",
        "",
        "## Training Integrity",
        "",
        f"- Checkpoint SHA-256: `{training['checkpoint_sha256']}`",
        f"- Expert archive SHA-256: `{training['expert_dataset_sha256']}`",
        f"- Data provenance: `{training['data_provenance']}`",
    ]
    if training["data_provenance"] == "locally_collected_expert_episodes":
        lines.extend(
            [
                f"- Accepted expert episodes: `{training['accepted_episodes']}/{training['requested_episodes']}`",
                f"- Expert rejection rate: `{100.0 * training['rejection_rate']:.2f}%` (limit `{100.0 * training['maximum_rejection_rate']:.2f}%`)",
                f"- All accepted demonstrations safe/cooperative: `{training['all_accepted_safe']}/{training['all_accepted_cooperative']}`",
            ]
        )
    else:
        lines.extend(
            [
                f"- Archive source balance: `{training['source_balance']}`; selected sequences balanced: `{training['source_selection_balanced']}`",
                f"- All source demonstrations safe/cooperative: `{training['all_sources_safe_and_cooperative']}`",
            ]
        )
        for source in training["source_rows"]:
            lines.append(
                f"- Source {source['source_index']}: `{source['selected_sequences']}/{source['original_sequences']}` sequences selected; "
                f"safe/cooperative `{source['all_accepted_safe']}/{source['all_accepted_cooperative']}`."
            )
    lines.extend(
        [
            f"- Training epochs with finite imitation loss: `{training['training_epochs']}` / `{training['all_losses_finite']}`",
            "",
            "## Fixed S1/S2 Regression",
        "",
        "| Scene | Execution | Cooperative Safe Capture | Collision | Boundary | Transit |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for scene, executions in aggregate["fixed_regression"].items():
        for mode, artifact in executions.items():
            metrics = artifact["metrics"]
            lines.append(
                f"| {scene} | {mode.upper()} | {_pct(metrics['cooperative_safe_capture_rate'])} | "
                f"{_pct(metrics['collision_rate'])} | {_pct(metrics['boundary_violation_rate'])} | "
                f"{_pct(metrics['transit_success_rate'])} |"
            )

    lines.extend(
        [
            "",
            "## V5 Random S3 Validation",
            "",
            "| Execution | Cooperative Safe Capture (95% Wilson CI) | Capture | Collision | Boundary | Transit | Time to Capture | Path / Defender | CBF Correction (mean / median / p95) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode, artifact in aggregate["s3_validation"].items():
        metrics = artifact["metrics"]
        low, high = metrics["cooperative_safe_capture_wilson_95"]
        correction = metrics["cbf_correction"]
        lines.append(
            f"| {mode.upper()} | {_pct(metrics['cooperative_safe_capture_rate'])} "
            f"({_pct(low)}, {_pct(high)}) | {_pct(metrics['capture_rate'])} | "
            f"{_pct(metrics['collision_rate'])} | {_pct(metrics['boundary_violation_rate'])} | "
            f"{_pct(metrics['transit_success_rate'])} | {_s(metrics['mean_capture_time_seconds'])} | "
            f"{metrics['mean_defender_path_length_m']:.2f} m | "
            f"{correction['mean']:.3f} / {correction['median']:.3f} / {correction['p95']:.3f} |"
        )

    pairing = aggregate["s3_scene_pairing"]
    lines.extend(
        [
            "",
            "## S3 Raw/CBF Pairing",
            "",
            f"- Static maps, initial positions, target profile, and episode seeds exactly paired: `{pairing['static_scenes_exactly_paired']}`",
            f"- Static-scene SHA-256: `{pairing['raw_static_scene_sha256']}`",
        ]
    )

    labels = {
        "observation_condition": "Observation condition",
        "obstacle_count": "Obstacle count",
        "planned_route_clearance_band": "Planned clearance proxy",
        "target_motion_mode": "Target motion",
    }
    lines.extend(["", "## CBF Failure Groups", ""])
    groups = aggregate["failure_indices"]["cbf"].get("groups", {})
    for field, label in labels.items():
        lines.extend([f"### {label}", "", "| Group | Episodes | Cooperative failure rate | Failure stages |", "| --- | ---: | ---: | --- |"])
        for group, values in groups.get(field, {}).items():
            stages = ", ".join(f"{stage}: {count}" for stage, count in values["failure_stages"].items()) or "none"
            lines.append(
                f"| {group} | {values['episodes']} | {100.0 * values['cooperative_failure_rate']:.1f}% | {stages} |"
            )
        lines.append("")

    lines.extend(["## Gate Decision", ""])
    for name, passed in aggregate["candidate_gates"].items():
        lines.append(f"- `{name}`: `{passed}`")
    lines.extend(
        [
            f"- Overall one-seed development candidate gate: `{aggregate['candidate_gate_passed']}`",
            "",
            "Raw actor and CBF execution are separate artifacts. CBF safety improvement is not attributed entirely to the learned actor. A passing development gate only permits the pre-registered next step; it does not open or replace the V5 next-locked seed block.",
            "",
        ]
    )
    return "\n".join(lines)


def render_policy_failure_report(aggregate: dict[str, Any]) -> str:
    """Render the short decision record that selects the next P2 branch."""
    raw = aggregate["s3_validation"]["raw"]["metrics"]
    cbf = aggregate["s3_validation"]["cbf"]["metrics"]
    raw_index = aggregate["failure_indices"]["raw"]["summary"]
    cbf_index = aggregate["failure_indices"]["cbf"]["summary"]
    fixed = aggregate["fixed_regression"]
    fixed_cbf = {
        scene: artifact["cbf"]["metrics"]["cooperative_safe_capture_rate"] for scene, artifact in fixed.items()
    }
    lines = [
        "# V5 Policy Failure Analysis",
        "",
        "This is a development-validation diagnostic for one training seed. It does not open the V5 next locked-test block.",
        "",
        "## Paired S3 result",
        "",
        f"- Raw/CBF static scenes exactly paired: `{aggregate['s3_scene_pairing']['static_scenes_exactly_paired']}`",
        f"- Raw Cooperative Safe Capture: `{raw['cooperative_safe_capture_successes']}/{raw['episodes']}`; collision `{100.0 * raw['collision_rate']:.1f}%`.",
        f"- CBF Cooperative Safe Capture: `{cbf['cooperative_safe_capture_successes']}/{cbf['episodes']}`; collision `{100.0 * cbf['collision_rate']:.1f}%`; boundary `{100.0 * cbf['boundary_violation_rate']:.1f}%`.",
        f"- Raw failure stages: `{raw_index['failure_stages']}`.",
        f"- CBF failure stages: `{cbf_index['failure_stages']}`.",
        "",
        "## Fixed-scene regression",
        "",
    ]
    for scene, rate in fixed_cbf.items():
        lines.append(f"- `{scene}` CBF Cooperative Safe Capture: `{100.0 * rate:.1f}%`.")
    if aggregate["candidate_gate_passed"]:
        decision = (
            "The shape-aware warm-start retained-BC checkpoint passes the one-seed "
            "development gate. Freeze the effective configuration, expert-archive "
            "provenance, checkpoint-selection rule, and CBF parameters, then train "
            "two additional independent seeds. Do not open seed block 647201 until "
            "all three checkpoints pass the same development gate."
        )
    else:
        decision = (
            "The raw actor fails before task-level pursuit in every S3 episode, while "
            "CBF removes collisions but leaves distributed timeouts. Together with the "
            "V4/V5 contract audit, this rejects the fresh V5 baseline as a candidate "
            "and selects P2-0 fixed-contract recovery: equal-sequence training on a "
            "newly collected fixed S1/S2 archive plus the frozen V5 random archive. "
            "Do not start MAPPO, change CBF margins, or open seed block 647201 before "
            "this data-only recovery passes fixed regression."
        )
    lines.extend(["", "## Decision", "", decision, ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-failure-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate = collect(args.run_dir.resolve(), args.evaluation_root.resolve(), args.run_id)
    args.output_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(aggregate), encoding="utf-8")
    if args.output_failure_md is not None:
        args.output_failure_md.write_text(render_policy_failure_report(aggregate), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md)}, indent=2))


if __name__ == "__main__":
    main()
