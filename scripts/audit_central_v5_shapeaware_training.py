"""Create an auditable two-stage training record for a V5 retained-BC run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return document


def _loss_audit(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Training log has no rows: {path}")
    losses = [float(row["action_mse"]) for row in rows]
    return {
        "epochs": len(losses),
        "all_losses_finite": all(math.isfinite(loss) for loss in losses),
        "first_action_mse": losses[0],
        "final_action_mse": losses[-1],
    }


def _effective_imitation(run_dir: Path) -> dict[str, Any]:
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("effective_imitation"), dict):
        raise ValueError(f"Missing effective_imitation in {run_dir / 'config.yaml'}")
    return config["effective_imitation"]


def _accepted_episodes(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("Expert manifest has no episode list")
    accepted = [episode for episode in episodes if episode.get("accepted", True)]
    if not accepted:
        raise ValueError("Expert manifest contains no accepted episodes")
    return accepted


def audit_fixed_stage(
    run_dir: Path,
    minimum_wall_examples_by_initial_distance: dict[float, int] | None = None,
) -> dict[str, Any]:
    manifest = _read_json(run_dir / "expert_dataset_manifest.json")
    settings = _effective_imitation(run_dir)
    accepted = _accepted_episodes(manifest)
    requested = int(settings["episodes"])
    accepted_count = int(manifest["accepted_episodes"])
    rejection_rate = float(manifest["expert_rejection_rate"])
    max_rejection_rate = float(settings.get("expert_max_rejection_rate", 1.0))
    layouts = Counter(str(episode.get("layout", "unknown")) for episode in accepted)
    wall_distances = Counter(
        float(episode["initial_side_distance"])
        for episode in accepted
        if str(episode.get("layout")) == "wall" and episode.get("initial_side_distance") is not None
    )
    required_wall_distances = minimum_wall_examples_by_initial_distance or {}
    wall_coverage_passed = all(
        wall_distances.get(distance, 0) >= minimum
        for distance, minimum in required_wall_distances.items()
    )
    safe = all(bool(episode.get("safe_capture_success")) for episode in accepted)
    cooperative = all(bool(episode.get("cooperative_requirement_met")) for episode in accepted)
    losses = _loss_audit(run_dir / "training.csv")
    return {
        "run_dir": str(run_dir.resolve()),
        "checkpoint_sha256": _sha256(run_dir / "checkpoint.pt"),
        "archive_sha256": _sha256(run_dir / "expert_sequence_dataset.npz"),
        "requested_episodes": requested,
        "accepted_episodes": accepted_count,
        "rejected_episodes": int(manifest["rejected_episodes"]),
        "collection_attempts": int(manifest["collection_attempts"]),
        "rejection_rate": rejection_rate,
        "maximum_rejection_rate": max_rejection_rate,
        "all_accepted_safe": safe,
        "all_accepted_cooperative": cooperative,
        "layout_accepted_episodes": dict(sorted(layouts.items())),
        "wall_initial_side_distance_accepted_episodes": {
            str(distance): count for distance, count in sorted(wall_distances.items())
        },
        "minimum_wall_examples_by_initial_distance": {
            str(distance): minimum for distance, minimum in sorted(required_wall_distances.items())
        },
        "wall_coverage_passed": wall_coverage_passed,
        "sequence_count": int(manifest["sequence_count"]),
        "frame_count": int(manifest["frame_count"]),
        "training": losses,
        "passed": (
            accepted_count == requested
            and len(accepted) == accepted_count
            and rejection_rate <= max_rejection_rate
            and safe
            and cooperative
            and wall_coverage_passed
            and losses["all_losses_finite"]
        ),
    }


def audit_retained_stage(run_dir: Path, fixed: dict[str, Any]) -> dict[str, Any]:
    manifest = _read_json(run_dir / "expert_dataset_manifest.json")
    initialization = _read_json(run_dir / "initialization.json")
    sources = manifest.get("reused_expert_datasets")
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("Retained manifest must contain exactly two reused expert archives")
    source_rows: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("manifest"), dict):
            raise ValueError("Retained source has no embedded manifest")
        nested = source["manifest"]
        accepted = _accepted_episodes(nested)
        source_rows.append(
            {
                "original_sequences": int(source["original_sequences"]),
                "selected_sequences": int(source["selected_sequences"]),
                "accepted_episodes": int(nested["accepted_episodes"]),
                "expert_rejection_rate": float(nested["expert_rejection_rate"]),
                "all_accepted_safe": all(bool(episode.get("safe_capture_success")) for episode in accepted),
                "all_accepted_cooperative": all(
                    bool(episode.get("cooperative_requirement_met")) for episode in accepted
                ),
            }
        )
    source_selection_balanced = len({row["selected_sequences"] for row in source_rows}) == 1
    source_safe_and_cooperative = all(
        row["all_accepted_safe"] and row["all_accepted_cooperative"] for row in source_rows
    )
    initialization_hash = initialization.get("sha256")
    warm_start_matches = isinstance(initialization_hash, str) and initialization_hash == fixed["checkpoint_sha256"]
    if source_rows[0]["original_sequences"] != fixed["sequence_count"]:
        raise ValueError("First retained source is not the supplied fixed-stage archive")
    losses = _loss_audit(run_dir / "training.csv")
    source_hashes = _read_json(run_dir / "source_hashes.json")
    return {
        "run_dir": str(run_dir.resolve()),
        "checkpoint_sha256": _sha256(run_dir / "checkpoint.pt"),
        "archive_sha256": _sha256(run_dir / "expert_sequence_dataset.npz"),
        "initialization_checkpoint": initialization.get("checkpoint"),
        "initialization_sha256": initialization_hash,
        "warm_start_matches_fixed_checkpoint": warm_start_matches,
        "source_balance": manifest.get("source_balance"),
        "source_selection_balanced": source_selection_balanced,
        "source_rows": source_rows,
        "all_sources_safe_and_cooperative": source_safe_and_cooperative,
        "sequence_count": int(manifest["sequence_count"]),
        "frame_count": int(manifest["frame_count"]),
        "training": losses,
        "source_hashes": source_hashes,
        "passed": (
            warm_start_matches
            and manifest.get("source_balance") == "equal_sequences"
            and source_selection_balanced
            and source_safe_and_cooperative
            and losses["all_losses_finite"]
        ),
    }


def _wall_requirements_from_preregistration(path: Path | None) -> dict[float, int]:
    if path is None:
        return {}
    document = _read_json(path)
    try:
        requirements = document["fixed_stage_quality_gate"]["minimum_accepted_wall_examples_per_initial_distance"]
    except (KeyError, TypeError) as error:
        raise ValueError("Pre-registration has no wall initial-distance quality gate") from error
    if not isinstance(requirements, dict) or not requirements:
        raise ValueError("Pre-registration wall initial-distance quality gate must be non-empty")
    parsed = {float(distance): int(minimum) for distance, minimum in requirements.items()}
    if any(minimum <= 0 for minimum in parsed.values()):
        raise ValueError("Pre-registration wall initial-distance minimum must be positive")
    return parsed


def collect(
    fixed_run_dir: Path,
    retained_run_dir: Path,
    preregistration: Path | None = None,
) -> dict[str, Any]:
    requirements = _wall_requirements_from_preregistration(preregistration)
    fixed = audit_fixed_stage(fixed_run_dir, requirements)
    retained = audit_retained_stage(retained_run_dir, fixed)
    return {
        "audit_type": "central_v5_shapeaware_retained_bc_training_audit",
        "not_an_evaluation": True,
        "fixed_stage": fixed,
        "retained_stage": retained,
        "pre_registration": (
            {"path": str(preregistration.resolve()), "sha256": _sha256(preregistration)}
            if preregistration is not None
            else None
        ),
        "candidate_training_integrity_passed": fixed["passed"] and retained["passed"],
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_markdown(audit: dict[str, Any]) -> str:
    fixed = audit["fixed_stage"]
    retained = audit["retained_stage"]
    lines = [
        "# Central V5 Shape-Aware Retained-BC Training Audit",
        "",
        "This record audits training provenance only. It is neither a development evaluation nor a locked test.",
        "",
        "## Fixed Shape-Aware Stage",
        "",
        "| Check | Result |",
        "| --- | --- |",
        f"| Requested / accepted expert episodes | {fixed['requested_episodes']} / {fixed['accepted_episodes']} |",
        f"| Rejected episodes / rejection rate | {fixed['rejected_episodes']} / {_pct(fixed['rejection_rate'])} |",
        f"| Safe / cooperative accepted demonstrations | {fixed['all_accepted_safe']} / {fixed['all_accepted_cooperative']} |",
        f"| Wall initial-distance coverage passes pre-registration | {fixed['wall_coverage_passed']} |",
        f"| Expert sequences / frames | {fixed['sequence_count']} / {fixed['frame_count']} |",
        f"| Training epochs / finite action-MSE | {fixed['training']['epochs']} / {fixed['training']['all_losses_finite']} |",
        f"| First / final action MSE | {fixed['training']['first_action_mse']:.8f} / {fixed['training']['final_action_mse']:.8f} |",
        f"| Checkpoint SHA-256 | `{fixed['checkpoint_sha256']}` |",
        "",
        "Fixed-stage accepted expert coverage:",
        "",
        "| Layout | Accepted expert episodes |",
        "| --- | ---: |",
    ]
    for layout, count in fixed["layout_accepted_episodes"].items():
        lines.append(f"| {layout} | {count} |")
    if fixed["minimum_wall_examples_by_initial_distance"]:
        lines.extend(
            [
                "",
                "Wall initial-distance pre-registration check:",
                "",
                "| Initial distance (m) | Accepted wall episodes | Required minimum |",
                "| ---: | ---: | ---: |",
            ]
        )
        for distance, minimum in fixed["minimum_wall_examples_by_initial_distance"].items():
            lines.append(
                f"| {distance} | {fixed['wall_initial_side_distance_accepted_episodes'].get(distance, 0)} | {minimum} |"
            )
    lines.extend(
        [
            "",
            "## Warm-Start Retained Stage",
            "",
            "| Check | Result |",
            "| --- | --- |",
            f"| Warm-start checkpoint hash matches fixed checkpoint | {retained['warm_start_matches_fixed_checkpoint']} |",
            f"| Archive source balance / selected sequence balance | {retained['source_balance']} / {retained['source_selection_balanced']} |",
            f"| All source demonstrations safe / cooperative | {retained['all_sources_safe_and_cooperative']} |",
        ]
    )
    for index, source in enumerate(retained["source_rows"]):
        lines.append(
            f"| Source {index}: original / selected sequences | {source['original_sequences']} / {source['selected_sequences']} |"
        )
    lines.extend(
        [
            f"| Total sequences / frames | {retained['sequence_count']} / {retained['frame_count']} |",
            f"| Training epochs / finite action-MSE | {retained['training']['epochs']} / {retained['training']['all_losses_finite']} |",
            f"| First / final action MSE | {retained['training']['first_action_mse']:.8f} / {retained['training']['final_action_mse']:.8f} |",
            f"| Retained checkpoint SHA-256 | `{retained['checkpoint_sha256']}` |",
            "",
            "## Source Provenance",
            "",
        ]
    )
    for path, digest in sorted(retained["source_hashes"].items()):
        lines.append(f"- `{path}`: `{digest}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Training integrity passes; a separate fixed/S3 development evaluation is still required."
            if audit["candidate_training_integrity_passed"]
            else "Training integrity fails; do not evaluate or open a locked-test seed block.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-run-dir", type=Path, required=True)
    parser.add_argument("--retained-run-dir", type=Path, required=True)
    parser.add_argument(
        "--pre-registration",
        type=Path,
        help="Optional P3-A pre-registration JSON with a fixed-wall coverage quality gate.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = collect(args.fixed_run_dir, args.retained_run_dir, args.pre_registration)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(audit), encoding="utf-8")


if __name__ == "__main__":
    main()
