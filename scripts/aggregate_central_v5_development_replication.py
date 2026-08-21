"""Audit three V5 development-validation summaries before any locked evaluation.

This tool treats training seeds, rather than episodes, as the replication unit.
It never reads a locked evaluation artifact and refuses summaries that claim to
be locked tests.  A non-passing development seed is evidence for a rejection
report, not a reason to rerun the same evaluation until it passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


FIXED_SCENES = ("s1_cylinder", "s1_box", "s1_wall", "s2")
REQUIRED_GATES = (
    "s3_cooperative_safe_capture_at_least_85_percent",
    "s3_collision_at_most_2_percent",
    "s3_boundary_at_most_2_percent",
    "s3_transit_at_least_99_percent",
    "all_fixed_cbf_at_least_98_percent",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON summary: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Summary must be a JSON object: {path}")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a JSON boolean")
    return value


def _rate(summary: dict[str, Any], *keys: str) -> float:
    value: Any = summary
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Missing summary field: {'.'.join(keys)}")
        value = value[key]
    if not isinstance(value, (float, int)):
        raise ValueError(f"Summary rate must be numeric: {'.'.join(keys)}")
    return float(value)


def _training_seed(run_dir: str) -> int:
    match = re.search(r"seed(\d+)", Path(run_dir).name)
    if match is None:
        raise ValueError(f"Could not infer retained training seed from run directory: {run_dir}")
    return int(match.group(1))


def summarize_input(path: Path) -> dict[str, Any]:
    summary = _read_json(path)
    if summary.get("evaluation_type") != "central_v5_retained_bc_development_validation":
        raise ValueError(f"Unexpected evaluation_type in {path}")
    if _require_bool(summary.get("not_a_locked_test"), f"not_a_locked_test in {path}") is not True:
        raise ValueError(f"Development replication input must not be a locked test: {path}")

    training = summary.get("training")
    if not isinstance(training, dict):
        raise ValueError(f"Missing training record in {path}")
    run_dir = training.get("run_dir")
    checkpoint_sha256 = training.get("checkpoint_sha256")
    if not isinstance(run_dir, str) or not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise ValueError(f"Training provenance is incomplete in {path}")

    gates = summary.get("candidate_gates")
    if not isinstance(gates, dict):
        raise ValueError(f"Missing candidate gates in {path}")
    gate_values = {name: _require_bool(gates.get(name), f"{name} in {path}") for name in REQUIRED_GATES}
    candidate_passed = _require_bool(summary.get("candidate_gate_passed"), f"candidate_gate_passed in {path}")
    if candidate_passed != all(gate_values.values()) and training.get("passed") is True:
        raise ValueError(f"Candidate gate is inconsistent with component gates in {path}")

    fixed_rates = {
        scene: _rate(summary, "fixed_regression", scene, "cbf", "metrics", "cooperative_safe_capture_rate")
        for scene in FIXED_SCENES
    }
    s3_metrics = summary.get("s3_validation", {}).get("cbf", {}).get("metrics")
    if not isinstance(s3_metrics, dict):
        raise ValueError(f"Missing S3 CBF metrics in {path}")
    if s3_metrics.get("episodes") != 60:
        raise ValueError(f"Expected 60 S3 development episodes in {path}")
    scene_pairing = summary.get("s3_scene_pairing")
    if not isinstance(scene_pairing, dict):
        raise ValueError(f"Missing S3 scene pairing in {path}")
    paired = _require_bool(scene_pairing.get("static_scenes_exactly_paired"), f"scene pairing in {path}")

    failures = [name for name, passed in gate_values.items() if not passed]
    if training.get("passed") is not True:
        failures.append("training_integrity")
    if not paired:
        failures.append("raw_cbf_scene_pairing")
    return {
        "training_seed": _training_seed(run_dir),
        "run_dir": run_dir,
        "checkpoint_sha256": checkpoint_sha256,
        "summary_path": str(path.resolve()),
        "summary_sha256": _sha256(path),
        "candidate_gate_passed": candidate_passed,
        "failed_gates": failures,
        "fixed_cbf_cooperative_safe_capture_rate": fixed_rates,
        "s3_cbf": {
            "episodes": s3_metrics["episodes"],
            "cooperative_safe_capture_rate": _rate(
                summary, "s3_validation", "cbf", "metrics", "cooperative_safe_capture_rate"
            ),
            "collision_rate": _rate(summary, "s3_validation", "cbf", "metrics", "collision_rate"),
            "boundary_violation_rate": _rate(
                summary, "s3_validation", "cbf", "metrics", "boundary_violation_rate"
            ),
            "transit_success_rate": _rate(
                summary, "s3_validation", "cbf", "metrics", "transit_success_rate"
            ),
        },
        "raw_cbf_static_scenes_exactly_paired": paired,
        "static_scene_sha256": scene_pairing.get("raw_static_scene_sha256"),
    }


def collect(summary_paths: list[Path]) -> dict[str, Any]:
    if len(summary_paths) != 3:
        raise ValueError("V5 development replication audit requires exactly three summaries")
    models = [summarize_input(path) for path in summary_paths]
    seeds = [model["training_seed"] for model in models]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Training seeds must be distinct: {seeds}")
    checkpoint_hashes = [model["checkpoint_sha256"] for model in models]
    if len(set(checkpoint_hashes)) != len(checkpoint_hashes):
        raise ValueError("Development replication inputs must use distinct checkpoints")

    models.sort(key=lambda model: model["training_seed"])
    passed = [model for model in models if model["candidate_gate_passed"]]
    decision = (
        "all_three_passed_open_locked_test"
        if len(passed) == 3
        else "replication_rejected_do_not_open_locked_test"
    )
    return {
        "evaluation_type": "central_v5_three_seed_development_replication_audit",
        "not_a_locked_test": True,
        "statistical_unit": "independent retained-BC training seeds",
        "required_training_seed_count": 3,
        "development_checkpoint_count": len(models),
        "development_candidate_pass_count": len(passed),
        "development_candidate_fail_count": len(models) - len(passed),
        "locked_seed_block": 647201,
        "locked_test_opened": len(passed) == 3,
        "decision": decision,
        "models": models,
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def render_markdown(aggregate: dict[str, Any]) -> str:
    lines = [
        "# Central V5 Three-Seed Development Replication Audit",
        "",
        "This report aggregates independent training seeds, not episodes. It uses only the V5 development block and does not open the locked block.",
        "",
        "## Candidate Results",
        "",
        "| Retained seed | Fixed CBF (cylinder / box / wall / S2) | S3 CBF | Collision | Boundary | Transit | Paired raw/CBF scenes | Gate |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for model in aggregate["models"]:
        fixed = model["fixed_cbf_cooperative_safe_capture_rate"]
        s3 = model["s3_cbf"]
        lines.append(
            f"| {model['training_seed']} | {_pct(fixed['s1_cylinder'])} / {_pct(fixed['s1_box'])} / "
            f"{_pct(fixed['s1_wall'])} / {_pct(fixed['s2'])} | "
            f"{_pct(s3['cooperative_safe_capture_rate'])} ({s3['episodes']}) | "
            f"{_pct(s3['collision_rate'])} | {_pct(s3['boundary_violation_rate'])} | "
            f"{_pct(s3['transit_success_rate'])} | {model['raw_cbf_static_scenes_exactly_paired']} | "
            f"{model['candidate_gate_passed']} |"
        )
    lines.extend(["", "## Provenance", ""])
    for model in aggregate["models"]:
        lines.append(
            f"- Seed `{model['training_seed']}`: checkpoint `{model['checkpoint_sha256']}`, "
            f"summary SHA-256 `{model['summary_sha256']}`."
        )
    lines.extend(["", "## Decision", ""])
    if aggregate["locked_test_opened"]:
        lines.append(
            "All three independent development checkpoints pass. The frozen V5 locked block 647201 may now be opened exactly once for the pre-registered locked evaluation."
        )
    else:
        failed = [model for model in aggregate["models"] if not model["candidate_gate_passed"]]
        details = "; ".join(
            f"seed {model['training_seed']}: {', '.join(model['failed_gates'])}" for model in failed
        )
        lines.append(
            "The three-seed replication requirement is not met; locked block 647201 remains unopened. "
            f"Failed development evidence: {details}."
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-summary", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate = collect(args.input_summary)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(aggregate), encoding="utf-8")


if __name__ == "__main__":
    main()
