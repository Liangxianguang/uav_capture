"""Audit the V4 retained-BC and V5 baseline data/training contracts.

The audit is intentionally read-only.  It compares the effective YAML and,
when available, the generated V5 run artifacts.  It does not inspect locked
test results and it never changes an experiment directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return document


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def _training_contract(document: dict[str, Any]) -> dict[str, Any]:
    settings = document.get("imitation")
    if not isinstance(settings, dict):
        raise ValueError("Training YAML must contain imitation mapping.")
    stages = settings.get("training_showcase_stages", [])
    if not isinstance(stages, list):
        raise ValueError("training_showcase_stages must be a list.")
    return {
        "seed": settings.get("seed"),
        "episodes": settings.get("episodes"),
        "expert_datasets": [str(item) for item in settings.get("expert_datasets", [])],
        "expert_dataset_source_balance": settings.get("expert_dataset_source_balance"),
        "training_obstacle_counts": settings.get("training_obstacle_counts"),
        "training_target_speed_scales": settings.get("training_target_speed_scales"),
        "training_target_motion_modes": settings.get("training_target_motion_modes"),
        "training_required_defender_zone_entries": settings.get("training_required_defender_zone_entries"),
        "action_scale_mode": settings.get("action_scale_mode"),
        "learning_rate": settings.get("learning_rate"),
        "hidden_dim": settings.get("hidden_dim"),
        "epochs": settings.get("epochs"),
        "training_showcase_stages": _canonical(stages),
    }


def _run_artifacts(run_dir: Path | None) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"V5 run directory does not exist: {run_dir}")
    result: dict[str, Any] = {"run_dir": str(run_dir)}
    for name in ("checkpoint.pt", "expert_sequence_dataset.npz", "expert_dataset_manifest.json", "config.yaml"):
        path = run_dir / name
        result[name] = {
            "present": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else None,
        }
    manifest_path = run_dir / "expert_dataset_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("expert_dataset_manifest.json must contain an object.")
        result["manifest"] = {
            key: manifest.get(key)
            for key in (
                "accepted_episodes",
                "rejected_episodes",
                "collection_attempts",
                "expert_rejection_rate",
                "expert_safe_capture_rate",
                "expert_cooperative_requirement_rate",
                "sequence_length",
                "sequence_count",
                "frame_count",
            )
        }
    dataset_path = run_dir / "expert_sequence_dataset.npz"
    if dataset_path.is_file():
        with np.load(dataset_path) as archive:
            result["dataset_shapes"] = {
                name: list(np.asarray(archive[name]).shape)
                for name in ("local_observations", "actions", "reset_masks")
                if name in archive.files
            }
    return result


def _retention_report_evidence(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"present": False, "fixed_archive_declared": False, "equal_sequence_balancing_declared": False}
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"V4 retention report does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    return {
        "present": True,
        "path": str(path),
        "sha256": _sha256(path),
        "fixed_archive_declared": "Fixed V4 expert archive" in text,
        "equal_sequence_balancing_declared": "equal-sequence balancing" in text,
        "checkpoint_sha256": (
            text.split("produces checkpoint SHA-256", 1)[1].split("`", 2)[1]
            if "produces checkpoint SHA-256" in text and "`" in text.split("produces checkpoint SHA-256", 1)[1]
            else None
        ),
    }


def audit(
    v4_config_path: Path,
    v5_config_path: Path,
    environment_config_path: Path,
    run_dir: Path | None = None,
    v4_retention_report: Path | None = None,
) -> dict[str, Any]:
    v4_document = _load_yaml(v4_config_path.resolve())
    v5_document = _load_yaml(v5_config_path.resolve())
    environment = _load_yaml(environment_config_path.resolve())
    v4 = _training_contract(v4_document)
    v5 = _training_contract(v5_document)
    retention_evidence = _retention_report_evidence(v4_retention_report)
    v4_datasets = v4["expert_datasets"]
    v5_datasets = v5["expert_datasets"]
    v5_uses_local_collection = not bool(v5_datasets)
    v4_has_fixed_archive = bool(
        any("shapeaware" in item or "fixed" in item for item in v4_datasets)
        or retention_evidence.get("fixed_archive_declared", False)
    )
    v5_has_fixed_archive = any("shapeaware" in item or "fixed" in item for item in v5_datasets)
    stage_difference = v4["training_showcase_stages"] != v5["training_showcase_stages"]
    findings = {
        "fixed_archive_inherited": bool(v5_has_fixed_archive),
        "v4_fixed_archive_declared": bool(v4_has_fixed_archive),
        "v5_uses_local_collection": v5_uses_local_collection,
        "equal_sequence_balance_preserved": bool(
            v5.get("expert_dataset_source_balance") == "equal_sequences"
            or retention_evidence.get("equal_sequence_balancing_declared", False) and v5_datasets
        ),
        "action_scale_mode_match": v4.get("action_scale_mode") == v5.get("action_scale_mode"),
        "hidden_dim_match": v4.get("hidden_dim") == v5.get("hidden_dim"),
        "zone_entry_requirement_match": (
            v4.get("training_required_defender_zone_entries")
            == v5.get("training_required_defender_zone_entries")
        ),
        "stage_contract_changed": stage_difference,
        "environment_config_sha256": _sha256(environment_config_path.resolve()),
        "v4_retention_report_available": bool(retention_evidence.get("present", False)),
    }
    findings["high_risk_data_contract_gap"] = bool(
        findings["v4_fixed_archive_declared"]
        and not findings["fixed_archive_inherited"]
        and findings["v5_uses_local_collection"]
    )
    return {
        "audit_type": "central_v4_v5_retained_bc_contract",
        "v4_config": str(v4_config_path.resolve()),
        "v5_config": str(v5_config_path.resolve()),
        "environment_config": str(environment_config_path.resolve()),
        "v4_contract": v4,
        "v5_contract": v5,
        "v4_retention_evidence": retention_evidence,
        "findings": findings,
        "run_artifacts": _run_artifacts(run_dir),
        "interpretation": (
            "V5 does not inherit the V4 fixed-scene archive and therefore cannot be treated as a faithful "
            "retained-BC reconstruction. Recover fixed-scene coverage before hard-example training."
            if findings["high_risk_data_contract_gap"]
            else "No high-risk fixed-archive gap was detected by the declared YAML contract."
        ),
    }


def render_markdown(document: dict[str, Any]) -> str:
    findings = document["findings"]
    v4 = document["v4_contract"]
    v5 = document["v5_contract"]
    lines = [
        "# V4/V5 Retained-BC Contract Audit",
        "",
        "This is a read-only contract audit. It does not open or inspect the V5 locked-test block.",
        "",
        "## Data-source comparison",
        "",
        "| Field | V4 retained | V5 baseline |",
        "| --- | --- | --- |",
        f"| Expert data source | `{', '.join(v4['expert_datasets']) or 'local collection'}` | `{', '.join(v5['expert_datasets']) or 'local collection'}` |",
        f"| Source balance | `{v4.get('expert_dataset_source_balance')}` | `{v5.get('expert_dataset_source_balance')}` |",
        f"| Training stages identical | `{not findings['stage_contract_changed']}` | `{not findings['stage_contract_changed']}` |",
        f"| Action scale mode | `{v4.get('action_scale_mode')}` | `{v5.get('action_scale_mode')}` |",
        f"| Required zone entries | `{v4.get('training_required_defender_zone_entries')}` | `{v5.get('training_required_defender_zone_entries')}` |",
        "",
        "## Findings",
        "",
    ]
    for name, value in findings.items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(["", "## Decision", "", f"{document['interpretation']}", ""])
    if document.get("run_artifacts"):
        lines.extend(["## V5 run artifact snapshot", "", "```json", json.dumps(document["run_artifacts"], indent=2), "```", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-config", type=Path, default=PROJECT_ROOT / "configs/capture_radius_recurrent_behavior_cloning_central_v4_s3_retained.yaml")
    parser.add_argument("--v5-config", type=Path, default=PROJECT_ROOT / "configs/capture_radius_recurrent_behavior_cloning_central_v5_baseline.yaml")
    parser.add_argument("--environment-config", type=Path, default=PROJECT_ROOT / "configs/capture_radius_pursuit_central_v4_flee.yaml")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--v4-retention-report", type=Path, default=PROJECT_ROOT / "CENTRAL_V4_S3_RETENTION_VALIDATION_REPORT.md")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = audit(args.v4_config, args.v5_config, args.environment_config, args.run_dir, args.v4_retention_report)
    args.output_json.write_text(json.dumps(document, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(document), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md)}, indent=2))


if __name__ == "__main__":
    main()
