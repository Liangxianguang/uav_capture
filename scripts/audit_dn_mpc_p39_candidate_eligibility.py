"""Audit P39 route-candidate eligibility without executing actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_report(dataset: Path, metadata_path: Path) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    labels = [str(value) for value in metadata["route_labels"]]
    with np.load(dataset, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "sample_type",
        "route_candidate_index",
        "route_geometry_valid",
        "labels_cbf_feasible",
        "earliest_failure_step",
        "scenario_index",
        "time_index",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"{dataset} is missing eligibility fields: {missing}")
    runtime = arrays["sample_type"] == 0
    if not np.any(runtime):
        raise ValueError(f"{dataset} has no runtime rows")
    family: dict[str, dict[str, Any]] = {}
    for candidate, label in enumerate(labels):
        rows = runtime & (arrays["route_candidate_index"] == candidate)
        geometry = arrays["route_geometry_valid"] >= 0.5
        cbf = arrays["labels_cbf_feasible"][:, 0] >= 0.5
        eligible = rows & geometry & cbf
        group_keys = sorted(
            {
                (int(scenario), int(time))
                for scenario, time in zip(
                    arrays["scenario_index"][rows], arrays["time_index"][rows]
                )
            }
        )
        eligible_groups = 0
        for scenario, time in group_keys:
            group = rows & (arrays["scenario_index"] == scenario) & (arrays["time_index"] == time)
            eligible_groups += int(np.all(geometry[group] & cbf[group]))
        family[label] = {
            "rows": int(np.sum(rows)),
            "geometry_valid_rows": int(np.sum(rows & geometry)),
            "geometry_valid_fraction": float(np.mean(geometry[rows])) if np.any(rows) else None,
            "first_step_cbf_feasible_rows": int(np.sum(rows & cbf)),
            "first_step_cbf_feasible_fraction": float(np.mean(cbf[rows])) if np.any(rows) else None,
            "eligible_rows": int(np.sum(eligible)),
            "eligible_row_fraction": float(np.mean(eligible[rows])) if np.any(rows) else None,
            "groups": len(group_keys),
            "fully_eligible_groups": int(eligible_groups),
            "fully_eligible_group_fraction": eligible_groups / len(group_keys) if group_keys else None,
            "branch_failure_rows_within_horizon": int(
                np.sum(rows & (arrays["earliest_failure_step"] <= 5))
            ),
        }
    return {
        "split": str(metadata["split"]),
        "dataset": str(dataset.resolve()),
        "dataset_sha256": _sha256(dataset),
        "metadata": str(metadata_path.resolve()),
        "metadata_sha256": _sha256(metadata_path),
        "runtime_rows": int(np.sum(runtime)),
        "families": family,
        "lower_detour_source_note": (
            "The route generator's lower branch checks lower_face_blocked for "
            "solid obstacles extending to z=0. Zero eligible lower_detour rows "
            "are therefore treated as a geometry-contract finding, not as a "
            "normal learned negative, until a separate ground-clearance route "
            "is designed."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "validation", "calibration"):
        parser.add_argument(f"--{split}-dataset", type=Path, required=True)
        parser.add_argument(f"--{split}-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    markdown = args.markdown_output.resolve()
    if output.exists() or markdown.exists() or (
        args.tensorboard_logdir.exists() and any(args.tensorboard_logdir.iterdir())
    ):
        raise FileExistsError("refusing to overwrite P39 eligibility outputs")
    reports = {
        split: _split_report(
            getattr(args, f"{split}_dataset").resolve(),
            getattr(args, f"{split}_metadata").resolve(),
        )
        for split in ("train", "validation", "calibration")
    }
    result: dict[str, Any] = {
        "audit_type": "dn_mpc_p39_candidate_eligibility",
        "development_only": True,
        "locked_test_opened": False,
        "action_executed": False,
        "splits": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# DN-MPC P39 Candidate Eligibility Audit",
        "",
        "**Status:** development-only; offline-only; no action executed.",
        "",
        "The audit separates route geometry validity, first-step Joint CBF feasibility, and complete candidate eligibility. It does not relax any safety gate.",
        "",
        "| Split | Route family | Geometry valid | First-step CBF feasible | Eligible rows | Fully eligible groups | Branch failure rows |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    with SummaryWriter(log_dir=str(args.tensorboard_logdir.resolve()), flush_secs=1) as writer:
        writer.add_text("Contract/eligibility", json.dumps({"development_only": True, "locked_test_opened": False, "action_executed": False}, sort_keys=True), 0)
        for split, report in reports.items():
            writer.add_scalar(f"P39/{split}/runtime_rows", report["runtime_rows"], 0)
            for label, values in report["families"].items():
                fmt = lambda value: "n/a" if value is None else f"{value:.2%}"
                lines.append(
                    f"| {split} | `{label}` | {fmt(values['geometry_valid_fraction'])} | "
                    f"{fmt(values['first_step_cbf_feasible_fraction'])} | "
                    f"{fmt(values['eligible_row_fraction'])} | "
                    f"{fmt(values['fully_eligible_group_fraction'])} | "
                    f"{values['branch_failure_rows_within_horizon']} |"
                )
                writer.add_scalar(f"P39/{split}/geometry_valid/{label}", values["geometry_valid_fraction"] or 0.0, 0)
                writer.add_scalar(f"P39/{split}/cbf_feasible/{label}", values["first_step_cbf_feasible_fraction"] or 0.0, 0)
                writer.add_scalar(f"P39/{split}/eligible/{label}", values["eligible_row_fraction"] or 0.0, 0)
                writer.add_scalar(f"P39/{split}/fully_eligible_groups/{label}", values["fully_eligible_group_fraction"] or 0.0, 0)
        writer.add_text("Contract/lower_detour", reports["validation"]["lower_detour_source_note"], 0)
    lines += [
        "",
        reports["validation"]["lower_detour_source_note"],
        "",
        "This audit is a P39 eligibility gate. It does not authorize JEPA retraining, Ledger-Lite, online route override, or locked testing.",
    ]
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
