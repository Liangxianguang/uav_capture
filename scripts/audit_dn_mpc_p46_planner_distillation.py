"""Build and audit planner-distillation labels from a P39 route archive.

The analytic DN-MPC planner is the teacher.  A row receives a positive target
only when its candidate is the planner-selected route and that route is
geometry-valid and first-step CBF-feasible.  Ineligible candidates and explicit
planner abstentions remain ``-1`` and are never fabricated into negatives.
This script is archive-only: it executes no action, changes no CBF gate, and
does not train a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter


ROUTE_COUNT = 12
ROUTE_LABELS = (
    "nominal",
    "left_detour",
    "right_detour",
    "upper_detour",
    "lower_detour",
    "radial_out",
    "formation_split",
    "formation_contract",
    "braking",
    "safe_intercept",
    "visibility_hold",
    "verified_safe_hold",
)
REQUIRED_ARRAYS = (
    "route_candidate_index",
    "route_geometry_valid",
    "labels_cbf_feasible",
    "sample_type",
    "scenario_index",
    "time_index",
    "planner_selected_candidate_index",
    "previous_selected_candidate_index",
    "planner_route_switch_outcome",
)
TRACE_ARRAYS = (
    "independent_cbf_trace_present",
    "selected_cbf_verified_feasible",
    "nominal_cbf_verified_feasible",
    "safe_hold_cbf_verified_feasible",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, metadata_path: Path, expected_split: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("split") != expected_split:
        raise ValueError(f"Expected {expected_split} metadata, got {metadata.get('split')!r}")
    if metadata.get("candidate_profile") != "obstacle_route_v1" or int(metadata.get("candidate_count", 0)) != ROUTE_COUNT:
        raise ValueError("P46 requires the obstacle_route_v1 12-candidate contract")
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError("P46 requires a closed development archive")
    planner_contract = metadata.get("planner_selection_identity_contract", {})
    if not isinstance(planner_contract, dict) or planner_contract.get("enabled") is not True:
        raise ValueError("P46 requires the planner selection identity contract")
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_ARRAYS).difference(archive.files))
        if missing:
            raise ValueError(f"{path} is missing {missing}")
        arrays = {name: np.asarray(archive[name]) for name in (*REQUIRED_ARRAYS, *TRACE_ARRAYS) if name in archive.files}
    count = int(arrays["sample_type"].shape[0])
    for name, value in arrays.items():
        if value.shape[0] != count or not np.isfinite(value).all():
            raise ValueError(f"{name} has inconsistent shape or non-finite values")
    if arrays["labels_cbf_feasible"].ndim != 2 or arrays["labels_cbf_feasible"].shape[1] < 1:
        raise ValueError("labels_cbf_feasible must contain a first-step column")
    return arrays, metadata


def _group_audit(arrays: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    sample_type = arrays["sample_type"]
    runtime = sample_type == 0
    candidate = arrays["route_candidate_index"]
    scenario = arrays["scenario_index"]
    time_index = arrays["time_index"]
    geometry = arrays["route_geometry_valid"] >= 0.5
    first_step_cbf = arrays["labels_cbf_feasible"][:, 0] >= 0.5
    planner_selected = arrays["planner_selected_candidate_index"].astype(np.int64, copy=False)
    previous_selected = arrays["previous_selected_candidate_index"].astype(np.int64, copy=False)
    planner_switch = arrays["planner_route_switch_outcome"].astype(np.int64, copy=False)
    labels = {
        "planner_eligible": np.zeros(runtime.shape, dtype=np.int8),
        "planner_teacher_label": np.full(runtime.shape, -1, dtype=np.int8),
        "planner_abstention": np.zeros(runtime.shape, dtype=np.int8),
        "planner_eligible_count": np.full(runtime.shape, -1, dtype=np.int16),
        "planner_teacher_selected": np.full(runtime.shape, -1, dtype=np.int16),
        "planner_selected_trace_present": np.zeros(runtime.shape, dtype=np.int8),
        "planner_selected_cbf_verified_feasible": np.full(runtime.shape, -1, dtype=np.int8),
    }
    groups = 0
    abstentions = 0
    invalid_selection = 0
    inconsistent_selection = 0
    selected_positive_rows = 0
    eligible_rows = 0
    trace_present_groups = 0
    route_counts: Counter[str] = Counter()
    eligible_route_counts: Counter[str] = Counter()
    selected_route_counts: Counter[str] = Counter()
    previous_known_rows = 0
    switch_rows = 0
    keys = sorted({(int(scenario[i]), int(time_index[i])) for i in np.flatnonzero(runtime)})
    for key in keys:
        group = runtime & (scenario == key[0]) & (time_index == key[1])
        groups += 1
        selected_values = planner_selected[group]
        previous_values = previous_selected[group]
        switch_values = planner_switch[group]
        if selected_values.size == 0 or not np.all(selected_values == selected_values[0]):
            inconsistent_selection += 1
            continue
        if not np.all(previous_values == previous_values[0]) or not np.all(switch_values == switch_values[0]):
            inconsistent_selection += 1
            continue
        selected = int(selected_values[0])
        previous = int(previous_values[0])
        switch = int(switch_values[0])
        if previous >= 0:
            previous_known_rows += 1
        if switch > 0:
            switch_rows += 1
        eligible_ids: list[int] = []
        for candidate_id in range(ROUTE_COUNT):
            rows = group & (candidate == candidate_id)
            if not np.any(rows):
                continue
            route_name = ROUTE_LABELS[candidate_id]
            route_counts[route_name] += 1
            is_eligible = bool(np.all(geometry[rows]) and np.all(first_step_cbf[rows]))
            if is_eligible:
                eligible_ids.append(candidate_id)
                eligible_route_counts[route_name] += 1
        eligible_count = len(eligible_ids)
        eligible_rows += int(np.sum(group & np.isin(candidate, eligible_ids)))
        selected_valid = selected in eligible_ids
        if selected >= 0 and not selected_valid:
            invalid_selection += 1
        if selected < 0 and eligible_count > 0:
            invalid_selection += 1
        abstention = int(eligible_count == 0)
        abstentions += abstention
        if selected_valid:
            selected_route_counts[ROUTE_LABELS[selected]] += 1
        trace_present = False
        if "independent_cbf_trace_present" in arrays:
            trace_present = bool(np.any(arrays["independent_cbf_trace_present"][group] > 0))
        if trace_present:
            trace_present_groups += 1
        selected_trace = -1
        if "selected_cbf_verified_feasible" in arrays:
            values = arrays["selected_cbf_verified_feasible"][group]
            selected_trace = int(np.any(values > 0)) if values.size else -1
        row_mask = group
        labels["planner_eligible"][row_mask] = np.isin(candidate[row_mask], eligible_ids).astype(np.int8)
        labels["planner_eligible_count"][row_mask] = eligible_count
        labels["planner_teacher_selected"][row_mask] = selected
        labels["planner_abstention"][row_mask] = abstention
        labels["planner_selected_trace_present"][row_mask] = int(trace_present)
        labels["planner_selected_cbf_verified_feasible"][row_mask] = selected_trace
        if selected_valid:
            teacher = row_mask & np.isin(candidate, eligible_ids)
            labels["planner_teacher_label"][teacher] = (candidate[teacher] == selected).astype(np.int8)
            selected_positive_rows += int(np.sum(candidate[teacher] == selected))
    runtime_rows = int(np.sum(runtime))
    eligible_group_count = groups - abstentions - inconsistent_selection
    report = {
        "runtime_rows": runtime_rows,
        "runtime_groups": groups,
        "eligible_groups": eligible_group_count,
        "abstention_groups": abstentions,
        "inconsistent_identity_groups": inconsistent_selection,
        "invalid_planner_selection_groups": invalid_selection,
        "eligible_rows": eligible_rows,
        "eligible_row_fraction": eligible_rows / runtime_rows if runtime_rows else None,
        "teacher_positive_rows": selected_positive_rows,
        "teacher_labeled_rows": int(np.sum(labels["planner_teacher_label"] >= 0)),
        "teacher_positive_fraction": selected_positive_rows / max(int(np.sum(labels["planner_teacher_label"] >= 0)), 1),
        "independent_trace_present_groups": trace_present_groups,
        "independent_trace_present_fraction": trace_present_groups / groups if groups else None,
        "previous_known_group_fraction": previous_known_rows / groups if groups else None,
        "planner_switch_group_fraction": switch_rows / groups if groups else None,
        "route_counts": dict(sorted(route_counts.items())),
        "eligible_route_counts": dict(sorted(eligible_route_counts.items())),
        "selected_route_counts": dict(sorted(selected_route_counts.items())),
    }
    return report, labels


def _write_labels(output_dir: Path, source_dataset: Path, source_metadata: Path, metadata: dict[str, Any], arrays: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "planner_distillation_labels.npz"
    np.savez_compressed(
        dataset_path,
        sample_type=arrays["sample_type"].astype(np.int8, copy=False),
        scenario_index=arrays["scenario_index"].astype(np.int64, copy=False),
        time_index=arrays["time_index"].astype(np.int64, copy=False),
        route_candidate_index=arrays["route_candidate_index"].astype(np.int64, copy=False),
        planner_selected_candidate_index=arrays["planner_selected_candidate_index"].astype(np.int64, copy=False),
        previous_selected_candidate_index=arrays["previous_selected_candidate_index"].astype(np.int64, copy=False),
        planner_route_switch_outcome=arrays["planner_route_switch_outcome"].astype(np.int8, copy=False),
        **labels,
    )
    output_metadata = {
        "dataset_version": "dn_mpc_planner_distillation_labels_v1",
        "task": "analytic_dn_mpc_teacher_labels_for_offline_route_distillation",
        "split": metadata["split"],
        "development_only": True,
        "locked_test_opened": False,
        "candidate_profile": metadata["candidate_profile"],
        "candidate_count": int(metadata["candidate_count"]),
        "source_dataset": str(source_dataset),
        "source_dataset_sha256": _sha256(source_dataset),
        "source_metadata": str(source_metadata),
        "source_metadata_sha256": _sha256(source_metadata),
        "teacher_source": metadata["planner_selection_identity_contract"],
        "teacher_label_contract": {
            "positive": "candidate equals planner_selected_candidate_index and is geometry/first-step-CBF eligible",
            "negative": "other geometry/first-step-CBF eligible candidate in a non-abstention group",
            "unknown": "ineligible candidate, offline row, or explicit planner abstention",
            "no_future_target_truth_used": True,
        },
        "array_shapes": {name: list(value.shape) for name, value in labels.items()},
    }
    (output_dir / "metadata.json").write_text(json.dumps(output_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "metadata": str(output_dir / "metadata.json"),
        "metadata_sha256": _sha256(output_dir / "metadata.json"),
        "source_split": metadata["split"],
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# DN-MPC P46 Planner-Distillation Label Contract Audit",
        "",
        "**Phase:** development-only, offline-only",
        "",
        "The analytic DN-MPC planner is the teacher. Geometry validity and first-step Joint-CBF feasibility are hard eligibility gates. Ineligible candidates and explicit abstentions remain unknown; no route or action is fabricated.",
        "",
        "| Split | Runtime groups | Eligible groups | Abstentions | Invalid selection groups | Teacher-labeled rows | Positive fraction | Trace groups |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "calibration", "validation"):
        report = result["splits"][split]["audit"]
        lines.append(
            f"| {split} | {report['runtime_groups']} | {report['eligible_groups']} | {report['abstention_groups']} | "
            f"{report['invalid_planner_selection_groups']} | {report['teacher_labeled_rows']} | "
            f"{report['teacher_positive_fraction']:.2%} | {report['independent_trace_present_groups']} |"
        )
    lines += ["", "## Gate decision", ""]
    all_reports = [result["splits"][split]["audit"] for split in ("train", "calibration", "validation")]
    passed = all(item["inconsistent_identity_groups"] == 0 and item["invalid_planner_selection_groups"] == 0 for item in all_reports)
    if passed:
        lines.append("- [x] Planner identity is group-consistent and selected routes are always eligible or explicit abstentions.")
    else:
        lines.append("- [ ] Planner identity/eligibility contract failed; do not train from these labels.")
    lines += [
        "- [x] Unknown labels are preserved for ineligible and abstention rows.",
        "- [x] No action was executed and no CBF/stale/OOD gate changed.",
        "- [ ] This archive does not authorize online JEPA override, Ledger-Lite or a locked benchmark.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "calibration", "validation"):
        parser.add_argument(f"--{split}-dataset", type=Path, required=True)
        parser.add_argument(f"--{split}-metadata", type=Path, required=True)
        parser.add_argument(f"--{split}-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.markdown_output.exists() or (args.tensorboard_logdir.exists() and any(args.tensorboard_logdir.iterdir())):
        raise FileExistsError("Refusing to overwrite P46 audit outputs")
    result: dict[str, Any] = {
        "audit_type": "dn_mpc_p46_planner_distillation_contract",
        "development_only": True,
        "locked_test_opened": False,
        "splits": {},
    }
    with SummaryWriter(log_dir=str(args.tensorboard_logdir.resolve()), flush_secs=1) as writer:
        for split in ("train", "calibration", "validation"):
            source_dataset = getattr(args, f"{split}_dataset").resolve()
            source_metadata = getattr(args, f"{split}_metadata").resolve()
            output_dir = getattr(args, f"{split}_output").resolve()
            arrays, metadata = _load(source_dataset, source_metadata, split)
            audit, labels = _group_audit(arrays)
            artifact = _write_labels(output_dir, source_dataset, source_metadata, metadata, arrays, labels)
            result["splits"][split] = {"audit": audit, "artifact": artifact}
            writer.add_scalar(f"Planner/{split}/runtime_groups", float(audit["runtime_groups"]), 0)
            writer.add_scalar(f"Planner/{split}/abstention_groups", float(audit["abstention_groups"]), 0)
            writer.add_scalar(f"Planner/{split}/teacher_positive_fraction", float(audit["teacher_positive_fraction"]), 0)
            writer.add_scalar(f"Planner/{split}/trace_present_fraction", float(audit["independent_trace_present_fraction"] or 0.0), 0)
            writer.add_text(f"Planner/{split}/route_counts", json.dumps(audit["selected_route_counts"], sort_keys=True), 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
