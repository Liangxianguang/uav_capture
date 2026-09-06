"""Aggregate the V21 independent selected/nominal/safe-hold CBF audit.

This is a development-only, read-only audit of frozen V21 reruns.  It does
not re-simulate an episode, change a CBF parameter, consume target future
truth, or execute any counterfactual action.  Every CBF-enabled cycle must
carry three finite, independently probed counterfactuals so an abort can be
attributed to action selection or to the shared safety-feasible set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


SEEDS = (20260911, 20260912, 20260913)
VARIANTS = ("m0", "m3", "a1", "a2")
EXPECTED_EPISODES = 20
NEGATIVE_SLACK_TOLERANCE = 1e-8
COUNTERFACTUAL_LABELS = ("selected", "nominal", "safe_hold")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"Empty trace: {path}")
    return rows


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_action(value: Any, label: str) -> None:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"Missing action array: {label}")
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(f"Malformed action row at {label}[{index}]")
        for component in item:
            if _finite(component) is None:
                raise ValueError(f"Non-finite action component at {label}[{index}]")


def _min_negative_slack(cbf: Mapping[str, Any]) -> tuple[str | None, float | None]:
    values = cbf.get("constraint_slacks")
    if not isinstance(values, Mapping):
        return None, None
    negative = [
        (str(name), number)
        for name, value in values.items()
        if (number := _finite(value)) is not None and number < -NEGATIVE_SLACK_TOLERANCE
    ]
    if not negative:
        return None, None
    return min(negative, key=lambda item: (item[1], item[0]))


def constraint_category(name: str | None) -> str:
    if not name:
        return "none"
    lowered = name.lower()
    if lowered.startswith("obstacle"):
        return "obstacle"
    if lowered.startswith("pairwise"):
        return "pairwise"
    if lowered.startswith("boundary") or lowered.startswith("altitude"):
        return "boundary"
    if lowered.startswith("speed") or lowered.startswith("acceleration"):
        return "dynamic_limit"
    if "target" in lowered:
        return "target"
    return "unknown"


def _probe_map(probes: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(probes, list):
        raise ValueError("independent_cbf_counterfactuals must be a list")
    if len(probes) != len(COUNTERFACTUAL_LABELS):
        raise ValueError(
            "independent_cbf_counterfactuals must contain exactly selected, nominal, safe_hold"
        )
    result: dict[str, Mapping[str, Any]] = {}
    for item in probes:
        if not isinstance(item, Mapping):
            raise ValueError("Each independent CBF counterfactual must be an object")
        label = str(item.get("label", ""))
        if label in result or label not in COUNTERFACTUAL_LABELS:
            raise ValueError(f"Invalid independent CBF counterfactual label set: {label!r}")
        if not isinstance(item.get("requested_action"), list):
            raise ValueError(f"Missing requested_action for independent probe {label!r}")
        _finite_action(item["requested_action"], f"independent_cbf_counterfactuals.{label}")
        result[label] = item
    if set(result) != set(COUNTERFACTUAL_LABELS):
        # Preserve the protocol order in output and reject missing/extra labels.
        missing = sorted(set(COUNTERFACTUAL_LABELS).difference(result))
        extra = sorted(set(result).difference(COUNTERFACTUAL_LABELS))
        raise ValueError(
            "Invalid independent CBF counterfactual label set: "
            f"missing={missing}, extra={extra}"
        )
    return result


def classify_counterfactuals(probes: Mapping[str, Mapping[str, Any]]) -> str:
    """Classify whether an abort is caused by selection or shared infeasibility."""

    accepted = tuple(_bool(probes[label].get("accepted")) for label in COUNTERFACTUAL_LABELS)
    if accepted == (False, False, False):
        return "all_three_infeasible"
    if accepted[0] is False and (accepted[1] or accepted[2]):
        return "selected_only_infeasible"
    if accepted[0] and not accepted[1] and not accepted[2]:
        return "selected_only_feasible"
    if accepted == (True, True, True):
        return "all_three_feasible_abort_inconsistent"
    return "mixed_counterfactual_outcome"


def _candidate_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    ranking = row.get("candidate_ranking")
    if not isinstance(ranking, Mapping):
        return {
            "present": False,
            "valid_count": None,
            "eligible_count": None,
            "selected_index": None,
            "selected_label": None,
            "execution_mode": None,
            "fallback_reason": None,
        }
    labels = ranking.get("candidate_labels")
    selected_index = ranking.get("selected_index")
    selected_label = None
    if isinstance(labels, list) and isinstance(selected_index, int) and 0 <= selected_index < len(labels):
        selected_label = str(labels[selected_index])
    valid = ranking.get("valid_mask")
    eligible = ranking.get("eligible_mask")
    return {
        "present": True,
        "valid_count": sum(_bool(value) for value in valid) if isinstance(valid, list) else None,
        "eligible_count": sum(_bool(value) for value in eligible) if isinstance(eligible, list) else None,
        "selected_index": selected_index,
        "selected_label": selected_label,
        "execution_mode": ranking.get("execution_mode"),
        "fallback_reason": ranking.get("fallback_reason"),
    }


def _compact_probe(probe: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label": str(probe["label"]),
        "route_id": probe.get("route_id"),
        "accepted": _bool(probe.get("accepted")),
        "verified_feasible": _bool(probe.get("verified_feasible")),
        "infeasible": _bool(probe.get("infeasible")),
        "timed_out": _bool(probe.get("timed_out")),
        "fallback_mode": str(probe.get("fallback_mode", "")),
        "solver_status": str(probe.get("solver_status", "")),
        "minimum_constraint_value": _finite(probe.get("minimum_constraint_value")),
        "action_correction_norm": _finite(probe.get("action_correction_norm")),
        "active_constraints": [str(value) for value in probe.get("active_constraints", [])]
        if isinstance(probe.get("active_constraints"), list)
        else [],
    }


def _validate_run(path: Path, seed: int, variant: str) -> dict[str, Any]:
    required = ("summary.json", "provenance.json", "scene_manifest.jsonl", "episodes.csv")
    for name in required:
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    summary = _json(path / "summary.json")
    provenance = _json(path / "provenance.json")
    metadata = summary.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Missing summary metadata: {path}")
    if metadata.get("development_only") is not True or metadata.get("locked_test_opened") is not False:
        raise ValueError(f"Run crossed development boundary: {path}")
    if int(metadata.get("training_seed", -1)) != seed:
        raise ValueError(f"Training seed mismatch: {path}")
    declared_variant = metadata.get("variant")
    if not isinstance(declared_variant, Mapping) or str(declared_variant.get("variant")) != variant:
        raise ValueError(f"Variant mismatch: {path}")
    if provenance.get("development_only") is not True or provenance.get("locked_test_opened") is not False:
        raise ValueError(f"Provenance crossed development boundary: {path}")
    traces = sorted((path / "step_traces").glob("episode_*.jsonl"))
    if len(traces) != EXPECTED_EPISODES:
        raise ValueError(f"Expected {EXPECTED_EPISODES} traces, found {len(traces)}: {path}")
    manifest_hash = sha256(path / "scene_manifest.jsonl")
    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping) or str(inputs.get("scene_manifest_sha256")) != manifest_hash:
        raise ValueError(f"Scene manifest hash mismatch in metadata: {path}")
    return {
        "path": str(path.resolve()),
        "seed": seed,
        "variant": variant,
        "summary": summary,
        "metadata": metadata,
        "manifest_sha256": manifest_hash,
        "summary_sha256": sha256(path / "summary.json"),
        "provenance_sha256": sha256(path / "provenance.json"),
        "traces": traces,
    }


def _audit_trace(
    run: Mapping[str, Any],
    trace_path: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    trace = _jsonl(trace_path)
    episode_index = int(trace[0].get("episode_index", -1))
    if episode_index < 0 or any(int(row.get("episode_index", -1)) != episode_index for row in trace):
        raise ValueError(f"Mixed or invalid episode index: {trace_path}")
    abort: dict[str, Any] | None = None
    probe_counts = Counter()
    for row in trace:
        for action_key in ("desired_action", "reachable_nominal_action", "requested_action", "executed_action"):
            _finite_action(row.get(action_key), f"{trace_path}:{row.get('step')}.{action_key}")
        if _bool(row.get("raw_unverified_executed")):
            raise ValueError(f"raw_unverified action was marked executed: {trace_path}:{row.get('step')}")
        probes = _probe_map(row.get("independent_cbf_counterfactuals"))
        for label, probe in probes.items():
            probe_counts[f"{label}.accepted"] += int(_bool(probe.get("accepted")))
            probe_counts[f"{label}.rejected"] += int(not _bool(probe.get("accepted")))
            probe_counts[f"{label}.timeout"] += int(_bool(probe.get("timed_out")))
        cbf = row.get("cbf")
        if not isinstance(cbf, Mapping):
            raise ValueError(f"Missing CBF diagnostics: {trace_path}:{row.get('step')}")
        if str(cbf.get("fallback_mode", "")) == "controlled_abort":
            if abort is not None:
                raise ValueError(f"Multiple controlled abort rows: {trace_path}")
            negative_name, negative_slack = _min_negative_slack(cbf)
            snapshot = _candidate_snapshot(row)
            abort = {
                "training_seed": int(run["seed"]),
                "variant": str(run["variant"]),
                "episode_index": episode_index,
                "step": int(row.get("step", -1)),
                "source_trace": str(trace_path.resolve()),
                "source_trace_sha256": sha256(trace_path),
                "classification": classify_counterfactuals(probes),
                "selected_accepted": _bool(probes["selected"].get("accepted")),
                "nominal_accepted": _bool(probes["nominal"].get("accepted")),
                "safe_hold_accepted": _bool(probes["safe_hold"].get("accepted")),
                "first_negative_constraint": negative_name,
                "first_negative_slack": negative_slack,
                "first_negative_category": constraint_category(negative_name),
                "cbf_solver_status": str(cbf.get("solver_status", "")),
                "cbf_minimum_constraint_value": _finite(cbf.get("minimum_constraint_value")),
                "cbf_active_constraints": [str(value) for value in cbf.get("active_constraints", [])]
                if isinstance(cbf.get("active_constraints"), list)
                else [],
                "candidate": snapshot,
                "independent_cbf_counterfactuals": [
                    _compact_probe(probes[label]) for label in COUNTERFACTUAL_LABELS
                ],
                "observation": {
                    "target_visible": row.get("observation", {}).get("target_visible")
                    if isinstance(row.get("observation"), Mapping)
                    else None,
                    "target_observation_age_steps": row.get("observation", {}).get("target_observation_age_steps")
                    if isinstance(row.get("observation"), Mapping)
                    else None,
                    "message_age_steps": row.get("observation", {}).get("message_age_steps")
                    if isinstance(row.get("observation"), Mapping)
                    else None,
                },
            }
    return {
        "training_seed": int(run["seed"]),
        "variant": str(run["variant"]),
        "episode_index": episode_index,
        "trace_steps": len(trace),
        "trace_sha256": sha256(trace_path),
        "probe_counts": dict(sorted(probe_counts.items())),
        "has_controlled_abort": abort is not None,
    }, abort


def audit_runs(input_root: Path) -> dict[str, Any]:
    input_root = input_root.resolve()
    runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        for variant in VARIANTS:
            path = input_root / f"jepa_safe_capture_v21_{variant}_seed{seed}"
            runs.append(_validate_run(path, seed, variant))
    manifests_by_seed: dict[int, str] = {}
    protocol_hashes: set[str] = set()
    environment_hashes: set[str] = set()
    cbf_contracts: set[str] = set()
    for run in runs:
        seed = int(run["seed"])
        previous = manifests_by_seed.setdefault(seed, str(run["manifest_sha256"]))
        if previous != str(run["manifest_sha256"]):
            raise ValueError(f"Scene manifest differs across variants for seed {seed}")
        inputs = run["metadata"].get("inputs", {})
        protocol_hashes.add(str(inputs.get("protocol_sha256")))
        environment_hashes.add(str(inputs.get("environment_config_sha256")))
        cbf_contracts.add(json.dumps(run["metadata"].get("cbf_contract", {}), sort_keys=True))
    if len(protocol_hashes) != 1 or len(environment_hashes) != 1 or len(cbf_contracts) != 1:
        raise ValueError("Protocol, environment, or CBF contract differs across the paired matrix")

    episode_rows: list[dict[str, Any]] = []
    abort_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for run in runs:
        episode_count = 0
        abort_count = 0
        probe_counts = Counter()
        for trace_path in run["traces"]:
            episode, abort = _audit_trace(run, trace_path)
            episode_rows.append(episode)
            episode_count += 1
            probe_counts.update(episode["probe_counts"])
            if abort is not None:
                abort_rows.append(abort)
                abort_count += 1
        overall = run["summary"].get("overall", {})
        if int(overall.get("cbf_controlled_abort_steps", -1)) != abort_count:
            raise ValueError(
                f"Summary/trace controlled-abort mismatch for {run['variant']}/{run['seed']}: "
                f"summary={overall.get('cbf_controlled_abort_steps')} trace={abort_count}"
            )
        run_rows.append(
            {
                "training_seed": int(run["seed"]),
                "variant": str(run["variant"]),
                "episodes": episode_count,
                "safe_capture_count": int(overall.get("safe_capture_count", -1)),
                "safe_capture_rate": _finite(overall.get("safe_capture_rate")),
                "cbf_controlled_abort_steps": abort_count,
                "raw_unverified_executed_steps": int(overall.get("raw_unverified_executed_steps", -1)),
                "collision_count": int(overall.get("collision_count", -1)),
                "boundary_violation_count": int(overall.get("boundary_violation_count", -1)),
                "pairwise_violation_count": int(overall.get("pairwise_violation_count", -1)),
                "probe_counts": dict(sorted(probe_counts.items())),
                "manifest_sha256": str(run["manifest_sha256"]),
                "summary_sha256": str(run["summary_sha256"]),
                "provenance_sha256": str(run["provenance_sha256"]),
            }
        )

    classification_counts = Counter(str(row["classification"]) for row in abort_rows)
    negative_categories = Counter(str(row["first_negative_category"]) for row in abort_rows)
    variant_abort_counts = Counter(str(row["variant"]) for row in abort_rows)
    all_three = sum(row["classification"] == "all_three_infeasible" for row in abort_rows)
    safety = {
        "collision_count": sum(row["collision_count"] for row in run_rows),
        "boundary_violation_count": sum(row["boundary_violation_count"] for row in run_rows),
        "pairwise_violation_count": sum(row["pairwise_violation_count"] for row in run_rows),
        "raw_unverified_executed_steps": sum(row["raw_unverified_executed_steps"] for row in run_rows),
    }
    result: dict[str, Any] = {
        "stage": "jepa_safe_capture_v21_independent_cbf_counterfactual_audit",
        "input_format": "v21_independent_probe_rerun",
        "development_only": True,
        "locked_test_opened": False,
        "input_root": str(input_root),
        "run_count": len(runs),
        "episode_count": len(episode_rows),
        "controlled_abort_count": len(abort_rows),
        "run_rows": run_rows,
        "episode_rows": episode_rows,
        "abort_rows": abort_rows,
        "classification_counts": dict(sorted(classification_counts.items())),
        "first_negative_category_counts": dict(sorted(negative_categories.items())),
        "abort_count_by_variant": dict(sorted(variant_abort_counts.items())),
        "all_three_infeasible_count": all_three,
        "all_three_infeasible_rate": all_three / max(len(abort_rows), 1),
        "safety_hard_gate": safety,
        "safety_hard_gate_pass": all(value == 0 for value in safety.values()),
        "scene_manifest_sha256_by_seed": {str(seed): value for seed, value in sorted(manifests_by_seed.items())},
        "protocol_sha256": next(iter(protocol_hashes)),
        "environment_config_sha256": next(iter(environment_hashes)),
        "cbf_contract": json.loads(next(iter(cbf_contracts))),
        "provenance": {
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "no_resimulation": True,
            "no_target_future_truth": True,
            "cbf_parameters_unchanged": True,
        },
    }
    return result


def _write_outputs(result: Mapping[str, Any], output_dir: Path, tensorboard_dir: Path) -> None:
    output_dir = output_dir.resolve()
    tensorboard_dir = tensorboard_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    if tensorboard_dir.exists() and any(tensorboard_dir.iterdir()):
        raise FileExistsError(tensorboard_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "independent_cbf_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fields = [
        "training_seed",
        "variant",
        "episode_index",
        "step",
        "source_trace",
        "source_trace_sha256",
        "classification",
        "selected_accepted",
        "nominal_accepted",
        "safe_hold_accepted",
        "first_negative_constraint",
        "first_negative_slack",
        "first_negative_category",
        "cbf_solver_status",
        "cbf_minimum_constraint_value",
        "candidate_present",
        "candidate_valid_count",
        "candidate_eligible_count",
        "candidate_selected_index",
        "candidate_selected_label",
        "candidate_execution_mode",
        "candidate_fallback_reason",
    ]
    csv_rows: list[dict[str, Any]] = []
    for row in result["abort_rows"]:
        candidate = row["candidate"]
        csv_rows.append(
            {
                "training_seed": row["training_seed"],
                "variant": row["variant"],
                "episode_index": row["episode_index"],
                "step": row["step"],
                "source_trace": row["source_trace"],
                "source_trace_sha256": row["source_trace_sha256"],
                "classification": row["classification"],
                "selected_accepted": row["selected_accepted"],
                "nominal_accepted": row["nominal_accepted"],
                "safe_hold_accepted": row["safe_hold_accepted"],
                "first_negative_constraint": row["first_negative_constraint"],
                "first_negative_slack": row["first_negative_slack"],
                "first_negative_category": row["first_negative_category"],
                "cbf_solver_status": row["cbf_solver_status"],
                "cbf_minimum_constraint_value": row["cbf_minimum_constraint_value"],
                "candidate_present": candidate["present"],
                "candidate_valid_count": candidate["valid_count"],
                "candidate_eligible_count": candidate["eligible_count"],
                "candidate_selected_index": candidate["selected_index"],
                "candidate_selected_label": candidate["selected_label"],
                "candidate_execution_mode": candidate["execution_mode"],
                "candidate_fallback_reason": candidate["fallback_reason"],
            }
        )
    with (output_dir / "independent_cbf_abort.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = [
        "# V21 Independent CBF Counterfactual Audit",
        "",
        "`development_only=true`; `locked_test_opened=false`. This is a read-only audit of the paired reruns.",
        "",
        f"- Runs: `{result['run_count']}`; episodes: `{result['episode_count']}`; controlled aborts: `{result['controlled_abort_count']}`.",
        f"- All three paths infeasible: `{result['all_three_infeasible_count']}/{result['controlled_abort_count']}` ({result['all_three_infeasible_rate']:.3f}).",
        f"- Safety hard gate: `{'PASS' if result['safety_hard_gate_pass'] else 'FAIL'}`.",
        "",
        "## Counterfactual classification",
        "",
        "| Classification | Episodes |",
        "|---|---:|",
    ]
    for label, count in sorted(result["classification_counts"].items()):
        lines.append(f"| `{label}` | {count} |")
    lines += [
        "",
        "## First negative constraint",
        "",
        "| Category | Episodes |",
        "|---|---:|",
    ]
    for label, count in sorted(result["first_negative_category_counts"].items()):
        lines.append(f"| `{label}` | {count} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "All-three-infeasible means the abort state has no accepted selected, reachable-nominal, or safe-hold primary CBF request under the frozen contract. It does not claim that an earlier route choice was optimal or that the CBF should be relaxed.",
        "",
        "This result directs the next fix toward earlier reachable candidate coverage, action-block design, and recovery feasibility. It is not evidence of a JEPA performance improvement and must not open the locked test.",
        "",
        "## Safety boundary",
        "",
        "- CBF margins, stale/OOD gates, and controlled-abort semantics were unchanged.",
        "- No raw-unverified action was executed.",
        "- No target future ground truth was consumed.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with SummaryWriter(log_dir=str(tensorboard_dir), flush_secs=1) as writer:
        writer.add_text(
            "Config/independent_cbf_audit",
            json.dumps(
                {
                    "stage": result["stage"],
                    "input_root": result["input_root"],
                    "development_only": True,
                    "locked_test_opened": False,
                },
                sort_keys=True,
            ),
            0,
        )
        writer.add_text("Provenance/inputs", json.dumps({
            "scene_manifest_sha256_by_seed": result["scene_manifest_sha256_by_seed"],
            "protocol_sha256": result["protocol_sha256"],
            "environment_config_sha256": result["environment_config_sha256"],
            "git_revision": result["provenance"]["git_revision"],
        }, sort_keys=True), 0)
        writer.add_text("Gates/status", json.dumps({
            "safety_hard_gate_pass": result["safety_hard_gate_pass"],
            "all_three_infeasible_rate": result["all_three_infeasible_rate"],
            "no_resimulation": True,
            "no_target_future_truth": True,
        }, sort_keys=True), 0)
        writer.add_scalar("Audit/controlled_abort_count", float(result["controlled_abort_count"]), 0)
        writer.add_scalar("Audit/all_three_infeasible_count", float(result["all_three_infeasible_count"]), 0)
        writer.add_scalar("Audit/all_three_infeasible_rate", float(result["all_three_infeasible_rate"]), 0)
        for label, count in sorted(result["classification_counts"].items()):
            writer.add_scalar(f"Audit/classification/{label}", float(count), 0)
        for label, count in sorted(result["first_negative_category_counts"].items()):
            writer.add_scalar(f"Audit/first_negative/{label}", float(count), 0)
        for row in result["run_rows"]:
            tag = f"{row['variant']}/seed{row['training_seed']}"
            writer.add_scalar(f"SafeCapture/{tag}", float(row["safe_capture_rate"]), 0)
            writer.add_scalar(f"CBF/{tag}/controlled_abort", float(row["cbf_controlled_abort_steps"]), 0)
            writer.add_scalar(f"Safety/{tag}/raw_unverified", float(row["raw_unverified_executed_steps"]), 0)
    accumulator = EventAccumulator(str(tensorboard_dir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required_text = {
        "Config/independent_cbf_audit/text_summary",
        "Provenance/inputs/text_summary",
        "Gates/status/text_summary",
    }
    missing = sorted(required_text.difference(tags.get("tensors", [])))
    events = sorted(path.name for path in tensorboard_dir.glob("events.out.tfevents.*"))
    if missing or not events:
        raise ValueError(f"TensorBoard audit incomplete: missing={missing}, events={events}")
    result_with_tb = dict(result)
    result_with_tb["tensorboard"] = {
        "logdir": str(tensorboard_dir),
        "event_files": events,
        "required_provenance": True,
    }
    (output_dir / "independent_cbf_audit.json").write_text(
        json.dumps(result_with_tb, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "stage": result["stage"],
                "input_format": result["input_format"],
                "development_only": True,
                "locked_test_opened": False,
                "protocol_sha256": result["protocol_sha256"],
                "environment_config_sha256": result["environment_config_sha256"],
                "git_revision": result["provenance"]["git_revision"],
                "tensorboard_logdir": str(tensorboard_dir),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    hash_manifest = {
        str(path.relative_to(output_dir)).replace("\\", "/"): sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.name != "hash_manifest.json" and path.is_file()
    }
    (output_dir / "hash_manifest.json").write_text(
        json.dumps(hash_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    args = parser.parse_args()
    if not args.development_only:
        raise ValueError("Independent CBF audit requires --development-only")
    result = audit_runs(args.input_root)
    _write_outputs(result, args.output_dir, args.tensorboard_dir)
    print(json.dumps({
        "controlled_abort_count": result["controlled_abort_count"],
        "classification_counts": result["classification_counts"],
        "first_negative_category_counts": result["first_negative_category_counts"],
        "all_three_infeasible_rate": result["all_three_infeasible_rate"],
        "safety_hard_gate_pass": result["safety_hard_gate_pass"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
