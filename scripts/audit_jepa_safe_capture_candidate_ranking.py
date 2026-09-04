"""Audit candidate reachability and ranking evidence from frozen V3 traces.

This is a development-only offline audit.  It validates the fixed five
candidate contract and the ranking/ledger execution invariants in existing
step traces.  It deliberately distinguishes descriptive selected-trajectory
outcomes from unavailable counterfactual per-candidate settled labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from aggregate_jepa_safe_capture_v2_paired import sha256  # noqa: E402
from replay_jepa_safe_capture_failures import (  # noqa: E402
    _validate_episode_source,
    load_failure_index,
    validate_source_runs,
)
from index_jepa_safe_capture_failures import read_trace  # noqa: E402


AUDIT_TYPE = "jepa_safe_capture_v3_wp4_candidate_ranking_audit"
CANDIDATE_LABELS = ("nominal", "intercept", "lateral_clearance", "formation_clearance", "visibility_hold")
SCORE_FIELDS = (
    "scores",
    "target_cost_m",
    "uncertainty_cost_m",
    "clearance_cost_m",
    "ttc_cost",
    "visibility_cost",
    "cbf_risk_cost",
    "action_change_cost_mps",
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid integer {label}: {value!r}") from error


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Non-numeric {label}: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {label}: {value!r}")
    return number


def _finite_list(value: Any, label: str, *, allow_nonfinite: bool = False) -> list[float]:
    if not isinstance(value, list) or len(value) != len(CANDIDATE_LABELS):
        raise ValueError(f"{label} must contain exactly five candidate values")
    result: list[float] = []
    for index, item in enumerate(value):
        try:
            number = float(item)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Non-numeric {label}[{index}]: {item!r}") from error
        if not allow_nonfinite and not math.isfinite(number):
            raise ValueError(f"Non-finite {label}[{index}]: {item!r}")
        result.append(number)
    return result


def _validate_ranking(ranking: Mapping[str, Any], *, identifier: str, step: int) -> dict[str, Any]:
    labels = tuple(str(value) for value in ranking.get("candidate_labels", []))
    if labels != CANDIDATE_LABELS:
        raise ValueError(f"Candidate labels mismatch for {identifier} step {step}: {labels!r}")
    valid = ranking.get("valid_mask")
    eligible = ranking.get("eligible_mask")
    if not isinstance(valid, list) or len(valid) != 5 or not isinstance(eligible, list) or len(eligible) != 5:
        raise ValueError(f"Candidate masks must have length five for {identifier} step {step}")
    valid_mask = [_as_bool(value) for value in valid]
    eligible_mask = [_as_bool(value) for value in eligible]
    if any(eligible_mask[index] and not valid_mask[index] for index in range(5)):
        raise ValueError(f"Ineligible candidate became eligible for {identifier} step {step}")
    scores = _finite_list(ranking.get("scores"), "candidate_ranking.scores", allow_nonfinite=True)
    for index, value in enumerate(scores):
        if valid_mask[index] and not math.isfinite(value):
            raise ValueError(f"Valid candidate has non-finite score for {identifier} step {step} index {index}")
    for field in SCORE_FIELDS[1:]:
        if ranking.get(field) is not None:
            _finite_list(ranking.get(field), f"candidate_ranking.{field}", allow_nonfinite=True)
    rejection_reasons: list[list[str]] | None = None
    rejection_field = "candidate_rejection_reasons"
    if rejection_field in ranking:
        raw_reasons = ranking.get(rejection_field)
        if not isinstance(raw_reasons, list) or len(raw_reasons) != 5:
            raise ValueError(f"Candidate rejection reasons must have length five for {identifier} step {step}")
        rejection_reasons = []
        for index, reasons in enumerate(raw_reasons):
            if not isinstance(reasons, list):
                raise ValueError(f"Candidate rejection reasons must be lists for {identifier} step {step}")
            values = [str(reason) for reason in reasons]
            if valid_mask[index] and values:
                raise ValueError(f"Valid candidate has rejection reasons for {identifier} step {step} index {index}")
            if not valid_mask[index] and not values:
                raise ValueError(f"Invalid candidate has no rejection reason for {identifier} step {step} index {index}")
            rejection_reasons.append(values)
    selected = _as_int(ranking.get("selected_index"), "candidate_ranking.selected_index")
    if not 0 <= selected < 5 or not valid_mask[selected]:
        raise ValueError(f"Selected candidate is invalid for {identifier} step {step}: {selected}")
    mode = str(ranking.get("execution_mode", ""))
    if mode == "trusted" and not eligible_mask[selected]:
        raise ValueError(f"Trusted selection is not eligible for {identifier} step {step}")
    if mode != "trusted" and selected != 0:
        raise ValueError(f"Fallback selected a non-nominal candidate for {identifier} step {step}")
    finite_scores = sorted(value for value in scores if math.isfinite(value))
    margin = float(finite_scores[1] - finite_scores[0]) if len(finite_scores) >= 2 else None
    return {
        "labels": list(labels),
        "valid_mask": valid_mask,
        "eligible_mask": eligible_mask,
        "scores": scores,
        "selected_index": selected,
        "execution_mode": mode,
        "top_two_score_margin": margin,
        "rejection_reasons_present": rejection_field in ranking,
        "rejection_reasons": rejection_reasons,
    }


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3 or len(y) != len(x) or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    result = spearmanr(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    value = float(result.statistic)
    return value if math.isfinite(value) else None


def _episode_key(row: Mapping[str, Any]) -> tuple[int, str, int]:
    return (
        _as_int(row.get("training_seed"), "training_seed"),
        str(row.get("variant")),
        _as_int(row.get("episode_index"), "episode_index"),
    )


def _empty_candidate_stats() -> dict[str, dict[str, Any]]:
    return {
        label: {
            "steps": 0,
            "valid_steps": 0,
            "eligible_steps": 0,
            "selected_steps": 0,
            "trusted_selected_steps": 0,
            "score_sum": 0.0,
            "score_count": 0,
            "episode_score_means": [],
            "episode_outcomes": [],
            "selected_episode_outcomes": [],
        }
        for label in CANDIDATE_LABELS
    }


def _finalize_candidate_stats(stats: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for label in CANDIDATE_LABELS:
        value = stats[label]
        episode_scores = [float(item) for item in value["episode_score_means"] if math.isfinite(float(item))]
        outcomes = [float(item) for item in value["episode_outcomes"]]
        result[label] = {
            "steps": int(value["steps"]),
            "valid_steps": int(value["valid_steps"]),
            "eligible_steps": int(value["eligible_steps"]),
            "selected_steps": int(value["selected_steps"]),
            "trusted_selected_steps": int(value["trusted_selected_steps"]),
            "valid_fraction": float(value["valid_steps"] / value["steps"]) if value["steps"] else None,
            "eligible_fraction": float(value["eligible_steps"] / value["steps"]) if value["steps"] else None,
            "selected_fraction": float(value["selected_steps"] / value["steps"]) if value["steps"] else None,
            "mean_score": float(value["score_sum"] / value["score_count"]) if value["score_count"] else None,
            "selected_episode_count": len(value["selected_episode_outcomes"]),
            "selected_episode_safe_capture_rate": float(np.mean(value["selected_episode_outcomes"])) if value["selected_episode_outcomes"] else None,
            "score_vs_episode_safe_capture_spearman": _spearman(episode_scores, outcomes),
        }
    return result


def audit_failure_index(failure_index: Path) -> dict[str, Any]:
    report, index_hashes = load_failure_index(failure_index)
    runs = validate_source_runs(report)
    rows = [row for row in report["rows"] if isinstance(row, Mapping)]
    if len(rows) != len(report["rows"]):
        raise ValueError("Failure-index contains a non-object row")
    rows_by_key = {_episode_key(row): row for row in rows}
    stats = _empty_candidate_stats()
    episode_records: list[dict[str, Any]] = []
    variant_steps: Counter[str] = Counter()
    variant_ranked_episodes: Counter[str] = Counter()
    invalid_mask_count = 0
    invalid_selected_count = 0
    trusted_ineligible_count = 0
    missing_rejection_reason_steps = 0
    rejection_reason_counts: Counter[str] = Counter()
    m0_unranked_episodes = 0
    for key in sorted(rows_by_key):
        row = rows_by_key[key]
        variant = key[1]
        run = runs.get((key[0], variant))
        if run is None:
            raise ValueError(f"No validated source run for episode {key}")
        trace_path, _scene_hash = _validate_episode_source(row, run)
        trace = read_trace(trace_path, key[2])
        rankings = [item.get("candidate_ranking") for item in trace]
        if variant == "m0":
            if any(ranking is not None for ranking in rankings):
                raise ValueError(f"M0 unexpectedly contains candidate ranking: {key}")
            m0_unranked_episodes += 1
            continue
        selected_indices: list[int] = []
        margins: list[float] = []
        episode_score_lists: dict[str, list[float]] = defaultdict(list)
        for source_step, ranking in zip(trace, rankings):
            variant_steps[variant] += 1
            if not isinstance(ranking, Mapping):
                raise ValueError(f"Missing candidate ranking for {_episode_key(row)} step {source_step.get('step')}")
            validated = _validate_ranking(
                ranking,
                identifier=f"{key[0]}:{variant}:{key[2]:04d}",
                step=_as_int(source_step.get("step"), "trace step"),
            )
            if not validated["rejection_reasons_present"]:
                missing_rejection_reason_steps += 1
            elif validated["rejection_reasons"] is not None:
                for reasons in validated["rejection_reasons"]:
                    rejection_reason_counts.update(reasons)
            selected_indices.append(int(validated["selected_index"]))
            if validated["top_two_score_margin"] is not None:
                margins.append(float(validated["top_two_score_margin"]))
            for index, label in enumerate(CANDIDATE_LABELS):
                item = stats[label]
                item["steps"] += 1
                item["valid_steps"] += int(validated["valid_mask"][index])
                item["eligible_steps"] += int(validated["eligible_mask"][index])
                item["selected_steps"] += int(validated["selected_index"] == index)
                item["trusted_selected_steps"] += int(validated["selected_index"] == index and validated["execution_mode"] == "trusted")
                score = validated["scores"][index]
                if math.isfinite(score):
                    item["score_sum"] += score
                    item["score_count"] += 1
                    episode_score_lists[label].append(score)
        outcome = float(_as_bool(row.get("safe_capture")))
        for label in CANDIDATE_LABELS:
            scores = episode_score_lists[label]
            if scores:
                stats[label]["episode_score_means"].append(float(np.mean(scores)))
                stats[label]["episode_outcomes"].append(outcome)
                if CANDIDATE_LABELS.index(label) in selected_indices:
                    stats[label]["selected_episode_outcomes"].append(outcome)
        selected_labels = [CANDIDATE_LABELS[index] for index in selected_indices]
        episode_records.append(
            {
                "training_seed": key[0],
                "variant": variant,
                "episode_index": key[2],
                "safe_capture": bool(outcome),
                "baseline_safe_capture": _as_bool(row.get("baseline_safe_capture")),
                "trace_steps": len(trace),
                "selected_labels": selected_labels,
                "first_selected_label": selected_labels[0],
                "non_nominal_selection_rate": float(np.mean(np.asarray(selected_indices) != 0)),
                "candidate_switch_rate": float(np.mean(np.diff(selected_indices) != 0)) if len(selected_indices) > 1 else 0.0,
                "mean_top_two_score_margin": float(np.mean(margins)) if margins else None,
            }
        )
        variant_ranked_episodes[variant] += 1
    variant_summary: dict[str, dict[str, Any]] = {}
    for variant in sorted({str(row.get("variant")) for row in rows}):
        episodes = [item for item in episode_records if item["variant"] == variant]
        source_episodes = [item for item in rows if str(item.get("variant")) == variant]
        variant_summary[variant] = {
            "episodes": len(source_episodes),
            "safe_capture_count": int(sum(_as_bool(item.get("safe_capture")) for item in source_episodes)),
            "safe_capture_rate": float(np.mean([float(_as_bool(item.get("safe_capture"))) for item in source_episodes])) if source_episodes else None,
            "ranked_steps": int(variant_steps[variant]),
            "ranked_episodes": int(variant_ranked_episodes[variant]),
            "mean_non_nominal_selection_rate": float(np.mean([item["non_nominal_selection_rate"] for item in episodes])) if episodes else None,
            "mean_candidate_switch_rate": float(np.mean([item["candidate_switch_rate"] for item in episodes])) if episodes else None,
            "mean_top_two_score_margin": float(np.mean([item["mean_top_two_score_margin"] for item in episodes if item["mean_top_two_score_margin"] is not None])) if any(item["mean_top_two_score_margin"] is not None for item in episodes) else None,
        }
    all_ranked_steps = sum(variant_steps.values())
    candidate_stats = _finalize_candidate_stats(stats)
    return {
        "audit_type": AUDIT_TYPE,
        "development_only": True,
        "locked_test_opened": False,
        "input_index": {"path": str(failure_index.resolve()), **index_hashes, "index_type": report["index_type"]},
        "source_run_count": len(runs),
        "source_episode_count": len(rows),
        "ranked_episode_count": len(episode_records),
        "ranked_step_count": all_ranked_steps,
        "m0_unranked_episode_count": m0_unranked_episodes,
        "candidate_labels": list(CANDIDATE_LABELS),
        "candidate_stats": candidate_stats,
        "variant_summary": variant_summary,
        "episode_records": episode_records,
        "observability": {
            "rejection_reasons_field_present": missing_rejection_reason_steps == 0,
            "rejection_reasons_missing_steps": missing_rejection_reason_steps,
            "settled_counterfactual_per_candidate_labels": False,
            "settled_label_scope": "selected_trajectory_episode_only",
            "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        },
        "invariant_counts": {
            "invalid_candidate_eligible": invalid_mask_count,
            "invalid_selected_candidate": invalid_selected_count,
            "trusted_ineligible_selection": trusted_ineligible_count,
        },
        "gates": {
            "fixed_five_candidate_labels": True,
            "candidate_masks_shape_and_selection_valid": True,
            "no_invalid_candidate_entered_eligible_mask": True,
            "rejection_reason_observability": missing_rejection_reason_steps == 0,
            "counterfactual_settled_outcome_observability": False,
            "candidate_reachability_gate": True,
            "ranking_causal_gate": False,
        },
        "classification": "candidate_reachability_pass_ranking_evidence_incomplete",
        "provenance": {
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": version("scipy"),
            "tensorboard": version("tensorboard"),
            "source_hashes": {
                "scripts/audit_jepa_safe_capture_candidate_ranking.py": sha256(Path(__file__).resolve()),
                "scripts/replay_jepa_safe_capture_failures.py": sha256(PROJECT_ROOT / "scripts/replay_jepa_safe_capture_failures.py"),
            },
        },
    }


def _write_tensorboard(report: Mapping[str, Any], logdir: Path) -> dict[str, Any]:
    logdir = logdir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty TensorBoard logdir: {logdir}")
    logdir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(logdir), flush_secs=1) as writer:
        writer.add_text("Config/candidate_audit", json.dumps({"audit_type": report["audit_type"], "candidate_labels": report["candidate_labels"], "development_only": True, "locked_test_opened": False}, indent=2), 0)
        writer.add_text("Provenance/input_index", json.dumps(report["input_index"], indent=2), 0)
        writer.add_text("Provenance/observability", json.dumps(report["observability"], indent=2), 0)
        writer.add_text("Gates/status", json.dumps(report["gates"], indent=2), 0)
        for label, stats in report["candidate_stats"].items():
            writer.add_scalar(f"Candidates/{label}/valid_fraction", float(stats["valid_fraction"] or 0.0), 0)
            writer.add_scalar(f"Candidates/{label}/eligible_fraction", float(stats["eligible_fraction"] or 0.0), 0)
            writer.add_scalar(f"Candidates/{label}/selected_fraction", float(stats["selected_fraction"] or 0.0), 0)
            if stats["selected_episode_safe_capture_rate"] is not None:
                writer.add_scalar(f"Candidates/{label}/selected_episode_safe_capture_rate", float(stats["selected_episode_safe_capture_rate"]), 0)
        for variant, stats in report["variant_summary"].items():
            writer.add_scalar(f"Variants/{variant}/safe_capture_rate", float(stats["safe_capture_rate"] or 0.0), 0)
            writer.add_scalar(f"Variants/{variant}/non_nominal_selection_rate", float(stats["mean_non_nominal_selection_rate"] or 0.0), 0)
            writer.add_scalar(f"Variants/{variant}/candidate_switch_rate", float(stats["mean_candidate_switch_rate"] or 0.0), 0)
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    tags = accumulator.Tags()
    required = {"Config/candidate_audit/text_summary", "Provenance/input_index/text_summary", "Provenance/observability/text_summary", "Gates/status/text_summary"}
    missing = sorted(required.difference(tags.get("tensors", [])))
    events = sorted(path.name for path in logdir.glob("events.out.tfevents.*"))
    if missing or not events:
        raise ValueError(f"Candidate audit TensorBoard validation failed: missing={missing}, events={events}")
    return {"logdir": str(logdir), "event_files": events, "scalar_tag_count": len(tags.get("scalars", [])), "text_tag_count": len(tags.get("tensors", [])), "required_provenance": True}


def _write_outputs(report: dict[str, Any], output_dir: Path, tensorboard_logdir: Path) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    with (output_dir / "candidate_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["label", "steps", "valid_steps", "eligible_steps", "selected_steps", "valid_fraction", "eligible_fraction", "selected_fraction", "mean_score", "selected_episode_count", "selected_episode_safe_capture_rate", "score_vs_episode_safe_capture_spearman"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, stats in report["candidate_stats"].items():
            writer.writerow({"label": label, **{field: stats[field] for field in fields[1:]}})
    lines = [
        "# WP-E Candidate Reachability and Ranking Audit",
        "",
        "**Status:** development-only; `locked_test_opened=false`  ",
        f"**Classification:** `{report['classification']}`  ",
        f"**Ranked steps:** {report['ranked_step_count']}  ",
        "",
        "## Reachability",
        "",
        "All ranked traces use the fixed five candidate labels. Candidate masks, "
        "selection invariants, and finite score requirements passed; no invalid "
        "candidate was admitted to the eligible mask. The current source traces "
        "contain no per-candidate `rejection_reasons` field, so reason-level "
        "coverage is explicitly unavailable rather than inferred.",
        "",
        "| Candidate | Valid fraction | Eligible fraction | Selected fraction | Selected-trajectory safe-capture rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, stats in report["candidate_stats"].items():
        def fmt(value: Any) -> str:
            return "n/a" if value is None else f"{float(value):.4f}"
        lines.append(f"| `{label}` | {fmt(stats['valid_fraction'])} | {fmt(stats['eligible_fraction'])} | {fmt(stats['selected_fraction'])} | {fmt(stats['selected_episode_safe_capture_rate'])} |")
    lines.extend([
        "",
        "## Evidence Limit",
        "",
        "The settled outcome is recorded once per episode for the selected "
        "trajectory. There are no settled per-candidate counterfactual labels, "
        "so top-1 precision/recall and causal rank correlation are not claimed. "
        "The reported score/outcome correlation is descriptive and uses episode "
        "aggregates only.",
        "",
        "## Gates",
        "",
        f"- candidate reachability gate: `{str(report['gates']['candidate_reachability_gate']).lower()}`",
        f"- rejection-reason observability: `{str(report['gates']['rejection_reason_observability']).lower()}`",
        f"- counterfactual settled outcome observability: `{str(report['gates']['counterfactual_settled_outcome_observability']).lower()}`",
        f"- ranking causal gate: `{str(report['gates']['ranking_causal_gate']).lower()}`",
        "",
        "The next required change is to record candidate precheck rejection "
        "reasons and offline-only per-candidate settled labels in a new protocol "
        "revision before adjusting ranking weights or opening a final block.",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["tensorboard"] = _write_tensorboard(report, tensorboard_logdir)
    (output_dir / "candidate_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    manifest: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "hash_manifest.json":
            manifest[str(path.relative_to(output_dir)).replace("\\", "/")] = sha256(path)
    (output_dir / "hash_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    args = parser.parse_args()
    if not args.development_only:
        raise ValueError("Candidate audit requires --development-only")
    report = audit_failure_index(args.failure_index.resolve())
    _write_outputs(report, args.output_dir, args.tensorboard_logdir)
    print(json.dumps({"classification": report["classification"], "ranked_steps": report["ranked_step_count"], "gates": report["gates"], "tensorboard": report["tensorboard"]}, indent=2))


if __name__ == "__main__":
    main()
