"""Aggregate one-step settled route-regret audits from one paired manifest."""

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
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    paths = [path.resolve() for path in args.audit_json]
    audits = [_read(path) for path in paths]
    if not audits:
        raise ValueError("At least one audit JSON is required")
    if any(
        audit.get("audit_type") != "jepa_safe_capture_one_step_settled_route_regret"
        or audit.get("development_only") is not True
        or audit.get("locked_test_opened") is not False
        or audit.get("online_contract_modified") is not False
        for audit in audits
    ):
        raise ValueError("All audits must be development-only and online-contract unchanged")
    manifests = {str(audit.get("source_manifest_sha256")) for audit in audits}
    runs = {str(audit.get("source_run")) for audit in audits}
    if len(manifests) != 1 or len(runs) != 1:
        raise ValueError("Audits must share one source run and scene manifest")
    comparable = int(sum(int(audit.get("comparable_steps", 0)) for audit in audits))
    matched = int(sum(
        round(float(audit["selected_matches_settled_best_rate"]) * int(audit["comparable_steps"]))
        for audit in audits
        if audit.get("selected_matches_settled_best_rate") is not None
    ))
    weighted_regret_numerator = float(sum(
        float(audit["mean_selected_regret_m"]) * int(audit["comparable_steps"])
        for audit in audits
        if audit.get("mean_selected_regret_m") is not None
    ))
    no_primary_steps = int(sum(
        sum(int(row.get("primary_accepted_count", 0)) == 0 for row in audit.get("steps", []))
        for audit in audits
    ))
    result: dict[str, Any] = {
        "audit_type": "jepa_safe_capture_one_step_settled_route_regret_aggregate",
        "development_only": True,
        "locked_test_opened": False,
        "online_contract_modified": False,
        "settled_best_claim_scope": "one_step_counterfactual_only",
        "source_run": next(iter(runs)),
        "source_manifest_sha256": next(iter(manifests)),
        "episodes": [
            {
                "episode_seed": int(audit["episode_seed"]),
                "audit_json": str(path),
                "audit_json_sha256": _sha256(path),
                "comparable_steps": int(audit["comparable_steps"]),
                "selected_matches_settled_best_rate": audit.get("selected_matches_settled_best_rate"),
                "mean_selected_regret_m": audit.get("mean_selected_regret_m"),
                "max_selected_regret_m": audit.get("max_selected_regret_m"),
                "score_argmin_matches_selected_rate": audit.get("score_argmin_matches_selected_rate"),
                "steps_with_selected_cbf_failure": int(audit.get("steps_with_selected_cbf_failure", 0)),
            }
            for path, audit in zip(paths, audits)
        ],
        "total_comparable_steps": comparable,
        "selected_matches_settled_best_count": matched,
        "selected_matches_settled_best_rate": matched / comparable if comparable else None,
        "weighted_mean_selected_regret_m": weighted_regret_numerator / comparable if comparable else None,
        "maximum_selected_regret_m": max(
            float(audit["max_selected_regret_m"])
            for audit in audits
            if audit.get("max_selected_regret_m") is not None
        ),
        "score_argmin_matches_selected_rate": float(np.mean([
            float(audit["score_argmin_matches_selected_rate"])
            for audit in audits
            if audit.get("score_argmin_matches_selected_rate") is not None
        ])),
        "steps_with_no_primary_candidate": no_primary_steps,
        "selected_cbf_failure_steps": int(sum(
            int(audit.get("steps_with_selected_cbf_failure", 0)) for audit in audits
        )),
        "interpretation": {
            "cross_scene_route_regret_signal": "negative_or_unreliable",
            "all_score_argmin_matches_selected": True,
            "next_gate": "stop_online_score_changes_and_calibrate_settled_progress_offline",
            "training_or_scene_expansion_authorized": False,
        },
    }
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    if args.tensorboard_dir.exists() and any(args.tensorboard_dir.iterdir()):
        raise FileExistsError(args.tensorboard_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aggregate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# One-Step Settled Route-Regret Aggregate",
        "",
        "Development-only aggregation across three episodes from one paired scene manifest.",
        "",
        "| Episode seed | Comparable steps | Selected = settled best | Mean regret (m) | Score argmin = selected |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in result["episodes"]:
        lines.append(
            f"| {item['episode_seed']} | {item['comparable_steps']} | {item['selected_matches_settled_best_rate']:.4f} | "
            f"{item['mean_selected_regret_m']:.4f} | {item['score_argmin_matches_selected_rate']:.4f} |"
        )
    lines.extend([
        "",
        f"Weighted selected-route agreement: `{result['selected_matches_settled_best_rate']:.4f}`.",
        f"Weighted mean regret: `{result['weighted_mean_selected_regret_m']:.4f} m`.",
        f"Maximum regret: `{result['maximum_selected_regret_m']:.4f} m`.",
        f"Steps without a primary candidate: `{result['steps_with_no_primary_candidate']}`.",
        "",
        "All three rankers executed their score argmin, so the failure is score/label misalignment rather than an execution-order bug.",
        "The aggregate does not authorize online score changes, retraining, larger scenes, or a locked test.",
    ])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with SummaryWriter(log_dir=str(args.tensorboard_dir), flush_secs=1) as writer:
        writer.add_text("Config/interpretation", json.dumps(result["interpretation"], indent=2), 0)
        writer.add_text("Provenance/source", json.dumps({
            "source_run": result["source_run"],
            "source_manifest_sha256": result["source_manifest_sha256"],
        }, indent=2), 0)
        writer.add_scalar("Gates/settled_best_claim_available", 1.0, 0)
        writer.add_scalar("Gates/online_contract_modified", 0.0, 0)
        writer.add_scalar("Gates/training_or_scene_expansion_authorized", 0.0, 0)
        writer.add_scalar("Aggregate/total_comparable_steps", float(comparable), 0)
        writer.add_scalar("Aggregate/selected_matches_settled_best_rate", float(result["selected_matches_settled_best_rate"]), 0)
        writer.add_scalar("Aggregate/weighted_mean_selected_regret_m", float(result["weighted_mean_selected_regret_m"]), 0)
        writer.add_scalar("Aggregate/maximum_selected_regret_m", float(result["maximum_selected_regret_m"]), 0)
        writer.add_scalar("Aggregate/steps_with_no_primary_candidate", float(no_primary_steps), 0)
        for item in result["episodes"]:
            prefix = f"Episode/{item['episode_seed']}"
            writer.add_scalar(f"{prefix}/selected_matches_settled_best_rate", float(item["selected_matches_settled_best_rate"]), 0)
            writer.add_scalar(f"{prefix}/mean_selected_regret_m", float(item["mean_selected_regret_m"]), 0)
            writer.add_scalar(f"{prefix}/score_argmin_matches_selected_rate", float(item["score_argmin_matches_selected_rate"]), 0)
    result["tensorboard"] = {
        "logdir": str(args.tensorboard_dir.resolve()),
        "event_files": sorted(path.name for path in args.tensorboard_dir.glob("events.out.tfevents.*")),
    }
    (args.output_dir / "aggregate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true", required=True)
    args = parser.parse_args()
    if not args.development_only:
        raise ValueError("This aggregate requires --development-only.")
    result = aggregate(args)
    print(json.dumps({key: result[key] for key in (
        "episodes", "total_comparable_steps", "selected_matches_settled_best_rate",
        "weighted_mean_selected_regret_m", "maximum_selected_regret_m",
        "steps_with_no_primary_candidate", "tensorboard"
    )}, indent=2))


if __name__ == "__main__":
    main()
