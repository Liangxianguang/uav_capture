"""Stratify P37 route-agreement failures without executing any action."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter


TIE_MARGIN = 0.005


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _utility(row: dict[str, str], weights: dict[str, float], *, predicted: bool) -> float:
    prefix = "predicted" if predicted else "truth"
    return (
        float(row[f"{prefix}_progress"]) / 0.3
        - weights["length"] * float(row["route_length"]) / 10.0
        - weights["escape"] * float(row[f"{prefix}_escape"]) / 2.0
        + weights["cbf"] * float(row[f"{prefix}_cbf"])
        - weights["switch"] * float(row["switch_penalty"])
    )


def _best(rows: list[dict[str, str]], weights: dict[str, float], *, predicted: bool) -> tuple[int, float]:
    scored = [
        (int(row["candidate"]), _utility(row, weights, predicted=predicted))
        for row in rows
    ]
    candidate, score = max(scored, key=lambda item: (item[1], -item[0]))
    return candidate, score


def _family(candidate: int, labels: list[str]) -> str:
    return labels[candidate] if 0 <= candidate < len(labels) else "unknown"


def diagnose(details_path: Path, audit_path: Path, metadata_path: Path) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    labels = [str(value) for value in metadata["route_labels"]]
    rows: list[dict[str, str]] = []
    with details_path.open(newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
    if not rows:
        raise ValueError("P38 details file is empty")
    model_names = sorted({str(row["model"]) for row in rows})
    if set(model_names) != set(audit.get("models", {})):
        raise ValueError("details and utility audit model names disagree")

    grouped: dict[tuple[str, str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["split"], int(row["scenario_index"]), int(row["time_index"]))].append(row)

    report: dict[str, Any] = {
        "audit_type": "dn_mpc_p38_route_agreement_diagnosis",
        "development_only": True,
        "locked_test_opened": False,
        "details_sha256": _sha256(details_path),
        "utility_audit_sha256": _sha256(audit_path),
        "metadata_sha256": _sha256(metadata_path),
        "route_labels": labels,
        "previous_route_source": (
            "planner_selected_candidate"
            if bool(audit.get("route_switch_penalty", {}).get("planner_selection_identity_used", False))
            else "frozen_actor_executed_route"
        ),
        "models": {},
    }
    for model_name in model_names:
        weights = {key: float(value) for key, value in audit["models"][model_name]["selected_weights"].items()}
        model_groups = {key[1:]: value for key, value in grouped.items() if key[0] == model_name}
        for split in sorted({key[0] for key in model_groups}):
            split_groups = [value for key, value in model_groups.items() if key[0] == split]
            eligible_counts: list[int] = []
            valid_groups: list[dict[str, Any]] = []
            abstention_groups = 0
            for group_rows in split_groups:
                eligible = [row for row in group_rows if _bool(row["eligible"])]
                eligible_counts.append(len(eligible))
                selected = int(group_rows[0]["selected"])
                if selected < 0:
                    abstention_groups += 1
                if len(eligible) < 2:
                    continue
                truth_best, truth_score = _best(eligible, weights, predicted=False)
                model_best, model_score = _best(eligible, weights, predicted=True)
                ordered_truth = sorted((_utility(row, weights, predicted=False) for row in eligible), reverse=True)
                valid_groups.append(
                    {
                        "truth_best": truth_best,
                        "model_best": model_best,
                        "selected": selected,
                        "truth_gap": float(ordered_truth[0] - ordered_truth[1]),
                        "previous_route": int(group_rows[0]["previous_route"]),
                        "truth_score": truth_score,
                        "model_score": model_score,
                    }
                )
            total = len(split_groups)
            model_truth = [item for item in valid_groups if item["model_best"] == item["truth_best"]]
            model_selected = [item for item in valid_groups if item["selected"] >= 0 and item["model_best"] == item["selected"]]
            near_tie = [item for item in valid_groups if item["truth_gap"] <= TIE_MARGIN]
            with_previous = [item for item in valid_groups if item["previous_route"] >= 0]
            predicted_switch = [item for item in with_previous if item["model_best"] != item["previous_route"]]
            planner_selection_change = [item for item in with_previous if item["selected"] >= 0 and item["selected"] != item["previous_route"]]
            family_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"groups": 0, "model_truth_matches": 0, "model_selected_matches": 0})
            selected_family_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"groups": 0, "model_truth_matches": 0, "model_selected_matches": 0})
            eligibility_by_family: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "eligible": 0})
            for group_rows in split_groups:
                for row in group_rows:
                    family = _family(int(row["candidate"]), labels)
                    eligibility_by_family[family]["rows"] += 1
                    eligibility_by_family[family]["eligible"] += int(_bool(row["eligible"]))
            for item in valid_groups:
                family = _family(item["model_best"], labels)
                family_stats[family]["groups"] += 1
                family_stats[family]["model_truth_matches"] += int(item["model_best"] == item["truth_best"])
                family_stats[family]["model_selected_matches"] += int(item["selected"] >= 0 and item["model_best"] == item["selected"])
                selected_family = _family(item["selected"], labels)
                selected_family_stats[selected_family]["groups"] += 1
                selected_family_stats[selected_family]["model_truth_matches"] += int(item["model_best"] == item["truth_best"])
                selected_family_stats[selected_family]["model_selected_matches"] += int(item["selected"] >= 0 and item["model_best"] == item["selected"])
            report["models"].setdefault(model_name, {"selected_weights": weights, "splits": {}})["splits"][split] = {
                "total_groups": total,
                "valid_groups_at_least_two_eligible": len(valid_groups),
                "zero_or_one_eligible_groups": sum(count < 2 for count in eligible_counts),
                "mean_eligible_candidates": float(np.mean(eligible_counts)) if eligible_counts else 0.0,
                "abstention_groups": abstention_groups,
                "abstention_fraction": abstention_groups / total if total else 0.0,
                "model_vs_truth": len(model_truth) / len(valid_groups) if valid_groups else None,
                "model_vs_selected": len(model_selected) / len(valid_groups) if valid_groups else None,
                "near_tie_groups": len(near_tie),
                "near_tie_fraction": len(near_tie) / len(valid_groups) if valid_groups else None,
                "groups_with_previous_route": len(with_previous),
                # The source is declared at report level.  Keep the historical
                # keys for machine compatibility, but expose source-specific
                # aliases so actor and planner decision chains cannot be mixed.
                "model_vs_previous_executed_route_fraction": len(predicted_switch) / len(with_previous) if with_previous else None,
                "planner_selected_vs_previous_executed_route_fraction": len(planner_selection_change) / len(with_previous) if with_previous else None,
                "model_vs_previous_selected_route_fraction": len(predicted_switch) / len(with_previous) if with_previous else None,
                "planner_selected_vs_previous_selected_route_fraction": len(planner_selection_change) / len(with_previous) if with_previous else None,
                "family_by_model_best": dict(sorted(family_stats.items())),
                "family_by_planner_selected": dict(sorted(selected_family_stats.items())),
                "eligibility_by_route_family": dict(sorted(eligibility_by_family.items())),
            }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    markdown = args.markdown_output.resolve()
    if output.exists() or markdown.exists() or (args.tensorboard_logdir.exists() and any(args.tensorboard_logdir.iterdir())):
        raise FileExistsError("refusing to overwrite P38 outputs")
    report = diagnose(args.details.resolve(), args.audit.resolve(), args.metadata.resolve())
    with SummaryWriter(log_dir=str(args.tensorboard_logdir.resolve()), flush_secs=1) as writer:
        writer.add_text("Contract/diagnosis", json.dumps({"development_only": True, "locked_test_opened": False, "previous_route_source": report["previous_route_source"]}, sort_keys=True), 0)
        for model_name, model_report in report["models"].items():
            writer.add_text(f"Utility/{model_name}/selected_weights", json.dumps(model_report["selected_weights"], sort_keys=True), 0)
            for split, values in model_report["splits"].items():
                for metric in ("model_vs_truth", "model_vs_selected", "near_tie_fraction", "abstention_fraction", "model_vs_previous_executed_route_fraction", "planner_selected_vs_previous_executed_route_fraction"):
                    value = values[metric]
                    if value is not None:
                        writer.add_scalar(f"P38/{model_name}/{metric}/{split}", float(value), 0)
                writer.add_scalar(f"P38/{model_name}/mean_eligible_candidates/{split}", values["mean_eligible_candidates"], 0)
                writer.add_scalar(f"P38/{model_name}/valid_groups/{split}", values["valid_groups_at_least_two_eligible"], 0)
                for family, family_values in values["family_by_model_best"].items():
                    writer.add_scalar(f"P38/{model_name}/model_best_family_groups/{split}/{family}", family_values["groups"], 0)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    previous_label = (
        "previous planner-selected route"
        if report["previous_route_source"] == "planner_selected_candidate"
        else "previous frozen-actor executed route"
    )
    lines = [
        "# DN-MPC P38 Route-Agreement Diagnosis",
        "",
        "**Status:** development-only; offline-only; no action executed.",
        "",
        "This report stratifies the learned-evaluator disagreement by route family, eligibility, near ties, route switching, and explicit CBF abstentions. It does not change the planner or safety filter.",
        f"The previous-route identity source for this archive is **{previous_label}**.",
        "",
    ]
    for model_name, model_report in report["models"].items():
        weights = model_report["selected_weights"]
        lines.append(f"## {model_name}")
        lines.append("")
        lines.append(f"Selected utility weights: `({weights['length']:g}, {weights['escape']:g}, {weights['cbf']:g}, {weights['switch']:g})`.")
        lines.append("")
        lines.append(f"| Split | valid groups | zero/one eligible | abstention | model vs truth | model vs selected | near tie | planner selected vs {previous_label} | model vs {previous_label} |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for split, values in sorted(model_report["splits"].items()):
            fmt = lambda value: "n/a" if value is None else f"{value:.2%}"
            lines.append(f"| {split} | {values['valid_groups_at_least_two_eligible']} | {values['zero_or_one_eligible_groups']} | {fmt(values['abstention_fraction'])} | {fmt(values['model_vs_truth'])} | {fmt(values['model_vs_selected'])} | {fmt(values['near_tie_fraction'])} | {fmt(values['planner_selected_vs_previous_executed_route_fraction'])} | {fmt(values['model_vs_previous_executed_route_fraction'])} |")
        lines.append("")
        lines.append("Route-family model-best counts (validation):")
        lines.append("")
        for family, family_values in sorted(model_report["splits"].get("validation", {}).get("family_by_model_best", {}).items()):
            lines.append(f"- `{family}`: {family_values['groups']} groups, model-vs-truth {family_values['model_truth_matches']}/{family_values['groups']}.")
        lines.append("")
    lines.append("P38 is a diagnosis gate. Candidate agreement remains a promotion requirement; no online JEPA override or Ledger-Lite is authorized by this report.")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
