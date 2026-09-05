"""Compare recorded selection with score-argmin on frozen settled branches.

This is an offline diagnostic.  It does not change the source replay, execute
new actions, or claim a full-episode policy result.  It isolates the effect of
top-two abstention and nominal-anchor decisions from the underlying score
ordering using already settled candidate outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object row: {path}")
                rows.append(value)
    if not rows:
        raise ValueError(f"Empty settled rows: {path}")
    return rows


def _settled_best(row: Mapping[str, Any], eligible: list[int]) -> int:
    return min(
        eligible,
        key=lambda index: (
            -int(bool(row["settled_candidates"][index]["settled_safe_capture"])),
            -int(bool(row["settled_candidates"][index]["settled_safety_ok"])),
            -float(row["settled_candidates"][index]["settled_progress_m"]),
            float(row["settled_candidates"][index]["settled_cbf_correction_norm_mps"]),
            index,
        ),
    )


def _score_argmin(row: Mapping[str, Any], eligible: list[int]) -> int:
    scores = [float(row["scores"][index]) for index in eligible]
    if not np.isfinite(scores).all():
        raise ValueError("Eligible scores must be finite")
    return min(eligible, key=lambda index: (float(row["scores"][index]), index))


def _policy_stats(rows: list[dict[str, Any]], selected_key: str) -> dict[str, Any]:
    safety: list[float] = []
    capture: list[float] = []
    progress: list[float] = []
    selected_not_best = 0
    selected_unsettled = 0
    decisions = 0
    for row in rows:
        eligible = [index for index, value in enumerate(row["eligible_mask"]) if bool(value)]
        if len(eligible) < 2:
            continue
        decisions += 1
        best = _settled_best(row, eligible)
        selected = int(row[selected_key])
        selected_not_best += int(selected != best)
        if selected not in eligible:
            selected_unsettled += 1
            continue
        outcome = row["settled_candidates"][selected]
        safety.append(float(bool(outcome["settled_safety_ok"])))
        capture.append(float(bool(outcome["settled_safe_capture"])))
        progress.append(float(outcome["settled_progress_m"]))
    return {
        "decisions": decisions,
        "selected_not_best_count": selected_not_best,
        "selected_not_best_rate": selected_not_best / max(decisions, 1),
        "selected_unsettled_count": selected_unsettled,
        "settled_decisions": len(safety),
        "settled_safety_rate": float(np.mean(safety)) if safety else None,
        "settled_safe_capture_rate": float(np.mean(capture)) if capture else None,
        "mean_settled_progress_m": float(np.mean(progress)) if progress else None,
    }


def analyze_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for row in rows:
        eligible = [index for index, value in enumerate(row["eligible_mask"]) if bool(value)]
        if len(eligible) < 2:
            continue
        score_index = _score_argmin(row, eligible)
        settled_index = _settled_best(row, eligible)
        normalized.append(
            {
                "training_seed": int(row.get("training_seed", -1)),
                "episode_index": int(row["episode_index"]),
                "step": int(row["step"]),
                "eligible_count": len(eligible),
                "recorded_selected_index": int(row["selected_index"]),
                "score_argmin_index": score_index,
                "settled_best_index": settled_index,
                "recorded_is_settled_best": int(int(row["selected_index"]) == settled_index),
                "score_argmin_is_settled_best": int(score_index == settled_index),
                "recorded_is_score_argmin": int(int(row["selected_index"]) == score_index),
            }
        )
        score_rows.append({**row, "selected_index": score_index})
    return {
        "decision_rows": len(rows),
        "multi_eligible_decisions": len(normalized),
        "recorded_selected": _policy_stats(rows, "selected_index"),
        "score_argmin": _policy_stats(score_rows, "selected_index"),
        "settled_best_upper_bound": {
            "decisions": len(normalized),
            "selected_not_best_count": 0,
            "selected_not_best_rate": 0.0,
            "recorded_agreement_rate": float(np.mean([item["recorded_is_settled_best"] for item in normalized])) if normalized else None,
            "score_argmin_agreement_rate": float(np.mean([item["score_argmin_is_settled_best"] for item in normalized])) if normalized else None,
        },
        "agreement": {
            "recorded_vs_score_argmin_rate": float(np.mean([item["recorded_is_score_argmin"] for item in normalized])) if normalized else None,
            "recorded_vs_settled_best_rate": float(np.mean([item["recorded_is_settled_best"] for item in normalized])) if normalized else None,
            "score_argmin_vs_settled_best_rate": float(np.mean([item["score_argmin_is_settled_best"] for item in normalized])) if normalized else None,
        },
        "rows": normalized,
    }


def _validate_tensorboard(logdir: Path) -> dict[str, Any]:
    events = sorted(path.name for path in logdir.glob("events.out.tfevents.*"))
    accumulator = EventAccumulator(str(logdir))
    accumulator.Reload()
    tags = accumulator.Tags()
    required = {
        "Config/abstention_counterfactual/text_summary",
        "Gates/status/text_summary",
    }
    missing = sorted(required.difference(tags.get("tensors", [])))
    if not events or missing:
        raise ValueError(f"TensorBoard validation failed: events={events}, missing={missing}")
    return {"logdir": str(logdir), "event_files": events, "scalar_tag_count": len(tags.get("scalars", []))}


def write_report(input_paths: list[Path], output_dir: Path, tensorboard_dir: Path) -> dict[str, Any]:
    if not input_paths:
        raise ValueError("At least one settled decision_rows.jsonl is required")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    if tensorboard_dir.exists() and any(tensorboard_dir.iterdir()):
        raise FileExistsError(tensorboard_dir)
    per_seed: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for path in input_paths:
        rows = _read_rows(path)
        seed = str(rows[0].get("training_seed", path.stem))
        if any(str(row.get("training_seed", seed)) != seed for row in rows):
            raise ValueError(f"Mixed training seeds in {path}")
        per_seed[seed] = analyze_rows(rows)
        all_rows.extend(rows)
    aggregate = analyze_rows(all_rows)
    result: dict[str, Any] = {
        "stage": "WP3_v21_abstention_counterfactual",
        "development_only": True,
        "locked_test_opened": False,
        "policy": {
            "recorded_selected": "source trace selected index, including nominal anchor and abstention",
            "score_argmin": "minimum finite score among the recorded eligible candidates, deterministic index tie-break",
            "settled_best": "offline safety/capture/progress oracle used only as an upper-bound diagnostic",
            "no_online_decision_change": True,
        },
        "inputs": {"decision_rows": [str(path.resolve()) for path in input_paths], "sha256": {str(path): sha256(path) for path in input_paths}},
        "per_seed": per_seed,
        "aggregate": aggregate,
        "gates": {
            "development_only": True,
            "locked_test_not_opened": True,
            "all_rows_have_finite_eligible_scores": True,
            "tensorboard_required": True,
        },
        "provenance": {
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "platform": platform.platform(),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "abstention_counterfactual.json").write_text(json.dumps(_jsonable(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# V21 abstention counterfactual diagnosis",
        "",
        "`development_only=true`; `locked_test_opened=false`. This compares frozen settled branches and does not alter online decisions.",
        "",
        "| Seed | Multi-eligible decisions | Recorded selected-not-best | Score argmin selected-not-best | Recorded settled safety | Score argmin settled safety |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, report in sorted(per_seed.items()):
        recorded = report["recorded_selected"]
        score = report["score_argmin"]
        lines.append(
            f"| {seed} | {report['multi_eligible_decisions']} | {recorded['selected_not_best_rate']:.3f} | "
            f"{score['selected_not_best_rate']:.3f} | {recorded['settled_safety_rate']:.3f} | {score['settled_safety_rate']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The score-argmin row is an offline counterfactual upper-bound diagnostic for removing abstention/anchor interference. It is not a deployable policy result and does not bypass CBF.",
        "",
        f"Aggregate recorded-vs-score-argmin agreement: `{aggregate['agreement']['recorded_vs_score_argmin_rate']}`.",
        f"Aggregate score-argmin-vs-settled-best agreement: `{aggregate['agreement']['score_argmin_vs_settled_best_rate']}`.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with SummaryWriter(log_dir=str(tensorboard_dir), flush_secs=1) as writer:
        writer.add_text("Config/abstention_counterfactual", json.dumps(result["policy"], indent=2), 0)
        writer.add_text("Gates/status", json.dumps(result["gates"], indent=2), 0)
        for index, (seed, report) in enumerate(sorted(per_seed.items())):
            writer.add_scalar(f"Seed_{seed}/recorded_selected_not_best_rate", report["recorded_selected"]["selected_not_best_rate"], index)
            writer.add_scalar(f"Seed_{seed}/score_argmin_selected_not_best_rate", report["score_argmin"]["selected_not_best_rate"], index)
            writer.add_scalar(f"Seed_{seed}/recorded_settled_safety_rate", report["recorded_selected"]["settled_safety_rate"] or 0.0, index)
            writer.add_scalar(f"Seed_{seed}/score_argmin_settled_safety_rate", report["score_argmin"]["settled_safety_rate"] or 0.0, index)
        writer.add_scalar("Aggregate/recorded_vs_score_argmin_rate", aggregate["agreement"]["recorded_vs_score_argmin_rate"] or 0.0, 0)
        writer.add_scalar("Aggregate/score_argmin_vs_settled_best_rate", aggregate["agreement"]["score_argmin_vs_settled_best_rate"] or 0.0, 0)
    result["tensorboard"] = _validate_tensorboard(tensorboard_dir)
    (output_dir / "abstention_counterfactual.json").write_text(json.dumps(_jsonable(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-rows", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_report([path.resolve() for path in args.decision_rows], args.output_dir.resolve(), args.tensorboard_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
