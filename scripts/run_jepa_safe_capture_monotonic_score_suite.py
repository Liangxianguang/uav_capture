"""Run a synthetic monotonicity audit for the safe-capture candidate ranker.

The suite is deliberately independent of episode outcomes.  It checks score
orientation, safety-gate precedence, and deterministic tie handling using a
controllable action-conditioned history stub.  It never changes an online
ranker configuration or consumes locked-test data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from encirclement3d.jepa_safe_capture_candidates import CANDIDATE_LABELS, SafeCaptureCandidateBatch
from encirclement3d.jepa_safe_capture_ranker import SafeCaptureJEPARanker, SafeCaptureRankerConfig


DEFENDER_COUNT = 2
CANDIDATE_COUNT = len(CANDIDATE_LABELS)


def _observation() -> dict[str, Any]:
    positions = np.zeros((DEFENDER_COUNT, 3), dtype=np.float64)
    positions[1, 1] = 1.0
    return {
        "defender_positions": positions,
        "target_belief_positions": np.full((DEFENDER_COUNT, 3), 2.0, dtype=np.float64),
        "target_belief_velocities": np.zeros((DEFENDER_COUNT, 3), dtype=np.float64),
        "target_visible": np.ones(DEFENDER_COUNT, dtype=bool),
        "target_observation_age_steps": np.zeros(DEFENDER_COUNT, dtype=np.float64),
        "obstacles": [{"shape": "cylinder"}],
    }


def _batch() -> SafeCaptureCandidateBatch:
    chunks = np.zeros((CANDIDATE_COUNT, 3, DEFENDER_COUNT, 3), dtype=np.float64)
    return SafeCaptureCandidateBatch(
        chunks=chunks,
        labels=CANDIDATE_LABELS,
        valid_mask=np.ones(CANDIDATE_COUNT, dtype=bool),
        rejection_reasons=tuple(() for _ in range(CANDIDATE_COUNT)),
    )


class _SyntheticHistory:
    """Deterministic model stub with one row per candidate."""

    defender_count = DEFENDER_COUNT
    predictor = SimpleNamespace(action_dim=3, input_dim=63, horizon_count=4)

    def __init__(self, fields: Mapping[str, np.ndarray]) -> None:
        self.fields = {name: np.asarray(value, dtype=np.float32) for name, value in fields.items()}

    def predict_candidates_multitask(self, actions: np.ndarray, *, horizon_index: int):
        if horizon_index != 2:
            raise AssertionError(f"suite expects horizon 2, got {horizon_index}")
        count = int(actions.shape[0])
        means = self.fields["means"][:count]
        std = self.fields["std"][:count]
        auxiliary = {
            name: value[:count]
            for name, value in self.fields.items()
            if name not in {"means", "std"}
        }
        return means, std, auxiliary


def _default_fields() -> dict[str, np.ndarray]:
    means = np.zeros((CANDIDATE_COUNT, DEFENDER_COUNT, 3), dtype=np.float32)
    means[..., 0] = 0.40
    return {
        "means": means,
        "std": np.full((CANDIDATE_COUNT, DEFENDER_COUNT, 3), 0.01, dtype=np.float32),
        "obstacle_clearance_lower_quantile": np.full((CANDIDATE_COUNT, DEFENDER_COUNT), 1.0, dtype=np.float32),
        "inter_agent_clearance_lower_quantile": np.full((CANDIDATE_COUNT, DEFENDER_COUNT), 1.0, dtype=np.float32),
        "pairwise_ttc": np.full((CANDIDATE_COUNT, DEFENDER_COUNT), 10.0, dtype=np.float32),
        "target_visibility_logit": np.full((CANDIDATE_COUNT, DEFENDER_COUNT), 10.0, dtype=np.float32),
        "cbf_intervention_logit": np.full((CANDIDATE_COUNT, DEFENDER_COUNT), -10.0, dtype=np.float32),
        "cbf_correction": np.zeros((CANDIDATE_COUNT, DEFENDER_COUNT), dtype=np.float32),
        "cbf_qp_feasibility_logit": np.full((CANDIDATE_COUNT, DEFENDER_COUNT), 10.0, dtype=np.float32),
    }


@dataclass(frozen=True)
class ScoreCase:
    name: str
    expected_selected_index: int
    expected_reason: str | None = None
    minimum_predicted_clearance_m: float = float("-inf")
    fixed_point_score_comparison: bool = False
    score_comparison_quantum_m: float = 0.0005
    score_tie_tolerance_m: float = 0.0005


def _case_fields(name: str) -> dict[str, np.ndarray]:
    fields = _default_fields()
    if name == "task_progress":
        fields["means"][0, :, 0] = 0.40
        fields["means"][1, :, 0] = 0.05
    elif name == "uncertainty":
        fields["std"][0] = 0.30
        fields["std"][1] = 0.01
    elif name == "clearance_gate":
        fields["means"][1, :, 0] = 0.01
        fields["obstacle_clearance_lower_quantile"][1] = 0.01
    elif name == "visibility":
        fields["target_visibility_logit"][0] = -10.0
        fields["target_visibility_logit"][1] = 10.0
    elif name == "ttc":
        fields["pairwise_ttc"][0] = 0.20
        fields["pairwise_ttc"][1] = 10.0
    elif name == "cbf_risk":
        fields["cbf_intervention_logit"][0] = 5.0
        fields["cbf_correction"][0] = 5.0
        fields["cbf_qp_feasibility_logit"][0] = -5.0
    elif name == "fixed_point_tie":
        pass
    else:
        raise ValueError(f"Unknown score case: {name}")
    return fields


def _cases() -> tuple[ScoreCase, ...]:
    return (
        ScoreCase("task_progress", expected_selected_index=1),
        ScoreCase("uncertainty", expected_selected_index=1),
        ScoreCase(
            "clearance_gate",
            expected_selected_index=0,
            expected_reason=None,
            minimum_predicted_clearance_m=0.15,
        ),
        ScoreCase("visibility", expected_selected_index=1),
        ScoreCase("ttc", expected_selected_index=1),
        ScoreCase("cbf_risk", expected_selected_index=1),
        ScoreCase(
            "fixed_point_tie",
            expected_selected_index=0,
            fixed_point_score_comparison=True,
            score_comparison_quantum_m=0.004,
            score_tie_tolerance_m=0.004,
        ),
    )


def _finite_or_none(value: float) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def evaluate_case(case: ScoreCase) -> dict[str, Any]:
    config = SafeCaptureRankerConfig(
        minimum_predicted_clearance_m=case.minimum_predicted_clearance_m,
        fixed_point_score_comparison=case.fixed_point_score_comparison,
        score_comparison_quantum_m=case.score_comparison_quantum_m,
        score_tie_tolerance_m=case.score_tie_tolerance_m,
    )
    ranker = SafeCaptureJEPARanker(_SyntheticHistory(_case_fields(case.name)), config=config)
    result = ranker.rank(_observation(), _batch())
    trace = result.trace.as_dict()
    passed = result.selected_index == case.expected_selected_index
    if case.expected_reason is not None:
        passed = passed and result.fallback_reason == case.expected_reason
    return {
        "name": case.name,
        "passed": bool(passed),
        "expected_selected_index": case.expected_selected_index,
        "selected_index": int(result.selected_index),
        "execution_mode": result.execution_mode,
        "fallback_reason": result.fallback_reason,
        "eligible_mask": list(trace["eligible_mask"]),
        "scores": list(trace["scores"]),
        "target_cost_m": list(trace["target_cost_m"]),
        "uncertainty_cost_m": list(trace["uncertainty_cost_m"]),
        "clearance_cost_m": list(trace["clearance_cost_m"]),
        "ttc_cost": list(trace["ttc_cost"]),
        "visibility_cost": list(trace["visibility_cost"]),
        "cbf_risk_cost": list(trace["cbf_risk_cost"]),
        "predicted_min_clearance_m": list(trace["predicted_min_clearance_m"]),
        "candidate_order": list(trace["candidate_order"]),
        "rank_abstention_reason": trace["rank_abstention_reason"],
        "config": config.as_dict() if hasattr(config, "as_dict") else {
            "minimum_predicted_clearance_m": _finite_or_none(config.minimum_predicted_clearance_m),
            "fixed_point_score_comparison": config.fixed_point_score_comparison,
            "score_comparison_quantum_m": config.score_comparison_quantum_m,
            "score_tie_tolerance_m": config.score_tie_tolerance_m,
        },
    }


def run_suite() -> dict[str, Any]:
    cases = [evaluate_case(case) for case in _cases()]
    return {
        "stage": "WP2_monotonic_score_suite",
        "development_only": True,
        "locked_test_opened": False,
        "candidate_contract": {
            "candidate_count": CANDIDATE_COUNT,
            "chunk_length_steps": 3,
            "jepa_role": "candidate_trajectory_evaluator_only",
        },
        "cases": cases,
        "all_cases_passed": all(bool(case["passed"]) for case in cases),
        "interpretation": {
            "score_is_cost": True,
            "safety_gate_precedes_task_progress": True,
            "synthetic_cases_are_not_task_performance_evidence": True,
        },
    }


def write_report(output_dir: Path, tensorboard_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    if tensorboard_dir.exists() and any(tensorboard_dir.iterdir()):
        raise FileExistsError(tensorboard_dir)
    result = run_suite()
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "monotonic_score_suite.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Synthetic monotonic score suite",
        "",
        "`development_only=true`; `locked_test_opened=false`. This is a ranker contract audit, not episode evidence.",
        "",
        "| Case | Expected index | Selected index | Execution mode | Eligible mask | Pass |",
        "|---|---:|---:|---|---|---:|",
    ]
    for case in result["cases"]:
        lines.append(
            f"| {case['name']} | {case['expected_selected_index']} | {case['selected_index']} | "
            f"{case['execution_mode']} | `{case['eligible_mask']}` | {case['passed']} |"
        )
    lines += [
        "",
        "## Gate",
        "",
        f"all_cases_passed: `{result['all_cases_passed']}`",
        "",
        "The suite checks cost orientation, safety eligibility before task progress, auxiliary-risk directions, and fixed-point tie handling. It does not establish safe-capture improvement.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with SummaryWriter(log_dir=str(tensorboard_dir), flush_secs=1) as writer:
        writer.add_text("Config/suite", json.dumps(result["candidate_contract"], indent=2), 0)
        writer.add_text("Interpretation", json.dumps(result["interpretation"], indent=2), 0)
        for index, case in enumerate(result["cases"]):
            writer.add_scalar(f"Cases/{case['name']}/passed", float(case["passed"]), index)
            writer.add_scalar(f"Cases/{case['name']}/selected_index", float(case["selected_index"]), index)
            writer.add_scalar(
                f"Cases/{case['name']}/eligible_count",
                float(sum(bool(value) for value in case["eligible_mask"])),
                index,
            )
        writer.add_scalar("Gates/all_cases_passed", float(result["all_cases_passed"]), 0)
    result["tensorboard"] = {
        "logdir": str(tensorboard_dir),
        "event_files": sorted(path.name for path in tensorboard_dir.glob("events.out.tfevents.*")),
    }
    (output_dir / "monotonic_score_suite.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_report(args.output_dir.resolve(), args.tensorboard_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
