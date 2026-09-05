"""Reliability-gated JEPA ranking for safe-capture v2 action chunks."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Mapping

import numpy as np

from .jepa_safe_capture_candidates import SafeCaptureCandidateBatch, SafeCaptureCandidateHistory
from .reliability import SafeCaptureReliabilityDecision, SafeCaptureReliabilityLedger


def _json_float(value: float) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _json_float_tuple(values: tuple[float, ...]) -> list[float | None]:
    return [_json_float(value) for value in values]


@dataclass(frozen=True)
class SafeCaptureRankerConfig:
    """Scoring constants frozen before development evaluation."""

    horizon_index: int = 2
    horizon_seconds: float = 0.30
    position_extent_m: float = 10.0
    max_cbf_correction_mps: float = 5.0
    obstacle_clearance_margin_m: float = 0.35
    inter_agent_clearance_margin_m: float = 0.15
    ttc_warning_seconds: float = 1.50
    target_weight: float = 1.0
    uncertainty_weight: float = 0.20
    clearance_weight: float = 1.0
    ttc_weight: float = 0.50
    visibility_weight: float = 0.25
    cbf_risk_weight: float = 0.75
    action_change_weight: float = 0.05
    nominal_anchor_margin_m: float = 1e-6
    # The tolerance is fixed before a paired block so CPU/CUDA score roundoff
    # cannot change a near-tied candidate decision.
    score_tie_tolerance_m: float = 5e-4
    # Fixed-point quantum used by the deterministic comparison profile.  Raw
    # scores remain in the trace for diagnosis; this key only affects ranking
    # and never bypasses the downstream CBF-QP.
    score_comparison_quantum_m: float = 5e-4
    # Fixed-point comparison is opt-in so legacy development protocols keep
    # their historical float ranking until a new protocol freezes the key.
    fixed_point_score_comparison: bool = False
    # Extra pre-registered band absorbs the largest observed CPU/CUDA score
    # drift.  It can only route a near-tie to nominal; CBF margins are
    # unchanged and every resulting action is still verified downstream.
    score_comparison_safety_band_m: float = 0.0
    # Optional P11 safeguards.  Zero/negative-infinity values preserve the
    # historical ranking behavior; a new protocol must opt into them.
    top_two_abstention_margin_m: float = 0.0
    minimum_predicted_clearance_m: float = float("-inf")
    candidate_hysteresis_margin_m: float = 0.0
    minimum_hold_steps: int = 0

    def __post_init__(self) -> None:
        if self.horizon_index < 0 or self.horizon_seconds <= 0.0 or self.position_extent_m <= 0.0:
            raise ValueError("horizon_index must be non-negative and horizon/extent must be positive.")
        if self.max_cbf_correction_mps <= 0.0 or self.ttc_warning_seconds <= 0.0:
            raise ValueError("max_cbf_correction_mps and ttc_warning_seconds must be positive.")
        if self.nominal_anchor_margin_m < 0.0:
            raise ValueError("nominal_anchor_margin_m must be non-negative.")
        if self.score_tie_tolerance_m < 0.0:
            raise ValueError("score_tie_tolerance_m must be non-negative.")
        if self.score_comparison_quantum_m < 0.0:
            raise ValueError("score_comparison_quantum_m must be non-negative.")
        if self.score_comparison_safety_band_m < 0.0:
            raise ValueError("score_comparison_safety_band_m must be non-negative.")
        if self.top_two_abstention_margin_m < 0.0 or self.candidate_hysteresis_margin_m < 0.0:
            raise ValueError("P11 ranking margins must be non-negative.")
        if self.minimum_hold_steps < 0:
            raise ValueError("minimum_hold_steps must be non-negative.")
        weights = (
            self.target_weight,
            self.uncertainty_weight,
            self.clearance_weight,
            self.ttc_weight,
            self.visibility_weight,
            self.cbf_risk_weight,
            self.action_change_weight,
        )
        if any(float(weight) < 0.0 for weight in weights):
            raise ValueError("Ranker weights must be non-negative.")


@dataclass(frozen=True)
class SafeCaptureRankingTrace:
    """JSON-friendly diagnostics for one ranking decision."""

    candidate_labels: tuple[str, ...]
    valid_mask: tuple[bool, ...]
    eligible_mask: tuple[bool, ...]
    scores: tuple[float, ...]
    target_cost_m: tuple[float, ...]
    uncertainty_cost_m: tuple[float, ...]
    clearance_cost_m: tuple[float, ...]
    ttc_cost: tuple[float, ...]
    visibility_cost: tuple[float, ...]
    cbf_risk_cost: tuple[float, ...]
    action_change_cost_mps: tuple[float, ...]
    predicted_min_clearance_m: tuple[float, ...]
    raw_predicted_min_clearance_m: tuple[float, ...]
    calibration_offset_m: tuple[float, ...]
    predicted_min_ttc_s: tuple[float, ...]
    predicted_uncertainty: tuple[float, ...]
    predicted_visibility: tuple[float, ...]
    predicted_cbf_risk: tuple[float, ...]
    ledger_states: tuple[str, ...]
    ledger_credits: tuple[float, ...]
    ledger_keys: tuple[str, ...]
    ledger_fallback_reasons: tuple[str | None, ...]
    candidate_rejection_reasons: tuple[tuple[str, ...], ...]
    selected_index: int
    execution_mode: str
    fallback_reason: str | None
    prediction_fault_fields: tuple[str, ...] = ()
    top_two_margin_m: float = float("inf")
    top_two_margin_comparison_m: float = float("inf")
    top_two_abstention_limit_m: float = float("inf")
    score_comparison_keys: tuple[int | None, ...] = ()
    fixed_point_score_comparison: bool = False
    score_comparison_quantum_m: float = 0.0
    candidate_order: tuple[int, ...] = ()
    rank_abstention_reason: str | None = None
    hysteresis_applied: bool = False
    hold_steps_remaining: int = 0
    # Runtime diagnostics are intentionally separate from ranking decisions.
    # They are measured at inference time and must never affect selection.
    jepa_inference_latency_ms: float = 0.0
    ledger_route_latency_ms: float = 0.0
    ranker_compute_latency_ms: float = 0.0
    rank_total_latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_labels": list(self.candidate_labels),
            "valid_mask": list(self.valid_mask),
            "eligible_mask": list(self.eligible_mask),
            "scores": _json_float_tuple(self.scores),
            "target_cost_m": _json_float_tuple(self.target_cost_m),
            "uncertainty_cost_m": _json_float_tuple(self.uncertainty_cost_m),
            "clearance_cost_m": _json_float_tuple(self.clearance_cost_m),
            "ttc_cost": _json_float_tuple(self.ttc_cost),
            "visibility_cost": _json_float_tuple(self.visibility_cost),
            "cbf_risk_cost": _json_float_tuple(self.cbf_risk_cost),
            "action_change_cost_mps": _json_float_tuple(self.action_change_cost_mps),
            "predicted_min_clearance_m": _json_float_tuple(self.predicted_min_clearance_m),
            "raw_predicted_min_clearance_m": _json_float_tuple(self.raw_predicted_min_clearance_m),
            "calibration_offset_m": _json_float_tuple(self.calibration_offset_m),
            "predicted_min_ttc_s": _json_float_tuple(self.predicted_min_ttc_s),
            "predicted_uncertainty": _json_float_tuple(self.predicted_uncertainty),
            "predicted_visibility": _json_float_tuple(self.predicted_visibility),
            "predicted_cbf_risk": _json_float_tuple(self.predicted_cbf_risk),
            "ledger_states": list(self.ledger_states),
            "ledger_credits": list(self.ledger_credits),
            "ledger_keys": list(self.ledger_keys),
            "ledger_fallback_reasons": list(self.ledger_fallback_reasons),
            "candidate_rejection_reasons": [list(value) for value in self.candidate_rejection_reasons],
            "selected_index": int(self.selected_index),
            "execution_mode": self.execution_mode,
            "fallback_reason": self.fallback_reason,
            "prediction_fault_fields": list(self.prediction_fault_fields),
            "top_two_margin_m": _json_float(self.top_two_margin_m),
            "top_two_margin_comparison_m": _json_float(self.top_two_margin_comparison_m),
            "top_two_abstention_limit_m": _json_float(self.top_two_abstention_limit_m),
            "score_comparison_keys": [
                None if value is None else int(value) for value in self.score_comparison_keys
            ],
            "fixed_point_score_comparison": bool(self.fixed_point_score_comparison),
            "score_comparison_quantum_m": float(self.score_comparison_quantum_m),
            "candidate_order": [int(value) for value in self.candidate_order],
            "rank_abstention_reason": self.rank_abstention_reason,
            "hysteresis_applied": bool(self.hysteresis_applied),
            "hold_steps_remaining": int(self.hold_steps_remaining),
            "jepa_inference_latency_ms": float(self.jepa_inference_latency_ms),
            "ledger_route_latency_ms": float(self.ledger_route_latency_ms),
            "ranker_compute_latency_ms": float(self.ranker_compute_latency_ms),
            "rank_total_latency_ms": float(self.rank_total_latency_ms),
        }


@dataclass(frozen=True)
class SafeCaptureRankingResult:
    """Selected desired action/chunk and its auditable decision trace."""

    selected_index: int
    selected_action: np.ndarray
    selected_chunk: np.ndarray
    execution_mode: str
    fallback_reason: str | None
    trace: SafeCaptureRankingTrace


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _minimum_pairwise_distance(positions: np.ndarray) -> float:
    if positions.shape[0] < 2:
        return float("inf")
    distances = [float(np.linalg.norm(positions[first] - positions[second])) for first in range(positions.shape[0]) for second in range(first + 1, positions.shape[0])]
    return min(distances) if distances else float("inf")


def _candidate_specific_separation(costs: np.ndarray) -> np.ndarray:
    """Return each candidate's distance to its nearest cost competitor.

    The separation is a candidate-level confidence signal.  A group-level
    top-two margin must not be copied to every candidate, because that makes
    the reliability ledger unable to distinguish the selected candidate from
    a clearly inferior alternative.  Non-finite costs are ignored and an
    isolated finite candidate receives zero separation.
    """

    values = np.asarray(costs, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Candidate costs must be a one-dimensional array.")
    result = np.zeros(values.shape, dtype=np.float64)
    finite = np.isfinite(values)
    finite_indices = np.flatnonzero(finite)
    if finite_indices.size < 2:
        return result
    finite_values = values[finite_indices]
    pairwise = np.abs(finite_values[:, None] - finite_values[None, :])
    np.fill_diagonal(pairwise, np.inf)
    result[finite_indices] = np.min(pairwise, axis=1)
    return result


def _conservative_margin_for_comparison(margin_m: float, quantum_m: float) -> float:
    """Return a deterministic lower-rounded margin for abstention decisions."""

    margin = float(margin_m)
    quantum = float(quantum_m)
    if not np.isfinite(margin) or quantum <= 0.0:
        return margin
    if margin <= 0.0:
        return 0.0
    # The lower rounding is intentional: a boundary value is routed to the
    # conservative nominal path on every device rather than selected as a
    # candidate because of BLAS/CUDA last-bit differences.
    return float(np.floor(margin / quantum) * quantum)


def _fixed_point_score_key(score_m: float, quantum_m: float) -> int | None:
    """Map one finite score to a deterministic integer comparison key.

    Scores are costs, so the smallest key wins.  The key uses round-half-up
    instead of Python's banker rounding and is kept separate from the raw
    floating-point score stored for diagnostics.  Non-finite scores are not
    eligible for ranking and therefore return ``None``.
    """

    score = float(score_m)
    quantum = float(quantum_m)
    if not np.isfinite(score):
        return None
    if not np.isfinite(quantum) or quantum <= 0.0:
        raise ValueError("A positive finite score comparison quantum is required.")
    scaled = score / quantum
    if not np.isfinite(scaled) or abs(scaled) >= float(np.iinfo(np.int64).max):
        raise ValueError("Score is outside the fixed-point comparison range.")
    # Scores are costs and are normally non-negative.  This branch keeps the
    # mapping symmetric if a future score contract permits negative costs.
    rounded = np.floor(scaled + 0.5) if scaled >= 0.0 else np.ceil(scaled - 0.5)
    return int(rounded)


def _fixed_point_score_keys(scores: np.ndarray, quantum_m: float) -> tuple[int | None, ...]:
    """Return JSON-friendly fixed-point keys for all candidate scores."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Scores must be a one-dimensional array.")
    return tuple(_fixed_point_score_key(value, quantum_m) for value in values)


class SafeCaptureJEPARanker:
    """Rank feasible chunks while enforcing v2 ledger abstention semantics.

    The ranker never calls a safety filter and never returns a claim that an
    action is safe.  ``execution_mode`` tells the downstream CBF/QP whether to
    use the selected candidate, nominal fallback, or its explicit hold path.
    """

    def __init__(
        self,
        history: SafeCaptureCandidateHistory,
        *,
        config: SafeCaptureRankerConfig | None = None,
        reliability_ledger: SafeCaptureReliabilityLedger | None = None,
        context_defaults: Mapping[str, Any] | None = None,
    ) -> None:
        self.history = history
        self.config = config or SafeCaptureRankerConfig()
        if self.config.horizon_index >= history.predictor.horizon_count:
            raise ValueError("Ranker horizon_index is outside the predictor output range.")
        self.reliability_ledger = reliability_ledger
        self.context_defaults = dict(context_defaults or {})

    def _context_base(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        visible = np.asarray(observation.get("target_visible", np.ones(self.history.defender_count)), dtype=np.float64)
        ages = np.asarray(
            observation.get("target_observation_age_steps", observation.get("message_age_steps", np.zeros(self.history.defender_count))),
            dtype=np.float64,
        )
        if visible.shape != (self.history.defender_count,) or ages.shape != (self.history.defender_count,):
            raise ValueError("target_visible and observation/message age must match defender count.")
        supplied_layout = "layout_signature" in self.context_defaults
        supplied_motion = "target_motion_mode" in self.context_defaults
        values = {
            "visibility_condition": float(np.mean(visible)),
            "observation_age_steps": float(np.mean(ages)),
            "obstacle_count": int(len(observation.get("obstacles", []))),
            "layout_signature": "unknown_layout",
            "target_motion_mode": "unknown_motion",
        }
        values.update(self.context_defaults)
        if self.reliability_ledger is not None and not (supplied_layout and supplied_motion):
            # A global calibration bucket is not permission to trust an
            # unlabelled deployment context. The caller must provide the
            # scenario provenance used to build the ledger.
            values["ood"] = True
        if not np.isfinite(visible).all() or not np.isfinite(ages).all():
            raise ValueError("Observation visibility/age values must be finite.")
        return values

    def _decision(
        self,
        base: Mapping[str, Any],
        *,
        minimum_clearance_m: float,
        pairwise_ttc_s: float,
        uncertainty: float,
        cbf_risk: float,
        candidate_separation_m: float,
    ) -> SafeCaptureReliabilityDecision:
        if self.reliability_ledger is None:
            return SafeCaptureReliabilityDecision("trusted", 1.0, 0, "ledger_disabled", None, False, False)
        context = dict(base)
        context.update(
            {
                "minimum_clearance_m": float(minimum_clearance_m),
                "pairwise_ttc_s": float(pairwise_ttc_s),
                "uncertainty": float(uncertainty),
                "cbf_risk": float(cbf_risk),
                "candidate_separation_m": float(candidate_separation_m),
            }
        )
        return self.reliability_ledger.decision(self.config.horizon_index, context)

    def _prediction_fault_result(
        self,
        candidate_batch: SafeCaptureCandidateBatch,
        chunks: np.ndarray,
        valid: np.ndarray,
        *,
        fault_fields: tuple[str, ...],
        jepa_latency_ms: float,
        ledger_latency_ms: float,
        rank_started_ns: int,
    ) -> SafeCaptureRankingResult:
        """Route a non-finite JEPA output to the explicit safe-hold path.

        This method intentionally does not invent replacement predictions or
        scores.  All candidates remain ineligible, trace values become JSON
        ``null`` through the evaluator's finite-value serializer, and the
        downstream evaluator sends the nominal-shaped hold request through the
        same Joint CBF-QP used by every other execution mode.
        """

        count = int(chunks.shape[0])
        reason = "non_finite_prediction"
        decision = SafeCaptureReliabilityDecision(
            "safe_hold",
            0.0,
            0,
            "invalid|non_finite_prediction",
            reason,
            False,
            False,
        )
        decisions = (decision,) * count
        rejection_reasons = tuple(
            tuple(dict.fromkeys((*candidate_batch.rejection_reasons[index], reason)))
            for index in range(count)
        )
        inf_values = (float("inf"),) * count
        nan_values = (float("nan"),) * count
        score_keys = (None,) * count
        elapsed_ms = (perf_counter_ns() - rank_started_ns) / 1_000_000.0
        trace = SafeCaptureRankingTrace(
            candidate_labels=tuple(candidate_batch.labels),
            valid_mask=tuple(bool(value) for value in valid),
            eligible_mask=(False,) * count,
            scores=inf_values,
            target_cost_m=inf_values,
            uncertainty_cost_m=inf_values,
            clearance_cost_m=inf_values,
            ttc_cost=inf_values,
            visibility_cost=inf_values,
            cbf_risk_cost=inf_values,
            action_change_cost_mps=inf_values,
            predicted_min_clearance_m=nan_values,
            raw_predicted_min_clearance_m=nan_values,
            calibration_offset_m=nan_values,
            predicted_min_ttc_s=nan_values,
            predicted_uncertainty=nan_values,
            predicted_visibility=nan_values,
            predicted_cbf_risk=nan_values,
            ledger_states=tuple(decision.state for decision in decisions),
            ledger_credits=tuple(float(decision.credit) for decision in decisions),
            ledger_keys=tuple(decision.key for decision in decisions),
            ledger_fallback_reasons=tuple(decision.fallback_reason for decision in decisions),
            candidate_rejection_reasons=rejection_reasons,
            selected_index=0,
            execution_mode="safe_hold",
            fallback_reason=reason,
            prediction_fault_fields=tuple(fault_fields),
            score_comparison_keys=score_keys,
            fixed_point_score_comparison=bool(self.config.fixed_point_score_comparison),
            score_comparison_quantum_m=float(self.config.score_comparison_quantum_m),
            jepa_inference_latency_ms=float(jepa_latency_ms),
            ledger_route_latency_ms=float(ledger_latency_ms),
            ranker_compute_latency_ms=float(max(elapsed_ms - jepa_latency_ms - ledger_latency_ms, 0.0)),
            rank_total_latency_ms=float(elapsed_ms),
        )
        return SafeCaptureRankingResult(
            selected_index=0,
            selected_action=chunks[0, 0].copy(),
            selected_chunk=chunks[0].copy(),
            execution_mode="safe_hold",
            fallback_reason=reason,
            trace=trace,
        )

    @staticmethod
    def _prediction_fault_fields(
        means: Any,
        stds: Any,
        auxiliary: Mapping[str, Any],
    ) -> tuple[str, ...]:
        """Return stable names for non-finite prediction heads."""

        fields: list[str] = []
        for name, value in (("target_displacement", means), ("target_uncertainty", stds)):
            if not np.isfinite(np.asarray(value)).all():
                fields.append(name)
        for name in sorted(auxiliary):
            if not np.isfinite(np.asarray(auxiliary[name])).all():
                fields.append(str(name))
        return tuple(fields)

    def rank(
        self,
        observation: Mapping[str, Any],
        candidate_batch: SafeCaptureCandidateBatch,
        *,
        previous_action: np.ndarray | None = None,
        previous_selected_index: int | None = None,
        hold_steps_remaining: int = 0,
    ) -> SafeCaptureRankingResult:
        rank_started_ns = perf_counter_ns()
        jepa_latency_ms = 0.0
        ledger_latency_ms = 0.0
        chunks = np.asarray(candidate_batch.chunks)
        if chunks.ndim != 4 or chunks.shape[0] != len(candidate_batch.labels) or chunks.shape[1] <= 0:
            raise ValueError("Candidate batch has an invalid chunk shape.")
        if chunks.shape[0] != 5 or chunks.shape[2:] != (self.history.defender_count, self.history.predictor.action_dim):
            raise ValueError("P4 requires exactly five [steps, defenders, action_dim] candidate chunks.")
        if not np.isfinite(chunks).all():
            raise ValueError("Candidate chunks must be finite before ranking.")
        if previous_selected_index is not None and not 0 <= int(previous_selected_index) < 5:
            raise ValueError("previous_selected_index must be in [0, 4].")
        if hold_steps_remaining < 0:
            raise ValueError("hold_steps_remaining must be non-negative.")
        valid = np.asarray(candidate_batch.valid_mask, dtype=bool)
        if valid.shape != (chunks.shape[0],):
            raise ValueError("Candidate valid_mask shape mismatch.")
        nominal = chunks[0, 0].copy()
        reference = nominal if previous_action is None else np.asarray(previous_action, dtype=np.float64)
        if reference.shape != nominal.shape or not np.isfinite(reference).all():
            raise ValueError("previous_action must be finite and match nominal action shape.")
        positions = np.asarray(observation.get("defender_positions"), dtype=np.float64)
        if positions.shape != nominal.shape or not np.isfinite(positions).all():
            raise ValueError("defender_positions must be finite and match action shape.")

        scores = np.full(5, np.inf, dtype=np.float64)
        target_cost = np.full(5, np.inf, dtype=np.float64)
        uncertainty_cost = np.full(5, np.inf, dtype=np.float64)
        clearance_cost = np.full(5, np.inf, dtype=np.float64)
        ttc_cost = np.full(5, np.inf, dtype=np.float64)
        visibility_cost = np.full(5, np.inf, dtype=np.float64)
        cbf_risk_cost = np.full(5, np.inf, dtype=np.float64)
        action_change_cost = np.full(5, np.inf, dtype=np.float64)
        min_clearance = np.full(5, np.nan, dtype=np.float64)
        raw_min_clearance = np.full(5, np.nan, dtype=np.float64)
        calibration_offset = np.full(5, np.nan, dtype=np.float64)
        min_ttc = np.full(5, np.nan, dtype=np.float64)
        uncertainty = np.full(5, np.nan, dtype=np.float64)
        visibility = np.full(5, np.nan, dtype=np.float64)
        cbf_risk = np.full(5, np.nan, dtype=np.float64)
        decisions: list[SafeCaptureReliabilityDecision] = [
            SafeCaptureReliabilityDecision("safe_hold", 0.0, 0, "not_evaluated", "invalid_candidate", False, False)
            for _ in range(5)
        ]
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size:
            jepa_started_ns = perf_counter_ns()
            try:
                means, stds, auxiliary = self.history.predict_candidates_multitask(
                    chunks[valid_indices, 0],
                    horizon_index=self.config.horizon_index,
                )
            except RuntimeError as error:
                if "non-finite" not in str(error).lower():
                    raise
                jepa_latency_ms = (perf_counter_ns() - jepa_started_ns) / 1_000_000.0
                return self._prediction_fault_result(
                    candidate_batch,
                    chunks,
                    valid,
                    fault_fields=("prediction_output",),
                    jepa_latency_ms=jepa_latency_ms,
                    ledger_latency_ms=ledger_latency_ms,
                    rank_started_ns=rank_started_ns,
                )
            jepa_latency_ms = (perf_counter_ns() - jepa_started_ns) / 1_000_000.0
            fault_fields = self._prediction_fault_fields(means, stds, auxiliary)
            if fault_fields:
                return self._prediction_fault_result(
                    candidate_batch,
                    chunks,
                    valid,
                    fault_fields=fault_fields,
                    jepa_latency_ms=jepa_latency_ms,
                    ledger_latency_ms=ledger_latency_ms,
                    rank_started_ns=rank_started_ns,
                )
            raw_obstacle = np.asarray(auxiliary["obstacle_clearance_lower_quantile"], dtype=np.float64) * self.config.position_extent_m
            raw_inter_agent = np.asarray(auxiliary["inter_agent_clearance_lower_quantile"], dtype=np.float64) * self.config.position_extent_m
            if self.reliability_ledger is not None:
                obstacle, inter_agent = self.reliability_ledger.calibrated_clearance_m(
                    self.config.horizon_index,
                    raw_obstacle,
                    raw_inter_agent,
                )
            else:
                obstacle, inter_agent = raw_obstacle, raw_inter_agent
            ttc = np.asarray(auxiliary["pairwise_ttc"], dtype=np.float64)
            visibility_probability = _sigmoid(np.asarray(auxiliary["target_visibility_logit"], dtype=np.float64))
            intervention_probability = _sigmoid(np.asarray(auxiliary["cbf_intervention_logit"], dtype=np.float64))
            correction = np.asarray(auxiliary["cbf_correction"], dtype=np.float64)
            qp_probability = _sigmoid(np.asarray(auxiliary["cbf_qp_feasibility_logit"], dtype=np.float64))
            base_context = self._context_base(observation)
            predicted_target = positions[None, :, :] + means.astype(np.float64) * self.config.position_extent_m
            future_defenders = positions[None, :, :] + chunks[valid_indices, 0].astype(np.float64) * self.config.horizon_seconds
            distances = np.linalg.norm(predicted_target - future_defenders, axis=2)
            raw_target_cost = np.mean(distances, axis=1)
            separation = _candidate_specific_separation(raw_target_cost)
            for local, candidate_index in enumerate(valid_indices):
                min_obstacle = float(np.min(obstacle[local]))
                min_inter = float(np.min(inter_agent[local]))
                raw_min = float(min(float(np.min(raw_obstacle[local])), float(np.min(raw_inter_agent[local]))))
                raw_min_clearance[candidate_index] = raw_min
                min_clearance[candidate_index] = min(min_obstacle, min_inter)
                calibration_offset[candidate_index] = min_clearance[candidate_index] - raw_min
                min_ttc[candidate_index] = float(np.min(ttc[local]))
                uncertainty[candidate_index] = float(np.mean(stds[local]))
                visibility[candidate_index] = float(np.mean(visibility_probability[local]))
                cbf_risk[candidate_index] = float(
                    np.max(
                        np.maximum.reduce(
                            [
                                intervention_probability[local],
                                np.clip(correction[local] / self.config.max_cbf_correction_mps, 0.0, 1.0),
                                1.0 - qp_probability[local],
                            ]
                        )
                    )
                )
                target_cost[candidate_index] = float(raw_target_cost[local])
                uncertainty_cost[candidate_index] = uncertainty[candidate_index] * self.config.position_extent_m
                clearance_cost[candidate_index] = float(
                    max(self.config.obstacle_clearance_margin_m - min_obstacle, 0.0)
                    + max(self.config.inter_agent_clearance_margin_m - min_inter, 0.0)
                )
                ttc_cost[candidate_index] = max(self.config.ttc_warning_seconds - min_ttc[candidate_index], 0.0) / self.config.ttc_warning_seconds
                visibility_cost[candidate_index] = 1.0 - visibility[candidate_index]
                cbf_risk_cost[candidate_index] = cbf_risk[candidate_index]
                action_change_cost[candidate_index] = float(np.mean(np.linalg.norm(chunks[candidate_index, 0] - reference, axis=1)))
                ledger_started_ns = perf_counter_ns()
                decisions[candidate_index] = self._decision(
                    base_context,
                    minimum_clearance_m=min_clearance[candidate_index],
                    pairwise_ttc_s=min_ttc[candidate_index],
                    uncertainty=uncertainty[candidate_index],
                    cbf_risk=cbf_risk[candidate_index],
                    candidate_separation_m=float(separation[local]),
                )
                ledger_latency_ms += (perf_counter_ns() - ledger_started_ns) / 1_000_000.0
                scores[candidate_index] = (
                    self.config.target_weight * target_cost[candidate_index]
                    + self.config.uncertainty_weight * uncertainty_cost[candidate_index]
                    + self.config.clearance_weight * clearance_cost[candidate_index]
                    + self.config.ttc_weight * ttc_cost[candidate_index]
                    + self.config.visibility_weight * visibility_cost[candidate_index]
                    + self.config.cbf_risk_weight * cbf_risk_cost[candidate_index]
                    + self.config.action_change_weight * action_change_cost[candidate_index]
                )

        nominal_decision = decisions[0]
        eligible = valid & np.asarray([decision.state == "trusted" for decision in decisions], dtype=bool)
        # Predicted safety quantities are only a ranking gate.  The CBF still
        # verifies every selected nominal/candidate action before execution.
        clearance_gate = np.isfinite(min_clearance) & (
            min_clearance >= float(self.config.minimum_predicted_clearance_m)
        )
        eligible &= clearance_gate
        if self.config.fixed_point_score_comparison:
            score_keys = _fixed_point_score_keys(
                scores,
                self.config.score_comparison_quantum_m,
            )
            eligible &= np.asarray([key is not None for key in score_keys], dtype=bool)
        else:
            # Keep keys in the trace even for legacy profiles, but do not let
            # this diagnostic field alter their historical float decisions.
            score_keys = (
                _fixed_point_score_keys(scores, self.config.score_comparison_quantum_m)
                if self.config.score_comparison_quantum_m > 0.0
                else tuple(None for _ in scores)
            )
        execution_mode = "trusted"
        fallback_reason: str | None = None
        rank_abstention_reason: str | None = None
        hysteresis_applied = False
        top_two_margin = float("inf")
        top_two_margin_comparison = float("inf")
        top_two_abstention_limit = float("inf")
        candidate_order: tuple[int, ...] = ()
        nominal_anchor_selected = False
        remaining_hold = max(int(hold_steps_remaining) - 1, 0)
        selected_index = 0
        if not bool(valid[0]):
            execution_mode = "safe_hold"
            fallback_reason = "nominal_infeasible"
        elif nominal_decision.state == "safe_hold":
            execution_mode = "safe_hold"
            fallback_reason = nominal_decision.fallback_reason
        elif nominal_decision.state == "fallback_nominal":
            execution_mode = "fallback_nominal"
            fallback_reason = nominal_decision.fallback_reason
        else:
            # A non-trusted alternative is never allowed to displace nominal.
            if self.config.fixed_point_score_comparison:
                eligible[0] = bool(
                    valid[0]
                    and nominal_decision.state == "trusted"
                    and clearance_gate[0]
                    and score_keys[0] is not None
                )
            else:
                eligible[0] = True
            trusted_indices = np.flatnonzero(eligible)
            if trusted_indices.size:
                if self.config.fixed_point_score_comparison:
                    ordered = np.asarray(
                        sorted(
                            (int(index) for index in trusted_indices),
                            key=lambda index: (int(score_keys[index]), index),
                        ),
                        dtype=np.int64,
                    )
                else:
                    ordered = trusted_indices[np.argsort(scores[trusted_indices], kind="mergesort")]
                candidate_order = tuple(int(index) for index in ordered)
                best_score = float(scores[ordered[0]])
                best_key = score_keys[int(ordered[0])]
                if ordered.size > 1:
                    top_two_margin = float(max(scores[ordered[1]] - best_score, 0.0))
                    if self.config.fixed_point_score_comparison:
                        second_key = score_keys[int(ordered[1])]
                        top_two_margin_comparison = float(
                            max(int(second_key) - int(best_key), 0)
                            * self.config.score_comparison_quantum_m
                        )
                    else:
                        top_two_margin_comparison = _conservative_margin_for_comparison(
                            top_two_margin,
                            self.config.score_comparison_quantum_m,
                        )
                if self.config.fixed_point_score_comparison:
                    # The comparison width is pre-registered in metres, then
                    # rounded upward to integer buckets.  This collapses a
                    # one-bucket CPU/CUDA boundary drift into a deterministic
                    # candidate-index tie without changing CBF geometry.
                    tie_width_m = (
                        self.config.score_tie_tolerance_m
                        + self.config.score_comparison_safety_band_m
                    )
                    tie_units = int(
                        np.ceil(
                            tie_width_m / self.config.score_comparison_quantum_m
                        )
                    ) if tie_width_m > 0.0 else 0
                    tied = np.asarray(
                        [
                            int(index)
                            for index in trusted_indices
                            if int(score_keys[int(index)]) <= int(best_key) + tie_units
                        ],
                        dtype=np.int64,
                    )
                else:
                    tied = trusted_indices[
                        scores[trusted_indices] <= best_score + self.config.score_tie_tolerance_m
                    ]
                # Candidate index is the deterministic secondary key.  This
                # prevents CPU/CUDA roundoff from changing a near-tied action.
                best = int(np.min(tied))
                nominal_anchor_selected = bool(best == 0 and int(ordered[0]) != 0)
                if best != 0:
                    if self.config.fixed_point_score_comparison:
                        best = 0 if int(score_keys[0]) <= int(score_keys[best]) + tie_units else best
                    elif scores[0] <= scores[best] + self.config.nominal_anchor_margin_m:
                        best = 0
                if best != 0 and (
                    self.config.top_two_abstention_margin_m > 0.0
                    # Treat the protocol's score tie tolerance as a
                    # deterministic numerical band around the abstention
                    # boundary.  This prevents CPU/CUDA roundoff from
                    # selecting different actions at nearly identical margins.
                    and top_two_margin_comparison
                    <= (
                        self.config.top_two_abstention_margin_m
                        + self.config.score_tie_tolerance_m
                        + self.config.score_comparison_safety_band_m
                    )
                    and bool(eligible[0])
                ):
                    best = 0
                    execution_mode = "fallback_nominal"
                    rank_abstention_reason = "top_two_margin_abstention"
                    fallback_reason = rank_abstention_reason
                elif previous_selected_index is not None and int(previous_selected_index) != 0:
                    previous = int(previous_selected_index)
                    previous_eligible = bool(eligible[previous]) and np.isfinite(scores[previous])
                    if previous_eligible and best != previous:
                        force_hold = hold_steps_remaining > 0
                        if self.config.fixed_point_score_comparison:
                            hysteresis_units = int(
                                np.ceil(
                                    self.config.candidate_hysteresis_margin_m
                                    / self.config.score_comparison_quantum_m
                                )
                            ) if self.config.candidate_hysteresis_margin_m > 0.0 else 0
                            within_hysteresis = (
                                int(score_keys[previous]) - int(score_keys[best])
                                <= hysteresis_units
                            )
                        else:
                            score_gap = float(scores[previous] - best_score)
                            within_hysteresis = score_gap <= self.config.candidate_hysteresis_margin_m
                        if force_hold or within_hysteresis:
                            best = previous
                            hysteresis_applied = True
                            remaining_hold = max(remaining_hold, self.config.minimum_hold_steps - 1)
                selected_index = best
                if nominal_anchor_selected and execution_mode == "trusted":
                    # Nominal was selected by the registered tie/anchor band,
                    # not because it was the discrete best score.  Expose one
                    # deterministic fallback state across CPU/CUDA backends.
                    execution_mode = "fallback_nominal"
                    rank_abstention_reason = "nominal_anchor_tie"
                    fallback_reason = rank_abstention_reason
            else:
                execution_mode = "fallback_nominal"
                fallback_reason = "no_trusted_candidate"
        if execution_mode != "trusted":
            selected_index = 0
        trace = SafeCaptureRankingTrace(
            candidate_labels=tuple(candidate_batch.labels),
            valid_mask=tuple(bool(value) for value in valid),
            eligible_mask=tuple(bool(value) for value in eligible),
            scores=tuple(float(value) for value in scores),
            target_cost_m=tuple(float(value) for value in target_cost),
            uncertainty_cost_m=tuple(float(value) for value in uncertainty_cost),
            clearance_cost_m=tuple(float(value) for value in clearance_cost),
            ttc_cost=tuple(float(value) for value in ttc_cost),
            visibility_cost=tuple(float(value) for value in visibility_cost),
            cbf_risk_cost=tuple(float(value) for value in cbf_risk_cost),
            action_change_cost_mps=tuple(float(value) for value in action_change_cost),
            predicted_min_clearance_m=tuple(float(value) if np.isfinite(value) else float("nan") for value in min_clearance),
            raw_predicted_min_clearance_m=tuple(float(value) if np.isfinite(value) else float("nan") for value in raw_min_clearance),
            calibration_offset_m=tuple(float(value) if np.isfinite(value) else float("nan") for value in calibration_offset),
            predicted_min_ttc_s=tuple(float(value) if np.isfinite(value) else float("nan") for value in min_ttc),
            predicted_uncertainty=tuple(float(value) if np.isfinite(value) else float("nan") for value in uncertainty),
            predicted_visibility=tuple(float(value) if np.isfinite(value) else float("nan") for value in visibility),
            predicted_cbf_risk=tuple(float(value) if np.isfinite(value) else float("nan") for value in cbf_risk),
            ledger_states=tuple(decision.state for decision in decisions),
            ledger_credits=tuple(float(decision.credit) for decision in decisions),
            ledger_keys=tuple(decision.key for decision in decisions),
            ledger_fallback_reasons=tuple(decision.fallback_reason for decision in decisions),
            candidate_rejection_reasons=tuple(
                tuple(str(reason) for reason in reasons)
                for reasons in candidate_batch.rejection_reasons
            ),
            selected_index=selected_index,
            execution_mode=execution_mode,
            fallback_reason=fallback_reason,
            top_two_margin_m=top_two_margin,
            top_two_margin_comparison_m=top_two_margin_comparison,
            top_two_abstention_limit_m=(
                float(
                    self.config.top_two_abstention_margin_m
                    + self.config.score_tie_tolerance_m
                    + self.config.score_comparison_safety_band_m
                )
                if self.config.top_two_abstention_margin_m > 0.0
                else top_two_abstention_limit
            ),
            score_comparison_keys=score_keys,
            fixed_point_score_comparison=bool(self.config.fixed_point_score_comparison),
            score_comparison_quantum_m=float(self.config.score_comparison_quantum_m),
            candidate_order=candidate_order,
            rank_abstention_reason=rank_abstention_reason,
            hysteresis_applied=hysteresis_applied,
            hold_steps_remaining=remaining_hold,
            jepa_inference_latency_ms=float(jepa_latency_ms),
            ledger_route_latency_ms=float(ledger_latency_ms),
            ranker_compute_latency_ms=float(
                max(
                    (perf_counter_ns() - rank_started_ns) / 1_000_000.0
                    - jepa_latency_ms
                    - ledger_latency_ms,
                    0.0,
                )
            ),
            rank_total_latency_ms=float((perf_counter_ns() - rank_started_ns) / 1_000_000.0),
        )
        return SafeCaptureRankingResult(
            selected_index=selected_index,
            selected_action=chunks[selected_index, 0].copy(),
            selected_chunk=chunks[selected_index].copy(),
            execution_mode=execution_mode,
            fallback_reason=fallback_reason,
            trace=trace,
        )
