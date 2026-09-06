"""Runtime adapter from obstacle routes to the existing JEPA/CBF stack.

The route generator deliberately owns geometry, while the historical ranker
continues to consume ``SafeCaptureCandidateBatch``.  This adapter bridges the
two contracts without changing legacy candidate profiles.  CBF probes are
read-only counterfactuals; the caller must still pass the selected request
through the normal Joint CBF ``filter`` method before execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .jepa_safe_capture_candidates import SafeCaptureCandidateBatch
from .obstacle_route_candidates import ObstacleRouteBatch


@dataclass(frozen=True)
class CBFRouteCounterfactual:
    """Serializable result of one read-only requested-action CBF probe."""

    label: str
    route_id: str | None
    requested_action: np.ndarray
    verified_feasible: bool
    infeasible: bool
    timed_out: bool
    fallback_mode: str
    solver_status: str
    minimum_constraint_value: float
    action_correction_norm: float
    active_constraints: tuple[str, ...]

    def __post_init__(self) -> None:
        action = np.asarray(self.requested_action, dtype=np.float64)
        if action.ndim != 2 or action.shape[1:] != (3,) or not np.isfinite(action).all():
            raise ValueError("Counterfactual requested_action must be finite with shape [defenders, 3].")
        object.__setattr__(self, "requested_action", action.copy())
        if not np.isfinite(self.minimum_constraint_value) and not np.isneginf(self.minimum_constraint_value):
            raise ValueError("minimum_constraint_value must be finite or -inf.")
        if not np.isfinite(self.action_correction_norm) and not np.isposinf(self.action_correction_norm):
            raise ValueError("action_correction_norm must be finite or +inf.")

    @property
    def accepted(self) -> bool:
        return bool(
            self.verified_feasible
            and not self.infeasible
            and not self.timed_out
            and self.fallback_mode == "none"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "route_id": self.route_id,
            "requested_action": self.requested_action.tolist(),
            "verified_feasible": bool(self.verified_feasible),
            "accepted": bool(self.accepted),
            "infeasible": bool(self.infeasible),
            "timed_out": bool(self.timed_out),
            "fallback_mode": self.fallback_mode,
            "solver_status": self.solver_status,
            "minimum_constraint_value": float(self.minimum_constraint_value),
            "action_correction_norm": float(self.action_correction_norm),
            "active_constraints": list(self.active_constraints),
        }


@dataclass(frozen=True)
class ObstacleRouteRuntimeBatch:
    """Route metadata, ranker-compatible batch, and optional CBF probes."""

    route_batch: ObstacleRouteBatch
    candidate_batch: SafeCaptureCandidateBatch
    cbf_counterfactuals: tuple[CBFRouteCounterfactual | None, ...]

    def __post_init__(self) -> None:
        if len(self.cbf_counterfactuals) != len(self.route_batch.candidates):
            raise ValueError("CBF counterfactual count must match route count.")
        if self.candidate_batch.chunks.shape != self.route_batch.chunks.shape:
            raise ValueError("Ranker batch and route batch chunks must have identical shape.")

    @property
    def labels(self) -> tuple[str, ...]:
        return self.route_batch.labels

    @property
    def chunks(self) -> np.ndarray:
        return self.candidate_batch.chunks

    @property
    def valid_mask(self) -> np.ndarray:
        return self.candidate_batch.valid_mask

    def as_dict(self) -> dict[str, Any]:
        return {
            "routes": self.route_batch.as_dict(),
            "candidate_batch": {
                "labels": list(self.candidate_batch.labels),
                "valid_mask": self.candidate_batch.valid_mask.tolist(),
                "rejection_reasons": [list(value) for value in self.candidate_batch.rejection_reasons],
            },
            "cbf_counterfactuals": [
                None if value is None else value.as_dict() for value in self.cbf_counterfactuals
            ],
        }


def route_batch_to_candidate_batch(route_batch: ObstacleRouteBatch) -> SafeCaptureCandidateBatch:
    """Convert route chunks to the unchanged ranker input contract."""

    return SafeCaptureCandidateBatch(
        chunks=np.asarray(route_batch.chunks, dtype=np.float64).copy(),
        labels=tuple(route_batch.labels),
        valid_mask=np.asarray(route_batch.valid_mask, dtype=bool).copy(),
        rejection_reasons=tuple(
            tuple(candidate.rejection_reasons) for candidate in route_batch.candidates
        ),
    )


def _counterfactual_from_diagnostics(
    *,
    label: str,
    route_id: str | None,
    requested_action: np.ndarray,
    diagnostics: Any,
) -> CBFRouteCounterfactual:
    return CBFRouteCounterfactual(
        label=label,
        route_id=route_id,
        requested_action=requested_action,
        verified_feasible=bool(getattr(diagnostics, "verified_feasible", False)),
        infeasible=bool(getattr(diagnostics, "infeasible", True)),
        timed_out=bool(getattr(diagnostics, "timed_out", False)),
        fallback_mode=str(getattr(diagnostics, "fallback_mode", "unknown")),
        solver_status=str(getattr(diagnostics, "solver_status", "unknown")),
        minimum_constraint_value=float(getattr(diagnostics, "minimum_constraint_value", -float("inf"))),
        action_correction_norm=float(getattr(diagnostics, "action_correction_norm", float("inf"))),
        active_constraints=tuple(str(value) for value in getattr(diagnostics, "active_constraints", ())),
    )


def probe_route_batch_with_cbf(
    route_batch: ObstacleRouteBatch,
    safety_filter: Any,
    observation: Mapping[str, Any],
) -> ObstacleRouteRuntimeBatch:
    """Probe valid route first steps and mark failures before JEPA ranking.

    This mirrors the established v2 prefilter semantics: only a primary,
    finite, non-timeout Joint CBF solve with no fallback makes a route eligible.
    The returned action chunks are never replaced by CBF output.
    """

    if safety_filter is None or not hasattr(safety_filter, "verify_requested_action"):
        raise ValueError("probe_route_batch_with_cbf requires a Joint CBF verification interface.")
    candidate_batch = route_batch_to_candidate_batch(route_batch)
    valid = np.asarray(candidate_batch.valid_mask, dtype=bool).copy()
    reasons = [list(values) for values in candidate_batch.rejection_reasons]
    probes: list[CBFRouteCounterfactual | None] = []
    for index, candidate in enumerate(route_batch.candidates):
        if not bool(valid[index]):
            probes.append(None)
            continue
        requested = np.asarray(candidate.action_chunk[0], dtype=np.float64)
        diagnostics = safety_filter.verify_requested_action(requested, observation)
        probe = _counterfactual_from_diagnostics(
            label=candidate.label,
            route_id=candidate.route_id,
            requested_action=requested,
            diagnostics=diagnostics,
        )
        probes.append(probe)
        if not probe.accepted:
            valid[index] = False
            reasons[index].append("cbf_timeout" if probe.timed_out else "cbf_infeasible")
    filtered_batch = SafeCaptureCandidateBatch(
        chunks=candidate_batch.chunks,
        labels=candidate_batch.labels,
        valid_mask=valid,
        rejection_reasons=tuple(tuple(dict.fromkeys(values)) for values in reasons),
    )
    return ObstacleRouteRuntimeBatch(
        route_batch=route_batch,
        candidate_batch=filtered_batch,
        cbf_counterfactuals=tuple(probes),
    )


def probe_independent_cbf_counterfactuals(
    *,
    selected_action: np.ndarray,
    nominal_action: np.ndarray,
    safe_hold_action: np.ndarray,
    observation: Mapping[str, Any],
    safety_filter: Any,
    selected_route_id: str | None = None,
) -> tuple[CBFRouteCounterfactual, ...]:
    """Probe selected, nominal, and safe-hold requests independently.

    These calls are deliberately independent even when two requests have the
    same numeric value.  Their separate labels make execution traces answer
    whether a failure belongs to route selection, the nominal anchor, or the
    explicit hold path.
    """

    if safety_filter is None or not hasattr(safety_filter, "verify_requested_action"):
        raise ValueError("Independent CBF counterfactuals require a Joint CBF interface.")
    requests = (
        ("selected", selected_route_id, selected_action),
        ("nominal", None, nominal_action),
        ("safe_hold", None, safe_hold_action),
    )
    results: list[CBFRouteCounterfactual] = []
    for label, route_id, action in requests:
        requested = np.asarray(action, dtype=np.float64)
        if requested.ndim != 2 or requested.shape[1:] != (3,) or not np.isfinite(requested).all():
            raise ValueError(f"{label} action must be finite with shape [defenders, 3].")
        diagnostics = safety_filter.verify_requested_action(requested, observation)
        results.append(
            _counterfactual_from_diagnostics(
                label=label,
                route_id=route_id,
                requested_action=requested,
                diagnostics=diagnostics,
            )
        )
    return tuple(results)


__all__ = [
    "CBFRouteCounterfactual",
    "ObstacleRouteRuntimeBatch",
    "route_batch_to_candidate_batch",
    "probe_route_batch_with_cbf",
    "probe_independent_cbf_counterfactuals",
]
