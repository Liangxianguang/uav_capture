"""Execution-settled reliability records for learned counterfactual ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .clearance_calibration import apply_head_offsets, offsets_for_horizon


_OBSERVATION_AGE_STATES = {"known", "never_received", "fresh", "delayed", "saturated"}


@dataclass(frozen=True)
class ReliabilityDecision:
    credit: float
    sample_count: int
    used_global_fallback: bool
    fallback_to_nominal: bool
    key: str


def _visible_bucket(value: float) -> str:
    return "visible" if float(value) >= 0.5 else "occluded"


def _message_age_bucket(value: float) -> str:
    normalized = float(value)
    if normalized <= 0.10:
        return "fresh"
    if normalized <= 0.35:
        return "delayed"
    return "stale"


def _clearance_bucket(value_m: float) -> str:
    if float(value_m) < 0.35:
        return "critical"
    if float(value_m) < 0.75:
        return "near"
    return "clear"


def _action_magnitude_bucket(value_mps: float) -> str:
    if float(value_mps) < 1.0:
        return "low"
    if float(value_mps) < 3.0:
        return "medium"
    return "high"


def make_context_key(
    horizon_index: int,
    visible_fraction: float,
    normalized_message_age: float,
    predicted_clearance_m: float,
    action_magnitude_mps: float,
) -> str:
    return "|".join(
        (
            f"h{int(horizon_index)}",
            _visible_bucket(visible_fraction),
            _message_age_bucket(normalized_message_age),
            _clearance_bucket(predicted_clearance_m),
            _action_magnitude_bucket(action_magnitude_mps),
        )
    )


def make_global_key(horizon_index: int) -> str:
    return f"h{int(horizon_index)}|global"


class ReliabilityLedger:
    """Read-only credit ledger derived from execution-settled validation data.

    Credit controls whether a learned ranking may be used.  It is deliberately
    conservative: sparse bins and unknown contexts fall back to the global
    horizon record, and a low score requests the deterministic nominal-action
    path.  The caller must still pass every selected action through CBF.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        if payload.get("ledger_type") != "jepa_v3_execution_settled_reliability":
            raise ValueError("Unsupported reliability ledger payload.")
        entries = payload.get("entries")
        policy = payload.get("decision_policy")
        if not isinstance(entries, dict) or not isinstance(policy, dict):
            raise ValueError("Reliability ledger requires entries and decision_policy mappings.")
        self.entries = entries
        self.minimum_sample_count = int(policy["minimum_sample_count"])
        self.minimum_credit = float(policy["minimum_credit"])
        if self.minimum_sample_count <= 0 or not 0.0 <= self.minimum_credit <= 1.0:
            raise ValueError("Invalid reliability ledger decision policy.")

    def decision(
        self,
        horizon_index: int,
        visible_fraction: float,
        normalized_message_age: float,
        predicted_clearance_m: float,
        action_magnitude_mps: float,
    ) -> ReliabilityDecision:
        key = make_context_key(
            horizon_index,
            visible_fraction,
            normalized_message_age,
            predicted_clearance_m,
            action_magnitude_mps,
        )
        entry = self.entries.get(key)
        used_global_fallback = entry is None
        if entry is None:
            key = make_global_key(horizon_index)
            entry = self.entries.get(key)
        if not isinstance(entry, dict):
            # Unknown future horizon or a malformed ledger is treated as OOD.
            return ReliabilityDecision(0.0, 0, True, True, key)
        credit = float(entry["credit"])
        sample_count = int(entry["sample_count"])
        fallback = sample_count < self.minimum_sample_count or credit < self.minimum_credit
        return ReliabilityDecision(credit, sample_count, used_global_fallback, fallback, key)


SAFE_CAPTURE_BUCKET_TOLERANCE_KEYS = (
    "visibility_fraction",
    "observation_age_steps",
    "clearance_m",
    "ttc_s",
    "uncertainty",
    "cbf_risk",
    "candidate_separation_m",
)


def normalize_safe_capture_bucket_tolerances(
    tolerances: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Return non-negative, JSON-stable boundary tolerances for ledger keys.

    A zero default preserves legacy v2/v3 ledgers.  New protocols bind their
    non-zero tolerances into the immutable ledger so calibration and runtime
    use the same conservative bucket semantics.
    """

    raw = dict(tolerances or {})
    unknown = sorted(set(raw).difference(SAFE_CAPTURE_BUCKET_TOLERANCE_KEYS))
    if unknown:
        raise ValueError(f"Unknown safe-capture bucket tolerance fields: {unknown}")
    normalized = {key: 0.0 for key in SAFE_CAPTURE_BUCKET_TOLERANCE_KEYS}
    for key, value in raw.items():
        tolerance = float(value)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(f"Safe-capture bucket tolerance {key} must be finite and non-negative.")
        normalized[key] = tolerance
    return normalized


def _safe_capture_visibility_bucket(value: float, *, boundary_tolerance: float = 0.0) -> str:
    tolerance = float(boundary_tolerance)
    return "visible" if float(value) - tolerance >= 0.5 else "occluded"


def _safe_capture_observation_age_bucket(value: float, *, boundary_tolerance: float = 0.0) -> str:
    age = float(value) + float(boundary_tolerance)
    if age <= 0.10:
        return "fresh"
    if age <= 0.35:
        return "delayed"
    return "stale"


def _safe_capture_clearance_bucket(value_m: float, *, boundary_tolerance: float = 0.0) -> str:
    clearance = float(value_m) - float(boundary_tolerance)
    if clearance < 0.35:
        return "critical"
    if clearance < 0.75:
        return "near"
    return "clear"


def _safe_capture_ttc_bucket(value_s: float, *, boundary_tolerance: float = 0.0) -> str:
    ttc = float(value_s) - float(boundary_tolerance)
    if ttc < 0.50:
        return "imminent"
    if ttc < 1.50:
        return "near"
    return "distant"


def _safe_capture_uncertainty_bucket(value: float, *, boundary_tolerance: float = 0.0) -> str:
    uncertainty = float(value) + float(boundary_tolerance)
    if uncertainty <= 0.10:
        return "low"
    if uncertainty <= 0.25:
        return "medium"
    return "high"


def _safe_capture_risk_bucket(value: float, *, boundary_tolerance: float = 0.0) -> str:
    risk = float(value) + float(boundary_tolerance)
    if risk < 0.25:
        return "low"
    if risk < 0.60:
        return "medium"
    return "high"


def _safe_capture_separation_bucket(value_m: float, *, boundary_tolerance: float = 0.0) -> str:
    separation = float(value_m) - float(boundary_tolerance)
    if separation < 0.05:
        return "low"
    if separation < 0.25:
        return "medium"
    return "high"


def make_safe_capture_context_key(
    horizon_index: int,
    visibility_condition: str,
    observation_age_bucket: str,
    obstacle_count: int,
    layout_signature: str,
    target_motion_mode: str,
    minimum_clearance_bucket: str,
    ttc_bucket: str,
    uncertainty_bucket: str,
    cbf_risk_bucket: str,
    candidate_separation_bucket: str,
) -> str:
    """Build the versioned full context key used by the v2 ledger."""

    fields = (
        f"h{int(horizon_index)}",
        f"vis={str(visibility_condition)}",
        f"age={str(observation_age_bucket)}",
        f"obs={int(obstacle_count)}",
        f"layout={str(layout_signature)}",
        f"motion={str(target_motion_mode)}",
        f"clear={str(minimum_clearance_bucket)}",
        f"ttc={str(ttc_bucket)}",
        f"unc={str(uncertainty_bucket)}",
        f"risk={str(cbf_risk_bucket)}",
        f"sep={str(candidate_separation_bucket)}",
    )
    return "|".join(fields)


def make_safe_capture_coarse_context_key(
    horizon_index: int,
    visibility_condition: str,
    observation_age_bucket: str,
    obstacle_count: int,
    target_motion_mode: str,
    minimum_clearance_bucket: str,
    uncertainty_bucket: str,
    cbf_risk_bucket: str,
) -> str:
    """Build a less sparse context key for deterministic ledger fallback."""

    fields = (
        f"h{int(horizon_index)}",
        f"vis={str(visibility_condition)}",
        f"age={str(observation_age_bucket)}",
        f"obs={int(obstacle_count)}",
        f"motion={str(target_motion_mode)}",
        f"clear={str(minimum_clearance_bucket)}",
        f"unc={str(uncertainty_bucket)}",
        f"risk={str(cbf_risk_bucket)}",
    )
    return "|".join(fields)


def make_safe_capture_global_key(horizon_index: int) -> str:
    return f"h{int(horizon_index)}|global"


@dataclass(frozen=True)
class SafeCaptureReliabilityDecision:
    """Immutable runtime decision for the checkpoint-bound v2 ledger."""

    state: str
    credit: float
    sample_count: int
    key: str
    fallback_reason: str | None
    used_coarse_fallback: bool
    used_global_fallback: bool


class SafeCaptureReliabilityLedger:
    """Read-only checkpoint-bound reliability ledger for safe-capture ranking.

    The ledger is deliberately not a safety certificate.  It only gates use of
    JEPA ranking features; every resulting action must still pass CBF.  Unknown
    or explicitly out-of-distribution contexts use safe-hold, while sparse or
    low-credit known contexts use nominal-action fallback.
    """

    LEDGER_TYPE = "jepa_safe_capture_v2_checkpoint_bound_reliability"
    LEDGER_TYPE_V3 = "jepa_safe_capture_v3_checkpoint_bound_reliability"
    LEDGER_TYPE_V3_CALIBRATED = "jepa_safe_capture_v3_checkpoint_bound_reliability_calibrated"
    SUPPORTED_LEDGER_TYPES = {LEDGER_TYPE, LEDGER_TYPE_V3, LEDGER_TYPE_V3_CALIBRATED}
    SUPPORTED_LEDGER_VERSIONS = {2, 3}
    REQUIRED_STATES = {"trusted", "fallback_nominal", "safe_hold"}

    def __init__(self, payload: Mapping[str, Any]) -> None:
        if payload.get("ledger_type") not in self.SUPPORTED_LEDGER_TYPES:
            raise ValueError("Unsupported safe-capture v2/v3 reliability ledger payload.")
        if payload.get("ledger_version") not in self.SUPPORTED_LEDGER_VERSIONS:
            raise ValueError("Safe-capture ledger_version must be 2 or 3.")
        if payload.get("not_a_locked_test") is not True:
            raise ValueError("Safe-capture v2 ledger must be development-only.")
        if payload.get("immutable_after_calibration") is not True:
            raise ValueError("Safe-capture v2 ledger must be immutable after calibration.")
        entries = payload.get("entries")
        policy = payload.get("decision_policy")
        source = payload.get("source")
        if not isinstance(entries, Mapping) or not isinstance(policy, Mapping) or not isinstance(source, Mapping):
            raise ValueError("Safe-capture v2 ledger requires entries, policy, and source mappings.")
        states = set(policy.get("states", []))
        if states != self.REQUIRED_STATES:
            raise ValueError(f"Safe-capture v2 ledger states must be {sorted(self.REQUIRED_STATES)}.")
        if source.get("checkpoint_sha256") is None or source.get("calibration_dataset_sha256") is None:
            raise ValueError("Safe-capture v2 ledger must bind checkpoint and calibration hashes.")
        if payload.get("ledger_type") == self.LEDGER_TYPE_V3_CALIBRATED:
            transform_hash = payload.get("clearance_calibration_sha256")
            if not isinstance(transform_hash, str) or len(transform_hash) != 64:
                raise ValueError("Calibrated safe-capture ledger must bind a calibration transform hash.")
            if source.get("clearance_calibration_sha256") != transform_hash:
                raise ValueError("Calibrated safe-capture ledger transform hash is not source-bound.")
        self.entries = dict(entries)
        self.policy = dict(policy)
        self.payload = dict(payload)
        self.clearance_calibration = payload.get("clearance_calibration")
        if payload.get("ledger_type") == self.LEDGER_TYPE_V3_CALIBRATED:
            if not isinstance(self.clearance_calibration, Mapping):
                raise ValueError("Calibrated safe-capture ledger requires clearance_calibration metadata.")
            # Validate every horizon at load time.  A malformed transform is a
            # provenance fault, not a reason to silently use raw predictions.
            rows = self.clearance_calibration.get("by_horizon")
            if not isinstance(rows, list) or not rows:
                raise ValueError("Calibrated safe-capture ledger has no horizon transforms.")
            for index in range(len(rows)):
                offsets_for_horizon(self.clearance_calibration, index)
        self.minimum_sample_count = int(policy["minimum_sample_count"])
        self.minimum_credit = float(policy["minimum_credit"])
        self.maximum_observation_age_steps = float(policy["maximum_observation_age_steps"])
        self.safe_hold_uncertainty_threshold = float(policy["safe_hold_uncertainty_threshold"])
        self.safe_hold_ttc_seconds = float(policy["safe_hold_ttc_seconds"])
        self.bucket_boundary_tolerances = normalize_safe_capture_bucket_tolerances(
            policy.get("bucket_boundary_tolerances")
        )
        if self.minimum_sample_count <= 0 or not 0.0 <= self.minimum_credit <= 1.0:
            raise ValueError("Invalid safe-capture v2 decision policy.")

    def calibrated_clearance_m(
        self,
        horizon_index: int,
        raw_obstacle_m: Any,
        raw_inter_agent_m: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the immutable checkpoint-bound transform, if present.

        Legacy v2/v3 ledgers return the raw metre values for backwards
        compatibility.  A calibrated ledger refuses malformed horizons rather
        than falling back to an uncalibrated safety gate.
        """

        obstacle = np.asarray(raw_obstacle_m, dtype=np.float64)
        inter_agent = np.asarray(raw_inter_agent_m, dtype=np.float64)
        if not np.isfinite(obstacle).all() or not np.isfinite(inter_agent).all():
            raise ValueError("Raw clearance predictions must be finite.")
        if self.clearance_calibration is None:
            return obstacle, inter_agent
        offsets = offsets_for_horizon(self.clearance_calibration, int(horizon_index))
        return apply_head_offsets(obstacle, inter_agent, offsets)

    @staticmethod
    def _context_values(context: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            "visibility_condition",
            "observation_age_steps",
            "obstacle_count",
            "layout_signature",
            "target_motion_mode",
            "minimum_clearance_m",
            "pairwise_ttc_s",
            "uncertainty",
            "cbf_risk",
            "candidate_separation_m",
        }
        missing = sorted(required.difference(context))
        if missing:
            raise ValueError(f"Safe-capture ledger context is missing fields: {missing}")
        values = dict(context)
        for name in ("visibility_condition", "observation_age_steps", "minimum_clearance_m", "pairwise_ttc_s", "uncertainty", "cbf_risk", "candidate_separation_m"):
            value = float(values[name])
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"Safe-capture ledger context field {name} is non-finite.")
            values[name] = value
        values["obstacle_count"] = int(values["obstacle_count"])
        values["observation_age_state"] = str(values.get("observation_age_state", "known"))
        if values["observation_age_state"] not in _OBSERVATION_AGE_STATES:
            raise ValueError(
                "Safe-capture ledger context field observation_age_state is invalid: "
                f"{values['observation_age_state']}"
            )
        return values

    def _keys(self, horizon_index: int, context: Mapping[str, Any]) -> tuple[str, str, str]:
        values = self._context_values(context)
        tolerances = self.bucket_boundary_tolerances
        visibility = (
            _safe_capture_visibility_bucket(
                float(values["visibility_condition"]),
                boundary_tolerance=tolerances["visibility_fraction"],
            )
            if isinstance(values["visibility_condition"], (float, int))
            else str(values["visibility_condition"])
        )
        age = _safe_capture_observation_age_bucket(
            float(values["observation_age_steps"]) / max(self.maximum_observation_age_steps, 1e-9),
            boundary_tolerance=tolerances["observation_age_steps"] / max(self.maximum_observation_age_steps, 1e-9),
        )
        clearance = _safe_capture_clearance_bucket(
            float(values["minimum_clearance_m"]),
            boundary_tolerance=tolerances["clearance_m"],
        )
        ttc = _safe_capture_ttc_bucket(
            float(values["pairwise_ttc_s"]),
            boundary_tolerance=tolerances["ttc_s"],
        )
        uncertainty = _safe_capture_uncertainty_bucket(
            float(values["uncertainty"]),
            boundary_tolerance=tolerances["uncertainty"],
        )
        risk = _safe_capture_risk_bucket(
            float(values["cbf_risk"]),
            boundary_tolerance=tolerances["cbf_risk"],
        )
        separation = _safe_capture_separation_bucket(
            float(values["candidate_separation_m"]),
            boundary_tolerance=tolerances["candidate_separation_m"],
        )
        full = make_safe_capture_context_key(
            horizon_index,
            visibility,
            age,
            values["obstacle_count"],
            values["layout_signature"],
            values["target_motion_mode"],
            clearance,
            ttc,
            uncertainty,
            risk,
            separation,
        )
        coarse = make_safe_capture_coarse_context_key(
            horizon_index,
            visibility,
            age,
            values["obstacle_count"],
            values["target_motion_mode"],
            clearance,
            uncertainty,
            risk,
        )
        return full, coarse, make_safe_capture_global_key(horizon_index)

    def decision(self, horizon_index: int, context: Mapping[str, Any]) -> SafeCaptureReliabilityDecision:
        try:
            values = self._context_values(context)
        except ValueError as error:
            if "non-finite" in str(error):
                return SafeCaptureReliabilityDecision(
                    "safe_hold", 0.0, 0, "invalid|non_finite_context", "non_finite_context", False, False
                )
            raise
        full_key, coarse_key, global_key = self._keys(horizon_index, values)
        if bool(values.get("ood", False)):
            return SafeCaptureReliabilityDecision("safe_hold", 0.0, 0, full_key, "ood", False, False)
        if values["observation_age_state"] == "never_received":
            return SafeCaptureReliabilityDecision(
                "safe_hold", 0.0, 0, full_key, "observation_never_received", False, False
            )
        age_tolerance_steps = self.bucket_boundary_tolerances["observation_age_steps"]
        if float(values["observation_age_steps"]) + age_tolerance_steps > self.maximum_observation_age_steps:
            return SafeCaptureReliabilityDecision("safe_hold", 0.0, 0, full_key, "stale_observation", False, False)
        if float(values["uncertainty"]) + self.bucket_boundary_tolerances["uncertainty"] > self.safe_hold_uncertainty_threshold:
            return SafeCaptureReliabilityDecision("safe_hold", 0.0, 0, full_key, "uncertainty_high", False, False)
        canonical_ttc = float(values["pairwise_ttc_s"]) - self.bucket_boundary_tolerances["ttc_s"]
        canonical_risk = float(values["cbf_risk"]) + self.bucket_boundary_tolerances["cbf_risk"]
        if canonical_ttc < self.safe_hold_ttc_seconds and canonical_risk >= 0.60:
            return SafeCaptureReliabilityDecision("safe_hold", 0.0, 0, full_key, "joint_ttc_cbf_risk", False, False)
        entry = self.entries.get(full_key)
        used_coarse = False
        used_global = False
        selected_key = full_key
        if not isinstance(entry, Mapping):
            entry = self.entries.get(coarse_key)
            used_coarse = isinstance(entry, Mapping)
            if used_coarse:
                selected_key = coarse_key
        if not isinstance(entry, Mapping):
            entry = self.entries.get(global_key)
            used_global = isinstance(entry, Mapping)
            if used_global:
                selected_key = global_key
        if not isinstance(entry, Mapping):
            return SafeCaptureReliabilityDecision("safe_hold", 0.0, 0, selected_key, "missing_bucket", used_coarse, used_global)
        credit = float(entry.get("credit", 0.0))
        sample_count = int(entry.get("sample_count", 0))
        if sample_count < self.minimum_sample_count or credit < self.minimum_credit:
            return SafeCaptureReliabilityDecision("fallback_nominal", credit, sample_count, selected_key, "low_credit", used_coarse, used_global)
        return SafeCaptureReliabilityDecision("trusted", credit, sample_count, selected_key, None, used_coarse, used_global)
