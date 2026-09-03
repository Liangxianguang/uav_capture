"""Execution-settled reliability records for learned counterfactual ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
