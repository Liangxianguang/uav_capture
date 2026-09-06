# P0 Message-Age Contract Audit

**Date:** 2026-09-07  
**Scope:** development-only simulation/SITL contract audit  
**Locked test:** not opened  
**Purpose:** distinguish an uninitialized message stream from a received-but-saturated stream before using communication age in the Reliability Ledger or failure attribution.

## Result

The synthetic state-machine audit passed all three required transitions:

| Case | Expected transition | Result |
|---|---|---|
| `never_received` | numeric compatibility age remains at the ceiling while semantic state stays `never_received` | pass |
| delayed delivery | age advances only when no packet is delivered, including while the target is visible | pass |
| saturation and recovery | `fresh -> delayed -> saturated -> fresh` after a new packet | pass |

The authoritative fields are `message_received` and `message_age_state`. The bounded numeric `message_age_steps` field remains only for frozen-actor compatibility.

Artifacts:

- Audit: `results/jepa_safe_capture_p0_message_age_contract_20260907/audit.json`
- Report: `results/jepa_safe_capture_p0_message_age_contract_20260907/report.md`
- TensorBoard: `results/jepa_safe_capture_p0_message_age_contract_20260907_tensorboard/`

## Historical V21 Trace Consequence

The V21 smoke traces used by the failure index contain `message_age_steps`, but do not contain `message_received` or `message_age_state`. Re-indexing those traces therefore produces:

- `message_age_semantics = legacy_numeric_only` for all 240 episodes;
- 408 control-step rows at the numeric ceiling whose semantic state is unresolved;
- 123 failed episodes labeled `communication_age_unresolved`;
- zero episodes labeled `communication_age_saturated` from these legacy traces.

The episode result is unchanged: `117/240 = 48.8%` safe capture, `116` primary `cbf_controlled_abort`, `7` timeout, and collision/boundary/pairwise/raw-unverified all zero. This is a diagnostic correction, not a controller change.

Re-indexed artifacts:

- `results/jepa_safe_capture_v21_failure_index_20260907_age_semantics/`
- `results/jepa_safe_capture_v21_current_tensorboard/failure_index_20260907_age_semantics/`

## Environment and Verification

The audit was run with the available conda environment `D:\download\anaconda3\envs\traj_pred_prep` after aligning TensorBoard dependencies to TensorBoard 2.19.0 and protobuf 5.29.6 for NumPy 2.3.5 compatibility. Focused regression tests passed: `14 passed`.

This change does not relax CBF margins, stale/OOD/non-finite gates, candidate eligibility, or `controlled_abort`. It does not authorize new training or L1-L3 expansion.

## Next Gate

Before interpreting communication-age labels as causal evidence, regenerate or replay a development trace contract that records `message_received`, `message_age_state`, target-observation age state, and rollout age at every control step. Only after that schema gate passes should communication-age-conditioned Ledger calibration be considered.
