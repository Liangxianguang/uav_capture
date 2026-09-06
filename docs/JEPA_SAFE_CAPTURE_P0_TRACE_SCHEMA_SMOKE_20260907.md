# P0 Trace-Schema Smoke

**Date:** 2026-09-07  
**Scope:** development-only L0 open execution-contract smoke  
**Locked test:** not opened  
**Purpose:** verify that newly generated runtime traces preserve communication and observation age semantics needed by the failure index.

## Contract

- Collection: `jepa_safe_capture_l0_l3_collection_v2.yaml`
- Scenario: `l0_open_nominal`, one episode
- Variant: `M0` nominal actor + CBF
- CBF: `strict_buffer`, anticipatory horizon `5`
- Device: CPU execution path on the RTX 5050 host
- Actor: `models/v5_development_exact_reactive_seed661606.pt`
- Raw/unverified actions: forbidden
- Development-only: true

## Result

| Metric | Result |
|---|---:|
| safe capture | `1/1` |
| control cycles | `150` |
| collision / defender boundary / pairwise | `0 / 0 / 0` |
| raw-unverified executed steps | `0` |
| CBF controlled abort | `0` |
| transit success | `100%` |
| CBF solver p95 | `5.22 ms` |
| cycle p95 | `21.73 ms` |

Every step trace contains `message_received`, `message_age_state`, `target_observation_received`, and `target_observation_age_state` in both `input_observation` and post-step `observation`. The smoke trace summary reports `message_age_semantics=explicit_or_received_inferred`, `fresh=600`, `message_age_saturated_rows=0`, and `message_age_unresolved_rows=0`.

Artifacts:

- Run: `results/jepa_safe_capture_p0_trace_schema_smoke_20260907/`
- TensorBoard: `results/jepa_safe_capture_p0_trace_schema_smoke_20260907_tensorboard/`

## Interpretation

This smoke establishes the trace schema for future development runs. It does not establish a JEPA improvement, a multi-seed result, or a replacement for the V4/V5 locked contracts. The historical V21 matrix remains numerically unchanged and is still labeled `communication_age_unresolved` where only the legacy numeric age ceiling was recorded.

The next allowed action is a small fixed-manifest contract audit of candidate coverage and earliest CBF infeasibility. New training remains gated by settled-ranking calibration and the P0 provenance checks.
