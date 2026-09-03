# Safe-Capture JEPA System P0 Contract Audit

**Date:** 2026-09-03
**Protocol:** `configs/jepa_safe_capture_v2_protocol.yaml`
**Protocol SHA-256:** `71a6e3e22058d921afe7ec11c0bde787d78d19aa5eb0df9e2d588caca630855f`
**Audit output:** `results/jepa_safe_capture_v2_protocol_audit.json`
**Status:** `passed`

## Contract Decision

P0 freezes the safety-first system boundary for the next experiment family:

- `safe_capture` is the primary endpoint; capture time is not an automatic rejection gate.
- JEPA is a candidate-trajectory evaluator/ranker only and cannot generate the final action.
- Candidates must be dynamics-feasible before JEPA scoring.
- The reliability ledger is calibration-only, checkpoint-bound, immutable after calibration, and has explicit `trusted`, `fallback_nominal`, and `safe_hold` states.
- CBF is enabled for baseline and candidate and is the final safety filter.
- Obstacle separation, inter-agent separation, boundary, altitude, speed, acceleration, and target-approach constraints are required.
- QP infeasibility or stale observations use deterministic safe-hold/nominal fallback through CBF.
- The rolling loop executes only the first action step and then re-observes/re-plans.
- `locked_test_opened=false` is a protocol invariant.

## Frozen Input Audit

The verifier found every currently required frozen input and recorded its byte hash:

| Artifact | SHA-256 |
|---|---|
| Frozen actor checkpoint | `535098773be05687e147043435649378532362d479bdc0375842970370ba40ba` |
| Environment config | `42bd4e158c5e314e0ece6add8038b32c384a7a2ca027e9387327656fccf751ad` |
| S3 scenario protocol | `7b90aaf8ae0bf65e886dc204b890eb69150a5c89e5637276e04c18f8506d3c78` |
| Reference scenes | `940b8140737f1a955ececbd6a4a518e434d02ed87e95596e5c1a512e61fd1c58` |
| Reference episodes | `88a85b50b7cc0cde6e162daec0e56bbf95f186cfa74e5c2b61cc9317f2bfc845` |

The full absolute paths, byte counts, and protocol hash are in
`results/jepa_safe_capture_v2_protocol_audit.json`. This result is a protocol
audit, not a new model-training or locked-test result.

## Required Next Gates

P0 permits P1 data work only. Before any new closed-loop claim, P1-P6 must
provide separate train/validation/calibration/development provenance, complete
TensorBoard training records, checkpoint-bound ledgers, zero-perturbation
regression, CBF safety evidence, and three-seed paired safe-capture statistics.

The verifier is `scripts/verify_jepa_safe_capture_protocol.py`; its tests are
`tests/test_jepa_safe_capture_protocol.py`. A failure in either file stops the
next phase and does not authorize reading a locked test.
