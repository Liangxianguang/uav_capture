# WP-D/F Reliability and CBF Fault Injection

**Date:** 2026-09-04  
**Status:** development-only; `locked_test_opened=false`  
**Output:** `results/jepa_safe_capture_v3_wp5_fault_injection_current/`  
**TensorBoard:** `results/jepa_safe_capture_v3_tensorboard/wp5_fault_injection_current/`

## Matrix

The deterministic matrix exercises the immutable reliability ledger and the
Joint CBF-QP using the development environment. It is not a task success-rate
evaluation and does not read a locked split.

| CBF scenario | Solver/fallback | Verified | Raw/unverified executed | Finite action | Latency (ms) |
|---|---|---:|---:|---:|---:|
| `nominal_feasible` | `success / none` | true | false | true | 1.869 |
| `safe_hold` | `success / safe_hold` | true | false | true | 1.824 |
| `nonfinite_request` | `nonfinite_request / controlled_abort` | false | false | true | 0.392 |
| `solver_timeout` | `timeout / controlled_abort` | false | false | true | 3.738 |
| `state_violation` | `state_safety_violation / controlled_abort` | false | false | true | 0.568 |
| `motion_infeasible` | `solver_failure / controlled_abort` | false | false | true | 27.743 |

Ledger cases and expected states:

| Injection | State | Reason |
|---|---|---|
| baseline | `trusted` | none |
| OOD | `safe_hold` | `ood` |
| stale observation | `safe_hold` | `stale_observation` |
| high uncertainty | `safe_hold` | `uncertainty_high` |
| non-finite context | `safe_hold` | `non_finite_context` |

## Gate Result

All CBF and ledger cases passed. Every fallback action was finite and no
unverified CBF result returned the original raw request. The measured matrix
p95 end-to-end latency was below 100 ms. Solver status, fallback mode, action
correction, minimum constraint, active constraints, and latency are retained in
the JSON output and TensorBoard provenance.

This proves the failure containment contract, not geometric feasibility under
all future scenes and not a safe-capture improvement. The next integration gate
must verify the same properties through the complete rolling-horizon evaluator.
