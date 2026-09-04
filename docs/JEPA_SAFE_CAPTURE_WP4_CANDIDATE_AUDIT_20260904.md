# WP-E Candidate Reachability and Ranking Audit

**Date:** 2026-09-04  
**Status:** development-only; `locked_test_opened=false`  
**Input:** `results/jepa_safe_capture_v3_wp1_failure_index_current/`  
**Output:** `results/jepa_safe_capture_v3_wp4_candidate_audit_current_next/`  
**TensorBoard:** `results/jepa_safe_capture_v3_tensorboard/wp4_candidate_audit_current_next/`

## Result

The audit validated the fixed five-candidate contract on all M3/A1/A2 source
traces: 6577 ranking steps over 180 episodes. Candidate labels, mask shapes,
finite score handling, selected-index validity, trusted eligibility, and
fallback-to-nominal invariants all passed. Every candidate had valid fraction
1.0 and no invalid candidate entered an eligible mask.

| Candidate | Valid fraction | Eligible fraction | Selected fraction | Selected-trajectory safe-capture rate |
|---|---:|---:|---:|---:|
| `nominal` | 1.0000 | 0.8040 | 0.2434 | 0.3250 |
| `intercept` | 1.0000 | 0.8039 | 0.5562 | 0.3397 |
| `lateral_clearance` | 1.0000 | 0.8036 | 0.0739 | 0.3820 |
| `formation_clearance` | 1.0000 | 0.8046 | 0.0426 | 0.3030 |
| `visibility_hold` | 1.0000 | 0.8039 | 0.0839 | 0.3934 |

The selected-trajectory rate is descriptive: a candidate is counted only in an
episode where it was actually selected at least once. It is not a
counterfactual candidate success rate.

## Gate Decision

`candidate_reachability_gate=true` and all execution invariants pass. The gate
for rejection-reason observability is false because the historical trace schema
does not contain per-candidate `rejection_reasons`; no reason was inferred from
the boolean mask. The counterfactual settled-outcome gate is also false because
each episode has one settled outcome for the selected trajectory, not five
settled outcomes under the same initial belief.

Therefore the classification is
`candidate_reachability_pass_ranking_evidence_incomplete`. Ranking weights must
remain frozen. A new protocol revision must record precheck reasons and
offline-only per-candidate settled labels before causal top-1 precision/recall,
rank calibration, or final-block readiness can be claimed.

Machine-readable output: `results/jepa_safe_capture_v3_wp4_candidate_audit_current_next/candidate_audit.json`.  
Hash manifest: `results/jepa_safe_capture_v3_wp4_candidate_audit_current_next/hash_manifest.json`.
