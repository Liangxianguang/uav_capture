# JEPA Safe-Capture v3 WP3 Reliability Ledger

> Calibration-only, checkpoint-bound, immutable runtime artifact. The ledger gates ranking features; CBF remains the safety proof boundary.

Checkpoint SHA-256: `8ff2531e64571c9e57cfd78e9023a8b49191e06d1c4e4fd00adfaec90b629185`
Calibration dataset SHA-256: `ea04eec8e255bcafa95386ef4c30e366e55723334b8d4985d6c94887b9a1a307`
Minimum credit/sample count: `0.65` / `128`

## Global Calibration Summary

| Horizon (s) | Samples | Credit | Target MAE (m) | Clearance MAE (m) | Collision rate | Boundary rate | Local/coarse/global forecast |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 78080 | 0.8188 | 0.3150 | 0.1914 | 0.0000% | 0.0000% | 99.79%/0.02%/0.19% |
| 0.2 | 78080 | 0.8078 | 0.3734 | 0.2322 | 0.0000% | 0.0000% | 99.85%/0.01%/0.14% |
| 0.3 | 78080 | 0.7864 | 0.4487 | 0.1908 | 0.0000% | 0.0000% | 99.74%/0.02%/0.24% |
| 0.5 | 78080 | 0.8080 | 0.4007 | 0.2196 | 0.0000% | 0.0000% | 99.76%/0.06%/0.18% |

## Runtime State Forecast

State counts: `{"fallback_nominal": 20038, "trusted": 292282}`
Fallback reasons: `{"joint_ttc_cbf_risk": 0, "low_credit": 20038, "missing_bucket": 0, "ood": 0, "stale_observation": 0, "uncertainty_high": 0}`
Unsafe rate by state: `{"fallback_nominal": 0.0, "trusted": 0.0}`

## Gates and Limits

- High-credit failure-rate gate: **PASS**.
- OOD/stale/hard-context safe-hold routing: **PASS**.
- Full context entries: `710`; total entries including coarse/global: `1187`.
- Current P1 QP-feasibility labels have no class variation. The QP head is retained for future data, but this ledger does not claim QP-feasibility calibration.
- Low credit requests frozen V5 nominal action followed by CBF. Unknown/OOD contexts request safe-hold followed by the declared CBF fallback ladder.
- This artifact is not a closed-loop safe-capture result and must not be tuned with S3 development outcomes.

## v3 Fallback Audit

`{"all_required_fallbacks_pass": true, "cases": {"non_finite": {"fallback_reason": "non_finite_context", "passed": true, "state": "safe_hold"}, "ood": {"fallback_reason": "ood", "passed": true, "state": "safe_hold"}, "stale": {"fallback_reason": "stale_observation", "passed": true, "state": "safe_hold"}}}`

OOD, stale, and non-finite contexts all require explicit safe-hold; this audit is separate from closed-loop performance.
