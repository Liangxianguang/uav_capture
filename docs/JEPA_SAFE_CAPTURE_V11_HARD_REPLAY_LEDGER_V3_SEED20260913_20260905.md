# JEPA Safe-Capture v3 WP3 Reliability Ledger

> Calibration-only, checkpoint-bound, immutable runtime artifact. The ledger gates ranking features; CBF remains the safety proof boundary.

Checkpoint SHA-256: `9fe66b66a6ea441807022c1fde71e61b578df3df6ab7265532761d70d6fab708`
Calibration dataset SHA-256: `ea04eec8e255bcafa95386ef4c30e366e55723334b8d4985d6c94887b9a1a307`
Minimum credit/sample count: `0.65` / `128`

## Global Calibration Summary

| Horizon (s) | Samples | Credit | Target MAE (m) | Clearance MAE (m) | Collision rate | Boundary rate | Local/coarse/global forecast |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 78080 | 0.8264 | 0.2930 | 0.1881 | 0.0000% | 0.0000% | 99.73%/0.01%/0.26% |
| 0.2 | 78080 | 0.8145 | 0.3536 | 0.1733 | 0.0000% | 0.0000% | 99.82%/0.03%/0.15% |
| 0.3 | 78080 | 0.7919 | 0.4337 | 0.1782 | 0.0000% | 0.0000% | 99.78%/0.04%/0.17% |
| 0.5 | 78080 | 0.8019 | 0.4035 | 0.1640 | 0.0000% | 0.0000% | 99.64%/0.03%/0.33% |

## Runtime State Forecast

State counts: `{"fallback_nominal": 20532, "trusted": 291788}`
Fallback reasons: `{"joint_ttc_cbf_risk": 0, "low_credit": 20532, "missing_bucket": 0, "ood": 0, "stale_observation": 0, "uncertainty_high": 0}`
Unsafe rate by state: `{"fallback_nominal": 0.0, "trusted": 0.0}`

## Gates and Limits

- High-credit failure-rate gate: **PASS**.
- OOD/stale/hard-context safe-hold routing: **PASS**.
- Full context entries: `773`; total entries including coarse/global: `1296`.
- Current P1 QP-feasibility labels have no class variation. The QP head is retained for future data, but this ledger does not claim QP-feasibility calibration.
- Low credit requests frozen V5 nominal action followed by CBF. Unknown/OOD contexts request safe-hold followed by the declared CBF fallback ladder.
- This artifact is not a closed-loop safe-capture result and must not be tuned with S3 development outcomes.

## v3 Fallback Audit

`{"all_required_fallbacks_pass": true, "cases": {"non_finite": {"fallback_reason": "non_finite_context", "passed": true, "state": "safe_hold"}, "ood": {"fallback_reason": "ood", "passed": true, "state": "safe_hold"}, "stale": {"fallback_reason": "stale_observation", "passed": true, "state": "safe_hold"}}}`

OOD, stale, and non-finite contexts all require explicit safe-hold; this audit is separate from closed-loop performance.
