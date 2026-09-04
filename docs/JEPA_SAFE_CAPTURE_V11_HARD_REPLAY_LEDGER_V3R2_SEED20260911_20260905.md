# JEPA Safe-Capture v3 WP3 Reliability Ledger

> Calibration-only, checkpoint-bound, immutable runtime artifact. The ledger gates ranking features; CBF remains the safety proof boundary.

Checkpoint SHA-256: `2317a9464f8001f27a5c028bb6b4c431c904af7bfc33bf43b3a1d05a5a9c6154`
Calibration dataset SHA-256: `ea04eec8e255bcafa95386ef4c30e366e55723334b8d4985d6c94887b9a1a307`
Minimum credit/sample count: `0.65` / `128`

## Global Calibration Summary

| Horizon (s) | Samples | Credit | Target MAE (m) | Clearance MAE (m) | Collision rate | Boundary rate | Local/coarse/global forecast |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 78080 | 0.8293 | 0.3050 | 0.4449 | 0.0000% | 0.0000% | 99.81%/0.03%/0.16% |
| 0.2 | 78080 | 0.8184 | 0.3582 | 0.4399 | 0.0000% | 0.0000% | 99.73%/0.03%/0.25% |
| 0.3 | 78080 | 0.8017 | 0.4329 | 0.4090 | 0.0000% | 0.0000% | 99.66%/0.01%/0.33% |
| 0.5 | 78080 | 0.8070 | 0.4175 | 0.3416 | 0.0000% | 0.0000% | 99.84%/0.03%/0.13% |

## Runtime State Forecast

State counts: `{"fallback_nominal": 15974, "trusted": 296346}`
Fallback reasons: `{"joint_ttc_cbf_risk": 0, "low_credit": 15974, "missing_bucket": 0, "ood": 0, "stale_observation": 0, "uncertainty_high": 0}`
Unsafe rate by state: `{"fallback_nominal": 0.0, "trusted": 0.0}`

## Gates and Limits

- High-credit failure-rate gate: **PASS**.
- OOD/stale/hard-context safe-hold routing: **PASS**.
- Full context entries: `629`; total entries including coarse/global: `1068`.
- Current P1 QP-feasibility labels have no class variation. The QP head is retained for future data, but this ledger does not claim QP-feasibility calibration.
- Low credit requests frozen V5 nominal action followed by CBF. Unknown/OOD contexts request safe-hold followed by the declared CBF fallback ladder.
- This artifact is not a closed-loop safe-capture result and must not be tuned with S3 development outcomes.

## v3 Fallback Audit

`{"all_required_fallbacks_pass": true, "cases": {"non_finite": {"fallback_reason": "non_finite_context", "passed": true, "state": "safe_hold"}, "ood": {"fallback_reason": "ood", "passed": true, "state": "safe_hold"}, "stale": {"fallback_reason": "stale_observation", "passed": true, "state": "safe_hold"}}}`

OOD, stale, and non-finite contexts all require explicit safe-hold; this audit is separate from closed-loop performance.
