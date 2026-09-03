# JEPA Safe-Capture v2 P3 三 Seed Reliability Ledger Aggregate

> Calibration-only evidence. Ledgers are checkpoint-bound and immutable at runtime; this is not a closed-loop result or a locked test.

Seeds: `20260911, 20260912, 20260913`

## Calibration Summary

| Horizon (s) | Global credit | Target MAE (m) | Clearance MAE (m) | Collision rate | Boundary rate | Local coverage | Coarse coverage | Global coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.8194 +/- 0.0050 | 0.2679 +/- 0.0071 | 0.1904 +/- 0.0105 | 0.0005 +/- 0.0000 | 0.0005 +/- 0.0000 | 0.9973 +/- 0.0005 | 0.0002 +/- 0.0002 | 0.0025 +/- 0.0005 |
| 0.2 | 0.8222 +/- 0.0049 | 0.3167 +/- 0.0069 | 0.1891 +/- 0.0161 | 0.0005 +/- 0.0000 | 0.0005 +/- 0.0000 | 0.9967 +/- 0.0009 | 0.0002 +/- 0.0000 | 0.0031 +/- 0.0009 |
| 0.3 | 0.8049 +/- 0.0052 | 0.3826 +/- 0.0108 | 0.1807 +/- 0.0096 | 0.0005 +/- 0.0000 | 0.0005 +/- 0.0000 | 0.9971 +/- 0.0003 | 0.0001 +/- 0.0001 | 0.0027 +/- 0.0004 |
| 0.5 | 0.7859 +/- 0.0019 | 0.5170 +/- 0.0079 | 0.1932 +/- 0.0086 | 0.0008 +/- 0.0000 | 0.0008 +/- 0.0000 | 0.9974 +/- 0.0006 | 0.0001 +/- 0.0001 | 0.0025 +/- 0.0006 |

## State and Fallback Aggregate

State totals: `{"fallback_nominal": 64069, "trusted": 857531}`
Fallback reasons: `{"joint_ttc_cbf_risk": 0, "low_credit": 64069, "missing_bucket": 0, "ood": 0, "stale_observation": 0, "uncertainty_high": 0}`

High-credit failure-rate gate: **PASS**.
OOD/stale/hard-context safe-hold routing: **PASS**.

## Interpretation

- All three calibration ledgers preserve the checkpoint, calibration dataset, metadata and protocol provenance.
- High-credit contexts have no higher settled unsafe rate than fallback contexts in each seed-level forecast.
- Calibration contains no QP-feasibility class variation, so the QP head is not treated as calibrated evidence.
- The ledger may gate candidate-ranking development. It is not a safety proof; strict multi-agent CBF/QP remains mandatory.
- No closed-loop safe-capture or locked-test claim is authorized by this aggregate.
