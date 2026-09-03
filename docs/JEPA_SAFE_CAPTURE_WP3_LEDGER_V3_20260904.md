# JEPA Safe-Capture v3 Reliability Ledger Aggregate

> Development-only calibration evidence. Ledgers are immutable, checkpoint-bound, and not safety certificates; all actions still require CBF.

Independent ledgers: `3`; locked test opened: `False`.

| Horizon (s) | Samples | Credit | Target MAE (m) | Clearance MAE (m) | Collision rate | Boundary rate | Ranking win rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 76800.0000 +/- 0.0000 | 0.8046 +/- 0.0113 | 0.2822 +/- 0.0090 | 0.1866 +/- 0.0066 | 0.0005 +/- 0.0000 | 0.0005 +/- 0.0000 | 0.6046 +/- 0.0164 |
| 0.2 | 76800.0000 +/- 0.0000 | 0.8088 +/- 0.0086 | 0.3268 +/- 0.0067 | 0.1746 +/- 0.0222 | 0.0005 +/- 0.0000 | 0.0005 +/- 0.0000 | 0.7403 +/- 0.0081 |
| 0.3 | 76800.0000 +/- 0.0000 | 0.7987 +/- 0.0068 | 0.3960 +/- 0.0057 | 0.2280 +/- 0.0359 | 0.0005 +/- 0.0000 | 0.0005 +/- 0.0000 | 0.7762 +/- 0.0019 |
| 0.5 | 76800.0000 +/- 0.0000 | 0.7686 +/- 0.0070 | 0.5333 +/- 0.0051 | 0.1601 +/- 0.0142 | 0.0008 +/- 0.0000 | 0.0008 +/- 0.0000 | 0.7961 +/- 0.0036 |

## Gates

- Runtime-valid ledgers: **PASS**.
- OOD/stale/non-finite fallback audit: **PASS**.
- High-credit failure-rate ordering: **PASS**.
- Eligible for closed-loop smoke only; this artifact does not authorize a locked test.
