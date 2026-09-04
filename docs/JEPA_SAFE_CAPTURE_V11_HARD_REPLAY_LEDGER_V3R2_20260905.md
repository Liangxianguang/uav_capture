# JEPA Safe-Capture v3 Reliability Ledger Aggregate

> Development-only calibration evidence. Ledgers are immutable, checkpoint-bound, and not safety certificates; all actions still require CBF.

Independent ledgers: `3`; locked test opened: `False`.

| Horizon (s) | Samples | Credit | Target MAE (m) | Clearance MAE (m) | Collision rate | Boundary rate | Ranking win rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 78080.0000 +/- 0.0000 | 0.8248 +/- 0.0054 | 0.3043 +/- 0.0110 | 0.2748 +/- 0.1473 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.7471 +/- 0.0084 |
| 0.2 | 78080.0000 +/- 0.0000 | 0.8136 +/- 0.0053 | 0.3618 +/- 0.0104 | 0.2818 +/- 0.1401 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.8257 +/- 0.0062 |
| 0.3 | 78080.0000 +/- 0.0000 | 0.7933 +/- 0.0077 | 0.4384 +/- 0.0089 | 0.2593 +/- 0.1298 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.8376 +/- 0.0057 |
| 0.5 | 78080.0000 +/- 0.0000 | 0.8056 +/- 0.0033 | 0.4073 +/- 0.0090 | 0.2417 +/- 0.0908 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.8603 +/- 0.0054 |

## Gates

- Runtime-valid ledgers: **PASS**.
- OOD/stale/non-finite fallback audit: **PASS**.
- High-credit failure-rate ordering: **PASS**.
- Eligible for closed-loop smoke only; this artifact does not authorize a locked test.
