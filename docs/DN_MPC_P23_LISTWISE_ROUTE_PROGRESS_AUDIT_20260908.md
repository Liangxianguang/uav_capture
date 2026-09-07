# DN-MPC P21 Route-Progress Label and Score-Direction Audit

**Status:** development-only; no online action or CBF policy was changed.

The audit consumes `D:\uav-capture\uav_capture\results\dn_mpc_jepa_safe_capture_dev\p23_listwise_candidate_ranking_audit\ranking_details.csv` and uses a tie margin of `0.005`.

| Split | groups | exact top-1 | tie-aware top-1 | informative top-1 | pairwise agreement | top-gap <= margin |
|---|---:|---:|---:|---:|---:|---:|
| train | 127 | 44.88% | 80.31% | 68.75% | 92.35% | 74.80% |
| validation | 172 | 29.65% | 61.63% | 51.43% | 83.47% | 79.65% |
| calibration | 200 | 31.00% | 68.50% | 53.57% | 84.47% | 86.00% |

## Interpretation

- `0.7965116279069767` of validation groups have a top-label gap no larger than the registered tie margin.
- Tie-aware top-1 is `0.6162790697674418`, while exact top-1 is `0.29651162790697677`.
- Among groups whose best label is separated by more than the informative margin, exact top-1 is `0.5142857142857142`.
- Pairwise agreement is `0.8346698113207547`; it is the more stable signal in this archive.

## Decision

Near ties explain part of the exact top-1 failure, but the informative-group top-1 result remains insufficient for direct route selection. Do not promote P18. Recalibrate or retrain route-progress with explicit tie-aware supervision, and keep DN-MPC/CBF as the executable authority.
