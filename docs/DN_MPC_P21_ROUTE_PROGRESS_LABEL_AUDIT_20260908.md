# DN-MPC P21 Route-Progress Label and Score-Direction Audit

**Status:** development-only; no online action or CBF policy was changed.

The audit consumes `D:\uav-capture\uav_capture\results\dn_mpc_jepa_safe_capture_dev\p20_p18_candidate_ranking_reaudit\ranking_details.csv` and uses a tie margin of `0.005`.

| Split | groups | exact top-1 | tie-aware top-1 | informative top-1 | pairwise agreement | top-gap <= margin |
|---|---:|---:|---:|---:|---:|---:|
| train | 127 | 29.13% | 75.59% | 40.62% | 88.82% | 74.80% |
| validation | 172 | 19.19% | 66.28% | 31.43% | 84.32% | 79.65% |
| calibration | 200 | 20.50% | 62.00% | 25.00% | 84.25% | 86.00% |

## Interpretation

- `0.7965116279069767` of validation groups have a top-label gap no larger than the registered tie margin.
- Tie-aware top-1 is `0.6627906976744186`, while exact top-1 is `0.19186046511627908`.
- Among groups whose best label is separated by more than the informative margin, exact top-1 is `0.3142857142857143`.
- Pairwise agreement is `0.8431603773584906`; it is the more stable signal in this archive.

## Decision

Near ties explain part of the exact top-1 failure, but the informative-group top-1 result remains insufficient for direct route selection. Do not promote P18. Recalibrate or retrain route-progress with explicit tie-aware supervision, and keep DN-MPC/CBF as the executable authority.
