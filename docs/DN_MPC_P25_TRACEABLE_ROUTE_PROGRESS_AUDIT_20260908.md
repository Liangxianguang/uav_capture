# DN-MPC P25 Traceable Archive Route-Progress Audit

**Status:** development-only; no online action or locked test was opened.

The P24 checkpoint was evaluated on the newly collected P25 traceable
archives. The route-progress metrics are the same as the ranking re-audit,
because P25 adds provenance and independent CBF traces without changing route
labels or predictions.

| Split | groups | exact top-1 | tie-aware top-1 | informative top-1 | pairwise agreement |
|---|---:|---:|---:|---:|---:|
| train | 136 | 59.56% | 94.12% | 97.06% | 94.88% |
| validation | 172 | 36.63% | 75.00% | 80.00% | 88.73% |
| calibration | 213 | 39.44% | 76.995% | 89.29% | 88.01% |

The validation informative result is materially above the P23 result, but the
exact top-1 gate is still below `50%`. Most validation groups remain near ties,
and the output slope remains compressed. P25 therefore closes traceability
only; it does not authorize route override.
