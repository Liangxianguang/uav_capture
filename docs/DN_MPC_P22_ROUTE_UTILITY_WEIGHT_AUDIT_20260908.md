# DN-MPC P22 Route-Utility Weight Audit

**Status:** development-only; no action was executed and no CBF rule was changed.

The sweep subtracts `lambda * route_length_m / 10` from both the offline route-progress label and P18 prediction. Route length is public candidate geometry and is available before CBF verification.

Calibration-selected weight: **`0.0`** (selected only from calibration).

| Split | lambda | exact top-1 | tie-aware top-1 | informative top-1 | pairwise agreement |
|---|---:|---:|---:|---:|---:|
| train | 0 | 29.13% | 75.59% | 40.62% | 88.82% |
| train | 0.01 | 22.83% | 70.08% | 34.29% | 86.87% |
| train | 0.02 | 20.47% | 75.59% | 38.89% | 88.60% |
| train | 0.05 | 23.62% | 81.89% | 41.67% | 89.64% |
| train | 0.1 | 25.20% | 84.25% | 44.44% | 91.75% |
| train | 0.2 | 25.20% | 84.25% | 44.44% | 95.38% |
| train | 0.3 | 25.20% | 84.25% | 44.44% | 96.80% |
| train | 0.5 | 25.20% | 84.25% | 44.44% | 97.63% |
| train | 1 | 25.20% | 84.25% | 44.44% | 98.38% |
| train | 2 | 25.20% | 84.25% | 44.44% | 98.70% |
| validation | 0 | 19.19% | 66.28% | 31.43% | 84.32% |
| calibration | 0 | 20.50% | 62.00% | 25.00% | 84.25% |
| calibration | 0.01 | 23.50% | 56.50% | 17.14% | 83.26% |
| calibration | 0.02 | 31.00% | 66.50% | 18.60% | 85.09% |
| calibration | 0.05 | 38.00% | 74.50% | 22.00% | 87.20% |
| calibration | 0.1 | 38.50% | 77.50% | 23.91% | 89.69% |
| calibration | 0.2 | 39.00% | 78.00% | 23.91% | 94.28% |
| calibration | 0.3 | 39.00% | 78.00% | 23.91% | 95.97% |
| calibration | 0.5 | 39.00% | 78.00% | 23.40% | 97.02% |
| calibration | 1 | 39.00% | 78.00% | 23.40% | 97.65% |
| calibration | 2 | 39.00% | 78.00% | 23.40% | 97.83% |

On validation at the calibration-selected weight, informative top-1 is `0.3142857142857143` and pairwise agreement is `0.8431603773584906`.

## Decision

The explicit route-length prior improves pairwise utility direction but does not reach the direct-selection promotion gate. Keep route length as a planner cost/regularizer, not as evidence that P18 is ready for online JEPA selection. Retraining remains gated on selected/nominal/safe-hold CBF traces and fresh OOD/disagreement calibration.
