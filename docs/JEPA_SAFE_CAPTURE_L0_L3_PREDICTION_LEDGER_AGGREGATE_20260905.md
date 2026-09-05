# JEPA L0-L3 Prediction and Ledger Aggregate

This is a development-only P2/P3 aggregate. It is not a closed-loop safe-capture result and did not open a locked test.

Seeds: 20260911, 20260912, 20260913
All prediction outputs finite: `True`
Every seed beats constant velocity at every horizon: `True`

| Horizon (s) | Position MAE mean +/- std (m) | Improvement over CV mean +/- std | Visibility AUC mean +/- std | QP feasibility Brier mean +/- std |
|---:|---:|---:|---:|---:|
| 0.1 | 0.0877 +/- 0.0042 | 0.6555 +/- 0.0167 | 0.7733 +/- 0.0065 | 0.000154 +/- 0.000000 |
| 0.2 | 0.1158 +/- 0.0046 | 0.7612 +/- 0.0096 | 0.7753 +/- 0.0088 | 0.000154 +/- 0.000000 |
| 0.3 | 0.1553 +/- 0.0070 | 0.7940 +/- 0.0093 | 0.7680 +/- 0.0122 | 0.000154 +/- 0.000000 |
| 0.5 | 0.1444 +/- 0.0034 | 0.8058 +/- 0.0046 | 0.7618 +/- 0.0182 | 0.000256 +/- 0.000000 |

## Interpretation

All three seeds pass the held-out prediction gate and show positive target-position improvement over the constant-velocity reference at all four horizons. The ledger reports zero unsafe rate in each routed state for every seed in the calibration replay. These findings establish prediction and routing evidence only; the L0-L3 `safe_capture` endpoint still requires paired rolling-horizon closed-loop evaluation.
