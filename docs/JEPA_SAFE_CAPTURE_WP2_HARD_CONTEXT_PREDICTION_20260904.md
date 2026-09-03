# JEPA v3 Hard-Context Prediction Gate

> Development-only held-out prediction evidence. This is not a closed-loop safe-capture result or a locked test.

Independent checkpoints: `3`.

| Horizon (s) | Target MAE (m) | Improvement vs CV | Seeds better | Obstacle clearance q10 MAE (m) | Inter-agent clearance q10 MAE (m) | Visibility Brier | CBF intervention Brier |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.3006 +/- 0.0112 | 0.0803 +/- 0.0344 | 3/3 | 1.8482 +/- 0.1836 | 0.2125 +/- 0.0022 | 0.1914 +/- 0.0052 | 0.0825 +/- 0.0067 |
| 0.2 | 0.3459 +/- 0.0100 | 0.2434 +/- 0.0219 | 3/3 | 1.8598 +/- 0.1108 | 0.2015 +/- 0.0370 | 0.1923 +/- 0.0018 | 0.0998 +/- 0.0151 |
| 0.3 | 0.4162 +/- 0.0070 | 0.3228 +/- 0.0114 | 3/3 | 1.8981 +/- 0.1457 | 0.2637 +/- 0.0295 | 0.1974 +/- 0.0034 | 0.1117 +/- 0.0164 |
| 0.5 | 0.5623 +/- 0.0083 | 0.4202 +/- 0.0086 | 3/3 | 1.7962 +/- 0.1568 | 0.1899 +/- 0.0249 | 0.1975 +/- 0.0039 | 0.1077 +/- 0.0050 |

## Decision

- All three hard-context weighted checkpoints emit finite predictions and improve target MAE over the constant-velocity baseline at every evaluated horizon.
- The auxiliary heads are available for ledger calibration, but prediction accuracy is not a safety certificate and does not establish safe-capture improvement.
- The next permitted step is independent calibration and reliability-ledger construction, followed by closed-loop smoke testing with the unchanged CBF contract.
- No locked test was opened.
