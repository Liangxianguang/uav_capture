# JEPA-v3 Multitask Three-Seed Prediction Aggregate

> Development-only prediction evidence. It is not a locked test and does not establish closed-loop capture improvement.

Training seeds: `20260911, 20260912, 20260913`.

| Horizon (s) | Target MAE (m) | Improvement vs CV | Seeds better than CV | Obstacle clearance MAE (m) | Inter-agent clearance MAE (m) | Visibility AUROC | CBF intervention AUROC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.1818 +/- 0.0206 | -0.0615 +/- 0.1202 | 1/3 | 0.8898 +/- 0.0607 | 0.1182 +/- 0.0183 | 0.5718 +/- 0.0021 | 0.9513 +/- 0.0028 |
| 0.2 | 0.2108 +/- 0.0122 | 0.2140 +/- 0.0454 | 3/3 | 0.8865 +/- 0.0605 | 0.1159 +/- 0.0166 | 0.5689 +/- 0.0056 | 0.9333 +/- 0.0036 |
| 0.3 | 0.2436 +/- 0.0167 | 0.3538 +/- 0.0442 | 3/3 | 0.8891 +/- 0.0695 | 0.1090 +/- 0.0126 | 0.5744 +/- 0.0001 | 0.9268 +/- 0.0043 |
| 0.5 | 0.2959 +/- 0.0121 | 0.4968 +/- 0.0205 | 3/3 | 0.8859 +/- 0.0573 | 0.1060 +/- 0.0112 | 0.5703 +/- 0.0034 | 0.9182 +/- 0.0058 |

## Run Provenance

| Seed | Best epoch | Best validation loss | Checkpoint SHA-256 | TensorBoard epochs | Histogram tags |
| ---: | ---: | ---: | --- | ---: | ---: |
| 20260911 | 7 | -3.522987 | `af4963f46e67497d29987621b16f061cf7077db0887cc465ceef5ee6f883d3b7` | 40 | 149 |
| 20260912 | 5 | -3.448652 | `f82e4f89e8361de2c8984c2cbc32c37286aa250ac75123cdc5bb87f7fb12a18e` | 40 | 149 |
| 20260913 | 5 | -3.500930 | `28469dc6c86c4b18c63a61b75d1507c2ae37e7e8cab51537369eaa2bfc138935` | 40 | 149 |

## Interpretation and Limits

- All three checkpoints pass the finite-output, held-out prediction, action-following, and TensorBoard provenance checks.
- Target prediction improves over constant velocity in all three seeds at 0.2, 0.3, and 0.5 seconds. At 0.1 seconds only one of three seeds improves, so the learned predictor must not be treated as uniformly superior at every horizon.
- CBF intervention AUROC is consistently high across horizons. Visibility AUROC is modest and obstacle-clearance error remains material; neither signal is a safety certificate or a license to bypass CBF.
- The next permitted use is an execution-settled reliability ledger with deterministic nominal-action fallback. Closed-loop paired development evaluation remains required before any claim that this model improves capture.

Action-following candidate separation: `0.000841 +/- 0.000118` normalized position units.

Prediction gates and action-following audits pass, but closed-loop paired development evidence is still required.
