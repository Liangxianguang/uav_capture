# JEPA Safe-Capture v2 P2 三 Seed Prediction Aggregate

> Development-only held-out prediction evidence. This is not a closed-loop result and not a locked test.

Model: `interaction_aware_action_conditioned_jepa_safe_capture_v2`
Training seeds: `20260911, 20260912, 20260913`
Validation samples per seed: `77400`
Validation dataset SHA-256: `48af3ce3bd83a7aa4d068d1f25c8311df706cf892c88d51690dd595c2643ccc7`

## Prediction Summary

| Horizon (s) | Target MAE (m) | Constant-velocity MAE (m) | Improvement vs CV | Seeds better than CV | Visibility AUROC | CBF intervention AUROC | QP feasibility AUROC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.2835 +/- 0.0114 | 0.3269 +/- 0.0000 | 0.1328 +/- 0.0348 | 3/3 | 0.6980 +/- 0.0019 | 0.7947 +/- 0.0297 | n/a |
| 0.2 | 0.3332 +/- 0.0101 | 0.4572 +/- 0.0000 | 0.2712 +/- 0.0220 | 3/3 | 0.6954 +/- 0.0018 | 0.8054 +/- 0.0255 | n/a |
| 0.3 | 0.4016 +/- 0.0130 | 0.6146 +/- 0.0000 | 0.3466 +/- 0.0212 | 3/3 | 0.6845 +/- 0.0024 | 0.7951 +/- 0.0294 | n/a |
| 0.5 | 0.5444 +/- 0.0101 | 0.9697 +/- 0.0000 | 0.4385 +/- 0.0104 | 3/3 | 0.6836 +/- 0.0021 | 0.6973 +/- 0.0134 | n/a |

## Checkpoint Provenance

| Seed | Prediction-gate SHA-256 | Checkpoint SHA-256 |
|---:|---|---|
| 20260911 | `3ab7244d449653758970bfd9fad5d3ff8c1a20aa90ccbea3b6379383f6f67169` | `3307c3935eabe0f6fb11a0dbe83ada0b4a4c610a1d96911a67c81cd6c66760e7` |
| 20260912 | `d190c8eb11268c8ec35eeadb13f39600271937df20f843625799caefe1a46b90` | `a95aaf56acce704aa7abec8bd3042309b2085cdca755a25f424ba1662ab4355c` |
| 20260913 | `0acd0e78cfe143a11b00275ae07a4170cd1c30ec3c4a2d1be0c82074247c2f1c` | `4fac01bbd49a0028485a87b07c10c1f27365c14a3298bc4f25f42f57c9072798` |

## Interpretation

- All three held-out checkpoints are finite and pass the declared P2 target-prediction gate.
- Target prediction improves over constant velocity for every seed at all four horizons in this validation summary.
- `qp_feasibility_auc` is `n/a` because the current P1 feasibility labels contain no positive/negative class variation. This is not evidence of a calibrated QP-feasibility predictor.
- Clearance, visibility, uncertainty and intervention heads remain ranking signals only. They are not safety certificates and cannot bypass CBF.
- The next authorized step is calibration-only reliability-ledger construction with immutable nominal fallback. Closed-loop paired safe-capture evaluation is still required.

## Decision

All three held-out P2 checkpoints pass the finite-output and target prediction gates. This authorizes calibration-only reliability-ledger development, not a closed-loop or locked-test performance claim.
