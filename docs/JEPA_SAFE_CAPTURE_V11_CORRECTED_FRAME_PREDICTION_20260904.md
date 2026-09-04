# JEPA Safe-Capture v2 P2 三 Seed Prediction Aggregate

> Development-only held-out prediction evidence. This is not a closed-loop result and not a locked test.

Model: `interaction_aware_action_conditioned_jepa_safe_capture_v2`
Training seeds: `20260911, 20260912, 20260913`
Validation samples per seed: `78080`
Validation dataset SHA-256: `a61c5c92ba6d9f8ac80e13e396297eb863ea2d59434d25b7f594d637049dfbe2`

## Prediction Summary

| Horizon (s) | Target MAE (m) | Constant-velocity MAE (m) | Improvement vs CV | Seeds better than CV | Visibility AUROC | CBF intervention AUROC | QP feasibility AUROC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.3245 +/- 0.0188 | 0.4408 +/- 0.0000 | 0.2638 +/- 0.0428 | 3/3 | 0.6987 +/- 0.0044 | 0.8079 +/- 0.0101 | n/a |
| 0.2 | 0.3846 +/- 0.0176 | 0.7253 +/- 0.0000 | 0.4698 +/- 0.0242 | 3/3 | 0.6938 +/- 0.0018 | 0.8051 +/- 0.0104 | n/a |
| 0.3 | 0.4638 +/- 0.0145 | 1.0488 +/- 0.0000 | 0.5578 +/- 0.0138 | 3/3 | 0.6877 +/- 0.0024 | 0.8014 +/- 0.0144 | n/a |
| 0.5 | 0.4370 +/- 0.0037 | 1.1172 +/- 0.0000 | 0.6088 +/- 0.0033 | 3/3 | 0.6818 +/- 0.0057 | 0.7056 +/- 0.0059 | n/a |

## Checkpoint Provenance

| Seed | Prediction-gate SHA-256 | Checkpoint SHA-256 |
|---:|---|---|
| 20260911 | `efcad79597be2ea72b9236bec88bcdec06dba790fc8fceb422c7d1522a388472` | `e638c5868a0e6047ad1cefb903973e0979ab5f47bde8b9a44889eb79775aa8d4` |
| 20260912 | `1e2b1515d566499f45f3c775f4da4c299953bc16526e6b0bf4789f2b4340b185` | `f3390bd321d6b9155570f8e8f47f4c072cd4c24c7e3c2afda61f00a34a5948a2` |
| 20260913 | `f82d5d9b370ac2ebdcb361c25953315a757738f0e0453ba307a0ab004d0aca87` | `c545915ab8540e468a6d863687677a9e059c1002aae8c17fa7e320f1c59052b5` |

## Interpretation

- All three held-out checkpoints are finite and pass the declared P2 target-prediction gate.
- Target prediction improves over constant velocity for every seed at all four horizons in this validation summary.
- `qp_feasibility_auc` is `n/a` because the current P1 feasibility labels contain no positive/negative class variation. This is not evidence of a calibrated QP-feasibility predictor.
- Clearance, visibility, uncertainty and intervention heads remain ranking signals only. They are not safety certificates and cannot bypass CBF.
- The next authorized step is calibration-only reliability-ledger construction with immutable nominal fallback. Closed-loop paired safe-capture evaluation is still required.

## Decision

All three held-out P2 checkpoints pass the finite-output and target prediction gates. This authorizes calibration-only reliability-ledger development, not a closed-loop or locked-test performance claim.
