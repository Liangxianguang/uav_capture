# JEPA Safe Capture P4 Head-Only Smoke Stop

**日期：** 2026-09-07  
**阶段：** development-only prediction audit  
**结论标签：** `prediction_signal_without_safe_risk_recall_stop`

## 1. 运行合同

The smoke initialized `interaction_aware_action_conditioned_jepa_route_identity_hard_negative_v2` from the frozen route-identity v1 checkpoint and trained only the five new hard-negative risk heads. The shared JEPA encoder and legacy route heads were frozen. No CBF margin, stale/OOD gate, Ledger route or runtime executor was changed.

| 项目 | 值 |
|---|---|
| train archive | `jepa_safe_capture_p3_hard_negative_archive_train_smoke4_v2_20260907` |
| validation archive | `jepa_safe_capture_p3_hard_negative_archive_validation_smoke4_v2_20260907` |
| calibration archive | `jepa_safe_capture_p3_hard_negative_archive_calibration_smoke4_v2_20260907` |
| base checkpoint | `jepa_route_identity_model_actor_train40_seed661606/checkpoint.pt` |
| base checkpoint SHA-256 | `4b68f8e61df978b79cb58113e6e960fc930912e992469c604abfe5f972a3bdf1` |
| head-only checkpoint | `jepa_route_identity_hard_negative_headonly_basev1_smoke_seed20260907/checkpoint.pt` |
| head-only checkpoint SHA-256 | `d81ea691a7888a440c7b98dfdf27cb372d826f09b08cc978fc22959732d890f9` |
| trainable parameters | `42,885` |
| epochs | `8` |
| device | `cuda` (RTX 5050 environment) |
| TensorBoard | `results/jepa_route_identity_hard_negative_headonly_basev1_smoke_seed20260907_tensorboard/` |

## 2. Training signal

On the 8-epoch smoke, validation risk losses decreased monotonically overall:

| head | validation epoch 1 | validation epoch 8 |
|---|---:|---:|
| stopping distance MSE | `0.12965` | `0.06872` |
| obstacle TTC MSE | `0.07462` | `0.03942` |
| boundary TTC MSE | `0.04645` | `0.03858` |
| pairwise TTC MSE | `0.09261` | `0.07261` |
| acceleration slack MSE | `0.00884` | `0.00548` |

This is evidence that the heads can fit the archive distribution. It is not evidence that the heads are safe enough for online ranking.

## 3. Held-out prediction audit

The audit used the frozen checkpoint on all three archives and logged MAE, RMSE, bias and correlation to a separate TensorBoard directory. The most important validation/calibration values are:

| head | validation MAE | validation correlation | calibration MAE | calibration correlation | calibration risk recall `<1s` |
|---|---:|---:|---:|---:|---:|
| stopping distance | `0.262` | `0.783` | `0.221` | `0.846` | n/a |
| obstacle TTC | `2.283 s` | `0.756` | `2.481 s` | `0.724` | `19.6%` |
| boundary TTC | `2.292 s` | `0.524` | `2.794 s` | `0.373` | `15.6%` |
| pairwise TTC | `2.804 s` | `0.461` | `3.124 s` | `0.353` | `3.5%` |
| acceleration slack | `0.077` | `0.480` | `0.072` | `0.624` | n/a |

The pairwise `<1s` recall is the blocking result. The model often predicts a long TTC for genuinely imminent pairwise risk, so a lower-risk candidate could be incorrectly preferred. Boundary TTC also degrades on calibration (`correlation 0.373`), showing distribution sensitivity.

## 4. Failure attribution

The failure is not a CBF safety violation: no online action was executed and no runtime safety gate was bypassed. It is a prediction/labeling limitation:

1. TTC is clipped at 10 seconds, making the safe majority much larger than the imminent-risk tail.
2. Smooth-L1 regression rewards average error and does not directly optimize recall for `<1s` or `<2s` hazards.
3. Pairwise TTC is sparse and interaction-dependent; a frozen backbone plus a single scalar regression head does not separate the high-risk tail reliably.
4. The `-1.0` acceleration-slack sentinel is excluded from its metric, but it still reduces the measured-slack coverage available to the head.

## 5. Stop decision

No full JEPA training, three-seed training, new Ledger, L0-L3 closed-loop run, or locked test was started after this audit. Increasing data or epochs without changing the objective would amplify a model that fits average TTC while missing the safety-critical tail.

## 6. Required next redesign

Before another training run, add explicit hazard classification/quantile objectives for TTC bands (`<=0.5s`, `<=1s`, `<=2s`), report recall at a fixed high precision, and use a censored/unknown mask for acceleration slack. Keep the continuous TTC regression as an auxiliary task, but do not let it stand alone as the Ledger risk signal. Re-run the same held-out audit first; only a materially improved pairwise and boundary hazard recall can authorize a new calibration Ledger and paired runtime test.

