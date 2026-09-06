# JEPA Safe Capture P3 Hard-Negative Archive

**日期：** 2026-09-07  
**阶段：** development-only train smoke  
**结论标签：** `archive_signal_ready_training_blocked`

## 1. 目的

P1 的 earliest-abort 审计显示，很多 route 在第一步可以通过 CBF，但在第 4/5 个 rollout step 因 stopping distance、障碍物/边界净空或联合约束而失效。本阶段把这些后续风险写成 action-conditioned counterfactual labels，为 JEPA 辅助头提供可审计的 hard negatives。

本阶段没有打开 locked split，没有修改 CBF margin、stale/OOD/non-finite gate，也没有执行 raw-unverified action。

## 2. 合同和运行

| 项目 | 值 |
|---|---|
| dataset version | `jepa_safe_capture_route_identity_hard_negative_v2` |
| split | `train` |
| episodes | 4 |
| frozen actor | `models/v5_development_exact_reactive_seed661606.pt` |
| CBF | `anticipatory_horizon_steps=5`, `barrier_mode=strict_buffer` |
| history / route chunk | `8 / 3` |
| sample stride | `16` |
| output | `results/jepa_safe_capture_p3_hard_negative_archive_train_smoke4_v2_20260907/` |
| TensorBoard | `results/jepa_safe_capture_p3_hard_negative_archive_train_smoke4_v2_20260907_tensorboard/` |
| dataset SHA-256 | `02ca878d6b89d5b955e22c696e770d748a31873717f2ef7d567fa03ce11ee342` |

The collector records protocol, environment, archive-config and actor hashes in `metadata.json`/`provenance.json`. TensorBoard contains archive counts, route coverage, CBF first-step feasibility, horizon failure and negative TTC/slack fractions.

## 3. Results

| 指标 | 结果 |
|---|---:|
| total samples | `1872` |
| runtime route rows | `1728` |
| offline boundary-shadow rows | `144` |
| geometry-valid runtime rows | `1240 / 1728 = 71.76%` |
| first-step CBF feasible | `1728 / 1728 = 100%` |
| horizon failure within steps 1/2/3/5 | `572 / 1728 = 33.10%` |
| boundary-clearance negative rows | `110` |
| route classes | 12 classes, `36` states each |

All new arrays have shape `[1872, 5]` and are finite:

| 标签 | 最小值 | 最大值 |
|---|---:|---:|
| stopping distance (m) | `0.0` | `2.0833` |
| obstacle TTC (s, clipped) | `0.0` | `10.0` |
| boundary TTC (s, clipped) | `0.0` | `10.0` |
| pairwise TTC (s, clipped) | `0.0` | `10.0` |
| acceleration slack | `-1.0` | `0.36` |

The `-1.0` acceleration-slack sentinel is reserved for failed/unknown diagnostics; successful rows contain measured slack. It is not interpreted as a CBF execution result.

## 4. Interpretation

The archive now contains the intended learning signal: first-step feasibility is not the bottleneck, while a substantial fraction of branches fail later in the horizon. This supports training stopping-distance/TTC/feasibility auxiliary heads and earlier braking or route switching.

This is **not** evidence of improved `safe_capture`, JEPA accuracy, or end-to-end control. The archive is train-only and was collected from a frozen actor. The P3 gate remains open until a non-overlapping validation archive contains auditable positive and negative feasibility, boundary and visibility classes.

## 5. Stop decision and next action

No JEPA training, L1-L3 expansion, or multi-seed benchmark was started after this smoke. The next bounded action is to collect validation and calibration archives with disjoint seeds, audit label direction/calibration, and only then run a small head-only JEPA training smoke. If paired safe-capture does not improve after that controlled experiment, stop further model/data scaling and attribute the failure using the existing route/Cbf/Ledger traces.

