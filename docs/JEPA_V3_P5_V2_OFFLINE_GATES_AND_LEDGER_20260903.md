# JEPA-v3 P5 v2 Offline Gates and Reliability Ledger

**阶段：** P5-B/P5-C  
**日期：** 2026-09-03  
**结论：** `pass; admitted_to_strict_zero_regression_only`  
**实验性质：** development-only；不构成 closed-loop capture improvement 或 locked-test 结论

## 1. 输入与边界

三个 P5 v2 checkpoint 均只在 v2 **validation** archive 上评估和建账本：

- validation NPZ SHA-256：`1c04b9556b95fbcc050678fc4ee3a1b62b45c9185bc928d904be18745ddfe51c`；
- validation metadata SHA-256：`28be533c4665ada49ccff4f1c1482c538b90e9f6e11a9b1ec3ed362a23e247af`；
- samples：`146,400`；action scale：`5.0`；constant action-chunk length：`3`；candidate count：`5`；
- 不使用 train archive、S3 development scenes、online execution 或 locked data 建 ledger；
- ledger 采用 offline execution-settled validation outcomes，运行时只允许请求 frozen V5 nominal action，之后仍必须通过 CBF。

## 2. Held-out prediction gates

| seed | checkpoint SHA-256 | all finite | auxiliary tasks | target beats CV horizons | gate |
|---:|---|---:|---:|---:|---:|
| 20260911 | `57741bbfdffb806d14043bc8620024f602eb412f7907f81e762e3d6af5b48c4f` | true | present | 4/4 | pass |
| 20260912 | `df9813a49db73216a336d3321ed7b96d8b0c8bddd83f4f786185a1445a6ed31f` | true | present | 3/4 | pass |
| 20260913 | `1318f9b62bc29e287b00e0dd4ded81208f4c00260d165c80b615204f0c1f0118` | true | present | 3/4 | pass |

Target-position MAE improvement relative to constant velocity by horizon (`0.1/0.2/0.3/0.5 s`) is:

| seed | 0.1 s | 0.2 s | 0.3 s | 0.5 s |
|---:|---:|---:|---:|---:|
| 20260911 | +4.56% | +26.96% | +38.21% | +51.34% |
| 20260912 | -29.66% | +10.35% | +29.06% | +45.87% |
| 20260913 | -5.84% | +23.25% | +36.02% | +48.90% |

所有 checkpoint 都满足既定的“至少一个预定义 horizon 优于 constant velocity”准入条件。seed 20260912/20260913 在 `0.1 s` 不优于 constant velocity，已保留为完整结果，不作选择性披露。

## 3. Action-following audit

在 4,096 个 held-out validation samples 上，对归一化最终候选动作施加 `0.02` 的正负轴向扰动。三个 checkpoint 均输出 finite 值，且候选 prediction separation 非零：

| seed | mean separation norm | mean antisymmetry norm | 结论 |
|---:|---:|---:|---|
| 20260911 | 0.00107995 | 0.00002979 | action-sensitive |
| 20260912 | 0.00097084 | 0.00001642 | action-sensitive |
| 20260913 | 0.00109821 | 0.00003115 | action-sensitive |

三 seed 的 antisymmetry 相对 separation 都偏低。审计脚本把这一点定义为 action-following mismatch 的**警告**，而非安全判定；既定协议没有为该比率定义自动拒绝阈值，且既有 P3/P4 已使用同量级诊断。因此本阶段按“非零 sensitivity、无数值不稳定”通过，但将此机制不确定性带入 P5/P6 failure analysis；不得把它描述为因果动作模型的证明。

## 4. Hash-bound reliability ledgers

三份 ledger 的 `source.checkpoint_sha256` 与上表对应 checkpoint SHA-256 严格一致；每份均绑定相同的 v2 validation NPZ/metadata hash。固定 policy 为：horizon index `3`（`0.5 s`）、minimum sample count `128`、minimum credit `0.65`、低信用或 OOD 时回退 frozen V5 nominal action，然后执行 CBF。

| seed | 0.5 s global credit | 0.5 s ranking win rate | validation fallback forecast | ledger decision |
|---:|---:|---:|---:|---|
| 20260911 | 0.8024 | 0.8311 | 0.97% | read-only, hash-bound |
| 20260912 | 0.7874 | 0.8253 | 1.61% | read-only, hash-bound |
| 20260913 | 0.8019 | 0.8247 | 0.84% | read-only, hash-bound |

`fallback forecast` 仅是在 ledger source validation rollouts 上的离线诊断，不是 closed-loop 安全率或捕获率，也不得按 S3 development outcome 更新。

## 5. Reproduction commands

```powershell
$seed = 20260911 # repeat for 20260912 and 20260913
$py = 'D:\miniconda3\envs\uav-encirclement-gpu\python.exe'

& $py scripts\evaluate_jepa_v3_multitask.py `
  --checkpoint results\jepa_v3_multitask_chunk3v2_seed$seed\checkpoint.pt `
  --dataset results\jepa_v3_chunk3v2_counterfactual_validation\counterfactual_multitask_dataset.npz `
  --metadata results\jepa_v3_chunk3v2_counterfactual_validation\metadata.json `
  --output results\jepa_v3_multitask_chunk3v2_seed$seed\prediction_gate.json --device cuda

& $py scripts\audit_jepa_action_following.py `
  --checkpoint results\jepa_v3_multitask_chunk3v2_seed$seed\checkpoint.pt `
  --dataset results\jepa_v3_chunk3v2_counterfactual_validation\counterfactual_multitask_dataset.npz `
  --metadata results\jepa_v3_chunk3v2_counterfactual_validation\metadata.json `
  --sample-count 4096 --perturbation 0.02 --device cuda `
  --output-json results\jepa_v3_multitask_chunk3v2_seed$seed\action_following_audit.json `
  --output-md results\jepa_v3_multitask_chunk3v2_seed$seed\action_following_audit.md

& $py scripts\build_jepa_v3_reliability_ledger.py `
  --checkpoint results\jepa_v3_multitask_chunk3v2_seed$seed\checkpoint.pt `
  --dataset results\jepa_v3_chunk3v2_counterfactual_validation\counterfactual_multitask_dataset.npz `
  --metadata results\jepa_v3_chunk3v2_counterfactual_validation\metadata.json `
  --minimum-sample-count 128 --minimum-credit 0.65 --device cuda `
  --output results\jepa_v3_multitask_chunk3v2_seed$seed\reliability_ledger.json `
  --report results\jepa_v3_multitask_chunk3v2_seed$seed\reliability_ledger_report.md
```

## 6. Decision

P5-B/P5-C 通过，唯一允许的下一步是 P5-D strict zero-perturbation paired regression。若该门失败，v2 checkpoint/ledger 不得进入 non-zero smoke 或 P6；若它通过，仍须先通过 seed-11 non-zero safety/efficiency smoke。无论 gate/ledger 结果如何，CBF 始终是最终安全过滤器。
