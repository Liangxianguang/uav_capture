# JEPA-v3 P5 v2 Archive Audit

**阶段：** P5-0  
**日期：** 2026-09-03  
**结论：** `pass; admissible_for_v2_training`  
**实验性质：** development-only；不是 V4/V5 locked test

## 1. 审计对象

本审计针对精度合同修复后重新采集的 chunk-3 counterfactual archive。旧版 `jepa_v3_chunk3_counterfactual_*` 不在本审计范围内，也不得与当前 runtime 混用。

| split | directory | NPZ SHA-256 | metadata SHA-256 |
|---|---|---|---|
| train | `results/jepa_v3_chunk3v2_counterfactual_train/` | `0d165646db5f0545115fa5f8cdb2bc6fd44b9ab2db5981e8de5b96963e84787c` | `6d6ae07cd74981dc9097a38cc3dc43bcac6baff82aed838e126f69bd963542c4` |
| validation | `results/jepa_v3_chunk3v2_counterfactual_validation/` | `1c04b9556b95fbcc050678fc4ee3a1b62b45c9185bc928d904be18745ddfe51c` | `28be533c4665ada49ccff4f1c1482c538b90e9f6e11a9b1ec3ed362a23e247af` |

Scenario-manifest hashes：

- train：`26a1cea0e95d3d77926232fd91863d148653a45255670d978ab20fd57b353ed1`
- validation：`48a3227a434e7db86e4d47a7a9521ab020b35a0bbba5892e672d3dfb7cef8737`

审计机器：NVIDIA RTX 5050；Conda Python `D:\miniconda3\envs\uav-encirclement-gpu\python.exe`；环境配置和源代码哈希记录在各自 metadata 中。

## 2. 固定合同检查

| 检查项 | 结果 |
|---|---:|
| train samples | `146400` |
| validation samples | `146400` |
| input shape | `N × 8 × 63` |
| action shape | `N × 8 × 3` |
| target horizons | `1, 2, 3, 5` steps |
| candidate groups | `29280` per split |
| candidates per state-agent group | exactly `5` |
| nominal candidate fraction | `0.2` |
| chunk length | `3` steps |
| candidate perturbation | `0.10 m/s` |
| action scale | `5.0`, matching frozen V5 actor |
| maximum normalized action magnitude | `0.999973` |
| all arrays finite | `true` |
| train/validation episode seeds disjoint | `true` |
| development/locked data used for training | `false` |
| target truth usage | offline labels only |

Scenario sample counts are balanced at `36,600` samples for each of the four collection scenarios in each split. Candidate indices are balanced at `29,280` samples each.

## 3. Reproduction command

```powershell
$py = 'D:\miniconda3\envs\uav-encirclement-gpu\python.exe'

& $py scripts\audit_jepa_v3_counterfactual_dataset.py `
  --dataset-dir results\jepa_v3_chunk3v2_counterfactual_validation `
  --compare-dataset-dir results\jepa_v3_chunk3v2_counterfactual_train `
  --output results\jepa_v3_chunk3v2_dataset_audit.json
```

机器可读审计结果保存在 `results/jepa_v3_chunk3v2_dataset_audit.json`。该结果目录默认不纳入 Git；本报告固化关键结果和哈希。

## 4. 决策

v2 archive 通过 P5-0，可用于后续三 seed replay-off multitask training。训练必须使用新命名 `jepa_v3_multitask_chunk3v2_seed<seed>`，不得复用旧 v1 checkpoint、旧 v1 TensorBoard 或旧 v1 ledger。

本审计不证明 JEPA 的闭环捕获提升，也不授权读取或运行任何 locked test。
