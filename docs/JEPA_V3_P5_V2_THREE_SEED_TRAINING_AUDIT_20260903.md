# JEPA-v3 P5 v2 Three-Seed Training Audit

**阶段：** P5-A  
**日期：** 2026-09-03  
**结论：** `pass; eligible_for_v2_offline_gates`  
**实验性质：** development-only；不是 V4/V5 locked test，也不是闭环捕获提升证据

## 1. 目的与不可变边界

本阶段只验证精度合同修复后的 v2 chunk-3 archive 能否在 RTX 5050 CUDA 环境中完成三次独立、可追溯的 replay-off multitask JEPA 训练。它不使用 S3 development 结果训练，不使用 P4 replay-on 权重，不运行 closed-loop control，也不读取 locked test。

所有 run 使用：

- train archive：`results/jepa_v3_chunk3v2_counterfactual_train/counterfactual_multitask_dataset.npz`，SHA-256 `0d165646db5f0545115fa5f8cdb2bc6fd44b9ab2db5981e8de5b96963e84787c`；
- validation archive：`results/jepa_v3_chunk3v2_counterfactual_validation/counterfactual_multitask_dataset.npz`，SHA-256 `1c04b9556b95fbcc050678fc4ee3a1b62b45c9185bc928d904be18745ddfe51c`；
- frozen V5 actor：`models/v5_development_exact_reactive_seed661606.pt`，SHA-256 `535098773be05687e147043435649378532362d479bdc0375842970370ba40ba`；
- action contract：`action_scale=5.0`、`K=5`、constant desired-action chunk `3` steps、history `8`、input/action dimension `63/3`；
- training contract：40 epochs、batch size 512、learning rate `1e-3`、weight decay `1e-5`、hidden/latent dimension `128/64`、CUDA；
- source hashes：trainer `19e01d796ebd71772852b666e382536c47d83ae6754093f135449a44598dcde8`，prediction runtime `245ed7611c82870748d1cef160a994629f507bda52bcccc6bdf5124f0dfd7a0e`，protocol `7684721d86ae1da55b0bd92cb896e0751b20076cdd233f638f1a15e74c8ccb6b`。

## 2. Training matrix and provenance

| seed | output directory | TensorBoard logdir | best epoch | best validation loss | elapsed (s) | checkpoint SHA-256 |
|---:|---|---|---:|---:|---:|---|
| 20260911 | `results/jepa_v3_multitask_chunk3v2_seed20260911/` | `results/jepa_v3_tensorboard/multitask_chunk3v2_seed20260911/` | 7 | -3.4680395546 | 419.32 | `57741bbfdffb806d14043bc8620024f602eb412f7907f81e762e3d6af5b48c4f` |
| 20260912 | `results/jepa_v3_multitask_chunk3v2_seed20260912/` | `results/jepa_v3_tensorboard/multitask_chunk3v2_seed20260912/` | 3 | -3.3252068600 | 409.41 | `df9813a49db73216a336d3321ed7b96d8b0c8bddd83f4f786185a1445a6ed31f` |
| 20260913 | `results/jepa_v3_multitask_chunk3v2_seed20260913/` | `results/jepa_v3_tensorboard/multitask_chunk3v2_seed20260913/` | 8 | -3.4427366556 | 415.59 | `1318f9b62bc29e287b00e0dd4ded81208f4c00260d165c80b615204f0c1f0118` |

每个 output directory 都包含 `checkpoint.pt`、`history.json` 和 `run_metadata.json`；每个 history 都恰有 40 条 epoch 记录。运行环境均为 Python `3.11.16`、PyTorch `2.7.1+cu128`、Windows 10，在 `uav-encirclement-gpu` Conda 环境的 NVIDIA RTX 5050 CUDA device 上完成。三次训练均明确记录 `replay.enabled=false`。

## 3. TensorBoard audit

每个 seed 都使用独立 event file，且 TensorBoard audit 结果一致：

| seed | train loss epochs | histogram tags | missing required scalar tags | missing required text artifacts | complete |
|---:|---:|---:|---:|---:|---:|
| 20260911 | 40 | 149 | 0 | 0 | true |
| 20260912 | 40 | 149 | 0 | 0 | true |
| 20260913 | 40 | 149 | 0 | 0 | true |

已记录的 scalar group 包括 `Loss/*`、`Target/*`、`Clearance/*`、`Visibility/*`、`Risk/*`、`Calibration/validation` 和 `Optimization/learning_rate`。Text artifacts 包括 protocol、train metadata、validation metadata、source hashes、model 和 optimizer 配置。参数及梯度 histogram 按既定间隔记录，审计中没有 NaN、Inf 或提前结束证据。

## 4. Regression tests

执行：

```powershell
& D:\miniconda3\envs\uav-encirclement-gpu\python.exe -m pytest `
  tests/test_jepa_v3_counterfactual_dataset.py `
  tests/test_jepa_v3_multitask.py `
  tests/test_prediction.py `
  tests/test_jepa_v3_zero_perturbation.py -q
```

结果：`18 passed in 2.10s`。

## 5. Decision

三个 v2 run 全部完成且通过 TensorBoard/provenance/基本回归测试，允许进入 P5-B 的 held-out prediction gate 和 action-following audit。这个准入不说明三 seed 的闭环收益，也不允许跳过 P5-D zero-perturbation strict regression。

旧 v1 chunk-3 checkpoint 及其 ledger 仍保持 `superseded_invalid_precision_contract` 状态；后续命令必须只引用本报告中的 v2 checkpoint 哈希。
