# V21 三 Seed Settled Ranking Aggregate

**阶段：** S1 / development-only
**代码 revision：** `b9693b4bb071b2e64545fe832188604707b10d41`
**协议 SHA-256：** `278623ceb7185a6c3ce23246e8a28693f025a2977fad95059ae5b0df9a03b014`
**环境配置 SHA-256：** `42bd4e158c5e314e0ece6add8038b32c384a7a2ca027e9387327656fccf751ad`
**TensorBoard：** `results/jepa_safe_capture_v21_current_tensorboard/settled_aggregate_v3/`
**Locked test：** 未打开

## 1. 输入与审计边界

聚合器读取三个已完成的 settled counterfactual 输出：

```text
results/jepa_safe_capture_v21_settled_seed20260911/
results/jepa_safe_capture_v21_settled_seed20260912/
results/jepa_safe_capture_v21_settled_seed20260913/
```

每个 seed 内的 M3/A1/A2 共用自己的 scene manifest；跨 seed 只要求 protocol 和 environment hash 一致。聚合器拒绝缺 seed、重复 `(variant, episode, step)`、源 gate 失败、hash 不一致或 decision count 不一致。它不推进模拟器，不改变 source trace，target truth 仅存在于 offline settled label。

## 2. 聚合结果

| Variant | Decisions | Selected-not-best | Selected settled safe | Best settled safe | Settled unsafe | Separation pass | Spearman | Kendall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M3 | 3182 | 41.6% | 1.9% | 2.0% | 2.6% | 89.1% | -0.307 | -0.290 |
| A1 | 3465 | 40.8% | 1.4% | 1.5% | 2.7% | 87.4% | -0.293 | -0.277 |
| A2 | 3337 | 41.1% | 1.6% | 1.7% | 2.4% | 88.6% | -0.317 | -0.298 |

### M3 per seed

| Seed | Decisions | Selected-not-best | Spearman | Separation pass |
|---:|---:|---:|---:|---:|
| 20260911 | 1189 | 35.3% | -0.319 | 91.9% |
| 20260912 | 917 | 39.1% | -0.294 | 87.7% |
| 20260913 | 1076 | 50.7% | -0.306 | 87.9% |

## 3. Gate decision

- Source settled gates：**PASS**。
- Ranking promotion gate：**FAIL**。
- 失败原因：M3 aggregate selected-not-best 高于 25%，seed 20260913 高于 40%，且三个 seed 的 Spearman 均为负。
- 当前标签：`ranking_unresolved`。
- 下一步：failure index、high-credit failure 分桶和 deterministic hard replay。

这份结果只说明当前 JEPA score 与 settled local-chunk outcome 存在系统失配；它不代表完整 episode policy outcome，也不改变 V4/V5 历史正式结论。不得用降低 CBF margin、放宽 ledger gate 或删除 controlled abort 来修正该结果。

## 4. 可复现实验与测试

聚合命令：

```powershell
$py = 'D:\download\anaconda3\envs\traj_pred_prep\python.exe'
$env:PYTHONPATH = "$PWD\src;$PWD\scripts"
& $py scripts/run_with_tensorboard_compat.py scripts/aggregate_jepa_safe_capture_v21_settled.py `
  --input-root results `
  --output-dir results/jepa_safe_capture_v21_current_settled_aggregate_v3 `
  --tensorboard-dir results/jepa_safe_capture_v21_current_tensorboard/settled_aggregate_v3 `
  --development-only
```

相关回归测试：`21 passed`，覆盖聚合器、settled counterfactual、candidate separation、rolling horizon 和 hash-bound ledger。
