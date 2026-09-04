# JEPA Safe-Capture WP-7 tie3 完整开发实验归档

**日期：** 2026-09-04
**状态：** development-only；`locked_test_opened=false`
**硬件：** NVIDIA GeForce RTX 5050；PyTorch 2.7.1+cu128
**协议：** tie3，`score_tie_tolerance_m=5e-4`
**实验规模：** 7 变体 x 3 seed x 40 validation episodes = 840 集

## 1. 实验合同

- JEPA 只评价传统规划器生成的 5 个候选 action chunks，不直接生成最终动作。
- 每个 chunk 为 3 个 control steps，只执行第一步，然后重新观测、重规划和过滤。
- 所有安全保留变体的 nominal、candidate 和 fallback 均经过同一个 Joint CBF-QP。
- CBF 不可行、timeout、non-finite、stale 或 OOD 时只能进入已验证的 safe-hold/nominal fallback；raw action 不得执行。
- A3 是 raw/no-CBF 诊断，不能纳入安全主结论。
- `safe_capture` 是第一指标；capture time 只作诊断，不抵消安全失败。

## 2. 完整结果

| 变体 | 含义 | safe-capture（3 seed 均值 +/- 样本 SD） | 碰撞 | defender 越界 | pairwise | CBF infeasible |
|---|---|---:|---:|---:|---:|---:|
| M0 | nominal planner + CBF | 50.0% +/- 0.0% | 0 | 0 | 0 | 51 |
| M1 | JEPA target/uncertainty + CBF | 39.2% +/- 3.8% | 0 | 0 | 0 | 72 |
| M2 | JEPA + ledger + target/uncertainty + CBF | 40.0% +/- 2.5% | 0 | 0 | 0 | 68 |
| M3 | JEPA + ledger + auxiliary safety ranking + CBF | 33.3% +/- 6.3% | 0 | 0 | 0 | 77 |
| A1 | M3 去掉 ledger + CBF | 29.2% +/- 7.2% | 0 | 0 | 0 | 84 |
| A2 | M3 去掉 clearance/visibility ranking + CBF | 35.8% +/- 10.1% | 0 | 0 | 0 | 72 |
| A3 | raw/no-CBF 诊断 | 0.0% +/- 0.0% | 120 | 0 | 39 | 0 |

M0、M1、M2、M3、A1 和 A2 的安全硬门均通过：碰撞、defender boundary 和
pairwise violation 全为 0，Transit 均为 100%。A3 三个 seed 共 120/120 集碰撞，
并有 39 集 pairwise violation，说明 CBF 是不可绕过的执行边界。

## 3. M3 配对结论

每个 M3 episode 与相同 seed、episode index、场景、目标运动和观测条件的 M0 episode
配对：

| seed | M0 | M3 | 配对 delta | improved | degraded | McNemar exact p |
|---:|---:|---:|---:|---:|---:|---:|
| 20260911 | 20/40 | 16/40 | -10.0 pp | 4 | 8 | 0.3877 |
| 20260912 | 20/40 | 13/40 | -17.5 pp | 4 | 11 | 0.1185 |
| 20260913 | 20/40 | 11/40 | -22.5 pp | 2 | 11 | 0.0225 |

跨 120 个配对 episode：improved/degraded/tied 为 `10/30/80`，平均配对差值为
`-16.7 pp`，固定 bootstrap 95% CI 为 `[-26.7, -7.5] pp`，非负 seed 为 `0/3`。
因此当前 M3 不能称为任务性能提升，也不能打开 locked test。

## 4. 可靠性和故障索引

聚合器的安全硬门为 PASS，但 reliability observability gate 为 FAIL，原因是 M0
seed 20260911 存在 1 个 CBF timeout。WP-8 故障索引（只读，不修改原始 run）显示：

- 840 集中 567 集未安全捕获；主因是 421 集 `cbf_controlled_abort`、26 集 timeout、
  120 集 A3 collision。
- M3 有 80 集 high-credit failure、31 集 fallback episode。
- 全矩阵有 197 集 `candidate_capture_regression`、92 集 candidate oscillation。
- 预测净空差距和可见性差距分别出现 28 和 19 集；当前 trace 没有 offline future
  target label，不能把它们写成 target drift 的因果证明。
- M3 的平均 candidate switch rate 约为 0.180，高于 M1/M2，提示排序在困难片段中
  过度切换或缺少稳定性惩罚。

机器可读产物：

- `results/wp7_tie3_aggregate/summary.json`
- `results/wp7_tie3_aggregate/report.md`
- `results/wp7_tie3_aggregate/run_metrics.csv`
- `results/wp7_tie3_aggregate/paired_comparison.json`
- `results/wp8_failure_index_tie3/failure_index.json`
- `results/wp8_failure_index_tie3/failure_index.csv`

## 5. 结论分类

本轮分类为 `insufficient_evidence_or_reject`：安全执行架构得到支持，A3 证明了 CBF
必要性，但 JEPA + ledger + auxiliary ranking 当前相对冻结 M0 发生一致的 safe-capture
回归，且 reliability observability gate 未通过。下一轮应定位失败机制并新建 protocol；
不得通过挑选 seed、修改统计口径或只报告 capture time 来掩盖回归。

## 6. 可复现命令

```powershell
$py='D:\\miniconda3\\envs\\uav-encirclement-gpu\\python.exe'
$env:PYTHONPATH="$PWD\\src;$PWD\\scripts"
& $py scripts/aggregate_jepa_safe_capture_v2_paired.py `
  --input-root results/wp7_tie3 `
  --output-dir results/wp7_tie3_aggregate `
  --stage full `
  --development-only
& $py scripts/index_jepa_safe_capture_failures.py `
  --input-root results/wp7_tie3 `
  --output-dir results/wp8_failure_index_tie3 `
  --tensorboard-logdir results/jepa_safe_capture_v3_tensorboard/wp8_failure_index_tie3 `
  --stage full `
  --input-format v2 `
  --development-only
```

相关完整执行计划见
`docs/JEPA_SAFE_CAPTURE_NEXT_EXECUTION_PLAN_20260904.md`。
