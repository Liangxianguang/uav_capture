# V21 三 seed paired smoke 汇总报告

**日期：** 2026-09-05

**阶段：** P4，development-only

**聚合输出：** `results/jepa_safe_capture_v21_smoke_aggregate/`

**TensorBoard：** `results/jepa_safe_capture_v21_smoke_aggregate/tensorboard/`

**protocol SHA-256：** `278623ceb7185a6c3ce23246e8a28693f025a2977fad95059ae5b0df9a03b014`

## 1. 主结果

| 变体 | seed 20260911 | seed 20260912 | seed 20260913 | 均值 +/- 样本 SD |
|---|---:|---:|---:|---:|
| M0 | 10/20 | 10/20 | 10/20 | 50.0% +/- 0.0% |
| M3 | 12/20 | 7/20 | 9/20 | 46.7% +/- 12.6% |
| A1 | 10/20 | 10/20 | 9/20 | 48.3% +/- 2.9% |
| A2 | 11/20 | 9/20 | 10/20 | 50.0% +/- 5.0% |

M3 相对同 seed M0 的 paired 结果：

| seed | M0 | M3 | delta | improved | degraded | McNemar exact p |
|---:|---:|---:|---:|---:|---:|---:|
| 20260911 | 10/20 | 12/20 | +10.0 pp | 2 | 0 | 0.5000 |
| 20260912 | 10/20 | 7/20 | -15.0 pp | 0 | 3 | 0.2500 |
| 20260913 | 10/20 | 9/20 | -5.0 pp | 1 | 2 | 1.0000 |
| **aggregate** | **30/60** | **28/60** | **-3.33 pp** | **3** | **5** | — |

episode-pair bootstrap 95% CI 为 `[-11.67, +5.00] pp`；只有 `1/3` seed 非负。

## 2. 安全与可观测性门

- M0/M3/A1/A2 的 collision、boundary、pairwise violation 均为 `0`。
- 所有 run 的 `raw_unverified_executed_steps=0`。
- 所有 CBF timeout 均为 `0`。
- 每个 CBF infeasible request 都进入 fallback；每个 unverified result 都是 controlled abort。
- TensorBoard 为每个 run 写入安全率、raw-unverified、controlled-abort 和 cycle p95，并保存 protocol、输入和决策 text provenance。

## 3. 结论

本轮 V21 smoke 的安全合同通过，但 M3 没有显示稳定的 safe-capture 控制收益，正式分类为 `useful_safety_fallback_only`。当前结果支持“JEPA + ledger + CBF 能在不牺牲安全硬门的情况下运行”，不支持“JEPA 已提升 safe-capture”。

因此：

- 不进入 40/60 集 paired development；
- 不打开 locked test；
- 不以 `mean_capture_time` 或 Transit 改写主结论；
- 不降低 CBF margin、不放宽 stale/OOD、不删除 controlled abort；
- 下一步转入 settled ranking、candidate separation、failure index 和困难片段因果 replay。
