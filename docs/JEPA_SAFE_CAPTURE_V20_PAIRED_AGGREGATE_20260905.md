# v20 M0/M3 三 seed paired development aggregate

**阶段：** WP2 / development-only  
**协议 SHA-256：** `b8a492faa9448bb0917c124908a044af6cb10813847afeedb78ce675446a2b99`  
**训练 seed：** `20260911`、`20260912`、`20260913`  
**统计单位：** `(training_seed, episode)`，共 60 个 episode pair  
**locked 状态：** `locked_test_opened=false`

## 配对结果

| Seed | M0 safe capture | M3 safe capture | Improved | Degraded | Tied | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 20260911 | 10/20 (50%) | 9/20 (45%) | 1 | 2 | 17 | -5 pp |
| 20260912 | 10/20 (50%) | 7/20 (35%) | 1 | 4 | 15 | -15 pp |
| 20260913 | 10/20 (50%) | 9/20 (45%) | 0 | 1 | 19 | -5 pp |
| **Pooled** | **30/60 (50%)** | **25/60 (41.7%)** | **2** | **7** | **51** | **-8.33 pp** |

Pooled paired bootstrap 95% CI 为 `[-18.33, +1.67] pp`。逐 seed exact McNemar 双侧 p 值分别为 `1.0000`、`0.3750`、`1.0000`。所有统计均保留 `controlled_abort` 在失败分母中。

## 安全审计

M0 和 M3 的三 seed 运行均为：

- collision：`0`
- defender boundary violation：`0`
- pairwise violation：`0`
- raw/unverified executed steps：`0`
- CBF timeout：`0`

CBF infeasible 请求都被路由到 fallback，且 `cbf_controlled_abort_steps == cbf_unverified_steps`。因此安全硬门和 reliability observability gate 通过；这不意味着 M3 的任务性能通过。

## 结论

当前 M3 在冻结 v20 replay 上没有显示 safe-capture 提升，三 seed 均为负 delta。结论分类为 `useful_safety_fallback_only`，而不是性能提升或 non-inferiority。该结果与 settled ranking audit 的负相关相互印证，下一步必须完成 score orientation、task/clearance/CBF-risk 符号、settled label、horizon 和 action-scale 诊断。诊断完成并通过人工单调性测试前，不得扩大到 40/60 集，也不得打开 locked test。

## 产物

- JSON：`results/jepa_safe_capture_v20_cpu_deterministic_paired_aggregate_v1/paired_aggregate.json`
- episode 配对 CSV：`results/jepa_safe_capture_v20_cpu_deterministic_paired_aggregate_v1/paired_episode_rows.csv`
- 配对明细：`results/jepa_safe_capture_v20_cpu_deterministic_paired_aggregate_v1/paired_comparisons.json`
- Markdown：`results/jepa_safe_capture_v20_cpu_deterministic_paired_aggregate_v1/report.md`
- TensorBoard：`results/jepa_safe_capture_v20_cpu_deterministic_tensorboard/paired_aggregate_v1/`
- 聚合脚本：`scripts/aggregate_jepa_safe_capture_v20_paired.py`
