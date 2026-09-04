# P2 Settled Ranking 与 Abstention 审计

**日期：** 2026-09-04
**范围：** `development_only=true`，`locked_test_opened=false`
**设备：** NVIDIA GeForce RTX 5050，CUDA 12.8，PyTorch 2.7.1+cu128
**审计输入：** 当前 V5/P1 M3 20 集、1,075 个 control cycles，以及同一冻结场景上的离线 settled counterfactual replay

## 1. 目的

本审计检查 action-conditioned JEPA 的候选排序是否与离线局部 settled outcome 对齐，并把
abstention、nominal anchor、候选切换、reliability credit 和 CBF abort 前状态放到同一份可重放
证据中。离线 settled outcome 只用于诊断，不会反馈给在线 evaluator，也不是完整 episode 的
策略结果。

## 2. 输入与可复现性

| 项目 | 路径/哈希 |
|---|---|
| 在线 trace run | `results/jepa_safe_capture_v5_p1_latency_m3_seed20260911/` |
| settled counterfactual | `results/jepa_safe_capture_v5_p2_settled_cf_m3_seed20260911/` |
| 聚合审计 | `results/jepa_safe_capture_v5_p2_ranking_audit_m3_seed20260911_v2/` |
| TensorBoard | `results/jepa_safe_capture_v5_tensorboard/p2_ranking_audit_m3_seed20260911_v2/` |
| ledger | `results/jepa_safe_capture_v4_t3_ledger_seed20260911_r2/reliability_ledger.json` |
| 结构测试 | `14 passed`（P2、settled counterfactual、历史 rank mismatch） |

审计器逐行校验 trace 与 settled rows 的 `(episode_index, step)`、training seed、variant、scene
manifest hash、development boundary 和 source raw-action gate。所有结构 gates 通过，且
`raw_unverified_executed=0`。

## 3. 结果

| Reliability bucket | Decisions | Selected-not-settled-best | Safety failure | CBF abort | Mean top-two margin (m) | Mean nominal displacement (m/s) |
|---|---:|---:|---:|---:|---:|---:|
| high (`credit >= 0.65`) | 1,052 | 26.8% | 11.1% | 1.0% | 0.001969 | 0.009884 |
| low/missing | 23 | 0.0% | 100.0% | 4.3% | n/a | approximately 0 |

全 run 的 settled ranking 结果为：

- selected-not-best：`282/1,075 = 26.23%`；
- settled rank Spearman：`-0.517`，Kendall：`-0.435`；
- source M3 safe-capture：`8/20 = 40.0%`；
- source collision、boundary、pairwise、CBF timeout 和 raw-unverified 均为 `0`；
- CBF abort 前状态共 `11` 条，均保留 selected/predicted-best/settled-best、top-two margin、
  nominal displacement、credit、execution mode 和 abort reason。

## 4. 解释

当前问题不是 CBF 放行了未经验证的动作，而是排序信号没有稳定转化为任务收益：

1. 高 credit 片段占绝大多数，但仍有约四分之一的 selected action 不是离线 settled 最优候选；
2. 低 credit 片段主要回退 nominal，说明 ledger 的安全拒答路径生效，但任务推进几乎为零；
3. top-two margin 较小且 nominal displacement 很低，表明候选差异常接近，排序噪声和 abstention
   语义会直接影响闭环选择；
4. 该结果只能归档为 `no_control_gain`，不能写成 JEPA 的 safe-capture 提升，也不能据此打开
   locked test。

## 5. 下一步

- [x] 完成 settled rows、online trace 和 provenance 的逐步 join；
- [x] 输出 selected-not-best、top-two margin、nominal displacement、candidate switch、
  credit bucket 和 CBF-abort pre-state；
- [x] 写入独立 TensorBoard 审计，保留配置、输入哈希、结构 gates 和混淆矩阵；
- [ ] 在独立 calibration evidence 上冻结或修订 `tie tolerance`、abstention margin、
  minimum predicted clearance、hysteresis 和 minimum hold steps；
- [ ] 只有新 protocol 通过 smoke 后，才运行三 seed paired development。

任何 ranking 权重、阈值或 hold 规则的改变都必须生成新的 protocol、scene manifest、ledger
revision、checkpoint/hash 和 TensorBoard run。`safe_capture` 仍是主指标，`mean_capture_time`
仅作为诊断。

## 6. 运行命令

```powershell
Set-Location D:\\uav-capture\\uav_capture
$env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"
& D:\\miniconda3\\envs\\uav-encirclement-gpu\\python.exe `
  scripts/audit_jepa_safe_capture_v5_p2_ranking.py `
  --trace-run results/jepa_safe_capture_v5_p1_latency_m3_seed20260911 `
  --settled-run results/jepa_safe_capture_v5_p2_settled_cf_m3_seed20260911 `
  --ledger results/jepa_safe_capture_v4_t3_ledger_seed20260911_r2/reliability_ledger.json `
  --output-dir results/jepa_safe_capture_v5_p2_ranking_audit_m3_seed20260911_v2 `
  --tensorboard-logdir results/jepa_safe_capture_v5_tensorboard/p2_ranking_audit_m3_seed20260911_v2 `
  --development-only
```
