# WP3 Ranking、Eligibility 与 Abstention 审计

**阶段：** development-only offline audit  
**日期：** 2026-09-05  
**locked 状态：** `locked_test_opened=false`  
**主指标边界：** 本阶段不打开 locked test，不把局部 settled branch 写成完整 episode 性能

## 1. 目的

WP2 的合成单调性 suite 已通过，但 V20 的真实 trace 仍显示大量 candidate fallback。WP3 分开评估：

1. 降低 predicted-clearance eligibility floor 是否能恢复有意义的候选分离；
2. 完全取消 top-two abstention/nominal anchor 后，score argmin 是否安全；
3. 一个仅在预测安全量不劣于 nominal 时允许替换的 guarded policy 是否值得进入 online smoke。

所有分析都消费冻结 V20 trace 和 settled counterfactual，未修改源 replay、CBF margin、ledger threshold 或在线动作。

## 2. Clearance-floor sensitivity

seed `20260911` 的离线 floor 曲线：

| 预测净空 floor | all-ineligible | multi-eligible | selected-not-best | mean Spearman |
|---:|---:|---:|---:|---:|
| `0.15 m`（V20） | 60.4% | 453 | 35.8% | -0.523 |
| `0.10 m` | 54.7% | 523 | 41.5% | -0.507 |
| `0.05 m` | 49.3% | 587 | 46.7% | -0.475 |
| `0.00 m` | 44.6% | 640 | 51.2% | -0.463 |

三 seed 的 `0.10 m` sensitivity：

| Seed | all-ineligible | selected-not-best | settled safety | mean progress | Spearman |
|---:|---:|---:|---:|---:|---:|
| 20260911 | 54.7% | 41.5% | 43.1% | 0.5297 m | -0.5065 |
| 20260912 | 39.4% | 55.9% | 57.6% | 0.5470 m | -0.5622 |
| 20260913 | 20.0% | 74.1% | 77.8% | 0.4123 m | -0.6183 |

结论：降低 floor 确实减少 all-ineligible，但没有稳定改善排序；在 seed 20260913 上 selected-not-best 反而更高。`0.00 m` 没有资格作为安全默认值，因为负预测净空不能被解释为安全候选。

产物：

- `results/jepa_safe_capture_v21_clearance_sensitivity_floor010_seed20260911/`
- `results/jepa_safe_capture_v21_clearance_sensitivity_floor010_seed20260912/`
- `results/jepa_safe_capture_v21_clearance_sensitivity_floor010_seed20260913/`
- `results/jepa_safe_capture_v21_clearance_sensitivity_floor005_seed20260911/`
- `results/jepa_safe_capture_v21_clearance_sensitivity_floor000_seed20260911/`
- 对应 TensorBoard：`results/jepa_safe_capture_v21_tensorboard/clearance_sensitivity_*`

## 3. Abstention counterfactual

对三 seed 的 V20 settled rows，比较 source trace 的 selected 和只按 finite score argmin 的离线策略：

| Policy | multi-eligible decisions | selected-not-best | settled safety | settled safe-capture | mean progress |
|---|---:|---:|---:|---:|---:|
| recorded selected（含 abstention/anchor） | 1878 | 92.4% | 97.2% | 3.11% | 0.4744 m |
| score argmin（去掉 abstention/anchor） | 1878 | 42.0% | 96.6% | 3.09% | 0.4853 m |

score argmin 与 settled-best 一致率为 `58.0%`，而 recorded selected 与 score argmin 一致率仅 `10.9%`。但取消 abstention 并没有带来 safe-capture 提升，还降低了局部 settled safety，因此不能直接创建“关闭 abstention”的在线 protocol。

对 nominal eligible rows 的 guarded policy 离线检查：只有 candidate 的预测 clearance 和 CBF-risk 都不劣于 nominal 时才允许替换。该规则的 settled safety 约 `97.2%`，但 candidate replacement 极少、safe-capture 不变，当前不足以支持 online smoke。

主诊断脚本：

- `scripts/audit_jepa_safe_capture_v5_settled_counterfactual.py --eligibility-floor ...`
- `scripts/diagnose_jepa_safe_capture_v21_abstention_counterfactual.py`
- `tests/test_jepa_safe_capture_v21_abstention_counterfactual.py`
- `results/jepa_safe_capture_v21_abstention_counterfactual/`
- `results/jepa_safe_capture_v21_tensorboard/abstention_counterfactual/`

## 4. 当前决策

- 保持 V20 的 CBF margin、controlled abort、fallback 顺序和 `locked_test_opened=false`。
- 不采用 `0.00 m` 或 `0.05 m` floor。
- 暂不在线关闭 top-two abstention；score argmin 只能作为离线上限诊断。
- 下一步优先修复非 finite JEPA 输出的显式 safe-hold 路径，并补充 fault-injection/replay 证据。
- 之后再训练/校准多任务安全头；新的在线 protocol 只有在 safety fault gate、settled ranking 和 TensorBoard/provenance gate 全部通过后创建。

## 5. 证据边界

本报告证明的是 eligibility 和 abstention 的局部诊断，不是完整 episode safe-capture 结果，也不是新的 locked-test 结论。所有 `controlled_abort` 和未 settled branch 都保留在 trace；任何 future online variant 仍必须经过同一个 Joint CBF-QP。
