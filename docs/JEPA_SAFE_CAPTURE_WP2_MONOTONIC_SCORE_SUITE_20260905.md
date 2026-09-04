# WP2 Synthetic Monotonic Score Suite

**阶段：** ranking contract audit，development-only  
**日期：** 2026-09-05  
**主指标边界：** 本阶段不运行 episode，不产生 safe-capture 性能结论  
**locked 状态：** `locked_test_opened=false`

## 目的

在创建新的 V21 ranking protocol 之前，先用可控的 action-conditioned history stub 检查 ranker 的方向性和优先级。suite 不读取 target ground truth、settled rows 或 locked split，也不修改在线权重、CBF margin、ledger threshold 或 fallback 顺序。

## 结果

最终产物：

- JSON：`results/jepa_safe_capture_v21_monotonic_score_suite_v2/monotonic_score_suite.json`
- Markdown：`results/jepa_safe_capture_v21_monotonic_score_suite_v2/report.md`
- TensorBoard：`results/jepa_safe_capture_v21_tensorboard/monotonic_score_suite_v2/`
- runner：`scripts/run_jepa_safe_capture_monotonic_score_suite.py`
- tests：`tests/test_jepa_safe_capture_monotonic_score_suite.py`

| Case | 检查内容 | 结果 |
|---|---|---:|
| `task_progress` | 任务进展更好时 cost 降低 | pass |
| `uncertainty` | 不确定性更低时 cost 降低 | pass |
| `clearance_gate` | 预测净空低于 `0.15 m` 时候选失去 eligibility | pass |
| `visibility` | 可见性更高时 cost 降低 | pass |
| `ttc` | TTC 更远时 cost 降低 | pass |
| `cbf_risk` | CBF intervention risk 更低时 cost 降低 | pass |
| `fixed_point_tie` | 相同 cost 使用固定 candidate index 稳定打破 tie | pass |

`all_cases_passed=true`。每个 case 的 score、各项 cost、eligible mask、selected index、candidate order 和 ranker config 均写入 JSON；同一 suite 的 case pass、selected index 和 eligible count 写入 TensorBoard。

## 安全解释

`clearance_gate` 中候选 1 具有更好的 target-distance cost，但 predicted minimum clearance 约为 `0.10 m`，低于 `0.15 m` 资格门，因此被标为不 eligible，nominal candidate 0 被选择。这只是预测筛选语义；真实执行仍必须经过同一个 Joint CBF-QP，预测资格不能替代几何安全证明。

## 验证

```text
tests/test_jepa_safe_capture_monotonic_score_suite.py
tests/test_diagnose_jepa_safe_capture_v20_ranking.py
tests/test_jepa_safe_capture_candidates.py
tests/test_jepa_safe_capture_v2_reliability.py
36 passed
```

## 结论边界

本阶段证明了 ranker 的合成输入合同和 TensorBoard 记录链路，不证明 JEPA 在真实冻结场景上提升 `safe_capture`。下一步仍需在独立 calibration evidence 上修订 eligibility/nominal-anchor/abstention protocol，并用 settled counterfactual 和 paired smoke 验证。
