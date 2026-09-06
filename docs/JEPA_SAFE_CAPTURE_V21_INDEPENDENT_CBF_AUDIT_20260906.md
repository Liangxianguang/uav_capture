# V21 Independent CBF Counterfactual Audit

**日期：** 2026-09-06  
**状态：** development-only；`locked_test_opened=false`  
**主指标：** `safe_capture`  
**目的：** 判断 V21 `controlled_abort` 是 selected candidate 的问题，还是当前状态下整个安全可行域都已不可行。

## 1. 执行合同

本阶段使用同一 V21 validation scene manifest、environment config、actor checkpoint、JEPA checkpoint、Ledger 和 CBF 参数，对三 seed、四变体重新运行 20 episodes。新增内容只有只读的三路 CBF counterfactual：

```text
selected request
reachable nominal anchor
current-velocity safe-hold request
```

每一路均调用相同的 `verify_requested_action` primary Joint CBF solve；不执行 counterfactual，不读取 target future truth，不改变 CBF margin、stale/OOD gate 或 `controlled_abort` 语义。

## 2. 结果

| 变体 | Seed 20260911 | Seed 20260912 | Seed 20260913 | 平均 |
|---|---:|---:|---:|---:|
| M0 | 10/20 (50%) | 10/20 (50%) | 10/20 (50%) | 50.0% |
| M3 | 10/20 (50%) | 11/20 (55%) | 7/20 (35%) | 46.7% |
| A1 | 10/20 (50%) | 10/20 (50%) | 8/20 (40%) | 46.7% |
| A2 | 11/20 (55%) | 9/20 (45%) | 9/20 (45%) | 48.3% |

所有 12 个 run 的安全硬门均通过：

```text
collision = 0
defender boundary violation = 0
pairwise violation = 0
raw-unverified execution = 0
```

总计 107 个 `controlled_abort`，共记录 321 个 abort counterfactual（每个 abort 三路；另有完整 step-level 三路记录）。在 107 个 abort 中：

| 分类 | Episodes |
|---|---:|
| `all_three_infeasible` | 107 |
| `selected_only_infeasible` | 0 |
| 其他 | 0 |

首个负 slack 的约束类别：

| 类别 | Episodes |
|---|---:|
| obstacle | 50 |
| pairwise | 34 |
| boundary | 23 |

三路 counterfactual 均无 timeout；每个 run 的独立探针 action 都是 finite。

## 3. 解释

本阶段没有证据支持“只有 JEPA 选错 candidate 导致 abort”。在记录到的 107 个 abort 状态，selected、nominal 和 safe-hold 的 primary CBF 请求都不可验证。因此当前主要瓶颈是：进入 abort 状态时已经没有满足固定动态约束的恢复动作，而不是简单的 score direction 或 nominal anchor 选择。

这不等于证明早一周期的路线选择最优，也不等于应该放宽 CBF。下一阶段应优先改进：

1. 更早触发可达的绕障/刹车/切向候选；
2. 增加候选块覆盖并保持同一 reachable projection；
3. 在 horizon 内评估最早 CBF 风险，而不是等到当前步不可行；
4. 用这些 abort 状态生成边界、pairwise 和 feasibility 负样本，再重训新 JEPA。

## 4. 与历史 116 个 abort 的关系

历史 `results/jepa_safe_capture_v21_failure_index_v4` 记录了旧 revision 下的 116 个 abort。本报告的 107 个 abort 来自当前 `26b5d62` revision 的新 rerun；虽然 scene manifest、协议和输入 checkpoint 保持绑定，但不能把两个数量直接合并，也不能据此声称 episode-level 结果完全 bitwise identical。历史结果继续保留为历史证据，本报告是当前执行合同下的独立审计证据。

## 5. 阶段裁决

当前结果未通过 M3 safe-capture 提升门：M3 相对 M0 的 paired deltas 为 `0 pp`、`+5 pp`、`-15 pp`，平均 `-3.3 pp`，只有 1 个 seed 提升。因此停止扩大 episode 数、停止打开 locked test，也不直接开始多任务 JEPA 训练。

在进入 WP3/WP4 前，必须先建立新的 route-aware negative counterfactual archive，并明确记录 route identity、最早 horizon failure、candidate eligibility 和 recovery feasibility。只有新 archive、checkpoint、calibration 和 Ledger 全部重新 hash-bind 后，才允许进行下一轮三 seed smoke。

## 6. 产物

- [独立审计 JSON](../results/jepa_safe_capture_v21_independent_cbf_audit_v1/independent_cbf_audit.json)
- [abort CSV](../results/jepa_safe_capture_v21_independent_cbf_audit_v1/independent_cbf_abort.csv)
- [TensorBoard](../results/jepa_safe_capture_v21_independent_cbf_tensorboard_v1/)
- [审计脚本](../scripts/audit_jepa_safe_capture_v21_independent_cbf.py)
