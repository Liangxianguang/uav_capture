# WP2：V21 固定点排序与候选分离审计

**日期：** 2026-09-05
**阶段：** development-only offline audit
**协议：** `configs/central_random_mixed_obstacle_s3_v5_v21_cpu_separation_gate_development_protocol.yaml`
**协议 SHA-256：** `25b17b915ccf8e7d97250c4e87520e8d8e1e7cd11857ed971037b81e92f45239`
**locked 状态：** `locked_test_opened=false`
**结果目录：** `results/jepa_safe_capture_v21_wp2_candidate_separation_audit/`
**TensorBoard：** `results/jepa_safe_capture_v21_tensorboard/wp2_candidate_separation/`

> 本阶段只消费冻结 V20 CPU/CUDA trace 和 settled rows，计算 V21 候选分离门的离线反事实。没有执行新的策略动作、没有改变 checkpoint/ledger/CBF margin，也不是新的 full-episode safe-capture 结果。

## 1. 审计目的

验证以下合同：

1. candidate separation 使用 candidate-specific nearest-competitor gap，而不是把 top-two gap 复制给所有候选；
2. separation gate 只作用于非 nominal 候选，candidate `0` 永远是 nominal anchor；
3. fixed-point score key 和 index tie-break 可重放；
4. CPU/CUDA 的 candidate order、selected index、abstention、CBF 验证字段和 termination 决策逐字段一致；
5. 任何情况下 `raw_unverified_executed=0`。

## 2. 输入与方法

- 三个训练 seed：`20260911`、`20260912`、`20260913`。
- 每个 seed 的 settled rows 与 CPU/CUDA replay 均为已冻结的 V20 development evidence。
- V21 counterfactual 参数：`minimum_candidate_separation_m=0.002 m`，`score_comparison_quantum_m=0.004 m`。
- 对于历史 V20 trace 中尚未保存 separation 字段的行，按 ranker 合同从 immutable `target_cost_m` 重建 separation；不读取 online target truth。
- settled outcome 只用于离线 `selected-not-best`、top-1 safety precision 和 safe-capture precision 诊断。

## 3. 结果

| Seed | Trace rows | 原始 multi-eligible | V21 counterfactual multi-eligible | 被 separation gate 拒绝的替代候选 | 拒绝率（原始替代候选） | CPU/CUDA 决策差异 | Raw unverified |
|---:|---:|---:|---:|---:|---:|:---:|---:|
| 20260911 | 1167 | 453 | 429 | 747 | 41.9% | 0 | 0 |
| 20260912 | 935 | 538 | 483 | 1022 | 47.9% | 0 | 0 |
| 20260913 | 1133 | 887 | 811 | 1609 | 45.4% | 0 | 0 |
| **aggregate** | **3235** | - | - | **3378** | - | **0** | **0** |

候选 separation 的中位数分别为 `0.002590 m`、`0.002372 m`、`0.002470 m`；第 10 百分位约为 `0.00026--0.00030 m`，说明接近等价的候选在真实 trace 中很常见。

在 V21 counterfactual eligible settled rows 上，source selected-not-best 约为 `76.5% / 64.8% / 68.1%`，fixed-point score argmin-not-best 约为 `45.9% / 42.4% / 41.9%`。这仍说明模型排序与 settled outcome 存在明显失配，不能把 separation gate 写成控制收益修复。

top-1 settled safety precision 为 `97.39% / 96.30% / 96.73%`，但 top-1 settled safe-capture precision 只有 `4.78% / 2.59% / 2.48%`。这些是局部 settled counterfactual 诊断，不是 episode-level safe-capture 结论。

source selected sequence 的 switch rate 为 `5.83% / 4.28% / 6.36%`；直接 V21 score counterfactual sequence 为 `21.18% / 20.99% / 27.65%`。这支持记录并审计候选抖动，但不证明低切换率本身带来更高捕获率。

## 4. Gate 结论

### 已通过

- [x] 7-case synthetic monotonic score suite 已通过。
- [x] separation gate 只拒绝非 nominal 候选，nominal anchor 保留。
- [x] 三 seed 共 `3235` 个冻结 trace rows 覆盖完整 settled rows。
- [x] CPU/CUDA candidate decision projection 逐字段一致，差异 `0`。
- [x] `raw_unverified_executed_count=0`。
- [x] TensorBoard event、Config/Provenance/Gates 和 Ranking scalar tags 存在。

### 未通过或尚不能宣称

- [ ] settled ranking mismatch 尚未消除；不能称为 JEPA 已改善 safe-capture。
- [ ] separation gate 的 rejection rate 较高，必须在 WP3/WP4 中验证 fallback 不会造成任务推进退化。
- [ ] 当前审计不替代真实 V21 paired smoke，也不能打开 locked test。

## 5. 对下一阶段的决定

1. 保持 `minimum_candidate_separation_m=0.002 m`、CBF margin、clearance floor、stale/OOD threshold、controlled-abort 语义不变。
2. 进入 WP3：按 V21 protocol 为三个 checkpoint 重新核验 hash-bound ledger，完成 OOD/stale/non-finite/provenance/unknown-horizon fault regression。
3. 进入 WP4 前必须证明 separation rejection、abstention 和 fallback 在 100/500-cycle rolling replay 中可确定性重放。
4. WP3/WP4 未全部通过前，不运行新的三 seed paired smoke，不扩大到 40/60 集，不打开 locked test。
5. 如果 WP5 仍没有 safe-capture 非劣，才进入多任务安全头和困难片段 replay；不通过调低 CBF 或删除 abort 获取表面收益。

## 6. 可复现实验命令

```powershell
$py = 'D:\\download\\anaconda3\\envs\\traj_pred_prep\\python.exe'
$env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = 'python'
& $py scripts/verify_jepa_safe_capture_protocol.py `
  --protocol configs/central_random_mixed_obstacle_s3_v5_v21_cpu_separation_gate_development_protocol.yaml
& $py scripts/audit_jepa_safe_capture_v21_candidate_separation.py `
  --protocol configs/central_random_mixed_obstacle_s3_v5_v21_cpu_separation_gate_development_protocol.yaml `
  --output-dir results/jepa_safe_capture_v21_wp2_candidate_separation_audit `
  --tensorboard-dir results/jepa_safe_capture_v21_tensorboard/wp2_candidate_separation
```

输出目录必须为空；脚本会拒绝覆盖已有结果。

## 7. 代码与测试

- 审计脚本：`scripts/audit_jepa_safe_capture_v21_candidate_separation.py`
- 单测：`tests/test_audit_jepa_safe_capture_v21_candidate_separation.py`
- ranker 变化：`src/encirclement3d/jepa_safe_capture_ranker.py`
- protocol verifier：`scripts/verify_jepa_safe_capture_protocol.py`
- paired evaluator：`scripts/evaluate_jepa_safe_capture_v2_paired.py`

本阶段 targeted safety regression：`75 passed`；审计脚本单测：`6 passed`；合计本轮相关测试：`81 passed`。

**最终结论：** V21 separation gate 的确定性和安全可审计性通过，但真实排序和 safe-capture 控制收益仍未通过；下一步是 WP3 ledger fault regression 和 WP4 rolling-horizon replay，而不是扩大样本或打开 locked test。
