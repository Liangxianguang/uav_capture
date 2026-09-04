# P11 Rank Guard Smoke 与安全语义审计

**日期：** 2026-09-04
**阶段：** development-only
**`locked_test_opened`：** `false`
**设备：** NVIDIA RTX 5050 / CUDA 12.8 / PyTorch 2.7.1+cu128
**审计脚本：** `scripts/audit_jepa_safe_capture_v5_rank_guard_smoke.py`

## 1. 本阶段目的

本阶段验证 P11 rank guard 是否满足可观测、安全回退和配对运行合同。重点修复了一个指标语义问题：CBF `controlled_abort` 返回的是有限 emergency action，虽然它是未验证且必须使 `safe_capture=false`，但不应被误报为执行了原始 candidate/nominal action。

当前区分三个字段：

- `cbf_unverified_steps`：CBF 输出未达到 `verified_feasible` 的步数，包含 `controlled_abort`；
- `cbf_controlled_abort_steps`：受控终止步数，属于安全失败；
- `raw_unverified_executed_steps`：没有经过已验证 CBF 路径而实际执行 raw 请求的步数，硬门必须为 0。

## 2. 实验输入

- baseline：`results/jepa_safe_capture_v4_p11_semantic_m0_seed20260911`
- candidate：`results/jepa_safe_capture_v4_p11_semantic_m3_seed20260911`
- paired scene manifest：`results/jepa_safe_capture_v4_p11_smoke_m0_seed20260911/scene_manifest.jsonl`
- protocol：`configs/central_random_mixed_obstacle_s3_v5_rank_guard_development_protocol.yaml`
- training seed：`20260911`
- episodes：20
- candidate contract：`K=5`、chunk length 3、execute-first-step-then-replan、tie tolerance `5e-4`

两次运行共享 scene manifest、actor checkpoint、protocol 和设备，M3 额外使用 JEPA checkpoint 与 P10 ledger。所有输出目录和 TensorBoard 目录均为新目录。

## 3. 结果

| run | safe capture | collision | boundary | pairwise | CBF unverified | controlled abort | raw unverified | CBF p95 (ms) | correction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 10/20 (50.0%) | 0 | 0 | 0 | 10 | 10 | 0 | 8.252 | 0.9400 |
| M3 | 10/20 (50.0%) | 0 | 0 | 0 | 10 | 10 | 0 | 7.468 | 0.2651 |

配对结果为：`improved/degraded/tied = 0/0/20`，paired delta `0.0 pp`。

M3 的 rank trace：

- ranking steps：1082；
- top-two margin abstention：168 次；
- candidate switch rate：0.0407；
- hysteresis steps：2；
- hold steps：43；
- invalid selection：0；
- non-finite score：0；
- raw-unverified trace steps：0；
- 缺失 raw-action 标志：0。

## 4. Gate 判定

P11 audit 的全部 gate 通过：

- development-only：通过；
- locked test 未打开：通过；
- scene pairing：通过；
- M0/M3 collision、boundary、pairwise：均为 0；
- M0/M3 `raw_unverified_executed_steps`：均为 0；
- trace 选择、score finite、raw-action 字段：通过；
- TensorBoard provenance、paired、gate 状态和 raw-action scalar：通过。

审计产物：

- `results/jepa_safe_capture_v4_p11_semantic_audit_seed20260911_v2/rank_guard_audit.json`
- `results/jepa_safe_capture_v4_p11_semantic_audit_seed20260911_v2/rank_guard_audit.md`
- `results/jepa_safe_capture_v4_tensorboard/p11_semantic_audit_seed20260911_v2/`

## 5. 结论边界

本阶段只证明 P11 的安全语义和 trace 可审计性已闭合。M3 与 M0 的 safe-capture 相同，故不能宣称 rank guard 带来任务收益。M3 的 CBF correction 较低只能作为执行代价诊断，不能替代 safe-capture 证据；10 个 controlled abort 仍是安全失败，需要在后续 settled counterfactual、ledger 校准和 rolling-horizon 分析中解释。

下一步：先进行 T2 settled counterfactual 排序诊断和 T3 ledger 时序/OOD 重校准，再运行 A1/A2 smoke；在这些门通过前不启动三 seed final block，也不打开 locked test。

## 6. 代码与测试变更

- evaluator 增加 `_raw_unverified_executed` 判定、episode/summary/trace/TensorBoard 计数；
- paired aggregate 增加 raw-action 字段并纳入 reliability gate；
- rank-guard audit 改为正向的 `locked_test_not_opened` gate，避免 `all(gates.values())` 被预期的 `locked_test_opened=false` 误判；
- 新增 raw-action 语义和 trace audit 单测。

Targeted tests：`32 passed`。`py_compile` 和 `git diff --check` 均通过。
