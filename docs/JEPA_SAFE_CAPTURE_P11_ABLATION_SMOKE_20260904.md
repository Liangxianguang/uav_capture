# P11 四变体 Smoke 消融归档

**日期：** 2026-09-04
**阶段：** development-only
**`locked_test_opened`：** `false`
**设备：** RTX 5050 CUDA
**training seed：** `20260911`
**episode block：** 20 集，同一 S3 scene manifest

## 1. 目的

在 P11 rank guard 安全语义审计通过后，使用同一 scene manifest 运行 M0、M3、A1、A2，判断 reliability ledger 和 clearance/visibility 排序项是否值得进入多 seed development block。该 smoke 不是正式统计结论，也不以 `95%` safe-capture 为目标。

## 2. 固定输入

- protocol：`configs/central_random_mixed_obstacle_s3_v5_rank_guard_development_protocol.yaml`
- scene manifest：`results/jepa_safe_capture_v4_p11_smoke_m0_seed20260911/scene_manifest.jsonl`
- actor：`models/v5_development_exact_reactive_seed661606.pt`
- JEPA：`results/jepa_safe_capture_v3_wp2_seed20260911/checkpoint.pt`
- ledger：`results/jepa_safe_capture_v4_p10_ledger_seed20260911/reliability_ledger.json`
- candidate contract：`K=5`、chunk length 3、execute-first-step-then-replan、tie tolerance `5e-4`
- CBF：同一 Joint CBF-QP、margin、solver、timeout 和 fallback 合同

四个运行和审计均使用已推送 revision `3cfa04fa410d997fe1af7f0abdbd527f583a7275` 的代码；所有输出目录和 TensorBoard 目录独立创建。

## 3. Smoke 结果

| 变体 | 组成 | safe capture | paired vs M0 | collision | boundary | pairwise | CBF timeout | CBF controlled abort | raw unverified |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | nominal + CBF | 10/20 (50.0%) | reference | 0 | 0 | 0 | 0 | 10 | 0 |
| M3 | JEPA + ledger + auxiliary rank + CBF | 10/20 (50.0%) | 0/0/20, 0.0 pp | 0 | 0 | 0 | 0 | 10 | 0 |
| A1 | JEPA + auxiliary rank + CBF，无 ledger | 9/20 (45.0%) | 0/1/19, -5.0 pp | 0 | 0 | 0 | 0 | 10 | 0 |
| A2 | JEPA + ledger + CBF，无 clearance/visibility rank | 10/20 (50.0%) | 0/0/20, 0.0 pp | 0 | 0 | 0 | 0 | 10 | 0 |

配对格式为 `improved/degraded/tied`。M3、A1、A2 的全部 smoke audit gate 均为 `true`，但 20 集不足以证明任务收益或不劣性。

## 4. Trace 与执行诊断

- M3：1082 ranking steps，168 次 top-two margin abstention，candidate switch rate `0.0407`，hysteresis 2 steps，hold 43 steps；CBF correction mean `0.2651`。
- A1：1137 ranking steps，204 次 abstention，candidate switch rate `0.0572`，hysteresis 14 steps，hold 69 steps；CBF correction mean `0.2678`。
- A2：1082 ranking steps，167 次 abstention，candidate switch rate `0.0407`，hysteresis 2 steps，hold 43 steps；CBF correction mean `0.2651`。
- M0：不使用 JEPA ranker，CBF correction mean `0.9400`。
- 四个运行的 raw-action trace 字段均完整，`raw_unverified_executed_steps=0`；10 个 CBF controlled abort 均被保留为安全失败，而非 raw action。

## 5. 产物

| 变体 | 运行目录 | audit 目录 | TensorBoard |
|---|---|---|---|
| M0 | `results/jepa_safe_capture_v4_p11_clean_m0_seed20260911` | `results/jepa_safe_capture_v4_p11_clean_m0_audit_seed20260911` | `results/jepa_safe_capture_v4_tensorboard/p11_clean_m0_seed20260911` |
| M3 | `results/jepa_safe_capture_v4_p11_clean_m3_seed20260911` | `results/jepa_safe_capture_v4_p11_clean_m3_audit_seed20260911` | `results/jepa_safe_capture_v4_tensorboard/p11_clean_m3_seed20260911` |
| A1 | `results/jepa_safe_capture_v4_p11_semantic_a1_seed20260911` | `results/jepa_safe_capture_v4_p11_semantic_a1_audit_clean_seed20260911` | `results/jepa_safe_capture_v4_tensorboard/p11_semantic_a1_seed20260911` |
| A2 | `results/jepa_safe_capture_v4_p11_semantic_a2_seed20260911` | `results/jepa_safe_capture_v4_p11_semantic_a2_audit_clean_seed20260911` | `results/jepa_safe_capture_v4_tensorboard/p11_semantic_a2_seed20260911` |

## 6. 结论和下一步

1. CBF 安全几何门在四个变体中均保持通过，且没有 raw/unverified action 执行。
2. M3 在本 smoke 中与 M0 持平；A1 出现一个 degraded episode，支持继续保留 ledger，但不能据此做正式因果结论。
3. A2 与 M3 持平，说明本 block 中 clearance/visibility 项尚未显示可测任务收益；仍需先完成 settled counterfactual 标签和独立 calibration，再决定是否调整权重。
4. 不启动 locked test；下一步是 T2 settled ranking、T3 ledger temporal/adversarial calibration 和 T6 rolling-horizon regression，之后才是三 seed paired block。

## 7. TensorBoard 与审计

每个 run 记录 configuration、provenance、Safety、CBF、Fallback、Ranking 和 latency scalar；每个 rank audit 记录 paired 结果、gate 状态和 raw-action scalar。M0/M3/A1/A2 的 audit 输出均包含 `all_gates_pass=true`。
