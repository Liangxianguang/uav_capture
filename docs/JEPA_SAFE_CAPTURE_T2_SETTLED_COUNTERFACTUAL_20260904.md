# T2 Settled Counterfactual Ranking Audit

**日期：** 2026-09-04
**阶段：** development-only，offline local-chunk audit
**`locked_test_opened`：** `false`
**输入：** floor015 protocol，seed `20260911`，同一 20-episode S3 scene manifest
**主关注：** settled safety/capture alignment；`mean_capture_time` 不参与准入

> 本审计从冻结运行的实际执行序列重建每个决策状态，对五个 eligible candidate 各自进行 3-step offline branch。每一步都经过同一个 Joint CBF-QP，target ground truth 只用于离线结算。它是局部 action-chunk counterfactual，不等价于重新运行一个完整候选策略的 episode 结果。

## 1. 固定输入与完整性门

- protocol：`configs/central_random_mixed_obstacle_s3_v5_rank_guard_floor015_development_protocol.yaml`
- environment：`configs/capture_radius_pursuit_central_v4_flee.yaml`
- baseline：`results/jepa_safe_capture_v4_p12_floor015_m0_seed20260911/`
- candidates：`results/jepa_safe_capture_v4_p12_floor015_{m3,a1,a2}_seed20260911/`
- protocol SHA-256：`7d4710f87805fade62b1e50c3a689cbf9f861917dddb3c8c940b397d129592ec`
- scene manifest SHA-256：`6a5fa0905a6b8391993fba3335452d1f0f3f1b8670749b45346a5ff71e3470ba`

三种候选运行均通过 source manifest hash、spec、scene hash、training seed 和 replay step-count 校验。审计 JSON 的 `all_gates_pass=true`；TensorBoard 每个变体均包含 Config、Provenance、Gates、Ranking、Settled、CBF 和 Calibration tags。

## 2. 决策级 settled 结果

| 变体 | decisions | selected-not-best | selected settled safe | best settled safe | selected safety | mean selected progress (m) | mean best progress (m) | score/progress Spearman | Kendall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M3 | 994 | 24.5% | 2.31% | 2.41% | 76.4% | 0.5660 | 0.5694 | -0.479 | -0.396 |
| A1 | 1,075 | 30.2% | 2.33% | 2.33% | 97.1% | 0.4658 | 0.4699 | -0.510 | -0.425 |
| A2 | 1,092 | 23.6% | 2.20% | 2.29% | 75.7% | 0.5287 | 0.5333 | -0.595 | -0.501 |

ranker score 采用“越低越优”，所以 score/progress 的负相关在方向上是预期的；它证明候选 score 并非完全随机，但不能证明已校准。`selected-not-best`、selected/best settled safety 差距和 score-softmax proxy 的高 ECE 仍表明排序存在可修复失配。

## 3. 按 paired episode outcome 分桶

paired label 继承同一 episode 的 M0 对比结果，不能当作独立样本。

| 变体/分桶 | decisions | selected-not-best | selected settled safe | best settled safe | selected safety | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| M3 degraded（1 episode） | 44 | 79.5% | 0.0% | 0.0% | 86.4% | -0.479 |
| M3 tied（19 episodes） | 950 | 22.0% | 2.42% | 2.53% | 75.9% | -0.479 |
| A1 degraded（2 episodes） | 294 | 48.3% | 0.34% | 0.34% | 99.0% | -0.499 |
| A1 tied（18 episodes） | 781 | 23.4% | 3.07% | 3.07% | 96.4% | -0.518 |
| A2 tied（20 episodes） | 1,092 | 23.6% | 2.20% | 2.29% | 75.7% | -0.595 |

degraded 分桶只有 1–2 个 episode，当前只能作为困难片段索引，不能作为可靠性因果估计。

## 4. 安全语义

- source M0/M3/A1/A2 运行的 collision、boundary、pairwise violation 和 CBF timeout 均为 0，`raw_unverified_executed_steps=0`。
- counterfactual branch 的 CBF unverified 是“候选被离线拒绝”的诊断信号，不是实际 raw action 执行；它被计入 settled safety failure，但不触发 source run raw-action gate。
- M3/A1/A2 的 selected branch 分别出现 21/31/18 个 local CBF-unverified steps，应在 T3 ledger 与 T4 辅助风险头校准中解释。

## 5. 结论边界与下一步

1. 当前 score 对短 chunk target progress 有方向性信息，但 `selected-not-best` 约 24%–30%，且 selected progress 略低于 settled best；不能把 ranker 称为已校准。
2. M3 degraded episode 的 selected-not-best 很高，优先作为困难片段重放对象；但样本太少，不能直接调整权重或宣称 ledger 因果提升。
3. score-softmax Brier/ECE 是由 score 构造的 proxy，不是模型输出的 capture probability；在独立 calibration 前不能用于部署阈值。
4. 继续执行顺序：T3 temporal/adversarial ledger 校准 -> T4 clearance/visibility/TTC/CBF 头与困难重放 -> T5/T6 rolling-horizon/CBF 回归 -> 三 seed smoke。未通过这些门前不运行三 seed final block，不打开 locked test。

## 6. 复现产物

每个目录均含 `settled_counterfactual.json`、`decision_rows.jsonl`、`report.md` 和独立 TensorBoard：

- `results/jepa_safe_capture_v4_p12_floor015_settled_cf_m3_seed20260911_final2/`
- `results/jepa_safe_capture_v4_p12_floor015_settled_cf_a1_seed20260911_final2/`
- `results/jepa_safe_capture_v4_p12_floor015_settled_cf_a2_seed20260911_final2/`
- `results/jepa_safe_capture_v4_tensorboard/p12_floor015_settled_cf_*_final2/`

所有产物保持 development-only；没有覆盖历史 V4/V5 archive。
