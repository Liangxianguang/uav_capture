# V21 完整闭环与 CBF-only 单 Seed 配对比较

**日期：** 2026-09-05
**实验性质：** development-only smoke，不是 locked test
**训练 seed：** `20260911`
**episode 数：** 20
**主指标：** episode-level `safe_capture`
**Locked test：** `locked_test_opened=false`

## 1. 比较定义

本实验在同一冻结 S3 scene manifest、同一 V21 protocol、同一 V5 actor、同一 episode seed 顺序下运行两条路径：

- **M0（CBF-only baseline）：** nominal planner + Joint CBF；不使用 JEPA 和 Reliability Ledger。
- **M3（完整闭环）：** nominal planner 生成候选，interaction-aware action-conditioned JEPA 评价候选，Reliability Ledger 路由，Joint CBF 过滤，rolling horizon 只执行第一步。

上一阶段 P5 的 CBF-QP audit 只有求解器和约束 case，没有 episode-level `safe_capture`，因此本报告使用同协议下的 M0 作为可比的 CBF-only 任务基线。

## 2. 结果

| 指标 | M0 CBF-only | M3 完整闭环 | 差异 |
|---|---:|---:|---:|
| safe capture | 10/20 = 50.0% | 12/20 = 60.0% | **+10.0 pp** |
| collision | 0 | 0 | 0 |
| boundary violation | 0 | 0 | 0 |
| pairwise violation | 0 | 0 | 0 |
| raw unverified executed | 0 | 0 | 0 |
| CBF infeasible / controlled abort steps | 9 / 9 | 8 / 8 | -1 |
| timeout episodes | 1 | 0 | -1 |
| control cycles | 1132 | 1189 | +57 |
| mean minimum clearance (m) | 0.4015 | 0.4006 | -0.0009 |
| mean CBF correction norm (m/s) | 0.9442 | 0.2754 | -0.6688 |
| mean capture time (s) | 6.69 | 8.59 | +1.90 |
| mean CBF p95 solve latency (ms) | 3.054 | 2.911 | -0.143 |

逐 episode 配对结果：

- improved：`2`
- degraded：`0`
- tied：`18`
- paired delta：`+2/20 = +10.0 pp`
- 改善 episode：`650002`（CBF abort -> safe capture）、`650007`（timeout -> safe capture）

## 3. 安全合同

本次 M0 和 M3 均通过开发安全硬门：没有 collision、boundary、pairwise 或 raw-unverified execution。M3 的所有 CBF failure 都保留为可观测 fallback/controlled-abort 语义，没有执行未经验证的 raw action。

这说明在这一组冻结场景中，完整闭环减少了一次 CBF controlled-abort 和一次 timeout，并多完成两次安全捕获。`mean_capture_time` 变差只作为诊断记录，不改变本实验的 safe-capture 主结论。

## 4. 结论边界

这次结果支持一个**单 seed、20 集的局部正向信号**，不能证明 JEPA 闭环已经稳定优于 CBF-only。已有 V21 三 seed、60 集 paired smoke 的总体结果仍是：

| 变体 | safe capture |
|---|---:|
| M0 | 30/60 = 50.0% |
| M3 | 28/60 = 46.7% |

对应三个 seed 的 M3 相对 M0 差异为 `+10 pp`、`-15 pp`、`-5 pp`，只有 `1/3` seed 非负。因此当前正式标签仍为 `ranking_unresolved` / `useful_safety_fallback_only`，不能据此扩大到 40/60 集或打开 locked test。

## 5. 输入与产物

- M0：`results/jepa_safe_capture_v21_full_loop_compare_m0_seed20260911/`
- M3：`results/jepa_safe_capture_v21_full_loop_compare_m3_seed20260911/`
- scene manifest：`results/jepa_safe_capture_v21_smoke_m0_seed20260911/scene_manifest.jsonl`
- protocol：`configs/central_random_mixed_obstacle_s3_v5_v21_cpu_separation_gate_development_protocol.yaml`
- actor：`models/v5_development_exact_reactive_seed661606.pt`
- JEPA checkpoint：`results/jepa_safe_capture_v11_hard_replay_seed20260911/checkpoint.pt`
- Reliability Ledger：`results/jepa_safe_capture_v21_ledger_seed20260911/reliability_ledger.json`
- TensorBoard：`results/jepa_safe_capture_v21_current_tensorboard/full_loop_compare_m0_seed20260911/`
- TensorBoard：`results/jepa_safe_capture_v21_current_tensorboard/full_loop_compare_m3_seed20260911/`

两条运行均记录了相同 protocol SHA-256 `278623ceb7185a6c3ce23246e8a28693f025a2977fad95059ae5b0df9a03b014`、相同 scene manifest SHA-256 `6a5fa0905a6b8391993fba3335452d1f0f3f1b8670749b45346a5ff71e3470ba`、RTX 5050 CUDA 环境和 `locked_test_opened=false`。

## 6. 下一步

1. 对所有 CBF abort 完成 candidate、nominal、safe-hold 三路独立 QP counterfactual，确认本次改善是否来自候选排序而非偶然场景差异。
2. 完成 score direction、candidate eligibility、action scale、horizon、solver initialisation 和 communication-age 语义回归。
3. 若 S3 诊断排除实现问题，再建立新 protocol/ledger，运行三 seed x 20 paired smoke。
4. 只有新三 seed smoke 的 aggregate `safe_capture` 不低于 M0 且至少 `2/3` seed 非负，才进入 40/60 development。
