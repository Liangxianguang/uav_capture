# JEPA Safe-Capture v11 Corrected-Frame P6 Smoke

**日期：** 2026-09-05  
**范围：** `development_only=true`，`locked_test_opened=false`  
**硬件：** NVIDIA GeForce RTX 5050，CUDA 12.8，PyTorch 2.7.1+cu128  
**协议：** `configs/central_random_mixed_obstacle_s3_v5_p2_corrected_frame_v11_development_protocol.yaml`  
**主指标：** `safe_capture`；`mean_capture_time` 仅为诊断。

## 1. P6 Smoke Aggregate

三个训练 seed（`20260911`、`20260912`、`20260913`）各使用同 seed 内共享的 paired scene manifest；每个变体 20 个 episode。aggregate 原始文件为 `results/jepa_safe_capture_v5_v11_corrected_frame_smoke_aggregate/summary.json`，SHA-256：
`86ee0ae4b3330b77e380b5604853667c1f9bfd784dac9953e677a4b42e004575`。

| 变体 | seed 20260911 | seed 20260912 | seed 20260913 | 合计 |
|---|---:|---:|---:|---:|
| M0：nominal + CBF | 10/20 (50%) | 10/20 (50%) | 10/20 (50%) | 30/60 (50.0%) |
| M3：JEPA + ledger + auxiliary rank + CBF | 9/20 (45%) | 10/20 (50%) | 9/20 (45%) | 28/60 (46.7%) |
| A1：JEPA + auxiliary rank，无 ledger + CBF | 9/20 (45%) | 12/20 (60%) | 10/20 (50%) | 31/60 (51.7%) |
| A2：JEPA + ledger，无 clearance/visibility rank + CBF | 9/20 (45%) | 11/20 (55%) | 10/20 (50%) | 30/60 (50.0%) |

M3 对 M0 的 paired delta 为 `-5 pp / 0 pp / -5 pp`；improved/degraded/tied 为 `3/5/52`；跨 seed 平均 delta 为 `-3.33 pp`，bootstrap 95% CI 为 `[-5.0, 0.0] pp`，只有 `1/3` 个 seed 非负。aggregate 分类为 `useful_safety_fallback_only`，不是 JEPA 控制收益证明。

## 2. Safety and Runtime Gates

四个变体、三个 seed 的 obstacle/target collision、defender boundary、pairwise violation 和 `raw_unverified_executed_steps` 均为 `0`。CBF controlled-abort 被显式保留并计数，不能计为成功。

12 个 latency audit 全部通过：

- 最大 cycle p95：`15.82469 ms`；
- 最大 CBF solver p95：`2.43385 ms`；
- 最大 queue-age p95：`60 steps`；
- latency/queue-age 字段 finite，trace/summary/episode 计数一致；
- 所有 run 的 `development_only=true`、`locked_test_opened=false`。

P5 的 Joint CBF-QP fault matrix、RTX 5050 rolling replay 和 CPU/CUDA deterministic replay 也全部通过。因此当前问题是任务推进和候选选择，而不是已观察到的执行安全越界。

## 3. Settled Counterfactual and Ranking

M3/A2 的六个 settled counterfactual replay 均 `all_gates_pass=true`，共覆盖 6,241 个左右的控制决策；所有 replay 都只在离线分支使用 simulator truth，不改变源运行。

| 变体/seed | selected-not-settled-best | Spearman | Kendall | 备注 |
|---|---:|---:|---:|---|
| M3 / 20260911 | 7.44% | -0.348 | -0.307 | low-credit 47 decisions，coverage 足够 |
| M3 / 20260912 | 30.49% | -0.564 | -0.484 | low-credit 15，coverage 不足 |
| M3 / 20260913 | 38.10% | -0.605 | -0.526 | low-credit 6，coverage 不足 |
| A2 / 20260911 | 7.39% | -0.398 | -0.358 | low-credit 47 decisions，coverage 足够 |
| A2 / 20260912 | 34.57% | -0.620 | -0.530 | low-credit 16，coverage 不足 |
| A2 / 20260913 | 38.67% | -0.605 | -0.523 | low-credit 6，coverage 不足 |

seed `20260912/20260913` 的 low-credit bucket 未达到预设的 20 条证据门，因此对应 ledger alignment 的 `all_gates_pass=false` 是“证据不足”，不是 ledger 正确性的反例。高信用 failure rate 仍不高于低信用桶，OOD/stale/non-finite fault fallback 和 ledger immutability 全部通过。

该结果把下一步优先级固定为：

1. 先修复 candidate ranking mismatch、候选 separation、abstention/nominal anchor 和 CBF-abort 关联；
2. 再在 train/calibration split 上训练 net-clearance、visibility、TTC、CBF-intervention 辅助头，并进行 hard-episode replay；
3. 重新生成 checkpoint-bound ledger 和全新 protocol 后，先做三 seed 20 集 smoke；
4. 只有新 smoke 安全硬门通过且 M3 相对 M0 不再系统性退化，才扩大到 40/60 集 development。

## 4. Temporal Ledger and TensorBoard

三个 seed 的 temporal ledger audit 均通过：`finite_prediction_trace=true`、`raw_trace_observable=true`、`raw_unverified_zero=true`、fault fallback 和 ledger immutability 全通过。对应审计文件 SHA-256：

| seed | temporal audit SHA-256 |
|---:|---|
| 20260911 | `0d40f2fee7f0bb3996f1d1612123157e204ce6ec08c9b64e9b7586f0ab2b60ea` |
| 20260912 | `33017d7ef1c367a56790c16ef88439c4d74c7dc2235f6a969471ffd1b9bf6951` |
| 20260913 | `2c1a89b3229c90ce6f1680854c673b39bfd6b29d869d71127e0f0c4ef6dd2dba` |

本阶段为 aggregate、settled、ranking、ledger alignment、temporal ledger 和 latency 每个运行建立了独立 TensorBoard 目录，均包含配置、输入 provenance、gate 状态和关键 scalar。主目录：

`results/jepa_safe_capture_v5_v11_corrected_frame_tensorboard/`

## 5. Decision

- P6 safety/provenance/runtime：`PASS`；
- P6 reliability coverage：seed `20260911` `PASS`，seed `20260912/20260913` `INSUFFICIENT_EVIDENCE`；
- P6 task non-inferiority：`FAIL`；
- 当前总分类：`useful_safety_fallback_only`；
- `locked_test_opened=false`，不打开 locked test；
- 不使用 `mean_capture_time` 抵消 `safe_capture` 下降；
- 不根据本 smoke block 直接调参或删除失败 episode。

本报告只归档当前 development 证据。它证明了 `JEPA -> Reliability Ledger -> safety-first ranker -> Joint CBF-QP -> rolling horizon` 的安全执行链可运行，但尚未证明该链路带来 safe-capture 控制收益。
