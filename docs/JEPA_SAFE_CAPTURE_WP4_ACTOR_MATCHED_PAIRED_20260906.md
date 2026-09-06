# WP4 Actor-Matched Paired Development

**日期：** 2026-09-06
**阶段：** development-only；`locked_test_opened=false`
**硬件：** NVIDIA GeForce RTX 5050；CUDA 12.8
**主指标：** episode-level `safe_capture`

## 实验目的

验证 WP1/WP2 的障碍物条件路线生成、actor-matched route JEPA、Reliability
Ledger、candidate CBF prefilter 和滚动重规划能否在同一冻结场景上超过
nominal actor + Joint CBF 基线。M0 和 M3 使用完全相同的 20 个 validation
episode、episode seed、障碍物布局、target motion 和 observation schedule。

## 配置与可复现输入

| 项目 | 值 |
|---|---|
| Protocol | `configs/central_random_mixed_obstacle_s3_route_v1_protocol.yaml` |
| Environment | `configs/capture_radius_pursuit_central_v4_flee.yaml` |
| Actor | `models/v5_development_exact_reactive_seed661606.pt` |
| JEPA | `results/jepa_route_identity_model_actor_train40_seed661606/checkpoint.pt` |
| Ledger | `results/jepa_route_identity_ledger_actor_train40_seed661606/reliability_ledger.json` |
| Candidate contract | 12 routes, `obstacle_route_v1`, 3-step chunk, first-step execution |
| Safety contract | CBF margin unchanged; stale/OOD/non-finite gates unchanged; raw action forbidden |
| TensorBoard | `results/wp4_actor_matched_paired_tb/{m0_seed20260911,m3_seed20260911}` |

运行入口是 `scripts/evaluate_jepa_safe_capture_v2_paired.py`。M0 首先生成
`results/wp4_actor_matched_paired_m0_seed20260911/scene_manifest.jsonl`；M3
复用该文件运行。配对审计入口是
`scripts/aggregate_actor_matched_paired_development.py`。

## 结果

| Variant | Safe capture | Collision | Defender boundary | Target boundary | Pairwise | Raw unverified | CBF abort steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 nominal + CBF | 7/20 (35.0%) | 0 | 0 | 0 | 0 | 0 | 12 |
| M3 route JEPA + Ledger + CBF | 9/20 (45.0%) | 0 | 0 | 3 | 0 | 0 | 5 |

M3 相对 M0 的 episode-level 配对为 **4 improved / 2 degraded / 14 tied**，
配对差值为 **+10 percentage points**。McNemar exact two-sided p-value 和
bootstrap interval 见本阶段生成的 `paired_aggregate.json`；20 集仍不足以
作为正式提升结论。

M3 路线统计：`30,144` 条候选生成，`17,434` 条几何有效，`17,393` 条通过
first-step CBF probe，`41` 条被拒绝。所有最终执行动作均通过 CBF；
`raw_unverified_executed_steps=0`。

## 解释

这是第一个在 actor-matched archive 和真实 route runtime 上同时成立的正向
development signal：M3 比 M0 多捕获 2 集，同时 controlled abort 从 12 步
降到 5 步，且 defender collision/boundary/pairwise 安全硬门保持为零。

但 M3 仍有 3 个 target-boundary violation、1,065 个 CBF fallback steps 和
6 个 timeout episode，因此不能把它写成“已完成提升”，也不能打开 locked
test。target boundary 与 defender safety gate 分开报告，避免掩盖目标动力学
越界问题。

## 阶段判定与下一步

- safety hard gate：PASS（defender collision/boundary/pairwise/raw-unverified 均为 0）。
- reliability gate：PASS（CBF timeout 为 0，abort/unverified 计数一致）。
- 当前分类：`positive_single_seed_development_evidence`。
- 下一步：在不改 CBF margin、不关闭 stale/OOD/non-finite gate、不执行 raw action 的前提下，使用同一协议运行其余两个 training seed 的 paired M0/M3 smoke，然后做三 seed aggregate。
- 如果三 seed 后 M3 的 paired delta 不再稳定为正，立即停止扩大规模，优先定位 route score、target belief、geometry-valid coverage 和 fallback 原因。

本报告不是 locked-test 结果；TensorBoard event files 和完整 episode/step
traces 保存在上述本地 `results` 目录中，Git 只提交可审计脚本与报告，不提交
被 `.gitignore` 排除的大型生成输出。
