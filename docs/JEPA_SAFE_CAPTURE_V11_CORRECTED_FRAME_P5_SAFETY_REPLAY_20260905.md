# v11 corrected-frame P5 安全执行与滚动回放报告

**日期：** 2026-09-05
**阶段：** P5 - Joint CBF-QP、rolling-horizon、设备一致性和实时性
**硬件：** NVIDIA GeForce RTX 5050，CUDA 12.8
**边界：** `development_only=true`，`locked_test_opened=false`

## 1. Joint CBF-QP fault matrix

输入为固定环境配置 `configs/capture_radius_pursuit_central_v4_flee.yaml`，seed `20260911`，9 类确定性 case，每类重复 20 次。原始审计：`results/jepa_safe_capture_v5_v11_corrected_frame_cbf_audit/audit.json`，SHA-256：`2a7eeba3257739ec8803518cf6472bdbea0479628a3aa029245f1d6a75acaf46`。

| gate | result |
|---|---:|
| all outputs finite | PASS |
| failed solve never executes raw request | PASS |
| zero-perturbation exact | PASS |
| repeated deterministic | PASS |
| state violation count | 0 |
| solver p95 | 0.708 ms |
| explicit fallback cases | 3 |
| infeasible cases | 3 |
| timeout cases | 1 |

non-finite request、运动学 infeasible 和 solver timeout 均返回显式 `controlled_abort`；它们被计为安全失败，但不执行未经验证的 raw action。

## 2. Rolling-horizon full-chain latency

使用 v11 corrected-frame M3 seed `20260911` 的 20 集、1,062 control cycles。原始审计：`results/jepa_safe_capture_v5_v11_corrected_frame_smoke_m3_seed20260911/latency_audit.json`，SHA-256：`e800a2a5def1a3ea277531daf8ef3775ba1d132b6d58c54af0016e41fd8b50fa`。

- 所有 latency/queue-age 字段存在且 finite；trace、CSV、summary 计数一致；
- cycle p95 `15.416 ms`，低于 `100 ms` contract；
- JEPA/ranker/CBF 全部可观测；
- `raw_unverified_zero=true`；
- `locked_test_not_opened=true`。

每个 control cycle 只执行 action chunk 的第一步，随后重新观测、更新 belief、生成候选并排序；审计没有发现长块 open-loop 执行。

## 3. CPU/CUDA deterministic replay

CUDA 与 CPU 使用同一 v11 protocol、actor checkpoint、JEPA checkpoint、ledger 和 scene manifest。原始审计：`results/jepa_safe_capture_v5_v11_corrected_frame_device_replay_audit/device_replay_audit.json`，SHA-256：`c9437644db4a2c9351bb2ba5f97c4568bffb934e42fe3057f2b060ed2c58a0f9`。

分类为 `cpu_cuda_safety_and_decision_equivalent`。20/20 episode 的安全结算一致，候选决策、CBF verification counts 和 rejection-reason schema 全部一致；两端 collision/boundary/pairwise 都为 0，raw-unverified 均为 0，CBF p95 均低于 100 ms。

## 4. P5 结论和下一步

P5 安全执行边界通过：CBF 是唯一执行入口，ledger/JEPA 只能影响候选排序或回退意图，所有失败均显式保留。该结果不等同于任务收益证明。

下一步进入 P6：对每个 training seed 生成独立 M0 manifest，运行 M0/M3/A1/A2 各 20 个 paired episodes；先完成 seed `20260913`，然后运行 aggregate、settled counterfactual、ledger alignment、temporal ledger、CBF、latency 和 deterministic replay 审计。只有 smoke 满足预注册 gate 才能进入 40 集 validation。
