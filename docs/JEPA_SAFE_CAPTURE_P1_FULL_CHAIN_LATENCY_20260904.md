# P1 全链路延迟与可观测性阶段报告

**日期：** 2026-09-04  
**范围：** development-only；`locked_test_opened=false`  
**设备：** NVIDIA RTX 5050；CUDA 12.8；PyTorch 2.7.1+cu128  
**协议：** `configs/central_random_mixed_obstacle_s3_v5_t3_recalibration_development_protocol.yaml`  
**主指标：** `safe_capture`；延迟仅用于实时性门，不替代任务指标

## 1. 目的

为 action-conditioned interaction-aware JEPA + reliability ledger + Joint CBF-QP + rolling horizon 闭环补齐逐 control-cycle 的延迟、输入队列年龄、回退和 provenance 记录。该阶段不改变候选选择、安全过滤或 episode 结算逻辑。

新增字段位于：

- `src/encirclement3d/jepa_safe_capture_ranker.py`：JEPA、ledger、ranker timing；
- `scripts/evaluate_jepa_safe_capture_v2_paired.py`：actor、candidate、CBF、环境、cycle、queue age 和 trace schema；
- `scripts/audit_jepa_safe_capture_v2_latency.py`：独立结构化审计和 TensorBoard 校验。

## 2. 20 集实际 replay

| 项目 | 结果 |
|---|---:|
| variant | M3 |
| training seed | `20260911` |
| episodes | 20 |
| control cycles | 1,075 |
| safe capture | `8/20 = 40.0%` |
| collision / boundary / pairwise | `0 / 0 / 0` |
| CBF timeout | 0 |
| CBF controlled abort | 11 |
| raw unverified executed | 0 |
| transit success | 95.0% |

该结果与已有 non-zero rolling replay一致，说明新增 timing 字段没有改变物理路径。它是 instrumentation/development evidence，不是 JEPA 任务提升结论。

## 3. Pooled latency（1,075 cycles）

| stage | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---:|---:|---:|---:|
| actor | 0.885 | 1.415 | 1.872 | 333.403 |
| candidate generation | 4.762 | 6.080 | 7.169 | 268.334 |
| JEPA inference | 3.637 | 4.748 | 5.595 | 266.766 |
| ledger route | 0.114 | 0.166 | 0.210 | 0.297 |
| ranker compute | 0.482 | 0.696 | 0.849 | 1.244 |
| rank total | 4.250 | 5.499 | 6.460 | 267.673 |
| CBF filter wall | 1.755 | 2.908 | 7.370 | 50.555 |
| CBF solver | 0.662 | 1.472 | 5.724 | 17.168 |
| environment step | 3.366 | 5.804 | 6.892 | 8.189 |
| complete cycle | 11.451 | 15.175 | 22.096 | 609.353 |

`cycle_total` p95 为 `15.175 ms`，低于当前 100 ms development budget；max 值来自 CUDA/进程 warm-up 和系统调度，只作为诊断保留。进入 SIL/HIL 前还需要单独制定 warm-up、watchdog 和 p99 预算，不能只依赖 p95。

输入队列年龄：p50 `2` steps，p95 `35.5` steps，p99/max `60` steps。该年龄来自在线 observation/message metadata，不读取 target ground truth；高年龄必须继续由 ledger 的 stale/OOD 回退处理。

## 4. 审计 gates

`results/jepa_safe_capture_v5_p1_latency_m3_seed20260911/latency_audit.json` 中以下 gates 全部为 `true`：

- `trace_schema_version=2`、latency contract 与声明阶段一致；
- 20 个 episode 的 CSV、summary、trace 数量一致；
- 1,075 个 cycle 的 10 段 latency 字段完整、finite、non-negative；
- 1,075 个 input queue-age 字段完整、finite；
- ranking trace 可观测；
- summary 与 trace 的 raw-unverified 数一致且均为 0；
- cycle p95 不超过 100 ms；
- development boundary 保持，locked test 未打开。

## 5. 可复现产物与 hash

运行结果：

- `results/jepa_safe_capture_v5_p1_latency_m3_seed20260911/`
- `results/jepa_safe_capture_v5_tensorboard/p1_latency_m3_seed20260911/`
- `results/jepa_safe_capture_v5_tensorboard/p1_latency_audit_m3_seed20260911/`

| 文件 | SHA-256 |
|---|---|
| `summary.json` | `1d359e76d24698b1c2fa4744a5bb30df2363264519283bc4bd8e99981541a46a` |
| `provenance.json` | `b9934fbec58668cd57636270e34940a9e7a2bdca83f5bb801505292f63aa682b` |
| `latency_audit.json` | `86cc21577af2d1d68cbccfba848992d2e3aeb1e4d0b222f8b0f010df658d1696` |
| evaluator TensorBoard event | `318694cb1c301f183cdcb2b30d12de364bf5de9cbe14a76a48ea95d0cd8f9cef` |
| audit TensorBoard event | `8ce97cc0bfb035cb721a3d05376fc647909e332f391d7d7461eaf837d7a6ebbd` |

代码 revision：`6a8c5e3e05bf3d36f2f81867f202b6f254ad72ff`。审计脚本 SHA-256：`c12732ee0404bc9a0ccf6f71e90d4c071b9a8538e8166c12bb5cc09df00ee0ac`。

## 6. 结论和下一步

P1 证明了完整闭环的延迟和回退状态可以逐步记录、独立审计并写入 TensorBoard；它没有证明 JEPA 提升 safe capture。当前下一步仍是：

1. 用 settled rows 修订或冻结 safety-first ranking、abstention、hysteresis 和 nominal anchor；
2. 对 clearance、visibility、TTC、CBF-risk 辅助头做 hard-bucket 校准和困难片段 replay；
3. 新建 protocol smoke，通过安全、provenance、TensorBoard 和 latency gates 后再运行三 seed paired development。

`mean_capture_time` 不参与安全门，95% 不作为硬目标；任何新的 collision、boundary、pairwise 或 raw action 都必须停止扩大实验规模。
