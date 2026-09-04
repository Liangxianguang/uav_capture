# P9 CBF 求解器可靠性修复归档

**日期：** 2026-09-04
**状态：** development-only；`locked_test_opened=false`
**协议：** `central_random_mixed_obstacle_s3_v4_p9_cbf_jacobian_development_protocol.yaml`
**硬件：** NVIDIA GeForce RTX 5050；PyTorch 2.7.1+cu128

## 1. 修复内容

在 Joint CBF-QP 的 SLSQP 求解中加入目标函数梯度和速度/加速度非线性约束的
解析 Jacobian。约束、margin、gamma、solver tolerance、fallback 顺序和
`max_latency_ms=100` 合同均未改变。该改动只减少有限差分开销，不放宽安全约束。

## 2. 回归证据

- CBF、paired evaluator 和 fault-injection targeted tests：`29 passed`。
- 固定压力审计 9 类场景、20 次重复：所有输出 finite、失败路径均有 fallback、
  raw request 未执行、重复结果确定；p95 求解延迟约 `0.80 ms`。
- 新 protocol M0 smoke（seed 20260911，20 集）：`11/20 = 55.0% safe_capture`，
  collision/boundary/pairwise 均为 0，CBF timeout 为 0，最大 run-level p95 约 9.80 ms。
- 同一 manifest 的 M3 smoke：`7/20 = 35.0% safe_capture`，collision/boundary/pairwise
  均为 0，CBF timeout 为 0，最大 run-level p95 约 8.02 ms。
- M0/M3 smoke 配对为 improved/degraded/tied `1/5/14`，delta `-20.0 pp`。

## 3. 解释

解析 Jacobian 修复了 CBF 求解的偶发 timeout 风险，但没有修复 JEPA 排序导致的任务
回归。因此 P9 只通过了 solver/reliability 子门，不能把 smoke 结果写成 JEPA 控制
提升。下一步仍需按 P8 replay 证据修复 high-credit 错误排序、CBF intervention 代价
和 candidate oscillation，然后才能重跑三 seed paired block。

## 4. 产物

- `src/encirclement3d/cbf_qp.py`
- `configs/central_random_mixed_obstacle_s3_v4_p9_cbf_jacobian_development_protocol.yaml`
- `results/wp9_cbf_jacobian_audit_20260904/`
- `results/wp9_smoke_tie3_m0_seed20260911/`
- `results/wp9_smoke_tie3_m3_seed20260911/`
- `results/jepa_safe_capture_v3_tensorboard/wp9_cbf_jacobian_audit_20260904/`
- `results/jepa_safe_capture_v3_tensorboard/wp9_smoke_tie3_m0_seed20260911/`
- `results/jepa_safe_capture_v3_tensorboard/wp9_smoke_tie3_m3_seed20260911/`

locked test 仍保持关闭。
