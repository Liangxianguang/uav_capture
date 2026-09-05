# V21 Rolling-Horizon / Joint CBF 审计报告

**日期：** 2026-09-05

**阶段：** P1，development-only

**设备：** CPU replay（CUDA 可用但本报告不把 CPU 结果称为 CUDA 等价）

**protocol：** `configs/central_random_mixed_obstacle_s3_v5_v21_cpu_separation_gate_development_protocol.yaml`

**protocol SHA-256：** `278623ceb7185a6c3ce23246e8a28693f025a2977fad95059ae5b0df9a03b014`

**审计输出：** `results/jepa_safe_capture_v21_rolling_audit_cpu_seed20260911_v3/`

**TensorBoard：** `results/jepa_safe_capture_v21_tensorboard/wp4_rolling_audit_cpu_seed20260911_v3/`

## 1. 运行结果

| replay | episodes | control cycles | safe capture | collision | boundary | pairwise | controlled abort | raw unverified | cycle p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| repeat1 | 20 | 1189 | 12/20 (60.0%) | 0 | 0 | 0 | 8 | 0 | 17.482 ms |
| repeat2 | 20 | 1189 | 12/20 (60.0%) | 0 | 0 | 0 | 8 | 0 | 15.858 ms |

两次 replay 均通过运行级安全门。`candidate_ranking`、requested/executed action、CBF 状态、fallback 和 termination 的确定性字段逐条一致，比较的 1189 行 `field_difference_count=0`。wall-clock latency 不参与策略等价比较，只作为性能诊断记录。

## 2. 审计门

- `development_only=true`：PASS
- `locked_test_opened=false`：PASS
- 至少 100 control cycles：PASS
- 至少 500 total control cycles：PASS
- episode/trace/control-cycle 数量一致：PASS
- 每周期 candidate ranking 和 first-step-replan 字段完整：PASS
- CBF failure 均有 fallback：PASS
- non-finite/unverified action 未执行：PASS
- collision/boundary/pairwise：均为 0
- `raw_unverified_executed=0`：PASS
- cycle p95 `<100 ms`：PASS
- repeat semantic trace equality：PASS

## 3. 解释边界

1189 个 control cycles 证明了长序列 rolling-horizon 执行合同和确定性 replay；它不等于 1189 个 hard-context 样本，也不证明 JEPA 提升了 safe-capture。当前 safe-capture 结果只能作为 development replay 记录，不能写成正式提升或 locked-test 证据。

本阶段修复了两个审计问题：

1. TensorBoard 2.4.1 在当前 NumPy/protobuf 环境下的 scalar tag 自检名称与写入名称不一致；自检现绑定实际 aggregate tag。
2. comparator 原先误把 latency 差异当成策略差异；现按 V21 合同排除所有 wall-clock 字段，只比较决策、安全和执行语义字段。

## 4. 下一步

P1 通过后执行同一 scene manifest 的 RTX 5050 CUDA M3 replay，再运行 CPU/CUDA comparator。P2/P3 通过前不扩大到三 seed paired smoke，不训练新 checkpoint，不修改 CBF margin，不打开 locked test。
