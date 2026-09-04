# v20 CPU deterministic replay：seed 20260911 阶段归档

**阶段：** WP3 单 seed 双设备决定性回放（development-only）  
**代码提交：** `ece741544ccdcc160d5064188cb5cf1747fdbfeb`  
**协议：** `central_random_mixed_obstacle_s3_v5_v20_cpu_deterministic_development_protocol.yaml`  
**协议 SHA-256：** `b8a492faa9448bb0917c124908a044af6cb10813847afeedb78ce675446a2b99`  
**场景：** 复用已冻结的 20 集 validation scene manifest，仅用于 paired replay  

## 结果

CUDA 和 CPU 均为 `9/20 = 45.0% safe_capture`，`0` collision、`0` defender boundary violation、`0` pairwise violation，`0` raw/unverified action。两侧各有 10 个 `cbf_controlled_abort`，并未从分母中删除。

设备审计结果为：

```text
cpu_cuda_safety_and_decision_equivalent
```

| 审计项 | 结果 |
|---|---:|
| safety-equal episodes | 20/20 |
| decision-equal episodes | 20/20 |
| CBF-equal episodes | 20/20 |
| numeric-equal episodes | 20/20 |
| step decision equal | 1167/1167 |
| step CBF equal | 1167/1167 |
| step numeric equal | 1167/1167 |
| candidate rejection reason schema | 1167/1167 |
| raw-unverified CUDA/CPU | 0/0 |

安全审计、输入 provenance、paired episode seeds、CBF verification count 和 RTX 5050 p95 solver latency 门全部通过。运行结果仍是 development evidence，不能替代三 seed paired performance 结论，也不能打开 locked test。

## 产物

- CUDA replay：`results/jepa_safe_capture_v20_cpu_deterministic_replay_m3_cuda_seed20260911_postcommit/`
- CPU replay：`results/jepa_safe_capture_v20_cpu_deterministic_replay_m3_cpu_seed20260911_postcommit/`
- device audit：`results/jepa_safe_capture_v20_cpu_deterministic_device_audit_seed20260911_postcommit/`
- TensorBoard：`results/jepa_safe_capture_v20_cpu_deterministic_tensorboard/`
- machine-readable evidence：`results/jepa_safe_capture_v20_cpu_deterministic_device_audit_seed20260911_postcommit/v20_seed20260911_evidence.json`

所有运行均包含 summary、episodes、scene manifest、step traces、provenance 和 TensorBoard；`development_only=true` 且 `locked_test_opened=false`。

## 结论和下一步

v20 已解决此前 actor backend 导致的 CPU/CUDA decision drift，并证明同一冻结输入下 1167 个 control-cycle 的候选决策、CBF 状态、动作和结算一致。该结果只证明 deterministic replay 和安全执行合同，不证明 JEPA 带来 safe-capture 提升。

下一步必须按计划推进 seed `20260912`、`20260913` 的同样审计；三 seed 设备门通过后，再做 settled ranking、100/500-cycle stress 和 M0/M3/A1/A2 paired smoke。不得通过放宽 CBF 或删除 controlled abort 追逐捕获率。
