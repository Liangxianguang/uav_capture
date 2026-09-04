# v20 三 seed CPU/CUDA deterministic device audit

**阶段：** WP1 / development-only  
**协议：** `configs/central_random_mixed_obstacle_s3_v5_v20_cpu_deterministic_development_protocol.yaml`  
**协议 SHA-256：** `b8a492faa9448bb0917c124908a044af6cb10813847afeedb78ce675446a2b99`  
**代码 revision：** `9fe8cb5be1c2727b669a0386b479cfc128e86aad`  
**locked 状态：** `locked_test_opened=false`

## 结果

三 seed 在相同冻结场景、episode seed、checkpoint、ledger 和 observation schedule 下完成 CPU/CUDA replay。actor 和 candidate ranking 均固定在 CPU，两个 backend 的候选决策、CBF 状态、动作、终止和安全结算逐字段一致。

| Seed | Safe capture | Control steps | Decision equal | CBF equal | Numeric equal | Collision | Boundary | Pairwise | Raw/unverified |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260911 | 9/20 (45.0%) | 1167 | 1167/1167 | 1167/1167 | 1167/1167 | 0 | 0 | 0 | 0/0 |
| 20260912 | 7/20 (35.0%) | 935 | 935/935 | 935/935 | 935/935 | 0 | 0 | 0 | 0/0 |
| 20260913 | 9/20 (45.0%) | 1133 | 1133/1133 | 1133/1133 | 1133/1133 | 0 | 0 | 0 | 0/0 |

跨 seed `safe_capture` mean 为 `41.67%`，sample SD 为 `5.77%`。CBF controlled-abort steps 分别为 `10`、`12`、`10`，均保留在失败分母和 trace 中。

## Gate

三 seed 均通过 `cpu_cuda_safety_and_decision_equivalent`，并通过以下 gates：输入 provenance、paired episode seeds、settled safety outcomes、candidate decisions、CBF verification counts、candidate rejection schema、raw-unverified zero 和 RTX 5050 p95 CBF latency。

## 产物

- 汇总 JSON：`results/jepa_safe_capture_v20_cpu_deterministic_device_audit_three_seed_final2/device_replay_three_seed.json`
- 汇总报告：`results/jepa_safe_capture_v20_cpu_deterministic_device_audit_three_seed_final2/report.md`
- TensorBoard：`results/jepa_safe_capture_v20_cpu_deterministic_tensorboard/device_audit_three_seed_final2/`
- 汇总脚本：`scripts/aggregate_jepa_safe_capture_v20_device_replay.py`
- 原始 seed audit：`results/jepa_safe_capture_v20_cpu_deterministic_device_audit_seed20260911_final/`、`seed20260912_final/`、`seed20260913_final/`

运行命令：

```powershell
$py = 'D:\download\anaconda3\envs\traj_pred_prep\python.exe'
$env:PYTHONPATH = "$PWD\src;$PWD\scripts"
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = 'python'
& $py scripts/aggregate_jepa_safe_capture_v20_device_replay.py `
  --output-dir results/jepa_safe_capture_v20_cpu_deterministic_device_audit_three_seed_final2 `
  --tensorboard-dir results/jepa_safe_capture_v20_cpu_deterministic_tensorboard/device_audit_three_seed_final2
```

## 结论和限制

该阶段证明了设备决定性和安全执行合同，不证明 JEPA 带来 safe-capture 提升。settled counterfactual ranking audit 仍显示三 seed 的 score 与 settled outcome 反向相关，因此下一步必须先完成 score orientation、settled label、horizon 和 action scale 诊断；在该阻断门通过前，不进入 40/60 集 development block，也不打开 locked test。
