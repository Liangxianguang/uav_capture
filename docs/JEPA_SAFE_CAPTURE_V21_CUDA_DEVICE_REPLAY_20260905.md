# V21 RTX 5050 CUDA Replay 与 CPU/CUDA Comparator 报告

**日期：** 2026-09-05

**阶段：** P2/P3，development-only

**GPU：** NVIDIA GeForce RTX 5050，CUDA `13.0`

**Python：** `D:\\download\\anaconda3\\envs\\traj_pred_prep\\python.exe`

**protocol：** `configs/central_random_mixed_obstacle_s3_v5_v21_cpu_separation_gate_development_protocol.yaml`

**protocol SHA-256：** `278623ceb7185a6c3ce23246e8a28693f025a2977fad95059ae5b0df9a03b014`

**CUDA run：** `results/jepa_safe_capture_v21_wp4_replay_m3_cuda_seed20260911/`

**comparator：** `results/jepa_safe_capture_v21_device_comparator_seed20260911/`

**TensorBoard：** `results/jepa_safe_capture_v21_tensorboard/wp4_replay_m3_cuda_seed20260911/` 和 `results/jepa_safe_capture_v21_tensorboard/wp4_device_comparator_seed20260911/`

## 1. CUDA replay

| episodes | control cycles | safe capture | collision | boundary | pairwise | CBF timeout | controlled abort | raw unverified | cycle p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 1189 | 12/20 (60.0%) | 0 | 0 | 0 | 0 | 8 | 0 | 17.5057 ms mean episode p95 |

其它诊断值：`transit_success_rate=0.95`、`mean_capture_time=8.5917 s`、`mean_min_clearance=0.4006 m`、`mean_cbf_action_correction_norm=0.2754`。这些诊断值不改变 `safe_capture` 结算，也不构成任务收益声明。

运行 metadata 确认：`device=cuda`、RTX 5050、`development_only=true`、`locked_test_opened=false`、JEPA 仅用于候选评价、所有实际动作通过 Joint CBF-QP。

## 2. CPU/CUDA comparator

CPU 对照使用同一 scene manifest 的 V21 CPU replay repeat1。比较绑定以下输入：protocol、environment config、actor checkpoint、JEPA checkpoint、reliability ledger 和 scene manifest；episode seeds 全部配对。

| comparator gate | result |
|---|---:|
| input provenance equal | PASS |
| paired episode seeds | PASS |
| settled safety outcomes equal | PASS |
| settled safety zero in both | PASS |
| candidate decisions equal | PASS |
| CBF verification counts equal | PASS |
| raw unverified execution zero in both | PASS |
| candidate rejection reason schema | PASS |
| CBF p95 latency under 100 ms | PASS |

最终分类为 `cpu_cuda_safety_and_decision_equivalent`。比较忽略 wall-clock latency，只对 candidate decision、ledger/ranker 路由、CBF verification、executed action、termination 和 safety settlement 做语义检查。

## 3. 结论边界

CPU 与 CUDA 在同一 manifest 下的决策和安全结算等价，证明了当前 RTX 5050 部署路径的设备一致性合同。它不证明 JEPA 相对于 M0 的 safe-capture 提升，也不替代三 seed paired smoke。当前 60% 仅是 20 集 development replay 的结果，不能写成正式泛化结论。

## 4. 下一步

P1-P3 已通过。下一步运行三 seed 的 M0/M3/A1/A2 paired smoke：每 seed 20 集，M0 先生成 manifest，其他变体复用同一 manifest。只有安全硬门和 paired 证据满足计划书出口条件，才允许扩大到 40/60 集 development；否则归档 `prediction_signal_no_control_gain`，不训练新模型或打开 locked test。
