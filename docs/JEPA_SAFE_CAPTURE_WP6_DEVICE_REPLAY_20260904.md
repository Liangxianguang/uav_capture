# WP-6 CPU/RTX 5050 Rolling-Horizon Replay Audit

**状态：** completed, development-only
**`locked_test_opened`:** `false`
**protocol revision：** tie3，`score_tie_tolerance_m=5e-4`
**protocol SHA-256：** `3e4f220f7e619bf316907aec6ad91e035cd6ba9b312a6e79f944a020855cce51`

## 输入与运行

CPU 和 CUDA 均使用同一 validation scene manifest、actor checkpoint、JEPA checkpoint、
reliability ledger、environment config 和 protocol。每侧 20 个 episode，所有输出均包含
`summary.json`、`episodes.csv`、逐步 JSONL、scene manifest、provenance 和 TensorBoard。

| 项目 | 路径 |
|---|---|
| RTX 5050 replay | `results/jepa_safe_capture_v3_wp6_replay_m3_seed20260911_cuda_tie3/` |
| CPU replay | `results/jepa_safe_capture_v3_wp6_replay_m3_seed20260911_cpu_tie3/` |
| 审计结果 | `results/jepa_safe_capture_v3_wp6_device_replay_audit_tie3_current/` |

## 结果

| 指标 | RTX 5050 | CPU |
|---|---:|---:|
| safe capture | 7/20 (35.0%) | 7/20 (35.0%) |
| collision / defender boundary / pairwise | 0 / 0 / 0 | 0 / 0 / 0 |
| target boundary violation | 0 | 0 |
| CBF controlled abort steps | 12 | 12 |
| raw/unverified execution | 0 | 0 |
| maximum CBF p95 latency | 25.82 ms | 25.41 ms |

## 等价性判定

- 20/20 settled safety outcomes 相同；
- 20/20 episode 的 CBF verification/fallback 计数相同；
- 746/746 step 的 CBF status 字段相同；
- candidate rejection reason schema 在 746/746 step 存在；
- 输入 provenance、scene manifest 和 episode seed 全部一致；
- 17/20 episode 的 candidate decision 完全一致，709/746 step 的 decision 字段一致；
- 3 个 episode 仍存在 candidate/ledger decision drift，但没有改变 settled safety outcome。

审计分类保持为 `cpu_cuda_safety_settlement_equivalent_decision_drift`。漂移来自不同设备
上的 JEPA 浮点预测累积和 context bucket/score 的近边界判定；tie3 已固定 tie policy，
但不声称 CPU 与 CUDA 的逐步动作 bitwise 相同。最终 development 主实验固定在 RTX 5050
上执行，CPU 仅作为安全结算和故障回退的一致性审计。

机器可读证据：
`results/jepa_safe_capture_v3_wp6_device_replay_audit_tie3_current/device_replay_audit.json`。
该 JSON 和 TensorBoard 均记录 `development_only=true`、`locked_test_opened=false`、
输入哈希、决策漂移计数和延迟门结果。

## WP-6 出口决定

安全闭环门通过，可以进入三 seed paired development final block；候选决策漂移作为
跨设备可复现性限制进入最终报告。final block 中禁止混用 CPU/CUDA 结果、禁止重新调
tie tolerance 或 ledger threshold，并继续要求所有最终动作经过 Joint CBF-QP。
