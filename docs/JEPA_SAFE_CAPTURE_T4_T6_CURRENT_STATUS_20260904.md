# JEPA Safe Capture T4-T6 当前状态与下一步执行计划

**日期：** 2026-09-04
**范围：** `development_only=true`，`locked_test_opened=false`
**设备：** NVIDIA GeForce RTX 5050，CUDA 12.8，PyTorch 2.7.1+cu128
**第一指标：** `safe_capture`；`mean_capture_time` 仅作诊断

> 本报告记录当前 T4-T6 的真实证据和剩余工作。离线预测头指标不能替代闭环 safe-capture；CBF correction 或 capture time 的变化不能抵消安全失败。

## 1. 已完成证据

### 1.1 Zero-perturbation identity regression

paired evaluator 现在仅在 `use_jepa=true && perturbation_mps==0.0` 时启用
`jepa_zero_perturbation_identity_bypass`：保留 JEPA/ledger checkpoint provenance，但不创建
candidate history、不运行 JEPA、不运行 ranker；actor desired action 直接进入与 M0 相同的 Joint
CBF-QP。非零扰动路径不变。

| run | episodes | safe capture | collision | boundary | pairwise | CBF timeout | raw unverified |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 20 | 10/20 (50.0%) | 0 | 0 | 0 | 0 | 0 |
| M3 zero bypass | 20 | 10/20 (50.0%) | 0 | 0 | 0 | 0 | 0 |

严格比较结果：

- `scenes_geometry_identical=true`；
- `field_difference_count=0`（96 个非诊断物理字段）；
- CBF correction、路径、终止、安全结算逐 episode 一致；
- CBF latency 和旁路标志被明确列为运行诊断字段，不参与物理等价判定；
- 第二次 m3 zero replay 与第一次 m3 zero replay 同样 `field_difference_count=0`。

产物：

- `results/jepa_safe_capture_v4_t3_zero_paired_v2_m0_seed20260911/`
- `results/jepa_safe_capture_v4_t3_zero_paired_v2_m3_seed20260911/`
- `results/jepa_safe_capture_v4_t3_zero_paired_v3_m3_seed20260911/`
- `results/jepa_safe_capture_v4_t3_zero_paired_v2_m3_seed20260911/zero_perturbation_comparison_v2.json`
- comparator SHA-256：`b7ce47de3edac928c347721e207eb8bfd0e72275807dc443200da3e9ab82cb3b`

### 1.2 Non-zero rolling-horizon deterministic replay

M3 的非零候选排序路径已完成两次独立 replay。每次包含 20 episodes 和 1,075 个 ranking/control
cycles；两次均为 `safe_capture=8/20=40.0%`，collision、boundary、pairwise、CBF timeout 和
raw-unverified 均为 0，controlled-abort 均为 11。逐 episode/step 的物理、安全、动作和 termination
字段 `field_difference_count=0`，CBF p95 分别为 `8.205 ms` 和 `6.991 ms`。

产物：

- `results/jepa_safe_capture_v4_t6_rolling_m3_repeat1_seed20260911/`
- `results/jepa_safe_capture_v4_t6_rolling_m3_repeat2_seed20260911/`
- `results/jepa_safe_capture_v4_t6_rolling_m3_repeat2_seed20260911/repeat_comparison.json`
- comparison SHA-256：`047d267633510f35f81c58cc65f7f232b2a16294699aeb1e5242afe0fd744fb3`

### 1.3 Joint CBF-QP 与故障回退

当前代码版本的 9 类 CBF deterministic cases、20 次 repeat 和 ledger fault matrix 均通过：

- 所有输出 finite；
- repeated deterministic=true；
- zero CBF action exact=true；
- infeasible/timeout/non-finite/state-risk 均有显式 fallback 或 controlled-abort；
- `raw_unverified_executed=0`；
- RTX 5050 CBF p95 低于 100 ms。

产物：

- `results/jepa_safe_capture_v4_t4_cbf_qp_audit_v2/audit.json`
- `results/jepa_safe_capture_v4_t4_fault_injection_v2/fault_injection.json`

### 1.4 Held-out safety auxiliary heads

独立 validation counterfactual 数据包含 146,400 samples，其中 hard subset 12,486；replay-off 和
replay-on 全部 finite，训练/验证边界未混入 development S3。

- replay-on 的 CBF-intervention Brier 在 4 个 horizon 均下降，0.1 s 的 hard-subset reduction 为 `0.0233`，95% CI `[0.0157, 0.0313]`；
- replay-on 在 hard subset 的 target position MAE 相比 replay-off 变差，0.1 s reduction 为 `-0.0262`，95% CI `[-0.0321, -0.0203]`；
- 因此结论是 `prediction_signal_mixed`，不是 safe-capture improvement；困难重放需要继续校准，不能直接替换闭环 checkpoint。

产物：

- `results/jepa_safe_capture_v4_t4_aux_head_audit_v2/replay_off_on_validation_comparison.json`
- SHA-256：`08c3dc9e99460142e9a0cd63010461eede2a9b989d8651e58604456ac15ae70e`

## 2. 当前未完成项

- [ ] 汇总 JEPA、ledger、ranker、Joint CBF-QP 的端到端 p50/p95/p99 latency、fallback 和 queue age。
- [ ] 用 WP-8 failure index 逐类复盘 high-credit failure、selected-not-best、低 margin、candidate oscillation、stale/OOD 和 controlled-abort；未能由 trace 证明的原因标为 `unresolved`。
- [ ] 根据 settled rows 冻结或修订 safety-first ranking；任何 weight/margin/hold 变化都创建新 protocol、manifest 和 ledger revision。
- [ ] 完成上述 instrumentation、ranking 和 replay gate 后运行新 protocol smoke；smoke 通过后才扩大至三 seed。

## 3. 下一步 TODO（严格顺序）

### T4.1 安全头校准与 replay

1. 固定当前 validation archive、replay manifest 和 checkpoint hash，不回写历史 archive。
2. 对 clearance、visibility、TTC、CBF intervention/feasibility 做每 horizon calibration：MAE、P90/P95、coverage、under-estimation、Brier/ECE、极端风险漏报。
3. 对 hard subset 分别检查 target drift、遮挡、急转、拥挤队形、CBF correction spike；保留 replay-on 负向 target 证据。
4. 若重新训练，三 seed 使用独立 optimizer/checkpoint/logdir；先 prediction gate，再闭环。

### T5.1 Settled ranking 冻结

1. 从 M3/A2 settled rows 计算 selected-not-best、Spearman/Kendall、top-two margin、switch/oscillation、credit buckets。
2. 固定筛选顺序：finite/reachability -> predicted safety lower bound -> ledger state -> task progress。
3. 保留 nominal exact anchor；low credit、missing bucket、margin 不足或 safety signal 冲突时走 nominal-CBF/safe-hold。
4. 用新 protocol 记录 tie tolerance `5e-4`、abstention margin、clearance floor、hysteresis、minimum hold；不得在最终三 seed 结果上调参。

### T6.1 Rolling-horizon 与安全闭环

1. [x] 选取同一 non-zero M3 run，固定前 100 个 control cycles，重复 replay 两次。
2. [x] 每步验证 trace 存在 belief、candidate、ledger、rank、CBF、executed action；只执行 chunk 第一步。
3. [x] 逐步断言所有 action finite、CBF verification/fallback 可解释、`raw_unverified_executed=0`，并检查 selected action 未绕过 CBF。
4. [x] 对两次 trace 只忽略 wall-clock latency 字段，其余决策、动作、安全状态和 termination 必须一致。
5. [ ] 将 latency 分成 JEPA、ledger、ranker、CBF、全链路；RTX 5050 p95 目标仍为 100 ms，超出则停止扩大规模。

### T7/T8 进入条件

- T4 auxiliary prediction gate：finite、标签非空、校准报告完整；
- T5 settled ranking：trace 可重放，high-credit failure 不高于 low-credit，或明确归档 `no_control_gain`；
- T6：100-cycle non-zero replay、zero regression、fault matrix、latency 和 schema tests 全通过；
- 新 smoke：M0/M3/A1/A2 的 collision、boundary、pairwise 均为 0，raw-unverified 为 0，timeout 为 0 或有 verified fallback；
- 以上均满足后，才运行 `20260911/20260912/20260913` 三 seed、每变体至少 40 集 paired development。

## 4. 结论边界

当前证据支持的系统性质是：JEPA 可以作为 action-conditioned interaction-aware 候选评价器，
ledger 能对风险上下文拒答，Joint CBF-QP 提供统一的不可绕过安全边界，zero regression 证明
JEPA 不改变 nominal actor-to-CBF 物理路径，rolling-horizon 结构已具备逐步 trace。当前尚未证明
JEPA 让闭环 safe-capture 提升；下一阶段应优先修复困难片段校准和排序失配，而不是继续扩大模型。

所有当前产物保持 development-only；未获得明确授权前不打开 locked test，也不把 95% 作为硬目标。

## 5. 预检说明

当前 S3 protocol 使用 `seed_blocks`、`episodes_per_split` 和 `s3` schema，由
`evaluate_random_central_mixed_obstacles.load_protocol` 校验。旧的
`verify_jepa_safe_capture_protocol.py` 只接受 `jepa_safe_capture_system_v2` schema，不能用于本
轮 S3 protocol；后续预检不得混用两个校验器。
