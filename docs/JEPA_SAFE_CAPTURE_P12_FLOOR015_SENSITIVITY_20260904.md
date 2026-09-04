# P12 Clearance-Floor Sensitivity 与 Temporal Ledger Audit

**日期：** 2026-09-04
**阶段：** development-only smoke 运行
**`locked_test_opened`：** `false`
**设备：** NVIDIA RTX 5050 / CUDA 12.8 / PyTorch 2.7.1+cu128
**training seed：** `20260911`
**episode block：** 20 集，同一 S3 scene manifest
**第一指标：** `safe_capture`；`mean_capture_time` 仅为次要诊断

> 本报告评估 P12 将 `minimum_predicted_clearance_m` 从 `0.35` 降到经独立 calibration 支持的 `0.15` 后，JEPA 候选是否真正参与排序。该阈值只影响预测筛选，不替代真实几何 Joint CBF-QP。所有结果仍是开发证据，不是 locked-test 结论。

## 1. 固定输入与实验合同

- protocol：`configs/central_random_mixed_obstacle_s3_v5_rank_guard_floor015_development_protocol.yaml`
- scene manifest：`results/jepa_safe_capture_v4_p12_floor015_m0_seed20260911/scene_manifest.jsonl`
- actor：`models/v5_development_exact_reactive_seed661606.pt`
- JEPA：`results/jepa_safe_capture_v3_wp2_seed20260911/checkpoint.pt`
- ledger：`results/jepa_safe_capture_v4_p10_ledger_seed20260911/reliability_ledger.json`
- candidate contract：`K=5`、3-step action chunk、execute-first-step-then-replan、top-two margin `0.0015 m`、hysteresis margin `0.001 m`、minimum hold `2` steps
- CBF contract：Joint `scipy_slsqp_joint_cbf_qp`，obstacle/inter-agent/boundary margin `0.35 m`，timeout `100 ms`
- protocol SHA-256：`7d4710f87805fade62b1e50c3a689cbf9f861917dddb3c8c940b397d129592ec`
- canonical scene manifest SHA-256：`6a5fa0905a6b8391993fba3335452d1f0f3f1b8670749b45346a5ff71e3470ba`

四个变体均使用同一 episode index、场景 hash、actor 和 protocol；只有 JEPA、ledger 与辅助评分开关不同。

## 2. Smoke 结果

| 变体 | 组成 | safe capture | paired vs M0 | collision | boundary | pairwise | CBF timeout | controlled abort | raw unverified |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | nominal + CBF | 10/20 (50.0%) | reference | 0 | 0 | 0 | 0 | 10 | 0 |
| M3 | JEPA + ledger + auxiliary rank + CBF | 9/20 (45.0%) | 0/1/19, -5.0 pp | 0 | 0 | 0 | 0 | 11 | 0 |
| A1 | JEPA + auxiliary rank + CBF，无 ledger | 8/20 (40.0%) | 0/2/18, -10.0 pp | 0 | 0 | 0 | 0 | 11 | 0 |
| A2 | JEPA + ledger + CBF，无 auxiliary clearance/visibility rank | 10/20 (50.0%) | 0/0/20, 0.0 pp | 0 | 0 | 0 | 0 | 10 | 0 |

配对格式为 `improved/degraded/tied`。所有 rank-guard audit 的 `all_gates_pass=true`；这些 20 集结果不能证明三 seed 下的因果提升或不劣性。

## 3. 候选资格与 rank trace

| 变体 | ranking steps | nominal eligible | intercept | lateral | formation | visibility | top-two abstention | switch rate | hysteresis steps | hold steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M3 | 994 | 78.5% | 27.9% | 28.1% | 28.1% | 27.9% | 184 | 0.0755 | 2 | 57 |
| A1 | 1,075 | 100.0% | 35.7% | 35.9% | 35.8% | 35.7% | 231 | 0.0987 | 19 | 87 |
| A2 | 1,092 | 77.4% | 28.8% | 28.9% | 28.8% | 28.8% | 227 | 0.0669 | 0 | 57 |

在四个 JEPA 运行中，所有 prediction 都 finite、没有 invalid selection；`clearance_floor` 时序信号分别触发 M3/A1/A2 的 `671/693/717` 个 ranking steps。降低 floor 后候选不再全部被 `0.35 m` 预测下界挡住，但有效候选仍主要集中在 nominal anchor，且 top-two margin abstention 很频繁。

## 4. CBF 与执行安全

- 四个变体 collision、defender boundary、target boundary、pairwise violation 均为 0。
- CBF timeout 均为 0；所有 CBF unverified steps 都明确记录为 `controlled_abort`，没有被伪装成成功。
- `raw_unverified_executed_steps=0`，且每条 ranking trace 都有 `raw_unverified_executed` 字段。
- mean CBF correction：M0 `0.9400`，M3 `0.2647`，A1 `0.2666`，A2 `0.2701`。该指标只能说明执行修正代价变化，不能代替 safe-capture。
- transit success 四个变体均为 `0.95`；最小几何净空均保持在 CBF margin 附近，未出现安全硬门回归。

## 5. Temporal reliability audit

审计产物：`results/jepa_safe_capture_v4_p12_floor015_temporal_ledger_audit_seed20260911/temporal_ledger_audit.json` 及 `report.md`。

- `all_gates_pass=true`，`development_only=true`，`locked_test_opened=false`。
- OOD、stale observation、non-finite context、高 uncertainty、联合 TTC/CBF risk、unknown horizon 六类故障均进入预期 `safe_hold` reason code。
- ledger SHA-256 before/after 相同：`aec138120f0e4ae5c21ac99f0bae317d8b71cce4a23b0c6c996ecc5a16c84751`。
- 四个运行 ranking prediction 全部 finite，trace raw 字段完整，raw-unverified 为 0。

## 6. 结论边界

1. P12 floor015 修复了 `0.35 m` 预测 floor 过度保守的问题，使非 nominal candidate 在一部分控制周期具备资格；但本 smoke 没有带来 safe-capture 提升。
2. M3 相对 M0 为 `-5.0 pp`，A1 为 `-10.0 pp`，A2 持平。A1 的负向结果与 ledger 对稳定执行的价值方向一致，但单 seed、20 集不足以做因果结论。
3. CBF 几何安全和 raw-action 语义均保持通过；controlled abort 仍是当前首要任务失败模式，必须在后续 settled counterfactual 中定位其发生前的候选、净空和预测误差。
4. 当前不能打开 locked test，也不能宣称 JEPA 已经提升围捕性能。下一阶段应优先完成排序-结算一致性、困难片段重放和三 seed paired development，而不是继续扩大模型规模。

## 7. 下一步 TODO 与准入门

- [x] T2：对 M0/M3/A1/A2 的 degraded/improved/tied episode 做 settled counterfactual replay，计算 selected-not-best、rank correlation、CBF correction 与局部 settled safe outcome 的关系；详见 `docs/JEPA_SAFE_CAPTURE_T2_SETTLED_COUNTERFACTUAL_20260904.md`。
- [ ] T3：在独立 temporal/adversarial calibration archive 上重校准 ledger；冻结 credit、OOD、stale、uncertainty、TTC/CBF risk 阈值并生成新 ledger hash。
- [ ] T4：补齐 clearance、visibility、TTC、CBF intervention/feasibility 头的校准与困难片段双次 deterministic replay；不得回写历史 archive。
- [ ] T5/T6：冻结 candidate score、hysteresis、hold 和 rolling-horizon 顺序，完成至少 100 个 control cycle 的 fallback、latency、zero-perturbation 与 replay audit。
- [ ] T7：在相同 manifest 上每 seed 先运行 M0/M3/A1/A2 各 20 集；任何安全、provenance 或 raw-action gate 失败都停止扩展。
- [ ] T8：仅在 T2-T7 通过后运行 `20260911/20260912/20260913` 三 seed paired development，每变体至少 40 集；以 safe-capture 为主指标，报告 paired delta、bootstrap CI 和 McNemar。
- [ ] locked test：在上述所有门通过且获得明确授权前保持关闭。

## 8. 复现产物

- 运行目录：`results/jepa_safe_capture_v4_p12_floor015_{m0,m3,a1,a2}_seed20260911/`
- rank-guard audits：`results/jepa_safe_capture_v4_p12_floor015_{rank_guard,a1_rank_guard,a2_rank_guard}_audit_seed20260911/`
- temporal ledger audit：`results/jepa_safe_capture_v4_p12_floor015_temporal_ledger_audit_seed20260911/`
- TensorBoard：`results/jepa_safe_capture_v4_tensorboard/p12_floor015_*/`

所有结果保持开发态，不覆盖既有 V4/V5 归档，也不把 archive-recovery checkpoint 作为历史 warm-start checkpoint。
