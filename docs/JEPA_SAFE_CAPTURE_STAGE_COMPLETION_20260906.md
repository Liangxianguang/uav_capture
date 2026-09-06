# JEPA Safe-Capture 阶段性完成与证据总结

**日期：** 2026-09-06
**阶段：** development-only；`locked_test_opened=false`
**设备：** NVIDIA GeForce RTX 5050，CUDA 13.0，PyTorch 2.9.1+cu130

## 结论先行

本阶段已经完成数据集生成、三 seed hard-replay JEPA 训练、独立校准与
Reliability Ledger、以及 L0-L3 七变体完整滚动闭环评估。结果证明了：

1. action-conditioned interaction-aware JEPA 能稳定预测目标轨迹，且在保留
   不确定性和安全辅助头时优于 constant-velocity 预测基线；
2. Ledger 的 hash binding、OOD/stale/non-finite 拒答和 CBF 安全边界均通过；
3. CBF-enabled 变体没有执行未经验证动作，也没有记录碰撞、边界或 pairwise
   违反；
4. 这些合同和预测证据尚未转化为 episode-level capture 提升。L0-L3 R2
   中 M3 相对 M0 的三 seed 平均配对差为 `-1.0 pp`（`1/3` seed 非负），因此
   当前不能宣称 JEPA 已经优于基线，也不能替代 V4 的 `75.3% +/- 6.5%`
   locked-test 结果。

## 证据链

### 数据与训练

| 项目 | 状态/证据 |
|---|---|
| train archive | SHA-256 `87ceb1a5e9866bde94ceab38bde7576c7b62679094d9d33370d93db07b0961b5` |
| validation archive | SHA-256 `44b80b144cc674c50886eb62ba34320e36805f33835da886971c15fc391500b6` |
| calibration archive | SHA-256 `16b0dc0e1e365eef78adf5d5d5fa876786f42dabb82f89cb0974374afd136719` |
| hard-replay weights | SHA-256 `e94a6d74aeab486dd47ee959a736cb80c375d97b7bd28c0346b001556c0f5a79` |
| training | 3 seed x 40 epoch，finite checkpoint，训练审计通过 |
| TensorBoard | 每个 seed 50 scalar、11 text、227 histogram，provenance 完整 |

三份 checkpoint 的 SHA-256 分别为：

- seed `20260911`: `2ee17218c8e8f08a97b1a164e97a22e62a7ac4d411d66c2f99262c914ec2d828`；
- seed `20260912`: `e8411f7cd5c8bc5db56feef84adc082e75f28310a3521932c363424943ced3d4`；
- seed `20260913`: `aa282e4eb75b2b2be4672526e5d4c09bdbaa29f08d82d716b0303914188ceebc`。

### 预测与 Ledger

| Horizon | JEPA 相对 constant-velocity 的平均位置误差改善 | 平均 Ledger credit | 平均候选 ranking win rate |
|---:|---:|---:|---:|
| 0.1 s | 59% | 0.923 | 0.801 |
| 0.2 s | 73% | 0.913 | 0.854 |
| 0.3 s | 77% | 0.898 | 0.866 |
| 0.5 s | 78% | 0.886 | 0.792 |

三 seed 的 Ledger 均满足 runtime validity、OOD/stale/non-finite 100% 回退、
各路由 unsafe rate 为 0，以及 high-credit failure 不高于 low-credit。
Ledger 仍是可信度路由器，不是安全证书；最终动作必须经过 Joint CBF-QP。

### L0-L3 完整闭环

固定同一 scene manifest（canonical SHA-256
`748d706ae6c2a064c92620200fe4c125a5be6f357c8bf85d7642235efee7a520`），
3 个训练 seed、7 个变体、每个运行 64 集，共 21 个运行和 1,344 条 step trace。

| 变体 | safe capture 均值 +/- seed SD | collision | boundary | pairwise | raw-unverified |
|---|---:|---:|---:|---:|---:|
| M0 | 46.9% +/- 0.0% | 0 | 0 | 0 | 0 |
| M1 | 48.4% +/- 3.1% | 0 | 0 | 0 | 0 |
| M2 | 40.1% +/- 5.0% | 0 | 0 | 0 | 0 |
| M3 | 45.8% +/- 5.5% | 0 | 0 | 0 | 0 |
| A1 | 52.6% +/- 6.3% | 0 | 0 | 0 | 0 |
| A2 | 47.4% +/- 5.0% | 0 | 0 | 0 | 0 |
| A3（raw/no-CBF 诊断） | 0.0% +/- 0.0% | 180 | 12 | 87 | 3192 |

M3 对 M0 的配对差为：seed `20260911` `-1.6 pp`、seed `20260912`
`-6.3 pp`、seed `20260913` `+4.7 pp`。因此自动分类为
`inconclusive_development_evidence`。安全变体 transit success 均为 100%。

## 解释与决策

- **已经有效的部分：** 预测表征、动作条件输入、可信度路由、CBF 最终
  执行边界和 rolling-horizon 记录链路均有可复现证据。
- **当前瓶颈：** L2 delayed/noisy 与 L3 S3 中可验证且有推进能力的候选不足，
  `controlled_abort` 和 all-candidates-ineligible 压低了 capture；不是通过
  放宽 CBF margin 可以解决的问题。
- **不能作出的结论：** 不能把 prediction improvement、zero collision 或
  单个 A1 seed 的较高 capture 写成完整 JEPA 闭环提升；不能打开 V4/V5 locked
  test 追逐目标数字。

## 下一阶段

1. 保留现有安全合同，继续按 failure index 分解 L2/L3 的 abort、候选可行率、
   nominal anchor 和 ranking mismatch；
2. 在 reachable projection 和同一 Joint CBF-QP 下增加 braking、左右切向、
   radial-out、formation expand/contract、safe-intercept 候选；
3. 用困难片段重放和 clearance/visibility/CBF-feasibility/action-consistency
   辅助头重新训练，并为新 checkpoint 建立新的 calibration archive 和 Ledger；
4. 通过 smoke gate 后再运行三 seed paired matrix，不修改旧 manifest，不打开
   locked test。

完整闭环表见
`docs/JEPA_SAFE_CAPTURE_L0_L3_R2_FULL_CLOSED_LOOP_REPORT_20260906.md`；
产物完整性和 TensorBoard 审计见
`docs/JEPA_SAFE_CAPTURE_L0_L3_R2_ARTIFACT_AUDIT_20260906.md`。
