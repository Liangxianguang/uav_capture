# WP3：V21 Hash-Bound Reliability Ledger 审计

**日期：** 2026-09-05  
**阶段：** development-only  主题：`Interaction-Aware Action-Conditioned JEPA + Reliability Ledger + CBF`  
**协议：** `configs/central_random_mixed_obstacle_s3_v5_v21_cpu_separation_gate_development_protocol.yaml`  
**协议 SHA-256：** `278623ceb7185a6c3ce23246e8a28693f025a2977fad95059ae5b0df9a03b014`  
**locked 状态：** `locked_test_opened=false`

## 1. 目的和边界

本阶段验证三份 checkpoint-bound ledger 能否在进入 rolling-horizon 闭环前拒绝 provenance 错配、OOD、陈旧观测和非 finite 预测。该审计不运行新的 episode policy，不读取 online target truth，也不宣称 safe-capture 控制收益。

JEPA 仍然只是候选轨迹评价器；所有实际控制动作的最终安全边界仍由同一个 Joint CBF-QP 提供。已有 CBF fault audit 独立验证 timeout、infeasible、non-finite request 和 controlled-abort 路径。

## 2. 固定输入

| 输入 | SHA-256 |
|---|---|
| calibration archive `counterfactual_safe_capture_v2.npz` | `ea04eec8e255bcafa95386ef4c30e366e55723334b8d4985d6c94887b9a1a307` |
| calibration metadata `metadata.json` | `531ce966d78cc448df4868bc071e507fa64bc9a7b1ee0d121ad367bba20ec6f0` |
| checkpoint seed `20260911` | `2317a9464f8001f27a5c028bb6b4c431c904af7bfc33bf43b3a1d05a5a9c6154` |
| checkpoint seed `20260912` | `8ff2531e64571c9e57cfd78e9023a8b49191e06d1c4e4fd00adfaec90b629185` |
| checkpoint seed `20260913` | `9fe66b66a6ea441807022c1fde71e61b578df3df6ab7265532761d70d6fab708` |

每个 checkpoint 使用相同的 calibration archive 独立推理并生成自己的 q=0.10 clearance transform，未复用其他 seed 的模型输出。

## 3. 结果

权威结果：

- `results/jepa_safe_capture_v21_hash_bound_ledger_audit_v2/hash_bound_ledger_audit.json`
- `results/jepa_safe_capture_v21_hash_bound_ledger_audit_v2/hash_manifest.json`
- `results/jepa_safe_capture_v21_hash_bound_ledger_audit_v2/report.md`
- `results/jepa_safe_capture_v21_tensorboard/wp3_hash_bound_ledger_v2/`

| Gate | 结果 |
|---|---:|
| development-only / locked closed | PASS |
| 三 seed 输入完整 | PASS |
| checkpoint/protocol/calibration hash binding | PASS |
| tampered provenance rejected | PASS |
| ledger file immutable before/after audit | PASS |
| OOD/stale/non-finite/unknown horizon/uncertainty/TTC-CBF fault matrix | PASS |
| raw unverified decision count | `0` |
| TensorBoard event and required tags | PASS |

三个 seed 的 q=0.10 calibration transform hash：

| Seed | Transform SHA-256 |
|---:|---|
| 20260911 | `efff8b5e0683ece9abd276f7fdfd4d687dcaa95ca261af85dccdbfe94b8e2bf3` |
| 20260912 | `e34367163d8c48b53927e10811be0a1082ed760d97436c6c1b97a2ba21504a44` |
| 20260913 | `064a3f68ae4287a52e943a273886ba59e65f52bbe08a88651d42758770478f75` |

这些差异来自三个 checkpoint 的独立预测残差，不代表协议不一致。

## 4. 故障语义

| 故障 | 预期状态 | 预期 reason |
|---|---|---|
| OOD | `safe_hold` | `ood` |
| stale observation | `safe_hold` | `stale_observation` |
| non-finite context | `safe_hold` | `non_finite_context` |
| uncertainty spike | `safe_hold` | `uncertainty_high` |
| joint TTC/CBF risk | `safe_hold` | `joint_ttc_cbf_risk` |
| unknown horizon | `safe_hold` | `missing_bucket` |
| tampered protocol provenance | audit reject | `source_hash_gate=false` |

ledger 在审计过程中只读；任何 fault 都不允许升级为 `trusted`，也不生成 raw execution permission。

## 5. 可复现命令

```powershell
$py = 'D:\download\anaconda3\envs\traj_pred_prep\python.exe'
& $py scripts/run_with_tensorboard_compat.py `
  scripts/audit_jepa_safe_capture_v21_hash_bound_ledger.py `
  --protocol configs/central_random_mixed_obstacle_s3_v5_v21_cpu_separation_gate_development_protocol.yaml `
  --calibration-dataset results/jepa_safe_capture_v2_p1_corrected_frame_calibration/counterfactual_safe_capture_v2.npz `
  --calibration-metadata results/jepa_safe_capture_v2_p1_corrected_frame_calibration/metadata.json `
  --seed 20260911 results/jepa_safe_capture_v11_hard_replay_seed20260911/checkpoint.pt results/jepa_safe_capture_v21_ledger_seed20260911/reliability_ledger.json `
  --seed 20260912 results/jepa_safe_capture_v11_hard_replay_seed20260912/checkpoint.pt results/jepa_safe_capture_v21_ledger_seed20260912/reliability_ledger.json `
  --seed 20260913 results/jepa_safe_capture_v11_hard_replay_seed20260913/checkpoint.pt results/jepa_safe_capture_v21_ledger_seed20260913/reliability_ledger.json `
  --output-dir results/jepa_safe_capture_v21_hash_bound_ledger_audit_v2 `
  --tensorboard-logdir results/jepa_safe_capture_v21_tensorboard/wp3_hash_bound_ledger_v2 `
  --development-only
```

## 6. 结论和下一步

WP3 已通过，可以进入 WP4 rolling-horizon/Joint CBF 长序列回归。该结果只证明 ledger 的 provenance、拒答和不可变性合同，不证明候选排序已经改善 safe-capture，也不允许打开 locked test。
