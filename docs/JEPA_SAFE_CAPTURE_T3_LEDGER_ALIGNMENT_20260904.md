# T3 Reliability Ledger Alignment Audit

**日期：** 2026-09-04
**阶段：** development-only，offline reliability audit
**`locked_test_opened`：** `false`
**输入：** T2 settled counterfactual rows，floor015 protocol，seed `20260911`

> 本阶段检查 ledger 的状态路由、不可变性、故障拒答和 settled safety 分桶。低信用 coverage 不足时明确报告 `insufficient_evidence`，不将缺失样本视为成功。

## 1. 审计输入

- ledger：`results/jepa_safe_capture_v4_p10_ledger_seed20260911/reliability_ledger.json`
- protocol：`configs/central_random_mixed_obstacle_s3_v5_rank_guard_floor015_development_protocol.yaml`
- T2 rows：`results/jepa_safe_capture_v4_p12_floor015_settled_cf_{m3,a1,a2}_seed20260911_final2/decision_rows.jsonl`
- ledger minimum credit：`0.65`
- minimum bucket decisions：`20`
- ledger SHA-256：`aec138120f0e4ae5c21ac99f0bae317d8b71cce4a23b0c6c996ecc5a16c84751`
- protocol SHA-256：`7d4710f87805fade62b1e50c3a689cbf9f861917dddb3c8c940b397d129592ec`

M3/A1/A2 的 T2 source gate、scene manifest 和 protocol hash 均一致。A1 是显式的 no-ledger 消融，不计入 ledger credit bucket 结论。

## 2. Credit bucket 结果

`failure` 在本审计中定义为 **settled safety failure**（不是三步内未捕获）；局部 safe-capture 作为独立诊断。

| 变体 | ledger decisions | high-credit decisions | low/missing decisions | high safety failure | low safety failure | safe-capture failure（high/low） | 可评估 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M3 | 994 | 988 | 6 | 23.2% | 100.0% | 97.7% / 100.0% | 否，low<20 |
| A2 | 1,092 | 1,086 | 6 | 23.8% | 100.0% | 97.8% / 100.0% | 否，low<20 |
| A1 | 1,075 | excluded | excluded | excluded | excluded | no-ledger control | 不适用 |

已有 6 个 low/missing 决策中，状态均为 `fallback_nominal`；high-credit safety failure 明显低于这 6 个 low-credit 样本，但样本量不足以建立可靠性门。

## 3. 故障拒答与安全语义

以下六类 fault injection 全部进入预期 `safe_hold`：

- `ood -> ood`
- `stale -> stale_observation`
- `non_finite -> non_finite_context`
- `uncertainty_high -> uncertainty_high`
- `joint_ttc_cbf_risk -> joint_ttc_cbf_risk`
- `unknown_horizon -> missing_bucket`

ledger 文件 hash before/after 相同，未发生在线更新。T2 source runs 的 `raw_unverified_executed_steps` 均为 0；counterfactual branch 的 unverified 仅作为离线候选风险记录。

## 4. Gate 判定

| Gate | 结果 |
|---|---|
| development-only / locked test closed | 通过 |
| scene/protocol provenance | 通过 |
| source T2 gates | 通过 |
| ledger immutable | 通过 |
| fault fallback matrix | 通过 |
| high-credit failure <= low-credit | 可计算但样本不足 |
| high/low bucket coverage >=20 | **不通过** |
| T3 overall | `all_gates_pass=false`，`insufficient_evidence` |

因此 T3 不能授权三 seed final block，也不能把当前 high/low 差异写成正式 ledger 提升。

## 5. 下一步

1. 从独立 calibration archive 生成足够的 low-credit、fallback_nominal、safe_hold、stale、OOD、急转和拥挤队形样本；不得从当前 development episode 反向调阈值。
2. 固定并重新审计 minimum sample count、minimum credit、credit decay/recovery、abstention hysteresis 和状态优先级。
3. 对新 ledger 运行 fault matrix、hash immutability、T2 settled safety 分桶和 TensorBoard audit。
4. 只有 high/low coverage 达标且 high-credit safety failure 不高于 low-credit，才进入 T4 辅助头/困难片段重放；之后再考虑三 seed smoke。

## 6. 复现产物

- `results/jepa_safe_capture_v4_p12_ledger_alignment_seed20260911_v2/ledger_alignment.json`
- `results/jepa_safe_capture_v4_p12_ledger_alignment_seed20260911_v2/report.md`
- `results/jepa_safe_capture_v4_tensorboard/p12_ledger_alignment_seed20260911_v2/`
- `scripts/audit_jepa_safe_capture_v5_ledger_alignment.py`

本阶段结论是“安全和拒答语义通过，可靠性因果证据不足”，所有结果保持 development-only。
