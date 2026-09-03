# WP6 Smoke Safety Audit: Boundary Semantics

**日期：** 2026-09-04  
**性质：** development-only diagnostic; `locked_test_opened=false`  
**审计对象：** M3 smoke, training seed `20260912`, episode index `19`  
**原始运行目录：** `results/jepa_safe_capture_v3_wp6_smoke_m3_seed20260912/`

## 结论

原始 smoke 结果中的 `boundary_violation=1` 不是 defender 越界。使用原始
scene manifest、actor checkpoint、JEPA checkpoint、reliability ledger 和
protocol 做 deterministic replay 后，唯一检测到的边界事件属于 target：

- target 在 step 104 首次从 `y=-10.0555282781 m` 越过下边界 `-10.0 m`；
- target boundary event 共 8 次（修改后的 replay 继续运行到 capture）；
- defender boundary event 共 0 次；
- CBF 每个执行步均为 `verified_feasible=true`，无 QP infeasible、timeout 或 unverified action；
- 修改后的 entity-specific safety 结算为 `safe_capture=true`，capture at `12.0 s`；
- legacy `world_violation_steps` 仍为 8，仅用于历史兼容，不再作为新 UAV 安全门的唯一来源。

因此，旧 evaluator 的以下组合是语义混淆，而不是 CBF 穿透：

```text
target boundary crossing
    -> legacy world_violation_steps > 0
    -> boundary_violation=True
    -> collision=True / safe_capture=False
```

README 的安全合同明确约束 defender 不得越界，因此新 development evaluator
将 target 越界保留为 `target_boundary_violation` 诊断，将
`defender_boundary_violation` 用作 UAV safety gate。历史 V4/V5 结果和原始
smoke 目录没有被改写；修正后的结果必须在新目录中重新生成。

## 审计输入与产物

审计脚本：`scripts/audit_jepa_safe_capture_v3_boundary_semantics.py`  
审计 JSON：`results/jepa_safe_capture_v3_wp6_boundary_audit_episode19_seed20260912_final/boundary_audit.json`  
审计 JSON SHA-256：`EE0D83CC31A46EF813B2AC45A86D253B4CBCE25EF8C01B4579BED91F2DE21840`  
TensorBoard：`results/jepa_safe_capture_v3_tensorboard/wp6_boundary_audit_episode19_seed20260912_final/`

TensorBoard 至少包含以下 provenance 和指标：

- protocol/checkpoint/ledger/scene manifest 输入及 hash；
- legacy aggregate、target event、defender event 计数；
- replay safe-capture、collision、legacy boundary 和 defender boundary clearance。

运行命令：

```powershell
Set-Location D:\uav-capture\uav_capture
$py = 'D:\miniconda3\envs\uav-encirclement-gpu\python.exe'
$env:PYTHONPATH = "$PWD\src;$PWD\scripts"
& $py scripts/audit_jepa_safe_capture_v3_boundary_semantics.py `
  --run-dir results/jepa_safe_capture_v3_wp6_smoke_m3_seed20260912 `
  --episode-index 19 `
  --output-dir results/jepa_safe_capture_v3_wp6_boundary_audit_episode19_seed20260912_final `
  --tensorboard-dir results/jepa_safe_capture_v3_tensorboard/wp6_boundary_audit_episode19_seed20260912_final `
  --device cuda
```

## 代码修正

- `CaptureRadiusPursuit3DEnv` 新增 `target_world_violation_steps` 和
  `defender_world_violation_steps`，保留 legacy aggregate 字段。
- 环境 info 新增 `target_boundary_violation` 和
  `defender_boundary_violation`。
- 新安全结算只把 defender boundary 视为 UAV safety failure；target boundary
  进入显式诊断和 TensorBoard。
- V2/V3 counterfactual data generator 的 boundary label 改为使用 defender
  boundary，避免后续训练把 target 越界当成 defender unsafe label。
- 新增 target-only、defender-only 和 target-boundary-after-capture 回归测试。

## 下一步准入

1. 以修正后的语义在独立新目录重跑 M0 三 seed、20 集 smoke；
2. M0 无安全硬门失败后，重跑 M3 三 seed、20 集 smoke；
3. 只有 M3 的 collision、defender boundary、pairwise、zero-perturbation、
   latency 和 provenance gates 全部通过，才允许运行 A1/A2；
4. 在 smoke 全部通过前，不启动三 seed final development，也不打开 locked test。

本审计只修复了边界标签归属，不能替代新的闭环 smoke，也不能单独证明
JEPA 提升 safe capture。
