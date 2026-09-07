# JEPA 路线恢复 G2 开发验证结果（2026-09-07）

## 结论

在同一冻结的 4 个 S3 场景上，将 route CBF 的只读因果 probe 从 1 步扩展到 3 步后，完整流程取得了 **4/4 = 100% safe capture**。4 个 episode 中均没有无人机碰撞、边界违规、编队间距违规、raw-unverified 执行、controlled-abort 或 timeout。

这说明“障碍物条件路线候选 + 多步 CBF 可行性验证 + verified-progress fallback”能够修复 G1b 中的单条 timeout。它仍然是 development-only 证据，不是多 seed 或 locked benchmark 结论：平均捕获时间从 G1b 的 11.03 s 增至 18.95 s，fallback 触发 457 次，说明当前 recovery 过于频繁，安全性提升是以进度效率为代价取得的。

## 配置合同

- 场景 manifest：`results/jepa_route_recovery_dev_g1b_seed20260911/scene_manifest.jsonl`
- G2 manifest SHA-256：`6ef41f43d4b37e894559a4ee78604d1f3dc0c08fe48a069e7b4140ad646d327b`
- actor：`models/v5_development_exact_reactive_seed661606.pt`
- JEPA：`results/jepa_route_identity_model_actor_train40_seed661606/checkpoint.pt`
- Ledger：`results/jepa_route_identity_ledger_actor_train40_seed661606/reliability_ledger.json`
- variant：M3（JEPA + Ledger + CBF）
- route profile：`obstacle_route_v1`，12 个候选，投影到 reachable dynamics
- CBF：horizon 3，strict buffer，obstacle/pairwise/boundary margin 均为 0.35 m
- route probe：3 steps；barrier-imminence gate 开启，threshold 0.05 m
- stale/OOD gate、raw-unverified 禁止和 controlled-abort 均保留
- TensorBoard：`results/jepa_route_recovery_dev_g2_seed20260911_tensorboard`

## Paired 结果

| 组别 | route probe | safe capture | collision | boundary | pairwise | raw-unverified | controlled-abort | timeout | 目标越界 | 平均捕获时间 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 原始 route + CBF | 1 | 3/4 (75%) | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 14.87 s |
| G1 stopping + fallback | 1 | 2/4 (50%) | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 13.90 s |
| G1b barrier-imminence | 1 | 3/4 (75%) | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 11.03 s |
| **G2 多步 probe** | **3** | **4/4 (100%)** | **0** | **0** | **0** | **0** | **0** | **0** | **2** | **18.95 s** |

所有组使用同一 manifest；G2 与 G1b 的 manifest 文件哈希一致。安全列统计的是无人机系统约束，目标越界单独报告，不把目标离开仿真边界误计为无人机安全违规。

## G2 运行审计

- control cycles：758
- CBF controlled-abort / infeasible / timeout：0 / 0 / 0
- route probe horizon failures：515
- candidate CBF prefilter：6304 accepted / 515 rejected
- verified-progress fallback：457 steps
- barrier-imminence：457 steps
- 最小障碍净空：0.3522 m（仍高于 0.35 m operational buffer）
- 最小 pairwise 净空：0.3504 m
- CBF solver p95：0.77 ms
- 完整 cycle p95：392.69 ms；主要开销来自候选路线生成（386.50 ms）

因此 G2 的安全验证本身没有超时，但当前完整闭环尚未满足 0.1 s 控制周期的实时性要求。候选生成需要后续做缓存、增量几何检查或降低重复 probe 次数；不能通过降低 CBF margin 来换取速度。

## 根因与解释

G1b 的 barrier gate 可以提前避免 abort，但以 1-step probe 和零 route hysteresis 运行时，恢复动作在大量连续步骤被触发。G2 的 3-step probe 能够区分“当前一步可行但后续即将失效”的路线，因而避免了 timeout；代价是更多候选被提前拒绝，队伍采取更保守的 verified-progress 路线，导致捕获时间增加。

本轮结果支持新方法的安全方向，但不支持以下未经验证的结论：

1. 不能据此宣称多 seed、L1-L3 或正式 locked benchmark 已达到 100%。
2. 不能把目标越界次数当作无人机安全成功。
3. 不能认为 recovery 触发 457 次已经是合理的最终策略。

## 下一步停止/继续条件

1. 保留 G2 作为当前 development reference，并复用同一合同做 3 seed paired replay。
2. 先实现 route hysteresis/minimum hold 或连续性奖励，目标是降低 fallback 次数和平均捕获时间，同时保持所有安全计数为 0。
3. 对候选生成做几何缓存和批量 CBF probe；若 cycle p95 仍高于 100 ms，停止扩大场景规模，优先修复实时性。
4. 只有在 3 seed 中 safe capture 不低于 G0、timeout 不增加、无人机安全违规仍为 0 时，才进入 L1-L3 开发验证。
5. 任一安全违规、raw-unverified 执行或 controlled-abort 增加时，立即停止扩展实验，回到对应 trace 做根因分析。

## 复现命令

```powershell
& D:\download\anaconda3\envs\traj_pred_prep\python.exe scripts/evaluate_jepa_safe_capture_v2_paired.py `
  --variant m3 --training-seed 20260911 --episodes 4 --split validation `
  --protocol configs/central_random_mixed_obstacle_s3_route_v1_protocol.yaml `
  --environment-config configs/capture_radius_pursuit_central_v4_flee.yaml `
  --actor-checkpoint models/v5_development_exact_reactive_seed661606.pt `
  --jepa-checkpoint results/jepa_route_identity_model_actor_train40_seed661606/checkpoint.pt `
  --reliability-ledger results/jepa_route_identity_ledger_actor_train40_seed661606/reliability_ledger.json `
  --scene-manifest results/jepa_route_recovery_dev_g1b_seed20260911/scene_manifest.jsonl `
  --output-dir results/jepa_route_recovery_dev_g2_seed20260911 `
  --tensorboard-dir results/jepa_route_recovery_dev_g2_seed20260911_tensorboard `
  --candidate-profile obstacle_route_v1 --candidate-cbf-prefilter `
  --jepa-perturbation-mps 0.10 --recurrent-reset-interval 1 `
  --cbf-horizon 3 --route-corridor-samples 65 --route-probe-horizon 3 `
  --stopping-distance-gate --verified-progress-fallback `
  --barrier-imminence-gate --barrier-imminence-threshold 0.05 `
  --device auto --development-only
```

验证：`tests/test_route_recovery.py`、`tests/test_obstacle_route_runtime.py`、`tests/test_jepa_safe_capture_wp1_failure_index.py` 共 `25 passed`。
