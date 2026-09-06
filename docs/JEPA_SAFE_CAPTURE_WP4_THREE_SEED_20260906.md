# WP4 Actor-Matched Three-Seed Paired Development

**日期：** 2026-09-06
**阶段：** development-only；`locked_test_opened=false`
**主指标：** episode-level `safe_capture`

## 实验内容

WP1/WP2 路线生成已接入完整 runtime。这里用固定的 actor-matched route JEPA
checkpoint、固定 Ledger 和同一 route protocol，分别执行协议 seed
`20260911`、`20260912`、`20260913`，每个 seed 包含 20 个 paired episodes。
每个 seed 先由 M0 生成 manifest，再由 M3 复用对应 manifest。

| 组件 | 值 |
|---|---|
| Protocol | `configs/central_random_mixed_obstacle_s3_route_v1_protocol.yaml` |
| Actor | `models/v5_development_exact_reactive_seed661606.pt` |
| JEPA | `results/jepa_route_identity_model_actor_train40_seed661606/checkpoint.pt` |
| Ledger | `results/jepa_route_identity_ledger_actor_train40_seed661606/reliability_ledger.json` |
| Runtime candidate contract | 12 routes, `obstacle_route_v1`, 3-step chunk, first-step execution |
| Safety contract | CBF margin、stale/OOD/non-finite gate 不变；raw action 禁止 |
| TensorBoard | `results/wp4_actor_matched_paired_tb/three_seed_aggregate` |

## 结果

| Protocol seed | M0 nominal + CBF | M3 route JEPA + Ledger + CBF | Improved / degraded / tied | Paired delta |
|---:|---:|---:|---:|---:|
| 20260911 | 7/20 (35%) | 9/20 (45%) | 4 / 2 / 14 | +10 pp |
| 20260912 | 7/20 (35%) | 9/20 (45%) | 4 / 2 / 14 | +10 pp |
| 20260913 | 7/20 (35%) | 9/20 (45%) | 4 / 2 / 14 | +10 pp |
| **Pooled** | **21/60 (35%)** | **27/60 (45%)** | **12 / 6 / 42** | **+10 pp** |

三个 M3 run 的 collision、defender boundary violation、pairwise violation 和
`raw_unverified_executed_steps` 都为 `0`；CBF timeout 都为 `0`，并且每个
run 的 `cbf_controlled_abort_steps` 与 `cbf_unverified_steps` 一致。M3 每个
run 有 3 个 target-boundary violation，这一目标动力学问题单独报告，不能被
defender 安全硬门掩盖。

## 重要的 provenance 结论

审计工具去除 manifest 顶层 `training_seed` 后得到同一个 canonical hash：

```text
1d9a29dda9e13f9357229c631c14476430c02a993df18ef129709fae17db04e2
```

三个 M3 run 使用同一个 JEPA checkpoint hash
`4b68f8e61df978b79cb58113e6e960fc930912e992469c604abfe5f972a3bdf1` 和同一个
Ledger hash。因此这组结果是 **三次协议 seed replay 的正向 development
signal**，不是三个独立随机初始化模型或三个独立场景块的泛化证据。不能据此
打开 locked test，也不能写成正式三 seed 提升。

## 阶段判定

- safety hard gate：PASS。
- reliability gate：PASS。
- 任务信号：M3 相对 M0 pooled `+10 pp`，但 McNemar/episode 样本仍不足以宣称正式提升。
- 当前分类：`positive_repeated_protocol_replay_signal_not_independent_seed_evidence`。

## 下一步

在同一 actor-matched train/validation archive 上用不同随机初始化训练两个新的
route JEPA checkpoint（例如训练 seed `661607`、`661608`），每个 checkpoint
分别重建 calibration archive 和 Ledger。之后使用独立 development scene
manifest 进行 M0/M3 paired smoke。若 distinct-model-seed 结果不再保持非负，
立即停止扩大规模，转向 route score、target belief、geometry-valid coverage
和 fallback 原因定位。所有新 run 继续保持 `development_only=true`，不打开
locked split。

完整逐 episode 数据、输入 hash 和 TensorBoard event 位于本地 `results` 目录；
生成输出不提交 Git，避免仓库膨胀。
