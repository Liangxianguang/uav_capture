# P2 route runtime：horizon=5 受限反事实 gate

**日期：** 2026-09-06  
**范围：** development-only，单一 actor-matched route 场景；未打开 locked test  
**目的：** 判断扩大 CBF anticipation horizon 是否能改善 route-aware M3 的在线捕获，同时检查实时性

## 1. 严格比较合同

两次运行使用同一：

- actor：`models/v5_development_exact_reactive_seed661606.pt`；
- actor-matched route JEPA：`results/jepa_route_identity_model_actor_train40_seed661606/checkpoint.pt`；
- actor-matched Reliability Ledger：`results/jepa_route_identity_ledger_actor_train40_seed661606/reliability_ledger.json`；
- route protocol、environment config 和 scene manifest；
- `obstacle_route_v1`、12 route candidates、primary CBF prefilter、recurrent reset=1；
- 只执行第一步并滚动重规划。

唯一变化是 CBF `anticipatory_horizon_steps`：3 vs 5。

## 2. 结果

| 配置 | safe capture | collision | boundary | pairwise | raw-unverified | CBF abort | timeout |
|---|---:|---:|---:|---:|---:|---:|---:|
| route M3, horizon=3 | 1/1 | 0 | 0 | 0 | 0 | 0 | 0 |
| route M3, horizon=5 | 1/1 | 0 | 0 | 0 | 0 | 0 | 0 |

因此在这个严格同场景单 episode 对比中，horizon=5 没有带来 episode-level safe-capture 增益。

## 3. 实时性结果

| 指标 | horizon=3 | horizon=5 |
|---|---:|---:|
| route candidate generation p95 | 349.27 ms | 353.29 ms |
| max queue age | 60 steps | 60 steps |
| route candidates | 1500 | 1500 |
| geometry-valid routes | 1127 | 1127 |
| primary CBF accepted | 1127 | 1127 |

`CBF solver` 本身仍低于安全合同延迟上限，但 route candidate generation 已成为主要运行时瓶颈。当前结果不能支持扩大到三 seed 或 40/60 episode。

## 4. 决策

- [x] 保留 horizon=5 作为开发参数，不替换历史 horizon=3 locked contract；
- [x] 停止 route-aware horizon sweep 和多 seed 扩展；
- [x] 不把 horizon=5 写成 safe-capture 提升；
- [x] 保留所有 CBF、Ledger、stale/OOD 和 raw-unverified 安全门；
- [ ] 下一阶段先优化 route candidate generation / CBF probe runtime，并检查队列年龄是否导致观测过期；
- [ ] 优化后重新做同一 scene manifest 的 paired replay，必须同时满足 safe-capture 不下降、硬安全门通过和 queue age/latency 可接受。

## 5. TensorBoard / provenance

本次运行：

- 结果：`results/p2_route_h5_m3_actor_scene20260911/`；
- TensorBoard：`results/p2_route_h5_m3_actor_scene20260911_tensorboard/`；
- 对照：`results/wp3_route_runtime_smoke_actor_full_m3_seed20260911/`及其 TensorBoard 目录。

两次 `summary.json` 和 `provenance.json` 都记录了 checkpoint、Ledger、protocol、scene manifest hash、CBF horizon、设备和 TensorBoard event file。

