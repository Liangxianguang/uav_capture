# P2 route geometry sampling performance gate

**日期：** 2026-09-06  
**状态：** development-only；单一 actor-matched route 场景；未打开 locked test  
**目的：** 在不改变 CBF 安全执行边界的前提下，降低 obstacle-route candidate generation 延迟

## 1. 修改

新增显式参数 `--route-corridor-samples`，默认值仍为 `65`，保持当前 route contract 的可复现行为。该参数只改变公共障碍物几何走廊的采样密度；每条候选仍必须经过 reachable projection、primary CBF prefilter、JEPA/Ledger routing 和最终 Joint CBF。

本阶段没有改变：

- CBF margin、速度/加速度约束；
- stale/OOD/non-finite gate；
- controlled abort；
- raw-unverified 禁止；
- obstacle route candidate 顺序和数量；
- actor、JEPA checkpoint、Ledger 或 scene manifest。

## 2. 同场景结果

固定 actor-matched route 场景、`horizon=5`、同一 manifest：

| corridor samples | safe capture | geometry-valid | primary CBF accepted | route p95 (ms) | cycle p95 (ms) | collision/boundary/pairwise/raw |
|---:|---:|---:|---:|---:|---:|---|
| 65 | 1/1 | 1127/1500 | 1127 | 353.29 | 358.66 | 0/0/0/0 |
| 17 | 1/1 | 1128/1500 | 1128 | 116.22 | 123.16 | 0/0/0/0 |
| 9 | 1/1 | 1129/1500 | 1129 | 75.39 | 83.10 | 0/0/0/0 |

CBF solver p95 约 `0.63–0.76 ms`，不是主要瓶颈。`corridor_samples=9` 在这个场景上把 route generation p95 降低约 `78.7%`，并使 cycle p95 首次低于 100 ms。

## 3. 决策

- [x] 将 `--route-corridor-samples` 作为显式、可审计的 development 参数；
- [x] 保留默认 `65`，不破坏历史复现；
- [x] 暂定 `9` 为下一轮性能验证候选；
- [ ] 不将单场景 `1/1` 写成 route-aware 方法提升；
- [ ] 在 `central_single`、`left_blocked`、`right_blocked`、`wall_single_gap` 微场景上验证 9 点采样不会改变应拒绝/应保留的路线；
- [ ] 在至少三个独立 development scene seed 上做 M3 paired replay，确认 safe capture 和 safety hard gates 不下降；
- [ ] 若任一困难场景出现几何漏检，回退到 17 或 65，不以延迟换取安全性。

## 4. TensorBoard / provenance

结果和 TensorBoard：

- `results/p2_route_h5_m3_actor_scene20260911/`
- `results/p2_route_h5_m3_actor_scene20260911_tensorboard/`
- `results/p2_route_h5_samples17_m3_actor_scene20260911_r2/`
- `results/p2_route_h5_samples17_m3_actor_scene20260911_r2_tensorboard/`
- `results/p2_route_h5_samples9_m3_actor_scene20260911/`
- `results/p2_route_h5_samples9_m3_actor_scene20260911_tensorboard/`

每个运行都保存 episode trace、summary、provenance、scene manifest、checkpoint/Ledger hash 和 TensorBoard event file。

