# P0 阶段结果：L0 recovery 三个独立场景 seed

**日期：** 2026-09-06  
**阶段：** P0 执行合同 / L0 recovery development gate  
**状态：** development-only；没有打开 locked test  
**硬件：** NVIDIA GeForce RTX 5050，CUDA 12.8  
**主指标：** `safe_capture`；平均捕获时间仅作诊断

## 1. 本阶段目的

验证计划书中的第一道闸门：在固定 recovery actor、`strict_buffer` CBF 和 anticipatory horizon=5 合同下，L0 open 是否能在不同场景 seed 中稳定完成捕获，并确认安全审计和 TensorBoard provenance 正常工作。

本阶段不是三训练 checkpoint 复现。三个运行共享同一个 actor：

`results/capture_radius_recurrent_behavior_cloning_l0_open_recovery_long_seed661606/checkpoint.pt`

因此结果只能表述为**同一 recovery actor 的三组独立场景 seed 验证**。

## 2. 固定合同

| 项目 | 值 |
|---|---|
| 场景 | L0 open，0 个障碍，nominal observation |
| 每个 seed | 8 episodes |
| actor | L0 recovery actor，recurrent reset interval=1 |
| CBF | `strict_buffer`，horizon=5 |
| operational buffer | obstacle / boundary / pairwise 均 0.35 m |
| candidate profile | legacy，5 candidates |
| projection | reachable dynamics projection=true |
| 执行 | 只执行 CBF 验证后的第一步，随后滚动重规划 |
| stale/OOD/non-finite gate | 保留 |
| controlled abort | 保留 |
| locked test | 未打开 |

为避免把不同标签的重复回放误称为独立 seed，本轮新增 `--development-seed`，显式改变场景 episode/layout seed。三个场景 seed 为 `27400101`、`27410101`、`27420101`。

## 3. 结果

| 场景 seed | episodes | safe capture | collision | boundary | pairwise | raw-unverified | CBF abort | mean capture time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 27400101 | 8 | 8/8 (100%) | 0 | 0 | 0 | 0 | 0 | 5.475 s |
| 27410101 | 8 | 8/8 (100%) | 0 | 0 | 0 | 0 | 0 | 4.625 s |
| 27420101 | 8 | 8/8 (100%) | 0 | 0 | 0 | 0 | 0 | 6.188 s |
| **合计** | **24** | **24/24 (100%)** | **0** | **0** | **0** | **0** | **0** | **5.429 s** |

其他合同检查：

- 最差最小 pairwise clearance：`0.3509 m`，没有 pairwise violation；
- 三组最高 CBF p95 solver latency：`5.86 ms`，低于合同 `100 ms` 上限；
- 三组均 `transit_success=100%`；
- TensorBoard 每组均生成 event file 和 provenance metadata；
- 31 个聚焦回归测试通过。

## 4. 解释和证据边界

这个阶段说明：

1. `horizon=5` recovery actor 在 L0 open 场景上具有稳定的安全捕获能力；
2. CBF 仍保持 strict operational buffer，没有通过执行 raw action 或删除 controlled abort 提高成功率；
3. P0 的执行合同和日志链路可以进入下一关。

这个阶段**不能**说明：

- JEPA 或 Reliability Ledger 已经带来提升，因为本阶段是 M0，没有加载 JEPA/Ledger；
- recovery actor 已完成三训练 seed 复制，因为 checkpoint 只有一个；
- L1/L2/L3 或 S3 已经解决；
- `horizon=5` 可以直接替换历史 V4 locked contract；
- 可以把本结果和历史 V4 `75.3%` 做不匹配协议下的直接比较。

## 5. TensorBoard 和 provenance

每个运行目录都包含 `summary.json`、`provenance.json`、`episodes.csv`、`scene_manifest.jsonl`、`scenes.jsonl`、`step_traces/` 和 TensorBoard event file：

- `results/p0_recovery_h5_m0_l0_independent_seed20260911/`
- `results/p0_recovery_h5_m0_l0_independent_seed20260912/`
- `results/p0_recovery_h5_m0_l0_independent_seed20260913/`
- 对应 TensorBoard 目录分别为上述路径加 `_tensorboard`

每个 provenance 记录：环境版本、RTX 5050/CUDA、git revision、collection config hash、scene manifest hash、CBF contract、recurrent reset interval 和 TensorBoard event file。

## 6. 停止和下一步规则

本阶段没有触发“连续无进步停止”条件：三个独立 L0 seed 均通过。因此下一步可以进入 **P1/P2 的固定微场景审计**，但仍不应直接扩大到完整 L1-L3 训练。

下一阶段必须先完成：

1. selected / nominal / safe-hold 三路独立 CBF counterfactual 的完整归因；
2. braking、left/right/upper/lower detour、formation expand/contract 候选覆盖审计；
3. 含 boundary / pairwise / CBF infeasible 负样本的 counterfactual archive；
4. 新候选合同下重新生成 calibration archive 和 hash-bound Reliability Ledger；
5. 用同一 scene manifest 做 M0/M3 paired L0 验证。

如果新候选或新 JEPA 在 L0 三 seed 上相对 M0 没有正向 paired delta，或者安全硬门不通过，应停止扩大训练，先读取 failure index 判断是候选覆盖、CBF 可行性、Ledger 过度 abstain 还是 JEPA 排序错误。

