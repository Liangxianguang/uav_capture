# P2 route-aware 候选路线 gate：停止扩大与失败归因

**日期：** 2026-09-06  
**阶段：** P2 obstacle-conditioned route runtime gate  
**状态：** development-only；停止扩大，不进入多 seed / 40-60 episode block  
**安全原则：** 不降低 CBF margin，不关闭 stale/OOD/non-finite gate，不执行 raw-unverified action

## 1. 触发停止条件

`obstacle_route_v1` 的最新 2-episode runtime smoke：

| 指标 | 结果 |
|---|---:|
| safe capture | 0/2 |
| collision | 0 |
| defender boundary | 0 |
| pairwise | 0 |
| raw-unverified | 0 steps |
| transit success | 2/2 |
| controlled abort | 1 episode |
| timeout | 1 episode |
| route candidates | 3180 |
| geometry-valid | 2774 |
| primary CBF accepted | 2764 |

安全过滤链路没有失效，但任务能力没有改善，且出现了任务终止。因此按照主计划，停止扩大 route profile 的训练和多 seed 评估。

## 2. 当前根因判断

已有 actor-matched route alignment replay 显示：

1. 旧 route archive 使用 `DynamicEncirclementController` 生成主状态，而在线 runtime 使用 frozen reactive actor，存在真实的 train/runtime distribution mismatch；
2. actor-matched archive 虽然把 fallback 从 `136/250` 降到 `41/250`，但 route predicted top-1 与真实 branch top-1 仍只有 `16.0%`，50-step safe capture 仍为 `0/1`；
3. pairwise sign agreement 约 `78.8%`，但 predicted progress `0.603 m` 高于实际 `0.240 m`，说明进度头没有充分建模 CBF correction 和动作执行后的轨迹；
4. route runtime smoke 中有一个周期出现 primary CBF 和 safe-hold 同时不可行，说明候选覆盖或接近障碍后的 stopping feasibility 仍不足；
5. 现有候选数量多、几何有效率高，并不等于有“能在未来 horizon 内捕获且可安全执行”的候选。

因此当前主要问题不是候选数量，也不是继续扩大随机数据，而是：

- actor-matched counterfactual archive 尚未完全绑定到在线执行合同；
- JEPA progress 排序没有对 CBF-filtered outcome 做校准；
- route candidate prefilter 主要检查第一步，未充分暴露后续 stopping / obstacle feasibility；
- safe-hold 在已经进入困难状态后才被调用，不能替代提前 braking / formation action。

## 3. 暂停事项

- [x] 暂停 `obstacle_route_v1` 多 seed 扩展；
- [x] 暂停 40/60 episode 大规模 route training；
- [x] 暂停继续增加随机候选数量；
- [x] 不将 `0/2` smoke 写成完整 M3 结论；
- [x] 保留所有安全硬门和 `controlled_abort`。

## 4. 恢复条件

恢复 route-aware 训练前必须完成：

1. 用 frozen runtime actor 重新生成 train/validation/calibration counterfactual archive；
2. 对每个 route 保存 CBF-filtered trajectory、最早风险 step、stopping distance、TTC、最终目标进度和 route identity；
3. 重新校准 progress / clearance / feasibility heads 和 route-bound Ledger；
4. 在同一 50-step alignment audit 上使 predicted-vs-actual route ranking 明显高于当前 `16.0%`，并在固定微场景得到非零 safe capture；
5. 先通过 L0 三场景 seed 的 M0/M3 paired gate，再允许进入 L1-L3。

在上述条件未满足前，任何新实验都只允许是失败 trace replay、标签审计、单元测试或小规模合同 smoke；不得继续堆叠模型规模。

## 5. 已记录的 TensorBoard / provenance

相关运行均已保存 TensorBoard event file、episode CSV、step trace、scene manifest 和 provenance：

- `results/wp2_route_runtime_smoke_v4_m3_seed20260911/`
- `results/wp2_route_runtime_smoke_tb_v4_m3_seed20260911/`
- `results/wp3_route_alignment_audit_actor_full_m3_seed20260911/`
- `results/wp3_route_alignment_audit_actor_full_m3_seed20260911_tb/`

这些目录是开发诊断产物，不是 locked benchmark。

