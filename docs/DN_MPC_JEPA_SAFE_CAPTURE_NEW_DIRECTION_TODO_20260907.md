# DN-MPC + JEPA + CBF 新方向 TODO 计划

**版本：** v1.0（development-only）
**日期：** 2026-09-07
**目标：** 在不破坏现有 M0/M3/G5、V4 locked 结果和 CBF 安全合同的前提下，引入面向对抗围捕的分布式极小极大 DN-MPC 规划器。

## 0. 不冲突原则

- 不删除或修改历史 V3/V4/V5 结果。
- 不覆盖现有 M0、M3 和 G5 输出；新方向使用独立目录、独立 checkpoint、独立 TensorBoard run。
- 原 Reliability Ledger 保留为旧 M3 对照；新系统使用 `Ledger-Lite`，两者不得混用 calibration archive。
- CBF margin、stale/OOD/non-finite gate、controlled abort、reachable projection 和 raw-unverified 禁止保持不变。
- 所有新结果均标记 `development-only`，在通过多 seed 前不得替代正式 locked benchmark。
- JEPA 不直接生成控制动作；DN-MPC 生成候选轨迹，CBF 决定最终可执行动作。

## 1. 新系统定义

### 1.1 模块职责

| 模块 | 职责 | 不能做的事 |
| --- | --- | --- |
| Delay-aware belief | 融合目标位置、速度、观测 age、障碍物和队友状态 | 不使用未来真值 |
| DN-MPC | 在目标最坏逃逸集合下规划多机协同路线 | 不能替代 CBF 安全证明 |
| JEPA evaluator | 评价候选轨迹的目标进度、净空、可见性和不确定性 | 不直接输出在线动作 |
| Ledger-Lite | 判断 JEPA 是否可信并触发降级 | 不能因预测不可信直接放弃所有捕获机会 |
| Joint CBF-QP | 对最终第一步动作做硬安全过滤 | 不能降低 margin 换成功率 |
| Rolling horizon | 每次执行第一步并重新规划 | 不能跳过独立候选验证 |

### 1.2 DN-MPC 目标

在规划窗口 `H=5` 起步，后续可测试 `H=8`，求解：

```text
min defenders max target_escape
    capture_funnel_cost
  + worst_case_escape_cost
  + formation_and_assignment_cost
  + route_switch_cost
  + control_smoothness_cost
```

约束优先级固定为：

```text
CBF feasibility > obstacle/boundary/pairwise safety
> formation feasibility > capture progress > capture time
```

## 2. P0：合同冻结和基线复核

- [ ] 记录 G5 固定 4 场景 manifest SHA-256、actor、JEPA、Ledger 和 CBF 参数。
- [ ] 记录 M0/M3 完整 L0-L3 总体结果：M0 `46.9%`，M3 `45.8% +/- 5.5%`。
- [ ] 记录 G5 结果：固定 4 场景 `4/4`，安全事件为 0。
- [ ] 建立新输出根目录：`results/dn_mpc_jepa_safe_capture_dev/`。
- [ ] 建立独立 TensorBoard 根目录：`results/dn_mpc_jepa_safe_capture_tensorboard/`。
- [ ] 为每次 run 保存 protocol、git revision、配置和输入 hash。

**P0 gate：** 如果旧基线无法在当前环境复现，停止新方向，不开始 DN-MPC 集成。

## 3. P1：DN-MPC 纯解析规划器

- [ ] 定义目标最坏逃逸集合：速度范围、转向范围、S-curve 和 delayed belief 误差。
- [ ] 定义局部无人机状态、邻居一致性状态和 formation role。
- [ ] 实现径向接近、左切向、右切向、制动、formation split/contract、safe intercept 候选。
- [ ] 实现 reachable dynamics projection。
- [ ] 实现分布式协调：每架无人机局部求解，交换目标预测、角色和编队约束残差。
- [ ] 使用 warm-start、route cache 和 active obstacle ID，避免每个周期从零求解。
- [ ] 输出 route ID、side、predicted cost、worst-case escape cost、solver status 和 residual。

**P1 gate：** 在无 JEPA、无 Ledger、仅 CBF 的 G5 固定场景上，DN-MPC 不能降低 safe capture，且安全事件必须为 0。

## 4. P2：路线状态机和切换抑制

状态机：

```text
approach -> pre_brake -> tangent_left/right -> encircle -> intercept -> capture
                          \-> safe_hold
```

- [ ] 保存 `active_obstacle_id`、`preferred_side`、`route_age` 和 `route_confidence`。
- [ ] 当前路线至少保持 2-5 个控制步。
- [ ] 只有新路线成本改善超过 hysteresis margin 才允许切换。
- [ ] 当前障碍物 ID 变化、路线 CBF 失效或目标逃逸方向突变时允许提前切换。
- [ ] 每次切换都记录原因，不允许因为候选分数微小抖动切换。

**P2 gate：** route switches 和 fallback steps 少于 G5，同时 timeout 不增加。

## 5. P3：JEPA 作为候选轨迹评价器

- [ ] 保留 action-conditioned interaction-aware JEPA 输入合同。
- [ ] 对每个 DN-MPC 候选预测：目标相对位移、route progress、obstacle clearance、boundary clearance、visibility、stopping distance 和 uncertainty。
- [ ] pairwise hazard 在 calibration gate 通过前只能作为辅助排序信号，不能作为硬安全证书。
- [ ] 新 checkpoint 必须使用独立 hard-negative archive，不能复用旧 Ledger。
- [ ] 训练 loss 使用安全优先级：feasibility/clearance > progress > capture time。
- [ ] 训练、validation、calibration 三个 split 必须 seed-disjoint。

## 6. P4：Ledger-Lite 可信度降级

状态定义：

- `trusted`：JEPA 参与候选排序；
- `degraded`：解析 DN-MPC 和 JEPA 混合评分；
- `abstain`：停止使用 JEPA 风险预测，但继续执行解析 DN-MPC + CBF。

检查项：

- [ ] 非有限输出；
- [ ] 观测 age 超限；
- [ ] OOD 距离；
- [ ] JEPA 与解析模型预测分歧；
- [ ] route progress 不一致；
- [ ] calibration credit 和 uncertainty。

`abstain` 不等于 `controlled_abort`。只有 DN-MPC 候选全部无法通过 CBF 时才允许 controlled abort。

## 7. P5：最终 CBF 和滚动闭环集成

- [ ] DN-MPC 只提交候选动作块，不直接执行。
- [ ] 每个候选经过 reachable projection、独立 CBF counterfactual 和最终 CBF。
- [ ] 最终只执行第一步动作。
- [ ] 保持 obstacle/boundary/pairwise margin `0.35 m`。
- [ ] 记录 selected、nominal、safe-hold 三路独立 CBF 结果。
- [ ] 记录 controlled abort、timeout、route switch、fallback、solver residual 和 CBF correction。

## 8. P6：数据和训练

- [ ] 采集目标最坏逃逸下的 DN-MPC candidate archive。
- [ ] 采集左右切向路线成对样本。
- [ ] 采集队友相对速度、拓扑变化和 pairwise transition hard negatives。
- [ ] 增加 `keep/switch/abort`、active obstacle ID 和 route regret 标签。
- [ ] 对每个 archive 写入 metadata、provenance、SHA-256 和 TensorBoard。
- [ ] 先训练 JEPA evaluator，不训练端到端动作策略。
- [ ] calibration gate 未通过时，不建立新 Ledger，不进入在线闭环。

## 9. P7：分阶段实验矩阵

| 阶段 | 配置 | 目的 | 进入条件 |
| --- | --- | --- | --- |
| S0 | M0 | 旧 nominal + CBF 基线 | 基线可复现 |
| S1 | DN-MPC + CBF | 验证规划器本身 | 安全事件 0 |
| S2 | DN-MPC + JEPA + CBF | 验证 JEPA 评价价值 | 不低于 S1 |
| S3 | DN-MPC + JEPA + Ledger-Lite + CBF | 完整新架构 | calibration gate 通过 |
| S4 | 三 seed paired replay | 稳定性验证 | S3 不退化 |
| S5 | L0-L3 development | 分层难度验证 | S4 通过 |

每次运行必须记录 TensorBoard：

```text
Capture/safe_capture
Safety/collision
Safety/boundary
Safety/pairwise
Safety/raw_unverified
Planning/route_switches
Planning/fallback_steps
Planning/worst_case_escape_cost
Planning/solver_time_p95
JEPA/uncertainty
Ledger/state_counts
CBF/correction_norm
```

## 10. 验收标准

### 固定 4 场景开发门

- [ ] safe capture 不低于 G5 的 `4/4`；
- [ ] timeout 不增加；
- [ ] collision、boundary、pairwise、raw-unverified 均为 0；
- [ ] route switches 少于 G5；
- [ ] controlled abort 不增加；
- [ ] control-cycle p95 目标低于 `100 ms`。

### 三 seed 门

- [ ] safe capture 不低于 M0；
- [ ] timeout 不高于 M0；
- [ ] 所有安全事件为 0；
- [ ] 结果在三个 seed 上方向一致；
- [ ] Ledger-Lite 不出现全局过度 abstain。

## 11. 停止规则

立即停止扩展并回放 trace 的条件：

- 任何 collision、boundary、pairwise 或 raw-unverified；
- controlled abort 增加；
- DN-MPC solver timeout；
- route switch 持续增加；
- calibration gate 连续两轮不通过；
- safe capture 低于对应基线且没有安全收益。

若 pairwise calibration 仍失败，则保留 DN-MPC 解析交互模型，JEPA 只做低权重候选评价，不再扩大模型容量。

## 12. 最终目标

最终系统应形成以下闭环：

```text
delay-aware belief
  -> distributed minimax DN-MPC
  -> JEPA counterfactual trajectory evaluation
  -> Ledger-Lite confidence routing
  -> strict Joint CBF-QP
  -> execute-first-step rolling horizon
```

该路线是新 development branch，不改写历史 V4/V5 结论；只有通过固定场景、三 seed 和 L0-L3 全部验收后，才考虑建立新的正式 benchmark。
