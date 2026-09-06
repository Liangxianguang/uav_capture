# JEPA + Reliability Ledger + CBF 围捕系统下一阶段主 TODO 计划

**版本：** Next Master Plan 2026-09-06  
**范围：** development-only 仿真 / SITL 验证；不打开历史 locked test，不直接用于真实飞行部署  
**硬件：** RTX 5050  
**主指标：** episode-level `safe_capture`  
**安全硬门：** `collision=0`、`boundary=0`、`pairwise=0`、`raw_unverified=0`；保留 `controlled_abort`

## 当前执行状态（2026-09-06）

- **P0 L0 recovery：已通过开发闸门。** 同一 recovery actor 在三个独立场景 seed、24 个 L0 open episode 上达到 `24/24 safe_capture`，所有安全硬门通过；结果见 [P0 report](JEPA_SAFE_CAPTURE_P0_L0_RECOVERY_THREE_SCENE_SEEDS_20260906.md)。这不是三训练 checkpoint 结论。
- **P2 route-aware runtime：已暂停扩大。** `obstacle_route_v1` 最新 smoke 为 `0/2 safe_capture`，虽无安全违规，但出现一次 controlled abort 和一次 timeout；结果见 [route gate stop report](JEPA_SAFE_CAPTURE_P2_ROUTE_GATE_STOP_20260906.md)。
- **当前唯一允许的下一步：** actor-matched counterfactual、route ranking/CBF failure replay 和标签/校准审计；在通过固定微场景 gate 前，不扩大 route 多 seed、40/60 episode 或新模型规模。
- **WP1 fixed micro-scene gate：已通过。** Corridor sampling `65/17/9` 在 `central_single`、左右封锁和 `wall_single_gap` 四个公开观测场景上保持相同的 valid/CBF-verified 路线集合；独立 4097 点复核的 geometry false accept 为 0。详见 [WP1 route sampling and paired replay report](JEPA_SAFE_CAPTURE_WP1_ROUTE_SAMPLING_AND_PAIRED_REPLAY_20260906.md)。
- **WP1 bounded runtime replay：已完成但不扩大。** 同一三场景 manifest 上 M0 为 `1/3`、M3 为 `2/3`，安全硬门均为 0；M3 的一个 timeout 由 Ledger `safe_hold` 过度拒绝触发。M1 去 Ledger 诊断在同一场景为 `1/1`，因此下一步是 stale/OOD Ledger 路由审计，不是降低 CBF 或立即重训。
- **WP1 Ledger abstention audit：已完成。** 在 M3 episode `646102` 的 250 步中，`safe_hold=250`，其中 `208` 步为“eligible=0 但至少一个 route CBF verified”；三路独立 CBF probes 为 `250/250` 全通过，安全硬门仍为 0。下一步只允许设计 bounded `cautious_reacquisition` 合同，不得把 stale/OOD 重标为 trusted。详见 [Ledger abstention audit](JEPA_SAFE_CAPTURE_WP1_LEDGER_ABSTENTION_AUDIT_20260906.md)。
- **WP1 bounded cautious reacquisition：已完成并停止扩大。** 新合同只允许三步 `visibility_hold`，仍保留 Ledger `safe_hold`、stale/OOD gate 和三路独立 CBF probes；同一三场景 replay 仍为 `2/3 safe_capture`，episode `646102` 仍 timeout，虽无安全回归。详见 [stop report](JEPA_SAFE_CAPTURE_WP1_CAUTIOUS_REACQUISITION_STOP_REPORT_20260906.md)。
- **WP1 三训练 seed smoke：已完成并停止扩大。** 在同 manifest 配对的 M0/M3 上，M0 pooled 为 `6/9`，M3 pooled 为 `3/9`，paired delta `-33.3 pp`；UAV collision/boundary/pairwise/raw-unverified 均为 0。retry checkpoint 同 manifest 仍为 `2/3`，未改变结论。当前标签为 `prediction_signal_no_control_gain`；详见 [three-seed stop report](JEPA_SAFE_CAPTURE_ACTIVE_SEARCH_THREE_SEED_STOP_REPORT_20260906.md)。下一步只做 settled route-regret、stale/Ledger 和 candidate ranking 离线归因，不继续训练或扩大 L1-L3。

## 0. 最终目标和当前判断

最终系统是一个安全增强的闭环：

```text
局部观测 / 障碍物几何
  -> 可解释候选路线与动作块
  -> reachable dynamics projection
  -> action-conditioned interaction-aware JEPA 反事实评价
  -> reliability ledger 可信度路由
  -> horizon-aware Joint CBF 过滤
  -> 只执行第一步
  -> 更新观测并滚动重规划
```

JEPA 只负责对候选轨迹进行预测和评价，不直接越过 CBF 产生最终执行动作。候选生成器必须知道当前可观测障碍物的位置、形状、可用绕行侧和编队空间；CBF 负责最终安全执行。

当前证据说明：

1. L0 open 的失败主要是 actor 晚刹车和 `horizon=3` 导致的联合约束不可行，不是实际碰撞，也不是 timeout。
2. 在同一 8 个冻结场景上，recovery actor + `horizon=5` 达到 `8/8`；加入 JEPA/Ledger 后仍为 `8/8`。这是开发证据，不是正式 locked 结论。
3. 完整 L0–L3 R2 结果仍只有 M0 `46.9%`、M3 `45.8%`，但所有安全事件为 0；因此主要瓶颈是“没有足够多的可验证捕获机会”，不是安全保护失效。
4. `physical_feasibility` CBF 开发消融将同一矩阵从 `5/64` 提升到 `8/64`，但不应被解读为保留原 `0.35 m` operational buffer 的正式提升。历史 `strict_buffer` 必须继续保留作基线。

## 1. 统一研究问题

> 在不牺牲物理安全不变量的前提下，如何让学习型世界模型和规划器更早提出、识别并执行可验证的捕获路线，从而减少因晚刹车、路线单一、过度拒绝和对抗目标预测漂移造成的 `controlled_abort`？

本计划只优化四个互相可定位的因素：

- **CBF 可行性：** 从“进入危险后拒绝”改为“提前估计 stopping distance / TTC / viability”。
- **候选覆盖：** 从围绕 nominal 的小扰动改为障碍物条件的不同路线和编队动作块。
- **JEPA 评价：** 从只预测目标位移扩展到净空、可见性、CBF 可行性和路线身份。
- **可信路由：** 让 Ledger 能区分模型不确定、信息过期、候选不可行和确实没有安全路线。

## 2. 永久不变的安全和信息边界

- 不降低历史 CBF margin，不删除 stale、OOD、non-finite 或 Ledger gate。
- 不执行未经 `reachable projection -> CBF-QP -> residual verification` 验证的动作。
- `controlled_abort` 是合法安全结果，必须保留并计入 `safe_capture=false`。
- 在线候选只使用当前可获得的局部观测、通信状态和障碍物几何；目标真实未来轨迹只作为离线 label。
- 所有新结果都标注 `development-only`，不覆盖 V4/V5 历史 locked 结果。
- 只在仿真/SITL 中验证，不把仿真成功率表述为真实飞行安全保证。

## 3. 当前优先级排序

### P0：先修执行合同，暂停大规模训练

这是下一步第一件事。没有这一步，继续增加数据集和模型只会把合同错误放大。

- [ ] 冻结一个新 protocol：actor checkpoint、候选 profile、动作尺度、chunk 长度、重规划频率、CBF horizon、`barrier_mode`、recurrent reset interval、观测信息边界。
- [ ] 为 protocol、scene manifest、checkpoint、calibration archive 和 Ledger 生成 SHA-256 provenance。
- [ ] 在每个 episode/step 记录：`requested_action`、`projected_action`、`cbf_action`、`executed_action`、最小 slack、最早失败 step、失败约束类型和 fallback reason。
- [ ] 独立运行 selected candidate、nominal anchor、safe-hold 三条 CBF counterfactual；三者不得共享已修改的状态或隐藏结果。
- [ ] 验证 `message_age` 状态机不会全量饱和，分别记录 observation age、communication age 和 model rollout age。
- [ ] 检查 candidate eligibility、score direction、nominal anchor、recurrent reset 和每条候选是否先经过 reachable dynamics projection。
- [ ] 固定历史基线：`strict_buffer + horizon=3`；固定开发分支：`horizon=5`，并单独标注 `physical_feasibility`。

**P0 通过条件：** 同一 scene manifest 重放可得到一致 trace；所有 abort 都能归因到 boundary / obstacle / pairwise / acceleration / stale-OOD / solver；没有 raw-unverified 执行；核心测试通过。

### P1：重设计 CBF 为“物理硬约束 + 运营风险诊断”

CBF 不能为了提高捕获率而放松物理安全，但也不能把可恢复的 operational buffer 误判为不可恢复。采用双层语义：

1. **物理硬层：** 无人机碰撞、世界边界、障碍物接触、pairwise physical contact、速度/加速度/可达动力学为硬约束。
2. **运营风险层：** `0.35 m` 等 buffer、预测净空、TTC、formation slack 和 time-to-boundary 用于排序、预警和提前制动；不能绕过物理层，也不能在报告中冒充保证。

实现任务：

- [ ] 加入 stopping-distance 与 time-to-boundary feasibility 检查，避免到最后一步才发现刹车不可行。
- [ ] 加入多步 `horizon=5` counterfactual；比较 horizon 3/5/7 的安全事件、abort、候选覆盖和延迟。
- [ ] 对 pairwise 约束加入 TTC、相对速度和编队拓扑；在 pairwise margin 激活前触发 formation expand/contract 或 braking。
- [ ] 将 CBF infeasibility 分解为约束兼容性、动作边界、solver failure 和信息过期，不用单一 `infeasible` 掩盖原因。
- [ ] 保留 `strict_buffer` 作为历史可复现基线；`physical_feasibility` 只作为明确命名的开发消融。
- [ ] 记录每次 margin/buffer 进入风险区的提前量，验证是否在真正不可行前已经减速或改道。

**P1 通过条件：** L0 recovery 三个 seed 均无安全违规；至少 95% 的 abort 有明确可解释根因；horizon 增大不造成 raw-unverified 或物理安全事件；端到端延迟仍满足仿真控制周期。

### P2：建立障碍物条件和交互感知候选动作块

候选必须代表真正不同的路线，而不是对 nominal 速度做固定小扰动。固定顺序并写入 protocol：

1. `nominal`
2. `left_detour`
3. `right_detour`
4. `upper_detour`
5. `lower_detour`
6. `radial_out`
7. `braking`
8. `formation_split`
9. `formation_expand`
10. `formation_contract`
11. `safe_intercept`
12. `verified_safe_hold`

实现任务：

- [ ] 每个障碍物由统一 SDF / signed-distance 接口计算前后、左右、上下 clearance、目标走廊阻塞和可用绕行侧。
- [ ] 对左/右/上/下绕行生成 waypoint/corridor，再通过 reachable dynamics projection 变成 action chunk。
- [ ] `braking`、`radial_out`、formation 动作必须显式改变速度或拓扑，而非只改变 JEPA 标签。
- [ ] 每个候选保存 route identity、障碍物 id、绕行侧、预计长度、目标进度、几何拒绝原因和 CBF 失败原因。
- [ ] 观测无法区分两侧时保留多个候选交给 JEPA；不得使用隐藏 ground truth 强行选择路线。
- [ ] 每步只执行第一步，执行后重新观测和重规划。

**P2 通过条件：** 固定微场景中 left/right/upper/lower 轨迹几何确实不同；窄通道能够触发 formation contract；所有候选都有独立 CBF 结果；候选覆盖率和“至少一条可验证候选”的比例可统计。

### P3：生成真正有负样本的 counterfactual archive

当前 archive 的安全和可行性负样本不足，导致 JEPA 不知道“看起来接近目标但未来会撞边界/编队”的候选。

数据分层：`train / validation / calibration / development` episode seed 完全不重叠；每个样本保存 scene hash、episode seed、protocol hash。

采样重点：

- earliest-abort 片段；
- 高速度接近边界、急刹车和 pairwise 临界片段；
- 左右绕行一侧可行、另一侧阻塞的片段；
- 窄通道、混合障碍、S-curve；
- dropout、观测噪声、通信延迟和消息过期；
- 目标突然转向、加速度变化和可见性丢失；
- 可行/不可行、低/高 buffer、低/高 uncertainty 的分层 counterfactual。

每个候选离线 label 至少包括：

- 目标相对位置/速度/加速度；
- 每个 horizon 的 obstacle、boundary、pairwise clearance 和 TTC；
- visibility probability、observation age、message age；
- CBF correction magnitude、feasibility、最小 slack、最早失败 step；
- route identity、side、route length、目标接近进度和 safe-capture proxy。

**P3 通过条件：** validation 同时含有正负 feasibility/boundary/visibility 类别；每类 hard sample 数量和采样权重可审计；任何 calibration episode 不出现在 train/development。

### P4：重新训练 action-conditioned interaction-aware JEPA

模型头和损失按安全决策需要配置，不追求无关的大模型规模：

- [ ] 保留 action-conditioned target latent / relative-state prediction。
- [ ] 增加 obstacle clearance、boundary clearance、pairwise clearance/TTC head。
- [ ] 增加 target visibility、message/observation age、CBF feasibility/slack head。
- [ ] 增加 action execution / CBF correction head，预测请求动作经过动力学和安全过滤后的风险。
- [ ] 增加 route identity/side consistency 和 route progress auxiliary loss。
- [ ] 使用 hard replay，但限制负样本权重，防止模型退化为“所有动作都不可信”。
- [ ] 在 RTX 5050 上先做小规模 smoke，再做三 seed 正式 development train；保存训练时间、显存、checkpoint hash 和配置。

建议的训练顺序：

1. 先冻结 backbone，只训练 clearance/feasibility/visibility heads，检查标签和方向。
2. 再联合训练 action-conditioned JEPA 和辅助头。
3. 最后用 hard replay 做小学习率校正，不直接覆盖基础 checkpoint。

**P4 通过条件：** held-out validation 上各辅助头方向正确；CBF feasibility 的 calibration 可计算；预测不确定性在 OOD/noisy 条件上升高；route identity 不因候选排列变化而漂移。

### P5：重建 calibration archive 和 Reliability Ledger

每个新 actor/JEPA/候选协议/CBF contract 都必须重新校准，禁止沿用旧 Ledger。

- [ ] calibration archive 使用与 training/development 不重叠的 scene/episode seed。
- [ ] 分别校准目标误差、boundary clearance、pairwise/TTC、visibility 和 CBF feasibility。
- [ ] Ledger 状态至少区分 `trusted`、`cautious`、`abstain`、`safe_hold`、`controlled_abort`。
- [ ] 将 stale、OOD、non-finite、低候选分离度和 CBF 不可行分别计数。
- [ ] 对“所有候选均拒绝”的状态增加诊断：真实无解、候选覆盖不足、预测不确定性过高，不能统称为模型不可信。
- [ ] 记录 Ledger 对 safe_capture 的影响：拒绝率、safe-hold 比例、候选可验证率和误拒率。

**P5 通过条件：** Ledger hash 与 checkpoint/calibration/protocol 绑定；离线 settled counterfactual 中 score 方向一致；所有 gate/fallback 可重放；不会因为 Ledger 过度保守而把所有候选拒绝。

### P6：分层闭环验证，不直接跳到大矩阵

每一层都使用同一 manifest、actor、candidate protocol、CBF contract 和 Ledger；M0/M3 必须 paired。

| 层级 | 场景 | 开发目标 |
|---|---|---:|
| L0 clean/open | 0–1 障碍、nominal observation | `safe_capture >= 90%` |
| L1 nominal | 3 障碍、nominal observation、S-curve | `safe_capture >= 90%` |
| L2 robust | dropout/noise/delay | `safe_capture >= 85%` |
| L3/S3 stress | 3–5 混合障碍、S-curve、delayed-noisy | `safe_capture >= 80%` |

验证顺序：

1. 单元测试：SDF、route side、waypoint、projection、CBF、JEPA heads、Ledger hash。
2. 四个固定微场景：左绕/右绕、wall、窄通道、编队重构。
3. L0 M0 三 seed；未通过不得训练/扩大 L1–L3。
4. L0 M3 三 seed；若 M3 低于 M0，先做 trace attribution，不急于继续堆数据。
5. L1–L3 三 seed paired development matrix。
6. 通过后再扩大到 40/60 episode block；locked test 仍需另行批准。

建议对照：

- `M0`：nominal planner + CBF；
- `M3`：obstacle-conditioned candidates + JEPA + Ledger + rolling-horizon CBF；
- `A1`：同候选但不使用 JEPA；
- `A2`：同候选但不使用 Ledger（仅开发诊断，仍不得执行 raw-unverified）；
- `A3`：raw/no-CBF 仅作为危险行为诊断，不参与安全结论。

**通过条件：** 至少三 seed 中 2 个 seed 的 M3 paired delta 不为负；所有安全硬门通过；若安全为 0 但捕获率没有提升，结果标签必须是 `prediction_signal_no_control_gain`，不能写成方法有效。

### P7：失败归因和论文证据链

每个 episode 必须归入一个主要失败原因，允许多标签但要有 primary cause：

- `late_braking / boundary_feasibility`；
- `pairwise_formation_feasibility`；
- `obstacle_route_missing`；
- `jepa_wrong_route`；
- `ledger_over_abstention`；
- `stale_or_ood`；
- `target_visibility_loss`；
- `controlled_abort_no_verified_action`；
- `timeout`。

归档内容：checkpoint、dataset、calibration、Ledger、protocol、scene manifest、SHA-256、TensorBoard、episode CSV、step traces、failure index、deterministic replay。

论文只报告三类结论：

1. **安全结论：** 是否保持 0 collision/boundary/pairwise/raw-unverified。
2. **能力结论：** M3 相对 M0 的 paired safe-capture delta 和 seed 一致性。
3. **机制结论：** 候选覆盖、提前制动、JEPA 辅助头和 Ledger 路由分别减少了哪类失败。

## 4. 近期执行清单（按实际顺序）

### 本周：只做合同和诊断

- [ ] 冻结 `horizon=5` recovery protocol 和 manifest。
- [ ] 完成三路独立 CBF counterfactual 与 message-age 修复回归。
- [ ] 运行 L0 recovery M0 三 seed，比较 `strict_buffer` 与 `physical_feasibility`，分开归档。
- [ ] 输出每个 abort 的约束类型、提前量和可恢复性。

### 下一阶段：先增加“可验证候选”，再训练 JEPA

- [ ] 完成 12 类候选的微场景 coverage report。
- [ ] 生成含 hard negative 的 counterfactual archive。
- [ ] 检查每个候选都能先投影、再 CBF 验证；删掉只在标签中存在、runtime 不能执行的候选。

### 再下一阶段：JEPA + Ledger 新合同

- [ ] 训练多任务 JEPA checkpoint。
- [ ] 用新 checkpoint 生成 calibration archive。
- [ ] 生成 hash-bound Ledger，并做离线 settled ranking audit。
- [ ] 先跑 L0 M0/M3 三 seed，再决定是否进入 L1–L3。

### 最终开发验证

- [ ] 完成 L0–L3 三 seed paired matrix。
- [ ] 完成 horizon、barrier mode、候选集合、Ledger 的最小消融。
- [ ] 只有在安全硬门和 provenance 完整时，才扩大 episode 数量或申请 locked test。

## 5. 不能再做的事情

- [ ] 不再只增加随机数据量而不增加困难/负可行性样本。
- [ ] 不再用 nominal 周围固定小扰动冒充绕障规划。
- [ ] 不再在新 checkpoint 上复用旧 candidate/CBF/Ledger 合同。
- [ ] 不再把 CBF-only、A1 或物理可行性消融写成完整 M3 提升。
- [ ] 不再通过降低 margin、关闭 stale/OOD、删除 controlled-abort 或执行 raw action 来提高 `safe_capture`。
- [ ] 不再在 L0 基准未稳定前扩大到复杂场景和大量 seed。

## 6. 预期的最终交付物

1. 一份冻结的 protocol 和 provenance manifest。
2. 一套障碍物条件候选路线/编队动作生成器。
3. 一个带 clearance、visibility、CBF feasibility、route consistency 辅助头的 action-conditioned JEPA。
4. 与新 checkpoint 和 calibration 完全 hash-bound 的 Reliability Ledger。
5. horizon-aware Joint CBF 和三路独立 counterfactual 审计。
6. L0–L3 三 seed paired development report，主指标为 `safe_capture`，并完整报告安全和失败归因。
7. 只有在上述证据成立后，才讨论历史 V4 `75.3%` 与新流程的公平比较。

**下一步唯一动作：** 先完成 P0 执行合同和 L0 recovery 三 seed 验证，再开始新 JEPA 数据集和训练；不要反过来先训练再猜问题在哪里。
