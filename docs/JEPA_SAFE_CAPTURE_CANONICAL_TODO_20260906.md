# Obstacle-Conditioned JEPA Safe-Capture: Canonical TODO

**状态：** 唯一执行入口，development-only；WP1/WP2 几何路线与 runtime smoke 已完成，route-bound JEPA 重训仍未完成

**日期：** 2026-09-06

**硬件：** RTX 5050

**主指标：** episode-level `safe_capture`

**安全原则：** 任何阶段都不得执行未经 CBF 验证的动作

## 1. 最终目标

针对四架无人机围捕单个对抗目标，完成下面的闭环系统：

```text
可观测局部障碍地图 / SDF
        -> 障碍物条件候选路线生成
        -> reachable dynamics 投影
        -> action-conditioned interaction-aware JEPA 反事实评价
        -> reliability ledger 可信度路由
        -> horizon-aware Joint CBF 验证与过滤
        -> 只执行第一步
        -> 更新观测并滚动重规划
```

JEPA 是轨迹评价器，不直接输出最终控制动作；但候选路线生成器必须使用当前可观测的障碍物几何，能够提出“左绕、右绕、上绕、下绕、分裂、刹车”等真正不同的路线方案。

## 2. 当前基线与确定的问题

最新 extended prefilter development smoke：

| 变体 | safe capture | CBF controlled abort | collision / boundary / pairwise |
|---|---:|---:|---:|
| A1 JEPA + CBF + 12-candidate prefilter | `2/8 = 25.0%` | 5 | `0 / 0 / 0` |
| M0 nominal + CBF | `3/8 = 37.5%` | 5 | `0 / 0 / 0` |

已确认的问题不是“CBF 没有保护”，而是控制在进入不可行区域后才触发保护：

1. 观测编码包含最近障碍物的相对位置、半径、高度和形状，但候选生成器没有使用障碍物几何决定绕行侧。
2. JEPA ranker 主要使用 obstacle/inter-agent clearance，缺少 boundary clearance 和未来多步 CBF feasibility。
3. 训练 archive 中 `labels_boundary`、`labels_collision` 和训练集 `labels_cbf_qp_infeasible` 基本没有负样本。
4. runtime 的 `extended_v1` 为 12 个候选，而旧 checkpoint 的训练合同为 5 个候选。
5. prefilter 只验证当前第一步；在最终 abort 周期，全部候选已经不可行，无法恢复任务进度。
6. 最新 A1 smoke 的 `use_ledger=false`，不是完整 JEPA + Ledger + CBF 闭环。

因此，不能通过继续增加随机扰动、降低 CBF margin 或删除 `controlled_abort` 来修复结果。

## 3. 不可改变的安全和信息边界

- `development_only=true`。
- `locked_test_opened=false`，新模型未通过开发门前不得打开 locked split。
- 不降低已有 CBF margin、速度约束、加速度约束或 pairwise 间隔。
- 不关闭 stale、OOD、non-finite 和 Ledger 信用门。
- 不执行 `raw-unverified` action。
- 保留 `controlled_abort`；它是安全结果，不是需要隐藏的失败。
- 在线只使用当前传感器/通信可得到的障碍物信息，不使用 simulator hidden ground truth。
- 目标真实未来轨迹只允许作为离线 label，不能泄漏到在线候选生成或排序。
- `safe_capture` 是唯一主指标；平均捕获时间仅作诊断。

## 4. WP0：清理入口并冻结新协议

### 任务

- [ ] 将本文件作为唯一执行入口。
- [ ] 新建一个 protocol，冻结 `candidate_profile`、候选顺序、chunk 长度、重规划频率、动作尺度和观测信息边界。
- [ ] 明确旧 5-candidate checkpoint 不能用于新的 12-candidate 运行。
- [ ] 为新协议生成唯一 SHA-256 manifest，绑定 environment、candidate generator、model、calibration 和 Ledger。
- [ ] 检查每个 run 的 provenance 是否记录实际 git revision；禁止使用旧 revision 的结果冒充新实现。

### 退出门

- [ ] protocol verifier 通过。
- [ ] candidate count、shape、动作尺度和 reachable projection 在 train/validation/calibration/development 完全一致。
- [ ] 没有打开 locked split。

## 5. WP1：建立可用于绕障的局部环境表示

### 已完成证据（2026-09-06）

- [x] 新增统一的 cylinder/box/wall signed-distance 和 route-corridor clearance 接口：`src/encirclement3d/obstacle_route_candidates.py`。
- [x] 四个确定性微场景路线审计通过，包含真实的只读 Joint CBF first-step probe：见 [WP1 obstacle-route audit](JEPA_SAFE_CAPTURE_WP1_OBSTACLE_ROUTE_AUDIT_20260906.md)。
- [x] 审计结果为 `development_only=true`、`locked_test_opened=false`；没有改变 CBF margin 或执行未经验证动作。

以下任务仍属于后续 runtime/观测合同工作，不能因为 WP1 几何审计通过而标记完成。

### 目标

让每架 UAV 能够从可观测信息中回答：障碍物在哪里、从哪一侧绕行、绕行后是否仍有目标可见性和编队空间。

### 任务

- [ ] 保留当前最近障碍物的相对几何，增加明确的 signed-distance / clearance-to-boundary 特征。
- [ ] 对每个障碍物计算 UAV 相关的局部坐标：前后、左右、上下、最近表面距离、预计穿越方向。
- [ ] 建立 obstacle-to-target corridor 特征：障碍物是否位于当前追击走廊、左右侧可用空间、上下侧可用空间。
- [x] 对 box、cylinder、wall 使用统一的局部 SDF 接口，不把形状判断散落在候选生成器中。
- [x] 对多障碍场景保留最近障碍物之外的阻塞关系，至少能识别窄通道和不可通行侧。
- [ ] 明确仿真中的障碍物观测是否代表真实可见信息；若引入障碍物 dropout/noise，单独记录其信息边界。

### 退出门

- [x] 单元测试覆盖 cylinder、box、wall 和单缺口窄通道；无障碍和完整局部地图信息仍需补充 runtime 回归。
- [x] 路线动作块在不同障碍物排序下保持确定性；旋转/平移不变性仍需加入观测特征回归。
- [x] 不使用 target ground truth 生成在线障碍特征。

## 6. WP2：障碍物条件候选路线生成

### 目标

候选不是围绕 nominal action 的无语义小扰动，而是不同的可解释路线方案。

### 候选集合

固定顺序并写入 protocol：

1. `nominal`：原始 actor 方案。
2. `left_detour`：绕当前主障碍左侧。
3. `right_detour`：绕当前主障碍右侧。
4. `upper_detour`：三维空间上方绕行。
5. `lower_detour`：三维空间下方绕行。
6. `radial_out`：先远离障碍物/拥挤区域。
7. `formation_split`：临时分裂后从两侧包抄。
8. `formation_contract`：收缩编队通过窄通道。
9. `braking`：减速并保持可行域。
10. `safe_intercept`：在可见性和捕获进度之间折中。
11. `visibility_hold`：保持目标可见并等待更稳定观测。
12. `verified_safe_hold`：只能作为 fallback，不能主动竞争。

### 任务

- [x] 候选生成器直接读取 `observation["obstacles"]` 或等价局部地图接口。
- [x] 对每个障碍物生成左/右/上/下 waypoint 或轨迹 corridor，而不是只给速度向量加固定扰动。
- [x] 候选包含 3 步 action chunk；逐周期更新目标 belief 的 runtime 接入仍待完成。
- [x] 每条候选先经过 speed、acceleration、slew 和 reachable dynamics projection。
- [x] 为每条候选记录 route identity、绕行障碍物 id、绕行侧、预计长度和几何可行性原因。
- [x] 当局部地图无法决定路线侧时，保留多个路线候选交给 JEPA 排序，不强行猜测。
- [x] 候选生成阶段不得调用最终 CBF 结果来制造标签泄漏。

### 退出门

- [x] 在固定障碍物场景中，left/right/upper/lower 候选的几何路径确实不同。
- [x] 所有候选都能通过相同的 reachable projection 和 schema 验证；几何阻塞候选会显式拒绝。
- [x] 形成 candidate coverage 报告：四个微场景覆盖 cylinder、wall、左右阻塞和单缺口；box 已由单元测试覆盖。

### WP1/WP2 完成边界

路线层已通过离线确定性审计，并已接入 JEPA ranker、reliability ledger、独立 CBF counterfactual 和滚动重规划 evaluator。runtime smoke 证据见 [WP2 route runtime smoke](JEPA_SAFE_CAPTURE_WP2_ROUTE_RUNTIME_SMOKE_20260906.md)。该 smoke 仍是开发验证，不能写成 safe-capture 提升；下一阶段必须先按失败 trace 建立 route-identity counterfactual archive 和新的 route-bound calibration ledger。

## 7. WP3：重新定义 JEPA 评价目标并生成数据

### 数据分层

建立互不重叠的 train、validation、calibration、development split；episode seed 和 scene hash 必须 disjoint。

### 每个候选的离线标签

- 未来目标相对位置、速度和加速度；
- 每个 horizon 的 obstacle clearance；
- 每个 horizon 的 boundary clearance；
- 每个 horizon 的 pairwise clearance 和 TTC；
- 目标可见性、观测年龄和通信年龄；
- CBF correction magnitude；
- CBF 是否可行、最小 constraint slack、最早失败 step；
- 候选路线长度、目标接近进度和局部 safe-capture proxy；
- obstacle side / route identity 一致性标签。

### hard counterfactual 采样

- [ ] 从 earliest-abort、近边界、高 correction、pairwise 临界、窄通道和遮挡片段采样。
- [ ] 保留足够的负 feasibility、负 boundary-clearance 和不良路线样本。
- [ ] 对可行/不可行候选做分层采样，避免全是“安全且简单”的正样本。
- [ ] 训练 hard replay 时单独报告各类困难样本数量和权重上限。
- [ ] 训练、calibration 和 development 不混用同一 episode seed。

### 模型头

- [ ] 保留 action-conditioned target latent/relative prediction。
- [ ] 新增 boundary-clearance head。
- [ ] 新增 horizon-level CBF-feasibility/slack head。
- [ ] 保留 obstacle、pairwise、visibility、TTC、correction 和 uncertainty heads。
- [ ] 增加 route progress / route identity consistency 辅助目标。

### 退出门

- [ ] validation 上 boundary 和 CBF feasibility 至少存在正负类别。
- [ ] feasibility、boundary 和 visibility calibration 可计算，不能因为单一类别而只报告 Brier 无法报告 discrimination。
- [ ] 新 checkpoint 的 model hash、dataset hash、训练 seed 和候选协议全部写入 metadata。

## 8. WP4：Horizon-aware JEPA ranker + Reliability Ledger

### 排序顺序

候选排序必须遵守：

1. 先剔除 non-finite、不可达、信息过期、OOD 或 Ledger 信用不足的候选；
2. 再比较未来 horizon 的最小 boundary/obstacle/pairwise slack；
3. 再比较目标接近进度、可见性和路线代价；
4. 最后使用 deterministic tie-break 和 nominal anchor。

### 任务

- [ ] Ranker 直接使用 predicted boundary clearance 和 horizon CBF feasibility。
- [ ] 将 route identity 和 obstacle side 纳入 trace，能回答选了哪条绕行路线。
- [ ] 对每个候选记录最早风险 step，而不只记录 horizon 末端值。
- [ ] Ledger 与新 checkpoint、calibration archive、candidate protocol 做 hash binding。
- [ ] 重建 calibration archive；禁止沿用 5-candidate checkpoint 的旧 Ledger。
- [ ] 在高不确定性或路线不可辨识时回退 nominal/safe-hold，并记录原因。

### 退出门

- [ ] 离线 settled counterfactual 中，score 与未来安全/捕获 proxy 的方向一致。
- [ ] 选择的路线不再系统性地劣于同一状态下的可行候选。
- [ ] OOD/stale/non-finite gate、fallback 和 tie-break 回归测试全部通过。

## 9. WP5：CBF 执行合同

- [x] 在当前 V21 development rerun 中对 selected candidate、nominal、safe-hold 分别进行独立 CBF counterfactual；107 个 abort 的三路结果见 [V21 independent CBF audit](JEPA_SAFE_CAPTURE_V21_INDEPENDENT_CBF_AUDIT_20260906.md)。
- [ ] prefilter 使用固定 horizon 的多步 rollout，报告最小 slack 和 earliest failure。
- [ ] 最终执行边界仍只有 Joint CBF；任何 JEPA score 不能替代 CBF。
- [ ] 每个周期只执行第一步，执行后重新读取观测并重规划。
- [ ] CBF infeasible 时保留 `controlled_abort`，不得偷偷执行 raw action。
- [ ] 记录 requested action、CBF corrected action、executed action、slack 和 fallback reason。

### 退出门

- [ ] 所有安全 counterfactual 和在线执行路径均 finite。
- [ ] CBF abort 不产生 collision、boundary 或 pairwise safety violation。
- [ ] abort trace 能区分 obstacle、boundary、pairwise、acceleration 和 solver failure。

## 10. WP6：集成测试与验证阶梯

### 测试顺序

1. **离线单元测试**：SDF、route side、waypoint、projection、JEPA head、Ledger hash。
2. **单场景闭环**：无障碍、单障碍左绕/右绕、wall、窄通道。
3. **L0**：0-1 个障碍、nominal observation。
4. **L1**：3 个障碍、nominal observation、S-curve。
5. **L2**：3 个障碍、dropout/noise/delay。
6. **L3/S3**：3-5 个混合障碍、S-curve、delayed-noisy。

### 变体

- M0：nominal planner + CBF；
- M3：obstacle-conditioned candidates + JEPA + Ledger + horizon CBF；
- A1：同候选、去掉 JEPA；
- A2：同候选、去掉 Ledger；
- A3：同候选、去掉 route-aware auxiliary heads。

所有变体必须使用同一 scene manifest、episode seed 顺序和 actor checkpoint。

## 11. 验收指标

以下是开发目标，不是已经获得的结果：

| 难度 | 目标 safe capture |
|---|---:|
| L0/L1 | `>= 90%` |
| L2 | `>= 85%` |
| L3/S3 | `>= 80%` |

硬安全门：

- collision = 0；
- defender boundary violation = 0；
- pairwise violation = 0；
- raw-unverified execution = 0；
- 所有 run 的 provenance、scene hash、protocol hash 完整。

比较门：

- M3 必须与 M0 做 paired comparison，不能只报告独立比例；
- 至少三 seed 中 2 个 seed 的 safe-capture delta 不为负，才可进入更大 episode block；
- 若安全通过但捕获率不增，结果标签为 `prediction_signal_no_control_gain`，不能写成方法提升。

## 12. WP7：结果归档和论文证据

- [ ] 每个 seed 保存 checkpoint、calibration archive、Ledger、protocol、scene manifest 和 SHA-256 manifest。
- [ ] TensorBoard 至少记录 train/validation loss、boundary/feasibility calibration、candidate route counts、Ledger states、CBF abort、safe capture 和 latency。
- [ ] 生成 episode-level CSV、step trace、failure index 和 deterministic replay。
- [ ] 单独报告“安全成功但未捕获”“候选路线被 CBF 拒绝”“JEPA 选错路线”“目标不可见”四类失败。
- [ ] 只把通过 paired gates 的结果升级为 promising development evidence；不把 smoke 写成正式 locked 结论。

## 13. 明确停止事项

- [ ] 不再新增没有依赖关系的 TODO 版本号。
- [ ] 不再用随机小扰动代替绕障路线候选。
- [ ] 不再用只含正样本的 archive 训练安全 feasibility head。
- [ ] 不再把 CBF-only、A1 消融和完整 M3 混称为完整闭环。
- [ ] 不在模型和 Ledger 未重新绑定前扩大到 40/60 episodes。
- [ ] 不为提高 `safe_capture` 而放宽安全约束。

## 14. 推荐执行顺序

```text
WP0 protocol freeze
  -> WP1 local obstacle/SDF representation
  -> WP2 obstacle-conditioned route candidates
  -> WP3 negative counterfactual archive + JEPA heads
  -> WP4 ranker + fresh Ledger
  -> WP5 horizon-aware CBF execution
  -> WP6 3-seed L0-L3 paired smoke
  -> WP7 40/60 development block and reporting
```

当前最近的三个动作只有：

1. [x] 完成 WP1/WP2 几何审计，证明候选路线真的会根据障碍物改变左右/上下绕行方向；
2. [ ] 将 `obstacle_route_v1` 接入 development evaluator，并先完成 selected/nominal/safe-hold 独立 CBF counterfactual 合同；
3. [ ] 生成包含 boundary/feasibility 负样本的新 archive，在新协议、新 checkpoint、新 Ledger 全部绑定后再跑三 seed paired smoke。

在这三步完成前，不继续堆叠更多模型、更多随机场景或更多计划文件。
