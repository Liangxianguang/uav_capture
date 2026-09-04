# 无人机集群对抗围捕：安全增强系统下一阶段 TODO 计划书

**版本：** 1.0  
**日期：** 2026-09-05  
**执行目录：** `D:\\uav-capture\\uav_capture`  
**硬件：** NVIDIA GeForce RTX 5050，CUDA 12.8，Conda `uav-encirclement-gpu`  
**系统路线：** Interaction-aware Action-conditioned JEPA + Reliability Ledger + Joint CBF-QP + Rolling Horizon  
**实验边界：** `development_only=true`，`locked_test_opened=false`  
**唯一主指标：** `safe_capture`；`mean_capture_time` 只作诊断，不抵消安全失败。

> 这份文件是下一阶段的执行计划，不是新的实验结果。任何模型、阈值、候选动作合同或 ledger 规则的改变，都必须生成新的 protocol、checkpoint、ledger、scene manifest、TensorBoard 目录和 SHA-256 记录。

## 1. 目标和最终验收标准

### 1.1 系统目标

实现一个面向无人机集群对抗围捕的安全闭环：世界模型只负责对传统规划器给出的候选轨迹进行隐空间反事实评价，不直接生成或执行控制动作；最终动作必须通过同一个 Joint CBF-QP，并采用滚动时域只执行第一个控制步。

```text
多机观测/通信历史
  -> interaction-aware belief state
  -> 传统规划器生成动力学可行 action chunks
  -> action-conditioned JEPA 反事实预测
  -> immutable reliability ledger 路由/拒答
  -> safety-first candidate ranking + nominal anchor
  -> Joint CBF-QP 过滤
  -> 执行第一步
  -> 重新观测、更新 belief、重新规划
```

### 1.2 必须同时满足的最终条件

- `safe_capture` 在冻结 paired development block 中相对于 M0 至少达到预先定义的非劣门；若要宣称正向提升，必须平均 paired delta > 0 且至少 2/3 seed 非负。
- 所有变体的 obstacle、target、defender-defender collision，boundary/altitude 和 pairwise separation violation 均为 0。
- `raw_unverified_executed_steps=0`；CBF timeout、infeasible、controlled abort 都必须单独统计，不能隐藏在捕获率中。
- OOD、stale observation、non-finite prediction、unknown horizon、provenance/hash mismatch 均只能进入 `safe_hold -> nominal CBF -> controlled_abort` 路径。
- 每个周期重新规划，action chunk 只执行第一步；重复 replay 的候选决策、安全结算和 rejection reason 必须一致。
- 所有结果可由代码 revision、Conda/GPU 环境、protocol、checkpoint、ledger、calibration archive 和 scene manifest 的 hash 完整追溯。

## 2. 当前证据快照（2026-09-05）

### 2.1 P5 安全执行边界

- Joint CBF-QP fault matrix：9 类 case，每类 20 次；输出 finite、失败不执行 raw request、zero-perturbation exact、重复确定性全部通过。
- 状态 violation 为 0，solver p95 为 `0.708 ms`，`raw_unverified_executed=0`。
- RTX 5050 M3 rolling replay：20 episodes、1,062 cycles，cycle p95 `15.416 ms`，低于 `100 ms` contract。
- CPU/CUDA replay：20/20 episode safety settlement、candidate decision、CBF verification count 一致。

结论：CBF、rolling horizon、设备一致性和 provenance 边界可进入下一阶段；这些结果本身不证明任务收益。

### 2.2 P6 corrected-frame smoke（每 seed 20 集）

| 变体 | seed 20260911 | seed 20260912 | seed 20260913 | 三 seed合计 |
|---|---:|---:|---:|---:|
| M0：nominal + CBF | 10/20 (50%) | 10/20 (50%) | 10/20 (50%) | 30/60 (50.0%) |
| M3：JEPA + ledger + auxiliary rank + CBF | 9/20 (45%) | 10/20 (50%) | 9/20 (45%) | 28/60 (46.7%) |
| A1：JEPA + auxiliary rank，无 ledger + CBF | 9/20 (45%) | 12/20 (60%) | 10/20 (50%) | 31/60 (51.7%) |
| A2：JEPA + ledger，无 clearance/visibility rank + CBF | 9/20 (45%) | 11/20 (55%) | 10/20 (50%) | 30/60 (50.0%) |

当前 M3 相对 M0 的 paired delta 为 `-5 pp / 0 pp / -5 pp`，所以本阶段只能登记为 `no_control_gain`，不能进入 40 集 validation，也不能写成 JEPA 已经提升 `safe_capture`。四个变体的 collision、boundary、pairwise 和 raw-unverified 均为 0；这说明当前瓶颈是任务推进/排序/拒答策略，而不是已观测到的安全越界。

## 3. 不可变安全与数据合同

### 3.1 执行边界

- JEPA 输出预测、uncertainty 和排序特征，禁止输出最终执行动作。
- Reliability ledger 只读、checkpoint/protocol/calibration hash 绑定，禁止在线更新 credit 或阈值。
- Ranker 只能在通过 finite/reachability/safety lower-bound 的候选中选择；nominal candidate 始终保留为 anchor。
- Joint CBF-QP 是唯一执行入口。`verified=true` 且 action finite 才能进入 executor。
- 任一失败的固定回退顺序：`separation-preserving safe-hold` -> `verified nominal through CBF` -> `controlled_abort`。
- 线上不得读取 target ground truth；ground truth 只能用于离线 settled label、episode 结算和审计。

### 3.2 固定候选合同

- 候选数 `K=5`：`nominal`、`intercept`、`lateral_clearance`、`formation_clearance`、`visibility_hold`。
- action chunk 长度为 3 control steps，线上只执行第 1 步。
- 固定 `score_tie_tolerance=5e-4 m`、`top_two_abstention_margin=1.5e-3 m`、`minimum_hold_steps=2`；修改即建立新 protocol。
- 每个候选进入 JEPA 前必须记录 shape、finite、speed、acceleration、slew、reachability 和拒绝原因。

### 3.3 `safe_capture` 结算

episode 只有同时满足以下条件才计为成功：在 time limit 内进入 capture radius `0.80 m`，且无任何碰撞、边界/高度、pairwise violation、CBF infeasible/timeout/unverified action 或 controlled abort。

## 4. 阶段总览和依赖

```text
P0 证据冻结与 P6 smoke 收口
  -> P1 失败归因与可解释性审计
  -> P2 训练/校准改进分支（只用 train/calibration）
  -> P3 candidate ranking + abstention 工程修复
  -> P4 三 seed 20 集 smoke 复验
  -> P5 三 seed 40/60 集 paired development
  -> P6 robustness stress + SIL/HIL readiness
  -> P7 locked-test preregistration 或负结果归档
```

依赖规则：P1 未完成不得调权重；P2/P3 的每个候选 revision 都要先通过离线 gate；P4 未通过不得扩大到 40/60 集；P5 未形成稳定安全与任务收益证据不得打开 locked test。

---

## 5. P0：冻结环境、完成 P6 smoke 证据闭环

**目标：** 把当前 smoke 变成不可篡改的基线，并确认所有后续比较使用同一场景和同一安全合同。

### TODO

- [ ] 检查工作树，保留现有 E1/V5/tmp 改动；不使用 `git add .`、`git reset`、`git checkout` 或删除 `tmp/`。
- [ ] 保存当前代码 revision、Conda 包清单、Python/PyTorch/CUDA/GPU 信息。
- [ ] 为 M0/M3/A1/A2 每个 seed 记录 `summary.json`、`episodes.csv`、`provenance.json`、`scene_manifest.jsonl`、TensorBoard event 和所有输入 hash。
- [ ] 运行 P6 三 seed aggregate，输出逐 episode pairing、improved/degraded/tied、paired delta、bootstrap 95% CI 和 exact McNemar。
- [ ] 运行 settled-counterfactual、ledger alignment、temporal ledger、CBF/fallback、latency、provenance 和 TensorBoard 完整性审计。
- [ ] 生成 `P6 corrected-frame smoke report`，明确分类为 `no_control_gain`，除非审计发现数据缺失或安全异常导致 `INSUFFICIENT_EVIDENCE`。
- [ ] 只提交本阶段新增报告/测试；push 失败时保留本地 commit 并记录网络错误。

### P0 出口门

- [ ] 四变体 x 三 seed 矩阵完整，episode pairing 和 canonical manifest hash 一致。
- [ ] 所有安全硬门通过：collision/boundary/pairwise/raw-unverified 均为 0，所有执行 action finite。
- [ ] 每个 control cycle 都有 candidate、ledger、ranker、CBF 和执行 provenance 字段。
- [ ] 若任何产物缺失、hash 冲突或 locked 字段异常，停止后续实验并标记 `INSUFFICIENT_EVIDENCE`。

## 6. P1：失败归因，不改模型和阈值

**目标：** 把 M3 相对 M0 的损失拆成可验证的机制，区分排序错误、ledger 拒答、CBF abort 和 target-motion mismatch。

### TODO

- [ ] 对所有 M3 非 safe-capture episode 建立 failure index：episode seed、场景障碍/运动模式、观测条件、终止原因、首次 abstain、首次 CBF fallback/abort、最小净空、target motion regime。
- [ ] 重新计算每个 control cycle 的 `selected_not_best`、top-1 safety precision、top-two margin、score separation、candidate reachability 和 selected candidate id。
- [ ] 对齐 settled counterfactual：比较 JEPA 预测排序与离线真实未来 safe outcome，按 `trusted/fallback_nominal/safe_hold` 分桶。
- [ ] 分别统计 ledger route 与 safe-capture 失败的条件概率，重点检查 `fallback_nominal` 是否过多、`safe_hold` 是否过早、high-credit 是否仍出现集中失败。
- [ ] 统计 CBF correction norm、infeasible/timeout/controlled-abort 与候选类型、预测 clearance、TTC、visibility 的关联。
- [ ] 按 target 急转、速度突变、S-curve、flee persistence、遮挡/延迟、3/5 obstacles、拥挤队形建立 failure heatmap。
- [ ] 对同一 episode 做 CPU/CUDA 和二次 replay；任何决策不一致先修复确定性，再讨论模型。

### P1 产物

`failure_index.csv`、`failure_index.json`、`ranking_mismatch_report.md`、`abstention_report.md`、`cbf_abort_correlation.md` 和一个可重放的失败 episode 清单。

### P1 出口门

- [ ] 每个失败 episode 有且只有一个主分类，允许附加次分类。
- [ ] 至少 95% 的失败 control cycles 能归入 `ranking_mismatch`、`ledger_abstention`、`cbf_abort`、`candidate_unreachable` 或 `target_prediction_drift`。
- [ ] P1 期间不修改 checkpoint、ledger、阈值和 validation manifest。

## 7. P2：训练和校准改进分支（只用 train/calibration）

**目标：** 增强 JEPA 的安全预测信号，优先提升 safe-capture 的可判别性，而不是追求更低的 capture time。

### 7.1 数据与标签

- [ ] 从 train split 新建 counterfactual archive；禁止把 P6 validation episode 直接回灌训练。
- [ ] 为每个 belief/action chunk 生成多 horizon settled labels：目标相对位移、净空下界、障碍净空、pairwise 净空、visibility、TTC、CBF intervention/abort 风险、target motion regime。
- [ ] 对 hard replay 片段加权：急转、加速度突变、遮挡、通信延迟、候选 separation 消失、CBF correction 变大和预测残差连续上升。
- [ ] 保留 normal/hard/OOD 三类独立标识；OOD 只用于拒答和校准审计，不用于伪造正常训练样本。

### 7.2 模型 revision

- [ ] 保持 interaction-aware history encoder 和 action-conditioned latent transition 不变，先新增 prediction heads：`net_clearance`、`visibility`、`TTC`、`CBF_intervention`。
- [ ] 对每个 horizon 输出 uncertainty/quantile 或 conformal lower bound；ranker 使用安全下界，不使用未经校准的均值。
- [ ] 加入 action sensitivity regularizer：同一 belief 下不同 action chunk 必须产生可测的 latent/prediction separation。
- [ ] 对 hard replay 使用固定权重上限，避免模型只学会保守 safe-hold。
- [ ] 每个 revision 独立保存训练 config、dataset hash、checkpoint hash、TensorBoard 和训练审计。

### 7.3 离线 gate

- [ ] 所有输出 finite，训练可复现，TensorBoard loss/histogram/provenance 完整。
- [ ] action-following separation > 0，且不出现全候选 action-insensitive。
- [ ] clearance/TTC/visibility/CBF-intervention 的 calibration、Brier/ECE、hard-slice recall 均优于当前 checkpoint 或达到预设 non-regression 区间。
- [ ] 安全风险的 lower-bound 漏报率优先于平均 MAE；target displacement MAE 变差不能被忽略。
- [ ] 离线 gate 未通过的 revision 不进入闭环 smoke。

建议优先顺序：先做多任务安全头和 hard replay，再做结构性更换；在当前证据下不引入新的大模型，不改变 JEPA 为“评价器而非控制器”的定位。

## 8. P3：Safety-first candidate ranking、abstention 和 ledger 修复

**目标：** 将可用预测信号转化为控制收益，同时保持安全边界不变。

### TODO

- [ ] 冻结一个新的 ranking protocol revision，明确 score 组成、各项单位、权重、tie tolerance、top-two abstention、hysteresis 和 minimum hold。
- [ ] 使用安全优先词典序：先排除预测净空/TTC/visibility lower-bound 不满足的候选，再比较任务进展；禁止用高捕获概率抵消安全风险。
- [ ] 保留 nominal anchor；JEPA 只在 margin 足够且 ledger=`trusted` 时改变候选，否则走 nominal/hold。
- [ ] 对 `fallback_nominal` 与 `safe_hold` 分别记录“触发原因”和“最终是否经 CBF 验证”，不得合并为单一失败类。
- [ ] 增加短期滞回和最小保持，防止候选在接近分数下抖动；所有滞回行为纳入 deterministic replay。
- [ ] 增加候选 action chunk 的多样性和可达性统计；若所有候选在同一 belief 下近似等价，记录 `insufficient_candidate_separation` 并使用 nominal。
- [ ] 对 ranker-only、ledger-only、auxiliary-head 三条消融分支分别编号，禁止在同一 validation block 反复调参。

### P3 出口门

- [ ] settled `selected_not_best`、top-1 safety precision 和 high-credit failure rate 相对当前基线改善，或至少不恶化超过预设容差。
- [ ] abstention 不得通过执行 raw action 获得捕获率；`raw_unverified_executed=0` 保持不变。
- [ ] 100-cycle non-zero replay 的候选决策、CBF verification 和 safety settlement 逐字段一致。

## 9. P4：新 revision 三 seed 20 集 smoke

**目标：** 在扩大样本前验证新模型/ledger/ranker 没有安全和接口回归。

### 固定矩阵

| 变体 | 作用 |
|---|---|
| M0 | frozen nominal + CBF baseline |
| M3-new | JEPA + ledger + 新安全排序 + CBF |
| A1-new | 去掉 ledger，诊断 credit 路由作用 |
| A2-new | 去掉安全辅助排序，诊断多任务 head 作用 |

### TODO

- [ ] 用新 protocol、三 seed 对应 checkpoint 和 hash-bound ledger 生成新的 paired scene manifest。
- [ ] 每个 seed 先运行 M0，再把同一 manifest 复用于 M3-new/A1-new/A2-new。
- [ ] 每个变体运行 20 episodes，独立 output/TensorBoard root，禁止覆盖旧 smoke。
- [ ] 运行完整安全、latency、provenance、ledger、settled ranking 和 deterministic replay 审计。

### P4 硬门

- [ ] 安全硬门全部通过：collision/boundary/pairwise/raw-unverified/CBF timeout 均为 0。
- [ ] 三 seed 的 M3-new 相对 M0 不出现明显系统性退化；若任一 seed 出现 safety regression，立即停止扩大规模。
- [ ] 若任务收益不满足非劣，但安全和证据完整，分类为 `prediction_signal_no_control_gain`，回到 P1/P3，不准通过重复采样掩盖问题。

## 10. P5：三 seed 40/60 集 paired development

**进入条件：** P4 全部硬门通过，且新 revision 的失败机制已有可解释改善。

### TODO

- [ ] 预先冻结每个 seed 的 40 集 smoke-to-development block；若资源允许扩展到 60 集，扩展前不得查看新增结果后改协议。
- [ ] 对 M0/M3-new/A1-new/A2-new 做逐 episode 配对，保存全量 episodes、step traces、scene manifest 和 hashes。
- [ ] 统计每 seed 和 aggregate 的 safe-capture rate、95% CI、paired delta、bootstrap CI、exact McNemar、improved/degraded/tied。
- [ ] 单独报告 collision/boundary/pairwise、CBF fallback/abort/timeout、raw-unverified、transit、minimum clearance 和 queue age。
- [ ] 把 mean capture time 放在诊断表中，不得作为安全失败的补偿项。

### P5 决策规则

| 分类 | 条件 | 后续 |
|---|---|---|
| `promising_development_candidate` | safe-capture 平均 paired delta > 0、至少 2/3 seed 非负，安全硬门全通过，证据完整 | 进入 P6 robustness 和新的 locked preregistration 草案 |
| `safe_non_inferior` | 平均 paired delta >= 0，安全硬门全通过，但正向统计证据不足 | 可做安全部署研究，先补 robustness，不宣称提升 |
| `prediction_signal_no_control_gain` | 离线预测/排序改善，闭环没有净收益 | 保留负结果，继续机制修复或停止该分支 |
| `no_control_gain` | JEPA 与 M0 无稳定净收益 | 不打开 locked，转为 baseline/消融证据 |
| `BLOCKED_BY_SAFETY` | 任一碰撞、边界、pairwise、raw action 或不可解释 CBF 失败 | 立即停止扩大实验，修复安全链路 |

## 11. P6：robustness、SIL/HIL 和部署就绪

**目标：** 证明安全增强系统在分布变化和设备约束下仍保持可拒答、可回退、可追溯。

### TODO

- [ ] 观测 dropout/noise、通信 delay/dropout、target 急转/突变、障碍密度、初始侧距和队形拥挤做分层 stress matrix。
- [ ] 做 OOD/stale/non-finite/provenance mismatch fault injection；确认 100% 进入 safe-hold 或 nominal CBF，不执行 raw。
- [ ] 在 RTX 5050 上重复测量 JEPA、ledger、ranker、CBF、cycle total p50/p95/p99 和 queue age。
- [ ] 运行长序列 100/500/1000 control-cycle deterministic replay，审计 rollout 漂移、候选抖动、CBF correction 累积和 trace 完整性。
- [ ] 完成 SIL 接口合同：传感器时间戳、通信年龄、动作执行回执、CBF solver 状态、controlled abort 事件。
- [ ] 若进入 HIL，先用仿真/回放数据验证“只执行 verified action”不变，再引入真实飞控；HIL 不得绕过 CBF。
- [ ] 写部署故障手册：GPU不可用、JEPA超时、ledger损坏、solver不可行、通信中断、传感器陈旧时的固定动作。

### P6 出口门

- [ ] 所有 stress case 的安全硬门和 fallback 语义通过。
- [ ] cycle p95 低于 100 ms contract，且不存在长块 open-loop 执行。
- [ ] SIL/HIL trace 与仿真 trace schema 一致，能重建每次动作的 provenance。

## 12. P7：论文证据、版本发布和 locked-test 决策

### 仅在 P5/P6 通过后执行

- [ ] 生成系统架构图、数据流图、CBF 约束图和 reliability ledger 状态机图。
- [ ] 发布代码 revision、环境文件、protocol、checkpoint、ledger、calibration archive、scene manifest 和 SHA-256 manifest。
- [ ] 报告主结果时只使用 safe-capture；将 mean capture time、transit、clearance、latency 作为次级诊断。
- [ ] 同时公开正向、负向和不确定结果，明确区分 development 与 locked evidence。
- [ ] 只有分类为 `promising_development_candidate` 且 robustness/SIL gate 通过，才新建 locked-test preregistration；不得把旧 V4/V5 locked 结果改写成新模型结果。
- [ ] locked-test preregistration 一旦冻结，不得根据 locked 结果反向调参；若未达到条件，正式归档为安全保持或无控制增益。

## 13. 每次实验的统一执行清单

### 运行前

- [ ] 确认 `development_only=true`、`locked_test_opened=false`、split 不是 locked。
- [ ] 确认 output 和 TensorBoard 目录为空或由工具拒绝覆盖。
- [ ] 保存 protocol/checkpoint/ledger/calibration/manifest SHA-256。
- [ ] 记录 RTX 5050、CUDA、PyTorch、Conda 环境和 Git revision。
- [ ] 运行 targeted tests、`git diff --check`、protocol schema test。

### 运行中

- [ ] 只执行 Joint CBF-QP 返回的 finite、verified action 第一控制步。
- [ ] 记录 candidate rejection、ledger state/reason、ranker selection、CBF status、fallback 和 queue age。
- [ ] 不在线更新 ledger，不读取 target ground truth，不删除失败 step。

### 运行后

- [ ] 检查 summary/episodes/traces/provenance/manifest/TensorBoard 是否齐全。
- [ ] 检查 collision/boundary/pairwise/raw-unverified/timeout/infeasible/abort 分开计数。
- [ ] 做 paired aggregate、bootstrap/McNemar、failure index 和 deterministic replay。
- [ ] 生成独立 Markdown/JSON 报告，写入输入 hash、命令、环境和限制。
- [ ] 只提交本阶段相关文件，不把 README、E1、V5 或 `tmp/` 用户改动混入提交。

## 14. 建议的近期执行顺序

1. 完成 P0：P6 corrected-frame smoke aggregate 和全套审计。
2. 完成 P1：失败索引、排序失配、abstention、CBF abort 和 target drift 归因。
3. 仅在 train/calibration split 上完成 P2 多任务安全头 + hard replay 的离线 gate。
4. 冻结 P3 新 ranking/ledger revision，先做 replay 和小规模 smoke。
5. P4 三 seed 20 集复验；只有非劣和安全硬门通过才进入 P5。
6. P5 完成三 seed 40/60 集 paired development，再决定是 `promising_development_candidate`、`safe_non_inferior` 还是负结果。
7. P6 做 robustness/SIL/HIL；最后才讨论新的 locked-test preregistration。

## 15. 预期最终交付物

- `docs/JEPA_SAFE_CAPTURE_P6_CORRECTED_FRAME_SMOKE_REPORT_*.md`
- `docs/JEPA_SAFE_CAPTURE_FAILURE_ATTRIBUTION_*.md`
- `docs/JEPA_SAFE_CAPTURE_AUXILIARY_HEAD_REPLAY_*.md`
- `docs/JEPA_SAFE_CAPTURE_RANKING_LEDGER_REVISION_*.md`
- `docs/JEPA_SAFE_CAPTURE_THREE_SEED_DEVELOPMENT_*.md`
- `docs/JEPA_SAFE_CAPTURE_ROBUSTNESS_SIL_HIL_*.md`
- 对应的 JSON/CSV、TensorBoard event、trace、manifest 和 SHA-256 manifest。

最终论文表述必须遵循证据分类：CBF/ledger/rolling horizon 负责提供安全执行和拒答边界；JEPA 是否带来 safe-capture 控制收益，只有在新的三 seed paired development 和 robustness 证据通过后才能下结论。
