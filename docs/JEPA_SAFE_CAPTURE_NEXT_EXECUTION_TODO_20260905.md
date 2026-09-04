# 无人机集群对抗围捕
# Interaction-Aware Action-Conditioned JEPA + Reliability Ledger + Joint CBF-QP + Rolling Horizon
# 下一步执行版 TODO 计划书

**版本：** 2026-09-05 execution plan 1.0
**执行目录：** `D:\\uav-capture\\uav_capture`
**硬件：** NVIDIA GeForce RTX 5050，CUDA 12.8，Conda `uav-encirclement-gpu`
**当前阶段：** `development_only=true`
**locked 状态：** `locked_test_opened=false`
**第一指标：** `safe_capture`
**诊断指标：** `mean_capture_time`、transit、路径长度、CBF 修正量和延迟

> 本文件是下一步的可执行清单，不是新的实验结果。绝对 `95%` 不是本计划的硬目标；任何捕获时间改善都不能抵消 `safe_capture` 下降或安全失败。

## 0. 当前状态与近期目标

### 0.1 已确认事实

- [x] CBF-QP、rolling-horizon、CPU/CUDA replay、RTX 5050 latency 和 fault matrix 已通过当前安全接口审计。
- [x] corrected-frame P6 smoke 已完成：M0 `30/60=50.0%`，M3 `28/60=46.7%`，A1 `31/60=51.7%`，A2 `30/60=50.0%`。
- [x] smoke 中 collision、boundary、pairwise violation 和 `raw_unverified_executed` 均为 `0`。
- [x] failure attribution 已完成：失败主要以 `cbf_controlled_abort` 结束，瓶颈是排序失配、候选推进不足和 abstention/fallback 组合，不是已观察到的 raw action 绕过 CBF。
- [x] v11 hard-replay seed `20260911` 已训练并保存 `checkpoint.pt`、`history.json`、`run_metadata.json` 和 TensorBoard。
- [ ] v11 hard-replay seed `20260912`、`20260913` 尚未完成。
- [x] 已确认 v11 runner 为 `scripts/train_jepa_safe_capture_v3.py`；它保持 v2 模型类型并实现 `hard_context_weighted_v1`，与 seed `20260911` metadata 的 source hashes 对齐。

### 0.2 本轮完成定义

本轮只有同时满足以下条件才算完成：

1. 三个 training seed 的新 checkpoint 均可加载、可复现且具有完整 hash/provenance。
2. JEPA 只评价候选轨迹；最终执行动作只能来自同一个 `Joint CBF-QP` 的 `verified=true` 输出。
3. ledger 能对 OOD、stale、non-finite、uncertainty spike、provenance mismatch 明确拒答并回退。
4. 每次控制周期只执行 3-step action chunk 的第一步，随后重新观测、重规划和过滤。
5. 新 revision 先通过每 seed 20 集 smoke，再考虑每 seed 40/60 集 paired development。
6. 主结论以 `safe_capture` 为准，并同时报告安全失败、CBF abort、fallback 和证据覆盖率。

## 1. 不可变安全合同

### 1.1 信息边界

- 在线输入只允许 defender 状态、target belief、观测/通信历史、障碍几何、边界、动作历史和时间戳年龄。
- target ground truth 仅用于离线 settled label、counterfactual audit 和 episode 结算，字段标记 `offline_only=true`。
- train、validation、calibration、development、locked 按 episode/layout/seed 隔离。
- development 失败片段不得直接写回旧 archive；重训必须创建新 archive、checkpoint、protocol 和 hash。

### 1.2 候选动作合同

- 固定 `K=5`：`nominal`、`intercept`、`lateral_clearance`、`formation_clearance`、`visibility_hold`。
- 每个候选为 3 个 control steps 的 action chunk，线上只执行第 1 步。
- JEPA 只能输出未来预测、风险下界、uncertainty 和排序特征，不能输出最终动作。
- 候选进入 JEPA 前必须通过 finite、shape、speed、acceleration、slew、reachability 检查。
- nominal 永远保留为 anchor；候选无效或证据不足时不能删除 nominal。

### 1.3 `safe_capture` 结算

一个 episode 只有同时满足以下条件才为 `safe_capture=true`：

- 至少一个 defender 在 time limit 内进入 `0.80 m` capture radius；
- 无 obstacle/target/defender-defender collision；
- 无 boundary/altitude violation；
- 无 pairwise separation violation；
- 无 CBF infeasible、timeout、unverified action 或 `controlled_abort` 终止。

`controlled_abort` 是安全失败，必须单独统计；不得从分母中删除，也不得改写成普通 timeout。

### 1.4 固定故障回退

```text
separation-preserving safe-hold
  -> verified nominal through the same Joint CBF-QP
  -> controlled_abort
```

任何 CBF 失败、JEPA/ledger 超时、non-finite、OOD、stale 或 hash 不一致，都禁止执行 raw desired action。

## 2. 目标系统架构和接口

```text
多机观测/通信历史
  -> BeliefState + observation/message age
  -> 传统规划器生成 5 个动力学可行 action chunks
  -> action-conditioned interaction-aware JEPA
  -> immutable Reliability Ledger
  -> safety-first ranker + nominal anchor + hysteresis
  -> Joint CBF-QP
  -> 只执行第一控制步
  -> 重新观测、更新 belief、重新规划
```

### 2.1 BeliefState

至少记录 defender position/velocity、编队几何、pairwise TTC、target belief position/velocity、visibility、observation/message age、障碍与边界 margin、最近 `H` 个历史 token、上一周期 CBF 状态和 provenance id。在线不得读取 target truth。

### 2.2 JEPA 评价器

对每个候选和多个 horizon 输出：

- target relative displacement、velocity、acceleration；
- obstacle/inter-agent clearance lower quantile；
- pairwise TTC；
- target visibility 和 observation-age risk；
- CBF intervention probability、correction magnitude、QP feasibility；
- predictive uncertainty 或 calibrated residual。

安全预测量是排序和拒答信号，不是几何安全证明；几何安全证明仍由 CBF-QP 完成。

### 2.3 Reliability Ledger

ledger 在 calibration 后只读，并绑定 checkpoint、protocol 和 calibration archive hash：

| 状态 | 允许路径 |
|---|---|
| `trusted` | 允许使用 JEPA 进行安全优先排序 |
| `fallback_nominal` | 忽略 JEPA 排序，执行 nominal，经 CBF |
| `safe_hold` | 执行 separation-preserving hold，经 CBF |
| `controlled_abort` | hold/nominal 均不可验证时终止，不执行 raw |

credit 不是安全证书。低信用必须改变执行路径，而不是只给 score 加一个很小的惩罚。

## 3. 总体依赖图

```text
P0 证据/环境冻结
  -> P1 完成三 seed hard-replay training
  -> P2 checkpoint、prediction、auxiliary-head audit
  -> P3 calibration archive + hash-bound ledger
  -> P4 ranking/abstention/hysteresis protocol revision
  -> P5 100-cycle replay + CBF/latency regression
  -> P6 三 seed x 20 集 smoke
  -> P7 三 seed x 40/60 集 paired development
  -> P8 robustness + SIL/HIL readiness
  -> P9 论文证据归档和 locked-test 决策
```

未通过上游 gate 时不得扩大 episode 数、不得调 CBF margin、不得打开 locked test。

## 4. P0：环境、协议和证据冻结

**目标：** 防止旧 V4/V5、tmp 或不同 seed 的产物污染新比较。

- [ ] 检查 `git status --short`，保留用户已有 README/E1/V5/tmp 改动，不使用 `git add .`、reset 或删除 `tmp/`。
- [ ] 保存 Git revision、Conda 包清单、Python/PyTorch/CUDA/GPU 信息。
- [ ] 保存当前 protocol、train/validation/calibration archive、scene manifest、checkpoint 和 ledger 的 SHA-256。
- [ ] 为每个新 run 建立空的独立 `results/` 和 TensorBoard 目录；工具必须拒绝覆盖非空目录。
- [ ] 运行 `git diff --check`、协议 schema test、核心安全测试和 CUDA 可用性检查。
- [ ] 写入 `preflight.json`，包含命令、环境、输入 hash、`development_only=true` 和 `locked_test_opened=false`。

### P0 出口门

- [ ] 所有输入 hash 与 run metadata 一致；任何 mismatch 立即停止。
- [ ] 运行器明确拒绝 locked split 和在线 target truth。

## 5. P1：完成 v11 hard-replay 三 seed 训练

**目标：** 使用完全相同的 corrected-frame train/validation archive 和 hard-replay 规则完成 `20260912/20260913`，不把 smoke/development 数据回灌训练。

### P1-0 runner 兼容性门（必须先做）

- [x] 检查 v11 protocol 的 phase、TensorBoard、数据路径和训练字段；与 hard-context trainer 的运行时合同一致。
- [x] 确认 `train_jepa_safe_capture_v2.py` 不提供 hard-context CLI；因此不将它作为 v11 hard-replay runner。
- [x] 固定唯一 runner 为 `scripts/train_jepa_safe_capture_v3.py`；该 runner 的 hard-context 权重由离线安全标签计算，不读取 development/locked episode。
- [x] 运行 `--help`、CUDA 检查和 metadata/hash 对齐检查；完整训练在这些检查通过后执行。
- [x] runner、protocol、training config 和 source hashes 写入每个 run metadata；不覆盖已完成的 `20260911` checkpoint。

### P1-1 训练任务

- [ ] 训练 seed `20260912`，输出 `results/jepa_safe_capture_v11_hard_replay_seed20260912/`。
- [ ] 训练 seed `20260913`，输出 `results/jepa_safe_capture_v11_hard_replay_seed20260913/`。
- [ ] 两个 seed 使用 `40 epochs`、`batch_size=512`、`hidden=128`、`latent=64`、`lr=1e-3`、`weight_decay=1e-5`、`quantile=0.10`、CUDA。
- [ ] replay 使用 train-only hard weights，uniform draw fraction 不低于 `0.50`，hard weight cap 为 `8.0`。
- [ ] TensorBoard 分目录记录 config、dataset metadata、replay manifest、loss、task metrics、histogram、source hashes。
- [ ] 输出 `checkpoint.pt`、`history.json`、`run_metadata.json`，并记录最佳 validation epoch/loss。

### P1 出口门

- [ ] 两个 checkpoint 可加载，三 seed 的 model schema、task heads 和 action/history shape 一致。
- [ ] train/validation/calibration episode seed 无交集。
- [ ] 所有训练输出 finite；无 NaN/Inf、无目录覆盖、无 locked 字段异常。
- [ ] 训练 loss 下降只是必要条件，不等于闭环 safe-capture 提升。

## 6. P2：checkpoint、预测和安全辅助头审计

**目标：** 证明新 checkpoint 提供可用的安全预测信号，而不是只改善平均位移 MAE。

- [ ] 对三 seed 做 held-out validation prediction replay，禁止使用 development episode。
- [ ] 检查 displacement、velocity、acceleration 的 MAE/P50/P90/P95 与 constant-velocity baseline。
- [ ] 对 clearance lower bound 计算 coverage、underestimation rate、over-optimism rate 和安全下界违约率。
- [ ] 对 visibility、CBF intervention、QP feasibility 计算 Brier、ECE、AUROC/AUPRC 和 hard-slice recall。
- [ ] 对 TTC、observation age、CBF correction 计算极端风险漏报率。
- [ ] 检查同一 belief 下 5 个 action chunks 是否有非零 latent/prediction separation。
- [ ] 按急转、速度突变、S-curve、遮挡、消息延迟、拥挤队形和低净空分桶。
- [ ] 生成每 seed `prediction_metrics.json`、`prediction_audit.md`、CSV/曲线、TensorBoard audit 和 checkpoint manifest。

### P2 出口门

- [ ] 三 seed 所有输出 finite，主要 horizon 不劣于 constant-velocity。
- [ ] 安全辅助头具有非空覆盖，不能系统性过度乐观。
- [ ] action-following separation 非零且可复现；若为零，停止闭环接入。
- [ ] prediction gate 通过只说明“预测信号可用”，不说明“控制收益已证明”。

## 7. P3：calibration archive 和 reliability ledger

**目标：** 让 ledger 在预测漂移前拒答，并使所有拒答可追溯。

- [ ] 使用与 train、validation、development 不重叠的 calibration archive；记录 archive episode/layout/seed hash。
- [ ] 覆盖 nominal、stale、dropout、noise、message delay/dropout、target turn、速度突变、遮挡和障碍 density shift。
- [ ] 为每个 `(checkpoint, context bucket, horizon, head)` 记录样本数、误差、coverage、credit 和置信区间。
- [ ] 固定 `minimum_sample_count`、`minimum_credit`、uncertainty、risk、TTC 和 stale-age 阈值；查看闭环结果前冻结。
- [ ] 生成每个 checkpoint 独立的 hash-bound ledger，不复用旧 checkpoint 的 ledger。
- [ ] 注入 OOD、non-finite、unknown horizon、stale 和 provenance mismatch；100% 进入 fallback/safe-hold。
- [ ] 验证 ledger immutability：运行期间不能在线更新 credit、threshold 或 bucket 统计。
- [ ] 生成 `ledger.json`、alignment report、temporal report、fault report、hash manifest 和 TensorBoard。

### P3 出口门

- [ ] high-credit settled failure rate 不高于 low-credit bucket。
- [ ] 每次状态转移都有 reason code、时间戳、输入 hash 和回退动作。
- [ ] OOD/stale/non-finite/provenance fault 的 raw/unverified execution 为 `0`。
- [ ] bucket 证据不足时固定输出 `insufficient_evidence`，不得把缺证据当作成功。

## 8. P4：安全优先排序、abstention 和滞回

**目标：** 解决当前 `selected_not_settled_best`、负 settled rank correlation 和 CBF abort 推进不足，同时不放宽安全边界。

- [ ] 新建 ranking protocol revision，冻结候选、单位、权重、tie tolerance、abstention margin、minimum hold 和 hysteresis。
- [ ] 使用词典序：`finite/reachability -> safety lower bound -> ledger state -> task progress`。
- [ ] 预测 clearance/TTC/visibility lower bound 不满足时，禁止用任务进展分数抬高候选。
- [ ] 仅当 ledger=`trusted` 且 top-two margin 足够时允许 JEPA 改变候选；否则走 nominal-CBF 或 safe-hold-CBF。
- [ ] 记录每个候选的 progress、clearance lower bound、visibility、uncertainty、CBF cost、action-change cost、anchor penalty。
- [ ] 增加 candidate separation、selected-not-best、top-1 safety precision、switch rate、oscillation length 和 CBF correction trace。
- [ ] 候选近似等价时记录 `insufficient_candidate_separation`，不强行切换。
- [ ] 每个 ranking change 做离线 settled replay；不得根据 mean capture time 或单 seed 事后调权重。

### P4 出口门

- [ ] settled ranking mismatch 和 high-credit failure 相对当前 revision 改善，或至少不超过预设 non-regression 容差。
- [ ] abstention 不执行 raw action；`raw_unverified_executed=0`。
- [ ] 双次 replay 的候选选择、CBF verification、fallback 和结算逐字段一致。

## 9. P5：100-cycle rolling-horizon 与 Joint CBF 回归

- [ ] 对 zero-perturbation identity path 做逐字段回归。
- [ ] 对 non-zero candidate path 做至少 100 个 control cycles 的双次 deterministic replay。
- [ ] 验证每周期顺序固定为 observe -> belief -> candidates -> JEPA -> ledger -> rank -> CBF -> execute-first-step。
- [ ] 验证 action chunk 后续步骤绝不在 open-loop 中执行。
- [ ] 注入 JEPA timeout、ledger timeout、non-finite prediction、CBF infeasible、CBF timeout、通信中断和 solver fault。
- [ ] 验证所有 fallback 仍走同一个 Joint CBF-QP，记录 status、slack、active constraints、correction norm、latency。
- [ ] 在 RTX 5050 和 CPU 上重放，比较 candidate decision、verified action、safety settlement 和 trace hash。
- [ ] 汇总 cycle p50/p95/p99、queue age、CBF p95；当前 contract 为端到端 p95 不超过 `100 ms`。

### P5 出口门

- [ ] collision/boundary/pairwise/raw-unverified 均为 `0`。
- [ ] CBF timeout/infeasible 只产生显式 fallback 或 controlled abort，不返回 raw request。
- [ ] CPU/CUDA replay 结算一致；若浮点 tie 造成 decision drift，先冻结 tie policy 再进入 smoke。

## 10. P6：三 seed、四变体、每 seed 20 集 smoke

### 固定矩阵

| 变体 | 定义 | 目的 |
|---|---|---|
| M0 | frozen nominal + CBF | 基线 |
| M3-new | JEPA + ledger + auxiliary safety rank + CBF | 主候选 |
| A1-new | JEPA + auxiliary rank，无 ledger + CBF | ledger 消融 |
| A2-new | JEPA + ledger，无 clearance/visibility rank + CBF | auxiliary-head 消融 |

- [ ] 每个 training seed 先生成唯一 paired scene manifest；M0/M3/A1/A2 复用同一 manifest。
- [ ] 每个变体每 seed 运行 20 集；每个 output、TensorBoard 和报告目录全新创建。
- [ ] 记录 episode、step trace、scene、provenance、checkpoint/ledger hash 和命令行。
- [ ] 运行 safety、latency、ledger alignment、temporal ledger、settled ranking、failure index 和 deterministic replay。
- [ ] 统计 per-seed `safe_capture`、paired delta、improved/degraded/tied、95% CI、CBF abort、fallback 和 failure cause。

### P6 硬门

- [ ] 任一 collision、boundary、pairwise、raw-unverified 或不可解释安全回退，立即停止扩大规模。
- [ ] M3-new 不得相对 M0 在三 seed 中系统性退化；不能用平均值掩盖单 seed 安全回归。
- [ ] 若安全全通过但没有控制收益，分类为 `prediction_signal_no_control_gain` 或 `safe_non_inferior`，回到 P4，不重复采样掩盖结果。

## 11. P7：三 seed paired development（每 seed 40/60 集）

**进入条件：** P6 全部硬门通过，且 P2-P5 证据完整。

- [ ] 预注册 episode 数、seed、scene manifest、候选、CBF margin、ledger threshold 和统计方法。
- [ ] 运行 M0/M3-new/A1-new/A2-new；A3 raw/no-CBF 只作为独立风险诊断，不作为主方法。
- [ ] 不在 block 内调 threshold、CBF margin、candidate weight、chunk length 或 episode seed。
- [ ] 以 `(training_seed, episode)` 为统计单位，不把 timestep 当独立样本。
- [ ] 报告每 seed 与 aggregate 的 `safe_capture`、paired delta、bootstrap CI、exact McNemar 和 improved/degraded/tied。
- [ ] 单独报告 collision、boundary、pairwise、CBF abort/timeout/infeasible、fallback、raw-unverified、transit 和 minimum clearance。
- [ ] `mean_capture_time` 只放诊断表，不作为安全失败的补偿项。

### P7 结论分类

| 分类 | 条件 | 后续 |
|---|---|---|
| `promising_development_candidate` | 平均 paired delta > 0、至少 2/3 seed 非负、安全硬门全通过 | P8 robustness，并草拟新 locked preregistration |
| `safe_non_inferior` | 平均 delta >= 0、安全硬门全通过，但正向证据不足 | 做 robustness，不宣称提升 |
| `prediction_signal_no_control_gain` | 离线排序改善，闭环无收益 | 保留负结果，继续修复或停止分支 |
| `no_control_gain` | JEPA 与 M0 无稳定差异 | 归档为安全基础设施/消融 |
| `BLOCKED_BY_SAFETY` | 任一安全硬门失败 | 立即停止，修复 CBF/回退链 |

## 12. P8：robustness、SIL/HIL 和部署就绪

- [ ] 构建 observation dropout/noise、communication delay/dropout、target turn/acceleration、障碍密度、初始侧距和拥挤队形 stress matrix。
- [ ] 每类 stress 都审计 safe_capture、安全硬门、ledger 状态、fallback、CBF correction 和 controlled abort。
- [ ] 注入 GPU 不可用、JEPA 超时、ledger 损坏、hash mismatch、solver timeout、传感器陈旧和通信中断。
- [ ] 验证 100/500/1000 control-cycle 长序列 deterministic replay、候选抖动、rollout drift 和 trace 完整性。
- [ ] RTX 5050 上测量 JEPA、ledger、ranker、CBF 和总周期 p50/p95/p99。
- [ ] 完成 SIL 接口合同：传感器时间戳、通信年龄、动作回执、solver 状态和 controlled abort 事件。
- [ ] 如进入 HIL，先保持“只执行 verified CBF action”的执行边界，再接入真实飞控。
- [ ] 编写故障手册和恢复动作表；任何故障都不得放行 raw desired action。

### P8 出口门

- [ ] 所有 stress case 的安全硬门和 fallback 语义通过。
- [ ] 端到端 p95 不超过 `100 ms`，无长块 open-loop 执行。
- [ ] SIL/HIL trace schema 可与仿真 trace 对齐并重建每次动作 provenance。

## 13. P9：论文证据、发布和 locked 决策

仅在 P7/P8 通过后执行：

- [ ] 生成架构图、数据流图、ledger 状态机、候选排序和 Joint CBF 约束图。
- [ ] 发布代码 revision、Conda 环境、protocol、checkpoint、ledger、calibration archive、scene manifest、TensorBoard 和 SHA-256 manifest。
- [ ] 主表只把 `safe_capture` 作为 endpoint；均值捕获时间、transit、路径代价和 latency 为 secondary diagnostics。
- [ ] 同时公开正向、负向、不确定和 `insufficient_evidence` 结果，区分 development 与 locked evidence。
- [ ] 只有 `promising_development_candidate` 且 robustness/SIL gate 通过，才创建新的 locked-test preregistration。
- [ ] locked protocol 冻结后不得依据 locked 结果反向调参；不满足条件则保持 `locked_test_opened=false` 并归档。

## 14. 每次运行的统一检查表

### 运行前

- [ ] 目录为空且不会覆盖旧结果。
- [ ] protocol/checkpoint/ledger/calibration/scene/code/environment hash 已记录。
- [ ] `development_only=true`、`locked_test_opened=false`、split 合法。
- [ ] CUDA、PyTorch、GPU、TensorBoard 和 targeted tests 通过。

### 运行中

- [ ] 每个动作均为同一 Joint CBF-QP 返回的 finite、verified action 第一控制步。
- [ ] 记录 candidate validity、ledger state/reason、ranker selection、CBF status、fallback、queue age 和 latency。
- [ ] 不读取 online target truth，不在线更新 ledger，不删除失败 step。

### 运行后

- [ ] `summary.json`、`episodes.csv`、step traces、provenance、manifest、TensorBoard event 齐全。
- [ ] collision/boundary/pairwise/raw-unverified/timeout/infeasible/abort 分开计数。
- [ ] 完成 paired aggregate、failure index、settled ranking 和 deterministic replay。
- [ ] Markdown/JSON/CSV/TensorBoard 交叉一致，生成 hash manifest。
- [ ] 仅提交本阶段文件，不混入 README、E1、V5 或 `tmp/` 改动。

## 15. 近期实际执行顺序

1. [ ] 完成 P0 preflight，并确认 v11 trainer/protocol 兼容性。
2. [ ] 完成 hard-replay seed `20260912`。
3. [ ] 完成 hard-replay seed `20260913`。
4. [ ] 审计三份 checkpoint、validation prediction、auxiliary heads 和 TensorBoard。
5. [ ] 为三份 checkpoint 生成独立 calibration-bound ledger。
6. [ ] 冻结新的 safety-first ranking/abstention/hysteresis protocol。
7. [ ] 完成 100-cycle replay、CBF fault、latency 和 CPU/CUDA regression。
8. [ ] 运行三 seed x 四变体 x 20 集 smoke。
9. [ ] 只有 smoke 安全硬门通过且 M3 不再系统性退化，才运行三 seed x 四变体 x 40/60 集 paired development。
10. [ ] 最后运行 robustness/SIL/HIL readiness，并根据预定义分类决定是否创建新的 locked preregistration。

## 16. 禁止事项和停止规则

- 不把单 seed、smoke、prediction MAE 或 mean capture time 写成正式控制收益。
- 不以 `95%` 作为绝对成功门，也不以捕获时间抵消 safe-capture 失败。
- 不降低 CBF margin、关闭 stale/OOD 检查或放宽回退路径来追逐分数。
- 不把 `controlled_abort` 从失败分母中删除。
- 不将 archive-recovery checkpoint 冒充历史 V4 warm-start checkpoint。
- 不在 locked protocol 冻结前查看 locked split；本计划全程保持 `locked_test_opened=false`。

**最终判断标准：** 只有当 JEPA 的候选反事实排序、ledger 的可信度拒答、Joint CBF-QP 的硬安全边界和 rolling-horizon 的闭环重规划，在三 seed 困难场景中共同通过上述审计，才能把系统称为“安全增强的闭环围捕系统”。否则应明确归类为安全基础设施、预测信号或 development-only 负结果。
