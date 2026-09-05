# 无人机集群对抗围捕
# Interaction-Aware Action-Conditioned JEPA + Reliability Ledger + Joint CBF-QP + Rolling Horizon
# 下一步执行版 TODO 计划书

**版本：** 2026-09-05 execution plan 1.6
**执行目录：** `D:\\uav-capture\\uav_capture`
**硬件：** NVIDIA GeForce RTX 5050；已验证运行时为 `D:\\download\\anaconda3\\envs\\traj_pred_prep`（Torch `2.9.1+cu130`，CUDA 可用）。目标环境 `uav-encirclement-gpu` 尚未安装，`environment.yml` 的 Conda/pip 下载目前受镜像/网络阻塞。
**当前阶段：** `v12 calibrated-clearance development；排序收益与 CPU/CUDA 决定性修复中`
**locked 状态：** `locked_test_opened=false`
**第一指标：** `safe_capture`
**诊断指标：** `mean_capture_time`、transit、路径长度、CBF 修正量和延迟

> 本文件是下一步的可执行清单，不是新的实验结果。绝对 `95%` 不是本计划的硬目标；任何捕获时间改善都不能抵消 `safe_capture` 下降或安全失败。

## 0. 当前状态与近期目标

### 0.1 已确认事实

- [x] v11 hard-replay 三 seed (`20260911/20260912/20260913`) 的 checkpoint、history、metadata 和 TensorBoard 均已生成并可加载。
- [x] v12 q10 residual clearance calibration 已在独立 calibration split 完成；normalized-to-metre 转换、quantile 方法、样本数和 hash binding 已写入校准报告。
- [x] 旧 v12 三 seed 的 calibrated checkpoint-bound reliability ledger 已生成，ledger、calibration archive、protocol 和 builder hash 均可追溯（仅作旧协议对照）。
- [ ] 由于本计划新增 `score_comparison_quantum_m`，必须用新 protocol hash 重新生成三份 `v12_r3` checkpoint-bound ledger；不得复用旧 `_r2` ledger。
- [x] ledger fallback、temporal ledger、OOD/stale/non-finite/unknown-horizon/provenance fault、Joint CBF-QP fault 和 RTX 5050 latency audit 已完成。
- [x] 三 seed x M0/M3/A1/A2 x 20 集 v12 smoke 已完成，使用同一 paired scene manifest；collision、boundary、pairwise 和 `raw_unverified_executed` 均为 `0`。
- [x] 当前失败主要是 `cbf_controlled_abort`（240 个 episode 中 119 个），同时存在候选排序失配和候选推进不足；没有观察到 raw action 绕过 CBF。
- [ ] CPU/CUDA replay 尚未通过：abstention margin 在浮点边界附近漂移，不能宣称跨设备等价。
- [x] 本计划 1.4 的新 `preflight.json` 和输入 hash 快照已补齐；protocol hash 与六项冻结输入均已核验。

### 0.2 已完成的 v12 calibrated-clearance smoke 证据

- [x] aggregate：`results/jepa_safe_capture_v12_calibrated_clearance_smoke_aggregate/`；每个 seed 使用同一 paired manifest、20 个 episode、`safe_capture` 为主指标。
- [x] seed `20260911`：M0 `50%`，M3 `40%`，A1 `45%`，A2 `45%`；M3 paired delta `-10 pp`。
- [x] seed `20260912`：M0 `50%`，M3 `30%`，A1 `60%`，A2 `40%`；M3 paired delta `-20 pp`。
- [x] seed `20260913`：M0 `50%`，M3 `55%`，A1 `50%`，A2 `55%`；M3 paired delta `+5 pp`。
- [x] 聚合 M3 paired delta 为 `-8.33 pp`，非负 seed `1/3`，improved/degraded/tied=`4/9/47`；240 个 episode 中 safe capture `114/240`，`cbf_controlled_abort=119`，timeout `7`。
- [x] settled ranking 仍未达标：selected-not-best=`31.7%/39.0%/52.4%`，Spearman=`-0.498/-0.532/-0.586`。
- [ ] 当前分类：`prediction_signal_no_control_gain`；不能扩大到每 seed 40/60 集，不能打开 locked test，也不能把单 seed 正向结果写成正式提升。

### 0.3 本轮完成定义

本轮只有同时满足以下条件才算完成：

1. 三个 training seed 的新 checkpoint 均可加载、可复现且具有完整 hash/provenance。
2. JEPA 只评价候选轨迹；最终执行动作只能来自同一个 `Joint CBF-QP` 的 `verified=true` 输出。
3. ledger 能对 OOD、stale、non-finite、uncertainty spike、provenance mismatch 明确拒答并回退。
4. 每次控制周期只执行 3-step action chunk 的第一步，随后重新观测、重规划和过滤。
5. 新 revision 先通过每 seed 20 集 smoke，再考虑每 seed 40/60 集 paired development。
6. 主结论以 `safe_capture` 为准，并同时报告安全失败、CBF abort、fallback 和证据覆盖率。

### 0.4 下一轮立即执行队列（按此顺序，不跨门）

| 阶段 | 当前状态 | 具体动作 | 必须产生的证据 | 通过后才可进入 |
|---|---|---|---|---|
| A. 复核与冻结 | 已完成 | 保存当前 v12 输入、代码、环境和结果 hash；冻结旧结果只读 | 新 `preflight.json`、输入 hash manifest、v12 aggregate 快照 | B |
| B. 净空校准 | 已完成 | 独立 split q10 residual offset、单位转换、覆盖率审计 | 三 seed calibration report、calibration hash | C |
| C. 新 ledger | 已完成 | 用新 protocol/hash 生成 checkpoint-bound calibrated ledger；拒绝缺证据/OOD/stale/non-finite/provenance mismatch | 三份 `v12_r3` ledger、fallback audit、TensorBoard | D |
| D. 排序与决定性修复 | 进行中 | 先修 CPU/CUDA abstention 边界，再修 settled ranking、候选分离和 controlled-abort 推进 | ranker 单测、双设备 trace、settled report、protocol hash | E |
| E. 回归审计 | 阻塞 | 100-cycle replay、CBF fault matrix、latency、双设备 deterministic replay | 独立结果目录和 TensorBoard；安全硬门报告 | F |
| F. Smoke 门 | 已完成但未通过收益门 | 三 seed x M0/M3/A1/A2 x 20 paired episodes；不调参、不换 seed、不删除 abort | v12 aggregate、failure index、settled ranking | 回到 D |
| G. 扩展开发集 | 条件执行 | 仅当 F 的收益/安全/排序门同时通过时运行每 seed 40/60 集 | preregistration、完整 paired report | H |
| H. 鲁棒性与部署 | 条件执行 | dropout/noise/delay/target turn/拥挤/障碍密度和 SIL/HIL readiness | stress matrix、长序列 replay、接口审计 | I |
| I. 发布/锁定决策 | 条件执行 | 归档正负结果；只有 H 通过才考虑新的 locked preregistration | release manifest、论文表格、locked 决策记录 | 结束 |

**当前唯一优先事项：** P0 证据冻结已经完成；现在先完成 D 中的 CPU/CUDA 浮点边界修复并重跑双设备 replay，再处理 settled ranking、候选分离和 controlled-abort 推进。当前 v12 smoke 的 `safe_capture` 尚未证明 M3 有控制收益，不能直接扩大到 40/60 集，也不能用降低 CBF margin、删掉 `controlled_abort` 或调低安全阈值换取通过。

### 0.5 每阶段统一产物命名

新产物必须使用独立前缀 `jepa_safe_capture_v12_calibrated_clearance_*`，不得覆盖 v11 目录：

```text
results/
  jepa_safe_capture_v12_calibrated_clearance_calibration_seed<seed>/
  jepa_safe_capture_v12_calibrated_clearance_ledger_seed<seed>/
  jepa_safe_capture_v12_calibrated_clearance_rank_seed<seed>/
  jepa_safe_capture_v12_calibrated_clearance_smoke_<variant>_seed<seed>/
  jepa_safe_capture_v12_calibrated_clearance_smoke_aggregate/
  jepa_safe_capture_v12_calibrated_clearance_failure_index/
  jepa_safe_capture_v12_calibrated_clearance_settled_seed<seed>/
tensorboard/
  jepa_safe_capture_v12_calibrated_clearance/<stage>/seed<seed>/
```

每个目录都必须从空目录开始创建，并至少包含 `summary.json`、`run_metadata.json`、输入/代码 hash、命令行、`development_only=true`、`locked_test_opened=false`。任何脚本发现目标目录非空都必须停止。

### 0.6 本轮硬性停止规则

- 校准 offset 不是 q10、不是 calibration-only，或无法绑定 checkpoint hash：停止，不接入运行时。
- calibrated lower-bound 导致大量候选变为负值、单位不一致或 `finite` 检查失败：停止，修复标签/坐标转换，不调低 CBF margin。
- ledger 状态、排序选择和 CBF verified action 在双次 replay 中不一致：停止，不进行 smoke。
- 任一 collision、boundary、pairwise violation、CBF timeout/infeasible 未进入明确 fallback，或 `raw_unverified_executed > 0`：立即停止并标记 `BLOCKED_BY_SAFETY`。
- M3 在 smoke 中少于 2/3 seed 非负、平均 paired delta < 0、或 ranking mismatch 恶化：回到 D，不扩大 episode 数。
- 所有证据不足统一标记 `insufficient_evidence`；不得把“没有观测到失败”写成“已经证明安全”。

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
- [x] 写入 `preflight.json`，包含命令、环境、输入 hash、`development_only=true` 和 `locked_test_opened=false`；当前文件为 `results/jepa_safe_capture_v12_r3_preflight/preflight.json`。

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

- [x] 训练 seed `20260912`，输出 `results/jepa_safe_capture_v11_hard_replay_seed20260912/`。
- [x] 训练 seed `20260913`，输出 `results/jepa_safe_capture_v11_hard_replay_seed20260913/`。
- [x] 两个 seed 使用 `40 epochs`、`batch_size=512`、`hidden=128`、`latent=64`、`lr=1e-3`、`weight_decay=1e-5`、`quantile=0.10`、CUDA。
- [x] hard-context 训练使用 train-only 困难上下文权重，cap 为 `8.0`，不读取 development/locked episode。
- [x] TensorBoard 分目录记录 config、dataset metadata、loss、task metrics、histogram 和 source hashes。
- [x] 输出 `checkpoint.pt`、`history.json`、`run_metadata.json`，并记录最佳 validation epoch/loss。

### P1 出口门

- [x] 三个 checkpoint 可加载，三 seed 的 model schema、task heads 和 action/history shape 一致。
- [x] train/validation/calibration episode seed 无交集。
- [x] 所有训练输出 finite；无 NaN/Inf、无目录覆盖、无 locked 字段异常。
- [ ] 训练 loss 下降只是必要条件，不等于闭环 safe-capture 提升。

## 6. P2：checkpoint、预测和安全辅助头审计

**目标：** 证明新 checkpoint 提供可用的安全预测信号，而不是只改善平均位移 MAE。

- [x] 对三 seed 做 held-out validation prediction replay，禁止使用 development episode。
- [x] 检查 displacement、velocity、acceleration 的 MAE 与 constant-velocity baseline。
- [x] 对 clearance lower bound 计算误差、visibility/CBF/QP 头的 Brier/AUROC 和 finite gate。
- [x] 对 TTC、observation age、CBF correction 计算误差指标；QP feasibility 的 AUROC 因标签无类别变化记为 `n/a`。
- [ ] 检查同一 belief 下 5 个 action chunks 是否有非零 latent/prediction separation。
- [ ] 按急转、速度突变、S-curve、遮挡、消息延迟、拥挤队形和低净空分桶。
- [x] 生成三 seed prediction gate、aggregate report、TensorBoard audit 和 checkpoint manifest。

### P2 出口门

- [x] 三 seed 所有输出 finite，主要 horizon 均优于 constant-velocity。
- [x] 安全辅助头具有非空覆盖；QP feasibility 标签无类变化，未宣称其校准能力。
- [ ] action-following separation 非零且可复现；若为零，停止闭环接入。
- [ ] prediction gate 通过只说明“预测信号可用”，不说明“控制收益已证明”。

## 7. P3：calibration archive 和 reliability ledger

**目标：** 让 ledger 在预测漂移前拒答，并使所有拒答可追溯。

- [x] 使用与 train、validation、development 不重叠的 calibration archive；记录 archive episode/layout/seed hash。
- [x] calibration archive 覆盖 nominal、stale、dropout、noise、message delay/dropout、target turn、速度突变、遮挡和障碍 density shift。
- [x] 为每个 `(checkpoint, context bucket, horizon, head)` 记录样本数、误差、coverage、credit 和置信区间。
- [x] 固定 `minimum_sample_count=128`、`minimum_credit=0.65`、uncertainty/risk/TTC/stale-age 阈值。
- [x] 已为三个 checkpoint 生成旧版 hash-bound ledger；该产物只用于历史对照。
- [x] 按 v12 calibrated-clearance protocol 1.4 重新生成 `v12_r3` ledger；未复用旧 `_r2` 目录。
- [x] 注入 OOD、non-finite、unknown horizon、stale 和 provenance mismatch；全部进入 fallback/safe-hold。
- [x] 验证 ledger immutability：运行期间不能在线更新 credit、threshold 或 bucket 统计。
- [x] 生成三份 `ledger.json`、aggregate report、fallback audit、hash manifest 和 TensorBoard。

### P3 出口门

- [x] high-credit settled failure rate 不高于 low-credit bucket。
- [x] 每次状态转移都有 reason code、时间戳、输入 hash 和回退动作。
- [x] OOD/stale/non-finite/provenance fault 的 raw/unverified execution 为 `0`。
- [x] bucket 证据不足时固定输出 `insufficient_evidence`，不得把缺证据当作成功。

### P3.1 净空校准变更（已完成）

v11 ledger 只作为历史对照；v12 已把 clearance head 的 calibration residual 接入运行时，形成新的 calibrated ledger。校准只允许使用独立 calibration split，不能使用 smoke/development/locked episode。

对每个 checkpoint、head 和 horizon，固定计算：

```text
residual = y_clearance_m - raw_prediction_m
offset_q10 = quantile(residual, q=0.10, method=预先固定的方法)
calibrated_lower_bound_m = raw_prediction_m + offset_q10
```

- [x] 明确并记录 normalized-label 到米的转换：先将预测和标签都乘同一 `world_extent_m`，再计算 residual；禁止把归一化值与米混算。
- [x] 保存 residual 分布、`offset_q10`、样本数、确定性分位数置信区间、每个 context bucket 的 coverage 和 over-prediction rate。
- [x] 不使用 calibration mean、median 或事后挑选的 quantile 代替 `q=0.10`；quantile 方法和缺失值处理已写入 protocol。
- [x] 将 offset 写入 ledger 的 checkpoint-bound metadata，并把 calibration archive、metadata、protocol 和 builder hash 全部写入 `source`。
- [x] runtime ledger 只读取已绑定的 calibrated lower bound；未找到 offset、hash 不匹配、单位不匹配或样本不足时只能 `fallback_nominal`/`safe_hold`。
- [x] ranker 的 `minimum_predicted_clearance_m=0.15` 保持不变；校准只修正预测偏差，不放宽 CBF margin 或 clearance gate。
- [x] trace 同时记录 `raw_predicted_clearance_m`、`calibration_offset_m`、`calibrated_lower_bound_m` 和 `clearance_gate_pass`，可区分模型低估与真实低净空。
- [x] 新增并通过 q10 transform、normalized-to-meter conversion、checkpoint/ledger hash binding、missing-offset refusal、candidate eligibility、tie/abstention 和 CBF-only execution 测试（针对性测试 `20 passed`）。

**P3.1 状态：已通过。** 三 seed 的校准文件可独立重建，校准下界 finite，ledger fault routing 和 trace observability 已通过；新的 `v12_r3` ledger 已绑定当前 protocol hash。跨设备 replay 的最终一致性仍属于 P5 未决问题，不能由 P3.1 单独替代。

## 8. P4：安全优先排序、abstention 和滞回

**目标：** 解决当前 `selected_not_settled_best`、负 settled rank correlation 和 CBF abort 推进不足，同时不放宽安全边界。v12 已完成净空校准接入，但本节的 ranking gate 尚未通过。

- [x] 新建 ranking protocol revision，冻结候选、单位、权重、tie tolerance、abstention margin、minimum hold 和 hysteresis。
- [ ] 使用词典序：`finite/reachability -> safety lower bound -> ledger state -> task progress`；当前实现需要在修复浮点边界比较后重新验证。
- [ ] 预测 clearance/TTC/visibility lower bound 不满足时，禁止用任务进展分数抬高候选。
- [ ] 仅当 ledger=`trusted` 且 top-two margin 足够时允许 JEPA 改变候选；否则走 nominal-CBF 或 safe-hold-CBF。
- [ ] 记录每个候选的 progress、clearance lower bound、visibility、uncertainty、CBF cost、action-change cost、anchor penalty。
- [ ] 增加 candidate separation、selected-not-best、top-1 safety precision、switch rate、oscillation length 和 CBF correction trace。
- [ ] 候选近似等价时记录 `insufficient_candidate_separation`，不强行切换。
- [ ] 每个 ranking change 做离线 settled replay；不得根据 mean capture time 或单 seed 事后调权重。

### P4 出口门

- [ ] settled ranking mismatch 和 high-credit failure 相对当前 revision 改善，或至少不超过预设 non-regression 容差。
- [ ] abstention 不执行 raw action；`raw_unverified_executed=0`。
- [ ] 双次 replay 的候选选择、CBF verification、fallback 和结算逐字段一致；CPU/CUDA 在 abstention 边界仍有 drift。
- [ ] 但 settled ranking mismatch 门未通过；下一轮必须修复 rank objective/候选分离后重新审计，不能调低 CBF 安全边界。

### P4.1 下一轮立即执行的排序修复清单

- [ ] 先冻结本轮四变体结果、三份 settled decision rows、ledger 和 protocol hash；修复过程不得覆盖这些证据。
- [ ] 为修复建立新的 `v12_r3` 结果/TensorBoard 前缀，禁止覆盖已有 `_r2` smoke 及其报告。
- [ ] 将 abstention 比较改为跨设备确定性规则：优先使用固定小数/整数化比较；若仍使用浮点，显式加入安全 band，并把 band 写入 protocol/hash。
- [ ] 新增 margin 在 `0.002 m` 附近的边界单测，覆盖 `0.0019779`、`0.0020000`、`0.0020076` 等值，确保 CPU/CUDA 走同一路由。
- [ ] 完成 seed `20260911` 的 CPU/CUDA replay；当前 safety band 已从 `0.0005 m` 扩至 `0.001 m` 以覆盖第二个已观测 margin drift，需重新生成证据后再按相同输入重跑 `20260912/20260913`。
- [ ] 当前已知复现点：同一 step 的 CUDA margin 约 `0.0020076045`、CPU margin 约 `0.0019779083`，现有阈值 `0.0015 + 0.0005 = 0.002` 导致一端选 candidate、另一端 abstain；修复必须消除该分叉而不是调低 CBF margin。
- [ ] 按 `observation_condition`、`target_motion_mode`、obstacle count、clearance bucket、ledger state 和 candidate separation 分桶，定位负 Spearman 与 `selected_not_best` 的来源。
- [ ] 对每个候选同时计算 task progress、conservative clearance lower bound、pairwise TTC、visibility/age risk、CBF correction cost、action-change cost 和 ledger credit；保存未归一化值与单位。
- [ ] 将排序实现为硬安全词典序：finite/reachability -> clearance/TTC lower bound -> ledger state -> visibility/age -> task progress；任务分数不能抬高安全下界不足的候选。
- [ ] 对 clearance 使用校准后的低分位下界，并单独惩罚 clearance over-prediction；不得用平均 MAE 代替下界校准。
- [ ] 增加 candidate-separation gate：top-two 差小于 `0.0015 m` 或候选动作范数差小于固定阈值时，保持 nominal/上一动作，不强行切换。
- [ ] 保留 nominal anchor、minimum hold `2` 和 hysteresis `0.001 m`；所有 abstention/fallback 仍必须经过 Joint CBF-QP。
- [ ] 用离线 settled counterfactual 只做诊断，不回灌训练；只有明确发现训练标签/坐标错位时才建立新 archive 和新 checkpoint。
- [ ] 先用单元测试覆盖 score 单位、词典序、tie/abstention、ledger state routing、候选 separation 和 CBF-only execution，再跑三 seed smoke。

### P4.2 修复后的验收门

- [ ] 三 seed selected-not-best 相对 v12 `31.7%/39.0%/52.4%` 均不恶化，且 Spearman/Kendall 不再系统为负。
- [ ] high-credit failure rate 不高于 low-credit bucket；低信用覆盖不足时标记 `insufficient_evidence`，不得判定通过。
- [ ] m3 相对 m0 至少 `2/3` seed 非负，平均 paired delta 不低于 `0`；safe-capture 是唯一主门。
- [ ] collision、boundary、pairwise、CBF timeout/infeasible/unverified、controlled-abort、raw-unverified 均逐项解释并保持安全硬门。
- [ ] 通过后才进入三 seed x 四变体 x 40/60 集；任一门失败则回到 P4.1，不打开 locked test。

### P4.3 下一版模型/数据增强工作包（仅在 P4.2 未通过时执行）

**设计决策：** 暂不更换为更大的世界模型或直接动作生成器。当前证据表明 interaction-aware、action-conditioned JEPA 能产生可审计的预测信号，但闭环收益尚未稳定；优先修复“预测信号到安全决策”的转换链路。

#### WP-1：辅助任务定义与标签合同

- [ ] 在同一 belief/action-chunk 输入上增加未来净空下界（obstacle/inter-agent）、visibility/observation-age risk、pairwise TTC、CBF correction magnitude 和 QP feasibility 头。
- [ ] 每个 head 明确标签单位、预测 horizon、缺失值策略和 offline-only 字段；target truth 只能离线生成标签，不能进入在线输入。
- [ ] 对 clearance 使用 calibrated q10 lower bound；可见性使用概率校准指标；TTC 使用截断/分桶标签，避免极端值主导损失。
- [ ] 记录每个 head 的 label coverage、finite rate、分桶样本数和 train/validation/calibration episode hash。

#### WP-2：困难片段重放（hard-segment replay）

- [ ] 从 development 失败索引提取困难上下文候选：`cbf_controlled_abort`、低净空、目标急转/S-curve、观测/通信陈旧、候选排序反转和高 CBF correction。
- [ ] 只将去重后的上下文摘要写入新的 train archive；不把 development/locked 的 settled outcome 或 target truth 直接回灌旧 archive。
- [ ] 固定 hard-context 权重上限（当前合同 `cap=8.0`），并预注册采样比例、去重规则、episode/layout/seed 隔离和重放次数。
- [ ] 训练三 seed 新 checkpoint，保留旧 v11/v12 checkpoint 只读对照；每个 run 记录 archive、代码、协议和环境 hash。

#### WP-3：候选动作块与动作条件建模

- [ ] 保持 `K=5` 候选语义：`nominal`、`intercept`、`lateral_clearance`、`formation_clearance`、`visibility_hold`。
- [ ] 每个候选生成长度为 `3` 的动力学可行 action chunk；每周期只执行第一个控制步，第二步前重新观测、更新 belief、重新生成候选并重新过 CBF。
- [ ] 对同一 belief 的五个候选做 latent/prediction separation 审计；若输出对动作不敏感，停止闭环接入并先修 action conditioning。
- [ ] 对候选加入 finite、速度/加速度/slew、reachability、目标区域和 obstacle/boundary 几何预检；nominal 永远作为 anchor。
- [ ] 禁止 JEPA 直接返回执行动作；ranker 只能返回候选索引和诊断，最终动作必须来自同一个 `Joint CBF-QP` 且 `verified=true`。

#### WP-4：多任务训练和选择规则

- [ ] 固定主位移/速度预测损失、净空 q10 损失、可见性风险损失、TTC/CBF 代价损失和动作条件 latent separation 损失的初始权重；只在新的 development block 内按预注册网格选择，不按单 seed 结果事后调权重。
- [ ] 以 validation 的安全头 coverage、calibration error、action separation 和 settled ranking 为预测门；不以 training loss 或 `mean_capture_time` 替代控制门。
- [ ] 新 checkpoint 先通过 P2 prediction gate、P3 calibration/ledger gate、P4 rank/replay gate，再运行三 seed x 20 smoke。
- [ ] 若新 revision 仍为 `prediction_signal_no_control_gain`，归档为安全预测基础设施，不继续扩大样本掩盖负结果。

#### WP-5：模型 revision 的最小验收矩阵

| 层级 | 必测项目 | 通过标准 |
|---|---|---|
| 预测 | MAE/Brier/ECE/AUROC、finite、按困难片段分桶 | 安全 head 有覆盖，主要 horizon 不劣于 constant-velocity；不宣称控制收益 |
| 动作条件 | 五候选 latent/prediction separation、候选有效率 | separation 非零且跨 seed 可复现；否则停止 |
| 可信度 | ledger credit、OOD/stale/non-finite/provenance fallback | 拒答路径可追溯，raw/unverified execution 为 `0` |
| 排序 | settled Spearman/Kendall、selected-not-best、abstention/switch | 不劣于 v12 基线，不能系统负相关 |
| 控制 | M3-M0 paired safe-capture delta | 至少 `2/3` seed 非负且平均 delta `>=0`；安全硬门全通过 |
| 可复现 | CPU/CUDA、双次 trace、100-cycle replay | candidate、verified action、结算和 trace hash 等价 |

## 9. P5：100-cycle rolling-horizon 与 Joint CBF 回归

- [ ] 对 zero-perturbation identity path 做逐字段回归。
- [ ] 对 non-zero candidate path 做至少 100 个 control cycles 的双次 deterministic replay。
- [ ] 验证每周期顺序固定为 observe -> belief -> candidates -> JEPA -> ledger -> rank -> CBF -> execute-first-step。
- [ ] 验证 action chunk 后续步骤绝不在 open-loop 中执行。
- [x] 注入 JEPA timeout、ledger timeout、non-finite prediction、CBF infeasible、CBF timeout、通信中断和 solver fault 的 fallback 语义。
- [x] 验证所有 fallback 仍走同一个 Joint CBF-QP，记录 status、slack、active constraints、correction norm、latency。
- [ ] 在 RTX 5050 和 CPU 上重放，candidate decision、verified action、safety settlement 和 trace hash 等价；当前因 abstention margin 浮点边界尚未通过。
- [x] 汇总 cycle p50/p95/p99、queue age、CBF p95；三 seed m3 cycle p95 约 `20.7 ms`，低于 `100 ms` contract。

### P5 出口门

- [x] collision/boundary/pairwise/raw-unverified 均为 `0`。
- [x] CBF timeout/infeasible 只产生显式 fallback 或 controlled abort，不返回 raw request。
- [ ] CPU/CUDA replay 结算一致；当前已观察到 candidate decision drift，必须在 P4 修复后重新审计。
- [ ] 尚未完成独立 100-cycle/500-cycle/1000-cycle 长序列压力回放，保留到 P8。

## 10. P6：三 seed、四变体、每 seed 20 集 smoke

### 固定矩阵

| 变体 | 定义 | 目的 |
|---|---|---|
| M0 | frozen nominal + CBF | 基线 |
| M3-new | JEPA + ledger + auxiliary safety rank + CBF | 主候选 |
| A1-new | JEPA + auxiliary rank，无 ledger + CBF | ledger 消融 |
| A2-new | JEPA + ledger，无 clearance/visibility rank + CBF | auxiliary-head 消融 |

- [x] 每个 training seed 使用 paired scene manifest；M0/M3/A1/A2 复用同一 canonical manifest。
- [x] 每个变体每 seed 运行 20 集；每个 output、TensorBoard 和报告目录独立创建。
- [x] 记录 episode、step trace、scene、provenance、checkpoint/ledger hash 和命令行。
- [x] 运行 safety、latency、ledger alignment、temporal ledger、settled ranking、failure index 和 deterministic replay。
- [x] 统计 per-seed `safe_capture`、paired delta、improved/degraded/tied、bootstrap CI、CBF abort、fallback 和 failure cause。

### P6 当前结果与硬门

- [x] 任一 collision、boundary、pairwise、raw-unverified 或不可解释安全回退均未出现。
- [x] M3-new 未通过跨 seed 控制收益门：平均 paired delta `-8.33 pp`，只有 `1/3` seed 非负；未把结果写成提升。
- [x] 240 个 episode 中 safe capture `114/240`，`cbf_controlled_abort=119`，timeout `7`；controlled abort 保留在失败分母。
- [x] 安全硬门通过但没有控制收益，当前分类为 `prediction_signal_no_control_gain`，回到 P4，不重复采样掩盖结果。
- [ ] 由于 CPU/CUDA 决定性尚未通过，P6 结果只能作为 development evidence，不能升级为 locked evidence。

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

1. [x] 补齐本计划 1.4 的 P0 preflight/hash manifest，并确认 v11 trainer/protocol 兼容性。
2. [x] 完成 hard-replay seed `20260912` 和 `20260913`。
3. [x] 审计三份 checkpoint、validation prediction、auxiliary heads 和 TensorBoard。
4. [x] 为三份 checkpoint 生成独立 calibration-bound ledger（旧版对照）。
5. [x] 冻结新的 safety-first ranking/abstention/hysteresis protocol。
6. [x] 完成 CBF fault、latency 和当前 trace deterministic replay；[ ] CPU/CUDA regression 的最终等价门仍未通过。
7. [x] 运行三 seed x 四变体 x 20 集 smoke，并生成 aggregate/failure index。
8. [ ] 先修复 abstention margin 的跨设备浮点边界，完成 seed `20260911`，再完成 `20260912/20260913` 的 CPU/CUDA replay；`v12_r3` ledger 已就绪。
9. [ ] 修复 settled ranking mismatch、候选分离不足和 controlled-abort 推进不足；重新通过 P4 ranking/ledger gates。
10. [ ] 只有修复后 smoke 的 M3 至少 2/3 seed 非负且安全硬门全通过，才运行三 seed x 四变体 x 40/60 集 paired development。
11. [ ] 完成 100/500/1000-cycle stress、robustness/SIL/HIL readiness，再决定是否创建新的 locked preregistration。

## 16. 禁止事项和停止规则

- 不把单 seed、smoke、prediction MAE 或 mean capture time 写成正式控制收益。
- 不以 `95%` 作为绝对成功门，也不以捕获时间抵消 safe-capture 失败。
- 不降低 CBF margin、关闭 stale/OOD 检查或放宽回退路径来追逐分数。
- 不把 `controlled_abort` 从失败分母中删除。
- 不将 archive-recovery checkpoint 冒充历史 V4 warm-start checkpoint。
- 不在 locked protocol 冻结前查看 locked split；本计划全程保持 `locked_test_opened=false`。

## 17. 从当前状态开始的命令级执行顺序

以下顺序是下一次实际运行的唯一入口。所有 `<...>` 参数必须替换为本次新 revision 的绝对路径；不得复用已有输出目录。

### 17.1 环境和协议预检

```powershell
# 正式环境安装完成后优先使用它；当前网络阻塞时使用已验证的 GPU 环境。
conda activate traj_pred_prep
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
$env:PYTHONPATH='src;scripts'
python scripts/verify_jepa_safe_capture_protocol.py `
  --protocol configs/central_random_mixed_obstacle_s3_v5_v12_calibrated_clearance_development_protocol.yaml `
  --project-root . `
  --output results/jepa_safe_capture_v12_r3_preflight/preflight.json
python -m pytest -q tests/test_clearance_calibration.py tests/test_jepa_safe_capture_candidates.py
```

预检失败、CUDA 不可用或协议输出包含 `locked_test_opened=true` 时立即停止；不要在 base 环境中继续跑实验。正式环境恢复后，再用 `conda env create -f environment.yml` 创建 `uav-encirclement-gpu`，并重新执行相同预检；环境安装失败不改变实验协议，也不能用未记录的依赖继续产生正式证据。

### 17.2 修复并证明跨设备决定性

1. 在 ranker 中实现固定小数/整数化的 margin 比较或预注册安全 band。
2. 运行 margin 边界单元测试，至少覆盖 `0.0019779`、`0.0020000`、`0.0020076`。
3. 对 seed `20260911` 先生成 CUDA/CPU 成对目录，再运行：

```powershell
python scripts/audit_jepa_safe_capture_device_replay.py `
  --cuda-run results/<cuda_run> `
  --cpu-run results/<cpu_run> `
  --output-dir results/jepa_safe_capture_v12_r3_device_replay_seed20260911 `
  --tensorboard-logdir results/jepa_safe_capture_v12_r3_tensorboard/seed20260911/device_replay `
  --development-only
```

4. 只有 seed `20260911` 逐字段一致后，才对 `20260912`、`20260913` 重复；任一 seed drift 都回到 ranker 修复。

### 17.3 排序和困难片段分析

- [ ] 运行三 seed settled counterfactual audit，输出每候选未归一化分数、单位、ledger state、calibrated clearance、TTC、visibility、CBF cost 和最终选择。
- [ ] 生成 `selected_not_best`、Spearman/Kendall、top-1 safety precision、abstention、switch/oscillation、candidate separation 的分桶报告。
- [ ] 对 `cbf_controlled_abort`、低净空、目标急转、stale/dropout、候选反转片段建立 hard-segment index；只保留可回放的 belief/action 上下文摘要。
- [ ] 若问题来自标签或坐标错位，建立新 archive；否则只修改 ranker/ledger 规则，不重新解释旧结果。

### 17.4 新 revision 训练和 ledger

- [ ] 若 P4.2 仍失败，按 P4.3 WP-1/WP-4 固定辅助头和 hard-replay 采样，使用 `scripts/train_jepa_safe_capture_v3.py` 训练三个新 seed。
- [ ] 每个 seed 独立输出 checkpoint、history、metadata 和 TensorBoard；训练参数、hard weight cap、quantile 和 loss weights 写入 metadata。
- [ ] 运行 `scripts/build_jepa_safe_capture_v12_calibrated_ledger.py` 生成新 checkpoint-bound ledger；缺 calibration/hash/finite 证据时必须拒绝发布。
- [ ] 训练和 ledger 只允许 train/calibration split；development/locked 结果不得参与梯度或在线 credit 更新。

### 17.5 回归、smoke 和扩展

- [ ] 先做 100-cycle rolling replay、fault matrix、latency 和 deterministic replay；全部通过后再做三 seed x 四变体 x 20 smoke。
- [ ] smoke 统计 `safe_capture`、paired delta、controlled abort、fallback、collision/boundary/pairwise、raw-unverified 和 minimum clearance；`mean_capture_time` 只作诊断。
- [ ] 仅当 M3 至少 2/3 seed 非负、平均 paired delta `>=0`、排序不劣化、CPU/CUDA 等价且安全硬门全通过时，才执行每 seed 40/60 集。
- [ ] 40/60 集仍属于 development；P8 stress 和 SIL/HIL 完成前不得创建新的 locked test。

### 17.6 每次阶段结束的归档动作

- [ ] 写入 `summary.json`、`run_metadata.json`、`episodes.csv`、逐步 trace、failure index、TensorBoard event 和 SHA-256 manifest。
- [ ] 运行 `git diff --check` 和相关测试，确认 Markdown/JSON/CSV/TensorBoard 数值一致。
- [ ] 只提交本阶段源代码、测试、配置和计划文件；不提交或删除 `tmp/`，不覆盖旧结果。

**最终判断标准：** 只有当 JEPA 的候选反事实排序、ledger 的可信度拒答、Joint CBF-QP 的硬安全边界和 rolling-horizon 的闭环重规划，在三 seed 困难场景中共同通过上述审计，才能把系统称为“安全增强的闭环围捕系统”。否则应明确归类为安全基础设施、预测信号或 development-only 负结果。
