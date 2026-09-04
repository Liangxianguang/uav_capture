# 无人机集群安全围捕：下一步详细 TODO 与目标计划书

**版本：** v1.0（P11 rank guard 之后执行版）
**日期：** 2026-09-04
**执行目录：** `D:\\uav-capture\\uav_capture`
**硬件：** NVIDIA RTX 5050，Conda 环境 `uav-encirclement-gpu`
**实验范围：** `development_only=true`，`locked_test_opened=false`
**第一指标：** `safe_capture`
**次要指标：** CBF abort/fallback、碰撞与越界诊断、净空、延迟、路径代价、`mean_capture_time`

> 本计划的目标是完成一套可审计的安全闭环，而不是追逐某个 seed 的最高捕获率。`safe_capture` 必须先满足安全合同；平均捕获时间不能抵消碰撞、边界越界、机间净空破坏、CBF 失败或未验证动作。当前结果只允许归类为 development evidence，不能直接写成新的正式提升。

## 1. 最终目标

面向无人机集群对抗围捕/拦截任务，完成以下闭环：

```text
多机观测与通信历史
    -> interaction-aware belief state
    -> 传统规划器生成动力学可行候选 action chunks
    -> action-conditioned JEPA 反事实预测
    -> reliability ledger 可信度校验与拒答
    -> 安全优先候选排序、滞回与保守 abstention
    -> Joint CBF-QP 安全过滤
    -> 只执行第一控制步
    -> 重新观测、更新 belief、重新规划
```

### 1.1 必须证明的系统性质

- JEPA 是候选轨迹评价器，不直接生成最终控制动作。
- JEPA 同时利用候选 action 和多机交互上下文，能够区分目标运动、可见性、净空和 CBF 干预风险。
- Reliability ledger 在 OOD、stale observation、预测漂移、non-finite、信用不足或候选分离消失时显式 abstain。
- 所有 candidate、nominal、safe-hold 和 fallback 都经过同一个 Joint CBF-QP。
- QP infeasible、timeout 或 unverified 时绝不执行 raw desired action。
- 每个 action chunk 只执行第一步，下一控制周期重新预测和过滤，避免长 rollout 漂移。
- 三 seed、同一 paired scene manifest、完整 provenance 下，系统在 `safe_capture` 上不劣于冻结 nominal + CBF；若有提升，必须有配对和逐 seed 证据。

### 1.2 当前事实快照

| 项目 | 当前状态 | 解释 |
|---|---|---|
| P10 多任务预测/校准 | 已完成 | 三 seed、40 epoch provenance audit 通过；输出 finite，主要 horizon 优于 constant-velocity；安全相关头已完成独立 calibration |
| P11 rank guard | 已实现并完成本阶段提交 | 已加入 top-two abstention、预测净空下界、candidate hysteresis、minimum hold |
| P11 smoke | 安全语义审计通过 | M0 与 M3 均为 `10/20=50.0% safe_capture`；collision/boundary/pairwise=0；CBF timeout=0；CBF controlled abort=10；raw-unverified=0 |
| P11 smoke 解释 | 未证明任务改善 | M3 平均 CBF correction 约 `0.27`，M0 约 `0.94`，但 safe-capture 没有改善；不能把修正量下降写成性能提升 |
| 代码状态 | 工作区有未提交改动 | 只能选择性提交 P11 文件，不得整体 `git add .` 或清理用户的 E1/V5/tmp 改动 |

## 2. 不可变合同

### 2.1 数据边界

- 在线输入只允许使用 defender 状态、target belief、观测/通信历史、障碍几何、边界、动作历史和时间戳年龄。
- target ground truth 只能用于离线 settled label 和 episode 结算，并明确标记 `offline_only=true`。
- train、validation、calibration、development 和 locked-test episode/layout/seed 必须隔离。
- development 失败片段不能直接回灌旧训练 archive；如需重训，必须创建新的 archive、protocol、checkpoint 和 hash。
- 每个结果目录必须保存代码 revision、环境、命令、protocol、scene manifest、actor checkpoint、JEPA checkpoint、ledger、CBF 配置及其 SHA-256。

### 2.2 候选动作合同

- 候选数固定 `K=5`：`nominal`、`intercept`、`lateral_clearance`、`formation_clearance`、`visibility_hold`。
- action chunk 固定为 3 个 control steps；只执行第 1 步后 replan。
- 候选进入 JEPA 前通过 finite、shape、speed、acceleration、slew 和 reachability 检查。
- 不可达 candidate 不得进入模型排序，并写入 `candidate_rejection_reasons`。
- ranking 使用冻结的 `score_tie_tolerance_m=5e-4`；修改 tie policy 必须创建新 protocol。

### 2.3 安全合同

一个 episode 只有同时满足以下条件才算 `safe_capture=true`：

1. 至少一个 defender 在时间限制内进入目标 `0.80 m` capture radius；
2. 无 obstacle、target 或 defender-defender collision；
3. 无 defender boundary/altitude violation；
4. 无 pairwise separation violation；
5. 无 CBF infeasible、timeout、unverified action 或 controlled-abort 终止。

CBF 回退顺序固定为：

```text
separation-preserving safe-hold
    -> verified nominal through Joint CBF-QP
    -> controlled abort
```

`controlled_abort` 是安全失败，不是安全成功；但它也不能被误记为 `raw_unverified_executed`。报告必须分别统计：

- `raw_unverified_executed`：实际执行了未验证动作，硬门必须为 0；
- `cbf_controlled_abort_steps`：未执行 raw 动作但无法验证安全动作，计为安全失败；
- `cbf_timeout_steps`、`cbf_infeasible_steps`：故障诊断和回退覆盖率。

## 3. 目标架构与接口

### 3.1 BeliefState

每个控制周期构造不可变快照，至少包含：

- 多机位置、速度、队形几何和 pairwise TTC；
- target belief 位置/速度、visibility、observation age、message age；
- obstacle/boundary/altitude 净空；
- 最近 `H` 个 observation、action 和 communication token；
- 上一周期 CBF status、最小 slack、修正范数、fallback mode 和 solver latency；
- layout signature、target motion mode 和 provenance id。

### 3.2 JEPA 评价输出

对每个 candidate action chunk 输出未来评价向量，而不是控制量：

```text
target displacement / velocity / acceleration
obstacle-clearance lower quantile
inter-agent clearance lower quantile
pairwise TTC
visibility probability / observation-age risk
CBF intervention probability / correction magnitude / QP feasibility
predictive uncertainty / candidate disagreement
```

建议的安全优先 score：

```text
score(k) = task_progress
         + visibility_gain
         - clearance_lower_quantile_risk
         - pairwise_ttc_risk
         - cbf_intervention_cost
         - uncertainty_penalty
         - action_change_cost
         - nominal_anchor_penalty
```

预测净空、TTC 和 QP feasibility 只用于排序、ledger 和诊断，不能替代真实几何 CBF。

### 3.3 Reliability ledger

ledger 在 calibration 后只读，并绑定 checkpoint、calibration archive 和 protocol hash：

| 状态 | 触发条件 | 允许行为 |
|---|---|---|
| `trusted` | bucket 样本足够、credit 足够、uncertainty/stale age 在阈值内 | 允许 JEPA 候选排序 |
| `fallback_nominal` | 低 credit、bucket 缺失、候选分离消失、预测漂移 | nominal，经 CBF 过滤 |
| `safe_hold` | OOD、non-finite、过期观测、连续失败或 provenance 不一致 | safe-hold，经 CBF 过滤 |
| `controlled_abort` | safe-hold 和 nominal-CBF 都无法验证 | 终止并记录，不能计 safe capture |

credit 是可信度信号，不是安全证明。低 credit 必须改变执行路径，不能只在 score 中减一个小惩罚。

## 4. 分阶段执行路线

```text
T0 运行前冻结与工作区审计
  -> T1 P11 smoke audit 与 raw/unverified 语义闭合
  -> T2 settled counterfactual 排序诊断
  -> T3 reliability ledger temporal/adversarial 重校准
  -> T4 净空/可见性辅助预测与困难片段重放
  -> T5 候选 action chunk 与 ranker 冻结
  -> T6 rolling-horizon + Joint CBF 回归
  -> T7 A1/A2 smoke 与新 protocol gate
  -> T8 三 seed paired development
  -> T9 robustness stress、SIL/HIL readiness
```

未通过上游 gate，不得启动下游全量实验；任何失败都要保留产物并新建 protocol revision，不能通过删 episode 或事后调参消除。

## 5. T0：运行前冻结与预检

**目标：** 防止当前未提交改动、场景 manifest 或阈值变化污染后续结果。

### TODO

- [x] 确认 Conda 环境、PyTorch/CUDA 和 RTX 5050 可见。
- [x] 确认所有运行使用 `phase=development_only`、`locked_test_opened=false`，split 只能是 validation/development。
- [x] 固定新 protocol、scene manifest、actor/JEPA/ledger/CBF 配置和 score 权重。
- [x] 记录代码 revision、环境包清单和全部输入 SHA-256。
- [x] 为本轮建立空的独立 output root 和 TensorBoard root，拒绝覆盖非空目录。
- [x] 检查 `git diff --check`；忽略无关的 E1/V5/tmp 改动。
- [x] 运行完整测试前先运行与本阶段相关的 targeted tests。

### 预检命令

```powershell
Set-Location D:\\uav-capture\\uav_capture
$py = 'D:\\miniconda3\\envs\\uav-encirclement-gpu\\python.exe'
$env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"

& $py -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
& $py scripts/verify_jepa_safe_capture_protocol.py --protocol configs/central_random_mixed_obstacle_s3_v5_rank_guard_development_protocol.yaml
& $py -m pytest -q tests/test_jepa_safe_capture_candidates.py tests/test_audit_jepa_safe_capture_v4_calibration.py
git diff --check
git status --short
```

### T0 出口

- [ ] preflight manifest 写出环境、命令、hash 和 locked-test 状态。
- [ ] protocol 校验通过；否则停止，不运行任何 episode。

## 6. T1：P11 rank guard smoke audit

**目标：** 先证明 rank guard 的安全和可观测性，再判断它是否值得进入 A1/A2 或三 seed block。

### TODO

- [x] 对已有 M0/M3 smoke 运行 `audit_jepa_safe_capture_v5_rank_guard_smoke.py`。
- [x] 核对同一 training seed、同一 episode index、同一 scene hash 和同一 actor/JEPA/ledger hash。
- [x] 核对五个 candidate、3-step chunk、top-two margin、abstention reason、hysteresis、hold state 均出现在 trace。
- [x] 明确统计 `cbf_unverified_steps` 与 `cbf_controlled_abort_steps` 的含义，确认 controlled abort 不被误报为 raw unverified action。
- [x] evaluator 增加 `raw_unverified_executed` 计数并重跑 smoke。
- [x] 运行 rank guard 单测、audit 单测和 deterministic replay。

### 当前 smoke 的解释规则

- M0 和 M3 都为 `50.0% safe_capture`：rank guard 暂未显示任务收益。
- 两者 collision/boundary/pairwise=0：安全几何约束没有回归。
- M3 correction norm 较低：只能作为执行代价诊断，不能等同于更高 safe capture。
- 若所有 gates 通过，结论为“安全保持、任务收益未证明”，不是“模型提升”。

### T1 出口

- [x] `all_gates_pass=true`。
- [x] `raw_unverified_executed=0`，所有动作均有 CBF verification 或明确 controlled abort。
- [x] trace 能逐步解释选择、拒答、回退和终止。
- [x] 生成 `docs/JEPA_SAFE_CAPTURE_P11_RANK_GUARD_SMOKE_*.md`、audit JSON、TensorBoard 和 hash manifest。

## 7. T2：Settled counterfactual 排序诊断

**目标：** 找出 score 与最终 safe-capture/CBF outcome 失配的原因，避免凭单个 seed 调权重。

### TODO

- [ ] 对历史 degraded/improved/tied episode 做只读 settled counterfactual replay。
- [ ] 为每个 decision 记录五个 candidate 的 settled target progress、capture outcome、CBF correction、minimum clearance、visibility 和 termination outcome。
- [ ] 计算 top-1 precision/recall、Spearman/Kendall rank correlation、Brier/ECE、selected-not-best、top-two margin 和 switch rate。
- [ ] 统计高 credit 失败、低 credit 失败、margin 不足、预测净空过度乐观和 visibility 误判的分桶率。
- [ ] 只在独立 calibration evidence 支持时调整 score；每次调整创建新 protocol 和新 ledger/calibration manifest。
- [ ] 保留一个 nominal anchor，避免所有候选在高不确定性时远离可解释基线。

### T2 停止规则

- [ ] 如果排序因果仍无法从 trace 证明，标记 `unresolved`，不继续加大模型或扩大候选数。
- [ ] 如果 rank guard 只降低 CBF correction 而不改善 settled safe outcome，归档为执行代价变化，不写成任务收益。

## 8. T3：Reliability ledger temporal/adversarial 重校准

**目标：** 让 ledger 在预测漂移前拒答，并把控制权交给确定性的 nominal/safe-hold CBF 路径。

### 校准场景

- [ ] target 急转、速度突变、flee persistence、S-curve 和突发加速度；
- [ ] 遮挡、detection dropout、message delay、message dropout 和 observation age 增长；
- [ ] 障碍数量/密度 shift、队形拥挤、pairwise TTC 变小；
- [ ] non-finite 输入、checkpoint/ledger hash 不一致和候选分离消失；
- [ ] 连续预测残差增大、连续 credit 下降和高 credit settled failure。

### 固定规则

- [ ] 固定 `minimum_sample_count`、`minimum_credit`、uncertainty 上限、stale age 上限、OOD 规则和 risk threshold。
- [ ] 固定 credit decay/recovery、abstention hysteresis 和状态转移优先级。
- [ ] 校准结束后将 ledger 设为只读，运行期间禁止在线更新 threshold、credit 或 checkpoint。
- [ ] 每次决策写入 ledger key、state、credit、uncertainty、observation age、reason code、fallback mode 和 trace hash。

### T3 出口

- [ ] OOD/stale/non-finite/低 credit 均触发 100% 显式回退。
- [ ] high-credit settled failure rate 不高于 low-credit bucket；否则不能称 ledger 提升了可靠性。
- [ ] 每次 fallback 可由单条 trace 重放，`raw_unverified_executed=0`。
- [ ] 生成 ledger calibration report、bucket coverage、fault-injection matrix 和 TensorBoard。

## 9. T4：安全辅助预测与困难片段重放

**目标：** 把 JEPA 从“只预测目标位移”提升为可用于安全排序的交互预测器，同时防止困难样本污染评估。

### TODO

- [ ] 保留 target displacement 主头，并验证 velocity/acceleration consistency。
- [ ] 校准 obstacle/inter-agent clearance lower quantile、pairwise TTC、visibility、observation-age、CBF intervention 和 QP feasibility heads。
- [ ] 对每个 horizon 计算 MAE、P50/P90/P95、coverage、underestimation rate、Brier、ECE 和极端风险漏报率。
- [ ] 检查同一 belief 下五个 candidate 的 latent/action-following separation；若接近随机或塌缩，先修 action-conditioned representation。
- [ ] 将 motion-mode embedding（constant velocity、flee、turn、S-curve、突变）作为可选新 protocol 变更，不在当前 block 中隐式切换。
- [ ] 从 failure index 选取遮挡、漂移、CBF 修正过大、候选振荡、timeout 和 stale 各类代表样本，双次 deterministic replay。
- [ ] 无法从 trace 证明的原因标记 `unresolved`，不得凭经验补标签。

### T4 训练约束

- [ ] 若需要困难片段重训，建立新 archive/checkpoint/hash；不得改写历史 V4/V5 archive。
- [ ] 三 training seed 独立训练，记录每个任务 loss、梯度/激活统计、校准曲线和 provenance text。
- [ ] 只有当所有输出 finite、主要 horizon 优于 constant-velocity、辅助标签非空且校准可用时，checkpoint 才能进入闭环。

## 10. T5：候选动作块和 ranker 冻结

**目标：** 将 JEPA 的预测信号转化为可解释、可达、可安全过滤的候选排序。

### 候选库

- [ ] 保持当前五类候选作为主比较合同，不在 smoke 中增加候选数。
- [ ] `nominal` 保持 exact anchor；其余候选覆盖 intercept、侧向净空、队形净空和 visibility hold。
- [ ] 每个候选记录生成参数、动力学可达性、speed/slew projection、nominal distance 和 rejection reasons。
- [ ] 任何候选 invalid/reachability fail 均在进入 JEPA 前剔除。

### 排序和稳定性

- [ ] 先做 finite/reachability/预测安全下界/ledger 状态筛选，再比较 task progress。
- [ ] 固定 clearance lower quantile、visibility gain、CBF intervention cost、uncertainty、action-change 和 nominal-anchor 项。
- [ ] 固定 top-two abstention margin、minimum predicted clearance、candidate hysteresis 和 minimum hold steps。
- [ ] 记录 selected index、top-two margin、switch rate、oscillation length、hysteresis、hold、fallback 和最终 CBF action。
- [ ] 排序输出不能覆盖 CBF filtered action；ranker 只能提供 desired candidate。

### T5 出口

- [ ] ranker 单测、tie policy 测试、CPU/CUDA settled safety replay 通过。
- [ ] `selected-not-best`、high-credit failure 和 oscillation 有可解释的变化；没有证据时保留负结果。
- [ ] 新权重、新 margin 或新 hold 参数已写入新 protocol 并冻结。

## 11. T6：Rolling-horizon 与 Joint CBF 端到端回归

**目标：** 验证系统每个周期真正闭环，而不是只在 episode 末端依赖一次预测。

### TODO

- [ ] 验证固定顺序：timestamp check -> belief update -> candidate generation -> reachability -> JEPA batch -> ledger -> ranking -> Joint CBF -> execute first step。
- [ ] 注入 QP infeasible、timeout、non-finite request、stale observation、通信中断、多约束激活和单机状态异常。
- [ ] 验证 safe-hold 与 nominal fallback 均通过同一个 Joint CBF-QP。
- [ ] 做 zero-perturbation regression：关闭 JEPA 后，非 JEPA 字段与 M0 逐字段一致。
- [ ] 做至少 100 个随机 control cycle 的 raw/unverified assertion 和 deterministic trace replay。
- [ ] 记录 solver status、active constraints、minimum slack、correction norm、fallback reason、CBF latency 和全链路 latency。

### T6 硬门

- [ ] 所有实际执行动作 finite 且有 CBF verification；`raw_unverified_executed=0`。
- [ ] timeout 要么为 0，要么每次都有可验证 fallback；controlled abort 单独计入安全失败。
- [ ] JEPA + ledger + ranker + CBF 端到端 p95 latency 不超过 100 ms。
- [ ] candidate、nominal、safe-hold、fallback 接口和 trace schema 测试全部通过。

## 12. T7：新 protocol smoke

**目标：** 在扩大到三 seed 前，以同一 manifest 验证主方法和关键消融。

### 实验矩阵

| 变体 | 执行栈 | 用途 |
|---|---|---|
| M0 | nominal planner + Joint CBF-QP | 冻结安全基线 |
| M3 | JEPA + reliability ledger + auxiliary safety ranking + CBF | 主方法 |
| A1 | M3 去除 ledger，保留 CBF | ledger 消融 |
| A2 | M3 去除 clearance/visibility ranking，保留 CBF | 安全辅助排序消融 |
| A3 | raw/no-CBF | 仅诊断 CBF 必要性，不进入安全结论 |

### TODO

- [ ] 为本轮生成全新 scene manifest，train/calibration/development 不重叠。
- [ ] 每个主变体先运行每 seed 20 集；同一 manifest、episode index、layout、target motion 和 observation schedule 配对。
- [ ] A3 必须标记 `diagnostic_only=true`，不得作为可部署候选或主统计样本。
- [ ] 每次 smoke 后立即运行 aggregate、failure index、rank audit、ledger audit、CBF safety audit 和 TensorBoard audit。
- [ ] 任一安全硬门、provenance 门或 latency 门失败，停止扩展并保存 failure memo。

### T7 准入门

- [ ] M0/M3/A1/A2 的 collision、defender boundary、pairwise violation 为 0。
- [ ] 所有实际执行动作 raw/unverified 为 0；CBF timeout=0 或有已验证 fallback。
- [ ] scene pairing、zero-perturbation、trace 完整性和 hash 一致性通过。
- [ ] safe-capture 没有因 rank guard 引入明显回归；若没有改善，归档为安全保持但任务收益未证明。

### T7 当前预演记录

已完成 `training_seed=20260911` 的同 manifest 20 集预演，详见 `docs/JEPA_SAFE_CAPTURE_P11_ABLATION_SMOKE_20260904.md`：M0/M3/A1/A2 的几何安全门和 raw-action gate 均通过，M3 与 M0 持平，A1 为 `-5.0 pp`，A2 持平。由于尚未覆盖三个 training seed，本节不能勾选 T7 全量准入，也不能启动 T8 final block。

## 13. T8：三 seed paired development

**前置条件：** T0-T7 全部通过；protocol、scene manifest、score、ledger、CBF 和设备冻结后才可执行。

### 固定设置

- [ ] training seed 固定为 `20260911`、`20260912`、`20260913`。
- [ ] 每个变体每个 seed 至少 40 个 paired validation episodes，总计至少 120 对主比较 episode。
- [ ] 运行顺序固定：M0 -> M3 -> A1 -> A2；A3 另行诊断。
- [ ] 使用 RTX 5050 CUDA 作为主结果设备；CPU 只做安全结算/确定性审计，不混入任务率主结论。
- [ ] 每个 run 保存 `summary.json`、`episodes.csv`、`step_traces/`、scene manifest、provenance、TensorBoard 和 SHA-256。

### 统计规则

- [ ] 统计独立单位是 `(training_seed, episode_index)`，不能把 timestep、candidate 或 chunk 当独立样本。
- [ ] 主指标按 safe-capture 报告逐 seed rate、样本 SD、paired delta、improved/degraded/tied、bootstrap 95% CI 和 exact McNemar。
- [ ] 安全诊断分别报告 collision、boundary、pairwise、CBF timeout/infeasible、controlled abort、fallback、raw/unverified 和 latency。
- [ ] 按 motion mode、visibility、observation age、clearance、ledger state、CBF active constraint 和 candidate outcome 分桶。
- [ ] `mean_capture_time` 只能作为次要诊断；不可用于掩盖 safe-capture 下降。

### 预定义结论

| 结论标签 | 条件 | 允许表述 |
|---|---|---|
| `safe_capture_improvement_candidate` | 安全、可靠性、实时性、provenance 全通过；paired delta 非负且至少 2/3 seed 非负 | development 中存在可复现正向证据，仍不等于 locked-test 结论 |
| `safety_preserving_noninferior` | 安全无退化，但任务提升无法确认 | 架构保持安全且任务不劣，需继续优化排序/预测 |
| `prediction_signal_no_control_gain` | 离线预测/校准改善，但闭环 safe-capture 中性或负向 | 预测信号有效，但尚未转化为控制收益 |
| `rejected_for_safety` | 新增碰撞、越界、pairwise violation、raw action 或不可审计回退 | 停止该变体并归档失败 |
| `insufficient_evidence_do_not_open_locked_test` | seed、配对、统计或 provenance 不完整 | 不打开 locked test |

不设置“必须达到 95%”的硬目标；首要判断是安全不劣性、长尾失败率和可解释性。

## 14. T9：Robustness stress 与 SIL/HIL 准备

只有 T8 通过安全硬门后才执行：

- [ ] 运行独立 hard block，覆盖延迟、遮挡、通信丢包、目标急转、密度 shift、拥挤队形和单机失效。
- [ ] 测量 p50/p95/p99 的 JEPA、ledger、ranker、CBF 和全链路延迟、显存、CPU 占用和消息队列积压。
- [ ] 注入传感器冻结、进程重启、watchdog、通信中断和 checkpoint/hash mismatch。
- [ ] 验证 safe-hold、verified nominal、controlled abort、急停和恢复策略。
- [ ] 形成 SIL/HIL 报告、风险清单和部署前检查表；HIL 通过不等于真实飞行许可。
- [ ] 未完成安全审查前禁止真实无人机飞行测试。

## 15. TensorBoard、Git 与 provenance 纪律

### TensorBoard 必须包含

```text
Loss/target, Loss/velocity, Loss/acceleration, Loss/clearance,
Loss/visibility, Loss/cbf_risk, Loss/action_consistency,
Calibration/*, Reliability/*, Ranking/*, CBF/*, Safety/*,
Latency/*, Fallback/*, Provenance/*
```

- [ ] 每个 training seed 使用独立 logdir，训练至少记录 40 个 epoch 点。
- [ ] 每个 audit 写入 command、environment、protocol/checkpoint/ledger/scene hash 和 locked-test 状态。
- [ ] 必需 tag、事件文件或 provenance 缺失时，结果不得进入 paired block。

### 选择性提交顺序

```text
P11: feat(jepa): add settled rank guard
P12: feat(jepa): recalibrate reliability ledger
P13: test(jepa): verify rolling horizon safety contract
P14: docs(jepa): archive development smoke
P15: docs(jepa): archive paired development block
```

- [ ] 每个阶段只 `git add` 本阶段明确文件；不使用 `git add .`。
- [ ] commit 后记录 commit hash 和 push 状态；网络失败时保留本地 commit 并写入报告。
- [ ] 不覆盖历史 V4/V5 结果，不把 tmp 中 archive-recovery checkpoint 冒充历史 warm-start checkpoint。

## 16. 时间盒

| 时间盒 | 工作 | 交付物 |
|---|---|---|
| Day 0 | T0 冻结/预检 | preflight、protocol、hash manifest |
| Day 1 | T1 P11 smoke audit | audit JSON/MD、TensorBoard、raw/unverified 语义报告 |
| Day 2-3 | T2 settled ranking | counterfactual labels、rank audit、failure memo |
| Day 3-5 | T3 ledger 校准 | ledger、OOD/stale/adversarial calibration report |
| Day 4-6 | T4 辅助预测/困难重放 | prediction calibration、replay JSONL、训练 provenance |
| Day 6-7 | T5 ranker 冻结 | protocol revision、ranker tests、trace schema |
| Day 7-8 | T6 rolling/CBF 回归 | safety/fallback/latency audit |
| Day 9 | T7 smoke | M0/M3/A1/A2 20 集/seed gate |
| Day 10-14 | T8 paired development | 3 seed x 40 episodes x fixed variants |
| Day 15+ | T9 stress/SIL/HIL | robustness report、deployment readiness |

时间盒可以因训练或硬件耗时顺延，但不能跳过依赖和硬门。

## 17. Definition of Done

本轮只有满足以下全部条件才算完成：

1. JEPA、ledger、ranker、Joint CBF-QP 和 rolling-horizon 接口均有代码、测试和逐步 trace。
2. P11 smoke 的安全、raw/unverified、pairing、trace 和 TensorBoard audit 全部通过。
3. settled counterfactual 能解释排序失配；无法证明的原因明确标记 `unresolved`。
4. ledger 对 OOD、stale、non-finite、低 credit 和预测漂移确定性拒答，并保持只读绑定。
5. 每个控制周期只执行第一步，所有 candidate/nominal/fallback 都经过 Joint CBF-QP。
6. T7 smoke 通过安全、zero-perturbation、provenance 和 latency 硬门。
7. T8 三 seed paired development 完整运行，并以 `safe_capture` 为第一指标归档统计。
8. 所有结果保持 `development_only=true`、`locked_test_opened=false`；没有用户单独授权不得打开新的 locked test。

最终能够严谨主张的系统定义是：**JEPA 负责 action-conditioned interaction-aware 反事实候选评价，reliability ledger 负责可信度和拒答，Joint CBF-QP 负责不可绕过的安全边界，rolling horizon 负责闭环修正；是否存在任务收益只能由多 seed safe-capture 证据决定。**
