# 无人机集群对抗围捕安全增强系统
# 当前执行版 TODO 与目标计划书

**系统目标：** `Interaction-Aware Action-Conditioned JEPA + Reliability Ledger + Joint CBF-QP + Rolling Horizon`  
**执行目录：** `D:\\uav-capture\\uav_capture`  
**硬件：** NVIDIA GeForce RTX 5050  
**当前阶段：** V20 development-only 诊断；尚未进入新的 40/60 集开发块，也未打开 locked test  
**版本：** 2026-09-05，执行路线 V21 起草前的入口文件

> 这是后续实现和实验的主 TODO，不是新的实验结果。当前唯一主指标是 `safe_capture`。`mean_capture_time`、Transit、路径长度、CBF 修正量和延迟只能作为诊断指标，不能抵消安全捕获失败。绝对 `95%` 不作为本轮硬目标。

## 1. 最终目标

构建一条可以逐周期审计、拒答和回退的无人机集群围捕闭环：

```text
多机观测/通信历史
  -> immutable BeliefState
  -> 传统规划器生成动力学可行候选 action chunks
  -> action-conditioned interaction-aware JEPA 反事实评价
  -> immutable Reliability Ledger 可信度校验/拒答
  -> safety-first ranker + nominal anchor + hysteresis
  -> Joint CBF-QP 最终安全过滤
  -> 只执行第一控制步
  -> 重新观测、更新 belief、重新规划
```

系统要解决的不是“让世界模型直接输出动作”，而是：

1. 用 JEPA 比较候选轨迹在目标交互、净空、可见性和安全风险上的后果。
2. 用 ledger 在分布外、陈旧观测、预测不确定性上升或 provenance 不一致时拒绝模型排序。
3. 用同一个 Joint CBF-QP 对 nominal、JEPA 选择、safe-hold 和 fallback 做最终验证。
4. 用 rolling horizon 把长 rollout 拆成短候选块，每次只执行第一步并立即重规划。
5. 用逐步 trace 证明“模型只评价、CBF 才执行、安全失败不会被隐藏”。

## 2. 当前基线和阻塞结论

### 2.1 V20 三 seed device audit

协议：`configs/central_random_mixed_obstacle_s3_v5_v20_cpu_deterministic_development_protocol.yaml`  
协议 SHA-256：`b8a492faa9448bb0917c124908a044af6cb10813847afeedb78ce675446a2b99`  
训练 seed：`20260911`、`20260912`、`20260913`

| seed | safe capture | steps | collision | boundary | pairwise | raw/unverified |
|---:|---:|---:|---:|---:|---:|---:|
| 20260911 | 9/20 = 45% | 1167 | 0 | 0 | 0 | 0 |
| 20260912 | 7/20 = 35% | 935 | 0 | 0 | 0 | 0 |
| 20260913 | 9/20 = 45% | 1133 | 0 | 0 | 0 | 0 |

三 seed 平均 `safe_capture=41.67%`，sample SD `5.77%`。CPU/CUDA 的 decision、CBF 状态和安全结算逐字段一致；这证明设备决定性和执行安全合同，不证明 JEPA 带来任务收益。

### 2.2 M0/M3 paired aggregate

- M0：每个 seed `10/20=50%`。
- M3：`45% / 35% / 45%`，合计 `25/60=41.7%`。
- 配对结果：improved `2`、degraded `7`、tied `51`。
- pooled delta：`-8.33 pp`，bootstrap 95% CI `[-18.33, +1.67] pp`。
- 当前分类：`useful_safety_fallback_only`，不能称为 safe-capture 改进。

### 2.3 Ranking diagnosis 的真实问题

当前 ranking score 是 cost，按升序选择；因此 score 与 settled progress 的负相关方向本身是预期的，不能仅凭 `Spearman < 0` 判定 sign bug。

三 seed 中所有候选同时不可用的比例为 `60.4% / 42.0% / 21.6%`。当至少两个候选 eligible 时：

- score argmin 与 settled-best 一致率约为 `49.9% / 59.5% / 61.3%`；
- 最终 selected 与 settled-best 一致率仅约为 `9.5% / 6.9% / 7.0%`；
- selected 与 score argmin 一致率仅约为 `15.2% / 11.0% / 8.7%`。

因此当前优先级是：先分离 eligibility、nominal anchor、abstention 和真实 score 顺序的影响，再决定是否改模型或权重。禁止用增加 episode 数掩盖该阻塞。

## 3. 不可变安全与信息合同

### 3.1 信息边界

- 在线 `BeliefState` 只能包含 defender 状态、target belief、观测/通信历史、障碍几何、边界、动作历史、时间戳年龄和 provenance。
- target ground truth 只能用于离线 settled label、episode 结算和 failure audit，并显式标记 `offline_only=true`。
- train、validation、calibration、development、locked 必须按 episode/layout/seed 隔离。
- development 失败片段不得直接写回旧训练 archive；任何重训都创建新 archive、checkpoint、ledger、protocol 和 hash。

### 3.2 候选合同

- 候选数固定 `K=5`：`nominal`、`intercept`、`lateral_clearance`、`formation_clearance`、`visibility_hold`。
- 每个候选是 3 个 control steps 的 action chunk；只执行第一个控制步，然后重新观测和规划。
- 候选先经过 finite、shape、speed、acceleration、slew 和 reachability 检查，再进入 JEPA。
- JEPA 只能输出预测、uncertainty 和评分特征，不能生成最终动作、修改 CBF 约束或绕过 CBF。
- nominal 永远保留为 anchor；证据不足时只能走固定 fallback，不得删除 nominal。

### 3.3 safe-capture 结算

一个 episode 只有同时满足以下条件才是 `safe_capture=true`：

1. 至少一个 defender 在时限内进入目标 `0.80 m` capture radius。
2. 无 obstacle、target 或 defender-defender collision。
3. 无 boundary/altitude violation。
4. 无 pairwise separation violation。
5. 无 CBF infeasible、timeout、unverified action 或 `controlled_abort` 终止。

`controlled_abort` 是安全失败，必须保留在失败分母；`raw_unverified_executed`、`cbf_controlled_abort_steps`、`cbf_timeout_steps` 和 `cbf_infeasible_steps` 分开统计。

### 3.4 固定回退链

```text
separation-preserving safe-hold
  -> verified nominal through the same Joint CBF-QP
  -> controlled_abort / mission termination
```

任何 OOD、stale、non-finite、ledger/provenance mismatch、JEPA 超时或 CBF 失败都禁止执行 raw desired action。不得降低 CBF margin、放宽 stale age、关闭 OOD 检查、删除 controlled abort 或执行完整 action chunk。

## 4. 模块职责和接口

| 模块 | 输入 | 输出 | 绝对禁止 |
|---|---|---|---|
| Observation/Belief | 多机观测、通信、历史 | immutable `BeliefState`、mask、age、provenance | 读取 online target truth |
| Candidate Planner | belief、冻结 actor/规则 | 5 个 3-step chunks、可达性标记 | 生成未经过 finite/reachability 检查的动作 |
| JEPA Evaluator | belief + candidate chunk | 多 horizon latent、任务/安全预测、uncertainty | 直接输出最终执行动作 |
| Reliability Ledger | prediction、bucket、age、hash | `trusted`/`fallback_nominal`/`safe_hold`、reason code | 在线改变 credit/threshold |
| Safety-first Ranker | 候选特征、ledger、nominal anchor | selected candidate 或 abstention | 选择无效或安全下界不合格的候选 |
| Joint CBF-QP | desired action、状态、全部约束 | verified action、status、slack、active set | infeasible/timeout 时返回 raw action |
| Rolling Executor | verified action | 执行第一步、下一周期 replan | 执行完整 chunk、绕过 CBF |
| Trace/Provenance | 各模块输入输出 | step trace、episode summary、manifest | 删除失败步或覆盖旧结果 |

每个接口都必须有 schema test、finite test、故障注入 test 和 deterministic replay。特别要验证：`JEPA -> Ranker` 只影响候选选择，`Ranker -> CBF` 的 desired action 仍经过同一个 solver，`CBF -> Executor` 只允许 `verified=true` 且 finite 的动作。

## 5. 分阶段执行计划

### WP0：冻结工作区和证据入口

**目标：** 防止当前 V20、V5、E1、tmp 和旧结果互相污染。

- [ ] 记录 `git status --short`，保留所有用户已有未提交修改；禁止 `git add .`、reset、删除或清理 `tmp/`、NPZ、checkpoint 和历史 results。
- [ ] 保存 Git revision、Python、PyTorch、CUDA、GPU、Conda 包清单和 `pip freeze`。
- [ ] 保存当前 protocol、scene manifest、checkpoint、ledger、calibration archive 和代码的 SHA-256。
- [ ] 为每个新阶段建立独立、空的 `results/` 和 TensorBoard 目录；非空目录必须拒绝覆盖。
- [ ] 运行 `git diff --check`、协议 schema tests、CBF tests、device replay tests 和 ranking tests。
- [ ] 输出 `preflight.json`，写入 `development_only=true`、`locked_test_opened=false`、命令行和全部输入 hash。

**出口门：** 任何 hash、split、设备或 locked 状态不一致都停止，不运行 episode。

### WP1：完成 ranking diagnosis 的代码和测试

**目标：** 把“模型排序差”“候选资格过严”“nominal/abstention 过多”分开计量。

- [ ] 为 `scripts/diagnose_jepa_safe_capture_v20_ranking.py` 增加单元测试：空 eligible、单 eligible、多 eligible、缺失 trace、非 finite clearance、fallback reason 和比例计算。
- [ ] 提交诊断报告，固定输出每 seed 的 all-ineligible rate、multi-eligible count、score argmin/selected/settled-best 三种一致率、fallback reason 和 abstention reason。
- [ ] 在报告中明确 score 是 cost，负的 score-progress correlation 不自动等于 sign bug。
- [ ] 将 `selected-not-best`、CBF abort 前状态和 controlled abort 的候选上下文 join 到同一 failure index。

**出口门：** 结构测试通过；三 seed 诊断可重放；没有使用 locked truth 改变在线决策；报告不提出未经验证的权重修复。

### WP2：人工 monotonic score suite

**目标：** 在进入新实验前验证 score 的方向、单位、优先级和 tie 语义。

建立完全独立的合成输入集，覆盖以下断言：

- settled task progress 增大时，task cost 不应恶化，除非安全约束优先级触发。
- predicted clearance、TTC、CBF intervention risk、uncertainty 变差时，总 cost 不应变好。
- 安全 lower-bound 不足的候选，即使 task progress 较好，也必须被拒绝或排在安全候选之后。
- visibility 下降、observation age 增大、通信 dropout 增大时，ledger 状态只能保持或降级，不能升级。
- 完全相同的 score 使用固定 candidate index 和 tie band，不能因为设备浮点误差改变决策。
- NaN/Inf、未知 horizon、hash mismatch 必须产生固定 reason code，并进入 safe-hold/nominal fallback。

交付：`monotonic_score_cases.jsonl`、`monotonic_score_report.md`、测试文件和 TensorBoard scalar。测试只检查决策语义，不把合成数据当作任务性能证据。

**出口门：** 所有方向性、单位、tie、非 finite 和安全优先断言通过；否则停止，不修改 CBF margin。

### WP3：eligibility、nominal anchor 和 abstention 修复

**目标：** 减少无必要的“全候选不可用”，同时不削弱真实 CBF 安全边界。

按以下顺序做独立 protocol revision，每次只改变一个因素：

1. **净空 eligibility：** 评估 calibrated lower-bound 的单位、q10 offset、horizon 对齐和 `minimum_predicted_clearance_m` 的覆盖率。可以重定义预测资格规则，但不能把预测资格当成几何安全证明。
2. **nominal anchor：** 只有非 nominal 候选在安全 lower-bound、ledger state、finite/reachability 和 score margin 上同时合格时才允许替换 nominal。
3. **top-two abstention：** 在独立 calibration evidence 上选择 tie band 和 margin；不以 smoke 结果事后调阈值。
4. **hysteresis/minimum hold：** 约束 candidate oscillation，但每次仍只执行第一步；不得通过延长 open-loop chunk 解决抖动。

每个 revision 生成新 protocol hash、ledger revision、scene manifest、output root 和 TensorBoard root。旧目录只读，禁止覆盖。

**出口门：** all-ineligible rate 有可解释变化；raw/unverified 仍为 0；CBF margin、fallback 顺序和 controlled abort 语义不变。

### WP4：settled counterfactual 与候选排序审计

**目标：** 证明 score 方向修复后，候选选择是否与局部 settled safety/progress 一致。

- [ ] 对每个候选保留相同 belief、相同起始状态和相同 CBF/动力学条件，离线执行短 settled horizon。
- [ ] 只用 target truth 生成 `offline_only` settled label，不把 label 回灌在线 evaluator、ledger 或训练梯度。
- [ ] 报告 score argmin、最终 selected、settled-best 的一致率、selected-not-best、within-decision rank correlation、按安全失败/净空/可见性/急转/延迟分桶的结果。
- [ ] 对 degraded/improved/tied episode pair 做 failure replay，标注候选反转、all-ineligible、abstention、CBF abort 和 nominal hold。

**出口门：** score orientation、label horizon、action scale 和 eligibility 的含义均可解释；不能只因为相关系数变好就宣称闭环提升。

### WP5：多任务 interaction-aware JEPA

**目标：** 在排序合同稳定后，训练能够评价交互后果和安全辅助量的世界模型。

模型至少包含：

- target relative displacement、velocity、acceleration；
- obstacle/inter-agent clearance lower quantile；
- pairwise TTC；
- visibility、observation age 和通信质量；
- CBF intervention probability、correction magnitude 和 feasibility risk；
- multi-horizon uncertainty/quality head。

训练要求：

- 只使用 train split；validation/calibration/development/locked 永不参与梯度。
- 采用 target 急转、速度突变、flee persistence、S-curve、遮挡、低净空、高拥挤、通信延迟和候选反转的 hard-segment replay。
- 每个 seed 独立 checkpoint、history、metadata、TensorBoard 和 hash；不得用 development 失败片段直接追加旧 archive。
- 先审计 action-following separation 和辅助头 finite/calibration，再接入闭环。

**出口门：** 输出 finite；同一 belief 下不同 action chunk 有可复现的非零预测 separation；主要安全头有独立校准证据。预测 MAE 下降本身不构成 safe-capture 通过。

### WP6：Reliability Ledger 校准和故障审计

**目标：** 让模型在预测漂移之前拒答，并把拒答变成可统计的控制路径。

- [ ] 使用与 train/validation/development 不重叠的 calibration archive。
- [ ] 固定或预注册 `minimum_sample_count`、credit threshold、observation age、uncertainty、TTC/CBF risk 和 clearance calibration 规则。
- [ ] 每个 `(checkpoint, context bucket, horizon, head)` 记录样本数、误差、coverage、credit、settled safety failure 和置信区间。
- [ ] 注入 OOD、stale、non-finite、unknown horizon、checkpoint hash mismatch、ledger hash mismatch 和 provenance mismatch。
- [ ] 验证 high-credit settled failure 不高于 low-credit bucket；证据不足必须是 `insufficient_evidence`，不能当作 trusted。
- [ ] 验证 ledger immutable：运行期间不更新 credit、threshold 或 bucket 统计。

**出口门：** 全部 fault 进入预期 fallback；`raw_unverified_executed=0`；每次状态转移都有 reason code、输入 hash 和回退动作。

### WP7：Joint CBF-QP 和 rolling-horizon 回归

**目标：** 证明所有候选路径最终都受同一个安全边界控制。

- [ ] 对 nominal、JEPA selected、safe-hold、fallback nominal 运行同一个 Joint CBF-QP。
- [ ] 做 zero-perturbation identity replay：相同 desired action 必须产生相同 verified action、CBF status 和物理结算。
- [ ] 做 non-zero rolling replay：至少 100 个 control cycles，双次运行逐字段比较 decision、ledger、CBF、action、termination 和 trace。
- [ ] 注入 solver timeout、infeasible、non-finite desired action、异常状态和通信/观测冻结；验证固定回退链。
- [ ] 测量 RTX 5050 p95 latency，但不能为降低延迟而关闭安全约束或执行完整 chunk。

**出口门：** collision/boundary/pairwise/CBF timeout/raw-unverified 硬门通过；CPU/CUDA 离散决策和安全结算一致；100-cycle replay 完全可重放。

### WP8：新 protocol smoke

**目标：** 在扩大样本前筛除明显的安全或任务退化。

固定三 seed：`20260911/20260912/20260913`。每 seed、每 variant 先运行 20 集 paired smoke。

建议 variant：

| variant | 作用 |
|---|---|
| M0 | nominal + Joint CBF-QP 基线 |
| M1 | target-displacement-only JEPA + CBF |
| M2 | multi-task JEPA + CBF、无 ledger |
| M3 | full JEPA + immutable ledger + ranker + CBF + rolling horizon |
| A1 | full stack 去掉 auxiliary safety heads |
| A2 | full stack 去掉 ledger，仅作故障对照 |

每个 smoke 必须记录：safe capture、improved/degraded/tied、collision、boundary、pairwise、CBF infeasible/timeout、controlled abort、fallback、raw-unverified、minimum clearance、candidate eligibility、selected-not-best 和 latency。

准入只看 `safe_capture` 和安全硬门：

- 任一硬安全门失败：立即停止并标记 `BLOCKED_BY_SAFETY`。
- 任一 variant/seed 证据不足：标记 `insufficient_evidence`，不扩大规模。
- smoke 通过只意味着可以评估更大 development block，不等于正式提升。

### WP9：三 seed paired development

只有 WP1-WP8 全部通过后才能进入。每 seed、每核心 variant 至少 40 集，必要时扩展到 60 集；保持相同 scene manifest、episode pair、checkpoint、ledger 和 protocol。

固定报告：逐 seed/pooled `safe_capture`、sample SD、paired delta、bootstrap CI、exact McNemar、improved/degraded/tied、controlled abort 和所有安全失败。`mean_capture_time` 只能放在诊断表。

建议的任务性判定：

- 三 seed 中至少 2/3 非负且 pooled paired delta 不为负，才可称为 `safe_capture_improvement_candidate`；
- 若安全门通过但任务 delta 不为正，归类为 `safe_fallback` 或 `prediction_signal_no_control_gain`；
- 若证据不足，归类为 `insufficient_evidence_do_not_open_locked_test`。

这些是 development 判定，不是新的 locked-test 结论。不得为满足门槛删除 controlled abort、改变失败分母或事后调参。

### WP10：鲁棒性、长序列和 SIL/HIL readiness

在 WP9 的核心结果可解释后开展：

- detection dropout、observation noise、message delay/dropout；
- target 急转、速度突变、flee persistence、S-curve；
- 障碍密度、低净空、高拥挤和左右初始侧距；
- 单机故障、通信冻结、传感器冻结、GPU 不可用和 watchdog 超时；
- 100/500/1000-cycle long replay，以及真实计算预算下的 latency。

SIL/HIL 只验证接口和安全边界，不把仿真结果写成真实飞行性能；实飞前必须有独立飞控审查和明确授权。

### WP11：论文证据归档和 locked 决策

- [ ] 保存代码 revision、环境清单、protocol、manifest、checkpoint、ledger、calibration、命令、日志和结果 hash。
- [ ] 生成 `summary.json`、`run_metadata.json`、`episodes.csv`、逐步 trace、failure index、TensorBoard 和 Markdown 报告。
- [ ] 校验 JSON/CSV/Markdown/TensorBoard 数值一致。
- [ ] 归档正向、负向和安全 fallback 结果，不删除失败实验。
- [ ] 只有安全硬门、provenance、device determinism、ranking、ledger 和 paired statistics 全部通过后，才起草新的 locked-test preregistration；在明确授权前保持 `locked_test_opened=false`。

## 6. 产物和命名规则

V20 及历史目录只读。后续修复使用唯一前缀，例如：

```text
results/jepa_safe_capture_v21_ranking_diagnosis/
results/jepa_safe_capture_v21_monotonic_score_suite/
results/jepa_safe_capture_v21_protocol_preflight/
results/jepa_safe_capture_v21_ledger_seed<seed>/
results/jepa_safe_capture_v21_settled_seed<seed>/
results/jepa_safe_capture_v21_smoke_<variant>_seed<seed>/
results/jepa_safe_capture_v21_development_<variant>_seed<seed>/
results/jepa_safe_capture_v21_failure_index/
results/jepa_safe_capture_v21_tensorboard/<stage>/seed<seed>/
```

每个目录至少包含：

```text
summary.json
run_metadata.json
input_hashes.json
command.txt
development_only=true
locked_test_opened=false
```

有 episode 的运行还必须保存 scene manifest、`episodes.csv`、逐步 trace 和 failure index。任何工具发现目标目录非空都应停止，而不是覆盖。

## 7. 推荐执行命令入口

以下命令只用于当前 development 审计；运行前必须重新确认路径、hash 和目标目录为空：

```powershell
Set-Location D:\\uav-capture\\uav_capture
$py = 'D:\\download\\anaconda3\\envs\\traj_pred_prep\\python.exe'
$env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = 'python'

& $py -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
& $py -m pytest -q tests/test_jepa_safe_capture_protocol.py tests/test_audit_jepa_safe_capture_candidate_ranking.py tests/test_audit_jepa_safe_capture_v5_p2_ranking.py tests/test_audit_jepa_safe_capture_device_replay.py tests/test_cbf_qp.py
& $py scripts/diagnose_jepa_safe_capture_v20_ranking.py `
  --output-dir results/jepa_safe_capture_v21_ranking_diagnosis `
  --tensorboard-dir results/jepa_safe_capture_v21_tensorboard/ranking_diagnosis
```

新 protocol、ledger、checkpoint 或 smoke 运行不能直接复用旧 output root；必须先生成新的 preflight 和 hash manifest。

## 8. 硬停止规则

出现以下任一情况，立即停止当前阶段并回到对应上游阶段：

- protocol、checkpoint、ledger、calibration、scene manifest 或 code hash 不一致；
- CPU/CUDA 的 candidate order、selected index、ledger state、CBF status、action、termination 或安全结算不一致；
- score orientation、settled label、horizon 或 action scale 无法解释；
- 任意 collision、boundary、pairwise violation、未处理 CBF failure 或 `raw_unverified_executed > 0`；
- 预测 eligibility 变化却没有 calibration 证据；
- 为提高 safe-capture 而降低 CBF margin、关闭 OOD/stale、放宽失败语义或执行完整 action chunk；
- 把单 seed、20 集 smoke、局部 settled label、预测 MAE 或 capture time 写成正式控制收益；
- 试图访问或打开新的 locked-test split。

## 9. 完成定义

只有同时满足以下条件，系统才能被描述为“安全增强的 JEPA 闭环围捕系统”：

1. Belief、候选、JEPA、ledger、ranker、Joint CBF-QP、rolling executor 和 trace 都有可测试接口。
2. JEPA 从始至终只评价候选轨迹，最终动作只来自同一个 `verified` Joint CBF-QP 输出。
3. score orientation、eligibility、settled label、horizon 和 action scale 已审计，不能再出现无法解释的系统性排序反向。
4. OOD、stale、non-finite、低 credit、provenance mismatch 和 CBF 失败永不执行 raw/unverified action。
5. 三 seed 的 CPU/CUDA replay、100-cycle rolling replay 和 fault matrix 可逐字段重放。
6. 三 seed paired smoke 和后续 development block 以 `safe_capture` 为主结论，安全失败和 controlled abort 完整纳入分母。
7. 所有正负结果、环境、协议、模型、ledger、数据和输出都有 hash/provenance。
8. 未经明确授权，`locked_test_opened=false` 始终保持不变。

允许的最终结论类别：

```text
safe_capture_improvement_candidate
safe_capture_noninferior_safety_preserving
useful_safety_fallback_only
prediction_signal_no_control_gain
rejected_for_safety
insufficient_evidence_do_not_open_locked_test
```

**当前默认结论：** 安全执行基础设施和设备决定性已有证据，但 JEPA 排序与 settled outcome 的关系尚未达到闭环收益门。下一步先完成 WP1-WP4，再决定是否投入新的多任务训练和扩大三 seed 实验。
