# 无人机集群对抗围捕安全增强系统
# 下一步详细 TODO 与验收计划

**系统路线：** Interaction-Aware Action-Conditioned JEPA + Reliability Ledger + Joint CBF + Receding-Horizon Control  
**版本：** v1.0 execution plan  
**日期：** 2026-09-04  
**环境：** Windows / Conda `uav-encirclement-gpu` / NVIDIA RTX 5050  
**实验边界：** development-only，`locked_test_opened=false`  
**第一指标：** `safe_capture`；`mean_capture_time` 只作为次要诊断指标

> 本计划书是当前仓库状态下的执行顺序。任何安全硬门、数据边界、哈希或可回放性失败，都优先于提高捕获率。未经单独明确授权，不打开新的 locked test。

---

## 1. 最终目标

构建一条可审计、可回退的无人机集群围捕闭环：

```text
多机观测/通信历史
        -> interaction-aware belief state
        -> 冻结传统规划器生成候选 action chunks
        -> action-conditioned JEPA 反事实预测
        -> reliability ledger 可信度判定
        -> 安全优先候选排序
        -> Joint CBF-QP 最终过滤
        -> 只执行第一控制步
        -> 重新观测、重规划、重过滤
```

系统必须满足：

- JEPA 只能评价候选轨迹，不能直接生成或覆盖最终飞行动作。
- 在线模块不能读取 target ground truth；ground truth 仅用于离线标签和结算。
- 所有候选以及 nominal baseline 都经过同一个 Joint CBF-QP。
- Reliability ledger 是拒答和回退机制，不是安全证明。
- CBF 是不可绕过的最终执行边界。
- `safe_capture` 必须同时满足捕获、时间限制、无障碍/目标/机间碰撞、无 defender 越界和无未验证 CBF 终止。
- action chunk 只执行第一步，下一周期必须重新观测并 replan。

## 2. 当前状态和阻塞项

### 2.1 已完成

- [x] WP0/WP1：协议、输入边界、失败索引和数据 split 冻结。
- [x] WP2：hard-context weighted JEPA 三 seed 训练（`20260911/20260912/20260913`，40 epoch，CUDA/RTX 5050）。
- [x] WP2 held-out prediction gate：三个 seed、四个 horizon 的 target MAE 均优于 constant-velocity。
- [x] WP3：checkpoint-bound reliability ledger v3，包含 OOD/stale/non-finite fallback 和 high-credit failure gate。
- [x] P5 CBF 单元测试、确定性求解、显式 fallback 和 provenance/TensorBoard 审计。
- [x] 新 development protocol：`configs/central_random_mixed_obstacle_s3_v3_development_protocol.yaml`。

### 2.2 当前 smoke 证据

M3（JEPA + ledger + candidate ranking + CBF）20 集 smoke：

| training seed | safe capture | collision | boundary | pairwise | CBF controlled abort |
|---:|---:|---:|---:|---:|---:|
| 20260911 | 8/20 = 40% | 0 | 0 | 0 | 11 |
| 20260912 | 6/20 = 30% | 1 | 1 | 0 | 13 |
| 20260913 | 5/20 = 25% | 0 | 0 | 0 | 14 |

M0 rerun（nominal + CBF）为 10/20 = 50%，collision/boundary/pairwise 为 0，CBF controlled abort 为 9。

**当前结论：** M3 smoke 未通过安全硬门，因此不能启动 A1/A2，也不能启动三 seed 全量 block，更不能申请 locked test。当前结果只说明闭环可运行，不说明 JEPA 已带来安全捕获提升。

### 2.3 必须先解决的阻塞项

seed `20260912` 的 episode `19` 同时出现 `collision=True`、`boundary_violation=True`，但 trace 中 CBF `verified_feasible=True` 且没有负净空。当前 `pursuit_env.py` 把 target 与 defender 的越界都累加到同一个 `world_violation_steps`，存在“target 越界被计为 defender safety failure”的语义混淆风险。

在语义审计完成前，不得重跑全量并据此比较模型。

---

## 3. 不可变安全合同

### 3.1 固定运行参数

- 候选数 `K=5`：`nominal`、`intercept`、`lateral_clearance`、`formation_clearance`、`visibility_hold`。
- action chunk 长度为 3 个 control steps；只执行第 1 步后 replan。
- 控制周期 `0.1 s`；CBF anticipatory horizon 为 3 steps。
- obstacle、boundary、inter-agent surface margin 均为 `0.35 m`。
- CBF `gamma=0.25`，最大 correction norm `5.0 m/s`，p95 延迟目标不超过 `100 ms`。
- solver、tolerance、maxiter、timeout、seed block、scene manifest 在 smoke 通过后全部冻结。

### 3.2 安全硬门

对所有安全保留变体：

1. collision count 必须为 0；
2. defender boundary violation count 必须为 0；
3. pairwise violation count 必须为 0；
4. QP infeasible/timeout/unverified 必须显式 fallback，不能执行 raw action；
5. zero-perturbation 时非 JEPA 字段完全一致；
6. ground-truth leakage、split leakage、未配对 episode 或 provenance 缺失立即停止。

安全硬门失败时，保存 trace 和诊断报告，停止当前变体，并回退到冻结 nominal + CBF。不得用更高 safe-capture 或更短 capture time 抵消安全失败。

### 3.3 Safe-capture 统计门

仅在安全硬门通过后应用：

- mean paired safe-capture delta >= `0 pp`；
- 至少 `2/3` seed 非负；
- 每个 seed 的下降不超过预先声明的一个 episode resolution；
- 报告 paired improved/degraded/tied、bootstrap 95% CI 和 exact McNemar。

若 CI 下界不大于 0，只能称为 `positive development evidence` 或 `safety-preserving non-inferiority`，不能称为稳健提升。`mean_capture_time` 不作为否决 safe-capture 的主门。

---

## 4. 分阶段 TODO

## WP-A：边界语义和 smoke 失败审计（立即执行）

**目标：** 区分 target 越界、defender 越界、障碍碰撞和 CBF 失效，排除统计标签错误。

- [ ] 对 seed `20260912` episode `19` 做 deterministic replay，定位首次越界 step、实体类型和触发 constraint。
- [ ] 检查 `pursuit_env.py` 的 target `_enforce_world_bounds` 与 defender `_enforce_world_bounds` 调用路径。
- [ ] 增加独立计数器：`target_world_violation_steps`、`defender_world_violation_steps`、`target_boundary_violation`、`defender_boundary_violation`。
- [ ] 保留历史字段 `world_violation_steps` 的兼容输出，但在新协议中明确其组成和安全结算来源。
- [ ] 将 safe-capture 的 boundary gate 绑定到 defender 越界；target 越界作为独立 target termination/diagnostic，除非协议明确规定其属于任务安全失败。
- [ ] 增加 target-only、defender-only、同时越界、后续捕获四类回归测试。
- [ ] 用独立审计脚本重算三份 smoke 的 collision/boundary/safe-capture，不覆盖原始结果。

**验收：** episode 19 的主因可唯一解释；新旧字段映射有测试；报告明确说明历史结果是否受标签语义影响；完整测试不回归。

**产物：** `docs/JEPA_SAFE_CAPTURE_WP6_SMOKE_SAFETY_AUDIT_20260904.md`、独立 audit JSON、回归测试和新的 provenance hash。

## WP-B：失败片段 hard replay 与因果归因

**目标：** 把失败分解为预测漂移、ledger 错信、候选排序、CBF 过度修正、QP 不可行或任务几何原因。

- [ ] 使用 `scripts/index_jepa_safe_capture_failures.py` 建立只读 failure index。
- [ ] 每个类别至少选择 3 个代表 episode：遮挡/过期观测、CBF controlled abort、低净空、候选振荡、target motion shift。
- [ ] 按 `observation -> belief -> JEPA -> ledger -> rank -> CBF -> executed action -> termination` 逐步重放。
- [ ] 记录 ledger state/credit、candidate score margin、selected candidate、CBF active constraints、correction norm、solver status、最小净空和终止原因。
- [ ] 无法由 trace 证明的原因标记为 `unresolved`，不得强行归类。
- [ ] 统计 high-credit failure、fallback 后安全率、candidate switch rate 和 CBF abort 关联。

**验收：** 100% 失败 episode 有唯一主因或 `unresolved` 标签；同一 episode replay 两次的终止原因、动作和 trace hash 一致。

**产物：** `results/jepa_safe_capture_v3_failure_index/`、`docs/JEPA_SAFE_CAPTURE_WP1_FAILURE_REPLAY_20260904.md`。

## WP-C：JEPA reliability 增强（离线优先）

**目标：** 让模型在对抗目标、延迟观测和集群交互下能表达“不确定”，而不是只预测 target displacement。

- [ ] 保留 target displacement/velocity/acceleration 主头。
- [ ] 校准 obstacle clearance、inter-agent clearance、pairwise TTC、visibility、observation age、CBF intervention 和 QP feasibility heads。
- [ ] 为每个 head 计算 calibration error、lower-quantile coverage、Brier/AUC 或适用的 ranking metric。
- [ ] 增加 explicit motion-mode embedding：`flee_persistence`、`turn`、`s_curve`、突变加速度和速度突变。
- [ ] 增加 action-conditioned consistency/contrastive loss，检查不同候选块的未来表示是否方向一致且不塌缩。
- [ ] 用 ensemble disagreement、heteroscedastic residual 或 calibrated residual 产生 uncertainty；不把 uncertainty 当安全证明。
- [ ] 维持 train/validation/calibration/development episode-level split，禁止把失败 development episode 直接回灌训练。

**离线准入：** 三 seed 输出 finite；各主要 horizon 的 target MAE 优于 constant-velocity；辅助头标签非空且有校准证据；action-following 方向一致性为正。

## WP-D：Reliability Ledger v3 temporal/adversarial 校准

**目标：** 高置信但错误时及时 abstain，抑制 rollout 漂移和分布外失效。

- [ ] 在 calibration-only split 上增加 temporal credit drift：连续残差突增、credit 连续下降、candidate separation 消失即转 fallback。
- [ ] 注入急转、速度突变、遮挡、消息延迟、丢包、障碍密度变化和队形拥挤度 shift。
- [ ] 冻结最小 bucket 样本数、最低 credit、uncertainty 上限、stale age 上限和 OOD 规则。
- [ ] 明确状态转移：`trusted -> rank`，`fallback_nominal -> nominal-CBF`，`safe_hold -> separation-preserving hold/controlled abort`。
- [ ] 验证 OOD/stale/non-finite 输入 100% 显式 fallback，且 unverified path 不计 safe capture。
- [ ] 完成 checkpoint、calibration archive、protocol 和 builder revision hash 绑定后，将 ledger 设为只读。

**验收：** high-credit settled failure rate 不高于 low-credit；每次 abstain 有 reason code 和可回放 trace。

## WP-E：候选动作块和安全排序

**目标：** 让 JEPA 真正改善候选排序，而不是只依赖 CBF 保住安全。

- [ ] 冻结当前 `K=5`、3-step chunk 作为主比较合同。
- [ ] 所有候选先过 speed、acceleration、slew、可达性和 finite 检查；失败候选不得进入 JEPA。
- [ ] score 分解为 task progress、保守 clearance、visibility gain、CBF intervention cost、uncertainty penalty、action-change cost 和 nominal anchor penalty。
- [ ] clearance 使用 lower quantile；预测安全量只能排序，不得替代 CBF。
- [ ] 记录 score margin、rank stability、candidate switch、oscillation、CBF correction norm 和候选/nominal 差异。
- [ ] 做离线 rank consistency：预测 top-1 与 settled safe outcome、CBF intervention 和 capture outcome 分别统计。
- [ ] 如需探索 chunk `1/3/5` 或新候选库，必须另建 protocol，不能在同一 block 事后择优。

**验收：** zero-perturbation 非 JEPA 字段差异为 0；候选切换不过度；排序收益不能伴随新增安全失败。

## WP-F：Joint CBF-QP 和 fallback fault injection

**目标：** 确保任何上游模型失效都不会执行未过滤动作。

- [ ] 审计 obstacle、defender boundary、altitude、speed、acceleration、pairwise separation 和 anticipatory braking 约束。
- [ ] 注入 QP infeasible、solver timeout、non-finite input、stale observation、通信中断和多约束同时激活。
- [ ] 固定 fallback 顺序：separation-preserving safe-hold -> verified nominal-CBF -> controlled abort。
- [ ] 每个输出记录 solver status、verified feasibility、active set、slack、residual、latency、fallback reason。
- [ ] 做 agent 数量、队形密度、障碍数和边界压力的 latency/feasibility stress test。
- [ ] 增加接口测试，证明 ranker/JEPA 不能在 CBF 之后覆盖 action。

**验收：** 任意 infeasible/timeout 不执行 raw action；安全保留变体 collision/boundary/pairwise 为 0；p95 端到端延迟 <= 100 ms。

## WP-G：滚动时域闭环集成

**目标：** 验证完整 Simulation -> Planning -> Decision-making 链路。

每个控制周期严格执行：

1. 检查 observation timestamp、message age 和 finite；
2. 更新 interaction-aware belief；
3. 生成并筛选候选 action chunks；
4. 对候选 batch 做 JEPA prediction；
5. 执行 ledger decision；
6. 进行安全优先 ranking；
7. 调用 Joint CBF-QP；
8. 只执行第一个 control step；
9. 重新观测并记录完整 trace。

- [ ] JEPA、ledger、ranker 或 CBF 任一超时立即 fallback。
- [ ] 记录输入 hash、预测、uncertainty、credit、selected rank、CBF 修正、动作、延迟和 fallback。
- [ ] 做 CPU 与 RTX 5050 的数值容差和 deterministic replay 对照。
- [ ] 从空目录重跑 1 个 episode，核对 summary、episodes.csv、step trace、TensorBoard 和报告一致。

## WP-H：M0/M3 smoke 重跑与准入

**目标：** 在修正 boundary 语义和 replay 链路后，重新筛查闭环安全性。

运行顺序固定：

1. 先跑 M0 nominal + CBF（三 seed，各 20 集）；
2. M0 安全门通过后跑 M3（三 seed，各 20 集）；
3. M3 通过后才运行 A1（去 ledger）和 A2（去 clearance/visibility）；
4. A3 raw/no-CBF 只做故障诊断，不参与安全主结论。

每个变体/seed 使用独立空 results 和 TensorBoard 目录，固定同一 scene manifest、episode index、observation schedule 和 target motion。

**smoke 通过条件：**

- collision/boundary/pairwise 均为 0；
- no raw/unverified action executed；
- trace/provenance/TensorBoard 完整；
- zero-perturbation 通过；
- p95 latency <= 100 ms；
- ledger OOD/stale/non-finite fallback 率为 100%。

任何一项失败都停止该变体，保存 audit，不进入 final block。

## WP-I：三 seed paired development final block

**目标：** 获得可复核的跨 seed safe-capture 证据，不能只看单个 seed。

- [ ] smoke 全通过后冻结 checkpoint、ledger、protocol、scene hash、solver 和 episode seed 列表。
- [ ] 运行 M0、M3、A1、A2；每个 training seed 至少 40 个配对 episode。
- [ ] 同一 `(training_seed, episode_index)` 使用完全相同的 layout、target motion、observation condition 和初始状态。
- [ ] 保存 episode summary、step trace、candidate trace、ledger state、CBF diagnostics、TensorBoard 和命令行 provenance。
- [ ] 不在 final block 中调 threshold、margin、candidate weight、chunk length 或 episode seed。

**主比较：** M3 vs M0。  
**消融：** A1 测 ledger；A2 测 clearance/visibility heads；A3 仅证明 raw/no-CBF 不可部署。

## WP-J：统计、审计和研究结论

- [ ] 以 `(training_seed, episode)` 为独立单位，拒绝把 timestep/chunk 当独立样本。
- [ ] 报告三 seed mean、sample SD、paired delta、improved/degraded/tied、bootstrap 95% CI 和 exact McNemar。
- [ ] 逐 seed 报告 safe capture、collision、boundary、pairwise、CBF infeasible/timeout/unverified、fallback 和 high-credit failure。
- [ ] 按 motion mode、visibility、observation age、clearance、ledger state、CBF active constraint 分桶。
- [ ] 二次报告 capture time、路径长度、修正 norm 和 latency；不以 capture time 否决安全结果。
- [ ] JSON/CSV、TensorBoard 和 Markdown 报告做双向一致性校验。
- [ ] 结论只允许归类为：
  - `safe_capture_improvement_candidate`
  - `safety_preserving_noninferior`
  - `prediction_signal_no_control_gain`
  - `rejected_for_safety`
  - `insufficient_evidence_do_not_open_locked_test`

## WP-K：SIL/HIL（可选，且不得提前）

只有 WP-F 至 WP-J 通过后才进入：

- [ ] RTX 5050/CPU 测量 JEPA、ledger、ranker、CBF 和端到端 p50/p95/p99 latency。
- [ ] 注入传感器冻结、时间戳错误、网络中断、target 突变、单机失效、QP timeout 和显存压力。
- [ ] 验证故障时仍保持 obstacle、boundary、pairwise clearance，否则 controlled abort。
- [ ] 准备 watchdog、geofence、safe-hold、急停、返航/任务终止和人工接管接口。
- [ ] 形成部署包、回滚包和审计日志完整性报告；没有安全审查和明确授权不得实飞。

---

## 5. 实验矩阵

| 阶段 | 变体 | seed | episode | 说明 |
|---|---|---:|---:|---|
| smoke | M0 | 3 | 20/seed | nominal + CBF，先跑 |
| smoke | M3 | 3 | 20/seed | JEPA + ledger + ranking + CBF |
| smoke | A1 | 3 | 20/seed | 仅在 M3 通过后 |
| smoke | A2 | 3 | 20/seed | 仅在 M3 通过后 |
| final | M0/M3/A1/A2 | 3 | >=40/seed | 同一 paired scene block |
| diagnostic | A3 | 3 | 同一 block | raw/no-CBF，不进安全结论 |
| stress | M0/M3 | 3 | 独立 hard block | shift、拥挤、遮挡、延迟 |

推荐每个 final 变体使用独立目录：

```text
results/jepa_safe_capture_v3_wp7_<variant>_seed<seed>/
results/jepa_safe_capture_v3_tensorboard/wp7_<variant>_seed<seed>/
```

---

## 6. RTX 5050 / Conda 执行纪律

```powershell
Set-Location D:\uav-capture\uav_capture
$py = 'D:\miniconda3\envs\uav-encirclement-gpu\python.exe'
$env:PYTHONPATH = "$PWD\src;$PWD\scripts"
& $py -m pytest -q
```

- 训练使用 CUDA；评估可用 CUDA，另做 CPU 一致性审计。
- 每个长实验从空 output/logdir 启动，不覆盖旧 checkpoint、NPZ、TensorBoard 或报告。
- 不使用 `git add .`，不触碰用户已有 E1/V5 修改。
- 每阶段写入 TensorBoard scalar、histogram/text provenance、JSON/CSV 和 SHA-256。
- 每阶段完成后创建独立 conventional commit；网络可用时推送 `origin/main`，推送失败只记录失败原因，不删除本地提交。

---

## 7. 时间盒和完成定义

| 时间盒 | 工作 | 完成标志 |
|---|---|---|
| Day 0--1 | WP-A 边界语义审计 | episode 19 原因明确、回归测试通过 |
| Day 1--3 | WP-B hard replay | 失败有因果链或 unresolved |
| Day 3--5 | WP-C/D 预测与 ledger 校准 | drift/OOD/stale fallback 可验证 |
| Day 5--7 | WP-E/F ranker 与 CBF fault tests | zero-perturbation、安全硬门、延迟通过 |
| Day 7--8 | WP-G/H smoke | M0/M3 三 seed 全部安全通过 |
| Day 9--13 | WP-I final development | 3 seed x >=40 paired episodes |
| Day 14--15 | WP-J 统计与报告 | provenance、TensorBoard、JSON/CSV 一致 |
| Day 16+ | WP-K SIL/HIL | 仅在前置门全部通过后执行 |

本计划完成的定义：

1. 边界、碰撞、CBF、回退和 safe-capture 语义有明确代码和测试支持；
2. JEPA、ledger、ranker、CBF、replan 链路逐步 trace 可回放；
3. 至少三个 training seed、同一 paired scene block、每 seed 至少 40 集完成；
4. 所有安全保留变体通过 collision/boundary/pairwise/zero-perturbation 硬门；
5. 结果按 safe-capture 优先、逐 seed 和配对统计报告，capture time 仅为次指标；
6. 所有产物带 protocol/checkpoint/ledger/scene/code/environment hash，并能从空目录复现最小运行；
7. 在明确授权前，`locked_test_opened=false` 始终保持不变。

**最终判据：** 不是追求某一个 seed 的最高捕获率，而是证明“JEPA 反事实评价 + reliability ledger 拒答 + CBF 硬安全层 + 滚动时域重规划”在多 seed、困难场景和完整审计链路下能够持续提高或至少不损害 `safe_capture`，且世界模型幻觉、目标漂移和集群安全约束不会静默失效。
