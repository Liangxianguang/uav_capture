# 无人机集群对抗围捕安全增强系统
# 下一步详细 TODO 与验收计划

**系统路线：** Interaction-Aware Action-Conditioned JEPA + Reliability Ledger + Joint CBF + Receding-Horizon Control  
**版本：** v1.3 execution plan（V3 failure-index 完成后，进入 hard replay）
**日期：** 2026-09-04  
**环境：** Windows / Conda `uav-encirclement-gpu` / NVIDIA RTX 5050  
**实验边界：** development-only，`locked_test_opened=false`  
**第一指标：** `safe_capture`；`mean_capture_time` 只作为次要诊断指标

> 本计划书是当前仓库状态下的执行顺序。任何安全硬门、数据边界、哈希或可回放性失败，都优先于提高捕获率。未经单独明确授权，不打开新的 locked test。

## 0.1 当前唯一允许的执行路径

下面是当前工作区的总控清单。它比“直接跑更多 episode”优先级更高；每一项都必须在
上一项的出口条件满足后才能开始。所有运行均保持 `development_only=true` 和
`locked_test_opened=false`。

| 顺序 | 工作包 | 立即动作 | 必须产出 | 出口条件 |
|---|---|---|---|---|
| 1 | WP-B0 接口适配 | 让 failure-index 同时识别 `v3_wp6_*_boundaryfixed` 与后续 `v3_wp7_*`，保留 v2 兼容 | 脚本、目录发现/配对测试 | **已完成**：12 个 smoke run 全部被发现，episode `0..19` 完整 |
| 2 | WP-B1 失败索引 | 只读扫描 summary、episodes、scene manifest、step traces | `results/jepa_safe_capture_v3_wp1_failure_index_current/`、索引 JSON/CSV、TensorBoard provenance | **已完成**：scene manifest 唯一、配对一致、源结果目录未修改 |
| 3 | WP-B2 hard replay | 对代表性 degraded episode 逐步重放 belief -> JEPA -> ledger -> rank -> CBF -> action -> termination | replay JSONL、trace hash、因果审计 Markdown | 每个失败有唯一主因，或明确标记 `unresolved`；重复 replay 完全确定 |
| 4 | WP-E1 候选可达性 | 在 JEPA 前检查 speed/acceleration/slew/finite/reachable | candidate feasibility 表、被拒候选原因 | 不可达候选 `0` 个进入 JEPA；可达率和拒绝原因可解释 |
| 5 | WP-E2 排序诊断 | 统计 top-1 settled outcome、score margin、switch/oscillation、CBF correction | rank audit、candidate trace、分桶统计 | 排序信号与 settled safe outcome 的关系可解释；无安全回归 |
| 6 | WP-D/F 可靠性与安全注入 | 注入 stale、OOD、non-finite、急转、延迟、QP timeout/infeasible | fault-injection report、reason-code 覆盖、延迟报告 | 所有异常均显式 fallback；不执行 raw/unverified action；端到端 p95 <= 100 ms |
| 7 | 发展协议决策 | 根据 B/E/D/F 证据决定“修复并新建 protocol”或“归档回归证据” | decision memo、新 protocol（如需要） | 没有可验证修复假设时，不为了凑 episode 数量运行 final block |
| 8 | WP-I paired final development | 仅在前 7 项通过后，M0/M3/A1/A2，3 seed，每 seed >=40 集 | 独立 results、TensorBoard、provenance | 配对场景/episode/观测条件完全一致，安全硬门全通过 |
| 9 | WP-J 统计结论 | safe-capture 优先统计并交叉核对 JSON/CSV/TensorBoard | final audit、报告、结果分类 | 只能落入预定义结论类别；仍不自动打开 locked test |

### 0.1.1 五个不可混淆的证据门

本项目的主张必须逐层通过，而不是用一个指标替代另一层：

1. **JEPA prediction gate**：证明 action-conditioned interaction-aware 表示能在离线数据上区分目标运动、净空、可见性和交互风险；这不是控制收益证明。
2. **Reliability gate**：证明 ledger 在 OOD、stale、non-finite 和 temporal drift 时拒答并可回放；credit 不是安全证明。
3. **CBF safety gate**：证明所有候选和 nominal 都经同一个 Joint CBF-QP，QP 失败不执行 raw action；这是安全硬门。
4. **Closed-loop gate**：证明每个周期只执行第一步并重新观测、重规划、重过滤，且 trace、延迟和 fallback 完整。
5. **Task gate**：在安全门通过后，才比较 paired `safe_capture`；`mean_capture_time` 只作诊断，不抵消安全失败。

### 0.1.2 当前工作日的停止点

- 在 WP-B1 失败索引测试或 scene pairing 失败时停止，不运行 replay。当前 WP-B1 已通过，可进入 WP-B2。
- 在 WP-B2 无法确定失败因果链时标记 `unresolved`，不凭均值调 score 或 ledger 阈值。
- 在 WP-E1 发现候选不可达时先修复候选生成/过滤，不让不可达动作进入 JEPA。
- 在 WP-D/F 任一异常路径执行 raw/unverified action、缺少 reason code 或 p95 超时，停止所有 final block。
- 只有 WP-B/E/D/F 的报告和 manifest 均冻结后，才允许创建新的 40-episode paired development 目录。

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
- [x] V3 failure-index 接口：支持 `auto|v2|v3`，并校验 V3 boundary-fixed 命名、三 seed、四变体和 episode 配对。
- [x] V3 WP1 当前索引：12 个 smoke run、240 集；`safe_capture=89/240 (37.1%)`，`cbf_controlled_abort=143`，`timeout=8`。
- [x] V3 WP1 审计产物：JSON/CSV/Markdown/provenance/TensorBoard 均已生成；TensorBoard 含 29 个 scalar tags 和必需 provenance text tags。

### 2.2 当前 smoke 证据（boundary-fixed，development-only）

同一 paired S3 scene manifest、三 training seed、每 seed 20 episodes：

| variant | 20260911 | 20260912 | 20260913 | mean |
|---|---:|---:|---:|---:|
| M0 nominal + CBF | 10/20 (50%) | 10/20 (50%) | 10/20 (50%) | 50.0% |
| M3 JEPA + ledger + auxiliary rank + CBF | 8/20 (40%) | 7/20 (35%) | 5/20 (25%) | 33.3% |
| A1 M3 without ledger + CBF | 7/20 (35%) | 6/20 (30%) | 6/20 (30%) | 31.7% |
| A2 M3 without clearance/visibility rank + CBF | 8/20 (40%) | 7/20 (35%) | 5/20 (25%) | 33.3% |

四个变体的 collision、defender boundary、pairwise violation 均为 0，Transit 为
100%，p95 CBF latency 低于 40 ms；CBF controlled abort 已显式记录并排除在
`safe_capture` 之外。M3 相对 M0 的 paired delta 为 `-10/-15/-25 pp`，总体
`-16.67 pp`。

**当前结论：** 安全硬门和实时性门通过，但任务性能 non-inferiority 门未通过。
这证明闭环能安全运行，不证明当前 JEPA 排序带来收益，也不证明 JEPA 架构无效。
当前必须先做 WP-B/WP-E hard replay 和候选可达性审计，再冻结 final paired block；
不得直接把 smoke 负结果写成最终结论或打开 locked test。

### 2.3 当前执行起点

WP-A 边界语义审计和代码修正已经完成：原 episode 19 的越界全部属于 target，
defender 越界为 0。WP-H boundary-fixed smoke 也已完成，原始结果未覆盖，审计产物
见 `docs/JEPA_SAFE_CAPTURE_WP6_SMOKE_BOUNDARYFIXED_20260904.md`。

下一步固定为：

1. WP-B2：对 failure index 中的代表性 degraded episodes 做双次 deterministic hard replay；
2. WP-E：在 replay 证据基础上检查候选动作块的可达性、score margin、切换和 CBF 修正；
3. WP-D/F：根据证据校准 ledger temporal drift/abstain 和 fault-injection fallback，但不得先调参追逐捕获率；
4. 只有诊断完成后，才冻结 40-episode paired final development block。

`locked_test_opened=false` 在整个流程中保持不变。

### 2.4 当前真正的阻塞项

| 阻塞项 | 证据 | 处理顺序 |
|---|---|---|
| M3 任务性能回归 | 相对 M0 的 paired delta 为 `-10/-15/-25 pp`，总体 `-16.67 pp` | 先做 WP-B/WP-E 因果 replay，不先调阈值 |
| CBF controlled abort 偏多 | M0 每 seed 为 `9/9/9`；M3 为 `11/13/14` | 检查候选可达性、修正 norm、active constraints 和 fallback reason |
| ledger 机制是否过度保守 | A1 仅为 `31.7%`，没有显示去掉 ledger 能恢复性能 | 统计 `fallback_nominal`、credit drift、bucket coverage；不得凭直觉放宽门限 |
| failure index 接口 | 已完成 V3 目录发现、配对校验和回归测试 | 保留 v2 兼容；后续只扩展新 protocol，不覆盖当前索引 |
| reliability fault injection | smoke 只证明已观测运行的安全/延迟/provenance；尚未覆盖所有 OOD、stale、non-finite 注入组合 | WP-D/WP-F 完成后才标记 reliability gate 完成 |

因此，当前“可安全运行”不等于“已证明可部署”，当前“prediction gate 通过”也不等于
“控制收益已证明”。所有未覆盖的证据必须保持 `pending`，不得从 smoke 均值推断。

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

## WP-A：边界语义和 smoke 失败审计（已完成）

**目标：** 区分 target 越界、defender 越界、障碍碰撞和 CBF 失效，排除统计标签错误。

- [x] 对 seed `20260912` episode `19` 做 deterministic replay，定位首次越界 step、实体类型和触发 constraint。
- [x] 检查 `pursuit_env.py` 的 target `_enforce_world_bounds` 与 defender `_enforce_world_bounds` 调用路径。
- [x] 增加独立计数器：`target_world_violation_steps`、`defender_world_violation_steps`、`target_boundary_violation`、`defender_boundary_violation`。
- [x] 保留历史字段 `world_violation_steps` 的兼容输出，但在新协议中明确其组成和安全结算来源。
- [x] 将 safe-capture 的 boundary gate 绑定到 defender 越界；target 越界作为独立 target termination/diagnostic。
- [x] 增加 target-only、defender-only、target-boundary-after-capture 回归测试。
- [x] 用独立审计脚本重算 episode 19，未覆盖原始结果。

**验收：** episode 19 的主因可唯一解释；新旧字段映射有测试；报告明确说明历史结果是否受标签语义影响；完整测试不回归。

**产物：** `docs/JEPA_SAFE_CAPTURE_WP6_SMOKE_SAFETY_AUDIT_20260904.md`、
`docs/JEPA_SAFE_CAPTURE_WP6_SMOKE_BOUNDARYFIXED_20260904.md`、独立 audit JSON、
回归测试和 provenance hash。

## WP-B：失败片段 hard replay 与因果归因

**目标：** 把失败分解为预测漂移、ledger 错信、候选排序、CBF 过度修正、QP 不可行或任务几何原因。

- [x] 让 `scripts/index_jepa_safe_capture_failures.py` 同时支持当前
  `jepa_safe_capture_v3_wp6_*_boundaryfixed` 目录和后续 `v3_wp7_*` 目录；
  保留旧 v2 输入兼容，并为目录发现、episode 数和 boundary 字段增加测试。
- [x] 使用适配后的索引脚本建立只读 failure index；索引过程不得修改任何源结果目录。
- [ ] 从 `failure_index.csv` 固定选择 replay 集，不得临时按结果挑样本：M3 相对 M0 回归、high-credit abort、`fallback_nominal` abort、候选振荡、stale/noisy matched case、timeout 各至少 3 集；若类别不足，记录实际数量。
- [ ] 按 `observation -> belief -> JEPA -> ledger -> rank -> CBF -> executed action -> termination` 逐步重放。
- [ ] 每个 step 记录 observation/message age、finite/OOD、candidate validity/reachability、candidate scores/margins、ledger state/credit/fallback reason、CBF active constraints/correction norm/feasibility/solver status、executed action 和 termination。
- [ ] 每个样本至少执行两次 replay；保存输入哈希、逐步 trace 哈希、终止原因、动作序列哈希和环境版本。
- [ ] 无法由 trace 证明的原因标记为 `unresolved`，不得强行归类。
- [ ] 统计 high-credit failure、fallback 后安全率、candidate switch rate 和 CBF abort 关联。

**验收：** 100% 失败 episode 有唯一主因或 `unresolved` 标签；同一 episode replay 两次的终止原因、动作和 trace hash 一致。

**已完成产物：** `results/jepa_safe_capture_v3_wp1_failure_index_current/`（failure index、报告、provenance、TensorBoard）。
**待完成产物：** `results/jepa_safe_capture_v3_wp1_failure_replay_current/`（replay JSONL、逐步 trace、哈希清单）、`docs/JEPA_SAFE_CAPTURE_WP1_FAILURE_REPLAY_20260904.md`。

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

**禁止事项：** 不因 M3 smoke 低于 M0 就直接降低 credit threshold、扩大 stale age
上限或关闭 OOD；任何阈值变更必须写入新的 calibration manifest，并重新通过 WP-D
离线准入和 paired smoke。

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

**优先诊断顺序：** 先验证五类候选在当前速度/加速度/slew 约束下是否真实可达，再
看 score margin 和 top-1 settled outcome，最后才讨论 score 权重。不可达候选必须在
进入 JEPA 前被剔除并单独统计，不能让它们通过低分“参与排序”。

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

## WP-H：M0/M3/A1/A2 smoke 重跑与准入（已完成）

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
- 已运行路径中的 trace/provenance 完整。

任何安全硬门失败都停止该变体，保存 audit，不进入 final block。当前四个变体的
已观测安全与实时性门通过，但 M3/A1/A2 的 safe-capture 均低于 M0；WP-D/WP-F 的
OOD/stale/non-finite 故障注入门仍未完成。因此先进入 WP-B/WP-E 诊断，不立即运行
final block，也不把该结果写成 JEPA 提升。

## WP-I：三 seed paired development final block（诊断完成后）

**目标：** 获得可复核的跨 seed safe-capture 证据，不能只看单个 seed。

- [ ] WP-B/WP-E/WP-D 的诊断报告完成，并明确是否需要另建 development protocol。
- [ ] 冻结 checkpoint、ledger、protocol、scene hash、solver 和 episode seed 列表。
- [ ] 运行 M0、M3、A1、A2；每个 training seed 至少 40 个配对 episode。
- [ ] 同一 `(training_seed, episode_index)` 使用完全相同的 layout、target motion、observation condition 和初始状态。
- [ ] 保存 episode summary、step trace、candidate trace、ledger state、CBF diagnostics、TensorBoard 和命令行 provenance。
- [ ] 不在 final block 中调 threshold、margin、candidate weight、chunk length 或 episode seed。
- [ ] 如果 WP-B/WP-E 证明当前排序存在可解释的实现缺陷，先修复并建立新的
  development protocol；旧 smoke 结果保留为 negative control，不与新协议混合。
- [ ] 如果没有可验证的修复假设，直接将当前结果归档为 task-regression evidence，
  不为了凑 episode 数量而运行 final block。

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

### 7.1 立即执行顺序（当前工作日）

```powershell
Set-Location D:\\uav-capture\\uav_capture
$py = 'D:\\miniconda3\\envs\\uav-encirclement-gpu\\python.exe'
$env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"

# 0. 只读确认环境、版本和工作区边界
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
git status --short

# 1. V3 failure-index 回归测试（当前已通过：8 passed）
& $py -m pytest -q tests/test_jepa_safe_capture_wp1_failure_index.py

# 2. 只读重建/核对当前索引；不要覆盖已有目录
& $py scripts/index_jepa_safe_capture_failures.py `
  --input-root results `
  --input-format v3 `
  --output-dir results/jepa_safe_capture_v3_wp1_failure_index_next `
  --tensorboard-logdir results/jepa_safe_capture_v3_tensorboard/wp1_failure_index_next `
  --freeze-manifest results/jepa_safe_capture_v3_wp0_baseline_freeze_20260904/manifest.json `
  --stage smoke --development-only

# 3. WP-B2 hard replay 完成后，再运行候选可达性和完整测试
& $py -m pytest -q
```

索引产物必须写入新的空目录；不得用临时脚本绕过 failure-index 的命名、配对和
provenance 校验。若索引发现的运行目录不是同一 canonical scene manifest，立即停止
并重建 paired block。

### 7.1.1 WP-B2 replay 命令模板

以下命令是接口约定，脚本尚未完成前不得用手工 notebook 替代；命令执行仍保持
`development_only=true` 和 `locked_test_opened=false`：

```powershell
& $py scripts/replay_jepa_safe_capture_failures.py `
  --failure-index results/jepa_safe_capture_v3_wp1_failure_index_current/failure_index.csv `
  --output-dir results/jepa_safe_capture_v3_wp1_failure_replay_current `
  --tensorboard-logdir results/jepa_safe_capture_v3_tensorboard/wp1_failure_replay_current `
  --repeats 2 --development-only
```

Replay 必须拒绝缺失 source trace、scene hash 不匹配、episode seed 不匹配、非 finite
动作或 locked-test 标记；失败样本可以输出 `unresolved`，但不得静默跳过。

### 7.2 final block 的准入命令模板

final block 只能在 WP-B/WP-D/WP-E/WP-F 的验收条目全部勾选后执行。每个变体和 seed
使用空目录；以下是命名模板，不代表当前立即执行：

```powershell
& $py scripts/evaluate_jepa_safe_capture_v2.py `
  --config configs/central_random_mixed_obstacle_s3_v3_development_protocol.yaml `
  --variant m0 `
  --training-seed 20260911 `
  --episodes 40 `
  --output-dir results/jepa_safe_capture_v3_wp7_m0_seed20260911 `
  --tensorboard-dir results/jepa_safe_capture_v3_tensorboard/wp7_m0_seed20260911 `
  --device cuda
```

实际参数必须以脚本 `--help` 和冻结 protocol 为准；如果 CLI 不支持上述某个参数，
先更新 evaluator/测试/协议，再运行实验，不能通过手工改 summary 或绕过 provenance。

### 7.3 当前准入决策表

| 检查项 | 通过条件 | 不通过时的动作 |
|---|---|---|
| failure index | 所有 smoke 运行被发现、episode 0..19 完整、scene manifest 唯一且 paired | 停止 replay，修复目录发现/配对校验 |
| hard replay | 每个失败 episode 有唯一主因或 `unresolved`，重复回放 trace hash 一致 | 不调模型，先补 trace 或标记 unresolved |
| candidate feasibility | 五类候选的 speed/acceleration/slew/finite 检查通过率可解释，不可达候选不进入 JEPA | 修复候选生成/过滤，重新做 smoke |
| ranking evidence | top-1 settled safe outcome、score margin、switch rate 和 CBF correction 有可解释记录 | 保留负结果，不事后调权重 |
| ledger reliability | OOD/stale/non-finite 100% 显式 fallback；high-credit failure 不高于 low-credit | 冻结 ledger，重新校准并新建 manifest |
| CBF fault injection | infeasible/timeout 不执行 raw action；fallback 顺序可回放；p95 端到端延迟 <= 100 ms | 回退 nominal + CBF，禁止 final block |
| final paired block | M0/M3/A1/A2 同一 scene block、3 seed、每 seed >= 40 集，所有 provenance 完整 | 不凑 episode 数，归档为 insufficient evidence |
| safe-capture 结论 | 安全硬门全通过，且 M3 平均 paired delta >= 0 pp、至少 2/3 seed 非负 | 只能报告 regression/non-inferiority 或 reject |

### 7.4 当前明确不做的事情

- [ ] 不打开新的 locked test，不读取 locked-test split，不修改历史 V4/V5 结论。
- [ ] 不以 `95%` 作为硬目标；`safe_capture` 的安全不劣性和可审计性优先。
- [ ] 不因 mean capture time 变差而绕过 CBF、降低 margin 或取消 fallback。
- [ ] 不把 target 越界写成 defender 安全失败；新报告必须同时列出 target 与 defender 字段。
- [ ] 不把单 seed、smoke 均值、prediction MAE 或 TensorBoard 曲线单独写成控制收益。
- [ ] 不将 `tmp` 中的 archive recovery checkpoint 冒充历史 retained-BC warm start。

本计划完成的定义：

1. 边界、碰撞、CBF、回退和 safe-capture 语义有明确代码和测试支持；
2. JEPA、ledger、ranker、CBF、replan 链路逐步 trace 可回放；
3. 至少三个 training seed、同一 paired scene block、每 seed 至少 40 集完成；
4. 所有安全保留变体通过 collision/boundary/pairwise/zero-perturbation 硬门；
5. 结果按 safe-capture 优先、逐 seed 和配对统计报告，capture time 仅为次指标；
6. 所有产物带 protocol/checkpoint/ledger/scene/code/environment hash，并能从空目录复现最小运行；
7. 在明确授权前，`locked_test_opened=false` 始终保持不变。

**最终判据：** 不是追求某一个 seed 的最高捕获率，而是证明“JEPA 反事实评价 + reliability ledger 拒答 + CBF 硬安全层 + 滚动时域重规划”在多 seed、困难场景和完整审计链路下能够持续提高或至少不损害 `safe_capture`，且世界模型幻觉、目标漂移和集群安全约束不会静默失效。
