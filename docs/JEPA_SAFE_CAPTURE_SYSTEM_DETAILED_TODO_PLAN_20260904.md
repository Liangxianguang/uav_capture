# 无人机集群对抗围捕安全增强系统
# 详细 TODO、实验与验收计划书

**系统路线：** Interaction-aware Action-conditioned JEPA + Reliability Ledger + Joint CBF-QP + Receding-Horizon Control  
**版本：** v1.0  
**日期：** 2026-09-04  
**适用环境：** Windows、Conda `uav-encirclement-gpu`、NVIDIA RTX 5050  
**实验边界：** `development_only=true`，`locked_test_opened=false`  
**第一指标：** `safe_capture`  
**次要指标：** capture time、路径长度、CBF 修正代价、fallback 率、延迟

> 本计划的目标不是追逐某个 seed 的最高捕获率，而是完成一条可审计、可回退、可重复的安全闭环。`mean_capture_time` 不得抵消碰撞、边界越界、机间净空破坏或未验证控制输出等安全失败。当前结果只能支持“安全架构已具备正向证据”，不能直接声称 JEPA 已获得稳健控制收益。

## 1. 最终目标

针对四机协同围捕一个具有逃逸行为的目标，完成以下闭环：

```text
多机观测/通信历史
    -> interaction-aware belief state
    -> 冻结传统规划器生成动力学可行候选 action chunks
    -> action-conditioned JEPA 反事实预测
    -> reliability ledger 可信度判定
    -> 安全优先候选排序
    -> Joint CBF-QP 最终安全过滤
    -> 只执行第一控制步
    -> 重新观测、更新 belief、重新规划
```

系统必须证明：

1. JEPA 只评价和排序候选轨迹，不生成、覆盖或绕过最终控制动作。
2. JEPA 能在相同 belief 下区分不同候选对目标运动、可见性、净空和 CBF 干预的影响。
3. Reliability ledger 能在 stale、OOD、预测漂移、non-finite 或信用不足时拒答并回退。
4. Joint CBF-QP 是不可绕过的执行边界；任何 QP 失败都不能执行 raw/unverified action。
5. 滚动时域只执行 action chunk 的第一步，下一周期重新观测、预测、排序和过滤。
6. 在多 seed、困难场景和完整 provenance 下，`safe_capture` 不因引入学习模块而降低；若有提升，提升必须可复现且有配对证据。

## 2. 当前证据和边界

### 2.1 已完成

- P0 协议、数据边界、safe-capture 语义和 `locked_test_opened=false` 已冻结。
- P1 困难场景/反事实 archive、episode split、calibration split 和哈希审计已完成。
- P2 三 seed JEPA 训练及 held-out prediction gate 已完成，主要 horizon 的 target MAE 优于 constant-velocity。
- P3 checkpoint-bound reliability ledger v3 已完成，包含 OOD/stale/non-finite 回退接口。
- P5 Joint CBF-QP 的单元测试、确定性、zero-perturbation 和显式 fallback 审计已完成。
- P7 development 运行已完成：M0/M1/M2/M3/A1/A2/A3，三 seed、每 seed 40 集，未打开 locked test。
- 现有运行均记录了 summary、episodes、step traces、scene manifest、provenance 和 TensorBoard。

### 2.2 目前能说什么

P7 development 中，M3（JEPA + ledger + 辅助安全排序 + CBF）为 `36.7% +/- 10.1%`，M0（nominal + CBF）为 `32.5% +/- 0.0%`；120 个配对 episode 的平均差值为 `+4.17 pp`，bootstrap 95% CI 为 `[-4.17, +12.50] pp`，非负 seed 为 2/3。所有安全保留变体的 collision、boundary、pairwise violation 为 0；A3 raw/no-CBF 明确暴露碰撞风险。

因此当前结论只能是：

- 安全链路和故障可观测性得到支持；
- JEPA 排序存在正向开发证据，但尚未证明稳健控制收益；
- 下一步优先做失败重放、候选可达性、排序因果和 fault injection；
- 不因平均 capture time、单 seed 高分或 smoke 均值打开 locked test。

## 3. 不可变系统合同

### 3.1 信息和数据边界

- 在线输入只能是 defender 状态、target belief、观测/通信历史、障碍几何、边界、动作历史和时间戳年龄。
- target ground truth 只允许用于离线 settled label 和 episode 结算，字段必须标记 `offline_only=true`。
- train、validation、calibration、development、locked episode 按 episode/layout seed 隔离。
- 失败 development episode 不得直接回灌训练；若要重放训练，必须新建协议和新 archive。
- 每次运行必须写入 protocol、checkpoint、ledger、scene、代码、环境版本和命令行哈希。

### 3.2 动作合同

- 候选数固定 `K=5`：`nominal`、`intercept`、`lateral_clearance`、`formation_clearance`、`visibility_hold`。
- action chunk 固定为 3 个 control steps；只执行第 1 步。
- 每个候选进入 JEPA 前通过 finite、shape、speed、acceleration、slew 和 reachability 检查。
- 不可达候选不得进入 JEPA，必须记录拒绝原因。
- 所有 candidate 和 nominal 使用同一 Joint CBF-QP、margin、solver、tolerance 和 timeout。

### 3.3 安全合同

`safe_capture=true` 必须同时满足：

1. 至少一个 defender 进入 `0.80 m` capture radius；
2. 在 episode time limit 内完成；
3. 无 obstacle、target 或 defender-defender collision；
4. 无 defender boundary/altitude violation；
5. 无 pairwise separation violation；
6. 无 CBF infeasible、timeout、unverified action 或 controlled-abort 终止。

CBF 失败时的固定顺序为：

```text
separation-preserving safe-hold
    -> verified nominal through Joint CBF-QP
    -> controlled abort
```

禁止把 raw desired action 作为最后回退。所有回退必须有 reason code、solver status、feasibility、slack、correction norm 和 latency。

## 4. 目标架构与接口

### 4.1 Belief state

每个控制周期构造 `BeliefState_t`，至少包含：

- defender position/velocity、formation geometry 和 pairwise TTC；
- target belief position/velocity、visibility probability、observation age 和 message age；
- obstacle geometry、boundary/altitude margin 和局部 clearance；
- 最近 `H` 个 observation/action/communication tokens；
- 上一周期 CBF 状态、fallback 状态和 solver latency；
- layout signature、target motion mode 和 provenance id。

### 4.2 JEPA 评价器

对每个 candidate action chunk 输出未来评价向量，而不是控制动作：

```text
target displacement / velocity / acceleration
obstacle-clearance lower quantile
inter-agent clearance lower quantile
pairwise TTC
visibility probability / observation-age risk
CBF intervention probability / correction magnitude / QP feasibility
predictive uncertainty / ensemble disagreement
```

建议排序分解：

```text
score(k) = task_progress
         + visibility_gain
         - clearance_risk_lower_quantile
         - cbf_intervention_cost
         - uncertainty_penalty
         - action_change_cost
         - nominal_anchor_penalty
```

预测安全量只能用于排序和 ledger 校准，不能替代 CBF 的真实几何约束。

### 4.3 Reliability ledger

ledger 在 calibration 后只读，并绑定 checkpoint/protocol/calibration archive hash。建议状态机：

| 状态 | 触发条件 | 允许动作 |
|---|---|---|
| `trusted` | bucket 足够、credit 足够、uncertainty 和 stale age 在阈值内 | 允许 JEPA 排序 |
| `fallback_nominal` | credit 下降、候选分离消失、预测漂移或 bucket 缺失 | nominal，经 CBF 过滤 |
| `safe_hold` | OOD、non-finite、过期观测、连续失败或 hash 不一致 | safe-hold，经 CBF 过滤 |
| `controlled_abort` | safe-hold/nominal-CBF 也无法验证可行 | 终止并记录，不计 safe capture |

credit 不是安全证明；低信用必须改变执行路径，而不是只降低 score。

## 5. 工作包和详细 TODO

### WP-0：基线、协议和环境冻结（已完成，持续守护）

- [x] 冻结 RTX 5050、Conda 环境、Python/PyTorch/CUDA 版本。
- [x] 冻结 protocol、scene manifest、episode seed、checkpoint、ledger 和 CBF 参数。
- [x] 固定 `development_only=true` 和 `locked_test_opened=false` 运行时检查。
- [ ] 每次新运行前自动比较 manifest hash；不一致立即拒绝启动。
- [ ] 为每个阶段生成独立 output/logdir，不覆盖历史结果。

**出口：** 任一安全参数、split、hash 或 locked 标记不一致时运行器拒绝启动。

### WP-1：失败索引和 deterministic hard replay（当前最高优先级）

- [x] 从现有 V3 smoke 结果建立 failure index，保留源 run、episode、scene 和 trace identity。
- [x] 固定选择代表性失败：candidate regression、high-credit failure、fallback nominal、candidate oscillation、stale/noisy、timeout；六类各选 3 集，无 shortage。
- [x] 逐步重放 `observation -> belief -> JEPA -> ledger -> rank -> CBF -> action -> termination`。
- [x] 每一步记录 observation/message age、candidate validity/reachability、score/top-two margin、ledger state/credit/fallback、CBF feasibility/active constraints/slack/correction/latency 和 termination。
- [x] 每个样本重复 replay 两次，要求终止原因、动作序列和 canonical trace hash 完全一致；18/18 通过。
- [x] 无法由 trace 证明的因果标记为 `unresolved`，不得凭猜测归因。

**产物：** `failure_index`、`replay_manifest`、逐步 JSONL、hash manifest、Markdown 报告、TensorBoard。  
**出口：** 每个失败有唯一主因或 `unresolved`；重复 replay 完全确定。

### WP-2：JEPA 多任务和不确定性准入

- [x] 保留 target displacement 多 horizon 预测并完成三 seed prediction gate。
- [ ] 增加 velocity/acceleration consistency、clearance lower-quantile、pairwise TTC、visibility、observation-age、CBF intervention 和 QP feasibility heads。
- [ ] 增加 flee persistence、turn、S-curve、突变加速度等 motion-mode embedding。
- [ ] 增加 action-conditioned contrastive/consistency loss，验证不同候选产生可辨识未来表示。
- [ ] 采用 ensemble disagreement、heteroscedastic residual 或 calibrated residual 产生 uncertainty。
- [ ] 在 held-out calibration split 上计算 MAE、coverage、Brier/AUC、rank consistency 和 uncertainty calibration。

**出口：** 三 seed 输出 finite；主要 horizon 超过 constant-velocity；辅助头标签非空且有校准证据；不把 prediction gate 当成控制收益证明。

### WP-3：Reliability ledger temporal/adversarial 校准

- [ ] 在 calibration-only split 上测连续残差突增、credit 连续下降和 candidate separation 消失。
- [ ] 注入 stale observation、消息延迟/丢包、遮挡、急转、速度突变、障碍密度 shift 和队形拥挤度 shift。
- [ ] 冻结最小 bucket 样本数、最低 credit、uncertainty 上限、stale age 上限和 OOD 规则。
- [ ] 验证 `trusted -> fallback_nominal -> safe_hold -> controlled_abort` 状态转移可回放。
- [ ] 验证 OOD/stale/non-finite 100% 显式回退，且 raw/unverified action 数为 0。
- [ ] 锁定 ledger 文件和 hash，运行期间禁止在线更新 credit/threshold。

**出口：** high-credit settled failure rate 不高于 low-credit；每次 abstain 都有 reason code、状态转移和 trace。

### WP-4：候选动作块可达性和排序诊断

- [x] 对五类候选统计 finite、speed、acceleration、slew、reachability 通过率；当前 6577 ranking steps 的五类 valid fraction 均为 1.0。
- [ ] 统计每类候选被拒原因；历史 trace 未记录 `rejection_reasons`，已明确标记 observability gap；不可达候选不得进入 JEPA 的 invariant 已通过。
- [x] 记录 task progress、lower-quantile clearance、visibility gain、uncertainty、CBF cost、action-change cost 和 nominal anchor 的逐项 score（保留在 source trace 并完成有限性审计）。
- [x] 统计 selected-trajectory safe outcome、top-two margin、switch rate 和 oscillation；CBF intervention 继续由 WP-5 单独审计。
- [ ] 计算预测排序与 settled outcome 的 rank correlation、top-1 precision/recall 和分桶校准；历史 trace 缺少 per-candidate settled counterfactual label，已保持 gate false。
- [ ] 只有有明确证据时才调整 score；任何调整都新建 calibration manifest 和 protocol revision。

**出口：** 可达性 invariant 通过，但排序因果证据仍不完整；在新增 rejection reason 和 offline counterfactual label 前不得调权重或进入 final block。

### WP-5：Joint CBF-QP 故障注入和实时性

- [x] 已覆盖主要障碍、边界、pairwise、速度/加速度和 slew 约束的基础单测。
- [ ] 注入 QP infeasible、solver timeout、non-finite request、stale observation、通信中断、多约束同时激活。
- [ ] 验证所有候选和 nominal 走同一 CBF；验证 ranker/JEPA 不能覆盖 filtered action。
- [ ] 统计 agent 数量、障碍数、队形密度和边界压力下的 feasibility 和 p50/p95/p99 latency。
- [ ] 验证 infeasible/timeout/unverified 时 raw action 执行计数始终为 0。
- [ ] 记录 solver version、status、message、active set、minimum slack、residual、correction norm、fallback reason。

**出口：** collision/boundary/pairwise 为 0；raw/unverified action 为 0；端到端 p95 不超过 100 ms；否则停止进入 final block。

### WP-6：滚动时域闭环集成审计

- [ ] 固定每周期顺序：时间戳检查、belief 更新、候选生成、可达性过滤、JEPA batch、ledger、ranking、CBF、执行第一步。
- [ ] JEPA、ledger、ranker 或 CBF 任一超时立即走显式 fallback。
- [ ] 记录输入 hash、预测、uncertainty、credit、selected candidate、CBF 诊断、动作和全链路 latency。
- [ ] 做 CPU 与 RTX 5050 的 deterministic replay；允许浮点容差，但不得改变安全结算。
- [ ] 从空 output 目录重跑 1 episode，核对 summary、episodes、step traces、TensorBoard 和报告。

**出口：** 每个 episode 都有完整闭环 trace；只执行第一步；下一周期确实重新观测和 replan。

### WP-7：三 seed paired development final block

前置条件：WP-1、WP-3、WP-4、WP-5、WP-6 全部通过，并冻结新 protocol（如有修改）。

- [ ] 变体固定为 M0、M3、A1、A2；A3 raw/no-CBF 只作诊断。
- [ ] 使用同一 paired scene manifest、episode index、layout、target motion 和 observation schedule。
- [ ] 每个变体、每个 training seed 至少 40 个 episode；每个结果目录全新创建。
- [ ] 不在 final block 中调 threshold、CBF margin、candidate weight、chunk length 或 episode seed。
- [ ] 每个 run 写 summary、episodes.csv、step traces、scene manifest、provenance、TensorBoard 和命令行记录。
- [ ] 任一安全硬门失败，立即停止该变体并回退冻结 nominal + CBF；不得继续凑 episode 数。

**主比较：** M3 vs M0。  
**消融：** A1 去 ledger；A2 去 clearance/visibility 排序项；A3 仅诊断 CBF 必要性。

### WP-8：统计、审计和结论归档

- [ ] 以 `(training_seed, episode)` 为独立统计单位，不把 timestep/chunk 当独立样本。
- [ ] 报告逐 seed safe_capture、collision、boundary、pairwise、CBF abort、fallback、high-credit failure 和 latency。
- [ ] 计算 mean、sample SD、paired delta、improved/degraded/tied、bootstrap 95% CI 和 exact McNemar。
- [ ] 按 motion mode、visibility、observation age、clearance、ledger state、CBF active constraint 分桶。
- [ ] 对 JSON、CSV、TensorBoard、Markdown 做双向一致性检查。
- [ ] 将结论归入预定义类别：`safe_capture_improvement_candidate`、`safety_preserving_noninferior`、`prediction_signal_no_control_gain`、`rejected_for_safety`、`insufficient_evidence_do_not_open_locked_test`。

## 6. 实验矩阵

| 阶段 | 变体 | seed | episode | 目的 |
|---|---|---:|---:|---|
| hard replay | M0/M3/A1/A2 失败样本 | 现有 3 seed | 固定代表集 | 因果归因和确定性 |
| candidate audit | M0/M3 | 3 | 离线全 trace | 可达性、排序和切换 |
| fault injection | M0/M3 | 3 | 固定注入矩阵 | ledger/CBF 故障门 |
| paired development | M0/M3/A1/A2 | 3 | >=40/seed | 主 safe-capture 比较 |
| robustness stress | M0/M3 | 3 | 独立 hard block | 延迟、遮挡、拥挤、shift |
| raw diagnostic | A3 | 3 | 与 paired block 对齐 | 仅显示无 CBF 风险 |

统计原则：`safe_capture` 为首要指标；capture time、路径长度、CBF 修正和延迟只在安全硬门通过后作为诊断指标报告。

## 7. 验收门和停止规则

| 门 | 通过条件 | 不通过动作 |
|---|---|---|
| 数据/协议 | split、scene、checkpoint、ledger、代码 hash 一致 | 停止并重建 manifest |
| Replay | 失败有主因或 unresolved，双次 hash 一致 | 不调参，补 trace/标记 unresolved |
| Prediction | 三 seed finite，主要 horizon 优于 constant-velocity | 回到离线训练/标签审计 |
| Reliability | OOD/stale/non-finite 100% 显式 fallback | 冻结 ledger，重新校准 |
| Candidate | 不可达候选 0 个进入 JEPA，可达率可解释 | 修复候选生成/预检查 |
| Ranking | top-1 与 settled outcome 关系可解释，无安全回归 | 保留负结果，不事后调权重 |
| CBF | collision/boundary/pairwise=0，raw=0，p95<=100 ms | 回退 nominal + CBF，禁止 final block |
| Paired task | 平均 paired delta >= 0，至少 2/3 seed 非负 | 只能报告 non-inferiority/regression |
| Locked readiness | 所有前置门和统计报告完成 | 保持 `locked_test_opened=false` |

明确禁止：

- 不以 `95%` 作为硬目标；
- 不用 mean capture time 掩盖 safe-capture 下降或安全失败；
- 不降低 CBF margin、关闭 OOD/stale 检查或扩大 stale age 以追逐捕获率；
- 不把 target 越界错误写成 defender boundary violation；
- 不把单 seed、smoke 结果或 prediction MAE 写成正式控制收益；
- 不把 `tmp` 中 archive-recovery checkpoint 冒充历史 V4 warm-start checkpoint；
- 未经单独授权不读取或打开新的 locked-test split。

## 8. RTX 5050 执行纪律

```powershell
Set-Location D:\\uav-capture\\uav_capture
$py = 'D:\\miniconda3\\envs\\uav-encirclement-gpu\\python.exe'
$env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
& $py -m pytest -q
```

- 训练优先使用 CUDA；评估需补做 CPU 一致性审计。
- 每个阶段使用独立空 output/logdir；禁止覆盖历史 results、checkpoint、NPZ 或 TensorBoard。
- TensorBoard 必须有 config/provenance text、阶段 scalar、事件文件存在性检查和必要 tag 校验。
- Git 只提交明确列出的代码、配置、测试和报告文件，不使用 `git add .`。
- 每个工作包形成独立 conventional commit；网络/认证失败时保留本地 commit 并记录原因。

## 9. 建议时间盒和交付物

| 时间盒 | 工作 | 交付物 |
|---|---|---|
| Day 1--2 | WP-1 hard replay | replay JSONL、hash、因果报告 |
| Day 2--3 | WP-4 candidate audit | feasibility/ranking CSV、分桶报告 |
| Day 3--5 | WP-3/WP-5 fault injection | ledger/CBF 注入报告、延迟报告 |
| Day 5 | protocol decision | 新 protocol 或负结果归档 memo |
| Day 6--9 | WP-6 + smoke | 完整闭环 trace、M0/M3 smoke |
| Day 10--14 | WP-7 final development | 4 变体 x 3 seed x >=40 episodes |
| Day 15--16 | WP-8 statistics | paired comparison、CI、McNemar、最终报告 |
| Day 17+ | SIL/HIL（可选） | 仅在所有前置门通过后进行 |

## 10. 完成定义

本计划只有在以下条件全部满足时才算完成：

1. 系统架构的 JEPA、ledger、ranker、Joint CBF-QP 和 rolling loop 接口均有代码、测试和逐步 trace 支持。
2. 失败样本可 deterministic replay，所有无法证明的原因显式标记 `unresolved`。
3. 所有异常输入、QP 失败和模型低信用路径均不会执行 raw/unverified action。
4. 三个 training seed、同一 paired scene block、每 seed 至少 40 集完成并可独立复核。
5. 所有安全保留变体通过 collision、boundary、pairwise、zero-perturbation 和 latency 硬门。
6. 最终结论以 `safe_capture` 为第一指标，包含逐 seed 和配对统计；capture time 仅作诊断。
7. 结果、代码、协议、checkpoint、ledger、场景和环境均有 hash/provenance；从空目录可重跑最小 episode。
8. 在获得明确授权前，`locked_test_opened=false` 始终保持不变。

**核心判断：** 只有当 JEPA 的反事实排序、ledger 的可信度拒答、CBF 的硬安全边界和滚动时域重规划在多 seed 困难场景中共同通过上述证据门，才能把该方案称为“安全增强的闭环围捕系统”；否则应诚实归类为 prediction signal、safety infrastructure 或 development evidence，而不是稳健性能提升。
