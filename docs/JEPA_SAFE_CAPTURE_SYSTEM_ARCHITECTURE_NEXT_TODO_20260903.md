# 面向无人机集群对抗围捕的安全增强世界模型系统
# Interaction-Aware Action-Conditioned JEPA + Reliability Ledger + CBF
# 下一阶段 TODO 目标计划书

**版本：** v1.0
**制定日期：** 2026-09-03
**适用范围：** 无人机集群对抗围捕/拦截仿真、SIL、HIL 和后续受控实飞准备
**当前阶段：** development-only；未授权前不得打开新的 locked test
**GPU 环境：** NVIDIA RTX 5050，Conda `uav-encirclement-gpu`
**核心目标：** 把世界模型作为候选轨迹评价器，而不是控制动作生成器；在不牺牲集群安全的前提下，提高 safe capture 的可靠性。

---

## 1. 研究目标和当前证据

### 1.1 最终系统目标

针对三维无人机集群围捕对抗目标，建立以下闭环：

```text
多机观测/通信历史
        |
        v
交互状态表示与目标 belief
        |
        v
冻结传统规划器生成 K 个物理可行候选动作块
        |
        v
action-conditioned interaction-aware JEPA
  预测目标、障碍、机间净空、可见性和 CBF 干预风险
        |
        v
reliability ledger 校验预测信用、漂移和 OOD 状态
        |
        +--> 高信用：只重排序候选
        +--> 低信用/OOD：确定性回退 nominal
        |
        v
CBF/QP 最终安全过滤器
        |
        v
只执行动作块第一步，重新观测、重新预测、重新规划
```

JEPA 永远不能直接输出未经 CBF 过滤的飞行动作，不能读取在线 target truth，不能修改环境动力学，不能在线更新 actor 或 ledger，也不能自行声明“安全”。

### 1.2 P5-P7 已有结果的边界

当前三 seed development 结果为：

| 结果 | 数值 | 可以支持的结论 |
|---|---:|---|
| JEPA + CBF safe capture | `58/60`, `57/60`, `56/60` | 均值 `95.0% +/- 1.67%`，没有稳定提升 |
| Collision / boundary | `0/180` / `0/180` | 当前 CBF 集成未引入新的不安全终止 |
| Transit | `100%` | 任务链路可运行 |
| 配对结果 | `6` improved / `6` degraded / `168` tied | 排序收益与退化相互抵消 |
| 统计分类 | `prediction_improvement_no_control_gain` | 不打开 locked test |

因此下一阶段不是重新声称 JEPA 已经有效，而是验证和修复三类具体问题：

1. 世界模型是否在对抗目标、延迟观测和多机交互下发生 rollout 漂移或幻觉；
2. ledger 是否能在高置信但错误的候选排序前及时 abstain；
3. 候选轨迹排序是否真的提高 safe capture，而不只是保持 CBF 的安全外壳。

---

## 2. 不可变设计原则

以下原则从本计划开始写入所有配置和运行时校验，不因单个 seed 或单个场景修改：

| 原则 | 强制规则 |
|---|---|
| 评价器而非控制器 | JEPA 只产生候选轨迹的预测/评分特征，不直接产生最终 action |
| 候选先于模型 | 候选必须先通过动力学、速度、加速度和任务几何可行性检查 |
| 安全最后执行 | baseline 和 candidate 都经过同一 CBF；CBF 是最后一道动作过滤器 |
| 低可信度宁可不使用模型 | ledger 缺失、低信用、OOD、过期观测或校准失败时回退冻结 nominal + CBF |
| 重新规划 | 实际只执行候选块第一步，下一控制周期重新观测和排序 |
| 安全优先 | safe capture、collision、boundary、机间分离优先于 capture time 和路径长度 |
| 无数据泄漏 | train、validation、development 和 locked 场景、seed、ledger 标签严格分离 |
| 可审计 | 每次训练、推理、回退、CBF 修正和失败必须有 provenance、TensorBoard 或结构化日志 |
| 结果诚实 | 没有正向闭环证据时只能写 safety-preserving 或 prediction-only，不写 performance gain |

---

## 3. 系统架构规格

### 3.1 输入和 belief 状态

建立固定的 `BeliefState_t`，至少包含：

- 每个 defender 的位置、速度、当前意图和最近动作历史；
- target 的位置、速度、估计加速度、运动模式和观测年龄；
- 所有障碍物的几何描述、估计净空和边界距离；
- 每一对 defender 的相对位置、相对速度、通信状态和预测碰撞时间；
- 可见性、消息延迟、消息丢失、观测噪声和传感器新鲜度；
- 上一周期 nominal action、CBF 修正量、候选 index、ledger bucket 和回退原因。

禁止把在线 target ground truth 写入 `BeliefState_t`。ground truth 只能在离线标签结算阶段使用，并且必须在日志中标明 `offline_only=true`。

### 3.2 候选动作块生成器

传统规划器先生成物理可行的 `K` 个 desired-action chunks。候选库至少包含：

1. 冻结 V5 nominal chunk；
2. 沿捕获方向的保守侧向候选；
3. 维持机间净空的侧向/纵向候选；
4. 降速、悬停或绕障候选；
5. 在目标运动不确定时提高可见性或保持观测的候选。

候选动作块必须记录 `candidate_id`、动作幅值、chunk 长度、预检查结果和失败原因。候选数量、扰动范围和 chunk 语义在一次实验中冻结。实际执行仍是第一步后重新规划，不能把三步反事实标签误写成三步 open-loop 执行。

### 3.3 JEPA 评价器

对每个候选动作块 `a^(k)`，模型输出多时域预测：

```text
{target displacement, target velocity/acceleration,
 obstacle clearance lower quantile,
 inter-agent clearance lower quantile,
 visibility probability and observation age,
 CBF intervention probability and correction magnitude,
 predictive uncertainty / ensemble disagreement}
```

模型输出的是候选评价向量 `q^(k)`，不是控制动作。排序分数必须只由已声明的预测量、信用权重和安全代价组成，例如：

```text
score(k) = task_progress(k)
           - uncertainty_penalty(k)
           - clearance_risk(k)
           - cbf_intervention_risk(k)
           + visibility_gain(k)
           - action_change_cost(k)
```

`clearance_risk` 使用保守下分位数而不是均值；任何预测的安全量都不得绕过 CBF。

### 3.4 Reliability ledger

ledger v2 采用 checkpoint-bound、validation-only、只读结构，按以下上下文分桶：

- visibility condition 和 observation age；
- obstacle count、布局签名和局部最小净空；
- target motion mode、速度比例和对抗性变化；
- defender side、队形拥挤度和 pairwise time-to-collision；
- JEPA uncertainty、candidate separation 和 CBF intervention risk。

每个 bucket 记录样本数、预测误差、覆盖率、校准误差、safe-capture 条件成功率和置信区间。ledger 输出三态决策：`trusted`、`fallback_nominal`、`hard_stop_or_safe_hold`。低信用不能只降低 score，必须明确回退行为。

### 3.5 CBF 安全过滤器

CBF 对每个控制周期同时约束：

- obstacle separation；
- defender-defender pairwise separation；
- boundary/高度/速度/加速度约束；
- target capture zone 的接近约束；
- CBF QP 可行性、修正量上限和连续性。

若 QP 不可行或观测过期，按固定优先级执行：

```text
safe-hold / separation-preserving action
  -> frozen nominal action through CBF
  -> mission abort or controlled hover with explicit reason
```

不得因为追求 safe capture 而放宽机间净空、边界或障碍约束。CBF 修正量、QP infeasible 次数、最小净空和 fallback 必须逐 episode 记录。

---

## 4. P0：安全合同和数据边界冻结

### TODO

- [ ] 新建版本化配置 `configs/jepa_safe_capture_v2_protocol.yaml`。
- [ ] 写明 train/validation/calibration/development/locked 的目录、seed 和 hash。
- [ ] 固定控制周期、最大速度/加速度、最小障碍净空和最小机间净空。
- [ ] 定义 safe capture、unsafe capture、timeout、safe-hold、CBF infeasible 的互斥终止语义。
- [ ] 定义候选数量、chunk 语义、执行第一步后重规划和 nominal fallback 行为。
- [ ] 为每一项安全约束写出离散时间 CBF 不等式和单位测试。
- [ ] 将 `locked_test_opened=false` 作为运行时硬校验。

### 准入标准

- 配置通过 schema 校验；
- 同一配置能生成完整 provenance manifest；
- 任意试图读取 locked 数据或替换安全参数都会失败；
- 单元测试覆盖单位、边界、动作幅值和终止语义。

### 输出与提交

- `configs/jepa_safe_capture_v2_protocol.yaml`；
- `docs/JEPA_SAFE_CAPTURE_P0_CONTRACT_*.md`；
- P0 独立 `git commit`；
- 不启动训练，不打开 locked test。

---

## 5. P1：困难片段和安全标签归档

### TODO

- [ ] 从现有 development 运行中只提取失败上下文的**状态摘要**，不把同一 episode 直接回灌为训练样本。
- [ ] 新采集 delayed/noisy、flee-persistence、s-curve、高 obstacle count、高拥挤度和低 visibility 场景。
- [ ] 为每个 state-agent-candidate 生成多个 horizon 的 settled rollout 标签。
- [ ] 记录 target displacement、速度/加速度、obstacle/inter-agent clearance 下分位数、visibility、CBF 修正和 QP 可行性。
- [ ] 增加 action perturbation 方向对称样本，检验 action-following antisymmetry。
- [ ] 进行 episode-level split；同一 episode、layout seed 和近邻场景不得跨 split。
- [ ] 生成独立 calibration split，禁止参与 JEPA 参数训练。

### 数据合同

```text
每个 (episode_seed, time_index, agent_id) -> K 个候选
每个候选 -> 相同 state/history 下的多时域 settled outcome
target truth -> offline label only
development/locked episode -> never used by train or ledger fitting
```

### 准入标准

- 所有 arrays finite；每组候选数量一致；
- train/validation/calibration seed 不重叠；
- 场景、环境、actor、采集脚本和 protocol 都有 SHA-256；
- 通过 groupby/lexsort 审计，不依赖 NPZ 原始行连续性；
- hard-case archive 有独立 manifest，不覆盖历史 v2 archive。

### TensorBoard 与提交

- 采集阶段记录 `Data/*` scalar、场景计数、标签覆盖率和 hash text；
- 写入 `results/jepa_safe_capture_v2_data_audit.json`；
- P1 独立 `git commit`，archive/NPZ 默认不提交 Git。

---

## 6. P2：增强 action-conditioned interaction-aware JEPA

### 模型任务

- [ ] 保留 target displacement 多时域预测；
- [ ] 新增 target velocity/acceleration consistency head；
- [ ] 新增 obstacle clearance 的 lower-quantile 或 distributional head；
- [ ] 新增 inter-agent clearance 和 pairwise time-to-collision head；
- [ ] 新增 visibility probability、observation-age 和 stale-observation head；
- [ ] 新增 CBF intervention probability、correction magnitude 和 QP feasibility head；
- [ ] 新增 action-conditioned contrastive/consistency loss，强制不同候选动作产生可辨识的未来表示；
- [ ] 增加 ensemble、dropout 或 calibrated residual 以获得 uncertainty，而不是只用单一 MSE。

### 训练合同

- [ ] 只使用 P1 train archive；validation 和 calibration 只用于评估/校准；
- [ ] 固定 optimizer、batch size、epoch、随机 seed 和 precision；
- [ ] P4 replay-on 负结果不直接复用；任何重放策略必须作为新版本单独预注册；
- [ ] 训练时记录梯度、激活、预测分布和各任务损失，防止辅助任务压制 safety head。

### TensorBoard 必须记录

```text
Loss/total, Loss/target, Loss/velocity, Loss/clearance,
Loss/inter_agent, Loss/visibility, Loss/cbf_risk, Loss/action_consistency
Metric/*/MAE, Metric/*/Brier, Metric/*/AUROC, Calibration/*,
Uncertainty/*, Optimization/learning_rate, Data/*, Provenance/*
```

每个 seed 必须有 40 个以上 epoch 点、完整 text provenance、histogram、checkpoint SHA-256 和环境信息。训练结束后运行 TensorBoard audit，缺失任何必需 tag 都不能进入 P3。

### 准入标准

- 所有任务输出 finite；
- 至少一个预定义 horizon 的 target 预测优于 constant velocity；
- clearance/visibility/CBF 风险 head 有非空覆盖和可校准输出；
- action-following separation 非零，且方向一致性不能退化为随机；
- 三 seed 结果全部可追溯。

---

## 7. P3：Reliability Ledger v2 和 abstention 校准

### TODO

- [ ] 在独立 calibration split 上建立 local、global 和 OOD bucket；
- [ ] 对每个预测 head 计算 Brier、ECE、coverage、误差分位数和 conformal-style lower bound；
- [ ] 将 uncertainty、visibility、observation age、candidate separation 和 CBF risk 纳入信用函数；
- [ ] 定义最小样本数和最小信用阈值，并在看到 development 结果前冻结；
- [ ] 设计 `trusted/fallback/safe-hold` 三态策略，记录触发原因；
- [ ] 做 temporal drift audit：连续 rollout 中信用下降时必须触发回退；
- [ ] 做 adversarial shift audit：目标转向、速度突变、遮挡和消息延迟增加时不允许信用虚高；
- [ ] 将 ledger 与 checkpoint、calibration archive、protocol 的 hash 绑定；
- [ ] 完成后将 ledger 设为只读，运行时禁止在线更新。

### 安全准入门

- high-credit bucket 的 settled safe-capture 条件成功率必须显著高于 low-credit bucket；
- high-credit bucket 不能出现系统性高净空误判或 CBF infeasible；
- 所有 OOD/低信用样本都能观察到确定性 fallback；
- ledger 本身不得被描述为安全证明，CBF 仍是最后防线。

### 输出与提交

- `reliability_ledger_v2.json`、校准报告、bucket coverage 图和失败示例；
- TensorBoard `Calibration/*`、`Reliability/*`、`Fallback/*`；
- P3 独立 `git commit`。

---

## 8. P4：候选轨迹评价和排序改造

### TODO

- [ ] 将候选评价拆为任务收益、净空风险、可见性收益、CBF 代价和不确定性惩罚；
- [ ] 使用预测的保守净空下界，不使用 clearance 均值作为安全分数；
- [ ] 对预测冲突采用安全优先排序：任意 candidate 触发 obstacle/inter-agent 风险时降级；
- [ ] 加入 nominal anchor：排序不能任意远离冻结 nominal，除非预测信用和安全裕度同时满足；
- [ ] 记录候选排名、top-1 与 nominal 的差异、候选分数间隔和最终 CBF 修正；
- [ ] 评估 action chunk smoothness、候选切换频率和连续周期 oscillation；
- [ ] 对 candidate ranking 做 counterfactual rank consistency 检查；
- [ ] 不以单一 capture time 或路径最短作为 rank target。

### 预定义消融

| 变体 | JEPA | Ledger | CBF | 用途 |
|---|---|---|---|---|
| frozen nominal | - | - | on | 主基线 |
| JEPA evaluator | on | on | on | 主候选 |
| no ledger | on | off | on | 诊断漂移代价 |
| no clearance heads | on | on | on | 安全辅助任务消融 |
| no visibility heads | on | on | on | 观测鲁棒性消融 |
| no CBF | on/off | on/off | off | 仅仿真诊断，不能作为部署结果 |

所有消融必须使用同一场景、episode seed 和运行时版本；`no CBF` 结果不得参与安全主结论。

---

## 9. P5：CBF 和集群安全增强

### TODO

- [ ] 验证每对 defender 的离散 CBF 约束和安全裕度；
- [ ] 增加 obstacle、boundary、速度/加速度和高度约束的联合 QP 测试；
- [ ] 测试 QP infeasible、观测过期、通信丢失和多个约束同时激活的确定性 fallback；
- [ ] 设置 CBF correction norm、连续修正次数和最小净空的告警阈值；
- [ ] 增加 CBF 后动作与下一周期候选状态的一致性日志；
- [ ] 在 agent 数量、队形密度和障碍数增加时测试可行性和延迟；
- [ ] 确认 JEPA 只能在 CBF 前影响 ranking，代码层面拒绝后置覆盖；
- [ ] 将每次 QP 状态、约束活跃集合和 infeasible 原因写入结构化 trace。

### CBF 准入标准

- zero-perturbation candidate 与 frozen nominal + CBF 逐字段一致；
- 所有主评估 candidate 均无 collision/boundary；
- pairwise minimum separation 不得低于预先冻结的安全阈值；
- QP infeasible 必须进入已定义 safe-hold/fallback，不得静默继续；
- p95 CBF 和 JEPA 推理延迟低于控制周期预算，超时必须回退 nominal。

---

## 10. P6：全新 development block 的闭环验证

### 10.1 场景设计

- [ ] 新建与历史 60 集不同的 development block，至少覆盖 nominal、delayed/noisy、flee-persistence、s-curve、目标速度变化、3–5 个障碍和左右起始侧；
- [ ] 训练 seed 固定为至少 `20260911/20260912/20260913`，每个 seed 使用自己的 checkpoint 和 ledger；
- [ ] 使用相同 baseline scenes 与 candidate scenes 做逐 episode 配对；
- [ ] 先运行 20 episode smoke，再运行至少 60 episode/seed final block；
- [ ] 每个 seed 运行前锁定配置、checkpoint、ledger、代码和环境 hash；
- [ ] 不读取历史 locked block，不用当前 development 结果回调阈值。

### 10.2 主指标优先级

1. safe capture；
2. collision、boundary、pairwise separation、CBF infeasible；
3. transit success 和 safe-hold/fallback 率；
4. paired improved/degraded/tied；
5. 最小 obstacle/inter-agent clearance 和 CBF correction；
6. capture time、路径、候选切换、可见性和推理延迟。

capture time 必须报告，但不是单独的 rejection gate。

### 10.3 预注册的安全决策门

以下门必须在运行新 block 前写入 protocol，不得事后修改：

- **G-Safety hard gate：** candidate 在所有 seed/episode 中没有新的 collision 或 boundary violation；zero-regression 必须通过；
- **G-Safe-capture non-inferiority gate：** 三 seed 的 mean paired safe-capture delta 不低于 `0 pp`，且任一 seed 不得低于预先声明的一个 episode 分辨率；
- **G-Safe-capture positive claim：** mean paired delta 为正、至少 2/3 seed 非负，并通过预先指定的 paired bootstrap/McNemar 报告；否则只能称 non-inferior 或 safety-preserving；
- **G-Reliability gate：** 高信用失败率不高于低信用 bucket，且 OOD 能触发回退；
- **G-Realtime gate：** JEPA+ledger+CBF p95 延迟不超过控制周期预算；超时必须使用 nominal CBF；
- **G-Provenance gate：** 所有结果、TensorBoard、checkpoint、ledger、场景和命令可重建。

这里不要求绝对 safe-capture 达到 `95%`；重点是安全不退化和相对 safe-capture 证据，而不是追求单个最好 seed。

### 10.4 必报失败分桶

- collision、boundary、pairwise separation；
- CBF infeasible、CBF correction 过大、safe-hold；
- timeout、capture regression、transit failure；
- ledger nominal/global/OOD fallback；
- high-credit ranking error；
- clearance hallucination、visibility hallucination、target drift；
- candidate oscillation、动作突变和控制延迟。

每个失败必须关联 `training_seed`、`episode_seed`、layout、obstacle count、target motion、observation condition、ledger credit、selected candidate 和 CBF trace。

---

## 11. P7：SIL/HIL 和受控部署准备

只有 P6 通过 G-Safety hard gate 后才进入：

- [ ] 固定仿真步长、控制周期、通信延迟和传感器噪声，做 software-in-the-loop；
- [ ] 将 JEPA、ledger、planner、CBF 放入与真实计算预算一致的进程/容器；
- [ ] 测量 p50/p95/p99 latency、显存、CPU、消息队列积压和 watchdog；
- [ ] 进行硬件在环的动力学、通信和故障注入；
- [ ] 验证断网、传感器冻结、目标突变和单机失效时仍保持 separation/boundary；
- [ ] 在没有人工接管时禁止真实飞行；
- [ ] 制定 geofence、急停、safe-hold、返航和任务终止条件；
- [ ] 只在安全审查通过后形成实飞 preregistration 草案。

HIL 通过不等于实飞通过；任何真实飞行前必须重新审核安全合同和风险责任。

---

## 12. P8：统计、可复现性和阶段提交纪律

### 每个阶段强制动作

- [ ] 运行前保存 protocol、环境、代码、输入和 checkpoint hash；
- [ ] 训练阶段使用独立 TensorBoard logdir，记录 scalar、text、histogram 和配置；
- [ ] 运行结束立即写 `evaluation_metadata.json`，不能数日后凭记忆补写；
- [ ] 每个阶段只提交本阶段代码、测试和 Markdown，results/checkpoint/NPZ/TensorBoard 默认保留本地；
- [ ] 使用 conventional commit，提交后立即 push 并确认 `origin/main`；
- [ ] 不修改或覆盖历史 V4/V5 locked 报告、checkpoint 和 archive；
- [ ] 把失败结果和停止原因一起提交，不能只提交最好结果。

### 统计要求

- [ ] 主要独立单位是完整 `(training_seed, episode)` 或预先定义的 episode block；
- [ ] 三 seed 报告 mean、sample standard deviation 和 paired bootstrap 95% CI；
- [ ] safe capture 同时报告逐 episode配对、improved/degraded/tied 和 exact McNemar；
- [ ] 不把五个候选、时间步或 chunk 内动作当成独立样本；
- [ ] 所有 secondary metrics 不能掩盖 safe capture 或安全失败。

---

## 13. 建议的阶段性文件和提交顺序

| 阶段 | 主要文件 | 阶段完成定义 |
|---|---|---|
| P0 | protocol、CBF safety contract、schema tests | 安全语义和数据边界冻结 |
| P1 | archive generator、audit、manifest | 困难片段和 settled labels 无泄漏 |
| P2 | JEPA model、train script、TensorBoard audit | 三 seed 训练和多任务 gate 通过 |
| P3 | ledger v2、calibration report | 信用/回退可校准且 checkpoint-bound |
| P4 | candidate generator、ranker、ablation harness | 排序、anchor 和日志可复现 |
| P5 | CBF/QP safety tests、fallback trace | zero-regression 和 infeasible fallback 通过 |
| P6 | new development smoke/final、aggregate report | safe-capture-first 三 seed 证据完整 |
| P7 | SIL/HIL latency/fault report | 部署约束和故障响应验证 |
| P8 | preregistration/readiness audit | 只有证据足够才申请新 locked block |

建议提交消息格式：

```text
docs(jepa): freeze safe-capture v2 contract
feat(jepa): add calibrated clearance and visibility heads
feat(jepa): add checkpoint-bound reliability ledger v2
feat(jepa): harden multi-agent cbf fallback
test(jepa): add safe-capture paired evaluation gates
docs(jepa): record three-seed safe-capture development
docs(jepa): audit sil-hil locked readiness
```

---

## 14. 结果分类和停止规则

| 分类 | 条件 | 后续动作 |
|---|---|---|
| `safe_capture_improvement_candidate` | G-Safety、G-Reliability、G-Realtime 通过，三 seed safe capture 有一致正向配对证据 | 写新 preregistration 草案，仍不自动开 locked |
| `safe_capture_noninferior_safety_preserving` | 无新安全失败，safe capture 不退化，但无正向收益 | 保留 CBF/ledger 架构，继续改进评价器，不宣称性能提升 |
| `prediction_signal_no_control_gain` | 离线预测改善，闭环 safe capture 中性或不一致 | 分析 ranking/ledger，禁止用同一 block 调参 |
| `rejected_for_safety` | 新 collision/boundary、zero-regression 失败或 CBF 绕过 | 立即停止该变体，保留失败审计 |
| `insufficient_evidence_do_not_open_locked` | seed 矛盾、CI 过宽、provenance 不完整或场景泄漏 | 只报告不确定性，不能运行 locked |

任何安全硬门失败都优先于路径、时间或平均奖励结果。safe capture 是任务主指标；CBF safety 是不可妥协的约束。

---

## 15. 当前立即执行清单

### 本周第一批

- [ ] 完成 P0 protocol 和 CBF safety contract；
- [ ] 建立 calibration split 与困难场景 manifest；
- [ ] 明确 pairwise separation、obstacle clearance 和 safe-hold 的数值阈值；
- [ ] 在现有 runtime 加入高信用 ranking error、visibility、clearance 和 CBF trace；
- [ ] 为新增日志写测试，不运行 locked test。

### 本周第二批

- [ ] 训练三 seed 的 JEPA v2 多任务模型，并用 TensorBoard 完整记录；
- [ ] 在独立 calibration split 建立 ledger v2；
- [ ] 运行 action-following、distribution shift 和 zero-perturbation audit；
- [ ] 对所有阶段结果进行独立 git commit/push。

### 在新闭环实验前必须完成

- [ ] CBF/QP infeasible 和 safe-hold 测试通过；
- [ ] candidate rank 与 CBF 后 action 的 trace 可关联；
- [ ] 新 development block 已冻结且不与训练/校准重叠；
- [ ] G-Safety、G-Reliability、G-Realtime 的数值门已写入 protocol；
- [ ] smoke 通过后才启动三 seed final；
- [ ] 没有正向 safe-capture 证据时，不打开新的 locked test。

## 最终成功定义

本计划的成功不是“让某个 seed 达到最高捕获率”，而是形成一条可审计的安全关键链路：

```text
候选轨迹生成
  -> action-conditioned interaction-aware JEPA 评价
  -> reliability ledger 校准/回退
  -> CBF 形式化安全过滤
  -> 滚动时域闭环执行
  -> safe capture-first 逐 episode 证据
```

只有当这条链路在全新场景、多个训练 seed、完整 provenance 和 TensorBoard 记录下同时满足安全硬门，并且 safe capture 有一致配对证据时，才可以说该系统在无人机集群对抗围捕任务中“有效”。在此之前，最稳健的论文表述是：**世界模型评价器与 reliability ledger 可以在 CBF 保护下安全接入闭环，但其 safe-capture 增益仍需通过下一阶段独立验证。**
