# 下一阶段目标计划书：安全增强的无人机集群对抗围捕系统

**主题：** Action-Conditioned Interaction-Aware JEPA + Reliability Ledger + Joint CBF-QP + Receding-Horizon Control
**版本：** v1.0（P7 之后执行版）
**制定日期：** 2026-09-04
**适用范围：** 三维无人机集群对抗围捕/拦截仿真，后续 SIL/HIL 准备
**当前阶段：** development-only；`locked_test_opened=false`
**硬件：** NVIDIA RTX 5050，PyTorch 2.7.1+cu128，Conda 环境 `uav-encirclement-gpu`
**主目标：** 在不牺牲任何安全硬约束的前提下，提高 `safe capture` 的可重复性和长尾可靠性。

> 本文件是 P7 完成后的后续执行计划，重点是可验收的工程和实验任务。它不把当前结果写成已经证实的泛化提升，也不授权打开新的 locked test。

## 1. 研究目标与当前证据

### 1.1 要验证的系统假设

本阶段验证以下可证伪假设，而不是预先假定 JEPA 必然有效：

1. **评价假设：** action-conditioned interaction-aware JEPA 能在相同当前 belief 下区分候选动作块的未来任务收益、目标运动漂移、可见性和安全风险。
2. **可信度假设：** reliability ledger 能在世界模型预测不可靠、分布外或观测过期时 abstain，并把控制权交给确定性 nominal/CBF 回退。
3. **安全假设：** Joint CBF-QP 对障碍、边界、机间净空、动力学和高度约束形成最终执行边界；上游模型不能覆盖其结果。
4. **闭环假设：** 只执行动作块第一步并重新观测、重排序、重过滤，可以抑制长时域 rollout 漂移。
5. **任务假设：** 上述链路对 `safe capture` 的改善来自更好的候选排序和回退，而不是来自放宽安全 margin、泄漏 target truth 或挑选最好 seed。

### 1.2 P7 已完成结果的诚实边界

P7 full development 使用同一配对场景、7 个变体、3 个 training seed、每个 40 个 validation episode，共 840 集；没有读取 100 集 locked test。

| 变体 | Safe capture | 安全结果 | 解释 |
|---|---:|---|---|
| M0 nominal + CBF | 32.5% +/- 0.0% | collision/boundary/pairwise = 0 | 安全基线 |
| M1 JEPA + CBF | 33.3% +/- 3.8% | 0/0/0 | 预测接入，收益很小 |
| M2 JEPA + Ledger + CBF | 34.2% +/- 8.8% | 0/0/0 | 回退机制有效但方差较大 |
| M3 JEPA + Ledger + safety ranking + CBF | 36.7% +/- 10.1% | 0/0/0 | 当前最好的开发变体 |
| A1 去除 Ledger | 30.8% +/- 9.5% | 0/0/0 | 诊断 ledger 贡献 |
| A2 去除 clearance/visibility | 33.3% +/- 14.2% | 0/0/0 | 诊断安全辅助任务 |
| A3 raw/no CBF | 2.5% +/- 0.0% | 117 collision、45 pairwise violation | 仅诊断，不可部署 |

M3 相对 M0 的 120 个配对 episode：`16 improved / 11 degraded / 93 tied`，平均配对差值 `+4.17 pp`，固定 seed bootstrap 95% CI 为 `[-4.17 pp, +12.50 pp]`，非负 seed 为 `2/3`。因此当前结论是：

- 可以称为 **positive development evidence**：安全硬门、可靠性观测门通过，平均配对差值为正；
- 不能称为稳健统计提升：CI 跨 0，逐 seed McNemar 检验不显著；
- 不能打开新的 locked test；
- 下一步优先处理 reliability ledger、困难片段重放、净空/可见性辅助预测和候选动作块，而不是替换为更大的主干模型；
- `mean capture time` 只做次要诊断，不是主优化目标，也不是否决安全通过结果的理由。

## 2. 最终系统架构合同

```text
多机观测、通信历史、时间戳
          |
          v
BeliefState_t：目标运动 belief + 多机交互 + 障碍/边界 + 可见性
          |
          v
冻结传统规划器生成 K 个动力学可行候选动作块
          |
          v
Action-Conditioned Interaction-Aware JEPA
  预测目标运动、净空、机间 TTC、可见性、CBF 干预和不确定性
          |
          v
Reliability Ledger
  trusted / fallback_nominal / safe_hold
          |
          +-- trusted：允许候选重排序
          +-- low credit / stale：冻结 nominal + CBF
          +-- OOD / unverified / non-finite：safe-hold 或 controlled abort
          |
          v
候选安全优先排序（保守净空下界和不确定性惩罚）
          |
          v
Joint CBF-QP：最终且不可绕过的执行过滤器
          |
          v
只执行第一步 -> 重新观测 -> 重新预测 -> 重新规划
```

### 2.1 不可变运行时不变量

- JEPA 只能输出预测特征和候选评分，不能直接生成最终飞行动作。
- 在线 `BeliefState` 不得包含 target ground truth；ground truth 只用于离线标签和结果结算，并标记 `offline_only=true`。
- 所有候选，包括 nominal baseline，都必须通过同一个 Joint CBF-QP。
- 当前合同保持 `K=5`、常值动作块 `chunk_length=3`、只执行第一步后 replan；改变合同必须新建 protocol 和新实验块。
- 候选必须先通过速度、加速度、slew 和可达性检查；不可达候选不得进入 JEPA 排序。
- CBF QP infeasible、timeout、stale、OOD、non-finite 和未验证结果必须有显式 fallback，不能静默执行 raw action。
- `safe capture` 必须同时满足捕获半径、时间限制、无障碍碰撞、无边界越界、无机间碰撞、无 target-obstacle 安全失败和无 unverified-CBF 终止。
- reliability ledger 在 calibration 后只读、与 checkpoint/protocol/calibration archive hash 绑定，运行中禁止在线更新。

## 3. 下一阶段工作分解

以下工作包按依赖关系执行。每个工作包都有独立产物和停止条件；未通过前不得跳到后续全量实验。

### WP0：冻结 P7 基线和输入边界（已完成）

**目标：** 把当前 P7 结果变成不可被后续调参污染的参考基线。

- [x] 保存 P7 full 的 `summary.json`、`paired_comparison.json`、`run_metrics.csv`、`m3_seed_comparisons.csv`、报告和 TensorBoard 路径。
- [x] 计算并登记 checkpoint、ledger、protocol、scene manifest、评估脚本、核心源文件和 Conda/Python/PyTorch/CUDA 的 SHA-256/版本。
- [x] 复制一份只读的 P7 manifest 到新的 provenance 目录；后续实验不得覆盖原目录。
- [x] 新建版本化后续协议 `configs/jepa_safe_capture_v3_next_phase.yaml`，明确 `development_only=true` 和 `locked_test_opened=false`。
- [x] 固定安全参数：障碍净空、机间净空、边界/高度、最大速度/加速度、CBF `gamma`、最大修正量、QP timeout 和控制周期。
- [x] 固定主指标、统计方法、episode 配对规则和 seed block 后，再查看新模型的闭环结果。

**验收：** 任意 hash、split、seed、CBF margin 或 locked 标志不一致时，运行器拒绝启动；P7 参考运行可 deterministic replay。

**产物：** `configs/jepa_safe_capture_v3_next_phase.yaml`、`docs/JEPA_SAFE_CAPTURE_WP0_BASELINE_FREEZE_*.md`、provenance manifest。

### WP1：困难 episode 索引、重放与因果归因（索引审计已完成）

**目标：** 把“捕获失败”拆成可修复的预测、排序、回退或 CBF 原因，不用均值猜原因。

- [x] 从 P7 每个 seed 的 episode 表和逐步 trace 建立 failure index，不修改原始结果。
- [x] 为每个失败记录：`training_seed`、`episode_seed`、layout、障碍数量、target motion、visibility、observation age、ledger state/credit、selected candidate、score margin、CBF active constraints、solver status、fallback reason 和 termination reason。
- [x] 按可由当前 trace 证明的类别分桶：遮挡/可见性丢失、观测过期、CBF 过度修正、QP infeasible、timeout、动作振荡和预测 gap；target drift 明确标记为未观测。
- [ ] 对每一类至少选取代表性 episode，运行环境级 deterministic replay，检查预测 -> ledger -> rank -> CBF -> executed action -> failure 的完整链路（下一小阶段）。
- [x] 计算 high-credit 错误率、low-credit 错误率、fallback 后安全率和候选切换统计。
- [x] 将 hard-case index 与训练/校准 split 解耦；失败 episode 的原始发展结果不得直接回灌训练集。

**验收：** 100% 的失败 episode 可定位到唯一主因或明确标记 `unresolved`；无法重放的 episode 直接触发 provenance gate 失败。

**建议产物：**

- `scripts/index_jepa_safe_capture_failures.py`
- `scripts/replay_jepa_safe_capture_failures.py`
- `results/jepa_safe_capture_v3_failure_index/`
- `docs/JEPA_SAFE_CAPTURE_WP1_FAILURE_REPLAY_*.md`

### WP2：JEPA 多任务预测和交互表示增强

**目标：** 提高候选之间的可辨识未来表示，抑制只预测目标位移造成的幻觉。

- [ ] 保留 target relative displacement 主头，并加入 velocity/acceleration consistency。
- [ ] 增加 obstacle clearance lower-quantile/distributional head，不使用均值代表安全。
- [ ] 增加 inter-agent clearance、pairwise TTC 和队形拥挤度 head。
- [ ] 增加 target visibility、observation age、message delay/loss head。
- [ ] 增加 CBF intervention probability、correction magnitude 和 QP feasibility head。
- [ ] 对目标运动模式加入 explicit mode/belief embedding，例如 constant-velocity、flee-persistence、turn、S-curve 和突变加速度。
- [ ] 使用 action-conditioned consistency/contrastive loss，确保不同候选块产生方向一致且非塌缩的未来表示。
- [ ] 使用 ensemble、heteroscedastic output 或 calibrated residual 得到 uncertainty；不把 uncertainty 误写成安全证书。
- [ ] 保持 `interaction_group_slices` 与 observation schema 固定；任何输入维度变更必须更新协议和测试。
- [ ] 对 task loss 做预注册，防止目标位移 loss 压制 clearance/visibility/CBF heads。

**离线准入：**

- 所有输出 finite；
- 至少一个 horizon 的 target MAE 优于 constant-velocity baseline；
- clearance、visibility、TTC、CBF risk heads 有非空标签覆盖和可校准分数；
- action-following 的方向一致性和候选分离度为正；
- 三个训练 seed 的 checkpoint、训练配置和 TensorBoard 均可追溯。

**产物：** 更新后的 `src/encirclement3d/prediction.py`、`scripts/train_jepa_safe_capture_v3.py`、`scripts/audit_jepa_safe_capture_v3_training.py`、三 seed checkpoint、prediction audit 报告和 TensorBoard。

### WP3：Reliability Ledger v3 与 abstention 校准

**目标：** 让 ledger 对“高置信但错误”的 rollout 及时拒答，而不是只给 score 加惩罚。

- [ ] 在独立 calibration split 建立 local、coarse/global 和 OOD bucket；禁止使用 development/locked episode 拟合阈值。
- [ ] 以 visibility、observation age、obstacle count/layout、target motion mode/speed、minimum clearance、pairwise TTC、uncertainty、candidate score margin 和 CBF intervention risk 分桶。
- [ ] 对每个预测 head 计算 MAE 分位数、coverage、ECE、Brier、AUROC 和 calibration error；对 clearance 使用保守 lower bound。
- [ ] 预先冻结最小样本数、最小 credit、uncertainty 上限、stale age 上限和 OOD 判定规则。
- [ ] 明确三态动作：`trusted -> rank`、`fallback_nominal -> frozen nominal then CBF`、`safe_hold -> separation-preserving hold/controlled abort`。
- [ ] 加入 temporal drift audit：连续 rollout credit 下降、预测残差突增或 candidate separation 消失时必须 abstain。
- [ ] 加入 adversarial shift audit：目标急转、速度突变、遮挡、消息延迟、丢包和障碍密度变化时不能虚高 credit。
- [ ] 对 high-credit failure 做专门报告；若 high-credit 比 low-credit 更危险，立即拒绝当前 ledger。
- [ ] 将 ledger hash 绑定 checkpoint、calibration archive、protocol 和代码 revision；校准完成后设为只读。

**准入门：** OOD/stale/non-finite 的 fallback 触发率为 100%；unverified path 不得计为 safe capture；high-credit 失败率不得高于 low-credit；所有 fallback 都有可回放 trace。

**产物：** `reliability_ledger_v3.json`、`scripts/build_jepa_safe_capture_v3_reliability_ledger.py`、calibration report、bucket coverage 图、failure examples 和 ledger tests。

### WP4：候选轨迹评价、排序和动作块设计

**目标：** 让 JEPA 的预测信号真正改变 safe-capture，而不是只保持 CBF 的安全外壳。

- [ ] 先保留当前 `K=5`、3-step constant chunk 作为可比基线；任何新候选集合单独登记版本。
- [ ] 候选库至少包含：nominal anchor、捕获方向、保守侧向、减速/悬停、绕障和提高可见性的动作块。
- [ ] 为每个候选写入 dynamics feasibility、reachable speed/slew projection、nominal distance 和生成失败原因。
- [ ] 将 score 分解为 task progress、保守 obstacle/inter-agent clearance、visibility gain、CBF intervention cost、uncertainty penalty、action change cost 和 nominal anchor penalty。
- [ ] 对预测安全量使用 lower quantile；预测值只能排序，不能取代 CBF。
- [ ] 加入 score margin、rank stability、candidate switch rate、oscillation、CBF correction norm 和候选/nominal 差异日志。
- [ ] 设计离线 rank consistency：预测 top-1 与 settled safe outcome、CBF intervention 和 capture outcome 的一致性分别报告。
- [ ] 若探索 chunk length `1/3/5` 或非恒定动作块，先做离线 action-following/可达性审计，再作为新 protocol 分支运行，不能在同一 development block 事后选择。

**准入门：** zero-perturbation 时非 JEPA 字段与 nominal 完全一致；候选切换不过度；任何排名收益不能以新增安全失败为代价。

**产物：** `src/encirclement3d/jepa_safe_capture_candidates.py`、`jepa_safe_capture_ranker.py` 的版本化更新、rank audit、candidate trace schema 和单元测试。

### WP5：Joint CBF-QP 与故障回退强化

**目标：** 将集群安全约束落实为不可绕过的最后执行边界。

- [ ] 审计 obstacle、boundary、altitude、speed、acceleration、pairwise separation 和 target approach 的离散时间约束。
- [ ] 保留 anticipatory braking，对未来可达速度而非仅当前速度施加约束。
- [ ] 注入 QP infeasible、solver timeout、non-finite input、过期 observation、通信中断和多个同时激活约束，验证确定性 fallback。
- [ ] 固定回退优先级：separation-preserving safe-hold -> verified nominal-CBF -> controlled abort；每次必须有 reason code。
- [ ] 限制 CBF correction norm、连续修正次数和最小净空，记录 active set、solver status、求解时间和 residual。
- [ ] 做 agent 数量、队形密度、障碍数和边界压力的可行性/延迟 stress test。
- [ ] 代码层面禁止上游 ranker 在 CBF 之后覆盖 action；增加接口测试证明 JEPA 不能绕过过滤器。
- [ ] 将 `cbf_unverified`、`cbf_timeout`、`controlled_abort` 从 safe capture 中严格排除。

**准入门：** 主变体 collision、boundary、pairwise violation 均为 0；QP 失败均有显式 fallback；p95 总控制延迟不超过 100 ms，超时必须走 nominal-CBF 或 safe-hold。

**产物：** `src/encirclement3d/cbf_qp.py`、安全测试、fault-injection audit、CBF TensorBoard 和 failure trace。

### WP6：滚动时域闭环集成与可观测性

**目标：** 验证从 Simulation -> Planning -> Decision-making 的完整闭环，而不是只验证离线预测。

- [ ] 每个控制周期按固定顺序执行：观测时间戳检查 -> belief 更新 -> 候选生成 -> JEPA batch prediction -> ledger decision -> ranking -> Joint CBF-QP -> first-step execute。
- [ ] 每一步记录输入 hash、候选动作、JEPA 预测、uncertainty、ledger state/credit、selected rank、CBF 修正、执行动作、延迟和 fallback。
- [ ] 实际只执行第一步，下一周期重新观测；禁止将 3-step counterfactual label 当成 3-step open-loop deployment。
- [ ] 添加 watchdog：JEPA、ledger、ranker 或 CBF 任一超时即回退，且 timeout 不得被隐藏在成功计数里。
- [ ] 验证 CPU 与 RTX 5050 的数值容差、动作选择、终止原因和 seed replay 一致性。
- [ ] 从空输出目录重跑一个 episode，核对 `summary.json`、`episodes.csv`、逐步 trace、TensorBoard 和报告统计一致。

**产物：** 更新后的 `scripts/evaluate_jepa_safe_capture_v3_paired.py`、trace schema、runtime audit、latency profile 和回放脚本。

### WP7：多 seed 配对 development 验证

**目标：** 用新的、独立的 development block 验证 safe-capture 是否具有跨 seed 的可重复趋势。

#### Round A：P7 合同回归

- [ ] 用冻结的旧 P7 合同重跑 M0--M3、A1--A3，每个 `20260911/20260912/20260913`、40 episodes，确认代码/环境修改没有改变参考结论。
- [ ] 该回归只用于合同一致性，不能据此重新调参数。

#### Round B：下一版本新 development block

- [ ] 新建与训练/校准 episode 不重叠的 scene manifest，覆盖 nominal、delayed/noisy、flee-persistence、急转、S-curve、速度突变、低 visibility、3--5 障碍和左右起始侧。
- [ ] 运行前冻结 checkpoint、ledger、protocol、scene hash、3 个 training seed 和 episode seed 列表。
- [ ] 先运行每变体/seed 20 集 smoke；只有安全硬门通过才进入 final。
- [ ] final 至少 3 个 training seed，每 seed 至少 40 集；若要提高分辨率，另建 protocol 扩展到 60 集，不在运行中改变 episode 数。
- [ ] 主比较至少包括 M0、当前 M3、去 ledger、去 clearance/visibility；raw/no-CBF 仅保留为诊断。
- [ ] 所有候选与 M0 使用同一 episode index、layout、target motion 和 observation schedule 配对。
- [ ] 每个 seed 生成独立 checkpoint/ledger/provenance；不把一个 seed 的结果传播给其他 seed。

**主指标顺序：**

1. `safe_capture`；
2. collision、boundary、pairwise violation、minimum clearance、CBF infeasible/timeout/unverified；
3. transit、safe-hold、fallback 和 high-credit failure；
4. paired improved/degraded/tied、bootstrap CI、exact McNemar；
5. candidate switch、CBF correction、visibility、prediction drift 和 latency；
6. capture time、路径长度和能耗等次要指标。

### WP8：SIL/HIL 安全与实时性准备

只有 WP5--WP7 的安全硬门通过后才进入；它们不自动产生实飞结论。

- [ ] 固定控制周期、仿真步长、通信延迟、消息丢失和传感器噪声，运行 SIL。
- [ ] 在 RTX 5050 和 CPU 两条路径测量 p50/p95/p99 的 JEPA、ledger、ranker、CBF 和端到端延迟。
- [ ] 注入传感器冻结、时间戳错误、网络中断、target 突变、单机失效、QP timeout 和显存压力。
- [ ] 验证任何故障下仍保持 pairwise separation、boundary 和 obstacle clearance；无法保证时必须 controlled abort。
- [ ] 准备 watchdog、geofence、急停、safe-hold、返航/任务终止和人工接管接口。
- [ ] 形成部署版本、回滚包和审计日志完整性检查；没有安全审查和用户授权不得真实飞行。

**产物：** SIL/HIL latency-fault report、部署约束清单、故障响应矩阵和待审查的实飞 preregistration 草案。

### WP9：统计审计、论文结论与 locked test 申请

- [ ] 按 `(training_seed, episode)` 作为主要独立单位；不得把候选、时间步或 chunk 内动作当独立样本。
- [ ] 同时报告三 seed mean、sample standard deviation、paired delta、improved/degraded/tied、bootstrap 95% CI 和 exact McNemar。
- [ ] 报告 safe capture 与所有安全失败的逐 seed 分布；capture time 只放 secondary table。
- [ ] 对 high-credit/low-credit、motion mode、visibility、clearance、fallback 和 CBF active constraint 分桶。
- [ ] 用 TensorBoard 与 JSON/CSV 双向核对，发现不一致即拒绝 readiness。
- [ ] 结果只归类为 `safe_capture_improvement_candidate`、`safety_preserving_noninferior`、`prediction_signal_no_control_gain`、`rejected_for_safety` 或 `insufficient_evidence_do_not_open_locked_test`。
- [ ] 只有新 development block 在安全、可靠性、实时性、provenance 和统计门全部通过后，才起草新的 locked preregistration。
- [ ] 真正打开 locked test 必须获得单独明确授权；脚本默认拒绝 `locked_test_opened=true` 以外的隐式路径。

## 4. 预注册门槛和停止规则

### 4.1 安全硬门（不可协商）

对所有安全保留变体：

- candidate collision count = 0；
- boundary violation count = 0；
- pairwise violation count = 0；
- zero-perturbation 的非 JEPA 字段差异 = 0；
- CBF infeasible、timeout、unverified 不得记为 safe capture；
- 任意新安全失败立即停止当前变体，保存 trace，回退 frozen nominal + CBF。

### 4.2 Safe-capture 门

保持当前协议的最小非劣门以保证可比性：

- mean paired safe-capture delta >= `0 pp`；
- 每个 seed 的 delta 不低于预先声明的一个 episode resolution（当前 40 集为 `-1.667 pp`）；
- 至少 2/3 seed 非负；
- 结论必须基于配对 episode，而不是挑选单个最好 seed。

更强的“稳健提升”表述还需要 paired bootstrap 95% CI 下界 > `0 pp`，并给出逐 seed exact test；否则只能使用 development evidence 或 non-inferiority 表述。

### 4.3 Reliability 门

- high-credit bucket 的 settled failure rate 不高于 low-credit bucket；
- OOD、stale、non-finite、unverified 输入 100% 触发显式 fallback；
- fallback 的安全率和终止原因可逐 episode 回放；
- ledger 不得被描述为安全证明，CBF 仍是执行安全边界。

### 4.4 实时性与可复现门

- 端到端 p95 latency <= 100 ms；超时必须走已验证 fallback；
- 每个输出包含 protocol、代码 revision、环境、checkpoint、ledger、scene 和数据 hash；
- 所有 3 seed 运行目录完整，TensorBoard/JSON/CSV 一致；
- 任意输入 hash 漂移都必须重新生成 provenance，不得继续使用旧结论。

### 4.5 立即停止条件

- 新 collision、boundary 或 pairwise violation；
- raw action 绕过 CBF 或未验证 action 被执行；
- high-credit 比 low-credit 更危险；
- zero-perturbation regression；
- QP infeasible/timeout 没有显式 fallback；
- ground-truth leakage、split leakage 或 episode 未配对；
- 为提升 safe capture 而降低 CBF margin、增加捕获半径、改变终止语义或隐藏 timeout；
- 仅有一个 seed 改善、CI 过宽或 provenance 不完整。

## 5. 实验矩阵与资源安排

### 5.1 必做矩阵

| 阶段 | 变体 | seed | episode | 用途 |
|---|---|---:|---:|---|
| Round A 回归 | M0/M1/M2/M3/A1/A2/A3 | 3 | 40/seed | 验证 P7 合同没有被改坏 |
| Round B smoke | M0/M3/A1/A2 | 3 | 20/seed | 新版本安全快速筛查 |
| Round B final | M0/M3/A1/A2 | 3 | >=40/seed | 新版本主 paired evidence |
| stress | M0/M3 | 3 | 独立 hard block | 目标漂移、遮挡、丢包、密集队形 |
| raw/no-CBF | 仅诊断 | 3 | 同一 paired block | 证明 raw 不能部署，不参与主结论 |

### 5.2 RTX 5050 执行纪律

- 训练使用 `device=cuda`，评估可用 CUDA 或 CPU 做一致性审计。
- 每个长任务使用新的空输出目录，不覆盖历史 checkpoint、TensorBoard、NPZ 或报告。
- 并行任务不能共享可写 TensorBoard 目录；每个 seed/variant 使用独立 logdir。
- 先跑 `python -m pytest -q` 和 targeted audit，再启动长实验。
- results、tmp、NPZ archive、TensorBoard event 和大体积 trajectory 默认只保留本地，不加入 Git。

## 6. 文件级 Todo 与阶段产物

### 6.1 需要优先审计/修改的现有文件

- [ ] `src/encirclement3d/prediction.py`：多任务 head、interaction encoding、uncertainty 和 action consistency。
- [ ] `src/encirclement3d/reliability.py`：bucket、credit、abstention、hash-bound 只读 ledger。
- [ ] `src/encirclement3d/jepa_safe_capture_candidates.py`：候选可达性、nominal anchor、动作块日志。
- [ ] `src/encirclement3d/jepa_safe_capture_ranker.py`：安全优先 score、rank margin、switch/oscillation 统计。
- [ ] `src/encirclement3d/cbf_qp.py`：joint constraints、anticipatory braking、solver timeout 和 fallback。
- [ ] `scripts/evaluate_jepa_safe_capture_v2_paired.py`：闭环 trace、first-step replan 和开发边界校验。
- [ ] `scripts/aggregate_jepa_safe_capture_v2_paired.py`：paired statistics、fallback 分类和 CI 审计。
- [ ] `scripts/audit_jepa_safe_capture_v2_training.py`：TensorBoard/provenance 必需 tag。

### 6.2 建议新增的脚本/报告

- [ ] `scripts/index_jepa_safe_capture_failures.py`
- [ ] `scripts/replay_jepa_safe_capture_failures.py`
- [ ] `scripts/audit_jepa_safe_capture_v3_distribution_shift.py`
- [ ] `scripts/audit_jepa_safe_capture_v3_latency_faults.py`
- [ ] `docs/JEPA_SAFE_CAPTURE_WP1_FAILURE_REPLAY_*.md`
- [ ] `docs/JEPA_SAFE_CAPTURE_WP3_LEDGER_V3_*.md`
- [ ] `docs/JEPA_SAFE_CAPTURE_WP7_NEXT_DEVELOPMENT_*.md`
- [ ] `docs/JEPA_SAFE_CAPTURE_WP8_SIL_HIL_*.md`
- [ ] `docs/JEPA_SAFE_CAPTURE_READINESS_*.md`

### 6.3 测试要求

- [ ] candidate count/chunk semantics/nominal anchor/first-step execution。
- [ ] action reachability、slew projection 和 non-finite rejection。
- [ ] JEPA no-ground-truth boundary 和 action-following directionality。
- [ ] ledger local/global/OOD/stale/high-uncertainty 三态决策。
- [ ] CBF obstacle/pairwise/boundary/kinematic/anticipatory constraints。
- [ ] QP infeasible、timeout、controlled abort 和 verified fallback。
- [ ] safe capture 不接受 collision、boundary、pairwise 或 unverified-CBF episode。
- [ ] paired aggregator 拒绝缺失 seed、重复 episode、scene hash 漂移和 locked metadata。

## 7. 推荐时间盒

| 时间 | 工作 | 完成标志 |
|---|---|---|
| 第 0--1 天 | WP0 基线/协议/hash 冻结 | 输入和门槛不可变，P7 可重放 |
| 第 2--4 天 | WP1 失败索引和 hard replay | 每个失败有因果链或明确 unresolved |
| 第 5--8 天 | WP2 JEPA 多任务离线训练/审计 | 三 seed prediction gates 通过 |
| 第 7--9 天 | WP3 ledger v3 校准/漂移审计 | OOD/stale 100% fallback |
| 第 9--11 天 | WP4 候选块/ranker + WP5 CBF fault tests | zero-perturbation 和安全硬门通过 |
| 第 12 天 | WP6 集成 smoke | 无新安全失败，trace 完整 |
| 第 13--17 天 | WP7 Round A/B 多 seed development | 配对 episode 和三 seed 结果齐全 |
| 第 18--19 天 | 统计、TensorBoard、provenance audit | JSON/CSV/事件日志一致 |
| 第 20--22 天 | 可选 WP8 SIL/HIL | 故障响应和实时性报告完成 |
| 第 23 天以后 | WP9 readiness/preregistration | 仅在门槛通过并获授权后申请 locked |

时间盒只是工程安排，不是 safe-capture 数值承诺；任何安全停止条件都优先于进度。

## 8. Definition of Done

本阶段只有同时满足下列条件才算完成：

1. 新协议、scene/seed block、checkpoint、ledger 和代码 revision 均有 hash，且 `locked_test_opened=false`。
2. JEPA 只做候选轨迹评价，最终动作 100% 经过 Joint CBF-QP。
3. 每个 infeasible、timeout、OOD、stale、non-finite 和 controlled abort 都有显式、可回放的 fallback。
4. 多任务 JEPA 在独立 validation/calibration 上有可复核预测和校准证据，不依赖在线 target truth。
5. 至少 3 个 training seed、同一 paired scene block、每 seed 至少 40 个 episode 完整运行；失败 episode 可 deterministic replay。
6. 安全保留变体通过 collision、boundary、pairwise、zero-perturbation 和 CBF fallback 硬门。
7. safe-capture 结果按 paired improved/degraded/tied、三 seed统计和 bootstrap CI 报告；capture time 仅作为次指标。
8. TensorBoard、JSON、CSV、逐步 trace 和 Markdown 报告相互一致；从空目录可复现最小运行。
9. 结果诚实归类为 improvement candidate、safety-preserving/non-inferior、prediction-only、rejected 或 insufficient evidence。
10. 没有单独明确授权前，新的 locked test 保持关闭；仿真/SIL/HIL 结果不被写成实飞安全证明。

**最终研究判据：** 不是某个 seed 的最高捕获率，也不是更短的 mean capture time，而是证明“JEPA 反事实候选评价 + reliability ledger 拒答 + CBF 硬安全层 + 滚动时域重规划”能在多 seed、困难场景和完整审计链路中持续提高或至少不损害 `safe capture`，并且世界模型的幻觉、目标漂移和集群安全约束都不会静默失效。
