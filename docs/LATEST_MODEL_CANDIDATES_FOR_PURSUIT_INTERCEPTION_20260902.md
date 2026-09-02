# 围捕拦截任务的最新模型候选

日期：2026-09-02  
范围：基于当前仓库的 V3/V4/V5 运动学仿真结果，筛选理论上最可能改善部分观测、随机障碍、目标机动和执行安全的模型。本文不把预印本中的结果当作本项目已经验证的结论。

## 1. 当前问题画像

现有证据表明，固定 S1/S2 场景已经接近饱和，真正的瓶颈在随机 S3 和执行扰动：

- V4 正式 S3 locked：Retained BC + CBF 为 `75.3% +/- 6.5%`，碰撞和边界各约 `4.7%`。
- RTX 5050 上的 archive-faithful V4 三种子 validation：`85.83% +/- 3.82%`，但不是新 locked 结果。
- V5 exact-reactive 的单种子开发结果可达 `57/60 = 95.0%`，三种子完整 gate 未通过。
- E1-prime 将捕获门槛降到 90% 后，只有 E0 通过；加速度限制、动作延迟和噪声下的 E1-E6 仍然失败。

因此，最值得优化的不是再加大一个 MLP，而是：

1. 从延迟/漏检观测中恢复可控制的目标和障碍状态；
2. 对候选动作的未来风险进行预测；
3. 让多个 UAV 产生非冗余、可执行的拦截轨迹；
4. 用安全层处理预测误差、动作饱和和边界约束。

## 2. 候选模型排名

| 优先级 | 模型/论文 | 核心机制 | 对本项目的理论帮助 | 与现有代码的匹配 | 主要风险 |
| ---: | --- | --- | --- | --- | --- |
| 1 | **SkyJEPA**，arXiv:2606.23444 | 潜在动力学 + 物理可解释 prober + 长时域预测 + sampling-based control | 直接针对四旋翼长时域预测、动作规划和 sim-to-real；可预测目标相对运动、无人机状态和 clearance | 极高；可接在现有 observation encoder 与 CBF 之前 | 论文为 under review；原始工作更偏视觉和单机，需改成结构化多 UAV 输入 |
| 2 | **Action-conditioned JEPA + model-based safety shield**，arXiv:2608.17496 | 对候选动作 chunk 预测任务进度与物理风险，学习排序，确定性 shield 强制执行 | 与当前 `Policy + CBF` 形式几乎同构，可把 CBF 从单步修正升级为短时域风险筛选 | 极高；保留 CBF，新增 action-conditioned risk/progress heads | 目前是 simulation-only 预印本；安全保证来自 shield，不是 JEPA 本身 |
| 3 | **Temporal-Distance-JEPA**，arXiv:2607.25337 | 从无奖励轨迹挖掘有向时间距离和 rollout consistency，作为规划代价 | 围捕不是单纯“离目标越近越好”，还要经历接近、分散、封锁和安全捕获阶段；时间距离可提供阶段进度 | 高；可把 `capture radius`、zone entry、clearance 组合成进度标签 | 时间排序可能把错误的专家行为固化，需要反事实和失败轨迹 |
| 4 | **PCDP: Planner-Conditioned Diffusion Policy**，arXiv:2608.16229 | 条件扩散生成多模态长时域轨迹，再做邻近 agent 组合重排 | 围捕需要左绕/右绕、分工和非冗余轨迹；多模态比单步均值动作更适合拦截 | 高；先作为高层 8-16 步 waypoint/chunk proposer，低层仍用现有控制器和 CBF | 推理成本较高；若没有良好 planner 多样性，重排收益会很小 |
| 5 | **ConfAL-WM**，arXiv:2608.25572 | 对世界模型预测输出 dense confidence，按任务/帧/局部区域主动选数据 | 可优先采样碰撞、timeout、低 clearance 和高 CBF correction 的失败片段，缓解当前几百条 archive 数据不足 | 高；不改策略结构，先用于 dataset curation 和 hard-case replay | 该方法面向视频 WAM；需把 confidence 改成结构化状态/风险 confidence |
| 6 | **AgilePE**，arXiv:2608.14135 | Prioritized Fictitious Self-Play、历史对手池、执行器/通信延迟随机化 | 直接对应追逃任务；历史目标策略池能避免只学单一 flee policy，执行随机化对应 E1-E6 | 中高；可替换/扩展当前固定 rule-expert 目标生成器 | 预印本 under review，报告含真实部署但不能直接迁移到当前四机 CBF 合同 |
| 7 | **Latent Activation Editing (LAE)**，arXiv:2509.20623 | 在线检测危险 latent，推理时编辑策略激活，不更新权重 | 对已有 V4/V5 checkpoint 是低成本安全修补，理论上可在碰撞前提早切换规避行为 | 中高；可放在 actor latent 与 action head 之间 | 需要碰撞前激活数据；可能损害捕获效率，且不提供形式安全保证 |
| 8 | **SHIELD**，arXiv:2505.11494 | 学习随机动力学 residual，在概率 CBF 中约束风险 | E1-prime 显示实际执行动力学是瓶颈；概率 CBF 可显式处理动作延迟、噪声和模型不确定性 | 中高；扩展现有 CBF 为 uncertainty-aware CBF | 需要真实或高保真执行数据；仿真 residual 若失配会给出虚假置信度 |
| 9 | **Graph-Operator World Model**，arXiv:2608.20936 | 图结构动力学与条件 operator，支持形态/动力学参数变化 | 若未来加入不同 UAV 质量、最大速度、加速度或载荷，可提高跨平台泛化 | 中；当前四机同构、运动学模型较简单，收益暂不如 JEPA/PCDP | 对当前任务可能过度设计 |
| 10 | **Online Planning for Multi-UAV Pursuit-Evasion**，arXiv:2409.15866，RA-L 2025 | 逃逸者预测增强网络 + 自适应环境生成器 + 两阶段奖励修正 | 与本项目最接近的已发表多 UAV 追逃参考；可借鉴预测输入、对手/地图生成和奖励修正 | 高；适合作为非 JEPA 的强基线和训练协议参考 | 其 100% 结果不能直接与本项目比较，任务、动力学和数据合同不同 |

## 3. 最值得实现的组合

推荐的主线不是“单独换成 JEPA”，而是下面的分层结构：

```text
局部观测 + mask/age + 队友消息
              |
      Multi-UAV graph encoder
              |
   action-conditioned JEPA latent dynamics
       |                    |
  progress head        risk/clearance head
       |                    |
   PCDP/扩散轨迹 proposer ---- candidate reranking
              |
      deterministic CBF / fallback ladder
              |
       low-level velocity action
```

其中：

- JEPA 预测的是结构化 latent，不重建像素；输入可直接使用当前 63 维 observation、缺失 mask、`observation_age`、`message_age` 和执行噪声状态。
- 预测必须以候选动作 chunk 为条件，否则无法判断“采取该动作后是否会撞障碍”。
- PCDP 只负责产生多个可能的拦截方案，不直接绕过 CBF。
- risk head 至少预测未来 `min_clearance`、collision probability、boundary probability、capture probability 和 timeout probability。
- CBF/fallback ladder 是硬约束；学习模型只能在 admissible candidates 中排序。

## 4. 建议的实验顺序

### P0：先验证数据和预测任务

用当前专家 archive 加上 E1-prime 的失败轨迹，构造窗口长度 16/32 的结构化 transition 数据。比较：

1. constant velocity；
2. GRU predictor；
3. transformer predictor；
4. action-conditioned JEPA。

指标不要只用 latent loss，还要报告 1/2/4/8 步的目标位置误差、clearance 误差、碰撞风险 AUROC、风险 calibration error 和 rollout drift。

### P1：固定场景回归

每个候选模型先通过 Cylinder、Box、Wall 和 S2 的固定 CBF 回归。任何固定场景退化都停止，不读取 S3 locked block。

### P2：三种子 S3 development

对每个通过 P1 的候选，使用相同的三训练 seed、相同的 development block、相同的 raw/CBF 配对。至少报告：

- Cooperative Safe Capture；
- collision、boundary、timeout；
- Transit；
- mean/max CBF correction；
- action execution error；
- 按 observation condition、obstacle layout 和 clearance band 分桶。

### P3：只有三种子都通过才开 locked

保留现有“development gate 通过后才打开 locked block”的规则。不能因为 JEPA 在单个困难 profile 上成功，就重写 V4/V5 历史结论。

## 5. 两周可执行 pilot

**第 1-3 天：数据管线**

- 统一保存 `obs_t`, `action_t`, `obs_{t+1:t+8}`、执行噪声、CBF 修正、clearance 和 termination label。
- 增加 observation mask/age 和 candidate action chunk。

**第 4-7 天：小模型预测对比**

- latent dimension 128 或 256；4 层以内 transformer/JEPA predictor；单张 RTX 5050 可训练。
- 只做 offline prediction，不改变 V5 策略。

**第 8-10 天：风险筛选器**

- 生成 8 个候选 action chunks；预测 progress/risk；CBF 过滤；空集合时执行保守 fallback。
- 与单步 actor + CBF 比较，不先训练新的 end-to-end policy。

**第 11-14 天：三 seed 小规模开发验证**

- 先用每 profile 20 回合 smoke，再用完整 60 回合 development。
- 目标是验证 delayed/noisy 和 E3-E6 的失败率是否下降，而不是追求单次 95%。

## 6. 结论

最值得优先实现的是 **SkyJEPA 思路 + 动作条件风险预测 + 现有 CBF**。如果需要一个更容易落地的第一版，则先实现 **GRU/Transformer action-conditioned predictor**，用同样的 risk head 和 candidate reranking 做对照；这样可以判断收益来自 JEPA 表征，还是仅来自多步预测。

PCDP 适合作为第二阶段的多 UAV 协调器，AgilePE 适合作为对手池和训练随机化协议，ConfAL-WM 适合作为失败数据选择器，LAE/SHIELD 适合作为安全增强方向。Graph-Operator WM 只有在引入异构 UAV 或动力学变化后才值得优先。

## 7. 参考链接

- [SkyJEPA (arXiv:2606.23444)](https://arxiv.org/abs/2606.23444)
- [Action-conditioned JEPA safety framework (arXiv:2608.17496)](https://arxiv.org/abs/2608.17496)
- [Temporal-Distance-JEPA (arXiv:2607.25337)](https://arxiv.org/abs/2607.25337)
- [JEPA-WAM (arXiv:2608.09381)](https://arxiv.org/abs/2608.09381)
- [ConfAL-WM (arXiv:2608.25572)](https://arxiv.org/abs/2608.25572)
- [Planner-Conditioned Diffusion Policy (arXiv:2608.16229)](https://arxiv.org/abs/2608.16229)
- [AgilePE (arXiv:2608.14135)](https://arxiv.org/abs/2608.14135)
- [Latent Activation Editing (arXiv:2509.20623)](https://arxiv.org/abs/2509.20623)
- [SHIELD (arXiv:2505.11494)](https://arxiv.org/abs/2505.11494)
- [Graph-Operator World Models (arXiv:2608.20936)](https://arxiv.org/abs/2608.20936)
- [Online Planning for Multi-UAV Pursuit-Evasion (arXiv:2409.15866)](https://arxiv.org/abs/2409.15866)

## 8. 2026-09-02 增量检索与可用性核验

本节补充了第一轮候选中遗漏、但更直接对应当前失败模式的工作。检索使用 arXiv 官方 API 和各论文 arXiv 摘要页；发表状态和代码链接仅以作者在该页面明确写出的信息为准。因大多数论文发布不足数月，**未使用引用次数来判断价值**。没有公开仓库不代表方法无效，但意味着本项目需要自行实现，不能把它当作可直接复现实验的基线。

| 候选 | 核验状态（2026-09-02） | 最匹配的问题 | 可放入本项目的位置 | 判断与边界 |
| --- | --- | --- | --- | --- |
| **Diff-MA-STL**: Generalizable Multi-Agent Planning from STL Specifications via Diffusion, [arXiv:2608.29490](https://arxiv.org/abs/2608.29490) | arXiv 备注为 **RA-L 2026 accepted**；[代码](https://github.com/jeappen/diff-ma-stl) 与项目页已公开 | 多机轨迹的非冗余性、碰撞、边界和阶段目标同时满足 | 作为 8--16 步 waypoint proposer；将捕获、障碍净空、UAV 间距、边界写成 differentiable STL predicates | **本轮最值得复用代码的协调规划候选**。STL 可以将现有硬指标显式写入高层规划，但仍需保留 CBF 处理模型误差和离散执行；论文不是追逃任务，不能预期直接提升捕获率。 |
| **Risk-Aware Belief CBF over Random Finite Sets**, [arXiv:2607.15016](https://arxiv.org/abs/2607.15016) | 预印本；未找到作者公开仓库 | 漏检、观测 age、动态障碍和目标位置的不确定性 | 用 belief/particle 表示目标和障碍，再将 CBF 从点估计距离替换为风险约束距离 | **最强的安全理论补件**。论文给出连续预测下安全集的 forward invariance 和离散更新条件，但原实验是水下机器人，且 particle 计算量会增加；它本身不改善围捕策略。 |
| **Self-Adaptive Learning + MPC for Tracking Unknown Dynamics**, [arXiv:2607.26370](https://arxiv.org/abs/2607.26370) | 预印本；Crazyflie 仿真和硬件实验；未找到代码 | evader 的 switching、随机或对抗性机动 | 维护常速度、GRU、Transformer/JEPA 等预测器池，在线按近期误差加权，并输出 MPC/候选排序的 target forecast | **比端到端大模型更适合先做的目标预测强基线**。其有限时间 near-optimal/no-regret 分析正对未知 target dynamics；理论保证依赖论文的在线预测设定，不自动扩展到四机、障碍和 CBF。 |
| **Less is More: Robust Zero-Communication 3D Pursuit-Evasion**, [arXiv:2603.08273](https://arxiv.org/abs/2603.08273) | IEEE 投稿中；未找到代码；4 pursuers vs. 1 evader | `message_age`、通信延迟与团队观测冗余导致的级联误差 | 对 actor 做 observation/channel ablation，并采用局部贡献 credit，而不是一味增加消息编码器 | **与本项目形态最接近的鲁棒性证据之一**。论文报告其 50-D 无通信接口在 Stage-5 成功率 `0.753 +/- 0.091`，略高于 83-D 全观测 `0.721 +/- 0.071`，但碰撞仍很高（`0.223`）；应把它理解为“删减脆弱信息”的假设，而不是可直接接受的安全方案。 |
| **Encirclement Guaranteed Finite-Time Capture**, [arXiv:2603.15278](https://arxiv.org/abs/2603.15278) | 预印本；未找到代码 | 围捕几何是否真正形成包围、而非只有一个 UAV 接近目标 | 为 progress head 或 high-level cost 加入“目标位于 pursuers 凸包内”和有效包围半径的连续 surrogate | **不是神经模型，却是最有价值的任务归纳偏置**。论文在二维、无界、未知 evader heading 的前提下证明保持 encirclement 的有限时间捕获；应先把其几何量做日志和辅助损失，再研究其能否在三维、有障碍和受限加速度下保持相关性。 |
| **Hydra: Navigation World Action Model**, [arXiv:2608.28995](https://arxiv.org/abs/2608.28995) | 最新预印本；在两个实体平台评测；未找到公开代码 | 在 latent 中联合采样、评估和执行候选轨迹的表征错位 | 可借鉴“离散高层 intent + 连续低层 trajectory”的双层接口 | 其 discrete latent planning + flow-matching execution 是有启发性的替代 JEPA 路线，但以视觉状态为中心。对当前 63 维结构化观测和小 archive 属于**第二阶段**，不宜先上。 |
| **Net-Carrying Drones via Competitive MARL**, [arXiv:2607.05939](https://arxiv.org/abs/2607.05939) | 预印本；未找到代码 | 目标策略非平稳、追逃策略对当前对手过拟合 | PFSP 历史 opponent pool 和低层控制随机化 | 它独立支持 AgilePE 的训练协议方向，说明 PFSP 是应保留的对手池基线。与 AgilePE 同属一条路线，**不应同时作为两个独立的“模型创新”计数**。 |
| **TERL**, [arXiv:2503.12395](https://arxiv.org/abs/2503.12395) | **IROS 2025 accepted**；[代码](https://github.com/ApricityZ/TERL) 已公开 | 规模扩大后 target assignment 和 attention-based coordination | 当任务扩展到多 evader 或更多 pursuer 时，作为 graph/transformer encoder 的实现参考 | 当前只有四机围一目标，target selection 的核心假设不成立，优先级低于 action-conditioned prediction 和 safety；其大规模 100% 数字不能与当前合同比较。 |

### 8.1 已有候选的代码与发表状态更正

为避免后续选型建立在错误前提上，第一轮清单中的以下状态已核验：

| 工作 | 核验后状态 | 含义 |
| --- | --- | --- |
| SkyJEPA | arXiv 备注为 under review；未在摘要页发现公开 GitHub 链接 | 保持第一优先级的**思想/架构**，但实现必须从小型结构化版本自建。 |
| Action-conditioned JEPA safety framework | 预印本；作者明确披露为 LIBERO-Long simulation-only，且离线 reranking 的 Level-3 significance test 未执行 | 架构与本项目最像，但其性能主张必须谨慎；只能借鉴 candidate-risk-shield 分层。 |
| Temporal-Distance-JEPA | [公开代码](https://github.com/HKBU-KnowComp/Temporal-Distance-JEPA) | 可作为 progress-cost 学习的实现参考，不等价于可直接接入 UAV policy。 |
| PCDP | [公开代码](https://github.com/marmotlab/PCDP) 与模型 | 是后续多模态 trajectory proposer 的可运行起点，仍需改为围捕目标和三维约束。 |
| AgilePE | arXiv 备注为 under review；未在摘要页发现公开 GitHub 链接 | 训练协议值得参考，不能假定可直接获得其环境和 checkpoint。 |
| SHIELD | arXiv 备注为 **IROS 2025 accepted**；未在摘要页发现公开 GitHub 链接 | 概率 CBF 的理论依据更可靠，但需要本项目自己的 dynamics-residual 数据。 |
| Online Planning for Multi-UAV Pursuit-Evasion | arXiv 备注为 **IEEE RA-L 2025 published** | 应作为已发表的直接追逃对照文献，而不是“最新预印本”的性能证据。 |

## 9. 基于核验后的优先级

下面是建议的实现顺序。每一步都只使用 development 数据和新建 experiment namespace；**不得改写 V4/V5 已锁定 block、历史训练 archive、阈值或 seed 合同**。

1. **A1 - 目标预测器池（最低风险，最高诊断价值）**：实现 constant-velocity、GRU、Transformer 和小型 action-conditioned JEPA 的统一接口；再加入 self-adaptive online weighting。先在 E1-prime 轨迹上比较 1/2/4/8 步目标误差、switch 后恢复时间和 capture-time 误差。
2. **A2 - 风险感知 CBF（直接针对 E1--E6）**：先从高斯/ensemble belief CBF 开始，不直接实现完整 SMC-PHD；输出 `p(collision)`、`p(boundary)` 与 conservative clearance，并与现有 deterministic CBF 配对报告。只有 calibration 改善后才考虑 particle/RFS 扩展。
3. **A3 - STL 约束的 candidate ranking（最可复用的开源路线）**：在不训练扩散器的第一阶段，先让现有 actor 产生 K 个 action chunks，以 STL proxy 对其排序，再交由 CBF 过滤。这样可以隔离“约束代价有效”与“diffusion proposer 有效”两种贡献。
4. **A4 - 通信/表示消融（成本低，可能解决鲁棒性）**：以 `message_age` 和丢包率分桶，对 team-coupled channels 做删除或 gate，而不是单纯加宽 RNN。应报告无通信、延迟通信和完整通信的同 seed 配对结果。
5. **A5 - 进度与围捕几何（理论归纳偏置）**：先记录 convex-hull containment、angular coverage、有效包围半径与 capture 之间的相关性；验证相关性后，再作为 Temporal-Distance-JEPA/actor 的辅助目标。
6. **A6 - 再考虑 PCDP 或 Diff-MA-STL 的完整 trajectory generator**：这一步计算量最大，且必须在 A1--A5 证明预测、风险与高层代价各自有增益之后进行。

## 10. 本轮新增参考链接

- [Diff-MA-STL (arXiv:2608.29490)](https://arxiv.org/abs/2608.29490), [code](https://github.com/jeappen/diff-ma-stl)
- [Risk-Aware Belief CBF over Random Finite Sets (arXiv:2607.15016)](https://arxiv.org/abs/2607.15016)
- [Self-Adaptive Learning and MPC for Tracking Unknown Dynamics (arXiv:2607.26370)](https://arxiv.org/abs/2607.26370)
- [Less is More: Zero-Communication 3D Pursuit-Evasion (arXiv:2603.08273)](https://arxiv.org/abs/2603.08273)
- [Encirclement Guaranteed Finite-Time Capture (arXiv:2603.15278)](https://arxiv.org/abs/2603.15278)
- [Hydra: Navigation World Action Model (arXiv:2608.28995)](https://arxiv.org/abs/2608.28995)
- [Net-Carrying Drones with Competitive MARL (arXiv:2607.05939)](https://arxiv.org/abs/2607.05939)
- [TERL (arXiv:2503.12395)](https://arxiv.org/abs/2503.12395), [code](https://github.com/ApricityZ/TERL)

## 11. 第二轮检索：最新可迁移模型与直接追逃方法（2026-09-02）

本轮使用 arXiv 官方 Atom API（T1）检索 `world model`、`action-conditioned`、`UAV pursuit-evasion`、`control barrier function` 和 `multi-agent` 组合查询，再逐条读取论文摘要页核对标题、版本日期和发表备注。以下条目是相对上一版清单新增或需要单独强调的候选；它们仍然是文献启发，**不是本仓库已经验证的模型结果**。

| 候选（核验日期） | 论文中可迁移的机制 | 对当前围捕/拦截系统的具体帮助 | 推荐接入点与最小验证 | 主要边界 |
| --- | --- | --- | --- | --- |
| **IMPACT: Attention Is the Interaction Map for Scalable Interaction-Aware World Model Training**（arXiv:2609.00161，2026-08-31） | 用交互感知的注意力/监督分配，减少静态背景对 action-conditioned world model 训练信号的淹没 | 让模型把容量集中在目标--追击机、追击机--追击机、追击机--障碍物的相对运动，而不是无关状态维度 | 在 63-D 结构化输入上加入 pairwise interaction gate；比较普通 JEPA 与 interaction-weighted JEPA 的多步目标误差和 clearance 误差 | 原工作面向视觉 world model；需要自行定义结构化交互图，不能直接沿用视觉指标 |
| **WorldEcho: Do Robotic World Models Really Follow Actions?**（arXiv:2608.24885，2026-08-25） | 专门测试 off-expert 动作是否真正改变预测未来，并用视觉完整性与 SE(3) 轨迹对齐诊断 action following | 直接对应当前 candidate reranking 的关键风险：如果 predictor 只会复现 expert 轨迹，候选动作排序会产生虚假置信度 | 对每个候选 action chunk 做 action-sensitivity test；报告预测差异、真实 rollout 差异和 action-following gap，再决定是否允许它进入 CBF 前排序 | 论文以视觉预测为主；在本项目中应换成真实位置/速度/clearance 的 action-following 指标 |
| **DreamLedger: Where to Refuse World-Model Imagination Using Execution-Settled Credit**（arXiv:2608.23863，2026-08-24） | 将预测可靠性记录为按工况、区域和预测时域索引的 execution-settled credit，并在使用预测前进行 gating | 可把 JEPA 的不确定性从一次性 log-variance 改成“该场景/障碍布局/观测 age 下历史上是否可信”，降低错误想象驱动的冒险动作 | 新增 per-condition reliability ledger；低 credit 时缩短 horizon、降低候选扰动或直接走 deterministic CBF fallback | 可靠性账本不是形式安全证明；必须保留现有 CBF 和失败回退，不可把 credit 当作安全证书 |
| **SAGE / Self-Aware Guided Exploration**（arXiv:2608.29772，2026-08-30） | 世界模型同时输出短时风险与模型不确定性（fear），将稀有失败转成定向数据收集 | 当前 archive 和 pilot 数据量有限，可优先挖掘碰撞、timeout、低净空、高 CBF correction 和观测丢失片段 | 先离线计算 hard-case priority，重采样训练 JEPA/risk head；用固定 validation block 检查是否改善长尾而非只改善平均 loss | 原工作是自动驾驶；主动采样改变训练分布，必须保留原始 validation 和清晰的数据版本记录 |
| **Instruct-to-Act: Decoupling Planning and Control for Instructable Agents**（arXiv:2608.26788，2026-08-27） | 高频 world-model controller 接受稀疏、低频的高层计划/意图 | 围捕可拆成 `approach → spread → encircle → intercept` 阶段，高层意图能减少低层策略在阶段切换时的抖动 | 将阶段 token 或 progress vector 作为 JEPA/reranker 的条件，不改变 V5 actor；固定场景先做阶段切换和 capture-time 回归 | 原工作含语言/VLM 规划，当前任务不需要 LLM；应使用可解释的几何阶段标签，避免引入额外幻觉源 |
| **Hydra: Navigation World Action Model**（arXiv:2608.28995，2026-08-29） | 在统一离散 latent 中同时进行候选采样、评估和执行，避免逐个解码高维未来 | 为未来的多 UAV intent/waypoint proposer 提供比单纯 JEPA 更紧的 planner--predictor 接口 | 先只借鉴离散 intent codebook；对 `left/right/above/below/hold` 等候选做 latent 评估，低层仍交给现有 CBF | 视觉实体平台模型较重；RTX 5050 和当前小型结构化 archive 不适合第一阶段完整复现 |
| **Evader-Agnostic Team-Based Pursuit Strategies in Partially-Observable Environments**（arXiv:2511.05812，2025-11-08） | 离线训练多种理性层级的对手，再在线分类当前 evader 并选择 best response | 适合解决当前固定 rule-expert 造成的对手分布过窄，以及目标切换/遮挡后的恢复慢问题 | 为 E1-prime 增加 opponent-policy pool；先只用于 validation 对手生成，比较单一 rule expert 与多策略池下的捕获和恢复时间 | 论文为两机城市环境；不能把其结果直接外推到四机三维障碍合同 |
| **CI-HRL: Decentralized Consensus Inference-based Hierarchical RL for Multi-Constrained UAV Pursuit-Evasion**（arXiv:2506.18126，2025-06-22） | 高层目标/定位与低层避障、编队控制分离，并通过邻居消息形成局部共识 | 对当前 `message_age`、通信延迟和队友冗余观测提供结构化建模思路；可把高层围捕进度与低层安全动作解耦 | 在现有 actor 外加轻量 consensus/progress head；按通信完整、延迟、丢包分桶做 paired test | 原任务是合作规避/覆盖，不是单目标捕获；只借鉴层次化接口和通信消融设计 |
| **SAGE-LLM**（arXiv:2602.23719，2026-02-27） | LLM 高层语义决策 + 图结构检索 + fuzzy-CBF 验证的两层架构 | 可作为未知障碍/突发威胁下的高层规则生成器研究参考，尤其适合把“封锁/绕行/撤退”转成可验证意图 | 不接入低层控制；仅离线把语义意图映射成 STL/CBF 可检查的阶段标签，测试是否提升解释性 | 预印本且依赖 LLM；对当前 63-D 结构化环境可能过度复杂，不能把语言模型输出当作安全保证 |

### 11.1 本轮新增候选的落地优先级

结合当前 V5 已有实现和 RTX 5050 资源，建议先做以下三个低风险验证：

1. **WorldEcho 风格 action-following 检查**：确认 action-conditioned JEPA 对非专家候选确实敏感；若不敏感，先修数据覆盖和训练目标，不扩大 S3 评估。
2. **IMPACT 风格 interaction gate**：在 63-D 结构化输入中显式区分 target/pursuer/obstacle/teammate 交互，和当前 JEPA 做同数据、同 seed 对照。
3. **DreamLedger + SAGE 风格可靠性与主动采样**：按条件记录预测兑现率，优先重放低净空、碰撞和 timeout 片段；所有模型选择仍由原始 validation block 决定。

其后再考虑 opponent-policy pool（Evader-Agnostic）、阶段条件控制（Instruct-to-Act/CI-HRL）和离散 latent planner（Hydra）。完整扩散或视觉 world model 暂不列为第一阶段主线，因为当前瓶颈首先是结构化状态预测、执行扰动和安全筛选，而不是像素生成质量。

### 11.2 可复现实验记录要求

- 每个新增模型建立独立的 `results/<model>_*` 命名空间，不覆盖 V4/V5 locked 输出。
- 统一记录 1/2/4/8 步目标位置误差、clearance 误差、action-following gap、风险 calibration、capture/collision/boundary/timeout 和 CBF correction。
- 先通过固定场景回归，再做同场景 paired development；至少三训练 seed 后才讨论是否值得开启新的 locked block。
- 预印本中的成功率、硬件部署和安全表述都只作为假设来源，不能改写本仓库已经锁定的 V4/V5 正式结论。

## 12. 本轮检索参考链接

- [IMPACT (arXiv:2609.00161)](https://arxiv.org/abs/2609.00161)
- [WorldEcho (arXiv:2608.24885)](https://arxiv.org/abs/2608.24885)
- [DreamLedger (arXiv:2608.23863)](https://arxiv.org/abs/2608.23863)
- [SAGE / Self-Aware Guided Exploration (arXiv:2608.29772)](https://arxiv.org/abs/2608.29772)
- [Instruct-to-Act (arXiv:2608.26788)](https://arxiv.org/abs/2608.26788)
- [Hydra (arXiv:2608.28995)](https://arxiv.org/abs/2608.28995)
- [Evader-Agnostic Team-Based Pursuit Strategies (arXiv:2511.05812)](https://arxiv.org/abs/2511.05812)
- [CI-HRL (arXiv:2506.18126)](https://arxiv.org/abs/2506.18126)
- [SAGE-LLM (arXiv:2602.23719)](https://arxiv.org/abs/2602.23719)

## 13. 第三轮检索：截至 2026-09-02 的新增可迁移方向

本轮再次通过 arXiv 官方 API 检索无人机 world model、embodied tracking、multi-agent CBF 与安全过滤关键词，并读取摘要核对方法边界。新增工作仍只作为候选依据，不改变已锁定的 V4/V5 结论，也不把论文中的成功率直接外推到本项目。

| 候选 | 论文中可迁移的机制 | 当前项目的最小接入方式 | 判断 |
| --- | --- | --- | --- |
| **AirDreamer: Generalist Drone Navigation with World Models**（[arXiv:2606.03252](https://arxiv.org/abs/2606.03252)） | world-model 环境理解 + 稀疏奖励，强调未见布局和局部最优逃逸 | 先把 JEPA 的预测目标扩展为 obstacle-relative occupancy/clearance 辅助头；不引入视觉生成 | 对随机混合障碍最有启发，但论文以导航为主，不能直接证明围捕提升；RTX 5050 上应先做结构化 clearance 头 |
| **PEACE: A Planner-Executor Agent with Constraint Enforcement for UAVs**（[arXiv:2606.00104](https://arxiv.org/abs/2606.00104)，[代码](https://github.com/erdemuysalx/PEACE)） | 高层 planner 与低层 executor 解耦，执行失败后有边界约束和有限重规划 | 将 `approach/spread/encircle/intercept` 作为低频阶段计划，低层仍使用 V5 actor + CBF | 适合解释性和故障恢复，不建议把 LLM 放入实时控制环；可先复用 planner-executor 接口思想 |
| **DeTrack / AaDWorlds: Altitude-Aware Dual World Model for Drone-Embodied Tracking**（[arXiv:2605.17451](https://arxiv.org/abs/2605.17451)） | 高低高度双 world model，处理目标可见性与飞行安全的冲突 | 给 JEPA 增加高度/视线分桶或 dual-head，分别预测 target motion 与 visibility/clearance | 对三维拦截的高度选择最直接；但其视觉 benchmark 很大，当前应先用结构化高度条件做小型消融 |
| **A Temporal Barrier Framework for Collision Avoidance in Multi-Agent Autonomous Aerial Vehicles**（[arXiv:2608.14239](https://arxiv.org/abs/2608.14239)） | 用 adversarial time-to-collision（aTTC）构造时间域 CBF，并用神经 surrogate 实时估计 | 在现有 CBF 前增加 aTTC 特征和离散 TTC 分桶；先只做日志/安全过滤对照 | 这是当前安全层最值得试的理论替代，摘要报告 3D pursuit 中更高 waypoint progress；仍必须在本项目动力学上重新验证 |
| **Scalable Tube-Tightened Multi-Agent Safety via Certified Constraint Reduction**（[arXiv:2608.25323](https://arxiv.org/abs/2608.25323)） | tube-tightened eCBF + Farkas certificate，只保留可证明足够的约束 | 当候选动作数或 UAV 数增加时，对 pairwise agent/obstacle CBF 做 certified reduction | 解决扩展规模后的 QP 计算瓶颈，不是策略提升模型；当前四机规模可作为后续效率实验 |
| **Runtime Safety Filtering for Learned Small UAS Separation Policies under GNSS Degradation**（[arXiv:2607.10014](https://arxiv.org/abs/2607.10014)） | 在有界观测不确定性下比较 action filtering 与 observation filtering | 对 delayed/noisy 条件增加 worst-case observation correction 对照，不把 CBF 过滤视为唯一方案 | 与当前 `message_age`/观测噪声失败模式高度相关；其摘要称 observation filtering 比 action filtering 更有效，但需保持本项目 CBF 安全层并做 paired test |
| **Individual CBF-Guided Diffusion for Safe Offline MARL**（[arXiv:2606.12640](https://arxiv.org/abs/2606.12640)） | 将个体 CBF 嵌入 diffusion trajectory generation，再用 inverse dynamics 执行 | 先把现有 candidate reranking 改成短 action-chunk diffusion proposer，CBF 仍做最终过滤 | 适合作为第二阶段高容量 proposer；当前 archive 较小，优先级低于 action-conditioned JEPA 和风险校准 |
| **Shared Voxel-Map-Based Cooperative Indoor UAV Guidance with MASAC**（[arXiv:2607.25728](https://arxiv.org/abs/2607.25728)） | shared world-frame voxel map + CTDE/MASAC，ego-aligned local crop | 若未来接入真实 LiDAR，用共享占据图替代当前手工障碍向量；先做 map encoder 接口烟雾测试 | 为多机空间融合提供工程路线，但当前仿真已有结构化障碍状态，不应把视觉/体素编码当作近期主线 |

### 13.1 更新后的建议

按“对当前瓶颈的直接性 × RTX 5050 可复现性”排序，建议顺序为：

1. **aTTC-CBF + worst-case observation filtering**：直接针对碰撞、低净空、观测延迟；保留现有 CBF 作为安全兜底。
2. **JEPA 的 clearance/visibility 多任务头**：吸收 AirDreamer 与 DeTrack 的结构，但保持 63-D 结构化输入，避免视觉 world model 成本。
3. **DreamLedger/SAGE reliability ledger**：按 target motion、observation condition、layout signature 和 horizon 记录预测兑现率，低信用时退回 deterministic CBF。
4. **阶段条件 planner-executor**：参考 PEACE/CI-HRL，把围捕阶段作为低频意图，不让 LLM 直接输出实时动作。
5. **Diffusion proposer 或 certified constraint reduction**：分别在数据量足够或 UAV 数量扩展后再做。

这些方向都必须沿用同一条验证纪律：固定场景回归 → 同场景 paired development → 至少三 seed → 只在预先声明的 gate 通过后才考虑新 locked block。
