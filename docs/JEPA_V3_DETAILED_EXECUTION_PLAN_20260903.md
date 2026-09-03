# Interaction-Aware Action-Conditioned JEPA + CBF
# JEPA-v3 详细执行计划

**计划版本**：v1.0

**制定日期**：2026-09-03

**适用仓库**：`Liangxianguang/uav_capture`

**适用硬件**：NVIDIA RTX 5050，CUDA GPU 环境 `uav-encirclement-gpu`

**计划性质**：development 计划，不是 V4/V5 locked test 方案

---

## 1. 计划摘要

本计划的主线是继续增强已经完成初步验证的 **interaction-aware action-conditioned JEPA + CBF**。当前证据表明，这条路线比普通 action-conditioned JEPA 更稳定，但收益仍然有限，且路径长度和捕获时间存在代价。因此下一阶段不直接更换为更大的视觉 world model、扩散模型或 LLM 控制器，而是围绕四个可诊断问题逐步增强：

1. JEPA 是否真正学习了候选动作对目标、净空和可见性的影响；
2. 在什么工况下可以相信 JEPA，什么时候应该回退到冻结 V5 actor + CBF；
3. 困难片段重放能否改善长尾失败，而不是只改善平均训练损失；
4. 短 action chunk 是否比当前单步微扰更能带来有效拦截意图，同时不增加安全风险。

最终目标不是预先承诺某个成功率数字，而是建立一套可审计的、可复现的模型增强链路，并用新的 development 证据决定是否值得预注册全新的 locked block。

核心执行链路为：

```text
冻结 V5 actor
    -> 生成物理可行候选 action/chunk
    -> interaction-aware JEPA 多任务预测
       (target displacement + clearance + visibility + risk)
    -> reliability ledger 判断是否可信
    -> candidate ranking
    -> 现有 CBF 最终安全过滤
    -> 只执行第一步并重新观测
```

**关键原则**：JEPA、reliability ledger 和 candidate ranking 只能影响候选动作偏好；CBF 仍然是最后的安全执行层，不能被不确定性分数或预测置信度替代。

---

## 2. 当前已知基线与证据

### 2.1 当前冻结 baseline

| 项目 | 固定值 |
| --- | --- |
| actor | `models/v5_development_exact_reactive_seed661606.pt` |
| 环境 | `configs/capture_radius_pursuit_central_v4_flee.yaml` |
| S3 development protocol | `configs/central_random_mixed_obstacle_s3_v5_protocol.yaml` |
| episode 数 | 60 |
| CBF | 开启 |
| recurrent reset interval | 1 |
| 场景配对 | 使用同一 `scenes.jsonl` 和 `episodes.csv` |
| baseline safe capture | `57/60 = 95.00%` |
| baseline collision / boundary | `0% / 0%` |
| baseline transit | `100%` |

### 2.2 已完成 JEPA-v2 development evidence

| 模型 | 三 seed safe capture | collision / boundary | paired capture delta | 结论 |
| --- | ---: | ---: | ---: | --- |
| 普通 action-conditioned JEPA | `93.89% +/- 0.96%` | `1.11% +/- 1.92%` / 同值 | `-1.11 +/- 0.96 pp` | 不作为主线 |
| interaction-aware JEPA | `96.11% +/- 0.96%` | `0% / 0%` | `+1.11 +/- 0.96 pp` | 当前主线 |

interaction-aware 三个 seed 分别为 `58/60`、`57/60`、`58/60`。逐 episode 配对结果共计 5 次改善、3 次退化、172 次持平。相对 baseline，平均总路径增加约 `3.55 m`，捕获时间也略有增加。

### 2.3 当前证据的边界

- 以上结果是 development evidence，不是新的正式 locked 结论。
- V4/V5 历史 locked protocol、checkpoint、archive 和正式报告不得修改。
- v1 action history 错位结果必须继续标记为 invalid smoke，不得进入新汇总。
- action-following 审计显示候选动作会改变预测输出，但尚未证明预测变化与真实 rollout 变化在 episode 层面充分一致。
- 预测 uncertainty 不是安全证明；低 uncertainty 也可能对应错误预测。

现有报告：

- `docs/JEPA_V2_INTEGRATION_DEVELOPMENT_REPORT_20260902.md`
- `docs/JEPA_V2_ACTION_FOLLOWING_AUDIT_20260902.md`
- `docs/LATEST_MODEL_CANDIDATES_FOR_PURSUIT_INTERCEPTION_20260902.md`

---

## 3. 研究问题与可证伪假设

### Q1：多任务预测是否比只预测目标位移更有用？

**H1**：加入 obstacle/teammate clearance、target visibility 和 CBF intervention risk 辅助头后，预测校准和候选排序质量改善，并且在三 seed paired control 中不增加碰撞或边界失败。

### Q2：reliability ledger 是否能降低长尾错误？

**H2**：按 motion mode、observation condition、layout signature 和 horizon 记录历史兑现率，并在低信用 bucket 回退到 V5 + CBF，可以减少错误想象造成的退化；如果只降低模型使用率而不改善结果，应如实判定为无效。

### Q3：困难片段重放是否改善 worst-case，而非只改善平均 loss？

**H3**：在保持至少 50% 原始均匀样本的条件下，hard-example replay 能降低 timeout、低净空和大 CBF correction 片段的失败率，同时不损害普通场景。

### Q4：action chunk 是否比单步微扰更适合拦截？

**H4**：长度 3--5 steps 的动力学可行 chunk 能提高候选动作的短期区分度，减少动作抖动；若路径代价或安全失败上升，则保留单步候选作为 fallback。

---

## 4. 不可变边界与数据治理

### 4.1 不得修改的内容

以下内容在整个 JEPA-v3 development 期间保持只读：

- V4/V5 locked protocol、locked seed block 和历史正式结果；
- `models/v5_development_exact_reactive_seed661606.pt`；
- 原有 V5 baseline 的场景、episode seed 和 transit evidence；
- 已完成 JEPA-v2 checkpoint、原始 train/validation 数据和 metadata；
- `src/encirclement3d/pursuit_env.py` 中用户已有的本地动力学草稿；
- 任何用于历史报告的 archive 文件。

### 4.2 新实验命名空间

建议统一使用以下路径，不覆盖已有结果：

```text
configs/jepa_v3_development_protocol.yaml
results/jepa_v3_counterfactual_train/
results/jepa_v3_counterfactual_validation/
results/jepa_v3_<model>_<seed>_<split>/
docs/JEPA_V3_*.md
scripts/*jepa_v3*.py
```

每个结果目录至少包含：

- `episodes.csv`；
- `scenes.jsonl`；
- `summary.json`；
- `evaluation_metadata.json`；
- 使用的 protocol 副本；
- checkpoint 或 checkpoint 的绝对路径和 SHA-256；
- 数据源 manifest 和 Git commit。

### 4.3 数据分割规则

- train：可用于训练和 hard-example replay；
- validation：用于模型选择、阈值选择和 smoke；
- development S3：用于冻结场景 paired control；
- locked test：本计划不打开，除非另行预注册并完成 gate。

任何 validation 或 S3 episode 都不得通过 replay 进入训练集。若需要从失败片段生成新数据，必须从 train split 中恢复相同类型的状态，或建立明确的 synthetic/counterfactual split。

---

## 5. 阶段总览与依赖关系

| 阶段 | 名称 | 主要产物 | 依赖 | 预计时间 |
| --- | --- | --- | --- | ---: |
| P0 | 冻结与预注册 | protocol、hash manifest、回归基线 | 当前 v2 证据 | 2--3 天 |
| P1 | Counterfactual 数据 | 同状态多动作真实短期标签 | P0 | 4--6 天 |
| P2 | 多任务 JEPA | clearance/visibility/risk heads | P1 | 7--10 天 |
| P3 | Reliability ledger | 信用表、阈值、回退策略 | P2 | 4--6 天 |
| P4 | Hard-example replay | 重采样器、训练对照 | P2 | 4--6 天 |
| P5 | Action chunk | 可行 chunk proposer/reranker | P3 | 7--10 天 |
| P6 | 三 seed 消融 | M0--M5 paired development | P2--P5 | 7--14 天 |
| P7 | 新 locked 前审计 | preregistration 或停止决策 | P6 | 2--4 天 |

依赖图：

```text
P0 -> P1 -> P2 -> P3 ----> P5 --+
             |                    |
             +-> P4 --------------+-> P6 -> P7
```

P3 和 P4 可以在 P2 的 prediction gate 通过后并行，但 P5 必须使用已经固定的 ledger 接口。P6 前不得根据 S3 development 结果反复修改候选评分权重。

---

## 6. P0：冻结当前证据并建立 v3 protocol

### 6.1 任务清单

- [ ] 创建 `configs/jepa_v3_development_protocol.yaml`。
- [ ] 记录 V5 actor、JEPA-v2 checkpoint、数据和场景文件 SHA-256。
- [ ] 记录 Python、PyTorch、CUDA、GPU 型号和驱动信息。
- [ ] 记录当前 baseline、interaction-aware v2 和 plain v2 的完整 summary。
- [ ] 将 v1 action-misaligned 结果明确列入 invalid artifacts。
- [ ] 定义新模型、阈值、候选数和 chunk length 的预注册范围。
- [ ] 规定任何 locked test 在 P7 前保持关闭。

### 6.2 建议 protocol 字段

```yaml
protocol_name: jepa_v3_interaction_aware_development
phase: development_only
not_a_replacement_for_locked_benchmarks: true
base_actor_checkpoint: ...
base_environment_config: ...
base_s3_scenes: ...
horizons_steps: [1, 2, 3, 5]
training_seeds: [20260911, 20260912, 20260913]
cbf_enabled: true
locked_test_opened: false
```

### 6.3 P0 验收门

P0 只有在以下条件全部满足时完成：

- 文件 hash、场景配对和 checkpoint hash 可独立复核；
- baseline 重播与当前记录一致；
- protocol 明确写出 `not_a_locked_test: true`；
- 所有新实验可通过独立目录追溯到 protocol 和 Git commit。

**建议提交**：`docs(jepa): preregister v3 development protocol`

---

## 7. P1：构建 counterfactual action-conditioned 数据集

### 7.1 目标

解决训练数据几乎全部来自 expert/V5 已执行动作的问题。模型必须看到“同一状态下，不同动作会导致什么真实短期后果”，否则 candidate reranking 可能只是在选择更像 expert 的动作。

### 7.2 每个样本的内容

每个快照至少保存：

| 标签 | 具体定义 |
| --- | --- |
| observation history | 长度 8 的 63-D policy-safe observation |
| action history | 与 observation 因果对齐的 outgoing actions |
| candidate action/chunk | 当前候选，已归一化并满足动力学限制 |
| target displacement | 0.1/0.2/0.3/0.5 s 的相对位移 |
| obstacle clearance | 每个时域内 UAV--障碍物最小净空 |
| inter-agent clearance | 每个时域内 UAV--UAV 最小间距 |
| visibility | 目标可见性、visible fraction、observation age |
| CBF intervention | 是否触发、修正范数和最大修正 |
| risk/termination proxy | collision、boundary、timeout 或近失败指标 |

### 7.3 候选采样策略

从 train trajectory 快照中采样：

- 1 个 nominal actor action；
- 1 个 zero perturbation action；
- 4--8 个相关扰动 action/chunk；
- 少量朝障碍、远离障碍、切向绕障和保持队形的定向候选。

每个候选必须通过：

1. 速度上限检查；
2. 加速度/动作变化检查；
3. 控制周期和 chunk 长度检查；
4. 数值有限性检查；
5. 不提前使用未来 target truth 的信息边界检查。

### 7.4 P1 评估

- 候选动作覆盖率：每个状态的有效候选数；
- action perturbation 分布；
- 各 motion mode、obstacle count 和 observation condition 的样本数；
- counterfactual rollout 的真实失败率；
- 与原 expert-only 数据的重复比例；
- train/validation hash 和状态来源审计。

### 7.5 P1 验收门

- 所有样本的 action history 因果对齐；
- 至少 4 个候选/状态的有效样本比例达到预设阈值；
- 每个主要工况均有样本，不允许某一工况占比超过总量的 50%；
- 至少 50% 样本保持原始或近似均匀分布；
- validation 和 development 场景没有进入训练文件；
- 随机抽样可重放真实标签。

**建议提交**：`data(jepa): add counterfactual development dataset protocol`

---

## 8. P2：训练多任务 interaction-aware JEPA

### 8.1 模型接口

保持当前 interaction-aware encoder 的 63-D 输入和 action-conditioned history 接口，不修改冻结 V5 actor 的 observation contract。模型新增输出：

```text
target_mean[h, 3]
target_log_variance[h, 3]
clearance_mean[h, risk_channel]
visibility_probability[h]
cbf_intervention_probability[h]
latent[h, d]
```

建议 risk channel 至少区分：

- obstacle clearance risk；
- inter-agent clearance risk；
- boundary risk；
- large-CBF-correction risk。

### 8.2 训练损失

```text
L = L_JEPA
  + lambda_target * L_target
  + lambda_clearance * L_clearance
  + lambda_visibility * L_visibility
  + lambda_risk * L_risk
  + lambda_calibration * L_calibration
```

初始实验只改变一个因素，避免无法归因：

| 版本 | 输出头 |
| --- | --- |
| M1 | 当前 v2 interaction-aware target displacement |
| M2 | M1 + clearance |
| M3 | M2 + visibility |
| M4 | M3 + intervention/risk |

### 8.3 训练与选择纪律

- 三个训练 seed 与 v2 保持一致，便于比较；
- 训练使用 CUDA，固定 batch、优化器、epoch 上限和 early stopping 规则；
- 每个 checkpoint 保存完整 model config、state dict、source hash 和数据 hash；
- 不用 S3 development 结果调学习率或损失权重；
- 先看 prediction gate，再看 control result；
- 不因为单个 seed 的峰值结果选择 checkpoint。

### 8.4 P2 预测验收门

每个候选模型必须报告：

- 0.1/0.2/0.3/0.5 s target position error；
- clearance MAE/RMSE 和分位数误差；
- visibility AUROC、Brier score 或 calibration curve；
- intervention/risk AUROC、AUPRC、Brier score；
- uncertainty coverage 和 negative log-likelihood；
- action-following separation、antisymmetry 和 non-trivial response；
- all-finite 和异常输出比例。

建议 gate：

- 三 seed target prediction 不弱于 v2 interaction-aware；
- 至少一个新增辅助任务在三 seed 中稳定改善；
- action-following audit 不退化；
- 低净空和大 CBF correction 区间不能出现系统性过度自信；
- 任何有限性或 shape contract 失败直接拒绝进入控制测试。

**建议提交**：`feat(jepa): add clearance and visibility auxiliary heads`

---

## 9. P3：建立 reliability ledger 与回退策略

### 9.1 设计目标

ledger 用历史执行结果记录“在什么条件下预测器可靠”，而不是把一次 forward pass 的 uncertainty 当作安全证书。

### 9.2 ledger 索引

至少按以下维度分桶：

- training seed；
- prediction horizon；
- `target_motion_mode`；
- `observation_condition`；
- message age；
- visible fraction；
- obstacle count/layout signature；
- minimum clearance；
- action perturbation/chunk magnitude。

### 9.3 每个 bucket 记录

- target displacement error；
- clearance error；
- visibility error；
- uncertainty coverage；
- candidate ranking win/tie/loss；
- CBF correction；
- collision、boundary、timeout；
- model-used 与 fallback-used 次数。

### 9.4 执行策略

| ledger 状态 | 策略 |
| --- | --- |
| 高信用且样本足够 | 使用 0.3--0.5 s 多任务 score |
| 中信用或样本不足 | 缩短 horizon，降低候选扰动 |
| 低信用或 OOD | 回退到冻结 V5 nominal action + CBF |
| 预测风险高 | 只允许安全候选，禁止激进 chunk |

ledger 的更新必须使用已完成 rollout 的 settled outcome，不能在同一 episode 中用未来标签即时修改当前决策。

### 9.5 P3 验收门

- ledger bucket 的样本量、更新时间和数据来源可审计；
- 在零扰动回归中，ledger 不改变冻结 V5 行为；
- 低信用回退逻辑不会绕过 CBF；
- ledger 开启后，至少在一个预注册困难分桶降低退化或风险指标；
- 如果 ledger 仅减少 JEPA 使用率但没有改善任何结果，判定为诊断工具而非有效增强。

**建议提交**：`feat(jepa): add reliability ledger fallback policy`

---

## 10. P4：困难片段重放与数据重加权

### 10.1 hard-example 定义

优先收集：

- timeout；
- collision 或 boundary violation；
- minimum clearance 低于阈值；
- CBF correction 高于阈值；
- target prediction error 大；
- prediction confidence 高但真实结果错误；
- delayed/noisy、burst、random-turn 等工况。

### 10.2 采样优先级

```text
P_i = w_failure * failure
    + w_clearance * low_clearance
    + w_cbf * high_cbf_correction
    + w_calibration * calibration_error
    + w_ood * out_of_distribution
```

权重必须在 train split 内预先确定。不得根据最终 development 60 episode 结果临时改变权重。

### 10.3 对照实验

只比较以下两个版本：

- `replay-off`：原始采样；
- `replay-on`：hard-example 重加权，且至少 50% 样本仍来自原始均匀池。

两个版本使用完全相同的 seed、训练步数、验证集和模型结构。

### 10.4 P4 验收门

- validation 不被 replay 污染；
- source rollout、片段 index、权重和 hash 完整记录；
- replay-on 的 hard-case 指标改善至少一个；
- 普通 nominal 分桶不出现显著安全退化；
- 失败不是通过人为删除困难样本获得。

**建议提交**：`feat(jepa): add hard-example replay for world-model training`

---

## 11. P5：从单步微扰升级为 action chunk

### 11.1 目标

当前 `0.005 m/s` 单步微扰已经能产生小幅正向证据，但它更像局部动作抖动，未必代表可执行的拦截意图。P5 将候选表示为 0.3--0.5 s 的短动作 chunk，并每次只执行 chunk 的第一步。

### 11.2 初始参数范围

| 参数 | 第一阶段值 |
| --- | --- |
| chunk length | 3 steps、5 steps 分别测试 |
| candidate count | K=5，必要时 K=8 |
| replanning | 每个控制周期重新规划 |
| CBF | 始终开启 |
| action change penalty | 继承 v2，另行预注册上限 |
| fallback | nominal actor action |

### 11.3 候选来源

- V5 nominal action chunk；
- 低曲率持续偏移；
- 朝目标拦截方向；
- 切向绕障方向；
- 增大 UAV 间距的队形修正；
- 降低 obstacle/inter-agent clearance risk 的安全候选。

所有候选先进行动力学可行性检查，再进入 JEPA 评分。

### 11.4 评分函数

```text
J(a[0:H]) = w_target * predicted_target_cost
          + w_clearance * predicted_clearance_risk
          + w_visibility * predicted_visibility_risk
          + w_uncertainty * predicted_uncertainty
          + w_change * action_change_cost
          + w_ledger * ledger_penalty
```

评分方向和归一化方法必须写入 protocol。不能在观察到某个 seed 退化后临时反转单个权重。

### 11.5 P5 回归要求

1. **零候选扰动回归**：候选集合只含 nominal action 时，逐 episode 与 V5 + CBF 完全一致；
2. **单步 v2 对照**：同场景比较单步和 chunk，记录动作变化率、CBF correction、路径和捕获时间；
3. **20 episode smoke**：先查异常、崩溃和安全失败；
4. **60 episode development**：仅在 smoke 通过后运行。

**建议提交**：`feat(jepa): rerank dynamically feasible action chunks`

---

## 12. P6：严格消融与三 seed paired development

### 12.1 最小消融矩阵

| 编号 | 配置 | 用途 |
| --- | --- | --- |
| M0 | frozen V5 actor + CBF | 固定 baseline |
| M1 | M0 + interaction-aware JEPA-v2 | 当前已验证主线 |
| M2 | M1 + clearance/visibility heads | 测试多任务预测 |
| M3 | M2 + reliability ledger | 测试可信度回退 |
| M4 | M3 + hard-example replay | 测试长尾修正 |
| M5 | M4 + action-chunk reranking | 测试短期拦截意图 |

M2--M5 每个版本先做 20 episode smoke。只有最终候选和必要的对照才进入完整 60 episode paired development；完整三 seed 必须至少覆盖 M1、M3、M5，若资源允许覆盖全部版本。

### 12.2 每次实验必须报告

**安全主指标**：

- safe capture rate/count；
- collision rate/count；
- boundary violation rate/count；
- timeout 和 safety failure；
- transit route feasibility/success。

**效率指标**：

- capture time；
- total defender path length；
- minimum clearance；
- mean/max CBF action correction；
- action change rate/chunk smoothness。

**预测指标**：

- target、clearance、visibility、risk 多时域误差；
- calibration；
- action-following gap；
- ledger used/fallback used 比例。

### 12.3 配对统计

每个 seed 内必须使用相同的：

- episode index；
- episode seed；
- layout seed；
- scenario metadata；
- transit evidence。

至少计算：

- improved / degraded / tied episode 数；
- safe capture percentage-point delta；
- joint-success capture-time delta；
- path-length delta；
- safety-failure only-in-method 与 only-in-baseline；
- 按 observation condition、motion mode、obstacle count 分桶的 delta。

不要把 180 个 episode 当作 180 个独立训练 seed，也不要只报告均值而隐藏退化 episode。

### 12.4 建议接受门槛

以下是 development gate，不是正式论文结论：

- 三 seed 中不出现比 M1 更差的 collision/boundary 安全趋势；
- 三 seed 平均 safe capture 不低于 M1，或改善证据与安全代价明确可解释；
- 至少 2/3 seed 的 paired capture delta 非负；
- 路径和捕获时间增加不超过预注册上限，建议先设为 10%；
- prediction、calibration、action-following 三类 gate 均通过；
- 所有失败 episode 都有可追溯的场景、候选、ledger 和 CBF 记录。

这里**不设置必须超过 95% 的绝对门槛**。95% 是当前 baseline 的观测值，不应变成为了过门而调参的目标。

**建议提交**：`test(jepa): add v3 paired development ablation protocol`

---

## 13. P7：是否开启全新的 locked block

只有在 P6 完成后三 seed development 后，才允许讨论新的 locked evaluation。P7 不允许使用新 locked 结果反向修改模型。

### 13.1 前置条件

- M5 的模型、ledger、候选生成器和 CBF 参数均已冻结；
- 新 seed block 与所有 development seed 不重叠；
- 新场景至少 100 episodes，或在 protocol 中解释统计功效；
- 主次终点、失败规则和统计单位预先写明；
- 所有 checkpoint、数据和 source hash 已发布；
- 执行人员无法在运行中查看并调参。

### 13.2 需要单独预注册

建议文件：`docs/JEPA_V3_LOCKED_EVALUATION_PREREGISTRATION.md`

必须明确：

- 新的 episode/layout seed block；
- checkpoint hash；
- JEPA horizon、candidate count、chunk length；
- ledger threshold 和 fallback 规则；
- CBF 参数；
- safe capture、collision、boundary、timeout、path、time 的主次终点；
- 单次执行和失败处理规则。

如果 P6 没有形成稳定的安全与收益证据，正确决策是**不打开 locked block**，并把结果写成开发阶段否定或未决结论。

---

## 14. 复现实验命令模板

以下命令为模板，实际路径和参数必须以最终 protocol 为准。所有命令使用 GPU conda 环境：

```powershell
$py = "D:\\miniconda3\\envs\\uav-encirclement-gpu\\python.exe"

& $py -m pytest -q

& $py scripts/generate_jepa_v3_counterfactual_dataset.py `
  --protocol configs/jepa_v3_development_protocol.yaml `
  --split train `
  --output-dir results/jepa_v3_counterfactual_train

& $py scripts/train_interaction_aware_jepa_multitask.py `
  --dataset results/jepa_v3_counterfactual_train/...npz `
  --metadata results/jepa_v3_counterfactual_train/metadata.json `
  --seed 20260911 `
  --output-dir results/jepa_v3_multitask_seed20260911

& $py scripts/evaluate_jepa_v3_prediction.py `
  --checkpoint results/jepa_v3_multitask_seed20260911/checkpoint.pt `
  --dataset results/jepa_v3_counterfactual_validation/...npz `
  --output results/jepa_v3_multitask_seed20260911/prediction_gate.json

& $py scripts/evaluate_random_central_mixed_obstacles.py `
  --method f2 `
  --checkpoint models/v5_development_exact_reactive_seed661606.pt `
  --protocol configs/central_random_mixed_obstacle_s3_v5_protocol.yaml `
  --environment-config configs/capture_radius_pursuit_central_v4_flee.yaml `
  --split validation --episodes 20 `
  --output-dir results/jepa_v3_m5_seed20260911_smoke `
  --use-cbf `
  --reference-episodes results/jepa_v2_control_baseline60/episodes.csv `
  --reference-scenes results/jepa_v2_control_baseline60/scenes.jsonl `
  --device cuda
```

命令模板中的新脚本名可以在 P0/P1 实现时调整，但 protocol、输出字段和信息边界必须保持不变。

---

## 15. 质量保证与审计清单

### 每个 checkpoint

- [ ] model type/config/state dict 完整；
- [ ] train/validation 数据 hash；
- [ ] source file hash；
- [ ] seed、optimizer、epoch、device metadata；
- [ ] prediction gate JSON；
- [ ] action-following audit；
- [ ] finite output 和 shape contract 测试。

### 每个 control run

- [ ] 场景和 episode 逐项配对；
- [ ] CBF 开启状态明确；
- [ ] recurrent reset interval 明确；
- [ ] candidate 数、扰动/chunk 参数明确；
- [ ] ledger 使用或 fallback 记录；
- [ ] episodes.csv、scenes.jsonl、summary.json 齐全；
- [ ] failure episode 可单独重放；
- [ ] 不存在 locked-test 标记误用。

### 每个阶段提交

- [ ] 只提交本阶段相关文件；
- [ ] 不提交用户已有 E1-prime 或 `tmp` 无关改动；
- [ ] commit message 使用 conventional commit；
- [ ] `pytest -q` 或对应目标测试通过；
- [ ] 文档写明结果是 development evidence；
- [ ] push 前检查 `git diff --check` 和 `git status`。

---

## 16. 风险、失败模式与停止规则

| 风险 | 早期信号 | 处置 |
| --- | --- | --- |
| counterfactual 覆盖不足 | action sensitivity 低、候选输出近似相同 | 增加 train-only 候选采样，暂停扩大 control |
| 多任务头互相干扰 | target gate 下降、clearance 也不改善 | 回退到 M1，分头训练或降低辅助损失 |
| ledger 过度回退 | JEPA 使用率极低且结果不变 | 保留 ledger 作为诊断，停止把它称为性能增强 |
| hard replay 过拟合 | hard-case 改善但 nominal 退化 | 提高均匀池比例，限制 replay 权重 |
| chunk 造成抖动 | action-change、CBF correction、path 明显上升 | 回退单步候选或缩短 chunk |
| CBF 频繁覆盖 JEPA | selected action 与 executed action 差异很大 | 优先改善候选可行性，不削弱 CBF |
| 预测置信度失校准 | 高置信错误集中在低净空/延迟场景 | 强制 ledger fallback，重新校准 risk head |
| 单 seed 偶然提升 | 三 seed 方差大、paired delta 符号不一致 | 不开 locked test，扩大 train-only 数据或停止该模块 |

### 16.1 立即停止条件

出现以下任一情况时，停止当前模块扩展并写失败报告：

- 任一新增模块在 smoke 中产生未记录的碰撞、边界越界或 NaN；
- 零扰动回归改变冻结 V5 行为；
- 发现 validation/locked 数据进入训练；
- action history 再次出现时间错位；
- CBF 被绕过、关闭或被 uncertainty 分数替代；
- 结果无法按 episode 和 checkpoint 完整重放。

---

## 17. 时间表与里程碑

| 周期 | 里程碑 | 输出 |
| --- | --- | --- |
| 第 1 周 | P0 完成，P1 数据协议和首批 train 数据 | protocol、manifest、dataset smoke |
| 第 2 周 | P1 完成，P2 M1/M2 训练 | counterfactual dataset、prediction gate |
| 第 3 周 | P2 M3/M4 完成 | 多任务 checkpoint 和 calibration report |
| 第 4 周 | P3 ledger 完成 | ledger schema、fallback smoke |
| 第 5 周 | P4 replay 对照完成 | replay-on/off prediction report |
| 第 6 周 | P5 chunk proposer 完成 | zero-perturbation、20 episode smoke |
| 第 7 周 | P6 首轮消融 | M0--M5 20 episode paired smoke |
| 第 8 周 | P6 三 seed development 完成 | aggregate JSON/MD、failure analysis |
| 第 9 周 | P7 决策 | locked preregistration 或停止/转向报告 |

RTX 5050 上优先采用小型结构化模型、混合精度和批量离线预测。若训练时间超出预算，应减少模型宽度或候选数量，不应删除安全评估、seed 或配对场景。

---

## 18. 最终决策框架

P6 完成后按以下顺序做决定：

1. **安全优先**：是否保持零 collision/boundary，或至少不劣于 M1；
2. **可重复性**：三 seed 的方向是否一致，是否存在单 seed 驱动的虚假提升；
3. **预测可信度**：辅助头和 action-following 是否与真实 rollout 对齐；
4. **效率代价**：路径、捕获时间、CBF correction 是否在可接受范围；
5. **复杂度收益比**：ledger、replay、chunk 的收益是否值得维护成本；
6. **是否开新 locked**：只有前五项全部有正向证据，才提交新的 locked preregistration。

建议的最终判定标签：

- `promising_development_candidate`：安全不退化，三 seed 有一致正向证据；
- `useful_safety_fallback_only`：主要价值是减少长尾风险，成功率未提升；
- `prediction_improvement_no_control_gain`：预测指标改善但闭环无收益；
- `rejected_for_instability`：出现安全、配对或可复现性问题；
- `insufficient_evidence_do_not_open_locked_test`：证据不足，不开启 locked。

本计划的预期结果不是保证 M5 一定超过当前 `95%` baseline，而是保证每一步都能回答“哪个模块有效、在哪些工况有效、代价是什么、是否值得继续”。

---

## 19. 推荐执行优先级

在资源有限时，严格按以下顺序执行：

1. **Counterfactual 数据覆盖与因果对齐**；
2. **Clearance/visibility 多任务头**；
3. **Reliability ledger 与 deterministic fallback**；
4. **Hard-example replay**；
5. **Action chunk candidate reranking**；
6. **新 locked evaluation**。

如果 P1 或 P2 不能证明动作敏感性、净空预测和校准质量，直接跳到更大的 diffusion/world model 没有诊断价值。只有在这条结构化路线已经显示清晰收益后，才考虑 PCDP、Diff-MA-STL 或视觉 world model 等高成本方向。

