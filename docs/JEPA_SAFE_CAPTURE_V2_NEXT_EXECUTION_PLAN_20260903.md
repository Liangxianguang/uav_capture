# 无人机集群对抗围捕安全增强系统
# Interaction-Aware Action-Conditioned JEPA + Reliability Ledger + CBF
# 下一步详细 TODO 与验收计划

**版本：** v2.1（P2/P3 完成后的当前执行版）
**日期：** 2026-09-03
**适用环境：** Windows、Conda `uav-encirclement-gpu`、NVIDIA RTX 5050
**运行阶段：** development-only；未经单独授权不得打开任何新的 locked test
**主目标：** safe capture 优先，建立可审计、可回退、可实时执行的闭环系统

---

## 0. 一句话目标

实现如下安全增强闭环，并证明它在同一冻结场景、三个训练 seed 上具有可复现的 safe-capture 行为：

```text
观测/通信历史
    -> interaction-aware belief state
    -> 传统规划器生成 K 个动力学可行 action chunks
    -> action-conditioned JEPA 预测未来与安全风险
    -> reliability ledger 决定 trusted / fallback_nominal / safe_hold
    -> 只对可信候选做任务-安全联合排序
    -> 严格多机 CBF/QP 最终过滤
    -> 只执行第一控制步
    -> 重新观测、重新排序、重新过滤
```

JEPA 是**轨迹评价器**，不是动作生成器；ledger 是可信度门控，不是安全证明；CBF/QP 是最后一道硬安全防线。

---

## 1. 当前状态与结论边界

### 1.1 已完成项

| 阶段 | 状态 | 已有证据 | 允许的结论 |
|---|---|---|---|
| P0 安全合同 | 完成 | `configs/jepa_safe_capture_v2_protocol.yaml`、协议测试 | 数据边界、候选语义、safe-capture 定义已冻结 |
| P1 反事实归档 | 完成 | train/validation/calibration 三份 rerun archive、manifest、SHA-256、TensorBoard 审计 | 数据可用于 P2；calibration 不进入优化器 |
| P2 多任务 JEPA | 完成 | seed `20260911/20260912/20260913`，每个 40 epoch，checkpoint 和 TensorBoard 审计 | 四个 horizon 的 target prediction 均优于 constant-velocity；仍是离线证据 |
| P2 闭环控制 | 未完成 | 尚无 P2-specific safe-capture paired aggregate | 不能声称 JEPA 已改善围捕成功率 |
| P3 ledger v2 | 完成 | 三个 checkpoint-bound calibration ledger、聚合报告、不可变运行时 API、测试 | 可门控候选排序开发；仍不是安全证明 |
| P4 候选排序 | 未完成 | 尚无当前 v2 的冻结 scorer/chunk protocol | 不得把 JEPA 预测直接接到动作执行 |
| P5 严格多机 QP-CBF | 未完成 | P1 的 QP 标签仍是 proxy；现有过滤器不能作为严格 QP 证明 | 必须实现 infeasible detection 和 safe-hold |
| P6 三 seed 闭环 development | 未完成 | 需要按本计划重新冻结运行配置 | 不得打开 locked test |

### 1.2 P2 三 seed 离线证据

当前三个 checkpoint 均满足：

- `interaction_aware_action_conditioned_jepa_safe_capture_v2` 类型正确；
- 40 epoch、all-finite、可严格加载；
- validation 为 held-out，calibration 仅记录 provenance；
- 每个 run TensorBoard 有 47 scalar、227 histogram、8 text；
- `locked_test_opened=false`。

当前 P2 checkpoint 和数据指纹如下，后续 P4--P6 必须引用这些文件或明确版本化的新文件：

| Seed | Checkpoint | SHA-256 |
|---:|---|---|
| 20260911 | `results/jepa_safe_capture_v2_p2_seed20260911/checkpoint.pt` | `3307c3935eabe0f6fb11a0dbe83ada0b4a4c610a1d96911a67c81cd6c66760e7` |
| 20260912 | `results/jepa_safe_capture_v2_p2_seed20260912/checkpoint.pt` | `a95aaf56acce704aa7abec8bd3042309b2085cdca755a25f424ba1662ab4355c` |
| 20260913 | `results/jepa_safe_capture_v2_p2_seed20260913/checkpoint.pt` | `4fac01bbd49a0028485a87b07c10c1f27365c14a3298bc4f25f42f57c9072798` |

正式 P2 validation dataset SHA-256 为
`48af3ce3bd83a7aa4d068d1f25c8311df706cf892c88d51690dd595c2643ccc7`；train dataset
SHA-256 为 `3186b05ce145303658b3fdb87ff5c3868ac8330170ad8f24515d93d9ced2ecfd`。

历史 `jepa_v3_*` replay、旧 P5/P6 和单 seed development 输出只能作为诊断或负结果，不能替代本计划的 v2 checkpoint、ledger 或三 seed paired evaluation；不得把它们混入新的训练、校准或主报告。

### 1.3 P3 已完成的 ledger 指纹

P3 已在同一 calibration split 上完成三个 seed 的独立校准，运行时只读，不允许在线更新。后续 P4--P6 必须使用下列 ledger，不得重新用 development episode 调阈值：

| Seed | Ledger | TensorBoard aggregate |
|---:|---|---|
| 20260911 | `results/jepa_safe_capture_v2_p3_rerun_ledger_seed20260911/reliability_ledger.json` | `results/jepa_safe_capture_v2_tensorboard/p3_rerun_ledger_aggregate` |
| 20260912 | `results/jepa_safe_capture_v2_p3_rerun_ledger_seed20260912/reliability_ledger.json` | 同上 |
| 20260913 | `results/jepa_safe_capture_v2_p3_rerun_ledger_seed20260913/reliability_ledger.json` | 同上 |

P3 聚合状态总量为 `trusted=857531`、`fallback_nominal=64069`，当前 calibration 未触发 `safe_hold`。high-credit unsafe-rate gate 和 OOD/stale/hard-context routing 均通过；由于 `labels_cbf_qp_feasible` 没有类别变化，QP 可行性仍不能宣称已校准。完整证据见 `docs/JEPA_SAFE_CAPTURE_P3_LEDGER_V2_20260903.md`。

target MAE 相对 constant velocity 的改善范围为：

| Horizon | seed 20260911 | seed 20260912 | seed 20260913 |
|---:|---:|---:|---:|
| 0.1 s | 16.8% | 9.8% | 13.3% |
| 0.2 s | 29.4% | 25.0% | 27.0% |
| 0.3 s | 36.9% | 32.8% | 34.3% |
| 0.5 s | 45.0% | 43.5% | 43.1% |

这些结果只能支持 `prediction_improvement_no_control_gain` 之前的离线阶段判断。P2 没有证明：

1. 候选排序选择了更好的动作；
2. 预测漂移能被 ledger 及时拒绝；
3. CBF-QP 在多约束同时激活时可行；
4. safe capture 相对冻结 V5 nominal 有稳定提升。

### 1.4 安全优先级

指标优先级固定为：

1. `safe_capture`；
2. collision、boundary、pairwise minimum separation、CBF-QP infeasible；
3. transit、safe-hold/fallback 行为和实时性；
4. 配对 improved/degraded/tied；
5. clearance、CBF correction、候选切换和可见性；
6. capture time、路径长度和显存/算力成本。

`mean capture time` 必须完整报告，但不是单独的 rejection gate；不要求绝对 safe-capture 达到 95%。

---

## 2. 不可变系统合同

以下规则写入配置、运行时检查和测试，后续不得因单个 seed 结果修改：

### 2.1 信息边界

- 在线输入只能使用观测、通信历史、动作历史、障碍几何和 observation/message age；
- 在线禁止读取 target ground truth；ground truth 只在 offline settled-label 结算使用；
- train、validation、calibration、development、locked episode/seed 不得交叉；
- `locked_test_opened=false` 是训练、评估和聚合脚本的硬校验。

### 2.2 动作和滚动时域

- 每周期生成 `K=5` 个候选；候选包含 nominal、拦截、侧向绕障、队形修正和可见性保持方向；
- 初始 chunk 长度固定为 3 个 control steps；只执行所选 chunk 的第一步；
- 下一周期必须重新观测、重新预测、重新排序；不得 open-loop 执行整个 chunk；
- 候选必须先通过速度、加速度、动作变化和数值有限性检查；
- JEPA 不得生成最终 action，也不得覆盖 CBF 输出。

### 2.3 安全合同

- baseline 和 candidate 必须使用同一个 CBF/QP 实现和同一安全参数；
- 同时约束 obstacle、defender-defender、boundary、altitude、speed、acceleration；
- QP infeasible、观测过期或推理超时必须进入确定性 safe-hold/fallback；
- 不得用降低安全 margin 换取 safe capture；
- 安全失败按 episode 记录，不能用其它 episode 的成功抵消。

---

## 3. 总体阶段路线

```text
P2-A 预测 gate aggregate [完成]
  -> P3 可靠性 ledger v2 校准与 abstention [完成]
  -> P4 候选 action-chunk 生成与 JEPA 排序
  -> P5 严格多机 CBF/QP 与 fallback
  -> P6 三 seed paired safe-capture development
  -> P7 可复现性审计与是否另行预注册 locked 的决策
  -> (可选) P8 SIL/HIL，不自动进入真实飞行
```

当前唯一允许的实现顺序是：P4 完成并通过回归后才能进入 P5；P5 未完成或 QP infeasible/fallback 未通过，禁止运行 P6 闭环结论；P6 未通过安全硬门不能申请新的 locked evaluation。

### 3.1 当前执行队列（从现在开始）

1. **P4 候选生成与排序：** 先完成模块、单元测试、zero-perturbation 和 action-following audit；此阶段只允许离线/synthetic replay，不运行正式闭环结论。
2. **P5 联合 CBF/QP：** 在 P4 回归通过后实现真实 solver、infeasible 检测和 fallback ladder；任何 proxy 结果都不能替代这一阶段。
3. **P6 paired development：** 只有 P5 的安全硬门和延迟门都通过，才冻结 M0--M3/A1--A3，先 smoke，再三 seed 全量运行。
4. **P7 readiness：** 汇总 provenance、TensorBoard、逐 episode trace 和统计区间；结果不足时保持 `insufficient_evidence_do_not_open_locked_test`，不为了得到正结果调整参数。

每个阶段结束时必须先生成报告和测试摘要，再进行该阶段的独立提交；不得把未完成阶段的结果写入正式 V4/V5 或 locked 结论。

---

## 4. P2-A：三 seed prediction aggregate（已完成）

P2-A 已由提交 `5a2c8f1` 完成。它只证明 held-out prediction gate，不证明闭环控制收益。

### 已完成与验收结果

- [x] 三个 `prediction_gate.json` 去重、验证 validation hash/model type，并拒绝 locked 标记；
- [x] 汇总 target、clearance、visibility、CBF intervention、observation-age 等 horizon 指标；
- [x] 三个 seed 在四个 horizon 上均优于 constant velocity；
- [x] `qp_feasibility_auc=n/a` 已明确记录，因为 P1 QP 标签无类别变化；
- [x] 训练审计、aggregate JSON、Markdown 报告和 TensorBoard provenance 已生成；
- [x] P2 已明确标记为“离线预测通过、闭环未验证”。

报告：`docs/JEPA_SAFE_CAPTURE_P2_AGGREGATE_20260903.md`。当前不需要重跑或修改 P2 checkpoint；除非输入 hash 发生变化，否则下一步直接使用 P3 ledger 进入 P4。

---

## 5. P3：Reliability Ledger v2 与 abstention（已完成）

**目标：** 识别“模型很自信但预测已漂移”的情况，并在候选排序前拒绝不可信预测。

### 5.1 Ledger 输入和分桶

在 calibration split 上离线结算每个 `(checkpoint, context bucket, horizon, head)`：

- visibility condition、observation age、message age；
- obstacle count、layout signature、minimum clearance；
- target motion mode、target speed scale、转向/加速度突变；
- defender side、队形拥挤度、pairwise TTC；
- JEPA uncertainty、candidate separation、预测 CBF intervention risk。

每个 bucket 至少保存 `sample_count`、误差均值/P50/P90/P95、coverage、Brier/ECE、settled failure rate、ledger version 和输入 hash。

### 5.2 已完成与验收结果

- [x] 生成三个 checkpoint-bound v2 ledger，并在 `SafeCaptureReliabilityLedger` 中实现只读 schema 校验；
- [x] 固定 `trusted`、`fallback_nominal`、`safe_hold` 三态，支持 exact context、coarse context、global horizon fallback；
- [x] 绑定 checkpoint、calibration metadata、protocol 和 builder source 的 SHA-256；
- [x] 设置 `immutable_after_calibration=true`，运行时拒绝写入；
- [x] 完成 high-credit failure-rate、OOD/stale/hard-context routing 和确定性测试；
- [x] 生成三 seed aggregate、TensorBoard provenance 和 JSON/Markdown 报告。

P3 的实际结果是 `trusted=857531`、`fallback_nominal=64069`，当前 calibration 没有触发 `safe_hold`。这表示 ledger 可以作为 P4 的候选排序门控，但不表示 QP 可行性或闭环安全已经证明。

### 5.3 P3 的保留限制

- `labels_cbf_qp_feasible` 没有类别变化，因此不能用 QP head 计算或宣称可行性校准；
- ledger 只能拒绝不可信的 JEPA 排序信号，不能替代 CBF/QP；
- P4/P6 不得使用 development episode 重新调整 ledger 阈值；
- 若 P4/P5 发现 high-credit 比 fallback 更危险，立即冻结 ledger，全部退回 frozen nominal + CBF。

报告：`docs/JEPA_SAFE_CAPTURE_P3_LEDGER_V2_20260903.md`。三份运行时 ledger 路径已列在 §1.3。

### 5.4 失败处理

若 ledger 不能区分高低信用，保留 checkpoint 作为预测研究结果，但运行时全部使用 frozen nominal + CBF；不得调低阈值伪造通过。

---

## 6. P4：候选动作块生成与轨迹重排序

**目标：** 验证 JEPA 是否真正改变了“候选选择”，而不只是改善离线预测。

### 6.1 固定第一版候选合同

| 项目 | 冻结值 |
|---|---:|
| 每周期候选数 | `K=5` |
| 主 chunk | 3 control steps |
| 诊断 chunk | 5 steps，仅在 3-step 通过后启用 |
| action perturbation | `0.10 m/s` |
| 执行 | 第一控制步后立即 replan |
| nominal anchor | 必须包含，且作为 tie-break |
| CBF | 所有 baseline/candidate 统一启用 |

### 6.2 TODO

- [ ] 新建候选生成模块，例如 `src/encirclement3d/jepa_safe_capture_candidates.py`；
- [ ] 新建排序模块，例如 `src/encirclement3d/jepa_safe_capture_ranker.py`；
- [ ] 候选 0 固定为 V5 nominal；其余候选分别表达拦截、侧向绕障、队形净空和可见性保持；
- [ ] 实现 action chunk encode/decode、动态可行性预检查和只执行第一步；
- [ ] 实现冻结的 score：task progress - uncertainty penalty - conservative clearance risk - CBF risk + visibility gain - action-change cost；
- [ ] 任何预测的安全量只用于排序，不得绕过 CBF；
- [ ] 引入 nominal anchor 和最小分数 margin，防止模型在低信用时远离 nominal；
- [ ] 记录 candidate id、rank、score、预测量、ledger state、fallback reason、CBF correction 和最终 action；
- [ ] 加入 action-following audit，验证不同候选确实产生非平凡且方向一致的预测差异；
- [ ] 增加 no-ledger、no-clearance-head、no-visibility-head、nominal-only 消融；
- [ ] 写 `docs/JEPA_SAFE_CAPTURE_P4_CANDIDATE_RANKING_20260903.md`。

### 6.3 P4 必测回归

- [ ] 全 nominal candidate 集合与 frozen V5 + CBF 逐字段一致；
- [ ] candidate 数量、chunk 长度、action scale、坐标轴和 replan 语义严格匹配协议；
- [ ] 非 finite、越界、超速、超加速度候选在进入 JEPA 前被拒绝；
- [ ] 空候选集执行明确的 nominal CBF fallback；
- [ ] 连续周期不能出现无界 candidate oscillation；
- [ ] 所有 ranking trace 可按 episode/time/agent 复盘。

---

## 7. P5：真实多机 CBF/QP、安全过滤与 safe-hold

**目标：** 把 P1 的 feasibility proxy 替换为真正的联合约束求解与不可行检测。该阶段是进入闭环评估的硬前置条件。

### 7.1 必须实现的约束

- obstacle separation；
- 每一对 defender 的 pairwise separation；
- world boundary、最低/最高高度；
- speed、acceleration 和 action slew rate；
- target approach/capture 约束只能作为任务约束，不能削弱安全约束。

### 7.2 TODO

- [ ] 新建独立的 QP-CBF 模块，例如 `src/encirclement3d/cbf_qp.py`，不把 projection proxy 继续命名为 QP；
- [ ] 为每个约束输出离散时间 barrier residual、active/inactive、slack 和单位；
- [ ] 对所有 defender 联合求解，而不是独立求解后再拼接；
- [ ] 固定 solver、容差、最大迭代、超时和 deterministic seed；
- [ ] 明确 infeasible、timeout、non-finite 和 stale observation 的区分；
- [ ] 实现 fallback ladder：`safe_hold -> nominal through CBF -> controlled hover/abort`；
- [ ] 任何 fallback 必须保留 pairwise separation、boundary 和 obstacle 安全约束；
- [ ] 记录 solver status、active constraints、slack、correction norm、solve latency 和 fallback reason；
- [ ] 若使用 OSQP/CVXPy，先测 CPU 延迟和部署依赖，再把版本写入 Conda environment lock；
- [ ] 新增单元测试、随机约束测试、同时激活约束测试、QP infeasible 测试和 determinism 测试；
- [ ] 做 zero-perturbation regression：candidate 和 nominal 经过同一 QP 后完全一致；
- [ ] 写 `docs/JEPA_SAFE_CAPTURE_P5_CBF_QP_AUDIT_20260903.md`。

### 7.3 P5 安全硬门

- [ ] 所有可执行 candidate 的 collision/boundary 计数为 0；
- [ ] pairwise minimum separation 不低于冻结安全阈值；
- [ ] QP infeasible 不得静默执行未经过滤动作；
- [ ] CBF correction 超阈值时必须告警并进入 fallback 分桶；
- [ ] p95 `JEPA + ledger + ranker + CBF` 延迟不超过 100 ms；超时执行 nominal CBF；
- [ ] 代码测试证明 JEPA 没有后置覆盖 CBF 输出。

如果不能实现真实 infeasible detection，P6 只能运行 deterministic nominal + 现有安全过滤器的诊断，不得声称完成安全增强系统。

---

## 8. P6：三 seed 配对 safe-capture development

**目标：** 在新闭环中回答“JEPA 评价器是否在不降低安全性的前提下改善 safe capture”。

### 8.1 评估矩阵

| ID | 执行栈 | 用途 |
|---|---|---|
| M0 | frozen V5 nominal + 同一 CBF/QP | 固定主基线 |
| M1 | P2 JEPA + nominal fallback + CBF/QP | 分离 ledger 前的模型作用 |
| M2 | P2 JEPA + ledger v2 + CBF/QP | 主安全增强候选 |
| M3 | P2 + ledger + candidate ranker + CBF/QP | 最终系统候选 |
| A1 | M3 去掉 ledger | 漂移/幻觉诊断 |
| A2 | M3 去掉 clearance/visibility heads | 辅助任务消融 |
| A3 | raw action/no CBF | 仅仿真诊断，不进入安全结论 |

所有模型与 M0 使用同一 episode seed、layout、初始状态、target motion 和 transit reference。M3 必须完成 `3 x 60 = 180` 个 paired development episodes。

### 8.2 场景覆盖

- nominal flee；
- delayed/noisy observation；
- target s-curve、速度突变、频繁随机转向；
- 3--5 个障碍和 narrow-channel 低净空；
- 高拥挤队形和 pairwise TTC 较低片段；
- 左右起始侧、不同 visibility 和 message age。

### 8.3 执行顺序

1. [ ] 清空新的 results namespace，写入 protocol、checkpoint、ledger、代码和环境 hash；
2. [ ] 每个变体/seed 先跑 20 paired smoke；
3. [ ] smoke 出现新 collision、boundary、非 finite、unpaired 或 zero-regression 失败时立即停止该变体；
4. [ ] smoke 通过后冻结所有参数，不再根据 smoke 成绩调权重、chunk、阈值或 seed；
5. [ ] 运行每个最终变体 `60 episodes/seed`；
6. [ ] 生成逐 episode trace、summary、paired statistics 和 TensorBoard evaluation audit；
7. [ ] 对失败逐条关联 training seed、episode seed、layout、target mode、observation condition、ledger state、candidate id 和 CBF trace。

### 8.4 主指标与统计

主指标：

- safe capture count/rate；
- collision、boundary、pairwise separation、CBF-QP infeasible；
- transit success、safe-hold 和 fallback rate。

次指标：

- paired improved/degraded/tied；
- 三 seed mean paired safe-capture delta 和 bootstrap CI；
- 必要时使用 McNemar/exact paired test；
- minimum clearance、CBF correction/intervention、candidate switch rate；
- capture time、path length、inference/CBF latency。

### 8.5 预注册决策门

**G1 安全硬门：** candidate collision = 0、boundary = 0、无新的 pairwise separation 违规；zero-perturbation 回归逐字段一致。任一失败即拒绝该变体。

**G2 safe-capture non-inferiority：** 三 seed 平均 paired delta 不低于 `0` 个百分点，至少 2/3 seed 非负；不要求绝对 95%。

**G3 正向主张：** 平均 paired delta 严格为正、至少 2/3 seed 非负，并报告预先指定的 bootstrap/McNemar 结果；否则只能称 non-inferior 或 safety-preserving。

**G4 reliability：** high-credit 失败率不高于 low-credit，OOD/低信用必须触发 fallback。

**G5 realtime：** p95 总控制延迟不超过 100 ms，超时有可观测 nominal CBF fallback。

**G6 provenance：** 所有输入、命令、输出、日志、hash 和失败 trace 可重建。

capture time/path 增加不能单独否决方案，但必须透明报告；若它们导致大量 timeout、safe-hold 或实时性失败，则按安全/任务失败处理。

### 8.6 结果分类

| 分类 | 含义 |
|---|---|
| `promising_development_candidate` | 三 seed 安全通过，safe capture 有一致正向证据，代价可解释 |
| `useful_safety_fallback_only` | 捕获未改善，但长尾风险、漂移或回退质量明显改善 |
| `prediction_improvement_no_control_gain` | 离线预测/校准改善，闭环没有净收益 |
| `rejected_for_instability` | 出现无法解释的安全失败、非确定性或严重实时性问题 |
| `insufficient_evidence_do_not_open_locked_test` | seed 冲突、区间过宽或审计不足 |

---

## 9. P7：审计、报告与是否申请新的 locked evaluation

### TODO

- [ ] 汇总 P2-A、P3、P4、P5、P6 的 JSON/Markdown/CSV；
- [ ] 对每个 checkpoint、ledger、protocol、archive、scene、source file 计算 SHA-256；
- [ ] 从空 results namespace 重跑最小复现实验；
- [ ] 审计所有 TensorBoard：标量、直方图、文本 provenance、评估标签和运行时延迟；
- [ ] 检查历史 V4/V5 locked 文件未被修改；
- [ ] 明确 prediction、safety-preserving、non-inferiority、positive development claim 的证据边界；
- [ ] 写 `docs/JEPA_SAFE_CAPTURE_V2_P7_READINESS_20260903.md`；
- [ ] 只有 P6 classified 为 `promising_development_candidate` 时，另写新的 preregistration；
- [ ] 新 locked block 必须得到用户明确授权，不能由脚本自动打开。

### 新 locked evaluation 的最低条件

1. P6 G1--G6 全部通过；
2. 三 seed 结果趋势可解释，不依赖单个最佳 seed；
3. 独立重跑能复现主趋势；
4. 新场景、seed、停止规则、主/次指标和统计方法先冻结；
5. 明确 locked 数据不可回流训练、ledger 或参数调节。

---

## 10. TensorBoard 与 provenance 要求

每次训练、校准和闭环评估均使用独立 logdir，禁止覆盖旧 run。至少记录：

```text
Loss/*
Target/*
Clearance/*
InterAgent/*
Visibility/*
Risk/*
Calibration/*
Reliability/*
CBF/*
Ranking/*
Fallback/*
Latency/*
Optimization/*
Data/*
Provenance/*
```

文本 provenance 必须包含：protocol、training config、model config、dataset metadata、checkpoint hash、ledger hash、scene hash、Git commit、Conda/Python/PyTorch/CUDA 版本和命令行。

闭环每个 episode 至少记录：

- safe capture/collision/boundary/timeout/transit；
- minimum obstacle/inter-agent clearance；
- CBF solver status、active constraints、infeasible、correction norm；
- candidate rank、score margin、selected candidate、switch rate；
- ledger state、credit、bucket、fallback reason；
- control/JEPA/ledger/CBF latency 和 watchdog timeout。

---

## 11. 文件与提交边界

### 建议新增文件

```text
scripts/aggregate_jepa_safe_capture_v2_prediction.py
scripts/build_jepa_safe_capture_v2_reliability_ledger.py
src/encirclement3d/jepa_safe_capture_candidates.py
src/encirclement3d/jepa_safe_capture_ranker.py
src/encirclement3d/cbf_qp.py
tests/test_aggregate_jepa_safe_capture_v2_prediction.py
tests/test_jepa_safe_capture_v2_reliability.py
tests/test_jepa_safe_capture_candidates.py
tests/test_cbf_qp.py
docs/JEPA_SAFE_CAPTURE_P2_AGGREGATE_20260903.md
docs/JEPA_SAFE_CAPTURE_P3_LEDGER_V2_20260903.md
docs/JEPA_SAFE_CAPTURE_P4_CANDIDATE_RANKING_20260903.md
docs/JEPA_SAFE_CAPTURE_P5_CBF_QP_AUDIT_20260903.md
docs/JEPA_SAFE_CAPTURE_V2_P6_DEVELOPMENT_20260903.md
docs/JEPA_SAFE_CAPTURE_V2_P7_READINESS_20260903.md
```

### 每阶段单独提交

| 阶段 | 建议提交信息 |
|---|---|
| P2-A | `docs(jepa): aggregate safe-capture v2 prediction gates` |
| P3 | `feat(jepa): add checkpoint-bound reliability ledger v2` |
| P4 | `feat(jepa): rerank feasible action chunks with jepa` |
| P5 | `feat(safety): add deterministic multi-agent cbf qp fallback` |
| P6 | `docs(jepa): record three-seed safe-capture development` |
| P7 | `docs(jepa): audit safe-capture v2 readiness` |

提交时只暂存当前阶段文件。`results/`、`tmp/`、NPZ archive、TensorBoard event 和用户已有的 E1/V5 改动不得混入阶段提交。每次提交后执行普通 `git push origin main`；不得 force push。

---

## 12. RTX 5050 执行安排

| 顺序 | 阶段 | 状态/主要耗时 | 完成标志 |
|---:|---|---|---|
| 1 | P2-A aggregate | 已完成 | 三 seed gate、审计和报告已生成 |
| 2 | P3 ledger | 已完成 | calibration-only ledger、三态回退和测试已完成 |
| 3 | P4 candidate ranking | **当前，1--2 天** | zero regression、action-following、smoke harness 完成 |
| 4 | P5 QP-CBF | 后续，1--3 天 | infeasible/fallback/latency 测试完成 |
| 5 | P6 smoke20 | 后续，0.5--1 天 | 每个 seed/变体安全 smoke 通过 |
| 6 | P6 final 3x60 | 后续，1--3 天 | 180 paired episodes、失败 trace 和统计完成 |
| 7 | P7 audit | 最后，0.5--1 天 | readiness decision 完成，不自动开 locked |

训练、评估和 QP 运行都使用：

```powershell
$py = 'D:\miniconda3\envs\uav-encirclement-gpu\python.exe'
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

每个长任务开始前确认输出目录不存在或为空；禁止覆盖已有 checkpoint、TensorBoard 或报告。

---

## 13. 失败、回退和停止规则

- 新 collision 或 boundary：立即停止当前变体，保存 trace，回退 nominal + CBF；
- zero-perturbation 不一致：停止，不继续调 ranker；
- non-finite、checkpoint/hash 不一致或数据泄漏：停止并重建输入；
- high-credit 比 low-credit 更危险：拒绝当前 ledger，保持 nominal；
- QP infeasible 没有 safe-hold：P5 不通过，不能进入 P6；
- p95 延迟超预算：记录 watchdog fallback，先做工程优化，不能隐藏超时；
- 三 seed 结果冲突：分类为 evidence insufficient，不打开 locked；
- capture time 变差但 safe capture 和所有安全硬门通过：保留并完整报告，不以时间单独否决；
- 任何“为了提高捕获率”而放宽 CBF margin、绕过 fallback 或回灌 validation/development 数据的行为：直接拒绝该版本。

---

## 14. 最终完成定义（Definition of Done）

本计划只有在以下条件全部满足时才算完成：

1. P2-A aggregate、P3 ledger、P4 ranker、P5 QP-CBF、P6 三 seed paired development 和 P7 audit 均有独立报告；
2. 所有代码、配置、测试、命令、checkpoint、ledger、scene 和数据 hash 可追溯；
3. TensorBoard 记录完整，且可从空目录重放；
4. candidate 永远经过 CBF，QP infeasible 永远进入显式 safe-hold/fallback；
5. safe capture、collision、boundary、pairwise separation 和 fallback 按 episode 审计；
6. 结果诚实区分 prediction-only、safety-preserving、non-inferiority 和 positive development claim；
7. 没有用户授权前，历史或新 locked test 均保持关闭。

**最重要的判断标准不是某个 seed 的最高成功率，而是：在安全不退化的前提下，JEPA 预测是否被可靠性账本正确使用，是否通过候选重排序产生可复现的 safe-capture 改善，以及所有失败是否都能被 CBF/fallback 解释和控制。**
