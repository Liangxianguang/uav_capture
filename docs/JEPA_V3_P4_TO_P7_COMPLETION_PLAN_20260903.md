# Interaction-Aware Action-Conditioned JEPA + CBF
# P4--P7 完整执行计划

**版本：** v1.1（续执行版）
**日期：** 2026-09-03
**性质：** development-only；不打开、不替代、不重解释 V4/V5 locked test
**执行环境：** Windows + `uav-encirclement-gpu` + NVIDIA RTX 5050/CUDA
**前序计划：** `docs/JEPA_V3_DETAILED_EXECUTION_PLAN_20260903.md`

---

## 1. 目标与核心结论标准

### 1.1 目标

完成以下闭环，判断 **interaction-aware action-conditioned JEPA + CBF** 是否在本项目的围捕拦截任务中具有可重复的开发价值：

```text
冻结 V5 actor
  -> 物理可行的候选动作/动作块
  -> interaction-aware、多任务、action-conditioned JEPA
  -> reliability ledger 的可信度判断与确定性回退
  -> 只改变候选排序
  -> CBF 最终安全过滤
  -> 每周期只执行第一步并重新观测
```

该方法试图处理的真实矛盾是：围捕控制需要在短时域内利用目标、障碍物、队友和可见性之间的交互预测来改善动作选择，但预测器的不确定性不能成为绕过安全机制的理由。JEPA 的职责是提供候选之间的比较信号；CBF 始终是实际执行前的最后安全层。

### 1.2 不设绝对捕获率门槛

本计划**不要求**最终候选达到预先设定的 `95%` 绝对安全捕获率。原因是冻结的 V5 development baseline 本身为单 seed 的 `57/60 = 95.0%`，把它当作所有后续模型的硬下限会掩盖有价值的安全回退、长尾改善或效率改善。

最终是否“有效”只依据预先规定的相对和配对证据：

1. CBF 始终启用，且任何新的 collision/boundary 失败都必须单独诊断，不能用平均捕获率抵消；
2. 最终候选相对冻结 V5 + CBF 的三 seed、同场景、逐 episode 配对结果；
3. 改善不能由单个训练 seed 驱动；
4. 路径长度、捕获时间、CBF 修正和动作抖动的代价必须透明报告；
5. 所有训练、数据、checkpoint、评估场景和代码版本必须可追溯。

### 1.3 禁止事项

- 不修改 V4/V5 locked protocol、历史 checkpoint、历史 archive 或正式报告；
- 不用 60-episode S3 development 结果调 replay 权重、ledger 阈值、chunk 长度或候选评分权重；
- 不把单 seed、20 episode smoke 或 prediction gate 写成正式捕获率提升；
- 不允许 JEPA、ledger、ranker 或 uncertainty 绕过 CBF；
- 不将 validation 或 S3 development 样本混入训练/replay 数据。

---

## 2. 当前冻结状态

### 2.1 已完成阶段

| 阶段 | 状态 | 已有证据/产物 | 可作出的结论 |
| --- | --- | --- | --- |
| P0：输入冻结 | 完成 | protocol、hash manifest、开发边界 | V3 仅用于 development，locked test 保持关闭 |
| P1：反事实数据 | 完成 | train/validation 各 `146,400` 样本、5 candidate/state-agent | action history 已按 actor `action_scale=5.0` 因果对齐；无开发/locked 数据泄漏 |
| P2：多任务 JEPA | 完成 | 三个 40-epoch CUDA runs、TensorBoard、prediction gates | 在 `0.2/0.3/0.5 s`，三 seed target MAE 均优于 constant velocity；这仍不是闭环捕获证据 |
| P3：reliability ledger | 完成 | hash-bound ledger、确定性 nominal fallback、零扰动回归 | 单 seed、20 episode 非零 smoke 为 `19/20` 对 `18/20`，仅可作为开发信号 |
| P4：困难片段 replay | 完成，未准入控制 | train-only weights、matched replay-off/on、held-out hard-subset audit | replay-on 在 hard/non-hard target 与 clearance 指标系统退化；保留为负结果，P5 回退 P3 |
| P5：动作块候选 | 未开始 | 无 | 不得抢跑 |
| P6：最终三 seed paired development | 未开始 | 无 | 不得作“JEPA 有效”结论 |
| P7：新 locked 前审计 | 未开始 | 无 | 是否预注册新 locked block 尚未决定 |

### 2.2 关键冻结事实

| 项目 | 冻结值/证据 |
| --- | --- |
| 基线 actor | `models/v5_development_exact_reactive_seed661606.pt` |
| actor SHA-256 | `535098773be05687e147043435649378532362d479bdc0375842970370ba40ba` |
| CBF | 每个学习动作路径均启用，且为最后过滤器 |
| S3 development | 60 episode、配对 `scenes.jsonl`/`episodes.csv`，只用于最终 development 评估 |
| P1 train hash | `6e8609484139fad93b427d8069f8f2517c472383962c036f2b6de9bd03c4b895` |
| P1 validation hash | `2176f09196a6c88787271ee7ee3f375311176163622c4df032fc078d107514d4` |
| P2 seeds | `20260911`, `20260912`, `20260913` |
| P3 零扰动回归 | 20 paired episodes、87 个非 JEPA 字段均为 0 差异；scene SHA-256 为 `1402bf6429814f7638625025bc75a3b4ca04ac3c0bc107eef13ac0cdf2a18b99` |

### 2.3 P4 的当前边界

P4 的 replay manifest 来自**训练集**，总计 `146,400` 样本，其中 `14,559` 个（`9.94%`）被标为困难样本；当前定义下它们均来自高 CBF correction，未观察到低净空、collision 或 boundary 标签。每个 epoch 的抽样中固定 `50%` 为均匀池，另 `50%` 通过权重 `3.0` 的困难样本抽样，因而不是删除普通样本的选择性训练。

已完成的匹配训练仅针对 seed `20260911`：

| 版本 | best validation loss | 结论边界 |
| --- | ---: | --- |
| replay-off | `-3.5229868505821855` | 与原始 seed `20260911` 行为一致，仍需完整 gate 文件 |
| replay-on | `-3.4300990752965377` | prediction gate 已通过；不能以总 validation loss 判断 replay 成败 |

因此 P4 的重点不是继续比较总 loss，而是验证高 CBF correction 子集、普通子集和最终闭环控制的差异。

---

## 3. 统一实验纪律

### 3.1 命名和目录

```text
results/jepa_v3_hard_replay/
results/jepa_v3_multitask_replay_{off,on}_seed<seed>/
results/jepa_v3_p5_<variant>_seed<seed>_<split>/
results/jepa_v3_p6_<variant>_seed<seed>_{smoke20,dev60}/
results/jepa_v3_tensorboard/<run-name>/
docs/JEPA_V3_P<stage>_*.md
```

每个训练或评估目录都必须保留：有效 protocol 副本或其 SHA-256、Git commit、输入数据与 metadata SHA-256、checkpoint SHA-256、设备信息、随机 seed、`episodes.csv`、`scenes.jsonl`、`summary.json` 和 `evaluation_metadata.json`。生成性结果目录保持忽略，不将大文件提交 Git。

### 3.2 TensorBoard 最低要求

所有新训练均写入独立 TensorBoard run，至少记录：

- `Loss/*`、`Target/*`、`Clearance/*`、`Visibility/*`、`Risk/*`、`Calibration/*`；
- `Optimization/learning_rate`、epoch/step、梯度或参数直方图；
- replay 时的 `Replay/uniform_draw_fraction` 与 `Replay/hard_draw_fraction`；
- `Dataset/train_replay_manifest`、protocol、source hashes、训练/验证 metadata 的 text artifact；
- 训练结束的 best epoch、best validation loss、checkpoint SHA-256 和 wall-clock time。

启动训练前必须先确认新的 logdir 为空；任何覆盖已有 run 的命令直接停止。

### 3.3 Git 交付纪律

每一阶段先完成报告和测试，再用单一、常规提交提交该阶段的代码与文档，随后普通 `git push origin main`。不纳入提交的内容包括 `results/`、`tmp/`、历史 archive、E1-prime 的用户工作及与 JEPA 阶段无关的变更。

建议提交边界：

| 阶段 | 提交信息 |
| --- | --- |
| P4 | `feat(jepa): add audited hard-example replay` |
| P5 | `feat(jepa): rerank feasible action chunks with jepa` |
| P6 | `docs(jepa): record three-seed development evaluation` |
| P7 | `docs(jepa): audit locked-test readiness decision` |

---

## 4. P4：困难片段重放

### 4.1 目的

检验 train-only hard-example replay 是否改善模型在高 CBF correction 片段上的预测和候选排序，而不是仅通过优化总体训练损失制造表面差异。

### 4.2 执行顺序

1. **完成可复现性检查。** 验证 weights、manifest、train dataset、metadata 和 protocol 哈希一致；确认 `uniform_fraction=0.50`，且权重源仅为 train split。
2. **补齐匹配的 held-out prediction gate。** replay-off 和 replay-on 都必须在同一个 P1 validation dataset 上生成 `prediction_gate.json`，CUDA、batch size 和 evaluator 版本一致。
3. **新增 hard-subset evaluator。** 用 manifest 定义的高 CBF correction 标签在 validation 内重建同一选择规则，分别报告 hard、non-hard、overall 三个子集。该 evaluator 只读 validation 数据，绝不回写 replay weights。
4. **比较预测与校准。** 对四个 horizon 分别报告 target MAE、clearance MAE、inter-agent clearance MAE、visibility AUROC/Brier、CBF intervention AUROC/Brier、coverage，以及 candidate ranking win/tie/loss。
5. **执行 action-following audit。** 对 replay-on checkpoint 验证候选动作改变时输出具备非平凡分离、方向一致性和有限值；用原 v3 audit 口径对 replay-off 作为对照。
6. **形成 P4 报告和判定。** 只在上述离线审计完成后决定是否允许 replay-on 进入 P5 的运行时链路。

### 4.3 P4 接受/停止门

| 类别 | 必须满足的条件 | 处理方式 |
| --- | --- | --- |
| 数据完整性 | 所有 manifest/hash 验证通过；没有 validation/S3 数据进入 replay | 任一失败即停止，重建权重 |
| 数值/接口 | 两个 checkpoint 在 held-out data 上 all-finite，shape 和 action-scale contract 正确 | 任一失败即拒绝 replay-on |
| 主任务 | replay-on 在 `0.2/0.3/0.5 s` 至少保留对 constant velocity 的正向 target prediction 证据 | 不满足则不进入控制 |
| 困难子集 | replay-on 必须在 high-CBF subset 至少改善一个预先报告指标，且不能让其它安全相关预测指标出现明显反向变化 | 无净收益则标为“无效重加权”，保留 P3 但不带 replay |
| 普通子集 | non-hard subset 不得出现系统性退化；使用 paired bootstrap CI 与逐 horizon 表共同判断 | 退化无法解释则拒绝 replay-on |
| 审计 | action-following、TensorBoard provenance、单元/集成测试通过 | 未通过不得进入 P5 |

“明显反向变化”必须由点估计、区间估计、每个 horizon 和 seed 分布共同报告，不可仅由单一总 loss 或挑选最优 horizon 判定。P4 的这轮开发对照使用 seed `20260911`，即使接受也只表示“可作为 P5 输入”，不等于三 seed 结论。

### 4.4 P4 可复现命令模板

```powershell
$py = 'D:\miniconda3\envs\uav-encirclement-gpu\python.exe'

& $py scripts/evaluate_jepa_v3_multitask.py `
  --checkpoint results/jepa_v3_multitask_replay_off_seed20260911/checkpoint.pt `
  --dataset results/jepa_v3_counterfactual_validation/counterfactual_multitask_dataset.npz `
  --metadata results/jepa_v3_counterfactual_validation/metadata.json `
  --output results/jepa_v3_multitask_replay_off_seed20260911/prediction_gate.json `
  --device cuda

& $py -m pytest -q tests/test_jepa_v3_replay.py
```

训练重跑必须使用 `scripts/train_interaction_aware_jepa_multitask.py` 的 `--train-replay-weights`、`--train-replay-manifest` 与独立 `--tensorboard-logdir`，并保持 manifest 的 `--replay-uniform-fraction 0.50`。

### 4.5 P4 产物

- `scripts/build_jepa_v3_hard_replay_weights.py` 与对应测试；
- replay-off/on 的完整 gate JSON；
- hard/non-hard/overall 对照 CSV/JSON 和统计脚本；
- `docs/JEPA_V3_P4_HARD_REPLAY_REPORT_20260903.md`；
- 一个仅包含 P4 文件的 Git commit 与远端 push。

---

## 5. P5：物理可行动作块候选与重排序

### 5.1 前置条件

只有 P4 被标为“可进入控制”或被明确排除且回退到 P3 model 时，才能开始 P5。P5 开始前先创建一个不可修改的 P5 protocol 增量，记录：checkpoint SHA-256、ledger SHA-256、候选数量、chunk 长度、速度/加速度约束、score 归一化、所有权重、测试 seed 和场景文件 SHA-256。

### 5.2 最小实现

第一轮只实现一个小而可诊断的配置：

| 项目 | 初始值 |
| --- | --- |
| 主 action chunk | 3 control steps |
| 诊断 chunk | 5 control steps，仅在 3-step 有效且安全后启用 |
| 每周期候选数 | `K=5` |
| 候选构成 | nominal、低曲率持续偏移、目标拦截、切向绕障、队形/净空修正 |
| 执行方式 | 每周期仅执行所选 chunk 的第一步，然后重新观测和重新规划 |
| learning action | 仅选择候选；不直接输出未约束控制量 |
| fallback | low-credit/sparse/OOD context 使用冻结 V5 nominal action |
| CBF | 始终在 ranker 之后执行 |

候选进入 JEPA 前必须通过速度、加速度、动作变化、数值有限性和环境坐标契约检查。候选评分使用已冻结的 target、clearance、visibility、risk、uncertainty、action-change 和 ledger penalty 项；不得根据同一 60-episode development 成绩回改任一权重。

### 5.3 P5 测试顺序

1. **单元测试：** action chunk encode/decode、只执行第一步、动态约束、fallback 和 CBF 顺序；
2. **零候选扰动回归：** 候选集合全为 nominal 时，逐 episode 必须与冻结 V5 + CBF 完全一致；
3. **单 seed 离线诊断：** 记录候选分数、预测风险、ledger fallback、CBF correction、动作变化率和 chunk 首步选择；
4. **20 episode paired smoke：** 与同一 baseline `scenes.jsonl` 对齐，先排除崩溃、安全失败和异常路径；
5. **单 seed 60 episode development：** 仅当 smoke 未出现 collision/boundary、且所有 provenance 完整时运行；
6. **冻结设计：** 单 seed 60 episode 只用于诊断。完成后不得以其结果再调参，直接进入 P6 的三 seed 评估或停止。

### 5.4 P5 通过门

- 零扰动回归为逐 episode 精确一致；
- action chunk 不改变 observer、actor、CBF 或环境的输入契约；
- 20 episode smoke 不出现新的 collision/boundary，且能解释所有 timeout/异常；
- action-change rate、mean/max CBF correction、路径长度和捕获时间均被报告；
- 相对 single-step P3/P4 对照，至少一个预先定义的闭环诊断有净收益，且没有无法解释的安全风险；
- 没有使用 S3 结果选择 chunk 长度、权重或检查点。

P5 不通过时，输出失败报告并将最终候选回退为 P3 或 P4；不为了“继续实验”扩展到更大候选集合或更长 chunk。

---

## 6. P6：最终三 seed、逐 episode 配对 development 评估

### 6.1 要回答的问题

P6 是唯一允许对“该 JEPA 方案是否在此 development 任务中有效”作出完整判断的阶段。它不能被单 seed prediction gate、单 seed smoke 或最优 checkpoint 替代。

### 6.2 评估矩阵

| ID | 执行栈 | 目的 |
| --- | --- | --- |
| M0 | frozen V5 actor + CBF | 固定 paired baseline |
| M1 | 已完成的 interaction-aware JEPA-v2 + CBF | 已有主线参考；不重新调参 |
| M3 | P2 multitask + P3 ledger + CBF | 确认可靠性回退本身的贡献 |
| M4 | P3 + 经 P4 接受的 replay checkpoint + CBF | 判断 replay 是否有闭环价值；若 P4 被拒绝则注明不运行 |
| M5 | P4/P3 最终可接受模型 + P5 action chunk + CBF | 最终候选 |

所有新训练使用 seeds `20260911/20260912/20260913`。每个模型 seed 与 M0 在相同的 episode index、episode seed、layout seed、motion metadata、初始条件和 transit reference 下运行，最终候选必须完成 `3 x 60 = 180` 个 development episodes。

### 6.3 两级执行

| 层级 | 规模 | 目的 | 准入条件 |
| --- | ---: | --- | --- |
| smoke | 每个新变体/seed 20 paired episodes | 发现接口、安全和极端效率问题 | 零扰动和测试已通过 |
| final development | 每个最终候选 seed 60 paired episodes | 估计三 seed 一致性与代价 | 对应 smoke 全部通过，配置已冻结 |

一旦 final development 开始，不再用其结果调参数。若观察到异常，只能停止、诊断并把对应版本判为失败；修复后的版本必须进入新的、明确版本化的开发循环。

### 6.4 必报指标

**安全和任务：**

- safe capture count/rate；
- collision、boundary、timeout count/rate；
- transit feasibility/success；
- 每个 failure 的 episode id、场景和根因分类。

**配对结果：**

- 对 M0 的 improved / degraded / tied；
- 每个 seed 的 paired delta，及三 seed mean +/- standard deviation；
- paired bootstrap confidence interval；对于二元捕获差异同时给出 McNemar 或 exact paired test，样本量不足时明确标注不确定性。

**效率与控制：**

- capture time、total defender path length；
- minimum obstacle/inter-agent clearance；
- mean/max CBF correction、intervention fraction；
- action-change rate、chunk smoothness、runtime fallback fraction；
- 平均控制时延、候选评估时延和 GPU/CPU device。

**预测到控制的对应：**

- target/clearance/visibility/risk 的 held-out 指标；
- action-following separation；
- ledger credit、local/global/OOD fallback 分解；
- high-CBF 和普通分桶的最终控制差异。

### 6.5 P6 决策规则

| 分类 | 证据要求 | 后续动作 |
| --- | --- | --- |
| `promising_development_candidate` | 三 seed 安全无新退化；平均 paired capture delta 为正，且至少 2/3 seed 非负；收益不只来自单一 seed；效率代价已量化且合理 | 进入 P7 审计 |
| `useful_safety_fallback_only` | 捕获改善不稳定，但 ledger/replay 在长尾分桶降低风险或退化，且无安全损害 | 保留为安全/诊断贡献，不声称捕获提升 |
| `prediction_improvement_no_control_gain` | 预测/校准改善，但三 seed 闭环 capture 与效率无净收益 | 记录负结果，不开 locked |
| `rejected_for_instability` | 任一配置出现无法解释的 collision/boundary、非确定性回归失败或严重效率退化 | 停止该候选，保留根因报告 |
| `insufficient_evidence_do_not_open_locked_test` | seed 之间冲突、样本不足或区间过宽 | 不开 locked；明确后续所需数据/实验 |

这里的“安全无新退化”不是一个 `95%` 捕获率阈值，而是 CBF 保护下逐 failure、逐 seed 的严格审计要求。

---

## 7. P7：可复现性审计与新 locked 前的决策

P7 不自动打开 locked test。它的唯一产物是一个独立审计，决定“是否值得另行预注册一个新的 locked evaluation”。

### 7.1 审计清单

- [ ] 每个最终 checkpoint、ledger、manifest、protocol 和 source 文件均有 SHA-256；
- [ ] 训练/评估命令能在 RTX 5050 环境中从空 results namespace 重放；
- [ ] 所有训练 run 有独立 TensorBoard log 和必需 text/scalar artifact；
- [ ] zero-perturbation regression、action contract、CBF order 和场景配对测试通过；
- [ ] P6 的所有 seed、所有 episode 和所有失败记录均可读取；
- [ ] 无 V4/V5 locked artifact 被改写；
- [ ] 计算开销、路径/时间代价、fallback 比例和失败模式已在报告中披露；
- [ ] 最终声明与证据等级一致，没有把 development 写成 locked improvement。

### 7.2 开新 locked 的最低条件

仅当以下条件同时成立，才可**另写一份新 preregistration** 请求是否打开新的 locked block：

1. P6 最终候选被分类为 `promising_development_candidate`；
2. 三 seed 结果具备可解释的一致性，而非只在一个 seed 或一个预选场景生效；
3. CBF 下没有新且未解释的安全失败；
4. 必要的效率代价没有超过 protocol 的报告阈值，且可被研究目标解释；
5. 独立重跑能复现主要趋势；
6. 新 locked protocol、seed block、停止规则、主/次指标和统计方法在运行前冻结。

任一条件不成立时，结论只能停留在 development，不因研究投入已多而降低证据标准。

---

## 8. 执行排期与检查点

以下是面向当前 RTX 5050 单机的保守工作安排；时间以完成的验收产物为准，不以日历日期替代质量门。

| 顺序 | 阶段 | 预计工作量 | 完成标志 |
| ---: | --- | --- | --- |
| 1 | P4 gate 与 hard-subset evaluator | 0.5--1 天 | replay-off/on 可比、测试通过、P4 report 完成 |
| 2 | P4 action-following 与 P4 提交 | 0.5 天 | 阶段判定、单一 Git commit、push 成功 |
| 3 | P5 protocol 和最小实现 | 1--2 天 | 代码测试、冻结 config、TensorBoard 预置 |
| 4 | P5 zero regression + smoke + 单 seed development | 1--2 天 | 安全/效率诊断完备、设计冻结或停止 |
| 5 | P6 三 seed 训练和 paired smoke | 2--4 天 | 所有候选 run 的 TensorBoard、20 episode gate |
| 6 | P6 60-episode final development 和统计 | 2--3 天 | `3 x 60` 配对结果、失败审计、最终分类 |
| 7 | P7 审计与报告 | 0.5--1 天 | locked readiness decision，不自动开锁 |

每次训练结束立即执行：检查 checkpoint/metadata/hash、保存 TensorBoard、运行关联测试、写入简短实验日志；不要等到多个阶段后再补 provenance。

---

## 9. 阶段完成定义

一个阶段只有同时满足以下条件才算完成：

1. 代码实现与相关测试通过；
2. 实验的输入、命令、输出、种子和哈希均已记录；
3. TensorBoard 包含该训练阶段的完整配置和过程；
4. 报告包含正向结果、负向结果、失败模式和结论边界；
5. 文档、代码和测试以该阶段为边界完成单独 Git commit，并成功 push；
6. 下一阶段的准入门通过，或明确记录“停止/回退”决定。

本计划优先建立可审计的证据链，而不是追逐单次最高捕获率。最终无论结论为正、负或证据不足，都应能回答：JEPA 在哪些观测/净空/CBF 工况下有帮助，作用机制是什么，安全和效率代价是什么，以及为何值得或不值得进入新的 locked test。
