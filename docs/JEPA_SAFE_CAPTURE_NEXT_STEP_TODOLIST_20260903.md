# 无人机集群安全增强围捕系统
# 下一步执行 TODO 计划：Interaction-Aware Action-Conditioned JEPA + Reliability Ledger + CBF

**版本：** 1.1
**日期：** 2026-09-03
**执行环境：** Windows + Conda uav-encirclement-gpu + NVIDIA RTX 5050
**实验性质：** development-only，未经单独授权不打开新的 locked test
**首要目标：** safe_capture 和安全不退化；mean capture time 仅作次要指标

> 本文件是从当前仓库状态开始执行的短路径计划。完整的系统合同和历史阶段记录见
> [JEPA_SAFE_CAPTURE_SYSTEM_MASTER_TODO_20260903.md](JEPA_SAFE_CAPTURE_SYSTEM_MASTER_TODO_20260903.md)。

---

## 1. 当前状态与未完成问题

### 1.1 已经具备的基础

- [x] P0：安全合同、信息边界、数据 split 和 locked_test_opened=false 规则。
- [x] P1：困难片段/反事实 archive 审计。
- [x] P2：三 seed action-conditioned JEPA 离线预测；四个 horizon 均优于 constant velocity。
- [x] P3：checkpoint-bound Reliability Ledger v2，运行时只读，支持 trusted、fallback_nominal、safe_hold。
- [x] P4：固定 K=5、3-step action chunk、第一步执行和候选排序接口。
- [x] P5：联合多机 CBF-QP、显式 infeasible/timeout、fallback ladder 和 deterministic audit。
- [x] P6-pre：paired evaluator、可达候选投影和 3-step anticipatory CBF 校准；三 seed smoke 尚未开始。

P5 开发审计已通过：13 个专项测试、三 seed audit、zero-perturbation exact、重复求解确定性和约 1.52--1.67 ms 的 p95 CBF 求解延迟。

P6 smoke 已完成全部 7 variants x 3 seeds x 20 episodes（420 个 paired episode）。过滤变体的 collision、boundary、pairwise 计数均为 0；A3 raw/no-CBF 为诊断路径，三个 seed 均出现 19/20 collision。P7 smoke aggregate 当前分类为 useful_safety_fallback_only：M3 相对 M0 的跨 seed paired delta 为 -5.0 个百分点，不能写成 JEPA 闭环提升。

### 1.2 不能从当前证据推出的结论

- P2 只证明离线预测，不证明闭环捕获提升。
- P3 只证明可以拒答和回退，不证明 QP feasibility 已校准。
- P4 只有接口和 synthetic evidence，不证明随机场景 safe-capture。
- 单 seed 的 V5 95% 不能替代三 seed paired result。
- 当前尚无 P6 三 seed 闭环 safe-capture 结论。

下一步唯一主问题是：在同一 episode、同一安全过滤器和同一冻结协议下，JEPA 作为候选轨迹评价器，叠加 ledger 与 CBF 后，是否能保持或提高 safe_capture，并让所有失败可解释、可回退、可复现？

### 1.3 当前执行阻塞：首个 M0 paired probe

paired evaluator 已经实现并通过 `py_compile`/CLI 检查，首个真实开发 probe 已运行：

- 输出目录：`results/p6_probe_m0_seed20260911`；
- `collision=0`、`boundary_violation=0`、`pairwise_violation=0`、`transit=100%`；
- 首个 episode 发生 `CBF controlled_abort`；
- `cbf_infeasible_steps=1`、`cbf_unverified_steps=1`；
- CBF p95 latency 约 `5.21 ms`；
- `safe_capture=0`。

这不是可以直接扩展到三 seed 的结果。它首先说明 actor 初始动作与当前联合 CBF-QP 的可行域/回退契约之间存在未诊断的接口问题。执行顺序必须固定为：

1. 读取该 episode 的 candidate、ledger、CBF 和 executed-action trace，定位不可行约束、请求动作、边界/间距余量及 fallback 原因；
2. 用确定性 replay 复现同一个首步，区分候选生成错误、坐标/单位错误、约束过紧、solver tolerance 和 fallback 验证错误；
3. 修复后重新跑同一 probe，并通过回归测试确认不可行/超时绝不执行 raw action；
4. 只有单 episode probe 通过后，才进入 20-episode smoke，再进入三 seed 全量。

在上述阻塞解除前，禁止宣称 V5 已完成、禁止聚合三 seed 主结果，也禁止以提高捕获率为目的放宽 CBF margin 或绕过安全层。

---

## 2. 不可变研究合同

### 2.1 Safe-capture 定义

一个 episode 只有同时满足以下条件才计为 Cooperative Safe Capture：

1. 至少一个 defender 进入目标 0.80 m capture radius；
2. 在 episode time limit 内发生；
3. 发生前没有 obstacle、defender-target 或 defender-defender collision；
4. 没有 defender 越过 3-D world boundary；
5. 没有因 CBF-QP infeasible、timeout、controlled abort 或未验证 fallback 而终止。

任何违反安全条件的“捕获”都记为 unsafe_capture=false，不能用 capture time 抵消。

### 2.2 信息边界

- 在线模块只能使用公开 observation、通信/动作历史、障碍几何、边界和时间戳/age。
- target ground truth 只能用于 offline settled-label 结算，不能进入 belief、JEPA、ledger、ranker 或 CBF 输入。
- train、calibration、development 和历史 locked 数据不可交叉。
- P6 全部运行保持 locked_test_opened=false。
- 不把 tmp 中恢复出的 archive 当作不存在的历史 warm-start checkpoint。

### 2.3 动作和安全边界

- 每周期生成固定 K=5 候选：nominal、intercept、lateral_clearance、formation_clearance、visibility_hold。
- 每个候选固定 3 个 control steps；只执行第一个控制步，然后重新观测和规划。
- JEPA、ledger 或 ranker 永远不能绕过 CBF-QP。
- 所有变体使用完全相同的 CBF margin、solver、tolerance、timeout、速度/加速度/slew 限制。
- CBF-QP 不可行或超时时，禁止执行未过滤的原始 action。
- CBF-QP 使用固定 `anticipatory_horizon_steps=3`，提前约束未来可达制动，避免在下一步才进入不可行状态。

---

## 3. 目标闭环架构和数据流

~~~text
公开观测/通信/动作历史
        |
        v
interaction-aware belief state
        |
        +--> 固定候选动作块生成器（K=5, horizon=3）
        |          |
        |          v
        |   action-conditioned JEPA
        |   target / clearance / TTC / visibility / risk heads
        |          |
        |          v
        |   Reliability Ledger v2
        |   trusted | fallback_nominal | safe_hold
        |          |
        |          v
        |   候选轨迹排序（只选第一步）
        |          |
        |          v
        |   Joint multi-agent CBF-QP
        |   obstacle + pairwise + boundary + kinematics
        |          |
        |          v
执行安全动作 -> 重新观测 -> 更新 belief -> 滚动 replan
~~~

### 3.1 每周期必须保留的状态

- observation/communication/action history 和 age/timestamp；
- defender positions/velocities、target belief、obstacle geometry、world bounds；
- 五个候选的 action chunk、JEPA 预测、uncertainty、clearance、visibility、TTC 和 score；
- ledger state、credit、bucket、fallback reason；
- CBF filtered action、solver status、active constraints、slack、correction norm、latency；
- episode/step seed、layout signature、target mode、observation schedule 和代码/checkpoint/ledger hash。

### 3.2 JEPA 头的职责

| Head | 用途 | 禁止事项 |
|---|---|---|
| target relative displacement/velocity | 预测候选动作下的追逃趋势 | 直接输出最终控制量 |
| obstacle clearance | 候选排序和困难片段诊断 | 替代真实障碍约束 |
| inter-agent clearance / pairwise TTC | 队形拥挤和碰撞风险排序 | 替代 pairwise CBF |
| target visibility / observation quality | 处理遮挡和 stale observation | 读取 hidden ground truth |
| CBF intervention risk | 估计候选会被安全层修改的概率 | 代替 CBF feasibility check |
| uncertainty / candidate disagreement | ledger credit 和拒答 | 用高置信度覆盖安全硬门 |

---

## 4. P6：三 seed paired closed-loop development

### 4.1 第一优先级：完成 paired evaluator 的校准、测试和闭环验收

已存在但尚未验收完成：

~~~text
scripts/evaluate_jepa_safe_capture_v2_paired.py
tests/test_jepa_safe_capture_v2_paired.py
scripts/aggregate_jepa_safe_capture_v2_paired.py
docs/JEPA_SAFE_CAPTURE_P6_THREE_SEED_PAIRED_DEVELOPMENT_20260903.md
~~~

实现/验收 TODO：

- [x] 从单一 EpisodeSpec 生成同一 episode 的 M0--M3/A1--A3 输入，确保 seed、初始状态、layout、target motion 和 observation schedule 完全相同。
- [x] 将 P4 candidate generator/ranker 接入真实滚动循环；不得继续使用 prediction-only evaluator 代替闭环。
- [x] 将每个候选和 nominal action 统一送入 JointCBFQPSafetyFilter；最终只能执行 filtered_action。
- [ ] 明确 trusted -> rank、fallback_nominal -> nominal、safe_hold -> safe hold 的运行路径并逐步记录。
- [ ] 处理 QP infeasible、timeout、solver_exception、nonfinite_request 和 stale/OOD observation；每类都要有可观测 fallback。
- [ ] 在 episode 结束时按 safe-capture 定义结算，不根据最后一个动作或距离阈值单独猜测成功。
- [x] 每步保存 candidate trace、ledger trace、CBF trace 和执行动作，支持 deterministic replay。
- [x] 启动时拒绝非空 output directory，写入 protocol/config/checkpoint/ledger/environment/git provenance。
- [x] 结果显式写入 development_only=true、locked_test_opened=false。
- [ ] 新增 `tests/test_jepa_safe_capture_v2_paired.py`，覆盖配对、CBF 全路径、zero-perturbation、safe-capture 结算和 provenance。
- [x] 修复并回归验证首个 M0/M3 probe 的 `controlled_abort` 根因：候选动作先投影到可达 slew envelope，CBF 加入 3-step anticipatory braking。

### 4.2 冻结评估矩阵

| ID | 执行栈 | 目的 |
|---|---|---|
| M0 | frozen V5 nominal + Joint CBF-QP | 新安全执行栈基线 |
| M1 | JEPA + nominal fallback + Joint CBF-QP | 区分 JEPA 评价信号的作用 |
| M2 | JEPA + Ledger v2 + Joint CBF-QP | 验证可信度门控和拒答 |
| M3 | JEPA + Ledger + candidate ranker + Joint CBF-QP | 最终安全增强系统 |
| A1 | M3 去除 ledger | 幻觉/漂移消融 |
| A2 | M3 去除 clearance/visibility heads | 安全辅助任务消融 |
| A3 | raw/no CBF | 仅故障诊断，不能进入安全主结论 |

固定输入：

- training seeds：20260911、20260912、20260913；
- 每个变体每个 seed 先做 20 个 paired smoke episodes；
- smoke 通过后每个变体每个 seed 做 60 个 paired episodes；M3 主结果为 3 x 60 = 180 episodes；
- 所有变体复用同一组 episode specifications，不能为某个变体另抽更容易的场景；
- 覆盖 nominal flee、S-curve/突变转向、延迟/噪声观测、3--5 个混合障碍、narrow channel、拥挤队形、左右起始侧和不同 communication delay。

### 4.3 Smoke 阶段验收

每个变体/seed 的 20 episodes 完成后自动检查：

- [ ] collision_count == 0；
- [ ] boundary_violation_count == 0；
- [ ] pairwise separation 无新增违规；
- [ ] 任意 QP infeasible/timeout 都有显式 fallback，绝不执行 raw request；
- [ ] 所有输出 finite，episode 配对关系完整；
- [ ] zero-perturbation 与 M0 的 filtered action 逐字段一致；
- [ ] high-credit 失败率不高于 low-credit；
- [ ] p95 总控制周期延迟 <= 100 ms；
- [ ] provenance 和 TensorBoard 字段完整。

任一硬门失败：停止该变体，保留完整 trace，先修复实现或将变体标记为 rejected，不继续扩大样本量。

### 4.4 P6 主指标和次指标

**主指标，按优先级排序：**

1. safe-capture count/rate；
2. collision、boundary、pairwise separation violation；
3. CBF-QP infeasible、timeout、controlled-abort；
4. fallback、safe-hold、stale/OOD 路由率；
5. transit success。

**次指标：** paired improved/degraded/tied、minimum clearance、CBF correction、candidate switch rate、visibility、各模块和总延迟、mean capture time、path length、显存占用。

mean capture time 不能单独否决一个安全且 safe-capture 不劣的方案；但如果时间变差导致 timeout、safe-hold 或实时性硬门失败，则按对应安全/任务失败处理。

### 4.5 P6 统计判定门

- **G1 Safety hard gate：** candidate collision=0、boundary=0、无新增 pairwise violation；所有 infeasible/timeout 显式 fallback；zero-perturbation 通过。
- **G2 Non-inferiority：** 三 seed 平均 paired safe-capture delta >= 0 个百分点，至少 2/3 seed 的 delta 非负。
- **G3 Positive development evidence：** G1 通过，平均 delta 严格为正，且至少 2/3 seed 非负；同时报告按 seed 的结果和预先固定的 bootstrap CI/McNemar exact test。
- **G4 Reliability：** high-credit failure rate 不高于 low-credit；OOD、stale、high-uncertainty 能触发 fallback。
- **G5 Realtime：** p95 总控制周期不超过 100 ms，timeout 有 nominal-CBF 或 safe-hold 路由。
- **G6 Provenance：** 每个 episode 可追溯到 seed、scene、checkpoint、ledger、代码 revision、配置和环境 hash。

捕获率低于历史单 seed 的 95% 不会自动判失败；安全硬门、配对公平性、可靠性和证据完整性优先。

---

## 5. P6 日志、文件和数据合同

### 5.1 推荐目录

每个 seed/variant 使用全新的目录，例如：

~~~text
results/jepa_safe_capture_v2_p6_paired_smoke_seed20260911_m3/
results/jepa_safe_capture_v2_p6_paired_seed20260911_m3/
results/jepa_safe_capture_v2_p6_paired_seed20260912_m3/
results/jepa_safe_capture_v2_p6_paired_seed20260913_m3/
results/jepa_safe_capture_v2_p6_tensorboard/seed20260911_m3/
~~~

禁止覆盖已有 results、checkpoint、TensorBoard、历史 V4/V5 报告或 tmp archive。

### 5.2 必须生成的文件

~~~text
protocol.json
effective_config.yaml
provenance.json
scene_manifest.jsonl
episodes.csv
summary.json
paired_comparison.json
step_traces/*.jsonl
candidate_traces/*.jsonl
cbf_traces/*.jsonl
~~~

每个 episode 至少包含：

~~~text
episode_seed, training_seed, variant, scene_id, layout_hash,
target_motion, observation_schedule, safe_capture, collision,
boundary_violation, pairwise_violation, transit_success,
cbf_infeasible, cbf_timeout, controlled_abort, fallback_count,
safe_hold_count, min_obstacle_clearance, min_pairwise_clearance,
selected_candidate, ledger_state, ledger_credit,
capture_time, total_latency_p95, checkpoint_sha256, ledger_sha256
~~~

TensorBoard 至少记录：Safety/*、CBF/*、Fallback/*、Reliability/*、Ranking/*、Visibility/*、Clearance/*、Latency/*、Episode/*、Provenance/*。

---

## 6. P6 测试 TODO

新增 tests/test_jepa_safe_capture_v2_paired.py，至少覆盖：

- [ ] 相同 EpisodeSpec 在所有变体间保持完全配对；
- [ ] M0 nominal anchor 和 M3 candidate 的 action chunk 长度、坐标轴和第一步执行语义正确；
- [ ] JEPA 永远只评价候选，不直接作为最终 action；
- [ ] ledger 三态路由和 stale/OOD fallback 可复现；
- [ ] 所有 candidate/nominal 都经过同一个 Joint CBF-QP；
- [ ] QP infeasible/timeout/non-finite 不会执行 raw request；
- [ ] zero-perturbation 的 M0/M3 filtered action exact；
- [ ] episode safe-capture 结算排除 unsafe capture；
- [ ] trace、hash、locked_test_opened=false 和 TensorBoard provenance 完整；
- [ ] 空 output directory、重复运行和 deterministic replay 行为正确。

当前专项与完整回归：`tests/test_jepa_safe_capture_v2_paired.py` 已加入，完整测试结果为 `270 passed`（17 个既有第三方警告，不影响通过）。

测试通过后再运行 smoke；禁止用手工临时脚本绕过 evaluator 的合同字段。

---

## 7. P7：聚合、失败复盘与 readiness audit

新增或完成：

~~~text
scripts/aggregate_jepa_safe_capture_v2_paired.py
docs/JEPA_SAFE_CAPTURE_P7_READINESS_AUDIT_20260903.md
~~~

### 7.1 聚合 TODO

- [ ] 按 variant、training seed、episode seed、motion mode、layout、visibility、observation age、ledger state 和 active CBF constraint 分桶。
- [ ] 输出逐 episode 表、每 seed 表、跨 seed 汇总和 paired delta。
- [ ] 对二元 safe-capture 使用配对计数；报告 improved/degraded/tied、McNemar exact 或等价 exact paired test。
- [ ] 报告 bootstrap CI 时固定重采样单位和随机 seed，不根据结果选择统计方法。
- [ ] 汇总 collision/boundary/pairwise/CBF failure 的绝对计数，不能只给百分比。
- [ ] 将每个失败 episode 关联到 candidate -> JEPA -> ledger -> rank -> CBF -> executed action 的因果链。
- [ ] 对高信用失败、连续 safe-hold、过度 candidate switching、CBF correction 过大和 visibility 丢失做专项诊断。
- [ ] 用 JSON、CSV 和 TensorBoard 三路交叉核对，防止只依赖单一汇总文件。

P7 smoke 聚合已生成：

~~~text
results/jepa_safe_capture_v2_p7_readiness_smoke_20260903/summary.json
results/jepa_safe_capture_v2_p7_readiness_smoke_20260903/report.md
results/jepa_safe_capture_v2_p7_readiness_smoke_20260903/paired_comparison.json
results/jepa_safe_capture_v2_p7_readiness_smoke_20260903/tensorboard/
~~~

该聚合已通过 canonical scene pairing、JSON/CSV/TensorBoard provenance 校验；60-episode full development 仍未开始。

### 7.2 结果分类

只能选择一个最终分类：

| 分类 | 条件 |
|---|---|
| positive_development_evidence | G1--G6 通过，三 seed 有可重复正向 paired safe-capture 证据 |
| safety_preserving_non_inferiority | G1--G6 通过，捕获不劣但没有稳定正向提升 |
| useful_safety_fallback_only | 捕获没有提升，但长尾风险/漂移/回退质量改善 |
| prediction_improvement_no_control_gain | 离线预测改善，闭环没有净收益 |
| insufficient_evidence_or_reject | 任一安全、可靠性、配对、公平性、实时性或 provenance 门失败 |

只有前两类且复盘资料完整时，才可以起草新的 preregistration；是否打开新的 locked test 仍需用户明确授权。

---

## 8. RTX 5050 执行顺序和命令模板

### 8.1 环境核验

~~~powershell
Set-Location D:\uav-capture\uav_capture
conda activate uav-encirclement-gpu
$env:PYTHONPATH = "$PWD\src;$PWD\scripts"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
python -m pytest -q tests/test_cbf_qp.py tests/test_jepa_safe_capture_v2_paired.py
~~~

如果 paired evaluator 或测试文件尚未实现，先完成代码和单测，不能调用 prediction-only evaluator 冒充闭环结果。

### 8.2 Smoke

~~~powershell
python scripts/evaluate_jepa_safe_capture_v2_paired.py --help
python scripts/evaluate_jepa_safe_capture_v2_paired.py --variant m0 --training-seed 20260911 --episodes 20 --split development --output-dir results/jepa_safe_capture_v2_p6_paired_smoke_seed20260911_m0 --tensorboard-dir results/jepa_safe_capture_v2_p6_tensorboard/seed20260911_m0 --device cuda --development-only
~~~

对 m0,m1,m2,m3,a1,a2,a3 和三个 training seed 重复。实际参数必须以脚本 --help 和冻结 protocol 为准。

### 8.3 Smoke 通过后的全量

~~~powershell
python scripts/evaluate_jepa_safe_capture_v2_paired.py --variant m3 --training-seed 20260911 --episodes 60 --split development --output-dir results/jepa_safe_capture_v2_p6_paired_seed20260911_m3 --tensorboard-dir results/jepa_safe_capture_v2_p6_tensorboard/seed20260911_m3 --device cuda --development-only
~~~

依次完成 M0--M3，再运行 A1--A3；每次都用新目录，不修改任何已存在结果。

### 8.4 聚合和审计

~~~powershell
python scripts/aggregate_jepa_safe_capture_v2_paired.py --input-root results --output-dir results/jepa_safe_capture_v2_p7_readiness_20260903 --development-only
~~~

---

## 9. 停止、回退和修复规则

- 新 collision、boundary 或 pairwise violation：立即停止变体，保留 trace，回退到 nominal + CBF。
- CBF infeasible/timeout 没有显式 safe-hold 或 nominal-CBF：该变体直接拒绝。
- zero-perturbation 不一致：停止实验，先修复 candidate/CBF 接口。
- high-credit 比 low-credit 更危险：冻结 ledger，回退 frozen nominal + CBF。
- checkpoint、ledger、protocol、scene 或 environment hash 不一致：停止并重建 provenance。
- p95 总延迟超过 100 ms：保留 watchdog/fallback 证据，先做工程修复，不能隐藏 timeout 或放宽安全约束。
- smoke 通过后不再基于捕获率调整权重、阈值、chunk、solver 或 episode seed。
- 不得用 ground truth 泄漏、回灌 development 数据、放宽 CBF margin 或绕过 fallback 换取更高捕获率。

---

## 10. 时间盒和提交边界

| 时间盒 | 工作 | 完成标志 |
|---|---|---|
| T0 | paired evaluator 设计、接口接线和 protocol freeze | schema、variant、seed、scene manifest 固定 |
| T1 | paired evaluator + unit tests | 所有动作经过 Joint CBF-QP，异常路径可测 |
| T2 | 每变体/seed 20 smoke | G1、G4、G5、G6 通过 |
| T3 | M0--M3 三 seed 全量 | M3 有 180 个可复盘 episode |
| T4 | A1--A3 消融 | 消融不改变安全合同，失败可解释 |
| T5 | P7 聚合和 readiness audit | 输出唯一结果分类和 CI |
| T6 | 可选 SIL/HIL 计划 | 仅在 P7 通过后进入，不转化为实飞结论 |

每阶段只提交对应代码、测试、报告和小型 manifest：

- P6：paired evaluator、aggregate、测试、P6 报告；
- P7：统计审计脚本、readiness 报告、README 索引；
- 不提交 tmp/、NPZ archive、TensorBoard event、大型 trajectory 或用户已有 E1/V5 修改；
- 提交前运行 git diff --check、相关 pytest 和 git status --short；
- 普通 git push origin main，不使用 force push。

---

## 11. 最终完成定义（Definition of Done）

本阶段完成必须同时满足：

1. P6 paired evaluator、P6 三 seed development 和 P7 readiness audit 都有独立产物；
2. JEPA 只做候选轨迹评价，最终动作 100% 经过 Joint CBF-QP；
3. infeasible、timeout、OOD、stale 和 non-finite 都有显式 fallback 且可审计；
4. safe-capture、collision、boundary、pairwise separation、CBF failure 和 fallback 均按 episode 统计；
5. 三 seed、同一 episode 配对、scene/hash/provenance 完整，结果可 deterministic replay；
6. TensorBoard 与 JSON/CSV 相互一致；
7. 结果被诚实归类为正向证据、安全不劣、仅回退收益、预测无控制收益或证据不足；
8. 没有用户明确授权前，新的 locked test 保持关闭；
9. mean capture time 只作为次要诊断，不掩盖任何安全硬门失败。

**研究成功标准：** 不是某一个 seed 的最高捕获率，而是证明“JEPA 反事实评价 + reliability ledger 拒答 + CBF 硬安全层 + 滚动时域 replan”能够在同一冻结协议下形成一条安全不静默失效、可解释、可复现的无人机集群围捕闭环。
