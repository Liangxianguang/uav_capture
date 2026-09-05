# 无人机集群对抗围捕安全增强系统
# V21 之后详细 TODO 与目标计划书

**版本：** 2026-09-05 / v21-continuation-2  
**执行目录：** `D:\\uav-capture\\uav_capture`  
**硬件目标：** NVIDIA GeForce RTX 5050  
**当前运行时：** `D:\\download\\anaconda3\\envs\\traj_pred_prep\\python.exe`，Torch `2.9.1+cu130`  
**实验边界：** `development_only=true`，`locked_test_opened=false`  
**主指标：** episode-level `safe_capture`  
**诊断指标：** `mean_capture_time`、Transit、最小净空、CBF 修正量、fallback/abort、候选切换率和控制周期延迟

> 本文件是从当前 V21 证据继续执行的计划，不是新的实验结果。绝对 `95%` 不是硬目标。任何捕获时间、路径长度或 CBF 修正量的改善，都不能抵消 `safe_capture` 下降或安全合同违规。

## 0. 当前执行状态（以此覆盖历史顺序）

- **S1 已完成：** 三 seed settled ranking/candidate-separation 聚合已提交为 `9d8933a`；source/hash gates 通过，但 ranking gate 未通过，当前标签为 `ranking_unresolved`。
- **S2 正在进行：** `scripts/index_jepa_safe_capture_v21_failures.py` 与针对性测试已加入工作区，下一动作是先完成测试、修正暴露问题，再对 12 个 V21 smoke run 建立只读 failure index。
- **当前禁止：** 在 S2 归因完成前，不训练新 JEPA、不修改 CBF margin/OOD/stale 阈值、不扩大 40/60 集、不打开 locked test。
- **主验收原则：** `safe_capture` 是唯一主指标；mean capture time、Transit、预测误差和 CBF correction 只用于解释，不能抵消安全捕获下降。

### S2 的立即命令

```powershell
$py = 'D:\\download\\anaconda3\\envs\\traj_pred_prep\\python.exe'
$env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"
& $py -m pytest -q tests/test_index_jepa_safe_capture_v21_failures.py

& $py scripts/run_with_tensorboard_compat.py `
  scripts/index_jepa_safe_capture_v21_failures.py `
  --input-root results `
  --settled-root results `
  --output-dir results/jepa_safe_capture_v21_failure_index `
  --tensorboard-logdir results/jepa_safe_capture_v21_current_tensorboard/failure_index `
  --development-only
```

### S2 通过条件

1. 12 个 run、每个 20 个 episode、三 seed 配对和 settled-row coverage 全部通过；
2. `collision`、`boundary_violation`、`pairwise_violation`、`raw_unverified_executed` 均为零，CBF abort/fallback 有完整 trace；
3. `failure_index.csv/json`、`report.md`、`provenance.json` 与 TensorBoard 的计数和 hash 一致；
4. 每个困难 bucket 至少有一个可重放样例；无法证明因果的样例标为 `unresolved`，不凭标签直接训练。

S2 失败时保留产物并修复索引/数据契约；不得用删行、改分母或放宽安全门“通过”。

## 1. 最终目标与系统边界

### 1.1 目标

构建一条面向无人机集群对抗围捕的可审计安全闭环：

```text
多机观测/通信历史
  -> interaction-aware BeliefState
  -> 传统规划器生成 K=5 个动力学可行候选 action chunks
  -> action-conditioned JEPA 反事实轨迹评价
  -> immutable Reliability Ledger 可信度校验/拒答
  -> safety-first ranker + nominal anchor + abstention/hysteresis
  -> Joint CBF-QP 统一安全过滤
  -> 只执行 action chunk 第 1 步
  -> 重新观测、重规划、重过滤
```

### 1.2 不可改变的责任边界

- JEPA 只能评价候选轨迹，不能生成、覆盖或直接执行最终控制动作。
- Reliability Ledger 在 calibration 后只读；不得在线更新信用或阈值。
- `nominal`、JEPA 选择候选、`safe-hold` 和 nominal fallback 都必须进入同一个 Joint CBF-QP。
- CBF 返回 `verified=true` 且 action finite 后，执行器才可以执行第一控制步。
- CBF 不可行、超时、non-finite、OOD、stale 或 provenance 不一致时，依次执行 `safe-hold -> nominal-CBF -> controlled_abort`；禁止 raw desired action。
- `controlled_abort` 是安全失败，必须保留在 episode 分母内，不能改写为 timeout 或删除。
- 在线控制不得读取 target ground truth；target truth 只用于离线 settled label 和 episode 结算。

### 1.3 `safe_capture` 定义

一个 episode 仅在以下条件同时满足时计为 `safe_capture=true`：

1. time limit 内至少一架 defender 进入 `0.80 m` capture radius；
2. 无 obstacle、target 或 defender-defender collision；
3. 无 boundary/altitude violation；
4. 无 pairwise separation violation；
5. 无 CBF infeasible、timeout、unverified action 或 `controlled_abort` 终止。

## 2. 当前基线：已完成与未完成

### 2.1 已冻结的 V21 输入

| 项目 | 当前值 |
|---|---|
| V21 protocol | `configs/central_random_mixed_obstacle_s3_v5_v21_cpu_separation_gate_development_protocol.yaml` |
| protocol SHA-256 | `278623ceb7185a6c3ce23246e8a28693f025a2977fad95059ae5b0df9a03b014` |
| training seeds | `20260911`, `20260912`, `20260913` |
| candidate 数量 | 5；`nominal/intercept/lateral_clearance/formation_clearance/visibility_hold` |
| action chunk | 3 control steps；只执行第 1 步后重规划 |
| predicted clearance floor | `0.15 m` |
| CBF margin | 保持现值，不得为追求捕获率下调 |
| locked 状态 | `locked_test_opened=false` |

### 2.2 已有证据

- 三 seed checkpoint-bound reliability ledger 已生成；checkpoint、protocol、calibration hash binding、tampered provenance rejection、immutable ledger、OOD/stale/non-finite/unknown-horizon/uncertainty/TTC-CBF fault matrix 均通过。
- V21 ledger 审计的 10 个核心测试已通过；TensorBoard 已写入 30 个 scalar tag 和 5 个 text tag。
- M3 CPU rolling replay repeat1/repeat2 均为 20 episodes、1189 control cycles、`safe_capture=12/20=60%`，collision/boundary/pairwise 均为 0，CBF infeasible/controlled abort 为 8，raw unverified 为 0；两次 trace、candidate ranking、CBF 和执行结果逐字段一致。
- zero-perturbation m0/m3 各 20 episodes，场景 geometry 一致，96 个非 JEPA 物理字段无差异；这是执行合同回归，不是 JEPA 收益证明。

### 2.3 尚未完成的关键事项

- rolling-horizon audit 尚未正式生成报告；必须确认 `100-cycle` coverage、`500-cycle hard-context` 语义、trace 完整性、fallback 和 latency。
- 尚未完成真正的 CUDA M3 replay；已有 CPU replay 不能称为 CPU/CUDA 等价验证。
- 尚未完成实际 CPU/CUDA comparator，对 candidate decision、CBF status、executed action、termination 和 safety settlement 做逐字段/量化后比较。
- 尚未运行 V21 之后三 seed 的 M0/M3/A1/A2 paired smoke，因此不能把当前单 seed 60% 写成跨 seed 结论。
- 尚未获得 safe-capture 的正式提升证据；后续若结果不提升，必须如实归档 `prediction_signal_no_control_gain` 或 `useful_safety_fallback_only`。

## 3. 总执行顺序与闸门

```text
P0 证据与环境冻结
  -> P1 rolling-horizon / Joint CBF audit
  -> P2 真 CUDA M3 replay
  -> P3 CPU/CUDA decision comparator
  -> P4 三 seed x M0/M3/A1/A2 x 20 集 paired smoke
  -> P5 settled ranking / failure index / aggregate
  -> P6 条件性三 seed x 40/60 集 paired development
  -> P7 多任务安全头与困难片段 replay（仅在 P6 无控制收益时）
  -> P8 robustness + SIL/HIL readiness
  -> P9 locked-test preregistration / 论文证据归档
```

上游闸门未通过时，不扩大 episode 数、不训练新模型、不修改 CBF margin、不打开 locked test。

## 4. P0：证据、环境和协议冻结

**目标：** 防止历史 V4/V5、`tmp/`、不同 seed 或 dirty worktree 污染 V21 比较。

### TODO

- [ ] 保存 `git rev-parse HEAD`、`git status --short`、代码 diff 摘要；保留用户已有改动，尤其不删除 `tmp/`，不执行 `git reset`、`checkout` 或 `git clean`。
- [ ] 保存 RTX 5050、driver、CUDA、PyTorch、NumPy、TensorBoard、Conda 包清单和 Python 路径。
- [ ] 保存 protocol、environment config、actor checkpoint、JEPA checkpoint、ledger、calibration archive、scene manifest 的 SHA-256。
- [ ] 为每个新运行建立空的 `results/` 和 TensorBoard 目录；脚本必须拒绝覆盖非空目录。
- [ ] 运行 `git diff --check`、V21 targeted tests、协议 schema 检查和 CUDA 可用性检查。
- [ ] 将命令行、环境变量 `PYTHONPATH`、设备、代码 revision 和 locked 状态写入 `preflight.json`。
- [ ] 明确当前 `pursuit_env.py` 用户本地动力学草稿不属于本轮修改范围。

### 出口门

- [ ] 所有输入 hash 在 `preflight.json` 和运行 metadata 中一致。
- [ ] 运行器拒绝 `locked_test` split 和 online target truth。
- [ ] 任何 provenance mismatch 都会在 episode 开始前停止。

## 5. P1：rolling-horizon 与 Joint CBF 长序列审计

**目标：** 证明系统每周期只执行第一步、持续重规划，且长序列中没有 raw action 绕过 CBF。

### TODO

- [ ] 使用 `scripts/audit_jepa_safe_capture_v21_rolling_horizon.py` 审计 CPU repeat1/repeat2。
- [ ] 设置 `--minimum-cycles 100`、`--hard-context-cycles 500`；单独检查“总 cycles 达到 500”与“hard-context 片段真正覆盖 500”不是同一件事。
- [ ] 检查每条 trace 是否具备 `candidate_ranking`、`cbf`、`executed_action`、`raw_unverified_executed`、完整 latency 字段。
- [ ] 检查 `requested_action != executed_action` 的 unverified 情况、CBF failure fallback、controlled abort、timeout 和 non-finite 路由。
- [ ] 汇总 actor、candidate generation、JEPA、ledger、ranker、CBF、environment 和 cycle-total 的 p50/p95/p99/max latency。
- [ ] 输出 JSON、Markdown、hash manifest 和独立 TensorBoard；不覆盖已有 WP4 目录。

### 出口门

- [ ] 两次 replay 均满足 100-cycle coverage；至少一个 run 满足真实 500-cycle hard-context coverage。
- [ ] episode/trace/control-cycle 数量一致，first-step-replan contract 全部为真。
- [ ] collision/boundary/pairwise 为 0，CBF failure 都有明确 fallback，`raw_unverified_executed=0`。
- [ ] cycle p95 不超过协议上限 100 ms。
- [ ] 未通过时标记 `BLOCKED_BY_ROLLING_OR_SAFETY_AUDIT`，不能进入 P2 之后的 smoke。

## 6. P2：RTX 5050 真 CUDA M3 replay

**目标：** 在与 CPU replay 完全相同的 V21 manifest、checkpoint、ledger 和 protocol 下，获得设备真实结果。

### 固定输入

- protocol：V21 CPU separation-gate protocol。
- scene manifest：`results/jepa_safe_capture_v21_wp4_replay_m3_cpu_seed20260911_repeat1/scene_manifest.jsonl`。
- seed：先 `20260911`，通过后再扩展到其他 seed。
- variant：`m3`。
- perturbation：`0.1 m/s`。
- device：`cuda`。

### 执行命令

```powershell
Set-Location D:\\uav-capture\\uav_capture
$py = 'D:\\download\\anaconda3\\envs\\traj_pred_prep\\python.exe'
$env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"
& $py scripts/run_with_tensorboard_compat.py scripts/evaluate_jepa_safe_capture_v2_paired.py `
  --variant m3 --training-seed 20260911 --episodes 20 --split validation `
  --protocol configs/central_random_mixed_obstacle_s3_v5_v21_cpu_separation_gate_development_protocol.yaml `
  --environment-config configs/capture_radius_pursuit_central_v4_flee.yaml `
  --actor-checkpoint models/v5_development_exact_reactive_seed661606.pt `
  --jepa-checkpoint results/jepa_safe_capture_v11_hard_replay_seed20260911/checkpoint.pt `
  --reliability-ledger results/jepa_safe_capture_v21_ledger_seed20260911/reliability_ledger.json `
  --scene-manifest results/jepa_safe_capture_v21_wp4_replay_m3_cpu_seed20260911_repeat1/scene_manifest.jsonl `
  --output-dir results/jepa_safe_capture_v21_wp4_replay_m3_cuda_seed20260911 `
  --tensorboard-dir results/jepa_safe_capture_v21_tensorboard/wp4_replay_m3_cuda_seed20260911 `
  --jepa-perturbation-mps 0.1 --device cuda --development-only
```

### TODO

- [ ] 运行前确认 GPU 名称为 RTX 5050、CUDA 可用且 device metadata 为 `cuda`。
- [ ] 运行后检查 `summary.json`、`provenance.json`、`episodes.csv`、`step_traces/`、TensorBoard 和 input hash manifest。
- [ ] 不将 CUDA 结果与 CPU 结果直接合并为任务率；先进入 P3 comparator。

### 出口门

- [ ] CUDA run 的 safety hard gates、fallback 语义、raw-unverified 和 latency 全通过。
- [ ] 若 CUDA 出现非 finite、solver failure 未 fallback、或 raw action，立即停止该设备实验。

## 7. P3：CPU/CUDA 决策与安全结算 comparator

**目标：** 区分“数值/设备差异”和“真正策略差异”，禁止仅凭 safe-capture 最终数字宣称设备等价。

### 比较字段

- candidate eligibility、eligibility reason、candidate scores、selected candidate、abstention、hysteresis/minimum hold；
- ledger state、reason code、fallback mode、uncertainty/OOD/stale flags；
- requested action、CBF verified status、solver status、fallback action、executed action；
- termination reason、capture event、collision/boundary/pairwise、controlled abort、timeout；
- scene geometry、episode seed、cycle index 和 trace row 数。

### 比较规则

- 离散字段要求完全相同。
- score 和连续 action 先按 protocol 的 fixed-point quantum 比较；不能把 wall-clock latency 纳入策略等价。
- 若设备导致 candidate/ranker drift，记录最早发生 drift 的 cycle 和输入 provenance；不通过时停止扩大实验。
- 安全结算必须单独比较；不能用“最终都没有碰撞”掩盖 candidate decision drift。

### 出口门

- [ ] candidate decision、ledger route、CBF status、executed action、termination 和 safety settlement 在比较规则下等价。
- [ ] `raw_unverified_executed=0` 且所有 CBF failure 均有同一 fallback 语义。
- [ ] 输出 `device_replay_comparison.json`、Markdown、TensorBoard 和 hash manifest。

## 8. P4：三 seed paired smoke

**目标：** 在新 V21 合同下先获得小规模、同场景、同 episode seed 的跨 seed 开发证据。

### 实验设计

- training seeds：`20260911/20260912/20260913`。
- 每 seed 先运行 M0 生成唯一 scene manifest；M3、A1、A2 逐 episode 复用该 manifest。
- 每变体每 seed 20 episodes；所有输出使用独立目录。
- M0：nominal planner + CBF baseline。
- M3：JEPA + ledger + auxiliary score + CBF。
- A1：去 ledger，诊断信用路由。
- A2：去 clearance/visibility ranking，诊断安全辅助头。
- A3：无 CBF 只作故障诊断，不能进入安全结论。

### 必须统计

- 主指标：每 seed 和 aggregate 的 episode-level `safe_capture`。
- paired：delta、improved/degraded/tied、exact McNemar 或等价配对检验。
- 安全：collision、boundary、pairwise、CBF infeasible、CBF timeout、controlled abort、fallback、raw-unverified。
- 诊断：Transit、capture time、minimum clearance、CBF correction norm、candidate switch/oscillation、abstention、selected-not-best、latency。
- 证据：checkpoint/ledger/protocol/manifest hash、TensorBoard tag 完整性和 trace 可重放性。

### 出口门

- [ ] 所有安全保留变体 collision/boundary/pairwise/raw-unverified 为 0。
- [ ] M3 的 CBF failure 都有 fallback；controlled abort 保留在分母。
- [ ] 若 M3 aggregate safe-capture 低于 M0，或 3 seed 中仅 0/1 个非负，结论为 `prediction_signal_no_control_gain`，不能扩展样本，转入 P5/P7 诊断。
- [ ] 若安全门通过且 M3 aggregate 不低于 M0、至少 2/3 seed 非负，才可申请 P6 40/60 集 paired development；这不是 95% 目标。

## 9. P5：settled ranking、failure index 与 reliability 诊断

**目标：** 找出 JEPA 预测、ledger 路由和最终安全捕获之间的失配，不通过调低安全阈值掩盖问题。

### TODO

- [ ] 为每个 decision 保存五候选的 offline settled progress、capture、clearance、visibility、CBF correction、feasibility 和 safety outcome。
- [ ] 计算 selected-not-best、top-1 safety precision/recall、Spearman/Kendall、candidate separation、abstention 和切换/振荡率。
- [ ] 按 ledger credit、visibility、observation age、TTC、CBF correction、target turn/acceleration 和 obstacle density 分桶。
- [ ] 单独列出 high-credit failure、low-credit failure、fallback、stale/OOD、timeout、controlled abort 和 rank mismatch。
- [ ] 做双次 deterministic replay；每个失败片段保留原始 trace 和 hash，不直接回灌旧训练 archive。
- [ ] 若修改 score、margin、hysteresis、minimum hold 或 candidate separation，必须创建新 protocol、calibration manifest、ledger 和结果目录。

### 诊断决策

| 现象 | 处理 |
|---|---|
| high-credit settled failure 高于 low-credit | ledger gate 失败，回到校准；不调 task score |
| selected-not-best 高、rank correlation 为负 | 冻结当前结果为 ranking_unresolved，先修排序/标签 |
| CBF 修正量下降但 safe-capture 不升 | 只记为执行诊断变化，不宣称任务收益 |
| 只有 controlled abort 多而碰撞为 0 | 保留安全结论，分析候选可行性/CBF 几何，不删除 abort |
| OOD/stale/non-finite 未进入 safe-hold | 立即停止，标记 `BLOCKED_BY_SAFETY` |

## 10. P6：条件性 40/60 集 paired development

**前置条件：** P1-P5 的安全、证据和决策门全部通过。

### TODO

- [ ] 新建独立 development manifest；不得把多个 20 集结果拼成伪 40/60 集。
- [ ] 固定 checkpoint、ledger、protocol、score、CBF margin、chunk length、episode seed 和 abort semantics。
- [ ] 至少覆盖 nominal/delayed/noisy、flee persistence/S-curve、target turn/acceleration、3--5 obstacles、拥挤队形和不同 initial side distance。
- [ ] 每 seed 先完成 40 episodes；若资源允许，再追加独立 60-episode block，仍需单独 manifest。
- [ ] 先汇总每 seed，再汇总 aggregate；报告 sample SD、paired CI、improved/degraded/tied 和完整安全失败矩阵。
- [ ] 主结论只使用 `safe_capture`；`mean_capture_time` 作为诊断表，不参与安全成功定义。

### 结果标签

- `safe_capture_improvement_candidate`
- `safe_capture_noninferior`
- `prediction_signal_no_control_gain`
- `useful_safety_fallback_only`
- `rejected_for_safety`
- `insufficient_evidence_do_not_open_locked_test`

## 11. P7：多任务安全头与困难片段 replay（条件分支）

**触发条件：** P4/P6 安全合同通过，但 M3 没有稳定 safe-capture 控制收益，或 P5 显示预测安全信号不足。

### 11.1 输入和表示

- defender-target、defender-defender 相对位置/速度、队形几何和 pairwise TTC；
- target belief 的 visibility、observation age、message age、历史动作和运动模式 embedding；
- obstacle/boundary 局部几何、预测净空、上周期 CBF 状态和通信 mask；
- permutation-invariant 或受控 agent-id 编码，避免集群排序改变导致 latent 漂移。

### 11.2 输出头

对每个候选 action chunk 和多个 horizon 输出：

```text
target displacement / velocity / acceleration
obstacle-clearance lower quantile
inter-agent clearance lower quantile / pairwise TTC
visibility probability / observation-age risk
CBF intervention probability / correction magnitude / QP feasibility
predictive uncertainty / candidate disagreement
```

这些量只用于候选排序、拒答和诊断；真实安全证明仍由 Joint CBF-QP 完成。

### 11.3 数据、训练和校准 TODO

- [ ] 只从 train split 建立新 counterfactual archive；development settled outcome 不原样回灌旧 archive。
- [ ] 困难片段覆盖急转、速度突变、遮挡/延迟、候选分离消失、CBF correction 变大、预测漂移和 controlled abort。
- [ ] 固定 hard-context 权重上限、采样比例、去重策略和 split 隔离；三 seed 独立训练，不共享 optimizer state。
- [ ] 记录每个预测头的 loss、MAE/Brier/ECE/AUROC、finite rate、coverage 和 uncertainty calibration。
- [ ] 先通过 prediction/calibration gate，再生成 checkpoint-bound ledger；不得跳过校准直接进入闭环。
- [ ] 对新 checkpoint 重复 P2-P6；旧 checkpoint 只读，旧结果保留为对照。

### 预测 gate

- [ ] 全部输出 finite。
- [ ] 至少一个主要 horizon 不劣于 constant-velocity baseline。
- [ ] clearance/TTC/visibility/CBF 风险头标签非空、不过度乐观且可校准。
- [ ] 明确写出“prediction gate 通过不等于 safe-capture 提升”。

## 12. P8：鲁棒性、SIL/HIL 和部署准备

**前置条件：** P6 或 P7 得到可复现的安全非劣/提升候选。

### Stress matrix

- [ ] detection dropout、observation noise、message delay/dropout、stale observation；
- [ ] target turn、突发加速度、flee persistence/S-curve；
- [ ] obstacle density、狭窄通道、队形拥挤和 initial side distance shift；
- [ ] 单机通信/执行失效、JEPA non-finite、ledger hash mismatch、GPU 不可用、进程重启；
- [ ] 每个条件均检查 safe-hold、nominal-CBF、controlled-abort 路由，不执行 raw。

### SIL/HIL TODO

- [ ] 固定 timestamp、通信年龄、执行回执、solver status、watchdog 和 abort 接口 schema。
- [ ] HIL 控制指令仍必须经过 CBF；HIL 通过不等于真实飞行许可。
- [ ] 记录端到端 cycle p50/p95/p99、丢帧、队列年龄和重启后的 provenance 恢复。
- [ ] 形成 failure injection matrix 和恢复时间报告。

## 13. P9：发布、论文和 locked-test 决策

### 只有以下条件全部满足，才允许考虑 locked test

- [ ] 三 seed development 结果、输入 hash、代码 revision、环境和命令完整可复现。
- [ ] safe-capture 主结论在 paired design 下成立；或明确归档为无控制收益，不包装成提升。
- [ ] collision/boundary/pairwise/raw-unverified 硬门全通过；CBF timeout/infeasible/controlled abort 全部公开。
- [ ] CPU/CUDA comparator 和 RTX 5050 最终任务运行结果分开报告。
- [ ] stress、SIL/HIL、ledger fault、rolling-horizon 和 zero-perturbation 审计全部有独立报告。
- [ ] 新 locked protocol、locked manifest、checkpoint、ledger 和 preregistration 已冻结；此前 development 结果不可回写 locked split。

否则最终标签为 `insufficient_evidence_do_not_open_locked_test`，继续保留所有正负结果。

## 14. 统一运行前后清单

### 运行前

- [ ] `development_only=true`、`locked_test_opened=false`、split 不是 locked。
- [ ] output/TensorBoard 目录为空；脚本拒绝覆盖。
- [ ] protocol/checkpoint/ledger/calibration/manifest/environment hash 已保存。
- [ ] RTX 5050、CUDA、PyTorch 和 Conda 环境已记录。
- [ ] targeted tests、schema tests、`git diff --check` 通过。

### 运行中

- [ ] 每周期只执行 Joint CBF-QP 返回的 finite + verified 第一控制步。
- [ ] 不读取 online target truth，不在线更新 ledger。
- [ ] 记录 candidate、eligibility、JEPA、ledger、rank、CBF、fallback、latency 和 provenance。
- [ ] 不删除失败 episode、controlled abort、non-finite trace 或 timeout。

### 运行后

- [ ] `summary.json`、`episodes.csv`、`step_traces/`、`provenance.json`、hash manifest、命令和 TensorBoard 齐全。
- [ ] JSON/CSV/Markdown/TensorBoard 的 episode 数、安全计数和 hash 双向一致。
- [ ] 运行 rolling audit、device comparator、failure index、settled ranking、ledger alignment、zero-perturbation 和 aggregate。
- [ ] 生成该阶段独立 Markdown 报告，并记录通过/失败门和下一步标签。

## 15. 结果目录和提交边界

新结果使用独立前缀，禁止覆盖历史目录：

```text
results/jepa_safe_capture_v21_rolling_audit_*
results/jepa_safe_capture_v21_wp4_replay_m3_cuda_seed<seed>/
results/jepa_safe_capture_v21_device_comparator_*/
results/jepa_safe_capture_v21_smoke_<variant>_seed<seed>/
results/jepa_safe_capture_v21_smoke_aggregate/
results/jepa_safe_capture_v21_settled_*/
results/jepa_safe_capture_v21_stress_*/
results/jepa_safe_capture_v21_tensorboard/<stage>/
```

每个阶段只选择性提交该阶段文件，禁止 `git add .`。建议提交顺序：

```text
test(jepa): verify v21 rolling horizon contract
test(jepa): verify v21 cuda decision equivalence
exp(jepa): run v21 three-seed paired smoke
docs(jepa): archive v21 smoke and settled ranking
train(jepa): add calibrated safety heads and hard replay
test(jepa): audit v21 robustness and sil hil contracts
```

## 16. Definition of Done

- [ ] JEPA 是 interaction-aware、action-conditioned 候选轨迹评价器，不是动作生成器。
- [ ] Reliability Ledger 对 provenance、OOD、stale、non-finite、低信用和 uncertainty 风险可拒答且可重放。
- [ ] 所有 desired action、fallback 和 safe-hold 均经过同一 Joint CBF-QP。
- [ ] rolling horizon 只执行第一步；100/500-cycle replay 和双设备 comparator 通过。
- [ ] 三 seed paired development 以 episode-level `safe_capture` 为唯一主指标。
- [ ] 所有 collision、boundary、pairwise、CBF failure、controlled abort、fallback、raw-unverified 和 latency 均公开。
- [ ] 结果带完整 code/environment/input hash；locked test 在证据充分前保持关闭。

**下一次实际动作：** 先完成上方 S2 failure-index 测试和真实索引；随后对 `candidate_capture_regression`、`high_credit_failure`、`cbf_controlled_abort`、`stale_observation`、`candidate_oscillation` 和 `clearance_prediction_gap` 进行双次 deterministic hard replay。只有 S2 归因报告完成后，才按 P1-P9 的条件闸门决定是否修复排序、训练新安全头或进入 paired smoke；在此之前不扩大样本、不降低安全阈值、不打开 locked test。
