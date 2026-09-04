# 无人机集群安全围捕系统：下一阶段 TODO 计划书

**版本：** v1.0
**日期：** 2026-09-04
**执行目录：** `D:\\uav-capture\\uav_capture`
**硬件：** NVIDIA RTX 5050
**运行范围：** development-only；`locked_test_opened=false`
**第一指标：** `safe_capture`；`mean_capture_time` 仅为次要诊断指标

> 本计划以 WP-7 tie3、WP-8 failure replay、P9 CBF solver audit 和 P11 rank-mismatch audit 为现状基线，目标是把“候选轨迹评价 + 可信度拒答 + CBF 安全过滤 + 滚动闭环”收敛为可审计的安全系统。计划不要求绝对捕获率达到 95%，也不允许用平均捕获时间抵消碰撞、越界、CBF 未验证动作或 safe-capture 回归。

## 1. 目标与系统边界

### 1.1 研究目标

实现以下闭环：

```text
观测/通信历史
  -> interaction-aware belief state
  -> 传统规划器生成动力学可行候选 action chunks
  -> action-conditioned JEPA 反事实预测
  -> reliability ledger 校准、信用与 abstention
  -> 安全优先候选排序/滞回
  -> Joint CBF-QP 过滤
  -> 只执行第一控制步
  -> 重新观测、更新 belief、重新规划
```

### 1.2 不可违反的边界

- JEPA 只能评价候选轨迹，不生成最终控制动作。
- `nominal`、所有 candidate、`safe-hold` 和 fallback 都必须经过同一个 Joint CBF-QP。
- QP 失败、超时、non-finite、OOD 或 stale 时，禁止执行 raw/unverified action。
- 在线输入不得包含 target ground truth；ground truth 只用于离线标签和 episode 结算。
- 每个 action chunk 固定 3 个 control steps，只执行第一步后 replan。
- 新实验使用新的 protocol、manifest、output root 和 checkpoint/ledger hash，不覆盖 WP-7 结果。

### 1.3 `safe_capture` 定义

一个 episode 只有同时满足下列条件才计为 `safe_capture=true`：

1. 至少一个 defender 在 time limit 内进入目标 `0.80 m` capture radius；
2. 无 obstacle、target 或 defender-defender collision；
3. 无 defender boundary/altitude violation；
4. 无 pairwise separation violation；
5. 无 CBF infeasible/timeout/unverified action 或 controlled-abort 终止。

target 越界单独记录为 diagnostic，不改写为 defender boundary failure。

## 2. 已知基线与当前判断

| 证据 | 当前结果 | 计划含义 |
|---|---:|---|
| WP-7 M0 nominal + CBF | 50.0% safe-capture；安全硬门通过 | 作为冻结主基线 |
| WP-7 M3 JEPA + ledger + auxiliary + CBF | 33.3%；相对 M0 配对 delta `-16.7 pp` | 停止宣称任务提升，先修排序/校准 |
| WP-7 M3 配对 | improved/degraded/tied = `10/30/80` | 失败必须逐 episode 解释 |
| WP-8 replay | 120/120 重放 hash 一致 | 可做因果和 settled counterfactual 审计 |
| P9 CBF | targeted 29 passed；smoke timeout=0 | solver 子门已改善，任务层仍未解决 |
| P11 rank audit | degraded switch rate `0.2161`，improved `0.1320`；degraded 全为 high-credit failure | score 与最终 settled outcome 失配 |
| A3 raw/no-CBF | 120/120 collision | CBF 是硬执行边界，不是可选消融 |

当前结论：安全执行架构已有正向证据；JEPA 排序的任务收益尚未证明，且当前 M3 是负向开发结果。下一步先做 P10 校准、P11 排序和 P12 ledger 修复，再决定是否重跑三 seed。

## 3. 阶段总览与依赖

```text
P0 冻结/预检
  -> P10 现有多任务头校准与不确定性审计
  -> P11 settled counterfactual 排序、滞回与保守 abstention
  -> P12 reliability ledger 重校准
  -> P13 rolling-horizon / zero-perturbation 集成回归
  -> P14 新 protocol smoke
  -> P15 三 seed paired development
  -> P16 SIL/HIL readiness（仅在 P15 安全门通过后）
```

P10、P11、P12 的离线审计可部分并行；P13 之后才能运行新的闭环 smoke；P15 之前不得打开 locked test。

## 4. P0：运行前冻结与工作区审计

### TODO

- [ ] 确认 Conda 环境 `uav-encirclement-gpu`、PyTorch/CUDA、RTX 5050 可见。
- [ ] 确认 `phase=development_only`、`locked_test_opened=false`，split 仅为 validation/development。
- [ ] 保存代码 revision、环境包清单、protocol、scene manifest、actor/JEPA/ledger/CBF 配置 SHA-256。
- [ ] 检查 `git diff --check`；忽略未完成的用户 E1/V5 修改，不执行整体清理。
- [ ] 为本轮建立独立根目录，例如 `results/jepa_safe_capture_v4_next_20260904/`。
- [ ] 建立本轮 TensorBoard 根目录，例如 `results/jepa_safe_capture_v4_tensorboard/`。
- [ ] 新建 protocol revision，明确候选数 `K=5`、chunk=3、tie tolerance=`5e-4`、CBF latency budget 和回退顺序。

### 预检命令

```powershell
Set-Location D:\\uav-capture\\uav_capture
$py = 'D:\\miniconda3\\envs\\uav-encirclement-gpu\\python.exe'
$env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
& $py scripts/verify_jepa_safe_capture_protocol.py --protocol <new_protocol.yaml>
git status --short
git diff --check
```

### 出口

- [ ] 所有输入 hash、环境信息和 protocol 已写入 `preflight.json`。
- [ ] locked-test 硬校验通过；否则停止后续任务。

## 5. P10：JEPA 多任务预测与校准审计

### 原则

当前 `InteractionAwareActionConditionedMultitaskJEPAPredictor` 已包含 target displacement、velocity/acceleration、obstacle/inter-agent clearance quantile、pairwise TTC、visibility、observation age、CBF correction/intervention 和 QP feasibility 头。不得重复添加同一结构；先证明现有头是否校准，只有发现明确缺口才创建新模型版本。

### TODO

- [ ] 审计训练/validation/calibration 的 episode、layout 和 seed 是否完全隔离。
- [ ] 以 calibration split 离线结算每个 `(checkpoint, context bucket, horizon, head)`。
- [ ] 对 target displacement/velocity/acceleration 计算 MAE、P50/P90/P95、constant-velocity 对照。
- [ ] 对 clearance lower-quantile 计算 coverage、underestimation/overestimation rate 和安全下界违约率。
- [ ] 对 visibility、CBF intervention、QP feasibility 计算 Brier、ECE、AUROC/AUPRC 及分桶可靠性曲线。
- [ ] 对 pairwise TTC、observation age、CBF correction 计算分位数误差和极端风险漏报率。
- [ ] 检查相同 belief 下 5 个 action chunks 的 latent/预测是否可辨识；记录 action-following separation 和方向一致性。
- [ ] 用 ensemble、MC-dropout 或 calibrated residual 只在现有 uncertainty 不足时增加不确定性估计，并新建 protocol。
- [ ] 不把 development 失败片段直接回灌旧训练 archive；如需困难片段重训，建立新的 archive/hash/seed。
- [ ] 三个 training seed 独立训练，记录 TensorBoard 的各任务 loss、梯度/激活统计、校准曲线和 provenance text。

### 建议命令模板

```powershell
& $py scripts/train_jepa_safe_capture_v2.py `
  --protocol configs/jepa_safe_capture_v2_protocol.yaml `
  --training-config configs/jepa_safe_capture_v2_training.yaml `
  --train-dataset results/jepa_safe_capture_v2_p1_train_rerun/counterfactual_safe_capture_v2.npz `
  --train-metadata results/jepa_safe_capture_v2_p1_train_rerun/metadata.json `
  --validation-dataset results/jepa_safe_capture_v2_p1_validation_rerun/counterfactual_safe_capture_v2.npz `
  --validation-metadata results/jepa_safe_capture_v2_p1_validation_rerun/metadata.json `
  --calibration-metadata results/jepa_safe_capture_v2_p1_calibration/metadata.json `
  --seed <20260911|20260912|20260913> `
  --device cuda `
  --output results/jepa_safe_capture_v4_next_train_seed<seed> `
  --tensorboard-logdir results/jepa_safe_capture_v4_tensorboard/train_seed<seed>

& $py scripts/evaluate_jepa_safe_capture_v2.py `
  --checkpoint results/jepa_safe_capture_v4_next_train_seed<seed>/checkpoint.pt `
  --dataset results/jepa_safe_capture_v2_p1_validation_rerun/counterfactual_safe_capture_v2.npz `
  --metadata results/jepa_safe_capture_v2_p1_validation_rerun/metadata.json `
  --output results/jepa_safe_capture_v4_next_validation_seed<seed>/metrics.json `
  --tensorboard-logdir results/jepa_safe_capture_v4_tensorboard/validation_seed<seed> `
  --device cuda
```

上面的第二条命令是 held-out validation 评估；calibration split 不应伪装成 validation，
应通过独立的 ledger/calibration builder 生成 calibration report 和只读 ledger。

### P10 准入门

- [ ] 所有输出 finite，且 checkpoint 可加载。
- [ ] 主要 horizon 的 displacement/velocity 预测不劣于 constant-velocity。
- [ ] clearance/visibility/CBF risk 头有非空覆盖，不出现系统性过度乐观。
- [ ] action-following separation 非零且方向一致，不接近随机。
- [ ] 三 seed 训练、校准和 hash 完整；否则停止，不接入闭环。

### 交付物

`prediction_calibration_report.md`、每 seed `metrics.json`、校准曲线/CSV、TensorBoard audit、checkpoint manifest 和新 protocol。完成后单独提交：`feat(jepa): audit multitask safety calibration`。

## 6. P11：Settled counterfactual 排序修复

### 目标

解决“高信用但最终失败”的 score 失配，而不是简单把 top-1 选择代码改成另一种 tie policy。

### TODO

- [ ] 对 WP-8 的 30 degraded、10 improved、80 tied 先做只读 settled counterfactual 标签分析。
- [ ] 固定安全优先分层：finite/reachability/预测安全下界/ledger 状态筛选在 task score 之前。
- [ ] 保留 5 个候选和 3-step chunk 合同，先不增加候选数。
- [ ] 评分至少包含 task progress、clearance lower quantile、visibility gain、CBF intervention cost、uncertainty penalty、action-change cost、nominal anchor penalty。
- [ ] 增加 top-two margin、rank stability、selected-not-best、switch rate、oscillation length、CBF correction 和 fallback probability 日志。
- [ ] 用 settled label 计算 top-1 precision/recall、Spearman/Kendall、Brier/ECE、分 bucket 的安全违约率。
- [ ] 当 top-two margin 小、visibility/clearance gap 突增、CBF risk 高或 high-credit failure pattern 命中时，执行 conservative abstention。
- [ ] 引入候选滞回和 minimum hold time；参数变化必须进入新 protocol 并重新跑 smoke。
- [ ] 用独立 calibration evidence 选权重，禁止按单个 seed 或 mean capture time 事后调参。

### 排序硬约束

```text
候选无效/不可达 -> 不进入 JEPA
预测安全下界不足 -> 不得因 task progress 被抬高
ledger abstain -> nominal-CBF 或 safe-hold-CBF
margin 不足/高信用失败模式 -> 保守拒答
所有最终动作 -> Joint CBF-QP
```

### 出口

- [ ] degraded replay 中 high-credit 错误排序下降。
- [ ] candidate switch/oscillation 相对 WP-7 有明确下降，或有可解释的无改善结论。
- [ ] 排序 trace 可与最终 CBF action/termination 一一关联。
- [ ] 变更通过 ranker 单元测试和 deterministic replay 测试。

## 7. P12：Reliability Ledger 重校准

### TODO

- [ ] 在独立 calibration split 注入目标急转、速度突变、遮挡、通信丢包、消息延迟、密度 shift、队形拥挤和 stale observation。
- [ ] 固定 `minimum_sample_count`、`minimum_credit`、uncertainty 上限、stale age 上限和 OOD 规则。
- [ ] 固定 credit decay/recovery、abstention hysteresis 和状态转移优先级。
- [ ] 四态路径固定为 `trusted -> JEPA ranking -> CBF`、`fallback_nominal -> nominal-CBF`、`safe_hold -> hold-CBF`、`controlled_abort -> terminate`。
- [ ] 每次决策写入 ledger key、state、credit、uncertainty、observation age、reason code、fallback mode 和 trace hash。
- [ ] 统计四种状态的 episode 比例、safe-capture、CBF intervention、abort、high/low-credit settled failure rate。
- [ ] 验证 high-credit failure rate 不高于 low-credit；不满足时禁止把 ledger 称为可靠性提升。
- [ ] 运行时 ledger 只读；禁止在线更新 threshold、credit 或 checkpoint。

### 出口

- [ ] OOD/stale/non-finite/低信用全部触发可验证回退。
- [ ] `raw_unverified_executed=0`。
- [ ] 每次 fallback 可由单个 trace 解释并可第二次重放。
- [ ] ledger、checkpoint、calibration archive 和 protocol hash 绑定。

## 8. P13：滚动时域与安全执行回归

### TODO

- [ ] 验证每个周期只执行 action chunk 的第一步，下一周期重新观测、更新 belief、预测、排序和 CBF。
- [ ] 验证 candidate、nominal、safe-hold、fallback 均调用同一个 Joint CBF-QP。
- [ ] 验证回退顺序严格为：`separation-preserving safe-hold -> verified nominal-CBF -> controlled abort`。
- [ ] 注入 QP infeasible、timeout、non-finite request、过期 observation、通信中断、多约束同时激活和单机状态异常。
- [ ] 做 zero-perturbation regression：关闭 JEPA 后，非 JEPA 字段逐字段与 M0 相同；JEPA 只能影响 candidate score/selection。
- [ ] 记录 monotonic start/end、solver status、iteration、active constraints、slack、correction norm、fallback reason 和 latency。
- [ ] 用 100 个随机 control cycles 做 raw/unverified action assertion 和 deterministic trace replay。

### P13 硬门

- [ ] 所有动作 finite 且均经过 CBF verification。
- [ ] `raw_unverified_executed=0`，CBF timeout 要么为 0，要么每次都有已验证 fallback。
- [ ] CBF/JEPA/ledger 端到端 p95 不超过 100 ms。
- [ ] interface、fallback、zero-perturbation、trace schema 测试全部通过。

## 9. P14：新 protocol smoke

### TODO

- [ ] 为本轮生成全新的 scene manifest，保持 train/calibration/development 不重叠。
- [ ] 每个 seed 先运行 20 集：M0、M3、A1、A2；A3 仅为 `diagnostic_only`。
- [ ] 同一 manifest、同一 episode index 配对运行 baseline 与候选。
- [ ] smoke 完成后立即运行 aggregate、failure index、rank audit、ledger audit、CBF safety audit 和 TensorBoard audit。
- [ ] 任一安全硬门、provenance 门或延迟门失败，停止扩展，不增加 episode 数量。

### smoke 准入门

- [ ] 安全保留变体 collision、defender boundary、pairwise violation 均为 0。
- [ ] raw/unverified action 为 0；timeout=0 或有可验证 fallback。
- [ ] zero-perturbation 通过，所有产物 hash/数量一致。
- [ ] 结果不是依赖单个随机重放或手工修改 manifest。

## 10. P15：三 seed paired development

仅当 P10-P14 全部通过后执行。不得覆盖 WP-7 tie3。

### TODO

- [ ] 固定三个 training seed `20260911/20260912/20260913`，每 seed 使用独立 checkpoint/ledger。
- [ ] 每 seed 至少 40 个 paired validation episodes，总计至少 120 对 episode。
- [ ] 运行顺序固定：M0 -> M3 -> A1 -> A2；A3 单独诊断且不进入安全主结论。
- [ ] 运行前冻结 score 权重、tie tolerance、CBF margin/gamma、chunk 长度、capture radius、episode manifest 和设备（RTX 5050 CUDA）。
- [ ] 每个 run 保存 `summary.json`、`episodes.csv`、step traces、scene manifest、provenance、TensorBoard 和 SHA-256。
- [ ] 聚合时以 `(training_seed, episode_index)` 为独立单位，不把 timestep/candidate/chunk 当独立样本。
- [ ] 报告逐 seed safe-capture、样本 SD、paired delta、improved/degraded/tied、bootstrap 95% CI 和 exact McNemar。
- [ ] 失败按 collision、boundary、pairwise、CBF abort/timeout、ledger fallback、high-credit ranking、clearance/visibility mismatch、target drift、oscillation、latency 分桶。

### 结论分类

| 分类 | 条件 | 后续 |
|---|---|---|
| `safe_capture_improvement_candidate` | 安全、可靠性、实时性、provenance 全通过；配对 delta 非负且至少 2/3 seed 非负 | 只写 development 正向证据，另行申请 locked preregistration |
| `safe_capture_noninferior_safety_preserving` | 安全无退化，delta 不能确认提升 | 保留架构，继续做评价器改进 |
| `prediction_signal_no_control_gain` | 预测校准改善但闭环 safe-capture 中性/负向 | 禁止宣称控制收益 |
| `rejected_for_safety` | 新 collision/boundary/pairwise、raw action 或不可审计回退 | 停止该变体并归档失败 |
| `insufficient_evidence_do_not_open_locked_test` | seed、配对、统计或 provenance 不完整 | 不打开 locked test |

不设置“必须达到 95%”的硬目标；首要判断是 safe-capture 安全不劣性和失败可解释性。

## 11. P16：SIL/HIL 准备（条件任务）

只有 P15 通过安全硬门后才允许开始：

- [ ] 固定仿真步长、控制周期、通信延迟、传感器噪声和 watchdog。
- [ ] 测量 p50/p95/p99 latency、显存、CPU、消息队列积压和 CBF 失败响应。
- [ ] 注入断网、传感器冻结、目标突变、单机失效和进程重启。
- [ ] 验证 separation、boundary、safe-hold、controlled-abort 和急停策略。
- [ ] 形成 SIL/HIL 报告和风险清单；HIL 通过不等于真实飞行许可。
- [ ] 未完成安全审查前禁止真实飞行。

## 12. TensorBoard、Git 与 provenance 纪律

### TensorBoard 必须记录

```text
Loss/target, Loss/velocity, Loss/acceleration, Loss/clearance,
Loss/visibility, Loss/cbf_risk, Loss/action_consistency,
Calibration/*, Reliability/*, Ranking/*, CBF/*, Safety/*,
Latency/*, Fallback/*, Provenance/*
```

- [ ] 每个 seed 独立 logdir；不少于 40 个 epoch 点。
- [ ] 写入配置、命令、环境、checkpoint/archive/ledger/protocol hash。
- [ ] 结果缺少必需 tag 时不能进入 paired block。

### Git 提交顺序

- [ ] P0：`docs(jepa): freeze next safe-capture protocol`
- [ ] P10：`feat(jepa): audit multitask safety calibration`
- [ ] P11：`feat(jepa): add settled rank safeguards`
- [ ] P12：`feat(jepa): recalibrate reliability ledger`
- [ ] P13：`test(jepa): verify rolling horizon safety contract`
- [ ] P14/P15：`docs(jepa): archive next paired development`
- [ ] 每次只选择性 `git add` 本阶段文件；不提交无关 E1/V5 修改；提交后记录 commit hash 和 push 状态。

## 13. 时间盒与停止规则

| 时间盒 | 任务 | 交付物 |
|---|---|---|
| Day 0 | P0 冻结与预检 | protocol、preflight、hash manifest |
| Day 1--3 | P10 校准/不确定性 | prediction/calibration report、三 seed metrics |
| Day 3--6 | P11 排序/滞回 | settled rank audit、ranker tests、new config |
| Day 5--7 | P12 ledger | ledger manifest、OOD/stale/credit report |
| Day 7--9 | P13 闭环回归 | safety/fallback/zero-perturbation audit |
| Day 9--10 | P14 smoke | 20-episode/seed smoke 与 gate memo |
| Day 11--15 | P15 paired development | 3 seed x 40 paired matrix、统计报告 |
| Day 16+ | P16 SIL/HIL | 仅在安全门通过后启动 |

任何安全硬门失败时：保存失败产物 -> 写 failure memo -> 停止该变体 -> 新建 protocol revision。不得挑 seed、删失败 episode、修改统计口径或用 capture time 掩盖 safe-capture 退化。

## 14. Definition of Done

本轮只有全部满足才算完成：

1. P10 safety heads 在独立 calibration split 有可复现的校准证据。
2. P11 settled counterfactual ranking 能解释并减少 high-credit failure/oscillation，或明确归档为无控制收益。
3. P12 ledger 对 OOD、stale、non-finite、低信用执行确定性 abstention，`raw_unverified_executed=0`。
4. P13 每周期只执行第一步，所有动作经 Joint CBF-QP，fallback trace 可重放。
5. P14 smoke 通过安全、实时性、provenance 和 zero-perturbation 硬门。
6. P15 三 seed paired development 完整运行并按 safe-capture 优先报告。
7. 所有结果均为 development-only；未获明确授权前 `locked_test_opened=false`。

最终可主张的系统性质必须是：**action-conditioned interaction-aware JEPA 负责反事实候选评价，reliability ledger 负责可信度和拒答，Joint CBF-QP 负责不可绕过的安全约束，rolling horizon 负责闭环修正；safe-capture 证据决定是否存在任务收益。**
