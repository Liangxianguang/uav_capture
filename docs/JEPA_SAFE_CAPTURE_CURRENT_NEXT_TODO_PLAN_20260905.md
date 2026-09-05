# 无人机集群对抗围捕
# Interaction-aware Action-conditioned JEPA + Reliability Ledger + CBF 下一步执行计划

**版本：** 2026-09-05 / v21-post-smoke-current
**执行目录：** `D:\\uav-capture\\uav_capture`
**硬件：** NVIDIA GeForce RTX 5050
**当前 revision：** `b9693b4bb071b2e64545fe832188604707b10d41`
**实验边界：** `development_only=true`，`locked_test_opened=false`
**唯一主指标：** episode-level `safe_capture`
**次级诊断：** mean capture time、Transit、minimum clearance、CBF correction、fallback/abort、候选切换率、周期延迟

> 这是一份当前执行入口，不是新的实验结果。绝对 95% 不再是硬目标。任何捕获时间、Transit、路径或 CBF 修正量改善，都不能抵消 safe-capture 下降或安全合同违规。

## 1. 研究目标和不可变边界

### 1.1 目标

为无人机集群对抗围捕构建一条可以审计、可以拒答、可以安全回退、可以滚动重规划的闭环：

```text
多机观测/通信历史
  -> interaction-aware belief state
  -> 传统规划器生成 K 个动力学可行候选 action chunks
  -> action-conditioned JEPA 反事实轨迹评价
  -> immutable reliability ledger 可信度校验/拒答
  -> safety-first candidate ranker + nominal anchor + abstention
  -> Joint CBF-QP 安全过滤
  -> 仅执行 action chunk 第 1 步
  -> 重新观测、重新规划、重新过滤
```

### 1.2 责任边界

- JEPA 是轨迹评价器，不生成最终控制动作，也不能绕过 CBF。
- 候选由传统规划器产生；所有候选、nominal、safe-hold 和 fallback 使用同一个 Joint CBF-QP。
- reliability ledger 只在离线 calibration split 拟合，在线只读；信用、阈值和 hash 不能在线更新。
- OOD、stale、non-finite、低信用、预测漂移、provenance mismatch 或 CBF 失败时，必须进入固定回退链：

```text
separation-preserving safe-hold
  -> verified nominal through the same Joint CBF-QP
  -> controlled_abort
```

- Rolling horizon 每周期只执行第一控制步；第 2/3 步必须等待新观测。
- 在线组件不能读取 target ground truth。ground truth 只用于离线 settled label 和 episode 结算，并标记 `offline_only=true`。
- `controlled_abort` 必须计入失败分母，不得改写为普通 timeout 或删除。

### 1.3 逐周期接口契约

| 组件 | 输入（在线可见） | 输出 | 失败时的行为 |
|---|---|---|---|
| BeliefState | 多机状态、历史观测/通信、时间戳年龄、障碍/边界几何、历史动作 | 固定 shape 的 belief、mask、age、belief hash | 缺失或过期时标记 stale，禁止伪造 target truth |
| Candidate Planner | belief、传统几何/动力学规则 | `K` 个 action chunks、类型、reachability、动作 hash | 不可行候选写 eligibility reason，不送入 JEPA |
| JEPA Evaluator | belief history、候选首动作/chunk、horizon | latent、目标进度、净空/TTC、visibility、CBF risk、uncertainty | non-finite/shape/horizon 错误时交给 ledger 拒答 |
| Reliability Ledger | JEPA 输出、context bucket、checkpoint/protocol hash | `trusted/degraded/fallback_nominal/safe_hold`、credit、reason code | 只读拒答并走固定回退链 |
| Safety-first Ranker | 可用候选、ledger 状态、nominal anchor、hysteresis | selected candidate、abstention、rank trace | 无可信分离时选 nominal 或 safe-hold |
| Joint CBF-QP | selected/nominal/safe-hold action、全体 UAV 状态、障碍/边界约束 | finite verified action、solver status、active set、slack、correction | infeasible/timeout/non-finite 时不可执行 raw |
| Rolling Executor | verified action 第一控制步、termination/watchdog | executed action、cycle trace、replan trigger | safe-hold -> nominal-CBF -> controlled-abort |

严格顺序固定为：

```text
observe -> belief/hash -> candidate validity -> JEPA batch -> ledger gate
-> safety-first rank -> Joint CBF-QP -> execute first step -> trace/settle
```

每一周期 trace 至少包含 `episode_seed`、`layout_hash`、`cycle_index`、`belief_hash`、candidate eligibility/reasons、JEPA outputs、ledger state/credit、selected index、abstention/hysteresis、CBF solver/active constraints/slack/correction、requested/executed action、latency、fallback 和 termination 字段。

### 1.4 Safe capture 定义

一个 episode 只有同时满足以下条件才是 `safe_capture=true`：

1. time limit 内至少一架 defender 进入目标 `0.80 m` capture radius；
2. 没有 obstacle、target 或 defender-defender collision；
3. 没有 world-boundary/altitude violation；
4. 没有 pairwise separation violation；
5. 没有未验证动作、未处理 CBF 失败或 `controlled_abort` 终止。

## 2. 当前证据基线

### 2.1 已完成的安全和执行证据

- CPU rolling-horizon repeat1/repeat2：每次 20 episodes、1189 control cycles，`safe_capture=12/20=60%`，collision/boundary/pairwise 为 0，controlled abort 为 8，`raw_unverified=0`。
- RTX 5050 CUDA M3 replay：20 episodes，`safe_capture=12/20=60%`，collision/boundary/pairwise 为 0，CBF controlled abort 为 8，timeout 为 0，`raw_unverified=0`。
- CPU/CUDA comparator：candidate decision、ledger/CBF status、executed action、termination 和 safety settlement 均等价；cycle p95 低于 100 ms。
- V21 三 seed、四变体 smoke 的安全硬门全部通过：collision、boundary、pairwise、CBF timeout 和 `raw_unverified_executed` 均为 0。
- hash-bound ledger、fault injection、non-finite safe-hold、zero-perturbation 和滚动执行合同均有独立测试/报告。

### 2.2 当前控制结果和真正阻断点

| 变体 | seed 20260911 | seed 20260912 | seed 20260913 | 三 seed均值 +/- 样本SD |
|---|---:|---:|---:|---:|
| M0 nominal + CBF | 10/20 | 10/20 | 10/20 | 50.0% +/- 0.0% |
| M3 JEPA + ledger + ranker + CBF | 12/20 | 7/20 | 9/20 | 46.7% +/- 12.6% |
| A1 JEPA + CBF（无 ledger） | 10/20 | 10/20 | 9/20 | 48.3% +/- 2.9% |
| A2 JEPA + ledger + CBF（无 clearance/visibility 排序项） | 11/20 | 9/20 | 10/20 | 50.0% +/- 5.0% |

M3 相对同 seed M0：`-3.33 pp`，episode-pair bootstrap 95% CI `[-11.67, +5.00] pp`，只有 `1/3` seed 非负。因此当前结论是 `useful_safety_fallback_only`，不是 JEPA 已提升 safe-capture。

settled counterfactual 还发现 M3 的 selected-not-best 分别为 `35.3%`、`39.1%`、`50.7%`，Spearman 分别为 `-0.319`、`-0.294`、`-0.306`。这说明候选评分和实际 settled outcome 仍有系统失配，是下一轮训练和大规模闭环实验的第一阻断门。

### 2.3 当前不允许的动作

- 不把单 seed 的 95% 写成正式结果。
- 不扩大 40/60 集、不打开 locked test，直到 ranking/ledger 诊断和安全门同时通过。
- 不降低 CBF margin，不放宽 OOD/stale，不删除 controlled abort，不把 raw policy 当安全结果。
- 不用 mean capture time、Transit、prediction MAE 或 CBF correction 单独宣称任务收益。

## 3. 目标指标、闸门和结果标签

### 3.1 永远优先的硬安全闸

以下任意一项失败，整个候选版本停止扩大实验并标记 `rejected_for_safety`：

- collision、boundary、pairwise separation 或未验证动作不为 0；
- CBF infeasible/timeout 没有明确 safe-hold -> nominal-CBF -> controlled-abort 路由；
- trace 缺少 candidate、ledger、CBF、executed action 或 termination 字段；
- online 读取 target truth、online 更新 ledger 或 provenance/hash 不一致；
- RTX 5050 cycle p95 超过 100 ms，或设备间决策不等价且未解释。

### 3.2 控制收益准入门

在一个新模型进入 40/60 集 development block 前，必须同时满足：

- 相对同一 episode manifest 的 M0，aggregate safe-capture 不低于 M0；
- 至少 `2/3` training seeds 的 paired delta 非负；
- paired 95% CI、improved/degraded/tied 和 McNemar/exact test 已报告；
- 所有安全保留变体的硬安全闸通过；
- 不以 mean capture time 或 Transit 替代 safe-capture。

### 3.3 排序/可信度准入门（先于控制收益门）

阈值必须在使用新 development 结果前写入 protocol。默认目标为：

- selected-not-best 不高于 25% aggregate，且没有 seed 高于 40%；
- score 与 settled safety/progress 不再三 seed 同向反转；Spearman/Kendall 必须至少不呈系统负相关；
- high-credit settled unsafe rate 不高于 low-credit bucket 加 5 pp；
- candidate separation、eligibility、abstention 和 nominal-anchor 触发率有完整分桶；
- reliability 校准报告 finite rate、coverage、Brier/ECE 或 AUROC，并证明低信用确实路由到回退。

这些是模型晋级门，不是可以事后调参的指标。未通过时标签为 `ranking_unresolved`，不得继续扩大 episode。

## 4. 当前执行队列

```text
S0 保护当前证据和环境
  -> S1 三 seed settled rows / ranking / candidate separation 聚合（立即）
  -> S2 failure index + deterministic hard replay（诊断）
  -> S3 修复 score、label、horizon、action scale 或 eligibility（仅在 S2 定位后）
  -> S4 多任务 interaction-aware JEPA + train-only hard replay（条件训练）
  -> S5 checkpoint-bound reliability ledger 重新校准
  -> S6 20 集三 seed paired smoke（新 protocol）
  -> S7 40/60 集 development + robustness stress（仅通过 S6）
  -> S8 SIL/HIL readiness、预注册和 locked-test 决策
```

上游步骤失败时，保留失败产物，修复后创建新的 protocol/checkpoint/ledger/output 前缀；禁止覆盖旧结果。

## 5. S0：证据、环境和数据保护

**状态：** 基本完成，后续每个新 run 仍必须执行。

### TODO

- [x] 保留用户现有 dirty worktree、`tmp/`、checkpoint 和 NPZ；不执行 `git reset`、`git checkout` 或 `git clean`。
- [x] 保存当前 revision、协议 hash、训练 seed、设备信息和现有结果目录。
- [ ] 为每个新 run 写入 `preflight.json`：Python 路径、Conda 包、GPU/driver/CUDA、`torch.cuda.is_available()`、git revision、protocol/checkpoint/ledger/scene hash、命令和 `locked_test_opened`。
- [ ] 处理 TensorBoard 与 protobuf 版本兼容性：优先固定环境中的兼容 protobuf 版本；临时使用 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 时必须在 metadata 记录。
- [ ] 所有输出目录先确认为空；脚本必须拒绝覆盖非空目录。
- [ ] 将固定 archive 路径和 SHA-256 写入数据清单；不把 `tmp` 中 archive 当作 Git 已发布资产。

### 出口

`preflight.json`、input hash manifest、命令和环境信息齐全，且所有运行仍为 development-only。

## 6. S1：settled ranking 与候选分离聚合（已完成）

**目标：** 解释 JEPA score 为什么没有可靠地对应真实安全捕获，不修改 CBF 安全参数。

**状态：** 已完成。详见 [V21 三 Seed Settled Ranking Aggregate](JEPA_SAFE_CAPTURE_V21_SETTLED_RANKING_AGGREGATE_20260905.md)。source gates 通过，但 ranking promotion gate 失败，当前标签为 `ranking_unresolved`。

### TODO

- [x] 对三个 seed 的 `decision_rows.jsonl` 做聚合。
- [x] 新增 `scripts/aggregate_jepa_safe_capture_v21_settled.py` 及对应单测；脚本只读已有 settled rows。
- [x] 校验候选 settled outcome、source gate、protocol/environment/scene hash 和 decision count。
- [x] 计算 selected-not-best、settled safety/progress、Spearman/Kendall、candidate separation 和 credit/visibility/TTC/CBF correction 分桶。
- [x] 生成独立 JSON、Markdown、TensorBoard 和 input manifest。
- [x] 聚合器拒绝缺 seed、重复 decision key、manifest/hash 不一致和错误的 episode 语义。

### 建议命令模板

```powershell
$py = 'D:\\download\\anaconda3\\envs\\traj_pred_prep\\python.exe'
$env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"
& $py scripts/audit_jepa_safe_capture_v21_candidate_separation.py `
  --protocol configs/central_random_mixed_obstacle_s3_v5_v21_cpu_separation_gate_development_protocol.yaml `
  --output-dir results/jepa_safe_capture_v21_current_candidate_separation `
  --tensorboard-dir results/jepa_safe_capture_v21_tensorboard/current_candidate_separation

& $py scripts/audit_jepa_safe_capture_v5_settled_counterfactual.py `
  --run results/jepa_safe_capture_v21_smoke_m3_seed20260911 `
  --baseline-run results/jepa_safe_capture_v21_smoke_m0_seed20260911 `
  --protocol configs/central_random_mixed_obstacle_s3_v5_v21_cpu_separation_gate_development_protocol.yaml `
  --environment-config configs/capture_radius_pursuit_central_v4_flee.yaml `
  --output-dir results/jepa_safe_capture_v21_current_settled_seed20260911 `
  --tensorboard-logdir results/jepa_safe_capture_v21_tensorboard/current_settled_seed20260911 `
  --development-only
```

对 seed `20260912`、`20260913` 使用独立输出目录重复 settled audit；不要合并不同 seed 的 scene manifest。

### 出口

- ranking 方向、horizon、action scale 和 eligibility mask 已定位，或明确标记 `ranking_unresolved`；
- 所有三 seed 的 settled audit `all_gates_pass=true`；
- 生成 `ranking_diagnosis.json`、`candidate_separation_audit.json` 和 `settled_aggregate.json`；
- 未定位前不得训练新模型或扩大 episode。

## 7. S2：失败索引和确定性困难片段重放（当前任务）

**目标：** 把失败归因到模型预测、ledger 路由、候选生成、CBF 可行性或环境结算，而不是只看 episode 成败。

**状态：** 进行中。S1 已确认 ranking unresolved，允许进入失败归因，但不允许扩大 episode 或打开 locked test。

### TODO

- [ ] 建立 failure index：每条失败关联 episode seed、layout hash、cycle、候选 index、score、credit、uncertainty、visibility、TTC、predicted/observed clearance、CBF status、fallback 和 termination。
- [ ] 优先重放：selected-not-best、high-credit failure、候选 separation 低、CBF correction 高、controlled abort、visibility 丢失、target 急转和 observation/message stale。
- [ ] 每个片段做两次 deterministic replay，比较 decision rows、settled rows 和 CBF trace；保留最早 divergence cycle。
- [ ] 将失败分为 `prediction_error`、`ranking_error`、`ledger_overtrust`、`candidate_unreachable`、`cbf_infeasible`、`observation_stale`、`environment_termination`。
- [ ] 失败片段不得直接回灌旧 archive；后续训练只能使用新的 train-only archive、manifest 和 replay weights。

### 出口

每个 failure bucket 有计数、样例索引、原因证据和重放 hash；无法归因的部分显式标记 `unresolved`。

## 8. S3：只在诊断完成后修复排序合同

**目标：** 先修正符号/尺度/标签/时域等可验证错误，再考虑改模型容量。

### 修复顺序

1. score direction：明确“越小越好/越大越好”，加入人工单调性测试；
2. horizon alignment：预测 horizon、action chunk 和 settled local rollout 必须相同；
3. action contract：统一 actor 输出、物理动作、clip、slew、单位和 replay action；
4. eligibility：finite、shape、速度、加速度、reachability 和 clearance gate 分开记录；
5. rank policy：安全优先、nominal anchor、top-two abstention、minimum separation 和 hysteresis；
6. 最后才允许冻结新 score 权重，并创建新的 protocol revision。

### 必须新增的回归测试

- 人工单调性：更安全的 clearance/TTC/visibility 候选不能因 task progress 被错误压过；
- 方向/尺度：固定输入下 score 方向、量化 quantum 和 tie policy 稳定；
- mask：invalid candidate 不进入 JEPA，也不成为 settled best；
- zero-perturbation：关闭 JEPA 后非 JEPA 字段逐字段等于 M0；
- CPU/CUDA：离散 decision、CBF status、executed action 和 termination 按 protocol 等价。

### 出口

新 ranking 在冻结三 seed trace 上不再系统性反向；否则停止并保留 `ranking_unresolved`。

## 9. S4：interaction-aware 多任务 JEPA 与困难片段训练

**触发条件：** S1-S3 通过，或已证明当前 score 合同正确但安全预测信号不足。

### 9.1 输入表示

- defender-target 相对位置/速度/加速度；
- defender-defender 相对位置、pairwise TTC 和编队几何；
- target belief、visibility、observation age、message age、通信 mask；
- obstacle/boundary 局部几何、历史动作、上周期 CBF 状态和执行噪声；
- permutation-invariant agent pooling 或受控 agent-id，避免队伍排序造成 latent 漂移。

### 9.2 输出头

对每个候选 action chunk 和 horizon `[1,2,3,5]` 输出：

```text
target displacement / velocity / acceleration
obstacle-clearance lower quantile
inter-agent clearance lower quantile / pairwise TTC
target visibility probability / observation-age risk
CBF intervention probability / correction magnitude / QP feasibility
predictive uncertainty / candidate disagreement
```

所有输出只用于评价、排序、拒答和诊断；不构成安全证书。

### 9.3 数据和训练 TODO

- [ ] 从 train split 重新生成 counterfactual archive，validation 保持 episode/layout/运动 seed 隔离。
- [ ] 困难片段覆盖急转、突发加速度、遮挡/延迟、候选分离消失、低净空、高 CBF correction 和 controlled abort。
- [ ] 使用 train-only replay weights；uniform draw fraction 至少 0.50，记录去重、采样上限和 archive hash。
- [ ] 三个 training seed 独立初始化、独立 optimizer state、独立 checkpoint；不共享 validation settled outcome。
- [ ] 记录 latent、target、clearance、visibility、CBF intervention 和 uncertainty 的 loss、MAE、Brier/ECE、AUROC、finite rate 和 coverage。

### 训练命令模板

```powershell
& $py scripts/generate_jepa_v3_counterfactual_dataset.py `
  --collection-config <train_collection_config.yaml> `
  --v3-protocol configs/jepa_v3_development_protocol.yaml `
  --output results/jepa_safe_capture_v22_archive_seed<seed> `
  --split train

& $py scripts/build_jepa_v3_hard_replay_weights.py `
  --dataset <train_dataset.npz> --metadata <train_metadata.json> `
  --protocol configs/jepa_v3_development_protocol.yaml `
  --output <train_replay_weights.npz> --manifest <train_replay_manifest.json>

& $py scripts/train_interaction_aware_jepa_multitask.py `
  --protocol configs/jepa_v3_development_protocol.yaml `
  --train-dataset <train_dataset.npz> --train-metadata <train_metadata.json> `
  --validation-dataset <validation_dataset.npz> --validation-metadata <validation_metadata.json> `
  --train-replay-weights <train_replay_weights.npz> `
  --train-replay-manifest <train_replay_manifest.json> `
  --output results/jepa_safe_capture_v22_model_seed<seed> `
  --tensorboard-logdir results/jepa_safe_capture_v22_tensorboard/seed<seed> `
  --seed <seed> --device cuda
```

### 预测准入门

- 全部输出 finite，shape 和 horizon 正确；
- 至少一个主要 horizon 不劣于 constant-velocity baseline；
- clearance、TTC、visibility 和 CBF risk 头不过度乐观且可校准；
- prediction gate 通过只允许建立 ledger，不等于 safe-capture 提升。

## 10. S5：reliability ledger 重新校准和路由

**目标：** 将“模型感觉可信”改成可验证的 execution-settled credit。

### TODO

- [ ] 用 calibration split 计算 `(context, obstacle/layout bucket, horizon, uncertainty, visibility, TTC, clearance)` 条件下的 settled credit。
- [ ] checkpoint、protocol、calibration archive、clearance transform 和 ledger 文件全部 hash-bound。
- [ ] ledger 状态固定为 `trusted`、`degraded`、`fallback_nominal`、`safe_hold`、`rejected_provenance`；在线禁止更新。
- [ ] low credit、stale、OOD、non-finite、unknown horizon 和高 uncertainty 必须触发回退；记录 reason code、credit、阈值和 stale age。
- [ ] 检验 high-credit settled unsafe rate 不高于 low-credit bucket 加 5 pp；否则 ledger gate 失败。
- [ ] 注入 hash mismatch、checkpoint mismatch、nan/inf、solver timeout、GPU unavailable 和进程重启 fault；任何 fault 不得执行 raw。

### 出口

immutable ledger、fault regression、ledger alignment、OOD/stale 路由和 checkpoint-bound provenance 全通过，才可进入新闭环 smoke。

## 11. S6：新 checkpoint 的三 seed paired smoke

**目标：** 先用小样本验证方向和安全，再决定是否扩大。

### 实验矩阵

| 变体 | 目的 | 必须保留的安全层 |
|---|---|---|
| M0 | nominal baseline | Joint CBF-QP |
| M3 | JEPA + ledger + safety-first ranker | Joint CBF-QP |
| A1 | 去 ledger，测量信用路由代价 | Joint CBF-QP |
| A2 | 去 clearance/visibility ranking 项 | Joint CBF-QP |
| A3 | raw policy 仅作故障诊断 | 不进入安全结论 |

### TODO

- [ ] 每个 training seed 先由 M0 生成一个 scene manifest；M3/A1/A2 逐 episode 复用该 manifest。
- [ ] 每个变体每 seed 20 episodes；每个目录独立、不可覆盖。
- [ ] 输出 safe-capture、完整安全失败矩阵、paired delta、CI、McNemar、settled ranking、ledger bucket、fallback、latency 和 hash。
- [ ] 运行 rolling-horizon audit、CPU/CUDA comparator、zero-perturbation 和 fault regression。

### 晋级规则

- 安全硬门失败：`rejected_for_safety`；
- 安全通过但 M3 aggregate 低于 M0，或仅 0/1 seed 非负：`prediction_signal_no_control_gain`；
- 安全通过、排序门通过且 M3 不低于 M0、至少 2/3 seed 非负：可申请 S7；
- 无论结果高低，都公开 controlled abort、fallback、CBF correction 和失败 episode。

## 12. S7：40/60 集 development、困难场景和鲁棒性

**前置条件：** S6 所有安全、排序、ledger 和 provenance 门通过。

### TODO

- [ ] 新建独立 40 集 manifest；不得将多个 20 集结果拼成伪 40 集。
- [ ] 先完成每 seed 40 episodes；资源允许时追加独立 60-episode block，仍使用新 manifest。
- [ ] 条件覆盖 nominal/delayed/noisy observation、message delay/dropout、target turn/acceleration、flee persistence/S-curve、3--5 obstacles、狭窄通道、拥挤编队和 initial side-distance shift。
- [ ] 每个条件同时报告 safe-capture 和安全失败，不用总体平均掩盖 hard bucket。
- [ ] 记录 visibility、TTC、predicted/observed clearance、CBF intervention、ledger state、fallback、abstention 和 cycle p50/p95/p99。
- [ ] 运行设备复现：RTX 5050 CUDA 为主，CPU 作为 comparator；设备差异必须记录最早 divergence cycle。

### 结果标签

`safe_capture_improvement_candidate`、`safe_capture_noninferior`、`prediction_signal_no_control_gain`、`useful_safety_fallback_only`、`rejected_for_safety`、`insufficient_evidence_do_not_open_locked_test`。

## 13. S8：SIL/HIL readiness 和 locked-test 决策

### SIL/HIL TODO

- [ ] 固定 observation timestamp、communication age、执行回执、CBF solver status、watchdog 和 abort interface schema。
- [ ] HIL 指令仍必须经过 CBF；HIL 通过不等于真实飞行许可。
- [ ] 注入单机通信/执行失败、GPU 不可用、进程重启、JEPA non-finite、ledger hash mismatch，并验证 safe-hold/nominal/abort 路由。
- [ ] 测量 queue age、丢帧、重启恢复时间和端到端 p50/p95/p99。

### 只有以下条件全部满足，才允许准备 locked block

- 三 seed development 与 paired manifest 完整可重放；
- safe-capture 的主结论成立，或明确归档为无控制收益；
- collision、boundary、pairwise、raw-unverified 硬门全通过；
- CBF failure、controlled abort、fallback 和 latency 全部公开；
- settled ranking 不再系统性反向，ledger calibration gate 通过；
- CPU/CUDA comparator、rolling-horizon、zero-perturbation、fault 和 stress 报告齐全；
- 新 locked protocol、scene manifest、checkpoint、ledger 和 preregistration 已冻结，development 结果不可回写 locked split。

否则保持 `locked_test_opened=false`，最终标签为 `insufficient_evidence_do_not_open_locked_test`。

## 14. 每次运行的验收清单

### 运行前

- [ ] split 不是 locked，`development_only=true`，输出目录为空；
- [ ] protocol、environment、actor、JEPA、calibration、ledger、scene manifest hash 已保存；
- [ ] RTX 5050/CUDA/PyTorch/Conda 信息已保存；
- [ ] targeted tests、protocol schema、`git diff --check` 通过；
- [ ] 不读取 online target truth，不在线更新 ledger。

### 运行中

- [ ] observe -> belief -> candidate -> reachability -> JEPA -> ledger -> rank -> Joint CBF -> execute first step 顺序不变；
- [ ] 只执行 finite + verified 的 CBF 第一控制步；
- [ ] 每周期保存 candidate eligibility、JEPA output、ledger state、rank、CBF active set/slack/correction、fallback、latency 和 provenance；
- [ ] 不删除失败 episode、controlled abort、timeout 或 non-finite trace。

### 运行后

- [ ] `summary.json`、`episodes.csv`、`step_traces/`、`provenance.json`、hash manifest、命令和 TensorBoard 齐全；
- [ ] JSON/CSV/Markdown/TensorBoard 的 episode 数、计数、hash 一致；
- [ ] 运行 settled ranking、failure index、ledger alignment、rolling audit、device comparator 和 aggregate；
- [ ] 生成独立阶段报告，明确 PASS/FAIL、结果标签和下一步，不改写历史报告。

## 15. 产物命名和提交边界

新产物使用独立前缀：

```text
results/jepa_safe_capture_v21_current_*/
results/jepa_safe_capture_v22_*/
results/jepa_safe_capture_current_tensorboard/*
docs/JEPA_SAFE_CAPTURE_CURRENT_NEXT_TODO_PLAN_*.md
```

每个阶段只提交该阶段的代码、测试、配置和报告；禁止 `git add .`。`tmp/`、NPZ、checkpoint、TensorBoard 和 episode trace 继续作为本地生成资产，必须通过 manifest/hash 引用。

## 16. Definition of Done

- [ ] JEPA 能利用多机交互和 action chunk 评价目标运动、净空、可见性和 CBF 干预风险；
- [ ] score、label、horizon、action scale 和 eligibility 已审计，settled ranking 不再系统性反向；
- [ ] reliability ledger 对 provenance、OOD、stale、non-finite、低信用和不确定性可拒答、可重放；
- [ ] 所有候选和回退均经过同一个 Joint CBF-QP，rolling horizon 只执行第一步；
- [ ] RTX 5050 上端到端 cycle p95 不超过 100 ms，CPU/CUDA comparator 通过；
- [ ] 新 checkpoint 至少三 seed paired smoke 以 episode-level safe-capture 为主指标；
- [ ] 只有在证据充分时才申请 locked test，否则如实保留负结果。

**下一次实际动作：** 先执行 S1 三 seed settled ranking/candidate separation 聚合，再执行 S2 failure index 和确定性困难片段重放。S1/S2 未通过前，不调 CBF 安全阈值、不训练新 JEPA、不扩大 40/60 集、不打开 locked test。
