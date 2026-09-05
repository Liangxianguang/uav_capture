# 无人机集群对抗围捕安全增强系统
# 下一步主 TODO 与目标计划书

**版本：** 2026-09-05 / v21-master-plan
**执行目录：** `D:\\uav-capture\\uav_capture`
**硬件：** NVIDIA GeForce RTX 5050
**系统路线：** Interaction-aware Action-conditioned JEPA + Reliability Ledger + Joint CBF-QP + Rolling Horizon
**当前阶段：** `development_only=true`；`locked_test_opened=false`
**唯一主指标：** `safe_capture`
**诊断指标：** `mean_capture_time`、Transit、最小净空、CBF 修正量、fallback/abort、控制周期延迟

> 这是一份可直接执行的工作计划，不是新的实验结果。`95%` 不作为硬目标；任何平均捕获时间改善都不能抵消 `safe_capture` 下降、CBF 失败、controlled abort 或安全合同违规。

## 1. 最终目标

构建一条可审计的安全闭环：

```text
多机观测/通信历史
  -> immutable BeliefState
  -> 传统规划器生成 K=5 个动力学可行 action chunks
  -> action-conditioned interaction-aware JEPA 反事实评价
  -> immutable Reliability Ledger 可信度校验/拒答
  -> safety-first ranker + nominal anchor + abstention/hysteresis
  -> Joint CBF-QP 统一安全过滤
  -> 只执行 action chunk 第 1 步
  -> 重新观测、重规划、重过滤
```

系统边界必须始终保持：

- JEPA 只评价候选轨迹，不能生成或直接执行最终动作。
- Reliability Ledger 只读，负责 OOD、stale、non-finite、低信用和 provenance mismatch 的路由。
- nominal、候选、safe-hold 和 fallback 全部经过同一个 Joint CBF-QP。
- Rolling horizon 每周期只执行第 1 步，禁止第 2/3 步脱离新观测 open-loop 执行。
- 所有安全结论以 episode-level `safe_capture` 为准。

## 2. 当前证据和限制

### 2.1 V20 三 seed 基线

| seed | safe capture | collision | boundary | pairwise | raw/unverified |
|---:|---:|---:|---:|---:|---:|
| 20260911 | 9/20 = 45% | 0 | 0 | 0 | 0 |
| 20260912 | 7/20 = 35% | 0 | 0 | 0 | 0 |
| 20260913 | 9/20 = 45% | 0 | 0 | 0 | 0 |
| **aggregate** | **25/60 = 41.67% +/- 5.77%** | **0** | **0** | **0** | **0** |

### 2.2 M0/M3 paired 事实

- M0：`30/60 = 50.0%`。
- M3：`25/60 = 41.7%`。
- paired delta：`-8.33 pp`，bootstrap 95% CI `[-18.33, +1.67] pp`。
- 当前结论：`prediction_signal_no_control_gain` / `useful_safety_fallback_only`，不能写成 JEPA 已经提升控制性能。

### 2.3 已完成和未完成

- WP1 non-finite prediction -> safe-hold：已完成，四类 fault 均通过，同一 CBF-QP 验证，`raw_unverified_executed=0`。
- WP2 monotonic score suite：7/7 synthetic cases 通过。
- V21 protocol 已建立：
  `configs/central_random_mixed_obstacle_s3_v5_v21_cpu_separation_gate_development_protocol.yaml`。
- V21 protocol hash：`25b17b915ccf8e7d97250c4e87520e8d8e1e7cd11857ed971037b81e92f45239`。
- WP2 真实 frozen settled replay、CPU/CUDA margin 边界审计和 candidate-separation 汇总尚未完成。
- 当前 targeted test 状态曾为 `55 passed, 1 failed`；失败是测试仍读取 `candidate_rejection_reasons`，应改为 `candidate_eligibility_reasons`。

### 2.4 当前 V21 固定参数

除非新 protocol 明确登记，不得修改：

| 参数 | 当前值 |
|---|---:|
| candidate 数量 | 5 |
| action chunk 长度 | 3 steps |
| 执行策略 | 只执行第 1 步后重规划 |
| predicted clearance floor | 0.15 m |
| score comparison quantum | 0.004 m |
| score safety band | 0.004 m |
| top-two abstention margin | 0.0015 m |
| minimum candidate separation | 0.002 m |
| CBF margin | 不变 |
| stale/OOD/ledger threshold | 不变 |

## 3. 不可变安全合同

### 3.1 `safe_capture` 定义

一个 episode 只有同时满足以下条件才成功：

1. 在 time limit 内至少一架 defender 进入 `0.80 m` capture radius；
2. 无 target、障碍物或 defender-defender collision；
3. 无 boundary/altitude violation；
4. 无 pairwise separation violation；
5. 无 CBF infeasible、timeout、unverified action 或 controlled abort。

`controlled_abort` 必须计入失败分母，并单独报告；不能改写成普通 timeout，也不能删除失败 episode。

### 3.2 固定回退链

```text
separation-preserving safe-hold
  -> verified nominal through the same Joint CBF-QP
  -> controlled_abort
```

任何 raw/unverified action 都禁止进入 executor。`raw_unverified_executed=0` 是硬门，不是优化目标。

### 3.3 数据边界

- 在线输入只能来自 defender 状态、target belief、通信/观测历史、障碍/边界几何、动作历史和时间戳年龄。
- target truth 只能用于离线 settled label 和 episode 结算，字段必须标记 `offline_only=true`。
- train、calibration、development、locked 的 episode、layout 和运动 seed 必须隔离。
- development 失败片段不能原样回灌旧训练 archive；重训必须创建新 archive、checkpoint、ledger 和 protocol。

## 4. 立即执行队列（今天先做）

### Step 1：修复测试语义，不改变运行时逻辑

- [ ] 在 `tests/test_jepa_safe_capture_candidates.py` 中，将 separation gate 的断言从 `candidate_rejection_reasons` 改为 `candidate_eligibility_reasons`。
- [ ] 保留 `candidate_rejection_reasons` 只表示 finite/dynamics invalid；不要把 eligibility 拒绝混入其中。
- [ ] 运行 targeted tests，目标为 `56 passed`。

### Step 2：验证 V21 protocol

```powershell
$py = 'D:\\download\\anaconda3\\envs\\traj_pred_prep\\python.exe'
$env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = 'python'
& $py scripts/verify_jepa_safe_capture_protocol.py `
  --protocol configs/central_random_mixed_obstacle_s3_v5_v21_cpu_separation_gate_development_protocol.yaml `
  --development-only
```

- [ ] 记录 protocol hash、checkpoint hash、calibration hash 和环境 hash。
- [ ] 确认 `locked_test_opened=false`、`target_truth_used_only_for_offline_labels=true`。

### Step 3：运行最小安全回归

- [ ] ranker/candidate tests。
- [ ] protocol/schema tests。
- [ ] monotonic score suite tests。
- [ ] non-finite safe-hold tests。
- [ ] ledger fault-injection tests。
- [ ] CBF、latency、zero-perturbation 和 fault regression tests。
- [ ] `git diff --check`，确认没有引入非 ASCII 控制字符、临时路径或错误 hash。

立即队列没有全部通过前，不训练新 checkpoint，不扩大 episode 数，不打开 locked test。

## 5. 工作包总览

```text
WP0 证据/环境冻结
  -> WP1 non-finite safe-hold（已完成，复核）
  -> WP2 fixed-point + candidate separation + settled replay
  -> WP3 ledger/provenance/fault regression
  -> WP4 rolling-horizon + Joint CBF 长序列回归
  -> WP5 三 seed x 20 paired smoke
  -> WP6 条件性多任务安全头 + hard replay
  -> WP7 独立 40/60 集 paired development
  -> WP8 robustness + SIL/HIL readiness
  -> WP9 发布/locked-test 决策
```

## 6. WP0：证据、环境和版本冻结

### 任务

- [ ] 保存 `git rev-parse HEAD`、`git status --short`、完整 dirty-file 清单。
- [ ] 保存 Python、PyTorch、CUDA、GPU、Conda package list 和完整命令。
- [ ] 对 protocol、checkpoint、calibration archive、ledger、scene manifest、源代码 revision 生成 SHA-256 manifest。
- [ ] 新结果目录和 TensorBoard 目录必须为空；非空立即停止。
- [ ] 每个 run 写入 `development_only=true`、`locked_test_opened=false`、`online_target_truth=false`。
- [ ] 保留 `tmp/`、NPZ、checkpoint 和用户已有未提交修改，不使用 `git add .`、reset、checkout 或删除操作。

### 产物

```text
results/jepa_safe_capture_v21_master_preflight/
  preflight.json
  environment.txt
  command.txt
  input_hash_manifest.json
tensorboard/jepa_safe_capture_v21_master/preflight/
```

### 出口门

任何输入 hash、split、环境或 locked 标记不一致，统一标记 `INSUFFICIENT_EVIDENCE`，不进入 WP2。

## 7. WP1：non-finite -> safe-hold 复核

### 任务

- [x] NaN/Inf clearance、uncertainty、visibility、TTC、CBF auxiliary 统一路由至 `safe_hold`。
- [x] 所有候选 `eligible=false`，reason 为 `non_finite_prediction`。
- [x] safe-hold 和 nominal 仍经过同一个 Joint CBF-QP。
- [x] CBF 无法验证时进入 `controlled_abort`。
- [x] JSON 中非 finite 数值序列化为 `null`。
- [ ] 在本轮 targeted regression 中重新确认上述五项，避免 WP2 变更造成回归。

### 出口门

- [ ] fault case 100% 进入 safe-hold 或 controlled abort。
- [ ] `raw_unverified_executed_count == 0`。
- [ ] 每个 fault 都有可重放 trace、CBF status、fallback reason 和 TensorBoard event。

## 8. WP2：固定点排序、候选分离和 settled replay

这是下一步第一优先级。

### 8.1 代码和单测

- [ ] 固定整数/小数 comparison key；原始 float 只用于诊断。
- [ ] 覆盖 separation margin `0.0019779`、`0.0020000`、`0.0020076` 的边界测试。
- [ ] 验证 nominal candidate `0` 永远保留为 anchor。
- [ ] 非 nominal candidate 分离不足时记录 `insufficient_candidate_separation`，不得强制切换。
- [ ] 所有 candidate 记录 `candidate_separation_m`、eligibility reason、rejection reason、score key 和 selected index。

### 8.2 冻结 trace 审计

使用 V20/V21 frozen trace，不能把 settled outcome 写回训练数据：

- [ ] fixed-point candidate order。
- [ ] CPU/CUDA candidate order、selected index、abstention 和 ledger state。
- [ ] selected-not-best。
- [ ] Spearman/Kendall rank correlation，并明确 cost/progress 方向。
- [ ] top-1 safety precision。
- [ ] abstention rate。
- [ ] candidate separation 分布和 separation-gate rejection rate。
- [ ] candidate switch rate、连续 oscillation length、fallback reason counts。
- [ ] 按 observation condition、target motion、obstacle count、clearance bucket、ledger state、候选数和 CBF active set 分桶。

### 8.3 审计产物

```text
results/jepa_safe_capture_v21_wp2_candidate_separation_audit/
  summary.json
  ranking_rows.csv
  settled_replay.json
  device_comparison.json
  failure_buckets.json
  input_hash_manifest.json
  command.txt
  report.md
results/jepa_safe_capture_v21_tensorboard/wp2_candidate_separation/
docs/JEPA_SAFE_CAPTURE_WP2_CANDIDATE_SEPARATION_20260905.md
```

TensorBoard 至少记录：

```text
Config/*
Provenance/*
Gates/*
Ranking/candidate_separation
Ranking/insufficient_candidate_separation
Ranking/selected_not_best
Ranking/abstention_rate
Ranking/switch_rate
Ranking/oscillation_length
Safety/raw_unverified_executed
```

### 8.4 出口门

- [ ] CPU/CUDA 的 candidate order、selected index、abstention、ledger、CBF action 和 termination 逐字段一致。
- [ ] synthetic monotonic suite 全部通过。
- [ ] separation gate 不引入 raw/unverified action。
- [ ] ranking 仍未解决时如实写 `ranking_unresolved`，不降低 CBF margin、clearance floor 或 ledger threshold。

## 9. WP3：Reliability Ledger、校准和 fault regression

### 任务

- [ ] 若 WP2 改变 protocol/comparison quantum，为三个 checkpoint 重新生成 hash-bound ledger。
- [ ] ledger 绑定 checkpoint、protocol、calibration archive、builder、代码 revision 和环境 hash。
- [ ] 验证四类运行路径：`trusted`、`fallback_nominal`、`safe_hold`、`controlled_abort`。
- [ ] 注入 OOD、stale、non-finite、unknown horizon、provenance mismatch、uncertainty spike、message dropout、target turn。
- [ ] 确认 calibration 后 ledger 只读，运行期间不能更新 credit、threshold 或 bucket statistics。
- [ ] 检查 high-credit failure、low-credit coverage、各状态占比和条件失败率。
- [ ] 每个 fault 都经过同一个 Joint CBF-QP，并保留 verified action、solver status、slack、active constraints 和 latency。

### 出口门

- [ ] OOD/stale/non-finite/provenance fault 100% 进入预注册回退路径。
- [ ] `raw_unverified_executed=0`。
- [ ] CBF timeout/infeasible 都有 verified fallback 或 controlled abort。
- [ ] coverage 不足时输出 `insufficient_evidence`，不能写成安全成功。

## 10. WP4：Rolling horizon 和 Joint CBF 长序列回归

### 任务

- [ ] zero-perturbation 逐字段 regression，确认 JEPA 只影响 score/selection，不改变物理执行接口。
- [ ] 至少两次 100-cycle deterministic replay。
- [ ] 至少一次 500-cycle hard-context stress；资源允许时补充 1000-cycle。
- [ ] 注入 QP infeasible、solver timeout、non-finite request、通信中断、多约束同时激活、单机失效和 target motion shift。
- [ ] 验证每周期顺序：`observe -> belief -> candidates -> JEPA -> ledger -> rank -> CBF -> execute-first-step -> trace`。
- [ ] 证明 chunk 第 2/3 步不会脱离新观测执行。
- [ ] 在 RTX 5050 测量 JEPA、ledger、ranker、CBF 和总周期 p50/p95/p99。

### 记录字段

`belief_hash`、`candidate_hash`、finite mask、ledger state/reason、selected index、CBF verified、active set、slack、correction norm、fallback、latency、termination。

### 出口门

- [ ] 重复 replay 的 canonical trace hash 一致。
- [ ] collision、boundary、pairwise、raw/unverified 均为 0。
- [ ] CBF timeout/infeasible 不得执行 raw action。
- [ ] p95 在已登记的控制周期预算内；超时必须能定位到模块。

## 11. WP5：三 seed paired smoke（每变体 20 集）

只有 WP1-WP4 全通过后执行。seed 固定为 `20260911`、`20260912`、`20260913`。

| 变体 | 配置 | 用途 |
|---|---|---|
| M0 | nominal planner + CBF | frozen baseline |
| M3 | JEPA + ledger + safety-first ranker + CBF | 完整系统 |
| A1 | JEPA + CBF，去 ledger 路由 | ledger 消融 |
| A2 | JEPA + ledger + CBF，去 clearance/visibility rank | 辅助头消融 |
| A3 | JEPA + ledger，去 CBF | 仅离线风险诊断，不进入安全结论 |

### 任务

- [ ] 每个 seed 先运行 M0，生成唯一 scene manifest。
- [ ] M3/A1/A2 逐 episode 复用同一 manifest。
- [ ] 每变体每 seed 独立运行 20 episodes，独立 output/TensorBoard 目录。
- [ ] 统计 safe-capture、paired delta、improved/degraded/tied、collision、boundary、pairwise、CBF abort/fallback、raw-unverified、minimum clearance 和 latency。
- [ ] 运行 aggregate、failure index、settled ranking、ledger alignment、CBF audit、zero-perturbation 和 TensorBoard completeness audit。

### 出口门

- [ ] 安全保留变体 collision/boundary/pairwise/raw-unverified 均为 0。
- [ ] 所有 CBF failure 有显式 fallback，controlled abort 保留在分母。
- [ ] M3 平均 paired delta `>= 0 pp`，且至少 `2/3` seed 非负，才允许 WP7。
- [ ] 若未达到收益门，归档 `prediction_signal_no_control_gain`，回到 WP2 或进入 WP6；不通过重复采样掩盖问题。

## 12. WP6：条件性多任务安全头和困难片段 replay

仅当 WP2 合同和 WP3/WP4 安全门正确，但 WP5 仍无任务收益时执行。此阶段不是为了追求更大模型，而是提高安全排序信号。

### 数据和标签

- [ ] 只从 train split 新建 counterfactual archive。
- [ ] 多 horizon 标签：target relative displacement/velocity/acceleration、obstacle/inter-agent clearance lower quantile、pairwise TTC、visibility、observation-age risk、CBF intervention/correction/feasibility。
- [ ] hard replay 覆盖急转、加速度突变、遮挡/延迟、候选分离消失、CBF correction 变大、预测残差连续上升、controlled abort。
- [ ] hard-context 权重上限固定为 `8.0`，固定采样比例、去重规则和 split 隔离。
- [ ] OOD 样本用于拒答/校准审计，不伪装成正常训练样本。

### 模型约束

- [ ] 保持 interaction-aware history encoder 和 action-conditioned latent transition。
- [ ] 新增/校准 `clearance`、`visibility`、`TTC`、`CBF intervention` 等安全辅助头。
- [ ] 使用 calibrated residual、quantile 或 conformal lower bound；ranker 使用安全下界，不使用未经校准的均值。
- [ ] 加入 action sensitivity regularizer，要求同一 belief 下候选产生可测 prediction/latent separation。
- [ ] 三 seed 独立 checkpoint；旧 checkpoint 只读对照。

### 预测出口门

- [ ] 所有输出 finite。
- [ ] 至少一个预注册 horizon 优于 constant-velocity baseline。
- [ ] 安全头有非空 coverage、calibration/ECE/Brier、hard-slice recall 报告。
- [ ] candidate separation 可复现。
- [ ] 预测 gate 通过不等于 safe-capture 控制收益通过，仍需重复 WP1-WP5。

## 13. WP7：独立 40/60 集 paired development

仅 WP5 通过安全硬门和非劣收益门后执行。

- [ ] 新建独立 40 集 manifest；若执行 60 集，必须是独立 block，不能将 20 集拼接成伪扩展。
- [ ] 冻结 checkpoint、ledger、protocol、score、CBF margin、chunk length、episode seed 和 abort semantics。
- [ ] 覆盖 nominal、delayed/noisy、flee persistence、S-curve、target turn/acceleration、3--5 obstacles、高拥挤度和不同侧距。
- [ ] 报告逐 seed/aggregate safe-capture、sample SD、paired delta、bootstrap 95% CI、exact McNemar、improved/degraded/tied。
- [ ] 分开报告 collision、boundary、pairwise、CBF infeasible/timeout、controlled abort、raw-unverified、fallback、minimum clearance、latency 和 mean capture time。

允许的结论标签：

`safe_capture_improvement_candidate`、`safe_capture_noninferior`、`prediction_signal_no_control_gain`、`rejected_for_safety`、`insufficient_evidence_do_not_open_locked_test`。

## 14. WP8：Robustness、SIL/HIL 和部署准备

- [ ] observation dropout/noise。
- [ ] message delay/dropout。
- [ ] target turn/acceleration shift。
- [ ] obstacle density、队形拥挤度、单机失效。
- [ ] GPU 不可用、进程重启、watchdog、solver timeout。
- [ ] 验证所有分布外故障都进入 safe-hold、nominal-CBF 或 controlled abort。
- [ ] SIL 记录时间戳、通信年龄、执行回执、solver、watchdog 和 abort interface。
- [ ] HIL 输出仍必须经过 Joint CBF-QP；HIL 通过不等于真实飞行许可。

出口要求：安全、provenance、实时性和 paired safe-capture 证据全部齐全；否则保持 `locked_test_opened=false`。

## 15. WP9：发布和 locked-test 决策

只有以下条件全部满足才允许起草新的 locked preregistration：

- [ ] WP0-WP8 产物完整且 hash 可追溯。
- [ ] 三 seed CPU/CUDA decision、CBF action、fallback、termination 和 safe settlement 逐字段一致。
- [ ] 所有实际动作 `verified=true`，`raw_unverified_executed=0`。
- [ ] 40/60 集 development 以 episode-level `safe_capture` 通过非劣或正向门。
- [ ] 失败、controlled abort、fallback、最小净空和延迟均公开报告。
- [ ] 没有使用 locked 数据调参或回灌训练。

否则发布负结果或 `insufficient_evidence`，不打开 locked test。

## 16. 统一 TensorBoard 合同

每个 stage/seed/variant 使用独立 logdir，至少写入：

```text
Config/*
Provenance/*
Gates/*
Safety/safe_capture
Safety/collision
Safety/boundary
Safety/pairwise_violation
Safety/raw_unverified_executed
Safety/controlled_abort
Reliability/state/*
Reliability/reason/*
Ranking/candidate_separation
Ranking/selected_not_best
Ranking/abstention_rate
Ranking/switch_rate
CBF/feasible
CBF/timeout
CBF/correction_norm
Latency/cycle_p50
Latency/cycle_p95
Latency/cycle_p99
```

训练阶段另记每个预测头的 loss、finite rate、coverage、Brier/ECE、uncertainty calibration 和 hard-slice recall。

## 17. 结果命名和 Git 边界

新产物使用独立前缀，不覆盖旧结果：

```text
results/jepa_safe_capture_v21_master_preflight/
results/jepa_safe_capture_v21_wp2_candidate_separation_audit/
results/jepa_safe_capture_v21_ledger_fault_regression/
results/jepa_safe_capture_v21_rolling_replay_<case>/
results/jepa_safe_capture_v21_smoke_<variant>_seed<seed>/
results/jepa_safe_capture_v21_smoke_aggregate/
tensorboard/jepa_safe_capture_v21_master/<stage>/seed<seed>/
```

每个目录至少包含 `summary.json`、`run_metadata.json`、`command.txt`、`input_hash_manifest.json`，并写入 `development_only=true` 和 `locked_test_opened=false`。

每个工作包只选择性 stage 本阶段文件，禁止 `git add .`。建议提交顺序：

1. `fix(jepa): complete candidate separation gate tests`
2. `audit(jepa): add v21 candidate separation replay`
3. `test(jepa): audit ledger and cbf fault routing`
4. `test(jepa): verify rolling horizon safety contract`
5. `exp(jepa): run v21 three-seed paired smoke`
6. `train(jepa): add calibrated safety heads and hard replay`

每次提交前运行 targeted pytest、`git diff --check` 和 `git status --short`；只在用户明确要求时 push。

## 18. 每次实验的执行清单

### 运行前

- [ ] output/TensorBoard 目录为空。
- [ ] protocol/checkpoint/ledger/calibration/manifest hash 已保存。
- [ ] `development_only=true`、`locked_test_opened=false`。
- [ ] targeted tests 和 protocol verifier 通过。

### 运行中

- [ ] 不读取 online target truth。
- [ ] 不在线更新 ledger。
- [ ] 只执行 Joint CBF-QP `verified=true` 的 finite action 第一步。
- [ ] 每周期保存 candidate、JEPA、ledger、ranker、CBF、fallback、latency 和 provenance trace。

### 运行后

- [ ] `summary.json`、`episodes.csv`、step trace、metadata、hash manifest、command 和 TensorBoard 齐全。
- [ ] collision、boundary、pairwise、timeout/infeasible、controlled abort、fallback、raw-unverified 分开统计。
- [ ] JSON/CSV/Markdown/TensorBoard 的 episode 数和安全计数相互核对。
- [ ] 运行 deterministic replay、failure index、settled ranking、ledger alignment 和 paired aggregate audit。

## 19. 总完成定义

只有同时满足以下条件，才能称为“安全增强的 JEPA 闭环围捕系统”：

1. JEPA 是 action-conditioned interaction-aware 候选评价器，不是动作生成器。
2. Ledger 能对 non-finite、OOD、stale、低信用和 provenance mismatch 拒答并可重放。
3. 同一个 Joint CBF-QP 是所有实际动作的唯一执行入口。
4. Rolling horizon 只执行第一步，100/500-cycle replay 可重现。
5. 三 seed CPU/CUDA 的决策、安全结算、fallback 和 termination 一致。
6. paired development 以 episode-level `safe_capture` 证明非劣或如实归档无收益。
7. 所有安全失败、controlled abort、fallback、延迟和最小净空都公开报告。
8. 代码、环境、protocol、checkpoint、ledger、calibration、manifest、命令和结果均有 hash/provenance。
9. 未经明确授权，`locked_test_opened=false` 始终成立。

**下一次实际执行动作：**先修复 `candidate_eligibility_reasons` 测试断言，完成 targeted safety regression，再执行 WP2 candidate-separation audit；在 WP2-WP5 通过前，不扩大训练规模、不打开 locked test，也不把当前 JEPA 写成已经提升 `safe_capture`。
