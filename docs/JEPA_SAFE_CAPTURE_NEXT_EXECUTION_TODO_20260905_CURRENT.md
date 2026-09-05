# 无人机集群对抗围捕安全增强闭环
# 下一步详细 TODO 与目标计划书（当前执行版）

**版本：** 2026-09-05 / V21-current

**执行目录：** `D:\\uav-capture\\uav_capture`

**硬件：** NVIDIA GeForce RTX 5050

**系统目标：** `Interaction-aware Action-conditioned JEPA + Reliability Ledger + Joint CBF-QP + Rolling Horizon`

**实验边界：** `development_only=true`，`locked_test_opened=false`

**主指标：** episode-level `safe_capture`

**诊断指标：** `mean_capture_time`、Transit、路径长度、最小净空、CBF 修正量、候选切换率、fallback/abort 数量和周期延迟

> 本计划不是新的实验结果。`mean_capture_time` 不作为优化目标，也不能抵消安全捕获失败。绝对 `95%` 不是硬目标；任何变体都必须先满足安全合同，再讨论任务性收益。

## 1. 目标和完成定义

最终交付是一条可以拒答、回退、重放和审计的闭环，而不是一个直接输出动作的世界模型：

```text
多机观测/通信历史
  -> immutable BeliefState
  -> 传统规划器生成 K=5 动力学可行候选 action chunks
  -> action-conditioned interaction-aware JEPA 反事实评价
  -> immutable Reliability Ledger 可信度校验
  -> safety-first ranker + nominal anchor + hysteresis
  -> 同一个 Joint CBF-QP
  -> 只执行 action chunk 第 1 步
  -> 重新观测、更新 belief、重新规划
```

完成定义：

- JEPA 只评价候选轨迹，不生成最终控制动作。
- Reliability Ledger 在 OOD、stale、non-finite、低信用、未知 horizon 或 provenance 不一致时拒答。
- nominal、JEPA selected、safe-hold、fallback 都经过同一个 Joint CBF-QP。
- 执行器只接受 `verified=true` 且 finite 的动作；`raw_unverified_executed=0`。
- 每周期只执行候选块的第一步，然后重新规划。
- 三个训练 seed 的 CPU/CUDA 离散决策、CBF 状态、执行动作和 episode 结算可确定性重放。
- 所有结论绑定 protocol、checkpoint、calibration、scene manifest、代码和环境 hash。

## 2. 不可变安全合同

### 2.1 `safe_capture` 定义

一个 episode 只有同时满足以下条件才记为 `safe_capture=true`：

1. 在时间上限内至少一台 defender 进入目标 `0.80 m` capture radius。
2. 无 target、障碍物或 defender-defender collision。
3. 无 boundary、altitude 或 pairwise separation violation。
4. 无 CBF infeasible、CBF timeout、unverified action 或 `controlled_abort`。
5. 所有执行动作都来自 Joint CBF-QP，且数值 finite。

`controlled_abort` 必须计入失败分母，并与普通超时、CBF infeasible、CBF timeout 分开统计。

### 2.2 固定候选和执行语义

- `K=5`：`nominal`、`intercept`、`lateral_clearance`、`formation_clearance`、`visibility_hold`。
- 每个候选为长度 `3` 的 desired-action chunk；在线只执行第 `1` 步。
- 候选进入 JEPA 前必须通过 finite、shape、speed、acceleration、slew 和 reachability 检查。
- `nominal` 永远保留为 anchor；候选分离不足时必须 abstain 或回退，不强迫选择近似等价候选。
- 预测净空、TTC、可见性和 CBF 风险只用于资格判断和排序，不能替代几何 CBF 安全证明。

### 2.3 固定回退链

```text
separation-preserving safe-hold
  -> verified nominal through the same Joint CBF-QP
  -> controlled_abort
```

任何预测或 ledger 故障都禁止执行 raw desired action。不得通过降低 CBF margin、放宽 stale 阈值、关闭 abstention 或删除失败分母获得表面提升。

## 3. 当前状态和证据边界

### 3.1 已完成

- **WP1 non-finite safe-hold：** NaN/Inf prediction、uncertainty 和 auxiliary head 会进入显式 `safe_hold`，reason code 固定，仍经 Joint CBF-QP，raw action 为零执行。权威目录：`results/jepa_safe_capture_v21_nonfinite_safe_hold_fault_audit_v2/`。
- **WP2 V21 candidate separation：** 当前 protocol SHA-256 为 `278623ceb7185a6c3ce23246e8a28693f025a2977fad95059ae5b0df9a03b014`；冻结 trace `3235` 行；CPU/CUDA decision difference `0`；`raw_unverified=0`；aggregate top-1 safety precision 约 `96.8%`。
- **WP2 相关测试：** 本轮安全回归 `75 passed`，candidate-separation audit `6 passed`，合计 `81 passed`。
- **WP3 hash-bound ledger：** 三 checkpoint 的外部 provenance、篡改拒绝、immutability 和 fault matrix 已通过；详见 `docs/JEPA_SAFE_CAPTURE_WP3_HASH_BOUND_LEDGER_20260905.md`。

### 3.2 尚未证明

- V21 separation gate 尚未消除 settled ranking mismatch，不能宣称控制收益。
- 尚未完成 100/500-cycle rolling-horizon 重放和 RTX 5050 延迟门。
- 尚未允许新的三 seed paired smoke、40/60 集 development 或 locked test。

当前可用结论只能是：**安全执行基础设施和确定性审计证据正在形成，JEPA 尚未证明 episode-level safe-capture 提升。**

## 4. 总体执行顺序

```text
WP0 证据/环境冻结（持续维护）
  -> WP3 完整 hash-bound ledger fault audit
  -> WP4 rolling-horizon + Joint CBF 长序列回归
  -> WP5 三 seed x M0/M3/A1/A2 x 20 集 paired smoke
  -> 条件 WP6 多任务安全头 + hard replay
  -> WP7 三 seed x 40/60 集 development
  -> WP8 robustness/SIL/HIL readiness
  -> 新 locked-test preregistration（需明确授权）
```

任何阶段的安全硬门、hash、provenance 或 deterministic replay 失败，都停止扩大样本，保留失败证据并创建新的 protocol revision。

## 5. WP0：证据、环境和版本冻结

**目标：** 保证本轮实验不会混用 V20/V21、旧 ledger、tmp archive 或用户已有 dirty worktree。

### TODO

- [ ] 保存 `git status --short --branch`、当前 revision 和 `git diff --check`；不撤销用户已有修改。
- [ ] 记录 Python、PyTorch、CUDA、GPU、Conda package list、`pip freeze` 和完整 PowerShell 命令。
- [ ] 对 V21 protocol、三个 checkpoint、calibration archive、ledger、scene manifest、关键源码生成 SHA-256 manifest。
- [ ] 每个新阶段使用全新的空 `results/` 与 TensorBoard 目录；非空目录拒绝覆盖。
- [ ] 写入 `development_only=true`、`locked_test_opened=false`、`target_truth_online=false`。
- [ ] 运行 protocol verifier、CBF tests、ledger tests、device replay tests、ranking tests。

### 产物和出口门

产物：`preflight.json`、`environment.txt`、`command.txt`、`input_hash_manifest.json`、TensorBoard `Config/*` 与 `Provenance/*`。

出口：任何 split、hash、设备、数据来源或 locked 状态不一致，标记 `INSUFFICIENT_EVIDENCE`，不得运行 episode。

## 6. WP3：三个 checkpoint 的完整 Reliability Ledger 审计

**状态：已通过。** 三个 ledger 已与 protocol、checkpoint、calibration archive 和 builder provenance 绑定；完整证据见 WP3 报告。该阶段仍不构成 safe-capture 控制收益结论。

### 固定输入

- checkpoint：`results/jepa_safe_capture_v11_hard_replay_seed20260911/checkpoint.pt`
- checkpoint：`results/jepa_safe_capture_v11_hard_replay_seed20260912/checkpoint.pt`
- checkpoint：`results/jepa_safe_capture_v11_hard_replay_seed20260913/checkpoint.pt`
- protocol：`configs/central_random_mixed_obstacle_s3_v5_v21_cpu_separation_gate_development_protocol.yaml`
- calibration archive：protocol 中冻结的 calibration NPZ 和 metadata。

### TODO

- [x] 为三个 checkpoint 分别生成 V21 hash-bound ledger；未复用旧 `_r2` 或不同 protocol 的 ledger。
- [x] 校验 ledger 内的 checkpoint SHA-256、protocol SHA-256、calibration archive SHA-256、builder/source revision。
- [x] 验证 calibration 后 ledger 文件只读，运行 fault audit 前后文件 hash 完全相同。
- [x] 注入并逐 case 审计：OOD、stale observation、non-finite context、unknown horizon、uncertainty spike、joint TTC/CBF risk 和 tampered provenance。
- [x] 验证安全状态和固定 reason code；故障请求全部进入 `safe_hold`。
- [x] 记录 provenance/hash 证据；缺失或篡改 provenance 被拒绝。
- [x] 验证 fault audit 中 `raw_unverified_executed=0`；闭环 CBF 执行安全由独立 CBF audit 覆盖。
- [x] 输出 JSON、Markdown、hash manifest 和 TensorBoard。

### 建议入口

```powershell
$py = 'D:\download\anaconda3\envs\traj_pred_prep\python.exe'
$env:PYTHONPATH = "$PWD\src;$PWD\scripts"
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = 'python'

& $py scripts/audit_jepa_safe_capture_v5_temporal_ledger.py --help
& $py scripts/audit_jepa_safe_capture_v5_ledger_alignment.py --help
```

实际运行时，每个 seed 使用独立的 `results/jepa_safe_capture_v21_ledger_seed<seed>/` 和 `results/jepa_safe_capture_v21_tensorboard/wp3_ledger_seed<seed>/`，不得覆盖旧输出。

### WP3 出口门（已通过）

- [x] 所有指定 fault 都进入预期安全路径。
- [x] hash-bound ledger 三 seed 全部通过，且 audit 前后 ledger hash 不变。
- [x] `raw_unverified_executed=0`，所有 fault decision 不执行 raw action。
- [x] OOD/stale/non-finite/provenance mismatch 不得进入 `trusted`。
- [x] TensorBoard 和 hash manifest 完整；该阶段不声称 safe-capture 收益。

WP3 已通过，可以进入 WP4；在 WP4 通过前仍不运行 paired smoke，也不训练新模型。

## 7. WP4：Rolling Horizon 与 Joint CBF 长序列回归

**状态：已通过（age-fix revision `bca003a`）。** CPU 与 RTX 5050 CUDA 各完成两次 20 集 replay；每次 1746 control cycles，逐字段 replay difference 为 0，cycle p95 分别为 14.52 ms 和 23.84 ms。详细证据见 `docs/JEPA_SAFE_CAPTURE_WP4_AGEFIX_ROLLING_AUDIT_20260906.md`。

**目标：** 证明候选评价器不会因长 rollout 漂移而绕过安全执行合同。

### 回归层级

1. **Zero-perturbation identity replay：** 相同 belief、候选、desired action 和随机状态，逐字段比较 selected index、ledger state、CBF status、verified action、termination 和 trace hash。
2. **100-cycle deterministic replay：** 同一输入运行两次，逐周期比较 decision、fallback、CBF active set、action 和 episode summary。
3. **500-cycle hard-context replay：** 覆盖低净空、候选分离消失、目标急转、观测陈旧、通信 dropout 和拥挤场景。
4. **故障注入：** CBF timeout、QP infeasible、non-finite desired action、异常状态、通信中断、单机故障、GPU 不可用和 watchdog 超时。

### TODO

- [x] 确认每个周期只执行第一个 action step；不得执行完整三步 chunk。
- [x] 验证 JEPA 只能改变候选排序，不能改变 CBF constraints、margin 或执行器入口。
- [x] 保存 canonical trace hash、episode trace、failure index 和 fault matrix。
- [x] 统计候选 switch rate、oscillation length、abstention、fallback、CBF correction norm、timeout/infeasible。
- [x] 在 RTX 5050 上测量 JEPA、ledger、ranker、CBF 和总控制周期的 p50/p95/p99 latency。
- [x] 对每个 solver failure 验证 `safe-hold -> verified nominal -> controlled_abort` 顺序。

### WP4 出口门

- [x] 双次 replay canonical trace hash 一致。
- [x] collision、boundary、pairwise、`raw_unverified` 均为 `0`。
- [x] timeout/infeasible/non-finite 均有显式 verified fallback 或 controlled abort。
- [x] 所有长序列 trace 完整，episode 统计与 JSON/CSV/TensorBoard 一致。
- [x] p95 延迟在预注册控制周期预算内；不得通过删除安全检查降低延迟。

## 8. WP5：三 seed paired smoke（每变体 20 集）

只有 WP3 和 WP4 全部通过后执行。固定训练 seed：`20260911`、`20260912`、`20260913`。

### 变体

| 变体 | JEPA | Ledger | CBF | 用途 |
|---|---:|---:|---:|---|
| M0 | off | off | on | nominal + Joint CBF baseline |
| M3 | on | on | on | 完整候选评价闭环 |
| A1 | on | off | on | ledger 消融/故障路由对照 |
| A2 | on | on | on | 去 clearance/visibility ranking 的辅助头消融 |
| A3 | on | on | off | 仅作 raw-risk 诊断，不进入安全主结论 |

### TODO

- [ ] 每个 seed 先运行 M0 生成唯一 paired scene manifest。
- [ ] M3/A1/A2 逐 episode 复用该 manifest；不得更换场景后比较。
- [ ] 每个 variant/seed 建立独立结果和 TensorBoard 目录。
- [ ] 统计 episode-level `safe_capture`、paired delta、improved/degraded/tied、collision、boundary、pairwise、CBF abort、fallback、raw-unverified、minimum clearance 和 latency。
- [ ] 运行 aggregate、failure index、settled ranking、ledger alignment、CBF safety 和 TensorBoard audit。

### smoke 判定

- 任一安全硬门失败：立即停止该变体，标记 `BLOCKED_BY_SAFETY`。
- M3 只有在至少 `2/3` seed paired delta 非负且 pooled delta 不为负时，才允许进入更大 development block。
- 若安全硬门通过但 M3 无任务性收益，归档为 `prediction_signal_no_control_gain` 或 `safe_fallback`，不强行训练或扩大样本。
- smoke 通过不等于正式提升，也不打开 locked test。

## 9. WP6：条件性多任务安全头与困难片段重放

**触发条件：** WP2/WP3/WP4 证明接口和安全合同正确，但 WP5 仍无 safe-capture 非劣或 ranking mismatch 明显。

### 训练目标

在保留目标相对位移预测的基础上，增加：

- target relative velocity/acceleration；
- obstacle/inter-agent clearance lower quantile；
- pairwise TTC；
- visibility 和 observation-age risk；
- CBF intervention probability、correction magnitude、QP feasibility risk；
- multi-horizon uncertainty/candidate disagreement。

### 数据和训练合同

- [ ] 只从 train split 建立新的 counterfactual archive；不得把 development settled outcome 原样回灌旧 archive。
- [ ] hard replay 覆盖 target turn、速度突变、flee persistence、S-curve、遮挡、通信延迟/丢包、低净空、高拥挤、候选分离消失和 controlled abort。
- [ ] 固定 hard-context 权重上限 `8.0`、采样比例、去重规则和 split 隔离。
- [ ] 三 seed 独立训练新 checkpoint；旧 checkpoint、旧 ledger 和旧结果只读。
- [ ] 新 checkpoint 先通过 finite、held-out prediction、辅助头 coverage/calibration 和 candidate separation gate，再生成新 ledger。
- [ ] 新 checkpoint 重新执行 WP3、WP4 和 WP5，不得直接替换主结果。

### WP6 出口门

- [ ] 所有 head finite，至少一个预注册 horizon 优于 constant-velocity baseline。
- [ ] clearance/TTC/visibility/CBF 头有独立校准证据和非空 coverage。
- [ ] 同一 belief 下候选之间有可复现的非零 prediction separation。
- [ ] prediction gate 通过不能单独写成 safe-capture 提升。

## 10. WP7：40/60 集 paired development

只有 WP5 smoke 通过后执行。

- [ ] 建立全新的 40 集 paired manifest；如资源允许，另建独立 60 集 block，不把 20 集拼成伪扩展。
- [ ] 冻结 checkpoint、ledger、protocol、score weights、CBF margin、chunk length、episode seeds 和 abort semantics。
- [ ] 覆盖 nominal、delayed/noisy、flee persistence、S-curve、target turn/acceleration、3--5 obstacles、拥挤度和不同初始侧距。
- [ ] 以 episode 为统计单位报告逐 seed/pooled `safe_capture`、sample SD、paired delta、bootstrap 95% CI、exact McNemar、improved/degraded/tied。
- [ ] 将 collision、boundary、pairwise、CBF infeasible/timeout、controlled abort、fallback、raw-unverified、minimum clearance、latency、mean capture time 分列报告。
- [ ] 按 motion mode、visibility、observation age、clearance、ledger state 和 CBF active set 分桶。

### 预定义结论标签

| 标签 | 条件 |
|---|---|
| `safe_capture_improvement_candidate` | 安全硬门通过，至少 2/3 seed 非负，pooled paired delta 不为负 |
| `safe_capture_noninferior` | 安全硬门通过，任务指标不劣但未证明提升 |
| `safe_fallback` | 主要价值是降低未验证执行或稳定故障回退 |
| `prediction_signal_no_control_gain` | 预测/校准改善，但闭环 safe-capture 无收益 |
| `rejected_for_safety` | 任一安全硬门或 provenance 合同失败 |
| `insufficient_evidence_do_not_open_locked_test` | 样本、coverage、replay 或 hash 证据不足 |

## 11. WP8：鲁棒性、SIL/HIL 与 locked 决策

- [ ] 注入 observation dropout/noise、message delay/dropout、target turn、障碍密度变化、拥挤度变化、单机故障、传感器冻结、GPU 不可用、进程重启和 watchdog stress。
- [ ] 做 100/500/1000-cycle long replay，确认故障不会使 raw action 穿透执行器。
- [ ] SIL 验证时间戳、通信年龄、执行回执、solver、watchdog 和 abort 接口；不把 SIL 结果写成真实飞行性能。
- [ ] HIL 仍使用同一个 CBF 入口；HIL 通过不等于真实飞行许可。
- [ ] 只有安全硬门、provenance、device determinism、ledger fault matrix、rolling replay 和 paired statistics 全部通过后，才起草新的 locked preregistration，并等待明确授权。

## 12. 每次运行统一清单

### 运行前

- [ ] `development_only=true`，`locked_test_opened=false`，非 locked split。
- [ ] output/TensorBoard 目录为空。
- [ ] protocol/checkpoint/ledger/calibration/scene/code/environment SHA-256 已保存。
- [ ] RTX 5050/CUDA/PyTorch/Conda 信息已写入 metadata。
- [ ] targeted tests、`git diff --check`、schema/preflight 已通过。

### 运行中

- [ ] 只执行 Joint CBF-QP 返回的 finite + verified action 第一步。
- [ ] 不读取 online target truth，不在线更新 ledger。
- [ ] 每周期写入 belief hash、candidate hash、JEPA finite mask、ledger state、selected index、CBF status、fallback 和 latency。
- [ ] 不删除失败 episode、controlled abort、non-finite trace 或 solver failure。

### 运行后

- [ ] `summary.json`、`episodes.csv`、`step_traces/`、`run_metadata.json`、`input_hash_manifest.json`、命令记录和 TensorBoard 齐全。
- [ ] JSON/CSV/Markdown/TensorBoard 的 episode 数和安全计数一致。
- [ ] 运行 deterministic replay、failure index、ledger alignment、settled ranking 和 paired aggregate。
- [ ] 报告中明确区分主指标、诊断指标和局部 settled counterfactual，禁止混写。

## 13. TensorBoard 最低记录合同

每个 stage/seed/variant 使用独立 logdir，至少包含：

```text
Config/*
Provenance/*
Safety/safe_capture
Safety/collision
Safety/boundary
Safety/pairwise_violation
Safety/raw_unverified_executed
Safety/controlled_abort
Reliability/state
Reliability/fallback_reason
Reliability/credit
Ranking/selected_not_best
Ranking/abstention_rate
Ranking/candidate_separation
CBF/feasible
CBF/timeout
CBF/infeasible
CBF/correction_norm
Latency/cycle_p50
Latency/cycle_p95
Latency/cycle_p99
```

训练阶段额外记录每个预测头的 loss、finite rate、coverage、calibration error、Brier/ECE 和 uncertainty 分桶。

## 14. 结果目录和 Git 边界

后续只使用新前缀，不覆盖 V20/V21 历史结果：

```text
results/jepa_safe_capture_v21_current_preflight/
results/jepa_safe_capture_v21_ledger_seed<seed>/
results/jepa_safe_capture_v21_tensorboard/wp3_ledger_seed<seed>/
results/jepa_safe_capture_v21_rolling_replay_<case>/
results/jepa_safe_capture_v21_tensorboard/wp4_<case>/
results/jepa_safe_capture_v21_smoke_<variant>_seed<seed>/
results/jepa_safe_capture_v21_tensorboard/wp5_<variant>_seed<seed>/
```

建议阶段提交顺序：

1. `test(jepa): complete v21 hash-bound ledger audit`
2. `test(jepa): verify v21 rolling-horizon safety contract`
3. `exp(jepa): run v21 paired smoke`
4. `train(jepa): add calibrated safety heads and hard replay`
5. `exp(jepa): run v21 paired development`

每次只选择性 stage 本阶段文件；禁止 `git add .`、reset、checkout、删除 `tmp/`、NPZ、checkpoint 或用户已有 dirty 文件。

## 15. 现在立刻执行的四件事

1. 重试推送已有本地 commit `c93d8bd`；推送失败不改写或删除本地 commit。
2. ~~完成三个 checkpoint 的 V21 hash-bound ledger，并生成 WP3 fault matrix、hash manifest 和 TensorBoard。~~ **已完成。**
3. 现在运行 zero-perturbation、100-cycle 和 500-cycle rolling replay；先完成 WP4 报告和延迟表。
4. 只有 WP3/WP4 全通过，才运行三 seed、M0/M3/A1/A2、每变体 20 集 paired smoke；在此之前不训练新模型、不扩大样本、不打开 locked test。

## 16. 最终 DoD

- [ ] JEPA 是 action-conditioned interaction-aware 候选评价器，而不是动作生成器。
- [ ] Reliability Ledger 对分布外、陈旧、非 finite、低信用和 provenance mismatch 可拒答、可回退、可重放。
- [ ] Joint CBF-QP 是唯一执行入口，`raw_unverified_executed=0`。
- [ ] rolling horizon 只执行第一步并且长序列可确定性重放。
- [ ] 三 seed smoke/development 以 episode-level `safe_capture` 为主结论。
- [ ] mean capture time 只作为诊断，不作为安全成功的替代指标。
- [ ] 所有正向、负向、abstention、fallback 和 controlled abort 证据均被保留并可追溯。
- [ ] 在没有明确授权前，`locked_test_opened=false` 始终保持不变。
