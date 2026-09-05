# 无人机集群对抗围捕：下一阶段详细 TODO 计划书

**系统路线：** Interaction-aware Action-conditioned JEPA + Reliability Ledger + Joint CBF-QP + Rolling Horizon  
**版本：** 2026-09-05-v4-current  
**执行目录：** `D:\\uav-capture\\uav_capture`  
**硬件：** NVIDIA GeForce RTX 5050  
**阶段：** development-only；`locked_test_opened=false`  
**唯一主指标：** `safe_capture`  
**诊断指标：** `mean_capture_time`、transit、路径长度、最小净空、CBF 修正量、fallback/abort、控制周期延迟

> 这份文件从当前 v21 证据继续执行，不替代历史 V3/V4/V5 报告。95% 不是硬目标；任何捕获时间改善都不能抵消 safe-capture 下降、controlled abort 或安全合同违规。

## 1. 目标和不可变架构

### 1.1 系统目标

把 JEPA 用作候选轨迹的 action-conditioned 反事实评价器，而不是动作生成器。完整闭环固定为：

```text
多机观测/通信历史
  -> interaction-aware BeliefState
  -> 传统规划器生成 K=5 个动力学可行 action chunks
  -> action-conditioned JEPA 预测未来和安全辅助量
  -> immutable Reliability Ledger 可信度校验/拒答
  -> safety-first ranker + nominal anchor + abstention/hysteresis
  -> Joint CBF-QP 统一安全过滤
  -> 只执行 action chunk 的第 1 步
  -> 重新观测、更新 belief、重新规划
```

### 1.2 安全合同

- `safe_capture=true` 必须同时满足：在时间上限内进入 `0.80 m` capture radius；无目标/障碍/机间碰撞；无边界或高度越界；无 pairwise separation violation；无 CBF infeasible/timeout；无 `controlled_abort`；无 `raw_unverified_executed`。
- 每个候选为长度 `3` 的 action chunk，在线只执行第一个控制步；后两步不能 open-loop 执行。
- 固定候选语义：`nominal`、`intercept`、`lateral_clearance`、`formation_clearance`、`visibility_hold`。
- JEPA 只能返回预测、uncertainty、风险下界和候选排序依据，不能写入最终执行动作。
- 所有实际动作，包括 nominal、safe-hold 和 fallback，都必须由同一个 `Joint CBF-QP` 返回 `verified=true` 后执行。
- 固定回退链：`separation-preserving safe-hold -> verified nominal-CBF -> controlled_abort`。
- target truth 只允许用于离线 settled label 和 episode 结算，不能进入在线 BeliefState、JEPA、ranker 或 ledger。
- `controlled_abort` 必须计入失败分母，不能改写成普通 timeout 或删除。

## 2. 当前证据快照

### 2.1 已完成的 v20 三 seed基线

| training seed | safe capture | collision | boundary | pairwise | raw/unverified | CPU/CUDA 决定性 |
|---:|---:|---:|---:|---:|---:|---|
| `20260911` | `9/20 = 45%` | 0 | 0 | 0 | 0 | 逐字段一致 |
| `20260912` | `7/20 = 35%` | 0 | 0 | 0 | 0 | 逐字段一致 |
| `20260913` | `9/20 = 45%` | 0 | 0 | 0 | 0 | 逐字段一致 |

聚合 `safe_capture=41.67%`，sample SD `5.77%`。这证明了安全执行合同和设备决定性，不证明 JEPA 带来控制收益。

### 2.2 已完成的 v20 paired 事实

- M0：`30/60 = 50.0%`。
- M3：`25/60 = 41.7%`。
- paired delta：`-8.33 pp`，bootstrap 95% CI `[-18.33, +1.67] pp`。
- collision、boundary、pairwise、raw-unverified 均为 `0`。
- 当前分类：`prediction_signal_no_control_gain`，不能扩大到 40/60 集，不能打开 locked test。

### 2.3 v21 排序和 abstention 诊断

- WP1 ranking diagnosis、WP2 monotonic score suite 和 WP3 eligibility/abstention audit 已完成。
- WP2 单调性：`7/7` cases passed，说明人工 score contract 方向测试可通过。
- 真实 trace 仍有大量拒答/排序失配：all-ineligible 约 `60.4% / 42.0% / 21.6%`（三 seed）；score argmin 与 settled-best 约 `49.9% / 59.5% / 61.3%` 一致。
- 三 seed aggregate 的多候选决策中，recorded selected-not-best `92.4%`，score argmin selected-not-best `42.0%`；取消 abstention 的离线反事实没有带来 safe-capture 提升，不能直接关闭 abstention。
- eligibility floor 从 `0.15 m` 降到 `0.10/0.05/0.00 m` 会减少 all-ineligible，但没有稳定改善排序；不采用降低 floor 作为安全默认值。
- 最新阻断项：JEPA clearance/uncertainty/auxiliary prediction 出现 NaN/Inf 时，ranker 仍可能直接抛出异常；必须明确进入 `safe_hold`，禁止 raw/unverified action。

## 3. 进入下一阶段前的证据规则

### 3.1 允许的结论

```text
safe_capture_improvement_candidate
safe_capture_noninferior_safety_preserving
prediction_signal_no_control_gain
rejected_for_safety
insufficient_evidence_do_not_open_locked_test
```

只能以完整 episode 的 `safe_capture` 作为控制收益结论。prediction MAE、局部 settled label、单个 seed、单个候选或 `mean_capture_time` 都不能替代 episode 指标。

### 3.2 任何阶段都不能做的事

- 不降低 CBF margin、pairwise separation、stale/OOD 阈值或 ledger 最低信用。
- 不删除、隐藏或缩小 `controlled_abort` 的失败分母。
- 不把 score-argmin 离线反事实当作可部署策略。
- 不把 development/locked 失败 episode 原样回灌旧训练 archive。
- 不删除或覆盖 `tmp/`、NPZ、checkpoint、历史 result 和 TensorBoard。
- 不打开新的 locked-test split。

## 4. 工作包总览和闸门

| WP | 工作包 | 当前状态 | 进入条件 | 通过后进入 |
|---|---|---|---|---|
| 0 | 证据/环境/版本冻结 | 部分完成 | 当前 v21 结果和 dirty worktree 可追溯 | WP1 |
| 1 | non-finite JEPA 显式 safe-hold | 下一步第一优先级 | 保留现有 CBF 合同 | WP2 |
| 2 | ranker 固定点、候选分离和 replay 修复 | 阻断中 | WP1 fault gate 通过 | WP3 |
| 3 | 三 seed ledger/provenance/fault regression | 已有基础，需按新 protocol 重验 | WP2 protocol hash 固定 | WP4 |
| 4 | 100/500-cycle rolling-horizon 回归 | 待开始 | WP1-WP3 全部通过 | WP5 |
| 5 | 三 seed x M0/M3/A1/A2 x 20 smoke | 禁止提前扩大 | WP4 全部安全门通过 | WP6 |
| 6 | 多任务 JEPA + 困难片段 replay | 仅在 WP2 未解决时执行 | 明确问题不是纯 ranker bug | WP7 |
| 7 | 40/60 集 paired development | 条件执行 | WP5 safe-capture 非劣 | WP8 |
| 8 | robustness/SIL/HIL/locked 决策 | 条件执行 | WP7 和部署审计通过 | 结束 |

## 5. WP0：证据、环境和版本冻结

### TODO

- [ ] 重试推送当前已完成的 `852604c` 审计 commit；只提交明确属于本阶段的文件，不使用 `git add .`。
- [ ] 保存 `git status --short`、`git rev-parse HEAD`、Python、Torch、CUDA、GPU、Conda 包清单和完整命令行。
- [ ] 保存 protocol、checkpoint、calibration archive、ledger、scene manifest、代码 revision 的 SHA-256。
- [ ] 确认每个新 output/TensorBoard 目录为空；脚本发现非空目录必须直接停止。
- [ ] 所有 run metadata 写入 `development_only=true`、`locked_test_opened=false`、`online_target_truth=false`。
- [ ] 运行 `git diff --check`、协议 schema test、核心安全测试和 CUDA 可用性检查。

### 产物

```text
results/jepa_safe_capture_v21_current_preflight/
  preflight.json
  input_hash_manifest.json
  environment.txt
  command.txt
tensorboard/jepa_safe_capture_v21_current/preflight/
```

### 出口门

任何 split、hash、protocol、环境或 locked 标记不一致，均标记 `INSUFFICIENT_EVIDENCE` 并停止，不进入后续实验。

## 6. WP1：non-finite JEPA -> safe-hold（下一步立即执行）

### 6.1 代码改动

- [ ] 在 `src/encirclement3d/jepa_safe_capture_ranker.py` 中将 clearance、uncertainty、visibility、TTC、CBF-risk 和 auxiliary head 的 NaN/Inf 统一路由为：
  - `execution_mode="safe_hold"`；
  - `fallback_reason="non_finite_prediction"`；
  - 所有候选 `eligible=false`；
  - 不执行 raw/unverified desired action。
- [ ] 保留原始 trace 字段，但 JSON 中非 finite prediction 序列化为 `null`；同时记录 head 名称、候选索引、horizon、ledger state、reason code 和输入 provenance hash。
- [ ] 确认 evaluator 对 `execution_mode="safe_hold"` 复用同一个 Joint CBF-QP separation-preserving hold 路径；不能建立绕过 CBF 的特殊分支。
- [ ] safe-hold 或 nominal 无法被 CBF 验证时，只能 `controlled_abort`，并保留失败记录。

### 6.2 fault-injection 测试

- [ ] 注入 NaN clearance prediction。
- [ ] 注入 Inf uncertainty。
- [ ] 注入 NaN visibility/TTC/CBF auxiliary head。
- [ ] 检查 `fallback_reason`、`execution_mode`、`eligible_mask`、`selected_index` 和 `raw_unverified_executed`。
- [ ] 检查 CBF status、active constraints、slack、correction norm、latency 和终止原因均可追溯。
- [ ] 添加单测：non-finite prediction 不能抛出未处理异常，不能返回可执行 raw action。

### 6.3 WP1 出口门

- [ ] NaN/Inf fault 全部进入 safe-hold 或后续 controlled abort。
- [ ] `raw_unverified_executed_count == 0`。
- [ ] 所有实际动作的 `verified=true` 来自同一 Joint CBF-QP。
- [ ] fault audit 结果和 TensorBoard event 可读取，`development_only=true`。

推荐产物：

```text
results/jepa_safe_capture_v21_nonfinite_safe_hold_fault_audit/
tensorboard/jepa_safe_capture_v21_current/wp1_nonfinite_safe_hold/
docs/JEPA_SAFE_CAPTURE_WP1_NONFINITE_SAFE_HOLD_20260905.md
```

## 7. WP2：排序、固定点和候选分离修复

### 7.1 先修决定性，不先调收益

- [ ] 固定 `score_comparison_quantum_m`、tie tolerance、abstention margin、minimum hold 和 hysteresis，并将其写入新的 protocol hash。
- [ ] 用固定小数/整数 comparison key 做跨 CPU/CUDA 的候选比较；原始 float 仅保留为诊断字段。
- [ ] 添加 margin 边界测试，覆盖 `0.0019779`、`0.0020000`、`0.0020076` 等已观察值，确保同一输入走同一路径。
- [ ] 保留 nominal anchor；候选 top-two separation 不足时输出 `insufficient_candidate_separation`，不能强行切换。
- [ ] 为每个候选记录未归一化值及单位：task progress、calibrated clearance lower bound、pairwise TTC、visibility/age risk、CBF correction cost、action-change cost、uncertainty、ledger credit。

### 7.2 词典序合同

固定比较顺序为：

```text
finite/shape/reachability
  -> conservative clearance/TTC lower bound
  -> ledger state and evidence coverage
  -> visibility/observation-age risk
  -> CBF intervention/correction cost
  -> task progress
  -> action-change and nominal-anchor tie break
```

任务进展分数不能抬高安全下界不足、不可达、OOD、stale 或 non-finite 候选。预测安全量只能用于候选资格和排序，不能替代 CBF 几何证明。

### 7.3 WP2 审计

- [ ] 重跑已通过的 7-case monotonic score suite。
- [ ] 重跑三 seed frozen settled rows；报告 selected-not-best、Spearman/Kendall、top-1 safety precision、abstention rate、candidate separation、switch rate 和 oscillation length。
- [ ] 分桶：observation condition、target motion mode、obstacle count、clearance bucket、ledger state、候选数量和 CBF active set。
- [ ] 对代表性失败片段标记 `orientation_error`、`label_mismatch`、`horizon_mismatch`、`scale_mismatch` 或 `unresolved`。
- [ ] 禁止使用新 smoke 结果事后修改排序权重。

### WP2 出口门

- [ ] CPU/CUDA 的 candidate order、selected index、abstention、ledger state、CBF action 和 termination 逐字段一致。
- [ ] 三 seed selected-not-best 相对当前基线不恶化；Spearman/Kendall 不再系统性为负，或明确归档 `ranking_unresolved`。
- [ ] 单调性、tie、abstention、nominal anchor 和 candidate separation 单测全部通过。

## 8. WP3：ledger、calibration 和安全 fault regression

### TODO

- [ ] 若 WP2 修改 protocol 或 comparison quantum，为三个 checkpoint 重新生成 hash-bound ledger；不得复用旧 revision。
- [ ] ledger 必须绑定 checkpoint、protocol、calibration archive、builder、环境和代码 hash。
- [ ] 验证 `trusted`、`fallback_nominal`、`safe_hold`、`controlled_abort` 四状态转移和 reason code。
- [ ] 注入 OOD、stale、non-finite、unknown horizon、provenance mismatch、uncertainty spike、message dropout 和 target turn。
- [ ] 检查 ledger calibration 后只读，运行期间不能更新 credit、threshold 或 bucket 统计。
- [ ] 每个 fault 都通过同一 CBF-QP，并记录 verified action；异常不能执行 raw action。
- [ ] TensorBoard 记录状态占比、reason code、raw-unverified、CBF fallback、latency 和 provenance。

### WP3 出口门

- [ ] 所有 OOD/stale/non-finite/provenance fault 100% 进入规定回退。
- [ ] `raw_unverified_executed=0`，所有 CBF timeout/infeasible 都有显式 fallback 或 controlled abort。
- [ ] 低信用覆盖不足输出 `insufficient_evidence`，不能被写成安全成功。

## 9. WP4：rolling-horizon 和 Joint CBF 长序列回归

### TODO

- [ ] 对 zero-perturbation path 做逐字段 regression。
- [ ] 至少两次 100-cycle deterministic replay，再做 500-cycle hard-context stress；条件允许时补 1000-cycle。
- [ ] 验证每周期固定顺序：`observe -> belief -> candidates -> JEPA -> ledger -> rank -> CBF -> execute-first-step -> trace`。
- [ ] 验证 action chunk 的第 2/3 步绝不会脱离新观测 open-loop 执行。
- [ ] 注入 QP infeasible、solver timeout、non-finite request、通信中断、多约束同时激活、单机故障和 target motion shift。
- [ ] 记录 belief hash、候选 validity、JEPA outputs、uncertainty、ledger state、selected index、CBF active set、slack、correction、latency 和 termination。
- [ ] 测量 RTX 5050 上 JEPA、ledger、ranker、CBF 和总周期 p50/p95/p99。

### WP4 出口门

- [ ] 重复 replay 的 canonical trace hash 一致。
- [ ] collision、boundary、pairwise、raw/unverified 为 0。
- [ ] CBF timeout/infeasible 必须进入 safe-hold、verified nominal 或 controlled abort。
- [ ] p95 在预注册控制周期预算内；任何超时都能定位到模块和回退路径。

## 10. WP5：三 seed paired smoke（20 集）

只有 WP1-WP4 全部通过才允许执行。

### 变体

| 变体 | JEPA | ledger | CBF | 用途 |
|---|---:|---:|---:|---|
| M0 | off | off | on | frozen nominal baseline |
| M3 | on | on | on | 完整系统 |
| A1 | on | off | on | ledger 消融 |
| A2 | on | on | on | clearance/visibility ranking 消融 |
| A3 | on | on | off | 仅风险诊断，不支持安全结论 |

### TODO

- [ ] training seed 固定为 `20260911/20260912/20260913`。
- [ ] 每个 seed 先运行 M0，生成唯一 `scene_manifest.jsonl`；M3/A1/A2 必须逐 episode 复用该 manifest。
- [ ] 每变体每 seed 运行 20 集，使用全新 output/TensorBoard 目录。
- [ ] 统计 episode-level safe-capture、paired delta、improved/degraded/tied、collision、boundary、pairwise、CBF abort、fallback、raw-unverified、minimum clearance 和 latency。
- [ ] 使用 TensorBoard 写入 config、hash、seed、manifest、每 episode 结果、故障计数和 aggregate summary。
- [ ] 以 `safe_capture` 做唯一主门；mean capture time 只写在诊断表。

### WP5 出口门

- [ ] 安全保留变体的 collision、boundary、pairwise、raw/unverified 均为 0。
- [ ] CBF failure 均有显式 fallback；`controlled_abort` 保留在失败分母。
- [ ] M3 至少 2/3 seed paired delta 非负，平均 paired delta 不低于 `0 pp`；否则归档 `prediction_signal_no_control_gain`，回到 WP2 或 WP6。
- [ ] 同时报告 paired bootstrap CI、exact McNemar 和每 seed 结果，不能只报告 aggregate 均值。

## 11. WP6：多任务 JEPA 和困难片段 replay（条件分支）

只有 WP2 证明 ranker contract 正确、但 WP5 仍无控制收益时才执行；不得用训练替代排序审计。

### 模型和标签

- [ ] 保留目标相对位移/速度/加速度预测。
- [ ] 增加 obstacle/inter-agent clearance lower-quantile、pairwise TTC、visibility、observation-age risk、CBF intervention probability、correction magnitude 和 QP feasibility heads。
- [ ] 添加 flee persistence、target turn、S-curve、突变加速度等 interaction/motion-mode embedding。
- [ ] 使用 calibrated residual、heteroscedastic 或 ensemble disagreement 估计 uncertainty。
- [ ] 对每个 `(episode, time, agent, candidate, horizon)` 保存标签单位、coverage、finite rate 和缺失值策略。
- [ ] clearance 使用预注册 q10 lower-bound；不能用平均 MAE 代替安全下界。

### hard-segment replay

- [ ] 从失败索引提取低净空、CBF controlled abort、预测漂移、候选 separation 消失、遮挡、通信延迟/丢包、急转和拥挤片段。
- [ ] 只把去重后的上下文摘要写入新的 train archive；不把 development settled outcome 原样回灌旧 archive。
- [ ] 固定 hard-context 权重上限 `8.0`、采样比例、去重规则和 split 隔离。
- [ ] 三 seed 训练新 checkpoint；旧 checkpoint 只读对照。
- [ ] 新 checkpoint 先通过 prediction/finite/calibration gate，再生成新 ledger，再做 WP1-WP5。

### WP6 出口门

- [ ] 所有预测 finite，至少一个预注册 horizon 优于 constant-velocity baseline。
- [ ] 安全辅助 head 有非空 coverage、校准误差和分桶报告。
- [ ] 同一 belief 下五个候选的 latent/prediction separation 非零且可复现；否则停止闭环接入。
- [ ] prediction gate 通过不等于控制收益通过。

## 12. WP7：40/60 集 paired development

只有 WP5 非劣门通过后执行。

- [ ] 新建独立 40 集 manifest；若资源允许，按预注册方案再做独立 60 集 block，不能把 20 集拼接扩充。
- [ ] final block 中冻结 checkpoint、ledger、protocol、score 权重、CBF margin、chunk length、episode seed 和 abort 语义。
- [ ] 覆盖 nominal、delayed/noisy、flee persistence、S-curve、target turn/acceleration shift、3--5 obstacles、高拥挤度和不同初始侧距。
- [ ] 以 episode 为统计单位，报告 safe-capture、paired delta、sample SD、bootstrap 95% CI、McNemar、improved/degraded/tied。
- [ ] 单独报告 collision、boundary、pairwise、CBF infeasible/timeout、controlled abort、raw-unverified、fallback、minimum clearance、latency 和 mean capture time。
- [ ] 分桶报告 motion mode、visibility、observation age、clearance、ledger state 和 CBF active constraints。

## 13. WP8：robustness、SIL/HIL 和 locked 决策

- [ ] 做 observation dropout/noise、message delay/dropout、target turn、障碍密度、拥挤度、单机失效、GPU 不可用、进程重启和 watchdog stress。
- [ ] SIL 保持 RTX 5050 的真实算力预算、通信延迟、传感器噪声和控制周期；任何 HIL 控制输出仍必须经过 CBF。
- [ ] 生成 reproducibility manifest、failure index、settled ranking audit、device audit、CBF audit、paired aggregate 和 TensorBoard index。
- [ ] 对 JSON/CSV/Markdown/TensorBoard 做数值和 hash 双向一致性检查。
- [ ] 只有所有安全、provenance、实时性和 paired safe-capture 门通过后，才起草新的 locked-test preregistration；在此之前保持 `locked_test_opened=false`。

## 14. 统一产物命名和 TensorBoard 合同

新 revision 必须使用独立前缀，禁止覆盖历史证据：

```text
results/jepa_safe_capture_v21_current_preflight/
results/jepa_safe_capture_v21_nonfinite_safe_hold_fault_audit/
results/jepa_safe_capture_v21_rank_fix_settled_seed<seed>/
results/jepa_safe_capture_v21_ledger_seed<seed>/
results/jepa_safe_capture_v21_rolling_replay_<case>/
results/jepa_safe_capture_v21_smoke_<variant>_seed<seed>/
results/jepa_safe_capture_v21_smoke_aggregate/
results/jepa_safe_capture_v21_development_<variant>_seed<seed>/
tensorboard/jepa_safe_capture_v21_current/<stage>/seed<seed>/
```

每个结果目录至少包含：

```text
summary.json
run_metadata.json
command.txt
input_hash_manifest.json
development_only=true
locked_test_opened=false
```

适用时还必须包含 `episodes.csv`、`step_traces/`、`scene_manifest.jsonl`、`failure_index.*` 和 TensorBoard event。TensorBoard 至少写入 `Config/*`、`Provenance/*`、`Gates/*`、safe-capture、所有安全计数、fallback/abort、latency 和每 seed 统计。

## 15. Git 提交边界和时间盒

每个工作包只提交本阶段源代码、测试、配置和报告；不提交用户无关 dirty 文件，不使用全量 staging。

| 时间盒 | 任务 | 独立提交建议 | 出口 |
|---|---|---|---|
| T0 | WP0 证据冻结、推送未发布审计 | `chore(jepa): freeze v21 current evidence` | preflight/hash 完整 |
| T1 | WP1 non-finite safe-hold | `fix(jepa): route non-finite predictions to safe hold` | fault gate 全通过 |
| T2 | WP2 ranker fixed-point/separation | `fix(jepa): make ranking deterministic and safety-first` | CPU/CUDA + settled replay |
| T3 | WP3 ledger/fault regression | `test(jepa): audit v21 ledger and cbf fallbacks` | raw-unverified 为 0 |
| T4 | WP4 rolling replay | `test(jepa): verify rolling-horizon safety contract` | 100/500-cycle 通过 |
| T5 | WP5 三 seed smoke | `exp(jepa): run v21 paired smoke` | safe-capture 非劣或如实归档负结果 |
| T6 | WP6 新模型/数据（若需要） | `train(jepa): add calibrated safety heads` | prediction gate 通过 |

每个 commit 前运行：

```powershell
& $py -m pytest -q <本阶段 targeted tests>
git diff --check
git status --short
```

push 失败时保留本地 commit 和错误日志，稍后重试；不能为了推送而重写历史或覆盖用户改动。

## 16. 总完成定义

只有同时满足以下条件，才能把系统描述为“安全增强的 JEPA 闭环围捕系统”：

1. JEPA 只评价候选轨迹；ledger 能对 OOD、stale、non-finite、低信用和 provenance mismatch 明确拒答。
2. 所有实际动作都经过同一个 Joint CBF-QP，`raw_unverified_executed=0`。
3. rolling horizon 每周期只执行第一步，并能在 100/500-cycle replay 中重现。
4. 三 seed CPU/CUDA 的决策、CBF action、fallback、termination 和安全结算逐字段一致。
5. 三 seed paired smoke 以 episode-level safe-capture 证明 non-inferiority，或明确归档 `prediction_signal_no_control_gain`。
6. 所有安全失败、controlled abort、fallback、延迟和最小净空都单独报告。
7. 代码、环境、协议、checkpoint、ledger、calibration、manifest、命令和结果都有 hash/provenance，并同步写入 TensorBoard。
8. 未经明确授权，`locked_test_opened=false` 始终不变。

**当前执行结论：** 先做 WP1 的 non-finite -> safe-hold 修复，再做 WP2 的固定点/候选分离和 settled replay；在这些门通过前，不训练更大的模型、不扩大 episode 数、不打开 locked test。当前证据支持“安全执行基础设施已成立”，尚不支持“JEPA 已改善 safe-capture”。
