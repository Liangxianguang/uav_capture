# 无人机集群对抗围捕安全增强系统：下一阶段详细 TODO 计划书

**系统路线：** Interaction-aware Action-conditioned JEPA + Reliability Ledger + Joint CBF-QP + Rolling Horizon  
**计划版本：** 2026-09-05-v3  
**实验阶段：** development-only  
**硬件：** NVIDIA GeForce RTX 5050  
**主指标：** `safe_capture`  
**安全硬门：** collision、boundary、pairwise separation、raw/unverified action 不得出现；CBF infeasible/timeout 必须显式回退且不得执行 raw  
**locked 状态：** `locked_test_opened=false`

> 本文件是下一阶段执行计划，不是新的实验结果。计划的核心原则是先证明“评价器排序正确且可拒答”，再扩大 episode 数；不能用 mean capture time、单个 seed 或 prediction MAE 掩盖 safe-capture 下降和安全失败。

## 1. 最终系统目标

实现以下安全闭环：

```text
多机观测/通信历史
  -> interaction-aware belief state
  -> 传统规划器生成动力学可行候选 action chunks
  -> action-conditioned JEPA 反事实轨迹评价
  -> reliability ledger 可信度校验/拒答
  -> 安全优先候选排序
  -> Joint CBF-QP 统一安全过滤
  -> 只执行 action chunk 的第一步
  -> 重新观测、重新规划、重新过滤
```

系统必须满足：

1. JEPA 只能评价和排序候选轨迹，不能直接生成、覆盖或执行控制动作。
2. 所有候选，包括 nominal，都经过同一个 Joint CBF-QP。
3. ledger 只在离线 calibration split 上拟合，运行期间只读并绑定 hash。
4. OOD、stale、non-finite、低信用、预测漂移或 CBF 失败时，只能走固定回退链。
5. 滚动执行只执行第一控制步，禁止一次性执行完整 action chunk。
6. `safe_capture` 只有在捕获、无碰撞、无越界、无机间净空破坏、无 CBF 失败、无 unverified action 且无 controlled abort 时才为真。

## 2. 当前证据基线

### 2.1 已完成的 v20 证据

协议：

```text
configs/central_random_mixed_obstacle_s3_v5_v20_cpu_deterministic_development_protocol.yaml
sha256 = b8a492faa9448bb0917c124908a044af6cb10813847afeedb78ce675446a2b99
git revision = 9fe8cb5be1c2727b669a0386b479cfc128e86aad
```

三 seed CPU/CUDA deterministic replay 已通过：

| seed | safe capture | control steps | decision equal | CBF equal | collision | boundary | pairwise | raw/unverified |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260911 | 9/20 = 45% | 1167 | 1167/1167 | 1167/1167 | 0 | 0 | 0 | 0/0 |
| 20260912 | 7/20 = 35% | 935 | 935/935 | 935/935 | 0 | 0 | 0 | 0/0 |
| 20260913 | 9/20 = 45% | 1133 | 1133/1133 | 1133/1133 | 0 | 0 | 0 | 0/0 |

三 seed汇总为 `41.67%`，sample SD 为 `5.77%`。该结果证明设备决定性和安全执行合同，不证明 JEPA 带来控制收益。

### 2.2 当前阻断问题

settled counterfactual ranking audit 已完成，但方向明显异常：

| seed | selected-not-best | Spearman | Kendall |
|---:|---:|---:|---:|
| 20260911 | 35.8% | -0.523 | -0.456 |
| 20260912 | 54.0% | -0.566 | -0.488 |
| 20260913 | 72.9% | -0.620 | -0.549 |

这意味着当前 CBF/回退链路安全，但 JEPA score 与 settled outcome 可能存在方向、符号、尺度或 horizon 对齐错误。该问题解决前：

- 不得扩大到 40/60 集 development block；
- 不得声称 JEPA 有 safe-capture 提升；
- 不得通过改变 CBF margin、关闭 stale/OOD 或删除 controlled abort 追逐捕获率；
- 不得打开新的 locked-test split。

## 3. 不可变安全合同

### 3.1 数据和信息边界

- 在线输入只能使用 defender 状态、target belief、观测/通信历史、障碍几何、边界、动作历史和时间戳年龄。
- target truth 只能用于 offline settled labels 和 episode 结算，不能进入在线 actor、JEPA、ranker 或 ledger。
- train、validation、calibration、development 和 locked 按 episode/layout seed 隔离。
- 失败 episode 不得直接回灌训练；重放训练必须建立新 archive、manifest 和 protocol revision。
- 每次运行保存 code、protocol、checkpoint、calibration、ledger、scene、环境和命令 hash。

### 3.2 动作和候选合同

- 固定 `K=5`：`nominal`、`intercept`、`lateral_clearance`、`formation_clearance`、`visibility_hold`。
- 固定 action chunk 为 3 个 control steps，在线只执行第 1 步。
- JEPA 前必须完成 finite、shape、速度、加速度、slew 和 reachability 预检查。
- 不可达候选不得进入 JEPA，必须写入拒绝原因。
- nominal 与所有 JEPA 候选共享 CBF margin、solver、tolerance 和 timeout。

### 3.3 固定回退链

```text
separation-preserving safe-hold
  -> frozen nominal through Joint CBF-QP
  -> controlled_abort
```

任一异常必须记录 `fallback_reason`、ledger state、solver status、active constraints、slack、correction norm、latency 和终止原因。`controlled_abort` 必须计入失败分母；CBF infeasible/timeout 可以作为被审计的故障注入或自然终止，但不得未处理或执行 raw action。

## 4. 工作包总览

| WP | 工作包 | 当前状态 | 进入条件 | 主要出口 |
|---|---|---|---|---|
| 0 | 证据归档和 provenance | 进行中 | v20 结果已存在 | 新旧 revision 可区分，输入 hash 完整 |
| 1 | v20 三 seed 设备审计正式汇总 | 待完成 | 三 seed audit 文件存在 | 正式 Markdown/JSON/TensorBoard |
| 2 | M0/M3 paired aggregate | 待完成 | paired episode 数据可读 | delta、CI、McNemar、失败统计 |
| 3 | settled ranking 方向诊断 | 阻断门 | WP2 结果和 counterfactual label | 找到或明确未找到根因 |
| 4 | score/label/horizon 修复与离线重验 | 待开始 | WP3 定位根因 | ranking 方向不再系统性反向 |
| 5 | JEPA 多任务和困难片段增强 | 待开始 | WP4 通过 | prediction/uncertainty 通过校准门 |
| 6 | reliability ledger 与 CBF fault regression | 部分完成 | WP4 或安全参数未变 | 所有异常零 raw action |
| 7 | rolling-horizon 长序列审计 | 待开始 | WP1、WP4、WP6 通过 | 100/500-cycle 确定性和实时性 |
| 8 | 三 seed paired smoke | 待开始 | WP3/WP4/WP6 通过 | M0/M3/A1/A2 可配对比较 |
| 9 | 40/60 集 development block | 禁止提前执行 | WP8 通过 | safe-capture 主比较 |
| 10 | SIL/HIL readiness 和 locked 决策 | 待开始 | WP9 和全部安全门通过 | 保持 locked 关闭或形成申请草案 |

## 5. WP0：环境、输入和 provenance 冻结

### TODO

- [ ] 保存 `git status --short`、`git rev-parse HEAD`、Conda 环境、Python、Torch、CUDA、GPU 和完整命令行。
- [ ] 计算 protocol、checkpoint、calibration archive、ledger、scene manifest 和代码 revision 的 SHA-256。
- [ ] 检查所有输出目录为空；禁止覆盖已有 result、checkpoint、NPZ 和 TensorBoard。
- [ ] 运行时强制检查 `development_only=true` 和 `locked_test_opened=false`。
- [ ] 保存当前 v20 与旧 v19 目录的映射，明确旧证据不可复用为 v20 证据。
- [ ] 提交未提交的 `aggregate_jepa_safe_capture_v20_device_replay.py` 前先运行单测和 dry-run。

### 产物

```text
results/jepa_safe_capture_v20_cpu_deterministic_preflight/
  preflight.json
  input_hash_manifest.json
  environment.txt
  command.txt
```

### 出口门

任何 hash、split、protocol 或 locked 标记不一致都必须停止，不能继续实验。

## 6. WP1：v20 三 seed 设备审计正式归档

### TODO

- [ ] 运行并审查 `scripts/aggregate_jepa_safe_capture_v20_device_replay.py`。
- [ ] 只读取以下权威目录：
  - `results/jepa_safe_capture_v20_cpu_deterministic_device_audit_seed20260911_final/`
  - `results/jepa_safe_capture_v20_cpu_deterministic_device_audit_seed20260912_final/`
  - `results/jepa_safe_capture_v20_cpu_deterministic_device_audit_seed20260913_final/`
  - `results/jepa_safe_capture_v20_cpu_deterministic_device_audit_three_seed_final/`
- [ ] 检查三 seed 的 protocol hash、checkpoint hash、episode manifest 和 git revision。
- [ ] 重新生成三 seed 汇总 Markdown，修正旧 seed11 报告仍引用 `ece7415` 的描述问题。
- [ ] 将所有 device gates 显式写入 JSON 和 TensorBoard。
- [ ] 为每个 seed 建立报告链接和 evidence index。

### 出口门

三 seed 必须全部满足 `cpu_cuda_safety_and_decision_equivalent`，且 decision、CBF、numeric、termination 和安全结算逐字段一致。该门通过只允许进入 WP2，不代表性能门通过。

## 7. WP2：M0/M3 三 seed paired aggregate

### 变体定义

- **M0：** frozen nominal + Joint CBF-QP。
- **M3：** JEPA evaluator + reliability ledger + safety-aware ranking + Joint CBF-QP。
- **A1：** JEPA + CBF，关闭 ledger，仅作漂移代价诊断。
- **A2：** JEPA + ledger + CBF，移除 clearance/visibility ranking terms。
- **A3：** raw/no-CBF，仅作独立风险诊断，不进入安全主结论。

### TODO

- [ ] 对相同 `scene_manifest`、episode index、observation schedule 和 episode seed 配对 M0/M3。
- [ ] 每个 seed 计算 safe-capture、collision、boundary、pairwise、CBF abort、fallback、raw-unverified 和 latency。
- [ ] 计算每个 episode 的 `M3 - M0` safe-capture delta。
- [ ] 统计 `improved/degraded/tied`，不能把 timestep、候选或 action chunk 当独立样本。
- [ ] 计算 seed-level mean、sample SD、paired bootstrap 95% CI 和 exact McNemar。
- [ ] 给出 controlled-abort 是否改变分母的显式审计；不得删除 abort。

### 出口门

输出必须能回答：M3 在哪些相同 episode 上优于、劣于或等于 M0，以及差异是否伴随安全错误。若 paired manifest 不完全一致，停止统计并重建输入。

## 8. WP3：settled ranking 根因诊断（当前第一阻断门）

### 必查项目

- [ ] **score orientation：** 验证 ranker 是按 score 越小越优还是越大越优，检查所有 sort、tie-break 和 nominal anchor 分支。
- [ ] **task-progress 符号：** 目标距离减少是否被奖励，距离增加是否被惩罚，检查单位是否为米。
- [ ] **clearance 项符号：** 净空越大是否导致风险越小；lower-quantile 变换是否重复取负或重复惩罚。
- [ ] **CBF-risk 项符号：** intervention probability、correction magnitude、QP infeasibility 的方向是否一致。
- [ ] **visibility 项符号：** visibility gain 与 observation-age risk 是否被混为同一项。
- [ ] **uncertainty 项：** uncertainty 越高是否必然不利，检查 calibration residual 和 ensemble disagreement 的尺度。
- [ ] **horizon 对齐：** score 使用的预测 horizon 必须与 settled local chunk horizon 一致，不能用短期 score 对比长期标签。
- [ ] **candidate action 尺度：** actor 输出、环境执行动作、settled action 和 replay action 必须使用同一单位、裁剪和归一化。
- [ ] **settled label 定义：** 明确 settled safe 的几何、碰撞、边界、pairwise、CBF feasibility 和 abort 语义。
- [ ] **selection bias：** 分别报告所有候选的 settled outcome、selected candidate outcome 和 best feasible candidate outcome。
- [ ] **episode/timestep 层级：** 说明 local settled label 只用于离线诊断，不把 1167/935/1133 个决策误写成 episode 数。

### 必做分析

- [ ] 用人工构造的单步样本验证每个 score term 的单调性。
- [ ] 对每个候选逐项输出 raw term、weighted term、fixed-point key 和 final rank。
- [ ] 计算 score 与 settled safety、progress、clearance、CBF intervention 的逐项相关性。
- [ ] 做 sign-flip、term-isolation 和 score-ablation 离线回放；所有修改仅写到新目录。
- [ ] 对 selected-not-best 的代表片段生成逐周期审计，标记 `orientation_error`、`label_mismatch`、`horizon_mismatch`、`scale_mismatch` 或 `unresolved`。

### 出口门

必须满足以下之一：

1. 找到明确根因，能在人工单调性测试和 settled replay 中复现并修复；或
2. 明确证据不足，保留 `ranking_unresolved`，不得调整权重并进入 WP5/WP8。

## 9. WP4：score、label 和 horizon 修复

### TODO

- [ ] 每次修复新建 protocol revision、calibration manifest、输出目录和 hash。
- [ ] 保持 CBF margin、stale/OOD 阈值、controlled-abort 语义和 action chunk 长度不变。
- [ ] 先在人工 synthetic monotonic suite 验证：task progress、clearance、visibility、uncertainty、CBF risk 各自方向正确。
- [ ] 在冻结的三 seed trace 上重新计算 settled ranking，不能用新 development 结果调权重。
- [ ] 报告 selected-not-best、Spearman、Kendall、top-1 safety precision/recall、top-two margin 和 switch rate。
- [ ] 比较修复前后每个 episode 的 selected candidate、CBF action、termination 和 safe-capture，不允许静默改变输入。

### 排序准入建议

```text
score(k) = - task_progress
         - visibility_gain
         + clearance_risk_lower_quantile
         + cbf_intervention_cost
         + uncertainty_penalty
         + action_change_cost
         + nominal_anchor_penalty
```

上式只是方向模板，最终形式必须由 WP3 的单位和实现证据确认，不能直接当作无条件改参授权。预测安全量只能影响排序，不能替代 CBF 的真实几何约束。

### 出口门

- 人工单调性测试全部通过；
- settled rank 不再出现三 seed 系统性反向关系；
- 预测排序改善不能以 collision、boundary、pairwise、raw action 或 abort 违规为代价；
- 若只改善 ranking correlation 而没有闭环收益，结论写为 `prediction_signal_no_control_gain`。

## 10. WP5：JEPA 多任务、交互建模和困难片段增强

只有 WP3/WP4 证明 score contract 正确后，才开始改变模型或训练数据。

### 模型输出

- [ ] 保留 target displacement 多 horizon head。
- [ ] 新增 target velocity/acceleration consistency head。
- [ ] 新增 obstacle-clearance lower-quantile head。
- [ ] 新增 inter-agent clearance 和 pairwise TTC head。
- [ ] 新增 visibility probability 和 observation-age risk head。
- [ ] 新增 CBF intervention probability、correction magnitude 和 QP feasibility head。
- [ ] 增加 target motion mode embedding：flee persistence、turn、S-curve、突变加速度。
- [ ] 使用 ensemble disagreement、heteroscedastic residual 或 calibrated residual 估计 uncertainty。
- [ ] 增加 action-conditioned contrastive/consistency loss，确认不同候选对应可辨识的未来表示。

### 数据和训练

- [ ] 构建 hard-fragment archive：低净空、遮挡、通信延迟、拥挤、急转、速度突变、high-credit failure。
- [ ] 按 episode 做 train/validation/calibration 隔离，禁止近邻 layout 跨 split。
- [ ] 为每个 `(episode, time, agent, candidate, horizon)` 保存 settled multi-task labels。
- [ ] 失败 development episode 不直接回灌；重放训练须有新协议。
- [ ] 三 seed 固定 optimizer、batch、epoch、precision、随机种子和 checkpoint 选择规则。
- [ ] TensorBoard 记录各 head loss、MAE、Brier/AUROC、coverage、uncertainty、rank consistency 和 provenance。

### 出口门

- 所有输出 finite；
- 至少一个预注册 horizon 优于 constant-velocity；
- clearance、visibility、CBF risk head 有非空标签覆盖和校准证据；
- 不能只报告预测改善，必须继续通过 WP8 的闭环安全和 paired gate。

## 11. WP6：Reliability Ledger 和安全回退回归

### ledger 合同

状态固定为：

| 状态 | 条件 | 允许动作 |
|---|---|---|
| `trusted` | bucket coverage 足够、credit 达标、uncertainty/stale 合格 | 允许 JEPA 排序 |
| `fallback_nominal` | credit 下降、candidate separation 消失、预测漂移 | nominal -> CBF |
| `safe_hold` | OOD、non-finite、过期观测、连续失败、provenance fault | safe-hold -> CBF |
| `controlled_abort` | safe-hold/nominal 均无法验证 | 终止并计失败 |

### TODO

- [ ] 三 seed 重新检查 ledger 与 checkpoint、protocol、calibration archive、clearance calibration 的 hash 绑定。
- [ ] 注入 stale observation、OOD context、non-finite score、unknown horizon、provenance mismatch、消息丢失和目标急转。
- [ ] 检查每个注入样本的状态转移、reason code、fallback action 和 CBF verification。
- [ ] 检查 ledger calibration 后只读，禁止在线更新 credit/threshold。
- [ ] 报告 high-credit failure rate、low-credit failure rate、abstention coverage 和误拒率。

### 出口门

所有 OOD/stale/non-finite/provenance fault 必须 100% 进入规定回退，`raw_unverified_executed=0`。任何一个异常样本绕过 CBF 都是 `BLOCKED_BY_SAFETY`。

## 12. WP7：Rolling-horizon 长序列和 CBF fault regression

### TODO

- [ ] 先运行至少两次 100-cycle deterministic replay，再运行 500-cycle hard stress；条件允许时补 1000-cycle。
- [ ] 注入 QP infeasible、solver timeout、non-finite request、通信中断、多约束同时激活、单机异常和 target motion shift。
- [ ] 每周期记录 belief hash、candidate validity、JEPA outputs、uncertainty、credit、selected index、CBF active set、slack、correction、latency 和 termination。
- [ ] 验证严格顺序：observe -> belief -> candidate -> reachability -> JEPA -> ledger -> rank -> CBF -> first-step execute -> trace。
- [ ] 验证不会执行完整 action chunk；chunk 长度和 first-step semantics 写入 trace。
- [ ] 测量 RTX 5050 上 JEPA、ledger、ranker、CBF 和 total cycle 的 p50/p95/p99。

### 出口门

重复长序列的 canonical trace hash 一致；collision、boundary、pairwise、raw/unverified 为 0；p95 在预注册控制周期预算内；超时始终进入 nominal CBF 或 safe-hold。

## 13. WP8：三 seed paired smoke

### TODO

- [ ] 固定 M0、M3、A1、A2；A3 仅诊断。
- [ ] 使用完全相同的 paired scene manifest、episode seeds、observation schedule、layout 和 target motion。
- [ ] 每变体每 seed 运行 20 集，所有目录全新创建。
- [ ] 运行前冻结 protocol、checkpoint、ledger、calibration 和 code hash。
- [ ] 统计 safe-capture、paired delta、improved/degraded/tied、collision、boundary、pairwise、CBF abort、fallback 和 latency。
- [ ] 按 motion、visibility、observation age、clearance、ledger state 和 active CBF constraint 分桶。

### 预注册判定

- 安全硬门：所有安全保留变体 collision、boundary、pairwise、raw/unverified 均为 0。
- non-inferiority：M3 平均 paired safe-capture delta 不低于 `0 pp`，且至少 2/3 seed 不为负。
- positive candidate：平均 paired delta 为正，且至少 2/3 seed 非负；同时报告 CI 和 McNemar，不能只报告均值。
- 失败分类：`prediction_signal_no_control_gain`、`safe_capture_noninferior_safety_preserving` 或 `rejected_for_safety`。

若 WP8 未通过，禁止进入 40/60 集；先回到 WP3、WP4 或 WP5。

## 14. WP9：40/60 集 development block

只有 WP0-WP8 全部通过才可执行。

### 设计

- [ ] 每 seed 至少 40 集；资源允许时按预注册方案增加独立 60 集 block。
- [ ] 覆盖 nominal、delayed/noisy、flee persistence、S-curve、目标急转/速度突变、3–5 障碍、高拥挤度、左右初始侧距。
- [ ] 训练 seed 固定为 `20260911/20260912/20260913`，每个 seed 使用绑定的 checkpoint 和 ledger。
- [ ] final block 中禁止调 score 权重、阈值、CBF margin、chunk length、episode seed 或 abort 语义。
- [ ] 每个 run 保存 summary、episodes.csv、step traces、scene manifest、provenance、hash 和 TensorBoard。

### 主报告

第一层：逐 seed 和 aggregate `safe_capture`。  
第二层：collision、boundary、pairwise、CBF infeasible/timeout、controlled abort、raw-unverified。  
第三层：paired delta、improved/degraded/tied、bootstrap CI、exact McNemar。  
第四层：capture time、路径、clearance、visibility、CBF correction、fallback 和 latency。

`mean_capture_time` 只能作为诊断指标，不能抵消 safe-capture 或安全硬门失败。

## 15. WP10：SIL/HIL readiness 和 locked 决策

### TODO

- [ ] 完成 100/500/1000-cycle 审计、watchdog、通信冻结、传感器冻结和 GPU 不可用手册。
- [ ] SIL 中保持真实计算预算、通信延迟、传感器噪声和控制周期。
- [ ] HIL 中确认真实飞控接口不能绕过 CBF；测试断网、传感器冻结、目标突变和单机失效。
- [ ] 生成 reproducibility manifest、failure index、settled ranking audit、CBF audit、device audit 和 paired aggregate。
- [ ] 对所有结果执行 JSON/CSV/Markdown/TensorBoard 双向一致性检查。
- [ ] 只有所有安全门、provenance 门和统计门通过后，才形成新的 locked-test preregistration 草案。

在明确授权前始终保持 `locked_test_opened=false`。HIL 通过也不等于可以直接实飞。

## 16. 统一产物命名

新 revision 使用唯一前缀，例如：

```text
results/jepa_safe_capture_v20_cpu_deterministic_preflight/
results/jepa_safe_capture_v20_cpu_deterministic_device_audit_three_seed_final/
results/jepa_safe_capture_v20_cpu_deterministic_paired_aggregate_v1/
results/jepa_safe_capture_v20_cpu_deterministic_ranking_diagnosis_v1/
results/jepa_safe_capture_v21_ranking_fix_settled_audit_seed<seed>/
results/jepa_safe_capture_v21_ledger_seed<seed>/
results/jepa_safe_capture_v21_smoke_<variant>_seed<seed>/
results/jepa_safe_capture_v21_development_<variant>_seed<seed>/
results/jepa_safe_capture_v21_tensorboard/<stage>/seed<seed>/
```

每个目录至少包含 `summary.json`、`run_metadata.json`、命令行、输入 hash、`development_only=true`、`locked_test_opened=false` 和适用的 `episodes.csv`/`step_traces/`/TensorBoard event。

## 17. RTX 5050 执行模板

```powershell
Set-Location D:\uav-capture\uav_capture
$py = 'D:\download\anaconda3\envs\traj_pred_prep\python.exe'
$env:PYTHONPATH = "$PWD\src;$PWD\scripts"
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = 'python'

& $py -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
& $py -m py_compile scripts/aggregate_jepa_safe_capture_v20_device_replay.py
& $py -m pytest -q tests/test_aggregate_jepa_safe_capture_v2_paired.py tests/test_jepa_safe_capture_v2_paired.py tests/test_jepa_safe_capture_v2_reliability.py
& $py scripts/verify_jepa_safe_capture_protocol.py --protocol configs/central_random_mixed_obstacle_s3_v5_v20_cpu_deterministic_development_protocol.yaml --development-only
```

当前不要静默切换到尚未安装的 `uav-encirclement-gpu` 环境。任何环境切换必须先导出 `conda list`、`pip freeze` 并写入 provenance。

## 18. 硬停止规则

- protocol、checkpoint、ledger、calibration、scene 或 code hash 不一致：停止并重建 preflight。
- CPU/CUDA 的 candidate order、selected index、ledger state、CBF status、action 或 termination 不一致：停止，不扩大 episode。
- settled ranking 方向未解释或 label/horizon 未对齐：停止，不调权重，不进入 final block。
- 任意 collision、boundary、pairwise violation、raw/unverified action 或未处理 CBF failure：标记 `BLOCKED_BY_SAFETY`。CBF failure 若按固定回退链处理，则记录其发生率和终止结果，不得从失败分母删除。
- 不得降低 CBF margin、关闭 OOD/stale、扩大 stale age、删除 controlled abort 或执行完整 chunk。
- 不得把单 seed、smoke、local settled label 或 prediction MAE 写成正式控制收益。
- 不得删除、覆盖或清理 `tmp/`、NPZ、历史 checkpoint 和历史 results。
- 不得访问或打开新的 locked-test split。

## 19. 推荐执行顺序和时间盒

| 时间盒 | 工作 | 交付物 |
|---|---|---|
| Day 1 | WP0 + WP1 | preflight、三 seed device audit 正式归档 |
| Day 1-2 | WP2 | M0/M3 paired aggregate 和统计审计 |
| Day 2-4 | WP3 | score/label/horizon 根因报告 |
| Day 4-6 | WP4 | 新 protocol revision、人工单调性和 settled replay |
| Day 6-10 | WP5/WP6 | 多任务 JEPA、ledger 校准、fault regression |
| Day 10-12 | WP7 | 100/500-cycle rolling replay 和 RTX 5050 latency |
| Day 12-14 | WP8 | 三 seed 20 集 paired smoke |
| Day 15+ | WP9 | 条件性 40/60 集 development block |
| 后续 | WP10 | SIL/HIL 和 locked readiness memo |

若任一出口门失败，时间表自动回退到对应工作包，不通过增加 episode 数来掩盖失败。

## 20. 完成定义

只有同时满足以下条件，才能称为“安全增强的 JEPA 闭环围捕系统”：

1. JEPA、ledger、ranker、Joint CBF-QP 和 rolling executor 均有可测试接口和逐周期 trace。
2. 三 seed CPU/CUDA replay 的离散决策、动作、终止和安全结算逐字段一致。
3. score orientation、settled label、horizon 和 action scale 已经审计，ranking 不再系统性反向。
4. OOD、stale、non-finite、低信用和 CBF 失败永不执行 raw/unverified action。
5. M0/M3/A1/A2 的三 seed paired smoke 可重放、可统计、可审计。
6. 后续 development block 中安全硬门为零回归，safe-capture 以 episode 配对证据为主结论。
7. 所有代码、环境、协议、checkpoint、ledger、calibration、scene、命令和结果都有 hash/provenance。
8. 未经明确授权，`locked_test_opened=false` 始终保持不变。

### 允许的最终结论类别

```text
safe_capture_improvement_candidate
safe_capture_noninferior_safety_preserving
prediction_signal_no_control_gain
rejected_for_safety
insufficient_evidence_do_not_open_locked_test
```

当前阶段的默认结论仍是：**安全执行基础设施和设备决定性已获得证据，但 JEPA 排序与 settled outcome 的反向关系尚未解决，因此不能声称控制收益，也不能进入最终大规模性能实验。**
