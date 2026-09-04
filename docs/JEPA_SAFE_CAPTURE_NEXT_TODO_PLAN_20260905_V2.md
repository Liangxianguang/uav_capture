# 无人机集群对抗围捕安全增强系统：下一步详细 TODO 计划书

**系统名称：** Interaction-aware Action-conditioned JEPA + Reliability Ledger + Joint CBF-QP + Rolling Horizon  
**计划版本：** 2026-09-05-v2（v20 deterministic development baseline）  
**适用任务：** 三维无人机集群对抗围捕、拦截与安全退出  
**运行硬件：** NVIDIA GeForce RTX 5050  
**当前环境：** `D:\download\anaconda3\envs\traj_pred_prep`，Python 3.11.14，Torch 2.9.1+cu130，CUDA 可用  
**主指标：** `safe_capture`  
**诊断指标：** capture time、transit、路径长度、最小净空、CBF 修正量、fallback/abort 率和端到端延迟  
**证据状态：** `development_only=true`，`locked_test_opened=false`；95% 不是硬门槛  

> 本文件是执行计划，不是新的实验结果。任何“提升”必须来自同一场景、同一 episode、三训练 seed 的配对证据；mean capture time 不能抵消 safe-capture 或安全性下降。

## 1. 总目标和系统边界

要实现的闭环为：

```text
观测/通信历史
  -> interaction-aware BeliefState
  -> 传统规划器产生物理可行候选 action chunks
  -> action-conditioned JEPA 反事实评价
  -> Reliability Ledger 可信度校验与拒答
  -> 固定点安全优先排序（含 nominal anchor 和滞回）
  -> Joint CBF-QP 统一安全过滤
  -> 仅执行第一控制步
  -> 重新观测、更新 belief、重新规划
```

不可变边界：

1. JEPA 是轨迹评价器，不生成、覆盖或直接执行控制动作。
2. 所有候选（包括 nominal）必须进入同一个 Joint CBF-QP；只有 `verified=true` 的输出可执行。
3. action chunk 固定为 3 个控制步，但在线只执行第 1 步。
4. target truth 只用于离线 settled labels 和 episode 结算，不能进入在线 belief、actor、JEPA 或 ledger。
5. ledger 在 calibration 后只读、与 checkpoint/protocol/calibration hash 绑定，禁止在线更新信用。
6. `controlled_abort` 计入失败分母，不能通过删除 abort 或执行 raw action 提高成功率。

## 2. 安全合同和结算语义

### 2.1 `safe_capture=true`

一个 episode 只有同时满足以下条件才算成功：

- 至少一个 defender 在时间上限内进入 `0.80 m` capture radius；
- 无 obstacle、target 或 defender-defender collision；
- 无 boundary/altitude violation；
- 全程满足 pairwise separation；
- 无 CBF infeasible、timeout、unverified action 或 `controlled_abort`。

### 2.2 固定回退链

```text
separation-preserving safe-hold
    -> 冻结 nominal 通过同一 Joint CBF-QP
    -> controlled_abort（显式 reason code）
```

任何 JEPA/ledger 超时、non-finite、OOD、stale、hash 不匹配或 QP 失败，都禁止执行 raw desired action。必须记录 `fallback_reason`、solver status、slack、correction norm 和 latency。

## 3. 目标架构接口

### 3.1 BeliefState

每周期固定 schema，至少包括：

- defender 位置/速度、最近动作历史、队形几何；
- target 的 belief 位置/速度/加速度、运动模式、visibility、observation/message age；
- 障碍几何、边界/高度余量、局部净空；
- defender pairwise 相对状态和 TTC；
- 上周期候选 index、ledger state、CBF status、修正量和 solver latency；
- `layout_signature`、时间戳和 provenance id。

### 3.2 候选生成器

固定 `K=5`：`nominal`、`intercept`、`lateral_clearance`、`formation_clearance`、`visibility_hold`。每个候选必须通过 finite、shape、速度、加速度、slew 和 reachability 预检查；拒绝候选不得进入 JEPA，并记录拒绝原因。

### 3.3 JEPA 评价器输出

对每个候选和多个 horizon 输出：

```text
target displacement / velocity / acceleration
obstacle-clearance lower quantile
inter-agent clearance lower quantile / pairwise TTC
visibility probability / observation-age risk
CBF intervention probability / correction magnitude / QP feasibility
predictive uncertainty / ensemble disagreement
```

推荐评分只用于排序，不用于安全证明：

```text
score(k) = task_progress
         + visibility_gain
         - clearance_risk_lower_quantile
         - cbf_intervention_cost
         - uncertainty_penalty
         - action_change_cost
         - nominal_anchor_penalty
```

### 3.4 Reliability Ledger

ledger 状态和动作必须是一一对应的状态机：

| 状态 | 触发条件 | 允许动作 |
|---|---|---|
| `trusted` | bucket 覆盖充分、信用达标、uncertainty/stale 在阈值内 | 允许 JEPA 重排候选 |
| `fallback_nominal` | 信用下降、候选分离消失、预测漂移或 bucket 缺失 | nominal -> CBF |
| `safe_hold` | OOD、non-finite、过期观测、连续失败或 provenance fault | safe-hold -> CBF |
| `controlled_abort` | safe-hold/nominal 也无法验证可行 | 终止、计失败、保留证据 |

ledger 需要按 visibility、observation age、target motion、障碍/拥挤度、clearance、uncertainty、candidate separation 和 CBF risk 分桶，保存样本数、误差、coverage、credit、失败率与置信区间。

### 3.5 Joint CBF-QP

联合约束 obstacle、defender-defender separation、boundary/altitude、速度/加速度/slew、捕获区接近和 QP 可行性。CBF 是最后执行边界，不能被 JEPA、planner 或 fallback 绕过。

### 3.6 Rolling executor

严格执行：`observe -> belief -> candidate -> reachability -> JEPA -> ledger -> rank -> CBF -> first-step execute -> trace`。下一周期必须重新观测和规划，禁止一次性执行完整 chunk。

## 4. 当前状态和唯一优先级

已具备：v12 校准数据、三 seed checkpoint-bound ledger、CBF 基础故障回退、RTX 5050 运行时和核心单测。当前 v19 只固定了 ranker CPU；actor 在设备间仍有约 `0.005 m/s` 输出差异，导致后续 score、selected index 和 trace 长度分叉。

因此第一优先级不是扩大 episode 数或继续调 score quantum，而是建立新的 v20 协议并验证 **actor + ranking 全部 CPU** 的决定性回放。v19 旧 hash、ledger 和 replay 不得冒充 v20 证据。

## 5. 分阶段 TODO 与出口门

### WP0 — 环境、输入和 provenance 冻结

- [ ] 保存 `git status --short`、Conda/Python/Torch/CUDA/GPU 信息和完整命令行。
- [ ] 为 protocol、checkpoint、calibration archive、ledger、scene manifest、代码 revision 计算 SHA-256。
- [ ] 创建全新的 v20 输出目录；非空目录禁止覆盖。
- [ ] 运行时硬校验 `development_only=true`、`locked_test_opened=false`、split 和 target-truth 边界。
- [ ] 运行 `git diff --check`、py_compile 和 targeted tests。

**出口门：** 生成 `preflight.json` 与输入 hash manifest；任何 hash/split 不一致立即停止。

### WP1 — v20 CPU deterministic ranking

- [ ] 新建 `configs/central_random_mixed_obstacle_s3_v5_v20_cpu_deterministic_development_protocol.yaml`，`protocol_version=20`、`profile=p20_cpu_deterministic_v1`、`ranking_device=cpu`、`actor_device=cpu`。
- [ ] evaluator 和 verifier 支持 v20；保持 `cbf_margin_changed=false`。
- [ ] ranker 使用预注册固定点 `score_comparison_quantum_m` 的整数 comparison key；保留原始 float 作为诊断。
- [ ] 对负值、非有限值、quantum 边界、相同 key、nominal tie、hysteresis 和 candidate index 增加单测。
- [ ] 增加 invariant：JEPA 只能改变 candidate ranking，不能写入执行 action。

**出口门：** 相同输入在 CPU/CUDA 上的 candidate order、abstention、hysteresis 和 selected index 完全一致。

### WP2 — 重新构建三 seed reliability ledger

- [ ] 使用 v20 protocol 为 `20260911/20260912/20260913` 重新生成 ledger、calibration report、diagnostics、fallback audit 和 TensorBoard。
- [ ] 重新运行 OOD、stale、non-finite、unknown horizon、provenance mismatch fault audit。
- [ ] 每份 ledger 绑定 checkpoint/protocol/calibration/builder hash，写明 bucket coverage 和版本。
- [ ] 将旧 v19 ledger 标记为历史证据，不复用。

**出口门：** 三份 ledger 都绑定 v20；所有异常样本 100% 进入规定回退，`raw_unverified_executed=0`。

### WP3 — seed 20260911 双设备决定性回放

- [ ] 复用同一 scene manifest、episode index、checkpoint、ledger、observation schedule；只改变执行 backend 为 CUDA/CPU。
- [ ] 逐字段比较 summary、trace 长度、candidate order、selected index、ledger state、CBF status、action、termination 和 trace hash。
- [ ] 比较 safe_capture 结算、collision、boundary、pairwise、CBF timeout/infeasible/abort、raw-unverified 和 latency。

**出口门：** 逐字段一致后才允许进入其他 seed；未通过时禁止扩大 episode 或修改 CBF margin。

### WP4 — 三 seed replay + settled ranking audit

- [ ] 对 `20260912/20260913` 重复 WP3。
- [ ] 从同一冻结 trace 生成 `offline_only=true` 的 settled counterfactual labels。
- [ ] 计算 selected-not-best、Spearman/Kendall、top-1 safety precision/recall、top-two margin、switch/oscillation rate。
- [ ] 按 nominal/delayed_noisy、急转、S-curve、遮挡、消息延迟、低净空和拥挤队形分桶。
- [ ] 无法由 trace 证明的因果标记 `unresolved`，不以推测替代。

**出口门：** 三 seed 无跨设备 decision mismatch，settled ranking 可重放且 coverage 完整。

### WP5 — 100/500-cycle rolling-horizon 和 CBF fault regression

- [ ] 至少 2 次 100-cycle deterministic replay；困难片段进行 500-cycle stress replay。
- [ ] 注入 QP infeasible、timeout、non-finite request、stale/OOD、通信中断、多约束激活和单机异常。
- [ ] 审计 safe-hold -> nominal CBF -> controlled_abort 顺序；验证完整 chunk 未被一次性执行。
- [ ] 逐周期记录 prediction、uncertainty、credit、selected candidate、active set、slack、correction 和 latency。

**出口门：** 长序列 trace 双次一致；collision/boundary/pairwise/raw-unverified 为 0；p95 满足控制周期预算。

### WP6 — JEPA 可靠性增强（在决定性基线通过后）

- [ ] 保留 target displacement head，新增 velocity/acceleration consistency、obstacle/inter-agent clearance lower-quantile、visibility/observation-age、CBF intervention/QP feasibility heads。
- [ ] 加入 target turn、flee persistence、S-curve、突变加速度等 motion-mode embedding。
- [ ] 使用 ensemble disagreement、heteroscedastic 或 calibrated residual 估计 uncertainty；限制 rollout 到校准 horizon。
- [ ] 构建 hard-fragment replay archive：低净空、遮挡、延迟、拥挤、急转和 high-credit failure；按 episode 做 train/validation/calibration 隔离。
- [ ] 为每个候选保存 settled multi-horizon labels，禁止把失败 development episode 直接回灌训练。
- [ ] 训练 TensorBoard 必须包含各 head loss、MAE/Brier/AUROC、coverage、uncertainty 和 provenance。

**出口门：** 三 seed 输出 finite；至少一个 horizon 优于 constant-velocity；辅助安全头有非空 coverage 和校准证据；不把 prediction gate 当作控制收益证明。

### WP7 — 三 seed paired development smoke

- [ ] 变体固定 M0（nominal+CBF）、M3（JEPA+ledger+CBF）、A1（去 ledger）、A2（去 clearance/visibility 排序）；A3 raw/no-CBF 仅诊断。
- [ ] 使用同一 paired scene manifest、episode index、layout、target motion 和 observation schedule。
- [ ] 每变体每 seed 20 集，独立输出 `summary.json`、`episodes.csv`、traces、manifest、provenance、hash 和 TensorBoard。
- [ ] 统计逐 seed safe_capture、paired delta、collision、boundary、pairwise、CBF failures、fallback、raw-unverified 和 latency。

**出口门：** 安全保留变体安全错误为 0；M3 至少 2/3 seed paired delta 非负且平均不低于 0；否则分类为 `prediction_signal_no_control_gain` 或回到 WP4/WP6。

### WP8 — 40/60 集 development block 和鲁棒性矩阵

- [ ] 只有 WP0–WP7 全部通过才扩展到每 seed 40 集，资源允许时预注册独立 60 集 block。
- [ ] 覆盖 target 急转/突变速度、detection dropout/noise、message delay/dropout、3–5 障碍、初始侧距和高拥挤度。
- [ ] 以 episode 为统计单位，计算 mean、sample SD、paired delta、bootstrap 95% CI、exact McNemar 和 improved/degraded/tied。
- [ ] 按 motion、visibility、age、clearance、ledger state、CBF active constraint 分桶。

### WP9 — SIL/HIL readiness、归档和 locked 决策

- [ ] RTX 5050 测量 JEPA、ledger、ranker、CBF 和 cycle total 的 p50/p95/p99；超时必须回退 nominal CBF。
- [ ] 完成 100/500/1000-cycle 长序列审计、watchdog、通信/传感器冻结和 GPU 不可用故障手册。
- [ ] SIL 通过后再做 HIL；真实飞控接口不得绕过 CBF。
- [ ] 生成 paired aggregate、failure index、settled ranking、device audit、CBF audit 和 reproducibility manifest。
- [ ] 只有所有前置门通过且获得明确授权，才新建 locked-test preregistration；此前始终保持关闭。

## 6. 实验矩阵

| 阶段 | 变体 | seed | 规模 | 目的 |
|---|---|---:|---:|---|
| deterministic replay | M0/M3 | 20260911→12→13 | 同一 20 集 manifest | 设备一致性 |
| settled ranking | M0/M3/A1/A2 | 3 | 全量冻结 trace | 排序因果 |
| CBF fault matrix | M0/M3 | 3 | 固定注入矩阵 | 安全回退 |
| paired smoke | M0/M3/A1/A2 | 3 | 20/seed | 快速安全门 |
| development block | M0/M3/A1/A2 | 3 | 40/seed，条件 60 | safe_capture 主比较 |
| stress/SIL | M0/M3 | 3 | 独立 hard block | 漂移、延迟、拥挤 |
| raw diagnostic | A3 | 3 | 与 paired block 对齐 | 仅展示绕过 CBF 的风险 |

## 7. 统一产物和命名

新 revision 统一使用 `v20_cpu_deterministic` 前缀，例如：

```text
results/jepa_safe_capture_v20_cpu_deterministic_preflight/
results/jepa_safe_capture_v20_cpu_deterministic_ledger_seed<seed>/
results/jepa_safe_capture_v20_cpu_deterministic_replay_<device>_seed<seed>/
results/jepa_safe_capture_v20_cpu_deterministic_settled_seed<seed>/
results/jepa_safe_capture_v20_cpu_deterministic_smoke_<variant>_seed<seed>/
results/jepa_safe_capture_v20_cpu_deterministic_aggregate/
tensorboard/jepa_safe_capture_v20_cpu_deterministic/<stage>/seed<seed>/
```

每个目录至少包含 `summary.json`、`run_metadata.json`、`episodes.csv`（如适用）、`step_traces/`、`scene_manifest.jsonl`、命令行、代码/protocol/checkpoint/ledger/scene hash、`development_only=true`、`locked_test_opened=false` 和 TensorBoard event。

## 8. RTX 5050 执行模板

```powershell
Set-Location D:\uav-capture\uav_capture
$py = 'D:\download\anaconda3\envs\traj_pred_prep\python.exe'
$env:PYTHONPATH = "$PWD\src;$PWD\scripts"
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = 'python'

& $py -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
& $py -m py_compile src/encirclement3d/jepa_safe_capture_ranker.py scripts/evaluate_jepa_safe_capture_v2_paired.py scripts/verify_jepa_safe_capture_protocol.py
& $py -m pytest -q tests/test_jepa_safe_capture_candidates.py tests/test_jepa_safe_capture_protocol.py tests/test_jepa_safe_capture_v2_paired.py
& $py scripts/verify_jepa_safe_capture_protocol.py --protocol configs/central_random_mixed_obstacle_s3_v5_v20_cpu_deterministic_development_protocol.yaml --development-only
```

若后续安装 `uav-encirclement-gpu`，必须先导出 `conda list`/`pip freeze` 并与当前环境比较；环境切换要写入 provenance，不得静默替换。

## 9. 硬停止规则

- protocol、checkpoint、ledger、calibration、scene 或代码 hash 不一致：停止并重建 preflight。
- CPU/CUDA 的 selected index、ledger state、CBF status、action 或 termination 不一致：停止，不扩大 episode。
- `raw_unverified_executed>0`、collision、boundary、pairwise violation 或未处理的 CBF timeout/infeasible：立即标记 `BLOCKED_BY_SAFETY`。
- 不得通过降低 CBF margin、关闭 OOD/stale 检查、扩大 stale age、删除 controlled_abort 或执行完整 chunk 来追逐捕获率。
- evidence 不足时使用 `insufficient_evidence_do_not_open_locked_test`，不得把“未观察到失败”写成“已证明安全”。
- 不删除/覆盖 `tmp/`、旧结果、NPZ 或 checkpoint；历史 v19 ledger/replay 不得复用为 v20 证据。

## 10. 完成定义

计划完成需同时满足：

1. JEPA、ledger、ranker、Joint CBF-QP 和 rolling executor 都有代码、schema test 和逐步 trace。
2. 三 seed CPU/CUDA replay 的离散决策、动作、终止和安全结算逐字段一致。
3. OOD、stale、non-finite、低信用和 CBF 失败路径永不执行 raw/unverified action。
4. 三 seed paired smoke 与后续 development block 以 episode 为单位可重放、可统计、可审计。
5. 安全保留变体通过 collision、boundary、pairwise、fallback、zero-perturbation 和 latency 硬门。
6. 结论以 `safe_capture` 为第一指标；capture time 仅作诊断，并逐 seed 报告配对统计。
7. 所有代码、环境、协议、checkpoint、ledger、calibration、scene、命令和结果都有 hash/provenance。
8. 在明确授权前，`locked_test_opened=false` 始终保持不变。

**最终判断：** 只有当 JEPA 的反事实评价、ledger 的可信度拒答、CBF 的硬安全边界和 rolling-horizon 重规划在三 seed 困难场景中共同通过上述证据门，系统才能被称为安全增强的闭环围捕系统；否则应诚实归类为 prediction signal、safety infrastructure 或 development evidence。
