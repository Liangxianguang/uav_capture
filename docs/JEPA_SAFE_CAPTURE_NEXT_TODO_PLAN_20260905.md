# 无人机集群对抗围捕：下一阶段详细 TODO 计划书

> **当前执行版：** 以 v20 CPU deterministic baseline 为准的更新计划见 [`JEPA_SAFE_CAPTURE_NEXT_TODO_PLAN_20260905_V2.md`](JEPA_SAFE_CAPTURE_NEXT_TODO_PLAN_20260905_V2.md)。本文件保留为早期执行记录。

**系统路线：** Interaction-aware Action-conditioned JEPA + Reliability Ledger + Joint CBF-QP + Rolling Horizon  
**计划版本：** 2026-09-05 current execution plan  
**执行目录：** `D:\\uav-capture\\uav_capture`  
**硬件：** NVIDIA GeForce RTX 5050  
**当前运行时：** `D:\\download\\anaconda3\\envs\\traj_pred_prep`，Python 3.11.14，Torch 2.9.1+cu130，CUDA 可用  
**目标 Conda 环境：** `uav-encirclement-gpu`（尚未安装，网络/镜像恢复后再补齐）  
**实验状态：** `development_only=true`，`locked_test_opened=false`  
**第一指标：** `safe_capture`  
**诊断指标：** `mean_capture_time`、transit、路径长度、CBF 修正量、fallback 率和延迟

> 这是一份下一步执行清单，不是新的实验结果。95% 不是硬目标；捕获时间不能抵消碰撞、边界、编队分离、CBF 失败或 controlled abort。

## 1. 目标

实现并验证一套安全增强的无人机集群对抗围捕闭环：

```text
多机观测/通信历史
  -> interaction-aware BeliefState
  -> 传统规划器生成候选 action chunks
  -> action-conditioned JEPA 进行隐空间反事实评价
  -> immutable Reliability Ledger 可信度路由/拒答
  -> safety-first ranker、nominal anchor、滞回
  -> Joint CBF-QP 统一安全过滤
  -> 只执行第一控制步
  -> 重新观测、更新 belief、重新规划
```

JEPA 只能评价候选轨迹，不能生成或直接执行动作。最终可执行动作必须来自同一个 `Joint CBF-QP` 的 `verified=true` 输出。

## 2. 当前事实和主要阻塞

### 2.1 已完成证据

- [x] v12 calibrated-clearance 协议、q10 净空残差校准和三 seed checkpoint-bound ledger 已生成。
- [x] OOD、stale、non-finite、unknown horizon 和 provenance fault 均有显式回退审计。
- [x] Joint CBF-QP 的安全过滤、故障回退、zero-perturbation 和 RTX 5050 延迟审计已通过当前测试。
- [x] seed `20260911` 的 CUDA/CPU 回放安全结算一致：collision、boundary、pairwise、raw-unverified 和 CBF timeout 均为 0。
- [x] 当前回放只执行每个 action chunk 的第一步，并在下一周期重新规划。

### 2.2 当前未完成证据

- [ ] CPU/CUDA trace 尚未逐字段决定性一致；部分 episode 的 trace 长度不同。
- [ ] ranker 当前只对 abstention margin 做保守量化，仍用原始浮点 `np.argsort(scores)` 排序。
- [ ] 已观察到 top-two margin 和 score ordering 在设备间漂移，漂移会改变后续状态、候选和 trace 长度。
- [ ] v12 smoke 不能证明 M3 的闭环控制收益；`safe_capture` 必须继续作为唯一主指标。
- [ ] 只有新 revision 通过双设备 replay、settled ranking 和安全硬门后，才允许扩大到 40/60 集。

### 2.3 典型漂移样本

以下样本仅用于回归测试，不得事后调参：

| 位置 | CUDA margin | CPU margin | 现象 |
|---|---:|---:|---|
| episode 0, step 31 | 0.00354045 | 0.00348010 | abstention 边界附近排序可能分叉 |
| episode 6, step 193 | 0.00346777 | 0.00352702 | top-two 比较方向可能分叉 |

当前比较上界约为 `0.003`，但上界只覆盖 abstention，不覆盖原始 score 排序，因此不能宣称决定性一致。

## 3. 不可变安全合同

### 3.1 safe-capture 定义

一个 episode 只有同时满足以下条件才计为 `safe_capture=true`：

1. 至少一个 defender 在 time limit 内进入 `0.80 m` capture radius；
2. 无 obstacle、target 或 defender-defender collision；
3. 无 defender boundary/altitude violation；
4. 无 pairwise separation violation；
5. 无 CBF infeasible、timeout、unverified action 或 `controlled_abort` 终止。

`controlled_abort` 必须计入失败分母，并与 `cbf_timeout`、`cbf_infeasible`、`raw_unverified_executed` 分开统计。

### 3.2 固定回退顺序

```text
separation-preserving safe-hold
  -> verified nominal through the same Joint CBF-QP
  -> controlled_abort
```

任何 JEPA/ledger 超时、non-finite、OOD、stale、hash 不匹配或 CBF 失败都禁止执行 raw desired action。

### 3.3 候选动作合同

- 候选数固定为 5：`nominal`、`intercept`、`lateral_clearance`、`formation_clearance`、`visibility_hold`。
- 每个候选固定为 3 个 control steps 的 action chunk。
- 在线只执行第一步，随后重新观测、重新排序和重新过滤。
- 候选进入 JEPA 前必须通过 finite、shape、speed、acceleration、slew 和 reachability 检查。
- `nominal` 永远保留为 anchor；不可信候选不能替换 nominal。
- CBF margin、几何约束、solver timeout 和安全回退阈值在本轮排序修复中不得改变。

## 4. 立即执行队列（必须按顺序）

### P0：运行环境和证据冻结

**目标：** 保证新结果不会污染旧 V4/V5/tmp 或混用协议。

- [ ] 保存 `git status --short`；保留现有用户改动和 `tmp/`，不得 reset、删除或覆盖。
- [ ] 保存 Python、Torch、CUDA、GPU、Conda 包清单和完整命令行。
- [ ] 计算 protocol、checkpoint、calibration archive、ledger、scene manifest 和代码 revision 的 SHA-256。
- [ ] 检查目标输出目录为空；任何非空目录必须停止，不允许覆盖。
- [ ] 验证 `development_only=true`、`locked_test_opened=false`、split 和 target-truth 信息边界。
- [ ] 运行 `git diff --check` 与当前核心测试。

**出口门：** 生成新的 `preflight.json` 和输入 hash manifest；任一 hash 或 split 不一致则停止。

### P1：修复真正的跨设备 ranking decision

**目标：** 让同一 belief、同一候选和同一 protocol 在 CPU/CUDA 上产生相同的离散决策。

- [ ] 读取 CUDA/CPU trace 的 common-prefix mismatch，定位每个 episode 的最早 score ordering drift。
- [ ] 在 `jepa_safe_capture_ranker.py` 中加入预注册的固定点 score comparison：
  - 保留原始 float score 作为诊断字段；
  - 用固定 `score_comparison_quantum_m` 转换为有界整数 comparison key；
  - 用整数 key 完成排序、tie、nominal anchor 和 hysteresis；
  - 对近似或无法区分的候选优先走 nominal/abstention；
  - 不改变 CBF 几何约束和 `verified` 语义。
- [ ] 明确负值、非有限值、quantum 边界、相同整数 key 和 candidate index 的稳定 tie-break。
- [ ] 确保 `top_two_margin_m` 仍保留原始诊断值，同时记录离散 `top_two_margin_comparison_m`。
- [ ] 增加跨设备和边界单测：margin `0.002977`、`0.003000`、`0.003527`、`0.003540`，score quantum 边界、tie、hysteresis、nominal anchor 和排序稳定性。
- [ ] 增加一个 invariant：JEPA 输出只能改变 candidate ranking；不得直接写入执行 action。

**出口门：** 纯 ranker 测试通过；相同输入在 CPU/CUDA 上的 candidate order、abstention、hysteresis 和 selected index 完全一致。

### P2：协议 revision、preflight 和 ledger 重建

**目标：** 任何排序规则变化都产生新的可追溯实验合同。

- [ ] 从当前 v12 protocol 新建下一 monotonic revision，记录 fixed-point ranking、tie-break 和 boundary policy。
- [ ] 保持 `cbf_margin_changed=false`、`locked_test_opened=false` 和 `development_only=true`。
- [ ] 重新计算 protocol hash；旧 preflight 和旧 ledger 不得复用。
- [ ] 用新 protocol 为 `20260911/20260912/20260913` 重新生成 checkpoint-bound ledger。
- [ ] 重跑 OOD/stale/non-finite/provenance fallback audit；确认每类 100% 进入规定回退路径。
- [ ] 每份 ledger 记录 checkpoint hash、calibration hash、protocol hash、builder hash、bucket coverage 和 TensorBoard event。

**出口门：** 三份 ledger 绑定同一新 protocol；所有 fallback、immutability 和 provenance 测试通过。

### P3：单 seed 双设备决定性回归

**目标：** 先用 seed `20260911` 证明新 revision 的闭环决定性，再扩展其他 seed。

- [ ] 使用同一 scene manifest、environment config、checkpoint、ledger、episode index 和 observation schedule。
- [ ] 分别运行 RTX 5050 CUDA 与 CPU replay；输出目录完全独立。
- [ ] 逐字段比较 episode summary、step trace、candidate order、ledger state、selected index、CBF status、action、termination 和 trace hash。
- [ ] trace 长度、controlled-abort step、fallback reason 和最早 mismatch 必须全部相同。
- [ ] 保留 `safe_capture`、collision、boundary、pairwise、CBF timeout/infeasible/abort、raw-unverified 和 latency 统计。

**出口门：** seed `20260911` 双设备逐字段一致。未通过时禁止跑其他 seed，不调 CBF margin。

### P4：三 seed 设备门和 settled ranking 审计

**目标：** 确认决定性不是单 seed 偶然现象，并量化 JEPA 排序是否与离线 settled outcome 一致。

- [ ] 对 seed `20260912`、`20260913` 重复 P3。
- [ ] 生成 `common_prefix_mismatch`、candidate score-order、top-two margin、hysteresis、switch rate 和 oscillation 报告。
- [ ] 从同一冻结 trace 生成 settled counterfactual labels；target truth 仅标记 `offline_only=true`。
- [ ] 计算 selected-not-best、Spearman/Kendall rank correlation、top-1 safety precision/recall 和按 ledger state 的分桶结果。
- [ ] 分桶覆盖 nominal/delayed_noisy、急转、S-curve、速度突变、遮挡、消息延迟、低净空和拥挤队形。
- [ ] 任何无法由 trace 证明的因果都标记为 `unresolved`，不得用推测替代。

**出口门：** 三 seed ranking decision 无跨设备 mismatch；settled ranking 结果可重放、可解释并含完整 coverage。

### P5：100-cycle rolling-horizon 与 Joint CBF 回归

**目标：** 证明长序列不会因 rollout 漂移或执行旁路破坏安全合同。

- [ ] 至少运行 100 个 control cycles 的两次 deterministic replay；再选取一组困难片段做 500-cycle stress replay。
- [ ] 审计每周期顺序：observe -> belief -> candidate -> reachability -> JEPA -> ledger -> rank -> CBF -> first-step execute -> trace。
- [ ] 验证完整 3-step chunk 不会被一次性执行。
- [ ] 记录每周期 prediction、uncertainty、credit、selected candidate、CBF active set、slack、correction norm、fallback 和 latency。
- [ ] 注入 CBF infeasible、timeout、non-finite request、stale/OOD、通信中断和多约束压力。
- [ ] 对所有故障验证 safe-hold -> nominal CBF -> controlled_abort 顺序和 `raw_unverified_executed=0`。

**出口门：** 100-cycle trace 双次一致；安全错误计数为 0；端到端 p95 满足当前 latency contract；否则禁止 smoke 扩展。

### P6：新 revision 的三 seed paired smoke

**目标：** 在固定小规模上检查排序修复是否改善或至少不破坏 safe-capture。

- [ ] 变体固定为 M0、M3、A1、A2；A3 raw/no-CBF 只作独立诊断，不参与安全结论。
- [ ] 使用同一 paired scene manifest、episode index、layout、target motion 和 observation schedule。
- [ ] 每个变体、每个 seed 运行 20 集；不在运行期间改阈值、换 checkpoint、换 seed 或删除 abort。
- [ ] 保存 `summary.json`、`episodes.csv`、step traces、scene manifest、run metadata、hash manifest 和 TensorBoard。
- [ ] 逐 seed 报告 safe-capture、paired delta、collision、boundary、pairwise、CBF timeout/infeasible/abort、fallback、raw-unverified 和 latency。

**出口门：**

- 所有安全保留变体 collision/boundary/pairwise/raw-unverified 均为 0；
- M3 至少 2/3 seed 的 paired delta 非负；
- M3 平均 paired delta `>= 0`，或明确归类为 `prediction_signal_no_control_gain`；
- ranking mismatch 不得恶化；
- 任一安全硬门失败立即停止该变体并回到 P1/P5。

### P7：扩展到每 seed 40/60 集 development block

**进入条件：** P0-P6 全部通过，且新 revision 的失败机制已有可解释改善。

- [ ] 预注册每个 seed 的 40 集 block；如资源允许，再预注册独立 60 集扩展 block。
- [ ] M0/M3/A1/A2 逐 episode 配对，保持完全相同场景和观察 schedule。
- [ ] 以 episode 为统计单位，不把 timestep 或 chunk 当独立样本。
- [ ] 计算 mean、sample SD、paired delta、bootstrap 95% CI、exact McNemar、improved/degraded/tied。
- [ ] 同时报告按 motion mode、visibility、observation age、clearance、ledger state 和 CBF active constraint 的分桶结果。
- [ ] 不得以 mean capture time 抵消 safe-capture 下降或安全失败。

**分类规则：**

| 分类 | 条件 |
|---|---|
| `promising_development_candidate` | 平均 paired delta > 0，至少 2/3 seed 非负，安全硬门全通过 |
| `safe_non_inferior` | 平均 paired delta >= 0，安全硬门全通过，但统计证据不足以称提升 |
| `prediction_signal_no_control_gain` | 预测/排序信号存在，但闭环没有净收益 |
| `rejected_for_safety` | 任一碰撞、边界、pairwise、raw action 或不可解释 CBF 失败 |
| `insufficient_evidence_do_not_open_locked_test` | 覆盖率、决定性或 provenance 不足 |

### P8：鲁棒性、部署和 SIL/HIL readiness

**进入条件：** P7 安全硬门通过。

- [ ] 对 detection dropout/noise、message delay/dropout、target 急转、突变加速度、障碍密度、初始侧距和拥挤队形建立 stress matrix。
- [ ] 运行 OOD/stale/non-finite/provenance mismatch fault injection；确认 100% fallback 且 raw-unverified 为 0。
- [ ] 在 RTX 5050 上测量 JEPA、ledger、ranker、CBF 和 cycle total 的 p50/p95/p99。
- [ ] 审计 100/500/1000-cycle 长序列的候选抖动、CBF correction 累积、ledger 状态转移和 trace 完整性。
- [ ] 完成 SIL 接口合同：时间戳、通信年龄、执行回执、solver 状态和 controlled-abort 事件。
- [ ] HIL 只在 SIL 通过后进行，真实飞控接口不得绕过 CBF。
- [ ] 写部署故障手册：GPU 不可用、JEPA 超时、ledger 损坏、QP 不可行、通信中断和传感器陈旧时的固定动作。

**出口门：** stress matrix、fallback、latency、长序列和 SIL/HIL 接口全部通过；否则保持 development-only。

### P9：最终统计、归档和 locked 决策

- [ ] 生成三 seed paired aggregate、failure index、settled ranking report、device audit、CBF audit 和 reproducibility manifest。
- [ ] JSON、CSV、TensorBoard、Markdown 做双向一致性检查。
- [ ] 汇总代码 revision、环境、protocol、checkpoint、ledger、calibration、scene、命令和所有 hash。
- [ ] 保留全部负结果、controlled abort 和 unresolved failure；不得只发布成功 episode。
- [ ] 只有在 P8 通过且有明确授权后，才新建 locked-test preregistration；本计划执行期间始终 `locked_test_opened=false`。

## 5. 实验矩阵

| 阶段 | 变体 | seed | episode | 目的 |
|---|---|---:|---:|---|
| device replay | M0/M3 | 20260911 -> 12 -> 13 | 同一 20 集 manifest | 决定性和 trace 一致性 |
| settled ranking | M0/M3/A1/A2 | 3 | 全量离线 trace | 选中候选与真实后果关系 |
| CBF fault matrix | M0/M3 | 3 | 固定注入矩阵 | 验证不可绕过安全边界 |
| paired smoke | M0/M3/A1/A2 | 3 | 20/seed | 新 revision 快速安全门 |
| development block | M0/M3/A1/A2 | 3 | 40/seed，条件扩展 60 | safe-capture 主比较 |
| robustness stress | M0/M3 | 3 | 独立 hard block | 分布变化和长序列 |
| raw diagnostic | A3 | 3 | 与 paired block 对齐 | 只显示无 CBF 风险 |

## 6. 统一产物和命名规则

所有新目录必须为空目录起跑，统一使用新 revision 前缀，例如：

```text
results/
  jepa_safe_capture_v13_fixedpoint_preflight/
  jepa_safe_capture_v13_fixedpoint_ledger_seed<seed>/
  jepa_safe_capture_v13_fixedpoint_replay_<device>_seed<seed>/
  jepa_safe_capture_v13_fixedpoint_settled_seed<seed>/
  jepa_safe_capture_v13_fixedpoint_smoke_<variant>_seed<seed>/
  jepa_safe_capture_v13_fixedpoint_failure_index/
  jepa_safe_capture_v13_fixedpoint_aggregate/
tensorboard/
  jepa_safe_capture_v13_fixedpoint/<stage>/seed<seed>/
```

每个结果目录至少包含：`summary.json`、`run_metadata.json`、命令行、代码/protocol/checkpoint/ledger/scene hash、`development_only=true`、`locked_test_opened=false` 和 TensorBoard event。

## 7. RTX 5050 执行模板

```powershell
Set-Location D:\\uav-capture\\uav_capture
$py = 'D:\\download\\anaconda3\\envs\\traj_pred_prep\\python.exe'
$env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = 'python'

& $py -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
& $py -m pytest -q tests/test_jepa_safe_capture_candidates.py tests/test_jepa_safe_capture_v2_reliability.py tests/test_jepa_safe_capture_v2_paired.py
& $py scripts/verify_jepa_safe_capture_protocol.py --protocol <new_protocol> --development-only
```

目标 `uav-encirclement-gpu` 环境恢复后，先导出并比较 `conda list`、`pip freeze`、Torch/CUDA/GPU 信息，再重复 P0；环境差异必须进入 provenance，不能静默切换。

## 8. 硬停止规则

- 任一输入 hash、split、checkpoint、ledger 或 manifest 不一致：停止并重建 preflight。
- CPU/CUDA 的 selected index、ledger state、CBF status、action 或 termination 不一致：停止，不扩大 episode 数。
- `raw_unverified_executed > 0`、collision、boundary、pairwise violation、CBF timeout/infeasible 未进入显式回退：立即标记 `BLOCKED_BY_SAFETY`。
- 发现通过降低 CBF margin、扩大 stale age、关闭 OOD 检查、删除 controlled abort 或执行完整 action chunk 才能提高捕获率：拒绝该修改。
- smoke 中 M3 少于 2/3 seed 非负、平均 paired delta < 0 或 ranking mismatch 恶化：回到 P1/P4，不凑 episode 数。
- 证据覆盖不足时使用 `insufficient_evidence`，不得把“未观察到失败”写成“已经证明安全”。
- 未经明确授权，不读取或打开 locked-test split。

## 9. 提交纪律

每个工作包独立 conventional commit，禁止 `git add .`：

1. `fix(jepa): make candidate ranking device deterministic`
2. `test(jepa): cover fixed point ranking boundaries`
3. `chore(protocol): freeze fixed point development revision`
4. `audit(jepa): complete three seed device replay`
5. `audit(safety): complete rolling horizon and cbf regression`
6. `docs(experiment): archive paired development decision`

提交前运行 `git diff --check`、targeted tests 和 provenance/hash 检查。不得提交 `tmp/`、checkpoint、NPZ 或未经审计的 `results/`。

## 10. 完成定义

本计划只有在以下条件全部满足时才算完成：

1. JEPA、ledger、ranker、Joint CBF-QP 和 rolling executor 接口均有代码、schema test 和逐步 trace。
2. 固定点 ranking 使三 seed CPU/CUDA replay 的决策和终止字段逐字段一致。
3. OOD、stale、non-finite、低信用和 CBF 失败路径永不执行 raw/unverified action。
4. 三 seed paired smoke 和后续 40/60 集 development block 均以 episode 为单位可独立复核。
5. 所有安全保留变体通过 collision、boundary、pairwise、zero-perturbation、fallback 和 latency 硬门。
6. 最终结论以 `safe_capture` 为第一指标，逐 seed 报告 paired statistics；`mean_capture_time` 只作诊断。
7. 所有代码、环境、协议、checkpoint、ledger、calibration、scene 和结果有 hash/provenance。
8. 在明确授权前，`locked_test_opened=false` 始终保持不变。

**核心判断：** 只有当 JEPA 的候选反事实排序、ledger 的可信度拒答、CBF 的硬安全边界和 rolling-horizon 重规划在三 seed 困难场景中共同通过这些证据门，才能把系统称为安全增强的闭环围捕系统；否则应诚实归类为 prediction signal、safety infrastructure 或 development evidence。
