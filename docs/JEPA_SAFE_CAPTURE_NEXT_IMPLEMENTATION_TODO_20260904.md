# 无人机集群对抗围捕安全增强系统
# 下一阶段实施 TODO 与验收计划

**版本：** v1.1（2026-09-04；已纳入 P1 全链路延迟审计结果）
**执行目录：** `D:\\uav-capture\\uav_capture`
**硬件：** NVIDIA RTX 5050；Conda 环境 `uav-encirclement-gpu`；PyTorch 2.7.1+cu128
**实验边界：** `development_only=true`；`locked_test_opened=false`
**主指标：** `safe_capture`
**诊断指标：** collision、boundary、pairwise separation、CBF abort/fallback、最小净空、延迟、路径代价、`mean_capture_time`

> 这份文件是下一阶段的实施入口，承接 [当前主 TODO](JEPA_SAFE_CAPTURE_CURRENT_MASTER_TODO_20260904.md)、[T4-T6 状态](JEPA_SAFE_CAPTURE_T4_T6_CURRENT_STATUS_20260904.md)、[T6 rolling replay](JEPA_SAFE_CAPTURE_T6_ROLLING_HORIZON_20260904.md) 和 [P1 全链路延迟报告](JEPA_SAFE_CAPTURE_P1_FULL_CHAIN_LATENCY_20260904.md)。旧计划仍保留为历史记录；本文件中的状态以最新结果为准。

## 1. 目标与不变边界

### 1.1 最终系统链路

```text
多机观测/通信历史
  -> interaction-aware belief state
  -> 传统规划器生成动力学可行候选 action chunks
  -> action-conditioned JEPA 反事实轨迹评价
  -> immutable reliability ledger 可信度/拒答
  -> safety-first ranker、nominal anchor、滞回
  -> Joint CBF-QP 安全过滤
  -> 只执行 action chunk 的第 1 步
  -> 重新观测、更新 belief、重新规划
```

### 1.2 研究问题

本阶段只回答一个核心问题：在不降低安全性的前提下，JEPA 评价器能否帮助传统规划器选择更安全、更有效的候选轨迹？模型不直接生成控制动作，最终执行动作只能来自同一个 `Joint CBF-QP`。

需要分别验证：

1. **预测信号：** 同一 belief 下，不同候选 action 对目标运动、可见性、净空、TTC 和 CBF 干预的预测具有可区分性。
2. **可信度：** ledger 在 OOD、stale、non-finite、高不确定性、候选分离消失和 hash 不一致时确定性拒答。
3. **执行安全：** candidate、nominal、safe-hold 和 fallback 均经过 CBF；QP 失败时不能执行 raw action。
4. **闭环稳定：** 每个 chunk 只执行第一步，下一周期重新规划，长 rollout 漂移不会累积为不可审计的动作。
5. **任务收益：** 三个训练 seed、同一 paired development block 下，`safe_capture` 相对 M0 至少不劣；只有配对证据一致时才讨论正向提升。

### 1.3 safe-capture 定义

完整 episode 只有同时满足以下条件才算 `safe_capture=true`：

- 至少一个 defender 在 time limit 内进入目标 `0.80 m` capture radius；
- 无 obstacle、target 或 defender-defender collision；
- 无 defender boundary/altitude violation；
- 无 pairwise separation violation；
- 无 CBF infeasible、timeout、unverified action 或 `controlled_abort` 终止。

`mean_capture_time` 只能作为诊断，不能抵消任何安全失败。`controlled_abort` 计为安全失败，但必须与 `raw_unverified_executed` 分开统计。

### 1.4 实现落点

| 工作包 | 主要代码入口 | 允许的改动 | 不得改变 |
|---|---|---|---|
| 候选生成/可达性 | `src/encirclement3d/jepa_safe_capture_candidates.py` | 候选 schema、拒绝原因、chunk 元数据 | `K=5`、chunk=3、只执行第一步 |
| JEPA 预测 | `src/encirclement3d/prediction.py`、`scripts/train_jepa_safe_capture_v3.py` | 多任务 head、uncertainty、batch latency | 不输出最终控制动作 |
| Reliability ledger | `src/encirclement3d/reliability.py`、ledger audit scripts | bucket、credit、三态拒答和 provenance | 只读、checkpoint-bound、OOD 必须回退 |
| Safety-first ranker | `src/encirclement3d/jepa_safe_capture_ranker.py` | 保守分数、abstention、hysteresis、nominal anchor | 不能选择不可达/不安全候选 |
| Joint CBF-QP | `src/encirclement3d/cbf_qp.py` | latency trace、fault diagnostics、fallback 记录 | 任何执行动作必须 verified |
| 闭环 evaluator | `scripts/evaluate_jepa_safe_capture_v2_paired.py` | step trace、latency breakdown、gate assertions | 不绕过 CBF，不改变 safe-capture 语义 |

代码修改每个阶段单独提交；results、checkpoint、NPZ 和 TensorBoard 不进入通用提交，也不得覆盖历史产物。

## 2. 当前事实基线

| 证据 | 已知结果 | 本阶段解释 |
|---|---|---|
| T3 ledger alignment/temporal audit | M3 `23/1052`、A2 `24/952` low/high；每 bucket >=20；OOD/stale/non-finite 进入 safe-hold | ledger 的可靠性和拒答语义已通过，尚未证明任务收益 |
| Zero-perturbation identity | M0 `10/20=50%`；M3 bypass `10/20=50%`；`field_difference_count=0` | JEPA 接入不改变 actor -> CBF 物理路径 |
| Joint CBF-QP fault audit | 9 deterministic cases、20 repeats；finite/fallback/deterministic；RTX 5050 CBF p95 <100 ms | CBF 是可审计的执行边界 |
| Non-zero rolling replay | 两次各 20 episodes、各 1,075 cycles；M3 `8/20=40%`；collision/boundary/pairwise/timeout/raw=0；逐字段一致 | rolling-horizon 的确定性通过，任务收益仍为 `no_control_gain` |
| Held-out auxiliary heads | 146,400 validation samples，12,486 hard samples；CBF-intervention Brier 改善，但 hard target MAE 变差 | 结论为 `prediction_signal_mixed`，不能直接换入闭环 |
| Settled ranking | M3 selected-not-best `26.23%`，A2 `25.41%`；Spearman `-0.517/-0.612`；rank guard 相对 M0 `-10/-5 pp` | 排序失配是当前首要算法问题 |
| 95% 观察值 | 仅为历史单 seed 开发结果 | 不作为本阶段硬目标或筛选条件 |

### 2.1 已完成，不重复做

- [x] zero-perturbation identity bypass 与逐字段 comparator；
- [x] CBF/QP fault matrix、显式 fallback、raw-action gate；
- [x] 新 T3 ledger alignment、temporal fault 和 provenance 校验；
- [x] non-zero M3 100-cycle 双次 deterministic replay；
- [x] 既有训练/校准 archive 和 TensorBoard 审计。
- [x] JEPA、ledger、ranker、CBF、环境 step 和 control-cycle 的全链路 latency instrumentation；
- [x] RTX 5050 上 20 集/1,075 cycles M3 replay、latency audit、TensorBoard gate 和 provenance 校验。

### 2.2 当前工作位置

- **已完成：** P1 全链路可观测性与实时性审计。JEPA/ledger/ranker/CBF/cycle p95 分别为
  `4.748/0.166/0.696/1.472/15.175 ms`，queue-age p95 为 `35.5 steps`；安全计数为
  collision/boundary/pairwise/CBF-timeout/raw-unverified 全部 `0`。
- **当前主问题：** P2 settled ranking 审计已完成，协议修订仍未冻结。M3 selected-not-best=`26.23%`、Spearman=`-0.517`；
  A2 selected-not-best=`25.41%`、Spearman=`-0.612`。M3 safe-capture=`8/20=40.0%`，低于旧 M0
  对照 `10/20=50.0%`，因此目前只能记为 `no_control_gain`，不能宣称 JEPA 带来任务提升。
- **P2 产物：** [P2 settled ranking audit](JEPA_SAFE_CAPTURE_P2_SETTLED_RANKING_AUDIT_20260904.md)，
  包含逐候选混淆矩阵、nominal displacement、switch、credit 和 CBF-abort pre-state，并已写入
  TensorBoard。
- **下一执行动作：** 基于 P2 证据冻结或修订 ranking/abstention protocol，再决定是否进入 P3
  校准、P4 smoke 和三 seed paired development；在此之前不扩大 episode 数、不调大模型、不打开
  locked test。

## 3. 总体执行顺序

```text
P0 冻结与预检
  -> P1 全链路 latency/provenance instrumentation
  -> P2 settled ranking 与 abstention 修订
  -> P3 安全辅助头校准与困难片段 replay
  -> P4 新 protocol smoke（每 seed/variant 20 集）
  -> P5 三 seed paired development（每 seed/variant >=40 集）
  -> P6 robustness stress 与 SIL/HIL readiness
  -> P7 统计汇总、论文表述和是否申请 locked test 的决策
```

依赖规则：P0 必须先完成；P1 可与 P2/P3 的离线分析并行；P2/P3/P1 全部通过后才进入 P4；P4 全部通过后才进入 P5；P5 安全门通过后才进入 P6。任一安全硬门失败，立即停止扩大规模并创建新 protocol revision。

## 4. P0：冻结实验合同和运行环境

### TODO

- [ ] 保留工作区现有 E1/V5/tmp 改动；禁止 `git add .`、reset 或删除无关文件。
- [ ] 从 `central_random_mixed_obstacle_s3_v5_t3_recalibration_development_protocol.yaml` 派生本阶段新 revision，记录候选、tie、margin、hold、CBF 参数和 seed。
- [ ] 创建全新的 scene manifest、results root 和 TensorBoard root；非空 output 目录必须拒绝覆盖。
- [ ] 固定 `K=5` 候选、chunk length=3、只执行第 1 步、capture radius=`0.80 m`、episode time limit 和所有 CBF 阈值。
- [ ] 将 train、validation、calibration、development 的 episode/layout/seed 分离；locked 数据继续禁止访问。
- [ ] 保存代码 revision、Conda 包清单、GPU/CUDA、命令行、protocol、actor/JEPA/ledger/checkpoint hash。
- [ ] 运行 `git diff --check`、schema tests 和当前 targeted tests。

### 预检命令

```powershell
Set-Location D:\\uav-capture\\uav_capture
$py = 'D:\\miniconda3\\envs\\uav-encirclement-gpu\\python.exe'
$env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
& $py -c "from pathlib import Path; from evaluate_random_central_mixed_obstacles import load_protocol; load_protocol(Path('configs/central_random_mixed_obstacle_s3_v5_t3_recalibration_development_protocol.yaml')); print('S3 protocol schema: OK')"
& $py -m pytest -q tests/test_jepa_v3_zero_perturbation.py tests/test_audit_jepa_safe_capture_v5_ledger_alignment.py tests/test_audit_jepa_safe_capture_fault_injection.py
git diff --check
git status --short
```

### 出口门

- [ ] `preflight.json` 存在且包含 environment/protocol/hash/locked-test 状态；
- [ ] protocol schema 通过，且运行时断言 `development_only=true`、`locked_test_opened=false`；
- [ ] targeted tests 全部通过；任一项失败不得启动 episode。

## 5. P1：补齐全链路可观测性和实时性（已完成）

### 目标

当前已有 CBF latency，但还缺少 JEPA、ledger、ranker 和全链路分解。先补齐这部分，避免将 CBF correction 或整体 episode time 误读为模型收益。

### TODO

- [x] 在单个 control cycle 记录 `belief_update_ms`、`candidate_generation_ms`、`reachability_ms`、`jepa_inference_ms`、`ledger_route_ms`、`ranker_ms`、`cbf_solve_ms`、`trace_write_ms` 和 `cycle_total_ms`。
- [x] 为每一项记录 p50/p95/p99、max、sample count、device、batch size 和 queue age。
- [x] 记录 JEPA/ledger/ranker/CBF 输入输出 finite 状态、fallback reason、ledger state、selected candidate、nominal distance、CBF status 和 active constraints。
- [x] 将 latency 与物理 deterministic comparator 分离；重复 replay 时只忽略 wall-clock 字段。
- [x] 对 CUDA warm-up、首次 inference、异常 fallback 和长序列 queue backlog 分桶统计。
- [x] 增加 TensorBoard tags：`Latency/JEPA_p95`、`Latency/Ledger_p95`、`Latency/Ranker_p95`、`Latency/CBF_p95`、`Latency/Cycle_p95`、`Fallback/*`、`Provenance/*`。
- [x] 为每个 trace 写 schema version，确保旧 trace 缺字段时审计器拒绝而不是静默填零。

### 验收门

- [x] 所有 1,000+ cycle replay trace 都能关联到唯一 episode/step/candidate；
- [x] RTX 5050 上 `cycle_total_p95` 不超过冻结控制周期预算，超时必须落入 verified nominal 或 safe-hold；
- [x] `raw_unverified_executed=0`；latency 异常不能改变 safe-capture 结算语义；
- [x] TensorBoard、JSON 和 CSV 的 cycle 数一致。

### 产物

已交付：`results/jepa_safe_capture_v5_p1_latency_m3_seed20260911/`、
`results/jepa_safe_capture_v5_tensorboard/p1_latency_m3_seed20260911/`、
`results/jepa_safe_capture_v5_tensorboard/p1_latency_audit_m3_seed20260911/`，以及
`docs/JEPA_SAFE_CAPTURE_P1_FULL_CHAIN_LATENCY_20260904.md` 和对应单测（targeted `36 passed`）。

## 6. P2：修复 settled ranking、abstention 和 nominal anchor（审计完成，协议决策待定）

### 目标

当前 selected-not-best 较高且 Spearman 为负，说明“预测变好”尚未转化为“选对候选”。这一阶段先修排序决策，不通过调大模型或降低安全门掩盖问题。

### 固定的安全优先顺序

```text
finite/shape/reachability
  -> predicted safety lower bound
  -> ledger state and credit
  -> task progress / visibility gain
  -> action smoothness and nominal distance
```

### TODO

- [x] 从现有 M3/A2 settled rows 重新计算 top-1 safety precision、selected-not-best、Spearman/Kendall、top-two margin、switch/oscillation 和 credit bucket failure。
- [ ] 将 score 显式拆分为 task progress、clearance lower-quantile、pairwise TTC risk、visibility gain、CBF intervention risk、uncertainty、action-change cost 和 nominal anchor。
- [ ] 使用保守净空下界，不使用 clearance 均值作为安全分数。
- [ ] 固定 `score_tie_tolerance_m=5e-4`、abstention margin、minimum predicted clearance、hysteresis 和 minimum hold steps；任何修改都新建 protocol/manifest/ledger revision。
- [ ] 低 credit、missing bucket、margin 不足、预测冲突或 high uncertainty 时选择 nominal-CBF 或 safe-hold，而不是只降低分数。
- [x] 保留 nominal exact anchor，记录每次 selected candidate 与 nominal 的距离、切换原因和最终 CBF 修正。
- [x] 对每个 settled decision 生成“模型选择 vs 离线最佳安全候选”的 confusion table 和可重放 JSON。
- [ ] 无法从 trace 证明的原因写成 `unresolved`，不得事后凭观察补标签。

### 验收门

- [ ] high-credit settled failure 不高于 low-credit，或结论明确写为 `no_control_gain`；
- [x] 选中候选、tie、abstention、hysteresis、fallback 和 CBF trace 可双次确定性重放；
- [x] 不能只凭 correction norm 下降宣称任务提升；safe-capture 仍需闭环验证。

## 7. P3：安全辅助头、校准和困难片段 replay

### 目标

在 held-out 结果为 `prediction_signal_mixed` 的前提下，针对 hard target degradation、clearance over/under-prediction、visibility 弱区分和 CBF risk 漏报做定向修复。

### 预测头和标签

- [ ] 保留多 horizon target displacement；
- [ ] 增加/校准 target velocity、acceleration 和 motion-mode consistency；
- [ ] 预测 obstacle/inter-agent clearance 的 lower quantile 或分布，而非只预测均值；
- [ ] 预测 visibility probability、observation age/staleness、pairwise TTC；
- [ ] 预测 CBF intervention probability、correction magnitude 和 QP feasibility；
- [ ] 输出 ensemble/MC-dropout/distributional uncertainty，并在 calibration split 上固定方法。

### TODO

- [ ] 固定现有 train/validation/calibration archive hash，禁止从 development 结果反向拟合阈值。
- [ ] 按 horizon、visibility、observation age、target motion、obstacle density、formation density 和 CBF active set 计算 MAE、P90/P95、coverage、under-estimation、Brier、ECE、AUROC。
- [ ] 对 hard replay 仅使用失败上下文的状态摘要；不得把同一 development episode 原样回灌训练 archive。
- [ ] 逐类重放急转、速度突变、flee persistence、S-curve、遮挡、通信延迟/丢包、拥挤队形和低候选分离。
- [ ] 检查辅助任务 loss 是否压制 target/action-consistency；固定 loss weights、optimizer、precision、epoch 和 seed。
- [ ] 每个训练 seed 单独保存 checkpoint、TensorBoard、manifest 和 hash；prediction gate 通过后才接入闭环。

### 验收门

- [ ] 所有输出 finite，所有安全标签非空且不系统性高估净空；
- [ ] 至少一个主要 horizon 优于 constant-velocity baseline；
- [ ] action-following separation 非零且方向一致；
- [ ] Brier/ECE/漏报按 hard bucket 有完整报告；
- [ ] 预测 gate 只标记 `prediction_signal_improved` 或 `prediction_signal_mixed`，不得直接标成 safe-capture improvement。

## 8. P4：新 protocol smoke gate

### 固定变体

| 变体 | JEPA | ledger | CBF | 用途 |
|---|---:|---:|---:|---|
| M0 | off | off | on | frozen nominal + CBF 主基线 |
| M3 | on | on | on | 完整系统候选 |
| A1 | on | off | on | ledger 消融 |
| A2 | on | on | on | 安全辅助排序消融 |
| A3 | on/off | on/off | off | 仅诊断，禁止作为安全结论 |

### TODO

- [ ] 生成新的 paired scene manifest；每个 seed 的 M0/M3/A1/A2 使用同一 episode/layout 顺序。
- [ ] 对 `20260911`、`20260912`、`20260913` 每个 seed、每个安全变体先运行 20 episodes；每个 run 独立 output/logdir。
- [ ] 运行前锁定 actor、JEPA、ledger、protocol、CBF、设备和命令 hash；不在 smoke 期间改权重或阈值。
- [ ] 运行结束立即生成 summary、episodes.csv、step traces、failure index、latency breakdown、TensorBoard audit 和 provenance manifest。
- [ ] 依次执行 aggregate、settled ranking、ledger alignment、temporal ledger、CBF audit、zero-regression comparator。

### Smoke 硬门

- [ ] M0/M3/A1/A2 的 collision、defender boundary、pairwise violation 均为 0；
- [ ] `raw_unverified_executed=0`；CBF timeout 为 0，或每次有 verified fallback；
- [ ] provenance、scene geometry、checkpoint/ledger hash 全一致；
- [ ] smoke 结果不得出现无法解释的 trace/schema 缺失；
- [ ] 任一门失败：停止进入 P5，保存完整失败产物并新建 protocol revision。

### 命令模板

```powershell
& $py scripts/evaluate_jepa_safe_capture_v2_paired.py `
  --variant m3 `
  --training-seed 20260911 `
  --episodes 20 `
  --split validation `
  --protocol <new_protocol.yaml> `
  --environment-config configs/capture_radius_pursuit_central_v4_flee.yaml `
  --actor-checkpoint <actor.pt> `
  --jepa-checkpoint <jepa.pt> `
  --reliability-ledger <ledger.json> `
  --scene-manifest <paired_scene_manifest.json> `
  --output-dir results/<smoke_m3_seed> `
  --tensorboard-dir results/jepa_safe_capture_v4_tensorboard/<smoke_m3_seed> `
  --device cuda `
  --development-only
```

M0/A1/A2 只替换 variant 和各自 output/logdir；不得为不同变体重新采样 paired manifest。

## 9. P5：三 seed paired development

### 固定设计

- [ ] 仅在 P1-P4 全部通过后执行；仍为 development-only，不打开 locked test；
- [ ] 三个 training seed：`20260911`、`20260912`、`20260913`；每 seed 使用自己的 checkpoint/ledger；
- [ ] 每个变体每个 seed 至少 40 个 paired episodes；如资源允许扩展到 60，但不能只挑高分 seed；
- [ ] 所有变体使用同一 paired scene manifest、episode index、layout 和 observation schedule；
- [ ] 顺序固定为 M0 -> M3 -> A1 -> A2；A3 只做 diagnostic；
- [ ] block 内不调整 score、credit threshold、CBF margin/gamma、chunk length、capture radius、seed 或设备。

### 主统计

统计单位是完整 `(training_seed, episode)`，不是 timestep、candidate 或 action chunk。必须报告：

- 每 seed 和总体 `safe_capture`、sample SD；
- paired improved/degraded/tied、paired delta、bootstrap 95% CI、exact McNemar；
- collision、boundary、pairwise、CBF infeasible/timeout/abort、safe-hold、nominal fallback、raw/unverified；
- 最小 obstacle/inter-agent clearance、CBF correction、candidate switch/oscillation；
- JEPA/ledger/ranker/CBF/cycle p50/p95/p99 latency；
- 按 motion mode、visibility、observation age、clearance、credit、ledger state、CBF active constraint 分桶。

### 预注册决策门

| 门 | 条件 |
|---|---|
| G-Safety | 安全保留变体无新的 collision、boundary、pairwise violation；raw/unverified 必须为 0 |
| G-Noninferiority | 三 seed mean paired safe-capture delta >= 0；任一 seed 不得出现超过预先声明的 episode-resolution 回归 |
| G-Reliability | high-credit failure <= low-credit；OOD/stale/non-finite 触发确定性 fallback |
| G-Realtime | 全链路 p95 在控制周期预算内；超时进入 verified nominal/safe-hold |
| G-Provenance | 所有结果、日志、hash、场景和命令可重建 |

若 G-Safety 通过但 G-Noninferiority 未通过，结论只能是 `prediction_signal_no_control_gain` 或 `insufficient_evidence_do_not_open_locked_test`。不设置绝对 95% 门槛，但所有 safe-capture 下降必须解释。

## 10. P6：困难场景鲁棒性与 SIL/HIL 准备

仅在 P5 安全门通过后执行：

- [ ] 目标急转、速度突变、flee persistence、S-curve；
- [ ] observation dropout、stale belief、通信延迟/丢包和可见性下降；
- [ ] 障碍数量/布局 shift、狭窄通道、拥挤队形和低 pairwise TTC；
- [ ] 单机失效、CBF 多约束同时激活、QP infeasible、进程重启、watchdog 和 hash mismatch；
- [ ] 测量 p50/p95/p99 latency、GPU 显存、CPU、消息队列积压和恢复时间；
- [ ] 验证 safe-hold -> verified nominal -> controlled-abort 的固定顺序；
- [ ] 形成 SIL/HIL 风险清单、急停/geofence/任务终止条件；
- [ ] 在安全审查完成前禁止真实飞行。

HIL 通过不等于真实飞行许可；任何实飞都必须重新审查动力学、通信、责任边界和安全合同。

## 11. 失败处理与停止规则

- **安全失败：** 任一安全保留变体出现 collision、boundary、pairwise 或 raw action，立即停止该变体和后续扩大规模。
- **CBF 失败：** QP infeasible/timeout/non-finite 必须进入 safe-hold 或 verified nominal；无法验证则 controlled-abort，不能放行 raw action。
- **ledger 失败：** high-credit failure 高于 low-credit，回到 P3；不允许通过调 task score 掩盖。
- **排序失败：** selected-not-best 或 oscillation 无改善，标记 `no_control_gain`，只做离线修订，不直接跑 full block。
- **预测失败：** hard subset degradation 或安全头漏报，保留负向证据，重新校准/重训并生成新 hash。
- **实时失败：** 全链路 p95 超预算，先优化/降级为 nominal-CBF，再决定是否继续；不以提高 capture rate 为理由放宽预算。
- **可复现性失败：** manifest、protocol、checkpoint、ledger、环境或 scene geometry 不一致，拒绝合并该 run。

## 12. 产物、命名和提交纪律

每个阶段都必须同时留下：

- `protocol.yaml`、`scene_manifest.jsonl`、`provenance.json` 和 `sha256_manifest.json`；
- `episodes.csv`、`step_trace.jsonl`、`summary.json`、`failure_index.json`；
- TensorBoard event files 和 audit JSON；
- 独立 Markdown 阶段报告，包含通过门、失败门、未决问题和停止原因。

建议命名：

```text
results/jepa_safe_capture_v5_<stage>_<variant>_seed<seed>/
results/jepa_safe_capture_v5_tensorboard/<stage>_<variant>_seed<seed>/
docs/JEPA_SAFE_CAPTURE_<STAGE>_20260904.md
```

选择性提交，禁止 `git add .`：

```text
feat(jepa): add full-chain latency provenance
feat(jepa): harden settled ranking abstention
feat(jepa): calibrate safety auxiliary heads
test(jepa): add paired smoke gates
docs(jepa): archive three-seed safe-capture development
docs(jepa): audit sil-hil readiness
```

NPZ、checkpoint、results 和 TensorBoard 默认保留本地；历史 V4/V5 报告不得覆盖。

## 13. 建议时间盒与每日检查点

| 时间盒 | 工作内容 | 当天必须交付 | 不满足时 |
|---|---|---|---|
| Day 0 | P0 冻结、环境/协议/manifest 审计 | `preflight.json`、新 protocol、hash manifest、测试日志 | 不启动 episode |
| Day 1 | P1 全链路 instrumentation | latency breakdown、schema test、TensorBoard audit | 只修日志，不扩展实验规模 |
| Day 2 | P2 settled ranking/abstention | rank manifest、selected-not-best 报告、双次 replay | 归档 `no_control_gain`，回到排序修订 |
| Day 3-4 | P3 安全头校准与 hard replay | 新 checkpoint/ledger（如确需重训）、calibration report | 保留 mixed/negative 证据，不接入闭环 |
| Day 5 | P4 新 smoke | M0/M3/A1/A2 每 seed 20 集及全部 audit | 任一硬门失败即停止扩大 |
| Day 6-8 | P5 三 seed paired development | 每变体每 seed >=40 集、aggregate、CI/McNemar | 只报告 development，不开 locked |
| Day 9+ | P6 robustness、SIL/HIL readiness | fault matrix、延迟压力、部署前风险清单 | 不进入真实飞行 |

每天结束前必须检查：`safe_capture` 与 safety counters 是否与 `episodes.csv` 一致、TensorBoard 是否有必需 tags、所有 output 是否有 provenance/hash、是否仍为 `development_only=true`。任何不一致都先修复审计链，不用手工改结果文件。

## 14. Definition of Done

本阶段只有以下条件全部满足才算完成：

1. P0 合同、数据边界、provenance 和 locked-test gate 冻结；
2. JEPA、ledger、ranker、CBF、cycle 的 latency 和 fallback 可逐 step 审计；
3. settled ranking 的 selected-not-best、credit、abstention、hysteresis 可重放；
4. 安全辅助头在独立 calibration split 有 finite、非空、不过度乐观的证据；
5. zero-regression、CBF fault matrix 和 non-zero rolling replay 仍通过；
6. M0/M3/A1/A2 三 seed smoke 全部通过安全、provenance、TensorBoard 和实时性门；
7. 三 seed、每变体至少 40 集的 paired development 完成，并按 `safe_capture` 优先报告；
8. 结果如实归类为 `safe_capture_improvement_candidate`、`safety_preserving_noninferior`、`prediction_signal_no_control_gain` 或 `insufficient_evidence_do_not_open_locked_test`；
9. 没有新的正向、多 seed、可审计 safe-capture 证据前，不打开 locked test。

最终允许的系统表述是：**action-conditioned interaction-aware JEPA 作为候选轨迹评价器，reliability ledger 负责可信度和拒答，Joint CBF-QP 负责不可绕过的安全边界，rolling horizon 负责闭环修正；是否存在任务收益只能由独立多 seed 的 safe-capture 证据决定。**
