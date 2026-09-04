# V5 下一阶段目标与 TODO 计划书

**系统名称：** Interaction-aware Action-conditioned JEPA + Reliability Ledger + Joint CBF-QP + Rolling Horizon
**版本：** v1.1（2026-09-04，v9 三 seed smoke 之后）
**执行目录：** `D:\\uav-capture\\uav_capture`
**设备：** NVIDIA RTX 5050，CUDA 12.8，PyTorch 2.7.1+cu128
**Conda 环境：** `uav-encirclement-gpu`
**实验边界：** `development_only=true`，`locked_test_opened=false`
**第一指标：** `safe_capture`
**诊断指标：** collision、boundary、pairwise separation、CBF abort/timeout/fallback、最小净空、延迟、路径代价、`mean_capture_time`

> 本文件是当前执行入口，承接 P1 latency 和 P2 settled-ranking 证据。它不是历史 V4 locked-test 结果，也不把单 seed 的 95% 观察值当作目标。所有新运行都必须写入新的 output root，不覆盖已有 results、checkpoint、TensorBoard 或 tmp 数据。

## 1. 目标、边界和最终系统

### 1.1 研究目标

在无人机集群对抗围捕任务中，验证以下安全增强链路能否在安全不下降的前提下改善候选轨迹选择：

```text
多机观测/通信历史
  -> interaction-aware belief state
  -> 传统规划器生成动力学可行候选 action chunks
  -> action-conditioned JEPA 反事实评价
  -> immutable reliability ledger 可信度校验与拒答
  -> safety-first ranker + nominal anchor + hysteresis
  -> Joint CBF-QP 最终安全过滤
  -> 只执行 action chunk 第 1 步
  -> 重新观测、更新 belief、重新规划
```

JEPA 是轨迹评价器，只能输出预测特征和候选排序依据；它不能生成或直接执行控制动作。所有实际执行动作必须来自同一个 `Joint CBF-QP` 的 `verified=true` 返回值。

### 1.2 不可变安全合同

- `safe_capture=true` 必须同时满足：在 time limit 内至少一架 defender 进入 `0.80 m` capture radius；无 obstacle/target/defender-pair collision；无 defender boundary/altitude violation；无 pairwise separation violation；无 CBF infeasible、timeout、unverified action 或 `controlled_abort`。
- `mean_capture_time` 只用于诊断，不得抵消安全失败。
- 候选数固定 `K=5`，候选类型为 `nominal`、`intercept`、`lateral_clearance`、`formation_clearance`、`visibility_hold`。
- action chunk 固定 3 个控制步，线上只执行第 1 步，然后立即 replan；不得执行完整 chunk 的 open-loop 动作。
- 候选进入 JEPA 前必须通过 finite、shape、速度、加速度、slew 和 reachability 检查；无效候选只记录拒绝原因，不得进入 ranker。
- 固定回退顺序为 `separation-preserving safe-hold -> verified nominal-CBF -> controlled_abort`；任何 raw desired action 都不能作为回退。
- online belief 不得读取 target ground truth；ground truth 只用于离线 settled label 和 episode 结算，并标记 `offline_only=true`。
- train、validation、calibration、development、locked 数据按 episode/layout/seed 隔离；locked 数据在本阶段保持关闭。

### 1.3 当前证据快照

| 证据 | 当前结果 | 结论 |
|---|---:|---|
| P1 RTX 5050 replay | 20 episodes / 1,075 cycles；JEPA p95 4.748 ms；ledger 0.166 ms；ranker 0.696 ms；CBF 1.472 ms；cycle 15.175 ms | 实时性和链路可观测性通过 |
| P1 安全计数 | collision、boundary、pairwise、CBF timeout、raw-unverified 均为 0 | 不能替代任务收益证据 |
| P2 M3 settled ranking | selected-not-best 282/1,075 = 26.23%；Spearman -0.517；Kendall -0.435 | 排序失配是当前首要算法问题 |
| P2 M3 闭环 | 8/20 = 40.0%，旧 M0 10/20 = 50.0% | 当前结论只能是 `no_control_gain` |
| P2 reliability | high-credit 1,052 decisions，failure 11.1%；low/missing 23 decisions，failure 100% | 拒答路径有效，但低信用任务推进不足 |
| 早期 V5 95% | 单 seed、开发验证 | 仅作可复查观察，不是多 seed 结论或硬门 |

因此下一阶段先做合同冻结、配对 smoke 和审计闭环，再决定是否修改预测头或排序权重；不通过安全门时不扩大 episode 数，不打开 locked test。

## 2. 验收状态定义

每个阶段必须标记以下之一：

- `PASS`: 所有本阶段硬门通过，可以进入依赖阶段。
- `PASS_WITH_LIMITATION`: 安全和 provenance 通过，但任务收益不足；只能进入诊断或修复分支。
- `NO_CONTROL_GAIN`: safe-capture 相对 M0 没有非劣证据，禁止宣称 JEPA 提升。
- `BLOCKED_BY_SAFETY`: 出现 collision、boundary、pairwise、raw action 或不可解释的 CBF 失败，立即停止扩大规模。
- `INSUFFICIENT_EVIDENCE`: 样本、bucket、manifest、hash 或审计字段不足，必须补证据。

## 3. P0: 冻结 v9 protocol 和运行环境

**目标：** 把 P2 审计后的规则冻结为可复现输入，防止运行中临时调权重或阈值。

### TODO

- [x] 选择性提交 `configs/central_random_mixed_obstacle_s3_v5_p2_ranking_audit_freeze_v9_development_protocol.yaml`；提交前保留工作区已有 E1/V5/tmp 改动，禁止 `git add .`、reset 或删除历史产物。
- [x] 记录 protocol v9 的 hash；确认 `phase=development_only`、`locked_test_opened=false`、`not_a_replacement_for_existing_locked_benchmarks=true`。
- [ ] 固定候选和排序合同：`K=5`、chunk=3、execute-first-step、tie tolerance `0.0005 m`、top-two abstention margin `0.0015 m`、minimum clearance `0.15 m`、hysteresis `0.001 m`、minimum hold `2`。
- [x] 固定环境配置、actor checkpoint、三个 JEPA checkpoint、三个 checkpoint-bound v9 ledger；如果 ledger 的 source protocol hash 不是 v9，先生成新的 ledger revision，不能静默复用旧 ledger。
- [ ] 校验 RTX 5050/CUDA/PyTorch/Conda 包清单、Git revision、命令行和 `PYTHONPATH`。
- [ ] 运行 targeted tests、`git diff --check` 和 protocol schema 检查。

### P0 预检命令

```powershell
Set-Location D:\\uav-capture\\uav_capture
$py = 'D:\\miniconda3\\envs\\uav-encirclement-gpu\\python.exe'
$env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
& $py -c "from pathlib import Path; from evaluate_random_central_mixed_obstacles import load_protocol; p=load_protocol(Path('configs/central_random_mixed_obstacle_s3_v5_p2_ranking_audit_freeze_v9_development_protocol.yaml')); assert p['phase']=='development_only' and p['locked_test_opened'] is False; print('protocol v9: OK')"
& $py -m pytest -q tests/test_jepa_safe_capture_v2_model.py tests/test_jepa_safe_capture_v2_training.py tests/test_jepa_safe_capture_latency.py
git diff --check
```

### P0 出口门

- [ ] `preflight.json` 包含 environment、protocol、all input hashes、command、Git revision 和 locked-test 状态。
- [ ] protocol schema、targeted tests、GPU 可用性全部通过。
- [ ] 任何 hash 不一致、非空 output 覆盖风险或 locked 字段异常都停止后续运行。

## 4. P1: 三个 seed 的 paired scene manifest 与 v9 ledger 对齐

**目标：** 为 `20260911`、`20260912`、`20260913` 建立独立但可配对的 validation 场景块，确保 M0/M3/A1/A2 在同一 seed 内逐 episode 比较，并让 ledger source protocol 与 v9 完全一致。

### TODO

- [ ] 为三个 training seed 分别准备新的 20-episode smoke manifest；manifest 只包含 validation split 的 layout、observation schedule、motion mode 和 episode seed，不包含 target ground truth。
- [ ] 对每个 seed 固定同一 manifest 给 M0、M3、A1、A2；不同变体不得重新采样场景或调整 episode 顺序。
- [ ] 预先计算 manifest SHA-256 和 canonical scene hash；写入每个 run 的 provenance。
- [ ] 确认每个 JEPA checkpoint 与 ledger 的 checkpoint SHA-256 一致，ledger 的 source protocol/checkpoint/calibration hash 完整。
- [x] 三个 seed 均已使用独立 calibration archive 生成 v9-bound ledger；不得使用 development episode 反向拟合 credit 或阈值。
- [ ] 为每个 seed 建立新的 results root 和 TensorBoard root，非空目录必须拒绝写入。

### 推荐输入映射

| seed | JEPA checkpoint（待 hash 校验） | ledger 要求 |
|---:|---|---|
| 20260911 | `results/jepa_safe_capture_v3_wp2_seed20260911/checkpoint.pt` | v9-bound revision，不能仅凭文件名认定兼容 |
| 20260912 | `results/jepa_safe_capture_v3_wp2_seed20260912/checkpoint.pt` | 用同 seed checkpoint 绑定的 v9 ledger |
| 20260913 | `results/jepa_safe_capture_v3_wp2_seed20260913/checkpoint.pt` | 用同 seed checkpoint 绑定的 v9 ledger |

### P1 出口门

- [ ] 三个 manifest 均可加载，episode 数、scene hash、protocol hash 可重建。
- [ ] 每个 ledger 的 `source.checkpoint_sha256` 等于对应 JEPA checkpoint SHA-256。
- [ ] ledger OOD、stale、non-finite、unknown horizon 和 provenance mismatch 的单元测试全部通过。
- [ ] manifest、protocol、checkpoint、ledger 的 provenance 关系没有缺失或冲突。

## 5. P2: v9 20 集 paired smoke（先 M0，再 M3/A1/A2）

**目标：** 在扩大到 40 集前，验证新 v9 合同不会引入安全回归、raw action 或不可追溯输出。

### 5.1 固定变体

| 变体 | JEPA | ledger | CBF | 作用 |
|---|---:|---:|---:|---|
| M0 | off | off | on | frozen nominal + CBF 基线 |
| M3 | on | on | on | 完整候选评价系统 |
| A1 | on | off | on | ledger 消融，诊断排序对信用的依赖 |
| A2 | on | on | on | 安全辅助排序消融 |

A3/no-CBF 只能作为明确标记的诊断，禁止用来支持安全结论，默认不纳入 smoke gate。

### 5.2 运行顺序与命令模板

每个 seed 的第一条命令必须是 M0；M0 生成的 `scene_manifest.jsonl` 作为 M3/A1/A2 的唯一 manifest 来源。以下模板中的 `<seed>`、`<jepa>`、`<ledger>` 和 `<root>` 必须在运行日志中展开为绝对路径。

```powershell
Set-Location D:\\uav-capture\\uav_capture
$py = 'D:\\miniconda3\\envs\\uav-encirclement-gpu\\python.exe'
$env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"
$protocol = 'configs/central_random_mixed_obstacle_s3_v5_p2_ranking_audit_freeze_development_protocol.yaml'
$envcfg = 'configs/capture_radius_pursuit_central_v4_flee.yaml'

& $py scripts/evaluate_jepa_safe_capture_v2_paired.py `
  --variant m0 --training-seed <seed> --episodes 20 --split validation `
  --protocol $protocol --environment-config $envcfg `
  --actor-checkpoint models/v5_development_exact_reactive_seed661606.pt `
  --output-dir results/jepa_safe_capture_v5_p2_smoke_m0_seed<seed> `
  --tensorboard-dir results/jepa_safe_capture_v5_tensorboard/p2_smoke_m0_seed<seed> `
  --device cuda --development-only

$manifest = 'results/jepa_safe_capture_v5_p2_smoke_m0_seed<seed>/scene_manifest.jsonl'
& $py scripts/evaluate_jepa_safe_capture_v2_paired.py `
  --variant m3 --training-seed <seed> --episodes 20 --split validation `
  --protocol $protocol --environment-config $envcfg `
  --actor-checkpoint models/v5_development_exact_reactive_seed661606.pt `
  --jepa-checkpoint <jepa> --reliability-ledger <ledger> `
  --scene-manifest $manifest `
  --output-dir results/jepa_safe_capture_v5_p2_smoke_m3_seed<seed> `
  --tensorboard-dir results/jepa_safe_capture_v5_tensorboard/p2_smoke_m3_seed<seed> `
  --device cuda --development-only
```

对 A1/A2 只替换 `--variant`、output-dir 和 TensorBoard dir；不得重新生成 manifest。每个 run 必须保留完整 stdout/stderr 和退出码。

### 5.3 每个 smoke run 的必备产物

- `summary.json`：safe_capture 和所有安全计数；
- `episodes.csv`：每个 episode 一行，不能只保留聚合值；
- `step_traces/` 或等价 `step_trace.jsonl`：每个 control cycle、候选、ledger、ranker、CBF 和执行状态；
- `scene_manifest.jsonl`：运行时拷贝，hash 必须与输入 manifest 一致；
- `provenance.json` 和 `sha256_manifest.json`；
- TensorBoard event file，包含 Config、Provenance、Safety、Reliability、Ranking、Latency 标签；
- latency audit、ledger audit、CBF/fallback audit 和失败索引。

### 5.4 Smoke 硬门

- [x] M0/M3/A1/A2 三 seed 的 collision、defender boundary、pairwise violation 均为 0。
- [x] `raw_unverified_executed=0`；CBF timeout 为 0；CBF infeasible 均记录为 controlled-abort/fallback，并单独计数。
- [ ] 所有执行 action finite；所有 `verified=true` 动作确实来自同一 Joint CBF-QP。
- [ ] candidate rejection、ledger state/reason、ranker selection、CBF diagnostics 在每个 cycle 都存在。
- [x] scene/protocol/actor/JEPA/ledger/environment hash 全部一致；TensorBoard、JSON、CSV cycle/episode 数一致。
- [ ] 任何硬门失败立即停止该 seed 的扩大运行，保留产物并创建新的 protocol revision；不得手改 summary 或删除失败 episode。

## 6. P3: smoke 聚合审计与排序诊断

**目标：** 先回答“选出的候选是否更接近安全最优”，再讨论任务率。

### TODO

- [x] 对 M0/M3/A1/A2 完成显式矩阵聚合；聚合器拒绝历史 v2/v3 目录，并强制检查每个 run 的 step traces 与 TensorBoard event file。
- [x] 对 M3/A2 三个 seed 运行 settled counterfactual audit，逐 `(episode_index, step)` join online trace 和 settled rows。
- [x] 计算 selected-not-settled-best、top-1 safety precision、Spearman、Kendall、top-two margin、nominal displacement、switch/oscillation。
- [x] 按 `high credit (>=0.65)`、low/missing credit、OOD、stale、non-finite、visibility、TTC、CBF active-set 分桶。
- [ ] 统计 ranker abstention、safe-hold、nominal fallback、CBF abort 前状态和最终 CBF correction；不要只看 correction norm。
- [x] 双次重放同一 run，比较除 wall-clock 外的逐字段结果；差异必须有明确分类和报告。
- [x] 对每个安全变体运行 ledger alignment、temporal ledger、CBF/fault、latency 和 provenance audit。

### P3 解释规则

- high-credit failure 高于 low-credit，或 settled ranking 仍为负相关：标记 `no_control_gain`，进入排序修复分支。
- low/missing credit 必须确定性回退；若仍执行 JEPA 选择或 raw action，标记 `BLOCKED_BY_SAFETY`。
- safe-capture 上升但安全计数非零：不能称为改进，只能标记安全失败。
- safe-capture 与 M0 接近但证据不足：标记 `INSUFFICIENT_EVIDENCE`，扩大样本前先补 provenance。

### v9 smoke 结果

v9 三 seed、四安全变体共 240 个 episode 的显式聚合已经完成。M0 为 `30/60=50.0%`，M3 为
`25/60=41.7%`，A1 为 `25/60=41.7%`，A2 为 `27/60=45.0%`。M3 相对 M0 的 seed delta 为
`-10/-10/-5 pp`，episode 配对为 improved/degraded/tied=`3/8/49`，bootstrap 95% CI 为
`[-10.0,-5.0] pp`。collision、defender boundary、pairwise、CBF timeout 和 raw-unverified
均为 0；CBF infeasible/controlled-abort 被显式记录，不能从 safe-capture 中删除。

因此 P3/P2 smoke 的任务结论为 `useful_safety_fallback_only` / `no_control_gain`，不能进入 40 集
validation，也不能打开 locked test。下一动作是冻结这批负向证据，进入排序/abstention 或安全辅助头
修复，建立新 protocol 和新 checkpoint 后再重跑 smoke。

## 7. P4: 40 集三 seed paired validation

**前置条件：** P0-P3 所有安全、结构和 provenance 硬门通过。smoke 通过后创建新的 40 集 protocol revision 和新的三份 manifest；不能把 20 集 manifest 重复拼接成 40 集。

### TODO

- [ ] 新 protocol 只改变 `episodes_per_split.validation=40` 或明确记录的必要变更；所有候选/CBF/ledger 规则保持不变。
- [ ] 重新为 `20260911/20260912/20260913` 生成 40 episode paired manifests。
- [ ] 每 seed 按 M0 -> M3 -> A1 -> A2 顺序运行；每变体 40 集，保存独立 output/TensorBoard/provenance。
- [ ] 汇总前运行结构一致性脚本，确认同 seed 的四个变体拥有相同 canonical manifest。
- [ ] 按完整 episode 统计，不把 timestep、candidate 或 control cycle 当作样本单位。

### 主统计

- 每 seed 和总体 `safe_capture`、sample SD、paired delta；
- paired improved/degraded/tied、bootstrap 95% CI、exact McNemar；
- collision、boundary、pairwise、CBF infeasible/timeout/abort、safe-hold、nominal fallback、raw/unverified；
- 最小障碍/机间净空、CBF correction、候选切换和 oscillation；
- JEPA/ledger/ranker/CBF/cycle p50/p95/p99 latency 与 queue age；
- motion mode、visibility、observation age、obstacle density、credit、ledger state、CBF active constraint 分桶。

### P4 决策门

| 门 | 条件 |
|---|---|
| G-Safety | 所有安全保留变体无 collision、boundary、pairwise；raw/unverified 为 0 |
| G-Noninferiority | 三 seed mean paired safe-capture delta >= 0；不能用单个高分 seed 抵消其它 seed 的下降 |
| G-Reliability | OOD/stale/non-finite 确定性 fallback；high-credit failure 不高于 low-credit，或如实报告失败 |
| G-Realtime | 全链路 p95 在控制周期预算内；超时进入 verified fallback |
| G-Provenance | 四变体、三 seed 的协议、场景、checkpoint、ledger、代码和环境可重建 |

这里不设“必须达到 95%”的绝对门槛。若 G-Safety 通过但 G-Noninferiority 未通过，最终表述只能是 `prediction_signal_no_control_gain` 或 `insufficient_evidence_do_not_open_locked_test`。

## 8. P5: 只有必要时才做模型/排序修复

如果 P3/P4 仍显示排序失配，按以下顺序修复。每一项都必须创建新的 protocol、dataset/archive、checkpoint、ledger revision、scene manifest 和 TensorBoard run。

### 8.1 排序器修复

- [ ] 把 score 分解为 task progress、clearance lower-quantile、pairwise TTC risk、visibility gain、CBF intervention risk、uncertainty、action-change cost 和 nominal anchor。
- [ ] 使用净空 lower bound/低分位数，不使用 clearance 均值作为安全分数。
- [ ] 低信用、margin 不足、预测冲突或高不确定性时显式 abstain 到 nominal-CBF 或 safe-hold，而不是只降低一个分数。
- [ ] 保持 tie tolerance、hysteresis 和 minimum hold 的协议绑定；修改后重新做 settled audit 和 zero-perturbation regression。

### 8.2 安全辅助头与校准

- [ ] 保留多 horizon target displacement，同时校准 target velocity、acceleration、motion-mode consistency。
- [ ] 增加 obstacle/inter-agent clearance lower-quantile、visibility probability、observation age、pairwise TTC、CBF intervention probability、correction magnitude 和 QP feasibility 预测。
- [ ] 使用 ensemble、MC-dropout 或 distributional uncertainty，并在独立 calibration split 固定校准方法。
- [ ] 按 horizon、visibility、observation age、target motion、obstacle density、formation density、CBF active set 报告 MAE、P90/P95、coverage、under-estimation、Brier、ECE、AUROC。
- [ ] 困难片段只使用失败上下文摘要进行 replay；不得把同一 development episode 原样回灌历史训练 archive。
- [ ] 覆盖急转、速度突变、flee persistence、S-curve、遮挡、通信延迟/丢包、拥挤队形和低候选分离。

### 8.3 修复后的准入门

- [ ] 所有 head 输出 finite，安全标签非空，且不存在系统性高估净空。
- [ ] 至少一个主要 horizon 优于 constant-velocity baseline。
- [ ] action-following separation 非零且方向一致。
- [ ] hard bucket 的 Brier/ECE/漏报率有完整报告。
- [ ] prediction gate 只能标记 `prediction_signal_improved` 或 `prediction_signal_mixed`，不能直接标记 safe-capture improvement；必须重新进入 P2-P4 闭环验证。

## 9. P6: 鲁棒性、SIL/HIL 和部署前安全审查

仅在三 seed paired validation 的 G-Safety、G-Provenance 和实时性门通过后执行：

- [ ] 目标急转、速度突变、flee persistence、S-curve、长时间遮挡；
- [ ] detection dropout、stale belief、通信延迟/丢包、观测噪声和 visibility 下降；
- [ ] 障碍数量/布局 shift、狭窄通道、拥挤队形、低 pairwise TTC；
- [ ] 单机失效、CBF 多约束同时激活、QP infeasible/timeout、进程重启、watchdog 和 hash mismatch；
- [ ] 测量恢复时间、GPU 显存、CPU、消息队列积压和各级 fallback 延迟；
- [ ] 验证 `safe-hold -> verified nominal -> controlled-abort` 顺序和急停/geofence/任务终止条件；
- [ ] 形成 SIL/HIL 风险清单。HIL 通过不等于真实飞行许可，真实飞行前必须重新审查动力学、通信和责任边界。

## 10. 失败处理和停止规则

- 出现任一 collision、boundary、pairwise 或 raw-unverified：立即停止当前变体和后续扩大规模，标记 `BLOCKED_BY_SAFETY`。
- CBF QP infeasible/timeout/non-finite：只能 safe-hold 或 verified nominal；二者都不能验证时 controlled-abort，绝不能放行 raw desired action。
- ledger provenance/hash 不一致：拒绝运行或拒绝合并结果；不得手工修改 ledger JSON。
- high-credit settled failure 高于 low-credit：回到排序/校准阶段，不通过调低安全权重掩盖。
- safe-capture 下降：保留完整失败证据，报告配对 delta 和主因，不用 `mean_capture_time` 掩盖。
- CPU/CUDA decision drift：只在最终 RTX 5050 上做主任务率结论，跨设备结果作为安全等价/限制报告，不混合统计。
- 任一 output 缺少 episode、step trace、TensorBoard 或 hash：标记 `INSUFFICIENT_EVIDENCE`，不得进入 aggregate。

## 11. 产物命名、提交和复现纪律

每个阶段必须留下：

```text
protocol.yaml
scene_manifest.jsonl
summary.json
episodes.csv
step_traces/ 或 step_trace.jsonl
provenance.json
sha256_manifest.json
latency_audit.json
ledger_audit.json
cbf_fallback_audit.json
TensorBoard event files
独立 Markdown 阶段报告
```

建议命名：

```text
results/jepa_safe_capture_v5_p2_smoke_<variant>_seed<seed>/
results/jepa_safe_capture_v5_p4_validation_<variant>_seed<seed>/
results/jepa_safe_capture_v5_tensorboard/p2_smoke_<variant>_seed<seed>/
docs/JEPA_SAFE_CAPTURE_V5_<PHASE>_20260904.md
```

代码和文档选择性提交，示例：

```text
chore(jepa): freeze p2 ranking development protocol
test(jepa): add paired smoke safety and provenance gates
feat(jepa): calibrate lower-quantile safety heads
feat(jepa): harden abstention and nominal anchor
docs(jepa): archive three-seed safe-capture development
```

禁止提交或覆盖：大型 NPZ、checkpoint、results、TensorBoard、历史 locked 报告和 `tmp` archive。需要清理 tmp 时另行确认并先做 hash 归档。

## 12. 时间盒和 Definition of Done

| 时间盒 | 交付物 | 通过条件 |
|---|---|---|
| Day 0 | protocol v9、preflight、hash manifest | development-only、locked closed、targeted tests pass |
| Day 1 | 三 seed manifest、v9-bound ledger | checkpoint/ledger/protocol hash 完整 |
| Day 2-3 | M0/M3/A1/A2 各 seed 20 集 smoke | 全部安全硬门和 provenance 门通过 |
| Day 4 | settled/ranking/ledger/CBF/latency aggregate | 可重放，明确 no-control-gain 或进入 validation |
| Day 5-7 | 三 seed、每变体 40 集 validation | safe_capture 优先的 paired 统计和 CI |
| Day 8+ | 困难场景鲁棒性、SIL/HIL 审查 | 无新的安全/审计缺口 |

本计划完成的最低定义是：

1. v9 合同、三 seed paired manifest 和所有输入 hash 冻结；
2. JEPA、ledger、ranker、CBF、rolling horizon 的每步输出可审计；
3. M0/M3/A1/A2 三 seed smoke 完成且无安全硬门失败；
4. 40 集 validation 按完整 episode 报告 safe-capture、配对 delta 和不确定性；
5. 任何正向结论都同时满足 safety、reliability、realtime、provenance 和多 seed 非劣门；
6. 在新的多 seed、可审计 safe-capture 证据出现前，保持 `locked_test_opened=false`。

最终系统表述应保持为：**action-conditioned interaction-aware JEPA 作为候选轨迹评价器，Reliability Ledger 负责可信度和拒答，Joint CBF-QP 负责不可绕过的安全边界，rolling horizon 限制预测漂移；任务收益只能由独立多 seed 的 safe-capture 配对证据决定。**
