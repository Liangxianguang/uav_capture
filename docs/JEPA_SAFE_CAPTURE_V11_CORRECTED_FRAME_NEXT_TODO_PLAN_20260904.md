# 无人机集群对抗围捕安全增强系统
# v11 corrected-frame 下一步详细 TODO 与验收计划

**版本：** v1.1（当前执行版）
**日期：** 2026-09-04
**执行目录：** `D:\\uav-capture\\uav_capture`
**硬件：** NVIDIA RTX 5050
**Conda：** `uav-encirclement-gpu`
**实验边界：** `development_only=true`，`locked_test_opened=false`
**首要指标：** `safe_capture`
**方案：** action-conditioned interaction-aware JEPA + reliability ledger + Joint CBF-QP + rolling horizon

> 本文件是当前执行入口。目标是完成一条可审计的安全闭环，而不是追逐某一个 seed 的最高捕获率。`mean_capture_time` 只作诊断，预注册的 `95%` 不是本阶段硬门。

## 0. 当前状态与下一步立即执行队列

本计划以当前工作区和已生成产物为准。已完成的离线阶段不重复运行；下一步从 P4 ledger 构建继续。

| 阶段 | 当前状态 | 可核对证据 | 下一动作 |
|---|---|---|---|
| P0 schema/provenance | DONE | v11 protocol、collection、corrected-frame metadata 回归通过 | 保持 hash 不变 |
| P1 corrected-frame archives | DONE | train/validation/calibration 各 64 episodes、78,080 samples；三份 dataset SHA-256 已写入报告 | 只读使用 |
| P2 三 seed training | DONE | seeds `20260911/20260912/20260913`，40 epochs；三 checkpoint hash 已归档 | 只读使用 |
| P3 held-out prediction gate | DONE | 四个 horizon 均优于 constant-velocity；平均 improvement 为 `26.38%/46.98%/55.78%/60.88%` | 不据此宣称控制收益 |
| P4 reliability ledger | DONE | 三 seed ledger、aggregate、fallback 和 TensorBoard 审计通过；见 [P4 报告](JEPA_SAFE_CAPTURE_V11_CORRECTED_FRAME_LEDGER_20260904.md) | 进入 P5 CBF/rolling replay |
| P5 CBF + rolling replay | PENDING | 必须在新 ledger 生成后重新绑定和审计 | fault matrix、确定性和延迟 |
| P6 paired smoke | PENDING | 尚未使用 v11 corrected-frame ledger 完成三 seed 配对 | M0/M3/A1/A2，各 20 集 |
| P7 development validation | GATED | 只能在 P6 M3 相对 M0 非劣后进入 | 三 seed、每变体至少 40 集 |
| P8/P9 stress + final report | PENDING | locked test 仍关闭 | 压力测试、统计归档、结论分级 |

### 0.1 P4 的精确修复记录

本阶段已修改 `scripts/build_jepa_safe_capture_v2_reliability_ledger.py` 的 metadata loader：

1. 将允许的数据版本扩展为 `jepa_safe_capture_v2_p1` 与 `jepa_safe_capture_v2_p1_corrected_frame`；
2. 对 corrected-frame metadata 强制检查 `target_relative_frame=post_action_defender_position`；
3. 强制检查 `label_frame_correction_version >= 1`、`split=calibration`、locked-test 关闭和 dataset SHA-256；
4. 保持旧版本的校验逻辑不变，不能用放宽检查来兼容新版本；
5. 为两个版本各添加通过/拒绝回归测试，错误信息必须包含失败字段和实际值。

修复后已按 seed 独立生成 ledger：

```text
results/jepa_safe_capture_v5_v11_corrected_frame_ledger_seed20260911/
results/jepa_safe_capture_v5_v11_corrected_frame_ledger_seed20260912/
results/jepa_safe_capture_v5_v11_corrected_frame_ledger_seed20260913/
```

每份 ledger 必须绑定对应 checkpoint SHA-256、corrected calibration archive SHA-256、v11 protocol SHA-256、bucket 定义、阈值、设备和生成命令；输出目录非空时拒绝覆盖。

## 1. 目标和结论边界

### 1.1 最终目标

针对四架无人机围捕一个具有逃逸、急转、遮挡和通信延迟的目标，完成以下闭环：

```text
观测/通信历史
  -> interaction-aware belief
  -> 传统规划器生成可达候选 action chunks
  -> action-conditioned JEPA 反事实评价
  -> immutable reliability ledger 可信度/拒答
  -> safety-first candidate ranking
  -> Joint CBF-QP 最终安全过滤
  -> 只执行 chunk 第 1 步
  -> 重新观测并滚动重规划
```

JEPA 只能评价候选轨迹，不能生成或覆盖最终控制动作；所有实际动作（包括 nominal、fallback 和 safe-hold）必须经过同一个 Joint CBF-QP。

### 1.2 `safe_capture` 定义

一个完整 episode 只有同时满足以下条件才计为 `safe_capture=true`：

- 至少一个 defender 在时间上限内进入目标 `0.80 m` capture radius；
- 无 obstacle、target 或 defender-defender collision；
- 无 defender boundary/altitude violation；
- 无 pairwise separation violation；
- 无 CBF infeasible、timeout、unverified action 或 controlled abort 终止。

任何安全失败都不能由较短的 capture time 抵消。`target_boundary_violation` 是诊断字段，不得误记为 defender boundary failure。

### 1.3 当前证据

| 项目 | 结果 | 当前含义 |
|---|---:|---|
| v10 M0 | 每 seed `10/20 = 50%` | nominal + CBF 基线 |
| v10 M3 | `40%, 40%, 45%` | 三 seed 均低于 M0，负向 development evidence |
| v10 M3 paired delta | `-10/-10/-5 pp`，均值 `-8.33 pp` | 不得进入 40 集 validation |
| v10 安全计数 | collision/boundary/pairwise/timeout/raw-unverified 均为 0 | CBF 和 fallback 仍保持安全边界 |
| v10 settled ranking | selected-not-best 约 `26.2%--59.9%`，相关性仍可能为负 | 排序因果失配尚未解决 |
| 根因候选 | `target_relative` 训练标签使用了动作前 defender frame | v11 必须用 post-action defender frame 重新生成数据和 checkpoint |
| v11 corrected-frame archives | train/validation/calibration 各 `64 episodes x 78,080 samples`；split seed 不重叠 | 数据和 frame/provenance gate 已通过 |
| v11 三 seed checkpoint | `20260911/20260912/20260913`，40 epochs，参数 finite | checkpoint 可加载且 hash-bound |
| v11 held-out prediction | 四个 horizon 相对 constant-velocity 平均 improvement `26.38%/46.98%/55.78%/60.88%` | 证明有预测信号，不等于 safe-capture 提升 |
| v11 控制收益 | 尚未完成 corrected-frame ledger 绑定后的 paired smoke | 在 P6 前不得宣称任务提升 |

因此下一轮首先验证 corrected-frame 标签是否修复闭环排序；不先换更大的模型，不先调 score 权重，不打开 locked test。

## 2. 不可变系统合同

### 2.1 动作和规划合同

- 候选数 `K=5`：`nominal`、`intercept`、`lateral_clearance`、`formation_clearance`、`visibility_hold`。
- 每个候选是长度为 3 的 constant action chunk；每次只执行第 1 个 control step。
- 候选在进入 JEPA 前通过 finite、shape、speed、acceleration、slew 和 reachability 检查；不可达候选不进入 JEPA。
- 下一周期必须重新读取 observation、更新 belief、生成候选、评价和过滤。

### 2.2 JEPA 输出合同

JEPA 输出未来评价量，而不是控制动作：

```text
target displacement / velocity / acceleration
obstacle-clearance lower quantile
inter-agent clearance lower quantile / pairwise TTC
visibility probability / observation-age risk
CBF intervention probability / correction magnitude / QP feasibility
uncertainty / ensemble disagreement
```

预测安全量只能用于排序和 ledger 校准，不能替代 CBF 的实时几何约束。

建议的显式 score 记录为：

```text
score(k) = task_progress
         + visibility_gain
         - clearance_lower_quantile_risk
         - pairwise_ttc_risk
         - cbf_intervention_cost
         - uncertainty_penalty
         - action_change_cost
         - nominal_anchor_penalty
```

每一项都必须按候选单独记录，禁止把全体候选的 top-two gap 复制给每个候选。

### 2.3 Reliability ledger 合同

ledger 在 calibration split 上离线校准，运行时只读，并绑定：checkpoint SHA-256、protocol SHA-256、calibration archive SHA-256。

| 状态 | 触发条件 | 执行路径 |
|---|---|---|
| `trusted` | bucket 足够、credit 足够、uncertainty/stale 在阈值内 | 允许 JEPA 排序 |
| `fallback_nominal` | credit 下降、候选分离不足、预测冲突或 bucket 缺失 | frozen nominal -> CBF |
| `safe_hold` | OOD、non-finite、过期观测、hash 不一致、连续异常 | safe-hold -> CBF |
| `controlled_abort` | safe-hold/nominal-CBF 也无法验证可行 | 终止并记录，不计安全捕获 |

固定回退顺序：`separation-preserving safe-hold -> verified nominal-CBF -> controlled-abort`。禁止把 raw desired action 当最后回退。

### 2.4 CBF 和实时合同

- candidate、nominal、fallback、safe-hold 共享同一 Joint CBF-QP、margin、gamma、solver、tolerance 和 timeout。
- CBF infeasible/timeout/non-finite 时，`raw_unverified_executed` 必须保持 0。
- 记录 solver status、message、active constraints、minimum slack、correction norm、fallback reason 和 latency。
- 端到端 control-cycle p95 目标为 `<=100 ms`；超时必须有已验证 fallback。

### 2.5 模块接口和故障闭环

每个模块只读写自己的契约字段，禁止通过隐式全局变量传递 target truth 或绕过安全边界。

| 模块 | 输入 | 输出 | 强制不变量 |
|---|---|---|---|
| `BeliefBuilder` | defender 状态、target 观测/通信历史、障碍和边界、时间戳 | `BeliefState`、observation age、OOD 标志 | 不含 target ground truth；所有数值 finite |
| `CandidateGenerator` | `BeliefState`、传统规划器 nominal | 5 个 action chunks、可达性结果、拒绝原因 | `K=5`、chunk=3；无效候选不进入 JEPA |
| `JEPAEvaluator` | belief history + 每个候选 chunk | 多 horizon 目标/安全辅助预测、uncertainty、候选级 score features | 只能评价，不能输出执行动作 |
| `ReliabilityLedger` | prediction features、belief provenance、离线校准表 | `trusted`/`fallback_nominal`/`safe_hold`/`controlled_abort` + reason | checkpoint/protocol/calibration hash 绑定；运行时只读 |
| `SafetyRanker` | 可达候选、JEPA features、ledger decision、上一动作 | selected candidate 或 fallback intent | 只做排序；不得跳过 CBF |
| `JointCBFQP` | selected/fallback desired action、当前状态、障碍/边界/队形约束 | `verified_action`、solver trace、fallback/abort | 唯一执行入口；失败不返回 raw action |
| `RollingController` | observation、上一周期 trace | 单步执行结果、完整 canonical trace | 每次只执行第一步，随后重新观测和规划 |

固定的单周期伪代码如下，审计 trace 必须能逐行对应：

```text
obs <- read_sensor_and_messages()
belief <- build_belief(obs, history)
assert finite(belief) and provenance_closed(belief)
candidates <- planner.generate(belief)
valid <- reachability_filter(candidates)
prediction <- jepa.evaluate(belief.history, valid)
ledger <- reliability.route(prediction, belief.provenance)
intent <- ranker.select(valid, prediction, ledger, previous_intent)
verified <- joint_cbf_qp.solve(intent.action, belief.safety_state)
if not verified.ok:
    verified <- joint_cbf_qp.solve(nominal_or_safe_hold, belief.safety_state)
if not verified.ok:
    terminate_controlled_abort()
else:
    execute(verified.action, exactly_one_control_step)
append_canonical_trace(obs, belief, candidates, prediction, ledger, intent, verified)
```

故障到动作的确定性映射：

| 故障 | ledger 状态 | 允许的动作 | 结算 |
|---|---|---|---|
| OOD、hash mismatch、non-finite belief/prediction | `safe_hold` | safe-hold 经 CBF；失败再 nominal-CBF | 不能执行 raw；失败为 controlled abort |
| stale observation、通信丢包、uncertainty 超阈值 | `safe_hold` 或 `fallback_nominal` | 按冻结优先级选择 safe-hold/nominal，经 CBF | 记录 reason code，不隐藏在成功率中 |
| 候选不可达、候选分离不足、预测冲突 | `fallback_nominal` | frozen nominal 经 CBF | 只计已验证动作 |
| Joint CBF infeasible/timeout | 不改变原 ledger 状态 | 进入下一回退动作；再次失败 controlled abort | `raw_unverified_executed=0` |

## 3. 代码、数据和产物布局

### 3.1 必须先修复的 schema/provenance 问题

- [x] v11 protocol 从父协议完整继承 collection/generator 所需字段；至少校验 `world.half_extent_xy_m`、`archive_contract.ttc_clip_seconds`、`archive_contract.cbf_max_correction_norm_mps` 和 `archive_contract.candidate_semantics`。
- [x] 明确 `v11 protocol` 与 `jepa_safe_capture_v2_collection.yaml` 的职责：协议负责版本/哈希/世界约束，collection 负责场景和 split seed；二者的有效 hash 都写入 metadata。
- [x] 修复 archive metadata 中任何硬编码父协议路径，使其始终写入实际传入的 v11 protocol 路径和 SHA-256。
- [x] 修复 `train_jepa_safe_capture_v3.py`，在 checkpoint 和 `run_metadata.json` 保存 `training_variant=corrected_post_action_frame_v1`、`target_relative_frame`、protocol hash 和 train/validation/calibration archive hash。
- [x] 为上述字段添加回归测试；schema 不通过时禁止生成 archive、训练或建 ledger。

### 3.2 新产物根目录（不得覆盖旧结果）

```text
results/jepa_safe_capture_v2_p1_corrected_frame_train/
results/jepa_safe_capture_v2_p1_corrected_frame_validation/
results/jepa_safe_capture_v2_p1_corrected_frame_calibration/
results/jepa_safe_capture_v5_v11_corrected_frame_seed20260911/
results/jepa_safe_capture_v5_v11_corrected_frame_seed20260912/
results/jepa_safe_capture_v5_v11_corrected_frame_seed20260913/
results/jepa_safe_capture_v5_v11_corrected_frame_ledger_seed20260911/
results/jepa_safe_capture_v5_v11_corrected_frame_ledger_seed20260912/
results/jepa_safe_capture_v5_v11_corrected_frame_ledger_seed20260913/
results/jepa_safe_capture_v5_v11_corrected_frame_smoke_*
results/jepa_safe_capture_v5_v11_corrected_frame_tensorboard/
```

每个目录必须独立保存 `metadata.json`、`archive_manifest.json` 或 `run_metadata.json`、checkpoint/ledger、TensorBoard event、命令记录和 SHA-256。非空目录一律拒绝覆盖。

### 3.3 代码 ownership 与提交边界

| 工作包 | 代码入口 | 交付和测试 | 单独提交主题 |
|---|---|---|---|
| schema/metadata | `scripts/generate_jepa_v3_counterfactual_dataset.py`、`scripts/train_jepa_safe_capture_v3.py` | archive/checkpoint metadata regression | `fix(jepa): bind corrected-frame provenance` |
| ledger | `scripts/build_jepa_safe_capture_v2_reliability_ledger.py`、`scripts/build_jepa_safe_capture_v3_reliability_ledger.py`、`src/encirclement3d/reliability.py` | pass/fail loader、fault、alignment、temporal audit | `fix(jepa): accept corrected-frame ledger archives` |
| prediction/ranking | `src/encirclement3d/prediction.py`、`src/encirclement3d/jepa_safe_capture_ranker.py` | candidate-specific separation、rank consistency、tie/hysteresis | `feat(jepa): audit safe candidate ranking` |
| safety/rolling | `src/encirclement3d/cbf_qp.py`、`src/encirclement3d/execution_safety.py`、`scripts/evaluate_jepa_safe_capture_v2_paired.py` | CBF fault matrix、100-cycle replay、latency | `test(safety): verify joint cbf rolling contract` |
| experiment/report | `scripts/evaluate_jepa_safe_capture_v2_paired.py`、aggregate/audit scripts、`docs/` | paired smoke/validation、hash manifest、final memo | `docs(jepa): archive v11 development evidence` |

每个提交只包含该工作包所需的源码、测试和文档；`results/`、checkpoint、`.npz`、TensorBoard 和 `tmp/` 不进入通用提交。现有 README、E1、V5 和用户未提交修改必须原样保留。

## 4. 分阶段 TODO

### P0：环境、协议和工作区冻结

- [x] 检查工作区状态，保留用户已有 E1/V5/tmp 改动；禁止 `git add .`、`git reset --hard`、`git checkout --` 和删除 `tmp`。
- [x] 激活 `uav-encirclement-gpu`，确认 Python、PyTorch、CUDA、RTX 5050 和 TensorBoard。
- [x] 运行现有 targeted tests、`git diff --check`，记录结果到 `results/.../preflight.json`。
- [x] 完成 v11 schema/provenance 修复并计算 protocol SHA-256。
- [x] 断言 `development_only=true`、`locked_test_opened=false`、locked split 不会被读取。

**出口：** schema、环境、测试、哈希和 locked boundary 全部通过；否则停止。

### P1：corrected-frame archive 生成

- [x] 运行 archive 生成器前先通过 `tests/test_jepa_safe_capture_v2_archive.py` 和 v3 corrected-frame 回归测试。
- [x] 使用同一个 corrected-frame protocol、同一个 collection config，分别生成 train、validation、calibration。
- [x] 保持 train/validation/calibration episode seed 完全不重叠；development 和 locked 只登记，不读取。
- [x] 校验每个 archive：
  - `target_relative_frame=post_action_defender_position`；
  - `inputs`、`action_history`、所有 labels finite；
  - 每个 sample 有完整的 5-candidate group；
  - history=8、horizon=`[1,2,3,5]`、chunk=3、candidate count=5；
  - target truth 只出现在离线 label；
  - metadata、scenario manifest、source hash 和 TensorBoard 完整。
- [x] 记录样本数、episode 数、nominal fraction、label shapes、seed 列表和 archive SHA-256。

**出口：** 三个 archive 通过结构、finite、split、frame、provenance 和 TensorBoard gate。

### P2：三 seed corrected-frame 训练

- [x] 固定模型结构、loss weights、optimizer、epoch=40、batch=512、precision 和 device；只改变训练 seed。
- [x] 使用 `20260911`、`20260912`、`20260913` 三个 seed，各自新建 checkpoint 目录。
- [x] checkpoint 必须保存 corrected-frame training variant、输入协议、数据 hash、best epoch 和 TensorBoard path。
- [x] 禁止在 validation 或 development 结果上调 threshold、score 权重、CBF 参数或采样分布。
- [x] 训练后运行 checkpoint loader 和 hash binding 测试。

**出口：** 三个 checkpoint 可加载、参数 finite、metadata 完整且 hash 可复核。

### P3：held-out prediction gate

- [x] 在 validation 上与 constant-velocity baseline 比较 target displacement 多 horizon MAE。
- [x] 分别报告 velocity、acceleration、clearance lower quantile、TTC、visibility、observation age、CBF intervention、QP feasibility。
- [x] 计算 uncertainty coverage、under-estimation rate、Brier/ECE/AUROC、action-following separation 和候选级 rank consistency。
- [x] 按 motion mode、visibility、observation age、障碍密度、队形密度和 CBF active set 分桶。
- [x] 明确标签为空、系统性高估净空、非 finite 或 action-conditioning 失效的情况。

**出口：** 三 seed finite；主要 horizon 至少有可解释的预测信号；否则记录 `prediction_signal_mixed`，不得接入闭环。

### P4：v11 reliability ledger 构建

- [x] 修改 `build_jepa_safe_capture_v2_reliability_ledger.py::_load_metadata()`，允许 `jepa_safe_capture_v2_p1_corrected_frame`，并强制检查 `target_relative_frame=post_action_defender_position` 与 `label_frame_correction_version>=1`。
- [x] 为旧版本和 corrected-frame 版本分别添加 metadata pass/fail 回归测试；失败时禁止建 ledger。
- [x] 只读取 corrected calibration archive，不读取 development 或 locked 数据；验证 calibration split 与 dataset SHA-256。
- [x] 为每个 checkpoint 构建独立 ledger，写入 calibration hash、checkpoint hash、protocol hash 和 bucket 定义。
- [x] 校准 trusted/fallback/safe-hold 的 credit、minimum sample count、uncertainty、stale age、candidate separation 和风险阈值。
- [x] 注入并审计：OOD、stale observation、non-finite context、低 credit、missing bucket、候选分离消失、急转、速度突变、通信丢包。
- [x] 验证状态转移和 reason code；运行期间 ledger 文件只读，并执行 hash before/after 不变性检查。
- [x] 三个 seed 的 ledger 生成使用独立 output/logdir，未复用或覆盖旧 ledger。

**出口：** OOD/stale/non-finite 100% 显式 fallback；`raw_unverified_executed=0`；high-credit failure rate 不高于 low-credit，或明确归档为失败证据。

**P4 执行命令模板：**

```powershell
Set-Location D:\\uav-capture\\uav_capture
conda activate uav-encirclement-gpu
$py = 'D:\\miniconda3\\envs\\uav-encirclement-gpu\\python.exe'
$env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"

& $py -m pytest -q `
  tests/test_jepa_safe_capture_v2_ledger_builder.py `
  tests/test_jepa_safe_capture_v2_reliability.py

& $py scripts/build_jepa_safe_capture_v3_reliability_ledger.py `
  --checkpoint results/jepa_safe_capture_v5_v11_corrected_frame_seed20260911/checkpoint.pt `
  --dataset results/jepa_safe_capture_v2_p1_corrected_frame_calibration/counterfactual_safe_capture_v2.npz `
  --metadata results/jepa_safe_capture_v2_p1_corrected_frame_calibration/metadata.json `
  --protocol configs/central_random_mixed_obstacle_s3_v5_p2_corrected_frame_v11_development_protocol.yaml `
  --output results/jepa_safe_capture_v5_v11_corrected_frame_ledger_seed20260911/reliability_ledger.json `
  --report results/jepa_safe_capture_v5_v11_corrected_frame_ledger_seed20260911/report.md `
  --tensorboard-logdir results/jepa_safe_capture_v5_v11_corrected_frame_tensorboard/ledger_seed20260911 `
  --device cuda
```

将 seed、checkpoint、output 和 TensorBoard 路径替换为 `20260912`、`20260913` 后重复执行。每次完成后立即运行 ledger alignment、temporal ledger、fault injection、TensorBoard event 和 provenance audit；任一 gate 失败都不能进入 P5。

### P5：Joint CBF-QP 和 rolling-horizon 回归

- [ ] 运行 CBF fault matrix：infeasible、timeout、non-finite desired action、state violation、多约束压力和通信中断。
- [ ] 对 candidate、nominal、fallback、safe-hold 比较执行合同，确保所有路径都经过同一 CBF。
- [ ] 每周期检查顺序固定为 timestamp -> belief -> candidate -> reachability -> JEPA -> ledger -> rank -> CBF -> execute-first-step。
- [ ] CPU 与 RTX 5050 各重放固定 episodes；比较 termination、动作序列、CBF status 和 canonical trace hash（忽略 wall-clock 字段）。
- [ ] 统计 JEPA、ledger、ranker、CBF、cycle total 的 p50/p95/p99；浮点 tie 必须使用冻结的 tie policy。

**出口：** collision/boundary/pairwise=0、raw/unverified=0、fallback 可重放、p95<=100 ms；否则停止扩大实验。

### P6：20 集 paired smoke

- [ ] 每个 training seed 先生成一份 M0 scene manifest；同 seed 的 M3/A1/A2 复用完全相同的 manifest、episode index、layout、motion 和 observation schedule。
- [ ] 运行 M0、M3、A1、A2，每个 seed 每个变体 20 episodes；A3 仅作 raw/no-CBF 诊断。
- [ ] smoke 期间冻结 checkpoint、ledger、protocol、CBF、score、threshold、chunk length 和 seed；不允许中途调参。
- [ ] 立即生成 summary、episodes.csv、step traces、provenance、TensorBoard 和 manifest hash。
- [ ] 运行官方 aggregate、settled counterfactual、ledger alignment、temporal ledger、CBF fault、latency 和 deterministic replay audit。

**Smoke gate：** 安全计数全为零、provenance 完整、manifest paired、TensorBoard 存在、无 schema 缺失；M3 相对 M0 至少非劣才可进入 P7。

### P7：三 seed paired development validation

只有 P0-P6 全部通过才执行：

- [ ] 每个 seed、每个安全变体至少 40 个 paired episodes；可扩展到 60，但不能筛 seed 或删除失败 episode。
- [ ] 统计单位为完整 `(training_seed, episode)`，不是 timestep、candidate 或 chunk。
- [ ] 主比较 M3 vs M0；A1 去 ledger，A2 去安全辅助排序，A3 只显示 CBF 必要性。
- [ ] 报告每 seed safe-capture、sample SD、paired delta、improved/degraded/tied、bootstrap 95% CI 和 exact McNemar。
- [ ] 同时报告 collision、boundary、pairwise、CBF abort/fallback、high-credit failure、最小净空、candidate switch/oscillation 和 latency。

**进入条件：** 安全变体无新的安全硬失败，三 seed 平均 paired safe-capture delta `>=0`，至少 2/3 seed 非负，且无未解释的 trace/provenance 缺失。

### P8：困难场景和实时性压力测试

- [ ] 低可见性、消息延迟/丢包、target burst/random turn/S-curve、狭窄通道、高密度队形和多障碍压力。
- [ ] 分别统计 trusted、fallback_nominal、safe_hold、controlled_abort 的安全结算和任务结算。
- [ ] 统计 queue age、GPU warm-up、异常 fallback、长序列 backlog 和候选切换抖振。
- [ ] 任何压力场景的 raw/unverified action 都必须为 0；controlled abort 必须被保留而非过滤。

### P9：最终统计、归档和论文结论

- [ ] 生成一份不可变 aggregate manifest，列出每个输入和输出文件的 SHA-256。
- [ ] 对 JSON、CSV、TensorBoard、Markdown 做双向计数和字段一致性检查。
- [ ] 将结果归类为：`safe_capture_improvement_candidate`、`safety_preserving_noninferior`、`prediction_signal_no_control_gain`、`rejected_for_safety` 或 `insufficient_evidence_do_not_open_locked_test`。
- [ ] 报告 v10 负向证据、v11 frame 修复是否改变结论、所有失败和 unresolved 原因。
- [ ] 仅在所有 development gate、统计报告和复现实验都完成后，另行申请是否打开 locked test；本计划默认保持关闭。

## 5. 命令模板

```powershell
Set-Location D:\\uav-capture\\uav_capture
conda activate uav-encirclement-gpu
$py = 'D:\\miniconda3\\envs\\uav-encirclement-gpu\\python.exe'
$env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
& $py -m pytest -q tests/test_jepa_safe_capture_v2_archive.py tests/test_jepa_safe_capture_candidates.py tests/test_jepa_safe_capture_v2_ledger_builder.py tests/test_jepa_safe_capture_v2_paired.py
git diff --check
```

生成 archive（完成 P0 schema 修复后）：

```powershell
& $py scripts/generate_jepa_safe_capture_v2_archive.py `
  --collection-config configs/jepa_safe_capture_v2_corrected_frame_v11_collection.yaml `
  --protocol configs/jepa_safe_capture_v2_corrected_frame_v11_protocol.yaml `
  --split train `
  --output results/jepa_safe_capture_v2_p1_corrected_frame_train `
  --tensorboard-logdir results/jepa_safe_capture_v5_v11_corrected_frame_tensorboard/archive_train
```

将 `train` 替换为 `validation` 和 `calibration`，并使用不同 output/logdir。训练、建 ledger、paired evaluator 使用各自脚本的 `--output`/`--tensorboard-logdir` 或 `--output-dir`/`--tensorboard-dir`，每次都指向新的非空拒绝目录。

三 seed 训练的必要输入为：corrected train/validation `.npz` 和 metadata、corrected calibration `metadata.json`、v11 protocol、v11 training config。ledger 构建必须显式传入 checkpoint、calibration dataset、metadata、report 和 TensorBoard 目录。

## 6. 统计和报告规则

### 6.1 主结果表

| seed | variant | episodes | safe_capture | collision | boundary | pairwise | CBF abort | fallback | p95 cycle |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260911 | M0/M3/A1/A2 | 20/40 |  |  |  |  |  |  |  |
| 20260912 | M0/M3/A1/A2 | 20/40 |  |  |  |  |  |  |  |
| 20260913 | M0/M3/A1/A2 | 20/40 |  |  |  |  |  |  |  |

### 6.2 强制解释

- `safe_capture` 是唯一主任务指标；capture time、路径长度、CBF correction norm 和 latency 只在安全硬门通过后作为诊断。
- 单 seed、20 集 smoke、settled local progress、prediction MAE 和 correction norm 都不能单独证明控制收益。
- 平均值必须同时给逐 seed 值、sample SD、paired delta、bootstrap CI 和 McNemar；不得删除失败 episode 或改变分母。
- 任何报告都必须明确 `development_only=true` 和 `locked_test_opened=false`。

## 7. 停止条件和失败处理

出现以下任一情况立即停止扩大 episode 数：

- 任一安全变体出现新的 collision、defender boundary、pairwise violation；
- CBF timeout/infeasible 导致 raw/unverified action；
- v11 archive 的 frame、split、finite、candidate group 或 hash gate 失败；
- ledger OOD/stale/non-finite 没有确定性 fallback；
- CPU/CUDA 物理结果不一致且无法解释；
- smoke 中 M3 相对 M0 退化，或 settled ranking 仍显示严重反向选择；
- TensorBoard、summary、episodes、trace 和 provenance 计数不一致。

失败时必须保存完整产物，写入 failure memo，标记主因或 `unresolved`，创建新的 protocol revision 后才能继续。禁止通过降低 CBF margin、放宽 stale age、关闭 ledger、修改统计口径或选择性删 episode 追逐捕获率。

## 8. 时间盒和 Definition of Done

| 时间盒 | 任务 | 交付物 |
|---|---|---|
| Day 1 | P0 schema/provenance 和 targeted tests | v11 protocol revision、preflight |
| Day 1--2 | P1 corrected-frame archives | 三 archive、metadata、hash、TensorBoard |
| Day 2--4 | P2 三 seed training | 三 checkpoint、history、run metadata |
| Day 4--5 | P3 prediction gate + P4 ledger | prediction report、三 ledger、fallback audit |
| Day 5--6 | P5 CBF/rolling replay | fault、latency、CPU/CUDA replay report |
| Day 6--7 | P6 smoke | 12 runs、aggregate、settled/ledger/CBF audits |
| Day 8--10 | P7 paired validation | 3 seed x 4 variant x >=40 episodes |
| Day 11--12 | P8/P9 | stress report、final development decision memo |

本计划完成的定义：

1. v11 corrected-frame 数据、模型、ledger、protocol 和运行结果均可由 hash 复核；
2. JEPA 始终是候选评价器，ledger 始终可拒答，所有动作始终经过 Joint CBF-QP；
3. rolling horizon 每次只执行第一步，完整 trace 可确定性重放；
4. safe-capture 主结果按三 seed paired episode 报告，安全硬门无偷换；
5. 若无任务提升，结果明确归档为 `prediction_signal_no_control_gain` 或 `safety_preserving_noninferior`，而不是强行宣称成功；
6. 未经额外授权，`locked_test_opened=false` 始终保持不变。

**最终研究主张：** action-conditioned interaction-aware JEPA 负责反事实候选评价，reliability ledger 负责可信度和拒答，Joint CBF-QP 负责不可绕过的安全约束，rolling horizon 负责闭环修正；系统是否真正改善围捕，最终只由多 seed 完整 episode 的 `safe_capture` 证据决定。
