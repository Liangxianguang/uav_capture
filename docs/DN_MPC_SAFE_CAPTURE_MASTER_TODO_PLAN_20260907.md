# DN-MPC + JEPA + CBF 安全围捕主 TODO 计划

**版本：** v1.0（development-only）
**日期：** 2026-09-07
**适用仓库：** `D:\\uav-capture\\uav_capture`
**新方向名称：** `DN-MPC-Hybrid`

## 0. 计划定位

本文件是新 DN-MPC 方向的执行主计划，建立在
[`DN_MPC_JEPA_SAFE_CAPTURE_NEW_DIRECTION_TODO_20260907.md`](DN_MPC_JEPA_SAFE_CAPTURE_NEW_DIRECTION_TODO_20260907.md)
之上，用于把设计拆成可以逐项验收的工程任务。它不是对 V3/V4/V5、M0/M3、G5
或既有 Reliability Ledger 结果的重写，也不是立即开启新的 locked benchmark。

本方向的核心思想是：

```text
delay-aware public belief
  -> obstacle-aware route candidates
  -> distributed minimax DN-MPC route selection
  -> JEPA counterfactual trajectory evaluation
  -> optional Ledger-Lite confidence routing
  -> strict Joint CBF-QP filtering
  -> execute first step and replan
```

DN-MPC 是规划与最坏情况排序器，不是安全证明器；JEPA 是候选轨迹评价器，
不直接输出动作；CBF 仍是最终动作边界。

## 1. 不冲突合同

- [ ] 保留所有历史报告、checkpoint、archive、manifest 和结果目录。
- [ ] 不修改 V4 locked evaluator、V5 released evaluator、M0/M3 统计口径或旧 Ledger。
- [ ] 新实验使用独立输出根目录、独立 TensorBoard run name、独立 checkpoint 和独立 calibration archive。
- [ ] 旧 Reliability Ledger 只用于旧 M3 对照；新方向的可信度模块命名为 `Ledger-Lite`，不能复用旧 calibration archive。
- [ ] 不降低 CBF margin，不关闭 stale/OOD/non-finite gate，不删除 `controlled_abort`，不允许 raw-unverified action。
- [ ] 不让新模型访问未来真值、仿真器内部障碍物真值或目标未来轨迹。
- [ ] 新方向所有结果标记为 `development-only`；通过三 seed paired replay、L0-L3 和审计后才可申请新的正式 benchmark。
- [ ] 如果新方向失败，只回滚新方向目录和新配置，不回滚旧路线代码与历史结果。

### 1.1 当前可引用基线

| 基线 | 当前证据 | 用途 |
| --- | ---: | --- |
| V4 random mixed S3 locked | `75.3% +/- 6.5%` | 正式历史参照 |
| V5 exact-reactive development | `57/60 = 95.0%`，单 seed | 仅开发参照 |
| 完整 M0 | `46.9%` | 新方向 paired baseline |
| 完整 M3 | `45.8% +/- 5.5%` | JEPA + Ledger 旧路线参照 |
| G5 固定 4 场景 | `4/4`，安全事件 0 | 新 planner 的第一道开发门 |

这些数字只用于比较，不能混合场景合同、seed、checkpoint 或结算定义。

## 2. Definition of Done

新方向只有同时满足以下条件，才算完成一套可复现闭环：

- [ ] DN-MPC 能在公开 belief 和已投影候选轨迹上稳定选择路线。
- [ ] 每个候选都经过 reachable projection、独立 CBF counterfactual 和最终 CBF。
- [ ] 每个控制周期只执行首步动作，然后重新观测、评价和规划。
- [ ] JEPA 只影响候选排序；JEPA 不可信时系统仍能降级到解析 DN-MPC + CBF。
- [ ] G5 固定场景不低于现有 `4/4` 开发基线，且 collision、boundary、pairwise、raw-unverified 均为 0。
- [ ] 三个独立 seed 的 paired replay 方向一致，且 safe capture 不低于对应 M0 基线。
- [ ] L0-L3 在 nominal、dropout、noise、delay 和混合障碍条件下完成分层报告。
- [ ] TensorBoard、配置、环境、代码 revision、输入 hash、checkpoint hash 和结果 JSON 可完整重放。
- [ ] 所有失败 episode 能通过 trace index 定位到路线、CBF、JEPA、Ledger 或执行合同原因。

绝对成功率目标（例如 L0/L1 超过 90%、L2 超过 85%、L3 超过 80%）保留为
研究目标，不作为绕过安全门的理由。首先要求安全事件为零并且不低于对应基线。

## 3. P0：冻结合同与工程隔离

### 3.1 基线冻结

- [x] 生成独立版本化 manifest：
  `results/dn_mpc_jepa_safe_capture_dev/baseline_manifest_g5_chunk5_seed20260907.json`。
- [x] 写入 G5 4 场景 manifest、M0/M3 manifest、环境配置、CBF 参数、capture radius、episode limit 和 seed。
- [x] 计算输入 archive、checkpoint、配置和代码 revision 的 SHA-256。
- [x] 记录当前 Python、PyTorch、CUDA、NumPy、TensorBoard 和 GPU 信息。
- [x] 保存 G5 严格 chunk-5 基线 replay 摘要，避免新方向的场景采样悄然变化。

### 3.2 输出隔离

- [ ] 结果根目录：`results/dn_mpc_jepa_safe_capture_dev/`。
- [ ] TensorBoard 根目录：`results/dn_mpc_jepa_safe_capture_tensorboard/`。
- [ ] 新 checkpoint 根目录：`results/dn_mpc_jepa_safe_capture_checkpoints/`。
- [ ] 新数据根目录：`results/dn_mpc_jepa_safe_capture_archives/`。
- [ ] 每次 run 旁边保存 `effective_config.yaml`、`provenance.json`、`summary.json` 和 `episodes.csv`。
- [ ] `.codegraph/` 只作为本地索引，不进入 Git；索引已用于确认 DN-MPC 当前尚未接入旧 evaluator。

### 3.3 P0 验收门

- [x] 旧基线 replay 可运行，结果和历史合同一致到允许的数值误差。
- [x] 新输出不覆盖任何已有文件。
- [x] 所有新 run 都能通过唯一 `run_id`、seed、git revision 和 config hash 追溯。

**不通过则停止：** 不开始训练，不接入旧闭环，不修改 CBF。

## 4. P1：统一候选轨迹接口

先统一接口，再写求解器和训练脚本。候选对象至少包含：

```text
route_id
route_family              # nominal / brake / tangent_left / tangent_right / radial_out / formation / hold
active_obstacle_id
preferred_side
action_chunk              # [horizon, defenders, 3], 已做 reachable projection
nominal_anchor
valid
projection_residual
source
```

### 4.1 候选生成

- [ ] 保留径向接近候选。
- [ ] 增加提前制动候选，使用 stopping-distance 约束。
- [ ] 增加左、右最近切向候选，记录切向方向和障碍物 ID。
- [ ] 增加 radial-out、formation expand/contract、safe intercept、verified safe-hold 候选。
- [ ] 对每个候选执行 reachable dynamics projection；投影失败的候选标记 invalid，不静默修正。
- [ ] 为每个候选生成 route identity，避免分数轻微变化造成路线抖动。
- [ ] 对候选数量、动作 shape、NaN/Inf、projection residual 写单元测试。

### 4.2 候选接口验收门

- [ ] 同一个 observation 在固定 seed 下候选顺序和 route ID 完全可重放。
- [ ] 生成器不读取目标未来真值。
- [ ] 候选动作块的每一步都能被下游 CBF 独立验证。

## 5. P2：纯解析 Distributed Minimax MPC

### 5.1 目标和假设集合

起始窗口使用 `H=5`、`dt=0.1 s`；只有解析 smoke 通过后才评估 `H=8`。
目标逃逸集合至少包括：

- 观测速度方向；
- 相对编队质心的 radial 方向；
- 左、右切向方向；
- 垂直上、下方向；
- 有界速度、加速度和观测延迟下的短时扰动。

优化目标按以下优先级实现：

```text
CBF feasibility
  > obstacle / boundary / pairwise safety
  > formation feasibility
  > worst-case escape cost
  > capture funnel progress
  > route switching and control smoothness
```

### 5.2 DN-MPC 实现任务

- [ ] 完成候选轨迹 rollout，不使用 simulator future state。
- [ ] 计算每个目标逃逸假设下的最小捕获距离和最坏逃逸代价。
- [ ] 计算 per-agent local cost、formation cost、smoothness cost 和 route switch cost。
- [ ] 输出 selected route、side、score、worst-case cost、solver status、residual 和 route age。
- [ ] 增加有限时间预算；超时返回可审计的 fallback，不静默执行未验证动作。
- [ ] 先使用确定性有限假设集合；未经对照证据，不引入不可审计的高维神经求解器。

### 5.3 当前实现状态

- [x] `src/encirclement3d/dn_mpc.py` 已建立 development-only 解析 planner。
- [x] `tests/test_dn_mpc.py` 已覆盖有效候选、无效候选、路线保持和局部成本输出。
- [x] 与 route-recovery、obstacle-route runtime 的回归为 `22 passed`。
- [x] `scripts/replay_dn_mpc_multicycle.py` 已完成 4 场景、72 周期 public-belief replay。
- [x] replay 已记录 route phase、terminal progress、stopping distance、route age 和切换原因到 JSONL/TensorBoard。
- [ ] 仍未接入旧 evaluator、JEPA 或在线 CBF runtime。

## 6. P3：路线状态机、迟滞与最近切向策略

状态机固定为：

```text
approach -> pre_brake -> tangent_left/right -> encircle -> intercept -> capture
                          \-> safe_hold
```

- [x] 保存 `active_obstacle_id`、`preferred_side`、`route_age`、`route_confidence`。
- [x] 路线最短保持时间从 2 步开始，当前 development contract 使用 3 步并覆盖测试。
- [x] 新路线只有在 score 改善超过 hysteresis margin 时才切换。
- [x] 两个已识别障碍物 ID 改变、当前路线 CBF 失效时允许提前切换；nominal/obstacle 转换仍使用迟滞。
- [x] 记录每次切换的触发原因和路线状态字段。
- [ ] `safe_hold` 只表示当前周期无可验证进攻动作，不直接宣告 episode 失败。
- [ ] controlled abort 只在全部候选和 safe-hold 均无法通过 CBF 时触发。

**P3 门：** route switch、fallback、controlled abort 不高于 G5 基线；否则先回放 trace，
不继续训练 JEPA。

**当前 P3 触发证据：** P2 mixed replay 的 route switch rate 为 `29.4%`
（5/17），因此 P3 状态机与迟滞扫描必须先完成，不能直接进入 JEPA 训练。

## 7. P4：CBF 分层接入

CBF 接入顺序固定为：

1. reachable projection；
2. candidate-level CBF counterfactual；
3. selected / nominal / safe-hold 三路独立验证；
4. 最终 Joint CBF-QP；
5. 只执行第一步；
6. 重新观测并进入下一周期。

- [x] 保持现有 obstacle、boundary、pairwise margin 和 acceleration limit。
- [x] 记录 nominal action、selected action、safe-hold action、CBF correction norm。
- [x] 记录每条约束的 active/infeasible 原因，不只记录一个 abort 标签。
- [x] 验证路线规划器不会绕过 CBF 或直接写入环境 action。
- [x] 首先只在 G5 固定 4 场景运行 DN-MPC + CBF，不接 JEPA。

**P4 门：** safe capture 不低于 G5 `4/4`，collision/boundary/pairwise/raw-unverified 为 0，
timeout 和 controlled abort 不增加。

## 8. P5：JEPA 候选评价器

### 8.1 输入和输出

输入保留 action-conditioned interaction-aware 合同：

- 当前公开观测和 message age；
- defender positions/velocities；
- target belief position/velocity；
- obstacle route identity、候选动作块和 active obstacle ID；
- 队友相对位置、速度和编队角色。

每个候选输出：

- 目标相对位移和 route progress；
- obstacle clearance、boundary clearance、pairwise clearance；
- stopping distance 和 CBF feasibility proxy；
- target visibility；
- uncertainty、OOD distance 和 rollout disagreement。

### 8.2 训练任务

- [ ] 未来目标相对位移预测。
- [ ] 未来最小障碍净空预测。
- [ ] pairwise 净空与 interaction transition 预测。
- [ ] 可见性和 observation age 影响预测。
- [ ] CBF 可行性、动作后风险和 route regret 辅助任务。
- [ ] 左右切向候选成对排序损失。
- [ ] 安全/可行性损失权重大于 capture-time 损失。

### 8.3 JEPA 接入门

- [ ] 训练、validation、calibration 按 scene seed 隔离。
- [ ] 新 checkpoint 建立新的 calibration archive 和 hash。
- [ ] pairwise hazard 在 calibration gate 未通过前只能作为低权重排序信号，不能作为安全证书。
- [ ] JEPA 排序版在 G5 不低于解析 DN-MPC + CBF；否则停止扩大模型。

## 9. P6：Ledger-Lite 可信度路由

Ledger-Lite 是可选的可信度降级层，不替代 DN-MPC，也不复制旧 Reliability Ledger：

```text
trusted  -> JEPA 参与候选排序
degraded -> 解析 DN-MPC 为主，JEPA 只提供受限辅助分数
abstain  -> 忽略 JEPA 风险预测，继续解析 DN-MPC + CBF
```

检查项：

- [ ] non-finite output；
- [ ] message age 超限；
- [ ] OOD distance；
- [ ] JEPA 与解析 rollout 分歧；
- [ ] route progress 不一致；
- [ ] calibration credit、uncertainty 和历史失败率。

`abstain` 不等于 `controlled_abort`。只有没有任何候选能通过 CBF 时才允许 abort。

## 10. P7：数据集和训练计划

### 10.1 数据采集阶段

- [ ] P7-A：无障碍和单障碍路线对，验证规划器方向和制动。
- [ ] P7-B：单个 cylinder/box/wall，采集最近左/右切向路线。
- [ ] P7-C：3-5 个混合障碍、S-curve、狭窄通道和队形变化。
- [ ] P7-D：dropout、噪声、delay、目标转向和速度突变。
- [ ] P7-E：专门采集 candidate keep/switch/abort 和 CBF infeasible hard negatives。

### 10.2 规模与切分

- [ ] pilot：每类至少 500 个 episode 或达到稳定失败模式覆盖；先用于接口 smoke。
- [ ] training：按场景族和目标逃逸模式扩展到至少 2,000 个有效窗口。
- [ ] calibration：独立 scene seed，至少 500 个窗口，不能从 training 随机切片得到。
- [ ] test：冻结场景和冻结 seed，禁止因结果不佳重采样。
- [ ] 保存 `dataset_manifest.json`、来源配置、生成 revision、seed 和 SHA-256。

### 10.3 训练纪律

- [ ] 先训练 JEPA evaluator，不训练端到端动作策略。
- [ ] 每轮训练都写 TensorBoard、effective config、checkpoint hash 和 validation summary。
- [ ] 连续两轮 calibration gate 不通过，停止扩大网络或追加数据，先分析标签/合同。
- [ ] 不把新 hard-negative checkpoint 接入旧 Ledger。

## 11. P8：TensorBoard 和审计规范

所有训练、smoke、评估和回放都必须写入 TensorBoard。至少记录：

```text
Capture/safe_capture
Capture/capture_time
Safety/collision
Safety/boundary
Safety/pairwise
Safety/raw_unverified
Safety/controlled_abort
Planning/route_switches
Planning/route_age
Planning/fallback_steps
Planning/worst_case_escape_cost
Planning/capture_cost
Planning/formation_cost
Planning/solver_time_ms_p95
Planning/projection_residual
JEPA/uncertainty
JEPA/ood_distance
JEPA/rollout_disagreement
JEPA/clearance_error
Ledger/state_counts
CBF/correction_norm
CBF/infeasible_constraint_counts
```

每个 run 的 `provenance.json` 还必须包含：

- git revision 和 dirty-file 列表；
- conda environment 名称和包版本；
- GPU/CUDA 信息；
- config、dataset、checkpoint、scene manifest hash；
- 当前 protocol 是否 `development-only`；
- stop/gate 判定和失败 trace 路径。

## 12. P9：分阶段实验矩阵

| 阶段 | 系统 | 场景 | 目的 | 通过条件 |
| --- | --- | --- | --- | --- |
| S0 | M0 + CBF | G5、L0 | 复核旧基线 | 合同一致 |
| S1 | DN-MPC + CBF | G5 4 场景 | 只验证规划器 | 安全事件 0，safe capture 不降 |
| S2 | DN-MPC + CBF | L0-L1 nominal | 验证制动和最近切向 | controlled abort 不增 |
| S3 | DN-MPC + JEPA + CBF | L0-L2 | 验证 JEPA 是否改善候选排序 | 不低于 S2 |
| S4 | DN-MPC + JEPA + Ledger-Lite + CBF | L0-L3 | 完整闭环开发验证 | calibration gate 通过 |
| S5 | S4 三 seed paired replay | 固定 manifest | 稳定性验证 | 方向一致且安全事件 0 |
| S6 | 冻结协议候选 locked block | 新场景 | 申请正式比较 | 所有审计材料齐全 |

每一阶段必须同时运行：

- success/safe_capture；
- collision、boundary、pairwise、raw-unverified；
- timeout、controlled abort、fallback；
- route switch 和 route age；
- CBF correction、solver time、projection residual；
- JEPA uncertainty/OOD；
- Ledger 状态计数。

## 13. 停止和回滚规则

出现任一条件，立即停止当前扩展并回放失败 trace：

- [ ] 任意 collision、boundary、pairwise 或 raw-unverified；
- [ ] controlled abort 或 timeout 相对对应 baseline 增加；
- [ ] DN-MPC solver 超时、候选未投影或 route identity 丢失；
- [ ] route switch 持续增加，或最近切向策略在同一障碍两侧来回切换；
- [ ] calibration gate 连续两轮不通过；
- [ ] safe capture 比对应基线下降超过 5 个百分点，且没有可量化安全收益；
- [ ] JEPA 只提高训练集分数，validation/calibration 不改善。

回滚顺序固定为：

```text
JEPA ranking -> JEPA disabled
Ledger-Lite -> abstain
DN-MPC candidate set -> smallest verified set
DN-MPC -> nominal route planner
新方向 -> 旧 M0/M3/G5 baseline
```

不得通过降低 CBF margin 或关闭 gate 来解除停止条件。

## 14. Git、提交和发布规则

- [ ] 每个阶段只提交逻辑相关文件，不提交 `results/`、TensorBoard event、checkpoint、npz 或 `.codegraph/`。
- [ ] 文档、测试、planner、配置和脚本分开形成可审计 commit。
- [ ] commit 使用 conventional commits，例如：
  - `docs(dn-mpc): add master safe-capture execution plan`
  - `test(dn-mpc): cover minimax route contract`
  - `feat(dn-mpc): add analytic candidate smoke`
  - `feat(dn-mpc): integrate verified route candidates`
- [ ] push 前检查 `git diff --check`、相关测试、配置 hash 和 README 入口。
- [ ] 网络不可用时保留本地 commit，并在报告记录未 push，不重写历史。

## 15. 立即执行顺序

接下来严格按以下顺序推进：

1. [x] 初始化本地 CodeGraph，并确认新 planner 与旧 evaluator 尚未耦合。
2. [x] 修正 DN-MPC 局部成本输出契约，完成 `22 passed` 回归。
3. [x] 新增 `scripts/smoke_dn_mpc_route_selection.py`，使用 synthetic route batch，输出 JSON 和 TensorBoard。
4. [x] 新增 `configs/dn_mpc_jepa_safe_capture_development.yaml`，明确 `development_only=true` 和 `locked_test_opened=false`。
5. [x] 完成 P0 manifest；最新严格 chunk-5 G5 结果为 `4/4`，见
   `docs/DN_MPC_P0_BASELINE_FREEZE_G5_CHUNK5_20260907.md`。
6. [x] S1 已通过（`4/4`），并完成 S2 planner-only L0/L1 slices（均 `8/8`）；见
   `docs/DN_MPC_S1_REPLAY_AND_ARTIFACT_INVENTORY_20260907.md`。接下来先审计路线状态机，再考虑 JEPA。
7. [ ] S1/S2 稳定后再采集 hard-negative archive 和训练 JEPA evaluator。
8. [x] 完成 P12 pairwise TTC label-semantics audit；结果见
   `docs/DN_MPC_P12_PAIRWISE_TTC_LABEL_AUDIT_STOP_20260908.md`。当前 TTC hazard
   与实际 strict-margin violation/CBF infeasibility 不等价，已停止继续调
   pairwise positive weight。
9. [x] P13 已完成四类显式 outcome label 合同并采集 disjoint calibration；结果见
   `docs/DN_MPC_P13_PAIRWISE_OUTCOME_LABEL_CONTRACT_STOP_20260908.md`。合同通过，
   但 valid runtime/interaction 的 strict-margin 正例为 `0%`，停止训练。
10. [x] P14 已完成 offline-only unsafe virtual probes；结果见
    `docs/DN_MPC_P14_VIRTUAL_PROBE_CALIBRATION_20260908.md`。strict-margin
    positive cell rate 为 `15.57%`、row rate 为 `40.33%`，且未执行任何
    raw-unverified action。数据可用于校准诊断，但 branch-failure 仍为 0，
    因此尚未通过 JEPA/Ledger promotion gate。
11. [x] P15 已完成独立 balanced calibration collection；结果见
    `docs/DN_MPC_P15_BALANCED_PAIRWISE_CALIBRATION_STOP_20260908.md`。采集和
    TensorBoard provenance 通过，但 runtime/interaction strict-margin 正例仍为
    `0%`，因此 promotion gate 未通过，未训练 JEPA、未创建 Ledger-Lite。
12. [ ] 新 calibration gate 通过后才训练下一版 JEPA evaluator 或实现
    Ledger-Lite；否则保留解析 DN-MPC + CBF。
13. [x] P16 已完成 P14/P15 source-bound multisource calibration bundle；结果见
    `docs/DN_MPC_P16_MULTISOURCE_CALIBRATION_BUNDLE_20260908.md`。source-binding
    和 provenance gate 通过，但 online promotion gate 仍未通过，未训练 JEPA、
    未创建 Ledger-Lite。
14. [ ] 建立独立 validation archive，验证 virtual strict-margin positives 与
    verified branch outcomes 的跨源校准后，才允许训练/在线集成。
    P17 已完成独立 source collection 和 provenance validation，但模型级
    cross-source calibration metric 尚未通过。
15. [x] 在 P17 validation 上实现并通过模型无关的 cross-source calibration
    metric；结果见 `DN_MPC_P18_SOURCE_STABILITY_AND_ROUTE_JEPA_TRAINING_20260908.md`。
    该 gate 只证明源稳定性，不授权在线集成。
16. [x] 在配对 P17 train archive 上完成一枚 development-only route-JEPA
    checkpoint；训练结果和哈希见 P18 报告。
17. [x] 完成 P18 checkpoint 的离线 candidate-ranking audit；结果见
    `DN_MPC_P19_P18_CANDIDATE_RANKING_AUDIT_20260908.md`。finite 输出和
    pairwise score direction 通过诊断阈值，但 validation top-1 只有
    `14.81%`，10/172 个 runtime groups 没有全体 defender 同时几何有效且
    首步 CBF 可行的候选；selected/nominal/safe-hold 独立 CBF trace 和
    OOD/disagreement binding 也缺失。因此 promotion gate 失败，未接入在线
    闭环。
18. [x] P20 修复 route-geometry fallback 并重采集同一 validation block；结果见
    `DN_MPC_P20_GEOMETRY_HOLD_REPAIR_AND_REAUDIT_20260908.md`。零 eligible
    group 从 `10/172` 降为 `0/172`，但 validation top-1 route-progress
    agreement 仍只有 `19.19%`，promotion gate 仍失败。
19. [x] 完成 route-progress label/score direction 审计，结果见
    `DN_MPC_P21_ROUTE_PROGRESS_LABEL_AUDIT_20260908.md`。validation 中
    `79.65%` 的 group 的 top-label gap 不超过 `0.005`，tie-aware top-1
    为 `66.28%`，但 informative-group exact top-1 只有 `31.43%`；pairwise
    agreement 为 `84.32%`。因此近似并列只能解释一部分失败，P18 仍不具备
    直接选路资格。
20. [ ] 使用 tie-aware/utility-aware route-progress 监督重新校准或训练
    evaluator；同时采集 selected/nominal/safe-hold 独立 CBF counterfactual
    和 OOD/disagreement calibration。P19-P21 全部 gate 通过后再进行三 seed
    paired replay。
21. [x] 完成 route-length utility weight sweep，结果见
    `DN_MPC_P22_ROUTE_UTILITY_WEIGHT_AUDIT_20260908.md`。calibration 按
    informative exact top-1 选择的权重为 `0.0`；validation 在该权重下仍为
    `31.43%` informative exact top-1。较大的 route-length 权重提高 pairwise
    utility direction，但没有形成可靠的直接选路证据，因此不接入在线闭环。
22. [ ] 重新定义面向捕获效用的 route label（包含进度、可行性、绕行代价和
    目标逃逸代价），在独立 calibration 后训练下一枚 evaluator；未通过
    selected/nominal/safe-hold CBF trace 和 OOD/disagreement gate 前不得三
    seed paired replay。
23. [x] 完成 P23 listwise route-JEPA 训练与审计；结果见
    `docs/DN_MPC_P23_LISTWISE_CANDIDATE_RANKING_AUDIT_20260908.md` 和
    `docs/DN_MPC_P23_LISTWISE_ROUTE_PROGRESS_AUDIT_20260908.md`。validation
    exact top-1 为 `29.65%`，informative top-1 为 `51.43%`，pairwise 为
    `83.47%`；相对 P18 有改善，但仍未通过 promotion gate。
24. [x] 完成 P24 listwise temperature=`0.005` 训练与审计；结果见
    `docs/DN_MPC_P24_LISTWISE_TEMP005_CANDIDATE_RANKING_AUDIT_20260908.md`
    和 `docs/DN_MPC_P24_LISTWISE_TEMP005_ROUTE_PROGRESS_AUDIT_20260908.md`。
    validation exact top-1 为 `36.63%`，informative top-1 为 `80.00%`，
    pairwise 为 `88.73%`。这是有效的离线改善，但 train/calibration
    仍有 zero-eligible groups，selected CBF trace 与 OOD/disagreement
    binding 仍缺失，因此保持 offline-only。
25. [ ] 实现 P25 traceability/calibration contract：在 train、validation、
    calibration 三个 split 记录 selected candidate、nominal anchor 和
    verified safe-hold 的独立 CBF counterfactual；建立 fresh OOD、rollout
    disagreement 和 abstention calibration。P25 全部 gate 通过前不得在线
    接入、创建 Ledger-Lite 或开展三 seed paired replay。

**当前第一开发目标不是追求更高的单次成功率，而是证明：在严格 CBF 和完整审计合同下，DN-MPC 能稳定选择一条可执行、少切换、面向目标的最近切向路线。**
