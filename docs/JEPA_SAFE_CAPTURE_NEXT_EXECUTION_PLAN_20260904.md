# 无人机集群对抗围捕安全增强系统
# 下一步详细 TODO 与执行计划

**版本：** v2.1（WP-6 tie3 安全等价闭合版）
**日期：** 2026-09-04
**适用环境：** Windows、Conda `uav-encirclement-gpu`、NVIDIA RTX 5050
**实验范围：** development-only；`locked_test_opened=false`
**第一指标：** `safe_capture`
**次要指标：** collision/boundary/pairwise、CBF abort/fallback、净空、延迟、路径代价、capture time

> 本文件是当前唯一的下一步执行入口。它将 action-conditioned interaction-aware JEPA 作为候选轨迹评价器，与 reliability ledger、Joint CBF-QP 和滚动时域闭环组合。任何任务性能结果都必须在安全、可靠性、可复现性和 provenance 门全部通过后才可解释；`mean_capture_time` 不能抵消 safe-capture 或安全失败。

## 1. 目标和系统边界

### 1.1 研究目标

针对四机协同围捕一个具有逃逸行为的对抗目标，建立如下闭环：

```text
观测/通信历史
    -> interaction-aware belief state
    -> 传统规划器生成动力学可行候选 action chunks
    -> action-conditioned JEPA 反事实预测
    -> reliability ledger 可信度/拒答判定
    -> 安全优先候选排序
    -> Joint CBF-QP 过滤
    -> 仅执行第一控制步
    -> 重新观测、更新 belief、重新规划
```

必须证明：

- JEPA 只评价和排序候选轨迹，不能生成或覆盖最终控制动作。
- ledger 能在预测漂移、OOD、stale、non-finite 和信用不足时拒答并回退。
- 所有 candidate、nominal 和 fallback 都经过同一个 Joint CBF-QP。
- CBF/QP 失败时绝不执行 raw 或 unverified action。
- 每个控制周期只执行 action chunk 的第一步，随后重新观测和规划。
- 多 seed、困难场景和完整 provenance 下，安全增强系统的 `safe_capture` 不劣于冻结 nominal + CBF；有提升时必须有配对证据。

### 1.2 safe-capture 定义

一个 episode 只有同时满足以下条件才计为 `safe_capture=true`：

1. 至少一个 defender 进入目标 `0.80 m` capture radius；
2. 在 time limit 内完成；
3. 无 obstacle、target 或 defender-defender collision；
4. 无 defender boundary/altitude violation；
5. 无 pairwise separation violation；
6. 无 CBF infeasible、timeout、unverified action 或 controlled-abort 终止。

target 越界必须单独记录为 target diagnostic，不得冒充 defender boundary failure。

## 2. 当前证据快照

| 工作包 | 当前状态 | 已有证据 | 下一步要求 |
|---|---|---|---|
| WP-B2 deterministic failure replay | 已完成 | 18 个代表性失败样本；六类各 3 个；双次 canonical trace hash 一致 | 将报告和 hash 清单纳入最终 provenance |
| WP-E candidate audit | 已完成 | 6577 个 ranking steps；五类 candidate `valid_fraction=1.0`；无 invalid candidate 进入 `eligible_mask` | 保留 rejection reason；补 settled counterfactual rank 统计 |
| WP-D/F fault injection | 已完成 | 6 个 CBF 场景、5 个 ledger 场景；finite fallback；`raw_unverified_executed=0`；p95 < 100 ms | 补滚动闭环多约束压力和通信故障矩阵 |
| WP-6 CPU/CUDA 基础 replay | 已完成但有风险 | 20/20 settled safety outcomes 相同；安全失败均为 0；原始 9/820 ranking steps 有浮点 drift | 已建立 tie3 protocol |
| WP-6 CUDA/CPU tie3 replay | 安全等价已闭合 | 两侧均 `safe_capture=7/20=35.0%`；安全失败均为 0；CBF abort 12；p95 <= 25.82 ms | 记录 3/20 episode 的已知 decision drift，固定 final 在 RTX 5050 |
| 三 seed final development block | 尚未开始 | 当前仍只有 smoke/replay 证据 | WP-6 tie3 安全出口和所有前置 gate 通过后再开始 |

当前 tie3 结果的定位是 **development replay**，不是三 seed 任务结论。历史 boundary-fixed smoke 中 M3 低于 M0 的结果仍需保留为负向开发证据，不能通过改写统计或选择 seed 消除。CPU/CUDA 只保证安全结算等价，不声称逐步动作 bitwise 等价。

## 3. 不可变合同

### 3.1 数据和运行边界

- 在线输入只能包含 defender 状态、target belief、观测/通信历史、障碍几何、边界、动作历史和时间戳年龄。
- target ground truth 只能用于离线标签和 episode 结算，并标记 `offline_only=true`。
- train、validation、calibration、development 和 locked episode 按 episode/layout seed 隔离。
- development 失败片段不得直接回灌训练；若要重训，必须新建 archive、protocol 和 checkpoint。
- 每次运行必须保存 protocol、checkpoint、ledger、scene、代码 revision、环境版本、命令行和 SHA-256。

### 3.2 候选动作合同

- 候选数固定 `K=5`：`nominal`、`intercept`、`lateral_clearance`、`formation_clearance`、`visibility_hold`。
- action chunk 固定为 3 个 control steps，执行第 1 步后立即 replan。
- 候选在进入 JEPA 前必须通过 finite、shape、speed、acceleration、slew 和 reachability 检查。
- 不可达 candidate 不得进入 JEPA，必须记录 `candidate_rejection_reasons`。
- 使用 `score_tie_tolerance_m=5e-4` 的确定性 tie policy；最终 protocol、代码和测试必须绑定同一版本。

### 3.3 安全回退合同

固定回退顺序：

```text
separation-preserving safe-hold
    -> verified nominal through Joint CBF-QP
    -> controlled abort
```

每次回退记录 reason code、ledger state/credit、solver status、feasibility、active constraints、slack、correction norm 和 latency。raw desired action 不得作为最后回退。

## 4. 立即执行清单：先闭合 WP-6

以下步骤必须按顺序执行。每一步使用新的输出目录，不覆盖已有结果。

### T0：运行前只读检查

- [ ] 确认当前工作目录为 `D:\\uav-capture\\uav_capture`。
- [ ] 确认 RTX 5050、CUDA、PyTorch、Conda 环境和 `PYTHONPATH`。
- [ ] 确认 protocol 的 `phase=development_only`、`locked_test_opened=false`。
- [ ] 检查 git diff，确认 tie-policy 变更没有混入无关 E1/V5 文件。
- [ ] 记录 protocol、actor、JEPA、ledger 和 scene manifest 的 SHA-256。

```powershell
Set-Location D:\\uav-capture\\uav_capture
$py = 'D:\\miniconda3\\envs\\uav-encirclement-gpu\\python.exe'
$env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
git status --short
git diff --check
```

### T1：运行 CPU tie3 replay

- [x] 使用与 CUDA tie3 完全相同的 scene manifest、actor checkpoint、JEPA checkpoint、ledger 和 protocol。
- [x] 使用全新目录 `results/jepa_safe_capture_v3_wp6_replay_m3_seed20260911_cpu_tie3/`。
- [ ] 保持 `--development-only`；不要传入 locked split。

```powershell
& $py scripts/evaluate_jepa_safe_capture_v2_paired.py `
  --variant m3 `
  --training-seed 20260911 `
  --episodes 20 `
  --split validation `
  --protocol configs/central_random_mixed_obstacle_s3_v3_development_protocol.yaml `
  --environment-config configs/capture_radius_pursuit_central_v4_flee.yaml `
  --actor-checkpoint models/v5_development_exact_reactive_seed661606.pt `
  --jepa-checkpoint results/jepa_safe_capture_v3_wp2_seed20260911/checkpoint.pt `
  --reliability-ledger results/jepa_safe_capture_v3_wp3_ledger_seed20260911/reliability_ledger.json `
  --scene-manifest results/jepa_safe_capture_v3_wp6_smoke_m3_seed20260911_boundaryfixed/scene_manifest.jsonl `
  --output-dir results/jepa_safe_capture_v3_wp6_replay_m3_seed20260911_cpu_tie3 `
  --tensorboard-dir results/jepa_safe_capture_v3_tensorboard/wp6_replay_m3_seed20260911_cpu_tie3 `
  --device cpu `
  --development-only
```

不得重新采样场景或手改 manifest。

### T2：审计 CPU/CUDA tie3 等价性

```powershell
& $py scripts/audit_jepa_safe_capture_device_replay.py `
  --cuda-run results/jepa_safe_capture_v3_wp6_replay_m3_seed20260911_cuda_tie3 `
  --cpu-run results/jepa_safe_capture_v3_wp6_replay_m3_seed20260911_cpu_tie3 `
  --output-dir results/jepa_safe_capture_v3_wp6_device_replay_audit_tie3_current `
  --tensorboard-logdir results/jepa_safe_capture_v3_tensorboard/wp6_device_replay_audit_tie3_current `
  --development-only
```

通过条件：

- [ ] 20/20 episode 的 settled safety outcome、termination reason 和 CBF verification 计数一致；
- [ ] collision、defender boundary、target boundary、pairwise violation 均为 0；
- [ ] `raw_unverified_executed=0`，所有动作 finite；
- [ ] candidate rejection reason 字段在全部 ranking steps 存在；
- [x] tie3 后 settled safety outcomes 和 CBF status 完全一致；3/20 episode 的 candidate/ledger decision drift 已报告，final 主实验固定在 RTX 5050，不混用 CPU/CUDA 任务率；
- [ ] CPU/CUDA 端到端 p95 latency 均不超过 100 ms；
- [ ] 两侧 provenance、scene hash、protocol hash 和 checkpoint hash 完整一致。

### T3：运行针对性测试

```powershell
& $py -m pytest -q `
  tests/test_jepa_safe_capture_candidates.py `
  tests/test_audit_jepa_safe_capture_candidate_ranking.py `
  tests/test_audit_jepa_safe_capture_fault_injection.py `
  tests/test_audit_jepa_safe_capture_device_replay.py `
  tests/test_replay_jepa_safe_capture_failures.py
```

- [ ] 测试通过后再运行完整 `pytest`。
- [ ] 任一测试失败时，保留失败输出，修复代码/测试并新建 protocol revision；不得修改旧结果。

### T4：更新报告、提交和可追溯性

- [x] 更新 `docs/JEPA_SAFE_CAPTURE_WP6_DEVICE_REPLAY_20260904.md`，明确 tie3 输入、结果和 decision drift。
- [x] 更新本计划的 WP-6 状态和实际产物路径。
- [ ] 检查 JSON、CSV、step traces、TensorBoard 和 Markdown 中的 episode 数、safe-capture、安全计数一致。
- [ ] 只提交 tie-policy 相关代码、配置、测试和报告；禁止 `git add .`。
- [ ] 创建独立 conventional commit；push 失败时保留本地 commit 和失败原因。

WP-6 的安全闭环出口在 T0--T4 通过后闭合。跨设备 decision drift 作为已知限制归档，未闭合前不得启动 final block；tie3 审计已通过该安全出口。

## 5. 三 seed paired development final block

### 5.1 前置准入

运行前必须在 checklist 中确认：

- [ ] WP-B2 replay 双次 hash 一致，所有失败有唯一主因或 `unresolved`；
- [ ] WP-E candidate 可达性通过，invalid candidate 不进入 JEPA；
- [ ] WP-D/F fault injection 中 OOD、stale、non-finite、QP infeasible/timeout 均显式 fallback，raw=0；
- [x] WP-6 tie3 CPU/CUDA 安全结算审计通过；
- [ ] protocol、scene manifest、checkpoint、ledger、solver 和 tie tolerance 已冻结；
- [ ] `locked_test_opened=false` 且 split 仅为 `validation`；
- [ ] 没有待解释的安全回归或 provenance 缺失。

若任一项不满足，选择“修复并新建 development protocol”或“归档为不足证据/任务回归”，不能为了凑 episode 数量直接进入 final。

### 5.2 固定实验矩阵

| 变体 | 含义 | 用途 |
|---|---|---|
| M0 | nominal planner + Joint CBF-QP | 安全保留基线和主比较对象 |
| M3 | JEPA + reliability ledger + auxiliary safety ranking + CBF | 主方法 |
| A1 | M3 去除 ledger + CBF | ledger 消融 |
| A2 | M3 去除 clearance/visibility ranking + CBF | 辅助安全排序消融 |
| A3 | raw/no-CBF | 仅诊断 CBF 必要性，不进入安全结论 |

- [ ] training seed 固定为 `20260911`、`20260912`、`20260913`。
- [ ] 每个安全变体每个 seed 至少 40 个 episode；总计 M0/M3/A1/A2 为 480 个配对 episode。
- [ ] A3 如运行，使用同一 paired block、独立目录，并明确标记 `diagnostic_only=true`。
- [ ] 变体之间使用相同 episode index、layout、target motion、observation condition 和初始状态。
- [ ] 每个 seed/variant 使用独立 results 和 TensorBoard 目录。

### 5.3 场景 manifest 规则

1. 先在全新目录生成一次 40-episode canonical validation manifest。
2. 对同一 paired block 的其他变体只读复用该 manifest；不得在变体间重新采样。
3. 三个 training seed 必须共享相同场景规格和 episode index；manifest hash 变化即停止。
4. manifest 必须覆盖 nominal、delayed/noisy、flee persistence、S-curve、左右起始侧和 3--5 个中心混合障碍。
5. 运行前保存 manifest、protocol 和所有输入文件的 SHA-256。

### 5.4 运行顺序

```text
M0 (3 seeds, 40 episodes) -> 安全硬门
    -> M3 (3 seeds, 40 episodes) -> 安全硬门
        -> A1/A2 (3 seeds, 40 episodes) -> 安全硬门
            -> A3 diagnostic（可选）
```

命令模板（实际参数以 `--help` 和冻结 protocol 为准）：

```powershell
& $py scripts/evaluate_jepa_safe_capture_v2_paired.py `
  --variant m0 `
  --training-seed 20260911 `
  --episodes 40 `
  --split validation `
  --protocol configs/central_random_mixed_obstacle_s3_v3_development_protocol.yaml `
  --environment-config configs/capture_radius_pursuit_central_v4_flee.yaml `
  --actor-checkpoint models/v5_development_exact_reactive_seed661606.pt `
  --jepa-checkpoint results/jepa_safe_capture_v3_wp2_seed20260911/checkpoint.pt `
  --reliability-ledger results/jepa_safe_capture_v3_wp3_ledger_seed20260911/reliability_ledger.json `
  --output-dir results/jepa_safe_capture_v3_wp7_m0_seed20260911 `
  --tensorboard-dir results/jepa_safe_capture_v3_tensorboard/wp7_m0_seed20260911 `
  --device cuda `
  --development-only
```

后续变体只替换 `--variant`、`--training-seed`、对应 checkpoint/ledger 和独立输出目录；不能在 final block 中调整 score 权重、CBF margin、credit threshold、chunk length、捕获半径或 episode seed。

### 5.5 Final block 硬门

每个 run 完成后立即检查：

- collision、defender boundary、pairwise violation 均为 0；
- CBF infeasible/timeout/unverified 均有显式 fallback，raw/unverified 执行数为 0；
- zero-perturbation 非 JEPA 字段差异为 0；
- 端到端 p95 latency <= 100 ms；
- summary、episodes.csv、step traces、scene manifest、provenance 和 TensorBoard 齐全；
- 任何安全硬门失败，立即停止该变体，保存 audit，回退冻结 nominal + CBF。

## 6. JEPA 模型增强路线（仅在安全闭环稳定后）

当前优先级不是换更大的 backbone，而是让世界模型对交互和安全风险有可校准的预测。

### 6.1 输入和表示

- [ ] defender-target、defender-defender 的相对位置/速度、TTC、队形几何和通信 mask；
- [ ] target belief 的 observation age、message age、visibility probability 和历史动作；
- [ ] obstacle/boundary 局部几何与 clearance；
- [ ] flee persistence、turn、S-curve、突变加速度和速度突变 motion-mode embedding；
- [ ] 集群实体采用 permutation-invariant 或显式 agent-id 受控编码，避免排序变化造成漂移。

### 6.2 多任务 action-conditioned 预测

保留 target displacement 主头，同时新增：

- target velocity/acceleration consistency；
- obstacle-clearance 和 inter-agent-clearance lower quantile；
- pairwise TTC；
- visibility 和 observation-age risk；
- CBF intervention probability、correction magnitude、QP feasibility；
- ensemble disagreement 或 heteroscedastic residual uncertainty。

每个候选 action chunk 必须单独编码，验证不同候选在相同 belief 下产生可辨识、方向一致的未来表示。预测安全量只能用于 ranking 和 ledger 校准，不能替代 CBF 几何约束。

### 6.3 训练和校准门

- [ ] train/validation/calibration/development 按 episode 隔离；
- [ ] 三个 training seed 独立训练，不共享 optimizer state 或 checkpoint；
- [ ] 报告多 horizon MAE、coverage、Brier/AUC、rank consistency 和 uncertainty calibration；
- [ ] 主要 horizon 优于 constant-velocity 仅作为 prediction gate，不得单独写成控制收益；
- [ ] calibration archive、checkpoint 和 ledger 绑定 hash，在线运行期间 ledger 只读。

## 7. Reliability ledger 完整化

状态机固定为：

| 状态 | 触发 | 执行路径 |
|---|---|---|
| `trusted` | bucket 足够、credit 足够、uncertainty/stale 在阈值内 | 允许 JEPA 排序，再过 CBF |
| `fallback_nominal` | credit 下降、预测漂移、候选分离消失或 bucket 缺失 | nominal + CBF |
| `safe_hold` | OOD、non-finite、过期观测、连续失败或 hash 不一致 | safe-hold + CBF |
| `controlled_abort` | safe-hold 和 nominal-CBF 均不可验证 | 终止，不计 safe capture |

- [ ] 在 calibration-only split 测连续残差突增和 credit decay；
- [ ] 注入通信丢包、急转、速度突变、遮挡、密度 shift 和队形拥挤；
- [ ] 固定最小 bucket、最低 credit、uncertainty 上限、stale age 上限和 OOD 规则；
- [ ] 统计 high-credit 与 low-credit settled failure rate，high-credit 不得更危险；
- [ ] 每次 abstain 都有 reason code、状态转移和可回放 trace；
- [ ] 禁止在线更新 credit/threshold，任何阈值改变必须新建 protocol 和 calibration manifest。

## 8. 候选排序和反事实诊断

### 8.1 排序分解

```text
score(k) = task_progress
         + visibility_gain
         - clearance_risk_lower_quantile
         - cbf_intervention_cost
         - uncertainty_penalty
         - action_change_cost
         - nominal_anchor_penalty
```

- [ ] 保存每个 candidate 的逐项 score、top-two margin、rank stability、switch rate 和 oscillation；
- [ ] 通过离线 settled counterfactual rollout 建立 candidate outcome label；ground truth 只在离线使用；
- [ ] 计算 top-1 precision/recall、Spearman/Kendall rank correlation、分桶校准和 CBF intervention 关联；
- [ ] 统计 rejection reason 分布，确保不可达 candidate 在 JEPA 前被拒绝；
- [ ] score 权重只有在独立 calibration evidence 支持时才能修改，并建立 protocol revision；
- [ ] 禁止根据单个 seed 或 mean capture time 事后挑选权重。

### 8.2 CBF 故障和实时性压力

- [ ] 覆盖 obstacle、boundary、altitude、speed、acceleration、slew 和 pairwise constraints；
- [ ] 注入 QP infeasible、timeout、non-finite request、state violation、通信中断和多约束同时激活；
- [ ] 增加 agent 数量、障碍密度、队形拥挤和边界压力的 latency/feasibility stress；
- [ ] 记录 solver version、status、active set、slack、correction norm、fallback reason 和 latency；
- [ ] 验证任何 ranker/JEPA 输出都不能覆盖 filtered action。

## 9. 统计、报告和结论分类

### 9.1 统计规则

- 以 `(training_seed, episode_index)` 为独立统计单位，不把 timestep/chunk 当独立样本；
- 主报告逐 seed safe-capture、collision、boundary、pairwise、CBF abort、fallback 和 high-credit failure；
- 报告 mean、sample SD、paired delta、improved/degraded/tied、bootstrap 95% CI 和 exact McNemar；
- 按 motion mode、visibility、observation age、clearance、ledger state 和 active constraint 分桶；
- capture time、路径长度、CBF correction 和 latency 只作为安全门通过后的诊断指标。

### 9.2 结论分类

结果只能归入以下类别之一：

- `safe_capture_improvement_candidate`：安全硬门通过，配对 delta 非负且证据支持跨 seed 改善；
- `safety_preserving_noninferior`：安全硬门通过，未观察到可确认提升，但不劣证据成立；
- `prediction_signal_no_control_gain`：prediction/calibration 通过，但闭环任务没有收益；
- `rejected_for_safety`：出现 collision、boundary、pairwise、raw action 或不可审计回退；
- `insufficient_evidence_do_not_open_locked_test`：seed、配对、统计或 provenance 不完整。

本计划不设置必须达到 `95%` 的硬目标。更重要的是 safe-capture 的安全不劣性、失败可解释性和闭环可复现性；任何更高捕获率都必须在这些条件之后解释。

## 10. 时间盒和交付物

| 时间盒 | 工作 | 交付物 |
|---|---|---|
| Day 0 | T0 环境、hash、工作区检查 | preflight log |
| Day 0--1 | T1 CPU tie3 replay | CPU replay results、TensorBoard、provenance |
| Day 1 | T2 device audit + T3 tests | tie3 audit JSON/Markdown、测试结果 |
| Day 1--2 | T4 报告和独立 commit | WP-6 closure report、commit hash |
| Day 2--4 | 失败/候选/ledger/CBF 诊断收口 | WP-B/E/D/F gate memo |
| Day 4--8 | M0/M3/A1/A2 smoke 或合同回归 | 每 seed 独立结果目录 |
| Day 8--14 | 三 seed x >=40 paired final development | 480 个配对 episode 的完整 provenance |
| Day 14--16 | 统计和一致性审计 | final audit、CI、McNemar、TensorBoard 对照 |
| Day 16+ | 多任务 JEPA/uncertainty 增强或 SIL/HIL | 新 protocol、训练报告、故障报告 |

时间盒不是性能承诺。安全硬门或 reproducibility 门失败时立即暂停下一阶段。

## 11. Definition of Done

只有全部条件满足才算完成本轮目标：

1. tie3 CPU/CUDA replay 的安全结算、已知动作漂移和 provenance 已审计；
2. JEPA、ledger、ranker、CBF 和 rolling loop 的接口有测试和逐步 trace；
3. OOD、stale、non-finite、QP infeasible/timeout 均不会执行 raw/unverified action；
4. 三个 training seed、同一 paired scene block、每 seed 至少 40 个 episode 完整运行；
5. 安全保留变体通过 collision、boundary、pairwise、zero-perturbation 和 latency 硬门；
6. safe-capture 按逐 seed、配对统计和 bootstrap CI 报告，capture time 仅作次指标；
7. JSON、CSV、TensorBoard、step traces 和 Markdown 双向一致，并可从空目录重跑最小 episode；
8. 结论按预定义类别归档，不能把单 seed 或 smoke 结果写成正式提升；
9. 未获得单独明确授权前，`locked_test_opened=false` 始终保持不变。

**最终判断标准：** 只有当“JEPA 反事实候选评价 + reliability ledger 拒答 + Joint CBF 硬安全层 + 滚动时域重规划”在多 seed 困难场景中共同通过安全、可靠性、实时性和可复现性证据门，才能称为安全增强的闭环围捕系统；否则应诚实归类为 prediction signal、safety infrastructure 或 development evidence。
