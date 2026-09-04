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
| 三 seed final development block | 已完成，但结果门未通过 | 21 个 run、840 集；安全硬门 PASS，M3 配对 delta=-16.7 pp，reliability gate FAIL | 先执行 P8--P13 修复，再用新 protocol 重跑 smoke |

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

- [x] 确认当前工作目录为 `D:\\uav-capture\\uav_capture`。
- [x] 确认 RTX 5050、CUDA、PyTorch、Conda 环境和 `PYTHONPATH`。
- [x] 确认 protocol 的 `phase=development_only`、`locked_test_opened=false`。
- [x] 检查 git diff，确认 tie-policy 变更没有混入无关 E1/V5 文件。
- [x] 记录 protocol、actor、JEPA、ledger 和 scene manifest 的 SHA-256。

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
- [x] 保持 `--development-only`；不要传入 locked split。

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

- [x] 20/20 episode 的 settled safety outcome、termination reason 和 CBF verification 计数一致；
- [x] collision、defender boundary、target boundary、pairwise violation 均为 0；
- [x] `raw_unverified_executed=0`，所有动作 finite；
- [x] candidate rejection reason 字段在全部 ranking steps 存在；
- [x] tie3 后 settled safety outcomes 和 CBF status 完全一致；3/20 episode 的 candidate/ledger decision drift 已报告，final 主实验固定在 RTX 5050，不混用 CPU/CUDA 任务率；
- [x] CPU/CUDA 端到端 p95 latency 均不超过 100 ms；
- [x] 两侧 provenance、scene hash、protocol hash 和 checkpoint hash 完整一致。

### T3：运行针对性测试

```powershell
& $py -m pytest -q `
  tests/test_jepa_safe_capture_candidates.py `
  tests/test_audit_jepa_safe_capture_candidate_ranking.py `
  tests/test_audit_jepa_safe_capture_fault_injection.py `
  tests/test_audit_jepa_safe_capture_device_replay.py `
  tests/test_replay_jepa_safe_capture_failures.py
```

- [x] 测试通过后再运行完整 `pytest`（targeted 27 passed，full 312 passed）。
- [x] 任一测试失败时，保留失败输出，修复代码/测试并新建 protocol revision；本轮无失败。

### T4：更新报告、提交和可追溯性

- [x] 更新 `docs/JEPA_SAFE_CAPTURE_WP6_DEVICE_REPLAY_20260904.md`，明确 tie3 输入、结果和 decision drift。
- [x] 更新本计划的 WP-6 状态和实际产物路径。
- [x] 检查 JSON、CSV、step traces、TensorBoard 和 Markdown 中的 episode 数、safe-capture、安全计数一致。
- [x] 只提交 tie-policy 相关代码、配置、测试和报告；禁止 `git add .`。
- [x] 创建独立 conventional commit；push 因网络 reset 失败，commit 已保留本地。

WP-6 的安全闭环出口在 T0--T4 通过后闭合。跨设备 decision drift 作为已知限制归档，未闭合前不得启动 final block；tie3 审计已通过该安全出口。

## 5. 三 seed paired development final block

### 5.1 前置准入

运行前必须在 checklist 中确认：

- [x] WP-B2 replay 双次 hash 一致，所有失败有唯一主因或 `unresolved`；
- [x] WP-E candidate 可达性通过，invalid candidate 不进入 JEPA；
- [x] WP-D/F fault injection 中 OOD、stale、non-finite、QP infeasible/timeout 均显式 fallback，raw=0；
- [x] WP-6 tie3 CPU/CUDA 安全结算审计通过；
- [x] protocol、scene manifest、checkpoint、ledger、solver 和 tie tolerance 已冻结；
- [x] `locked_test_opened=false` 且 split 仅为 `validation`；
- [x] 已知跨设备 decision drift 有报告；不存在未解释的安全回归或 provenance 缺失。

若任一项不满足，选择“修复并新建 development protocol”或“归档为不足证据/任务回归”，不能为了凑 episode 数量直接进入 final。

### 5.2 固定实验矩阵

| 变体 | 含义 | 用途 |
|---|---|---|
| M0 | nominal planner + Joint CBF-QP | 安全保留基线和主比较对象 |
| M1 | JEPA + target/uncertainty ranking + CBF，无 auxiliary safety terms | 基础 JEPA 排序消融 |
| M2 | JEPA + ledger + target/uncertainty ranking + CBF | ledger 与 auxiliary score 对照 |
| M3 | JEPA + reliability ledger + auxiliary safety ranking + CBF | 主方法 |
| A1 | M3 去除 ledger + CBF | ledger 消融 |
| A2 | M3 去除 clearance/visibility ranking + CBF | 辅助安全排序消融 |
| A3 | raw/no-CBF | 仅诊断 CBF 必要性，不进入安全结论 |

- [x] training seed 固定为 `20260911`、`20260912`、`20260913`。
- [x] 每个变体每个 seed 至少 40 个 episode；完整矩阵为 7 变体 × 3 seed × 40 集 = 840 集。
- [x] A3 使用同一 paired block、独立目录，并明确标记 `diagnostic_only=true`。
- [x] 变体之间使用相同 episode index、layout、target motion、observation condition 和初始状态。
- [x] 每个 seed/variant 使用独立 results 和 TensorBoard 目录。

### 5.3 场景 manifest 规则

1. 先在全新目录生成一次 40-episode canonical validation manifest。
2. 对同一 paired block 的其他变体只读复用该 manifest；不得在变体间重新采样。
3. 三个 training seed 必须共享相同场景规格和 episode index；manifest hash 变化即停止。
4. manifest 必须覆盖 nominal、delayed/noisy、flee persistence、S-curve、左右起始侧和 3--5 个中心混合障碍。
5. 运行前保存 manifest、protocol 和所有输入文件的 SHA-256。

### 5.4 运行顺序

```text
M0 (3 seeds, 40 episodes) -> 安全硬门
    -> M1/M2/M3 (3 seeds, 40 episodes) -> 安全硬门
        -> A1/A2 (3 seeds, 40 episodes) -> 安全硬门
            -> A3 diagnostic（同一 paired block）
```

命令模板（实际参数以 `--help` 和冻结 protocol 为准；目录名必须匹配聚合器）：

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
  --output-dir results/jepa_safe_capture_v2_p6_paired_full_seed20260911_m0 `
  --tensorboard-dir results/jepa_safe_capture_v3_tensorboard/wp7_full_seed20260911_m0 `
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

---

## 12. WP-7 tie3 完成后的当前状态

WP-7 的 7 变体 x 3 seed x 40 集矩阵已经完成，产物和数字以
`docs/JEPA_SAFE_CAPTURE_WP7_TIE3_FINAL_DEVELOPMENT_20260904.md` 为准。本轮不是
“继续增加 episode 数量”阶段，而是“定位回归并修复合同”的阶段：

| 项目 | 结果 | 决定 |
|---|---:|---|
| M0 nominal + CBF | 60/120 = 50.0% | 当前安全保留基线 |
| M3 JEPA + ledger + auxiliary + CBF | 40/120 = 33.3% | 相对 M0 配对 delta = -16.7 pp，停止宣称提升 |
| M3 非负 seed | 0/3 | 不打开 locked test，不调参后重算旧结果 |
| M3 improved/degraded/tied | 10/30/80 | 优先分析排序回归和候选切换 |
| 安全保留变体 collision/boundary/pairwise | 全为 0 | CBF 执行边界保持有效 |
| A3 raw/no-CBF collision | 120/120 | CBF 必要性得到诊断支持 |
| reliability observability gate | FAIL | 先修复 1 个 M0 CBF timeout |

### 12.1 当前唯一允许的下一步

- [x] 完成 21 个 run、每 run 40 集，并保存 JSON/CSV/step trace/manifest/provenance/TensorBoard。
- [x] 运行 paired aggregate，确认 canonical scene manifest 唯一且 120 对 episode 可配对。
- [x] 运行 WP-8 失败索引：567/840 集失败，421 次 CBF controlled abort，26 次 timeout，
  197 次 candidate capture regression，92 次 candidate oscillation。
- [ ] 完成 P8--P11 的失败因果、solver timeout、排序失配和 ledger 校准修复。
- [ ] 用全新 protocol revision 和全新 output root 进行 smoke，再决定是否重跑三 seed final。

当前必须保留 `locked_test_opened=false`。任何新 checkpoint、ledger、score 权重、CBF
阈值、chunk 长度或候选合同改变，都必须生成新的 protocol、manifest 和 provenance，
不得覆盖 `wp7_tie3`。

## 13. 下一阶段分工作包 TODO

### P8：M3 回归的因果定位（立即执行，1--2 天）

- [ ] 从 `results/wp8_failure_index_tie3/failure_index.csv` 取出 30 个 M3 相对 M0
  degraded episode、10 个 improved episode 和 80 个 tied episode。
- [ ] 每类至少选 6 个代表片段，按同一 episode seed 重放
  `belief -> candidate generation -> JEPA -> ledger -> rank -> CBF -> action -> termination`。
- [ ] 记录每个候选的 task progress、visibility、clearance、uncertainty、CBF
  intervention cost、top-two margin、ledger state 和 fallback reason。
- [ ] 为每个 degraded episode 标注唯一主因：错误高信用排序、预测净空过度乐观、
  可见性估计失配、candidate oscillation、CBF infeasible/abort、timeout 或目标 belief 漂移。
- [ ] 若无法形成唯一因果链，标为 `unresolved`，禁止凭均值修改权重。
- [ ] 输出 `results/wp8_failure_replay_tie3/`，包括 replay JSONL、trace hash、因果表和
  deterministic second replay hash。

**出口：** 30 个 degraded episode 全部有 `primary_cause` 或 `unresolved`；两次 replay
hash 完全一致；原始 WP-7 目录不被修改。

### P9：CBF/solver reliability gate 修复（与 P8 并行，1--2 天）

- [ ] 定位 M0 seed 20260911 的 1 个 CBF timeout：区分 solver 真实超时、日志计时错误、
  Windows 调度抖动和输入规模异常。
- [ ] 为每次 QP 调用记录 monotonic start/end、solver status、iteration、active set、
  slack、correction norm、fallback mode 和 reason code。
- [ ] timeout 必须转入有验证的 safe-hold 或 nominal-CBF fallback；不得继续使用旧的
  desired action，也不得将 timeout 记作 safe capture。
- [ ] 增加多约束同时激活、通信中断、non-finite、急转、拥挤队形和边界压力测试。
- [ ] 测试 CPU/CUDA 只要求 settled safety outcome 等价，不要求逐步动作 bitwise 等价；
  final 主实验固定 RTX 5050。
- [ ] 若需改变 solver timeout、margin、gamma 或 fallback 顺序，建立新 protocol revision，
  并重新计算所有输入哈希。

**出口：** timeout=0 或每个 timeout 都有明确、可验证、可回放的安全 fallback；
`raw_unverified_executed=0`；端到端 CBF p95 <= 100 ms；reliability observability gate PASS。

### P10：JEPA 多任务和不确定性增强（P8 之后，3--6 天）

- [ ] 保留 target displacement 头，同时加入 target velocity/acceleration consistency、
  obstacle-clearance lower quantile、inter-agent clearance、pairwise TTC、visibility、
  observation-age risk、CBF intervention probability 和 QP feasibility 头。
- [ ] 编码 defender-target/defender-defender 相对状态、TTC、队形几何、通信 mask、障碍/边界
  局部几何、motion-mode embedding；实体编码保持 permutation-invariant 或显式 agent-id 约束。
- [ ] 对每个候选 action chunk 单独编码，检查相同 belief 下不同候选的未来表示是否可分辨、
  方向一致；禁止只用无动作条件的 target prediction。
- [ ] 训练三组全新 seed；train/validation/calibration/development 按 episode 隔离，禁止
  直接把 WP-7 失败片段回灌旧 archive。
- [ ] 报告多 horizon MAE、coverage、Brier/AUC、rank consistency、uncertainty calibration，
  并与 constant-velocity 和旧 JEPA 对照。
- [ ] 只有 prediction/calibration gate 通过后，才把新 checkpoint 接入闭环；prediction gate
  通过不等于控制收益。

**出口：** safety-related heads 在 calibration split 有可复现的校准曲线，主要 horizon
的 rank consistency 不劣于旧模型；checkpoint、训练配置和 archive hash 完整。

### P11：candidate block 和排序修复（P8/P10 之后，2--4 天）

- [ ] 继续使用 5 个候选和 3-step chunk 作为合同基线，先修复评分与 settled outcome 的
  失配，再考虑增加候选数量。
- [ ] 采用安全优先的分层排序：先排除不可达/明显不安全候选，再比较 task progress；
  score 至少包含 clearance lower quantile、visibility gain、CBF intervention cost、
  uncertainty penalty、action-change cost 和 nominal anchor penalty。
- [ ] 增加 top-two margin、rank stability、candidate switch rate、oscillation length、
  CBF correction 和 fallback probability 的日志。
- [ ] 用离线 settled counterfactual label 计算 top-1 precision/recall、Spearman/Kendall、
  分桶 calibration 和 CBF intervention 相关性；ground truth 仅在离线使用。
- [ ] 对高信用但失败的候选设置 conservative abstention：当预测净空过度乐观、visibility
  gap 突增或 rank margin 太小时，退回 nominal/safe-hold，而不是强行相信 JEPA。
- [ ] 设定滞回和最小保持时间，抑制 M3 当前约 0.180 的 candidate switch rate；该参数改变
  必须进入新 protocol。

**出口：** degraded replay 中高信用错误排序显著减少；candidate regression 和 oscillation
有明确下降；所有候选仍经过同一 Joint CBF-QP。

### P12：reliability ledger 重新校准（P10/P11 之后，2--3 天）

- [ ] 用独立 calibration split 测连续残差突增、目标急转、速度突变、遮挡、通信丢包、
  density shift 和队形拥挤。
- [ ] 固定并绑定 hash：minimum bucket、minimum credit、uncertainty 上限、stale age 上限、
  OOD 规则、credit decay/recovery、abstain hysteresis。
- [ ] 统计 trusted、fallback_nominal、safe_hold、controlled_abort 四种状态的 episode 率、
  safe-capture、CBF abort 和 high/low-credit failure rate。
- [ ] 验证 high-credit failure rate 不高于 low-credit；若不满足，ledger 只能归类为
  `prediction_signal_no_control_gain`，不能作为可信度提升证据。
- [ ] 每次 abstain 记录 reason code、状态转移、credit、uncertainty、观测年龄和可回放 trace；
  在线不更新 threshold 或 checkpoint。

**出口：** ledger observability gate PASS，异常输入不会执行 raw/unverified action，且可由
单个 trace 解释每次 fallback/abstain。

### P13：滚动时域闭环集成回归（1--2 天）

- [ ] 验证每个控制周期只执行 action chunk 第一步，随后重新获取 observation、更新 belief、
  重新预测、排序和 CBF 过滤。
- [ ] 验证候选、nominal、safe-hold 和 fallback 都走同一 Joint CBF-QP；CBF 失败时按
  `separation-preserving safe-hold -> verified nominal-CBF -> controlled abort` 执行。
- [ ] 做 zero-perturbation regression：关闭 JEPA 后非 JEPA 字段应完全不变；打开 JEPA 只能
  改变 candidate score/选择，不能改写 filtered action 或安全约束。
- [ ] 保存 step-level ledger state、candidate scores、selected candidate、filtered action、
  active constraints、slack、latency、termination reason 和 trace hash。

**出口：** 关键接口测试全部通过；100 个随机控制周期中无 raw/unverified action；闭环 trace
可从空目录重放。

### P14：重新验证和统计（P8--P13 全部通过后，3--5 天）

- [ ] 新建 protocol、scene manifest、checkpoint 和 ledger 的独立目录，不覆盖 WP-7。
- [ ] 先跑每 seed 20 集 smoke：M0、M3、A1、A2；A3 只作为诊断，保持 `diagnostic_only=true`。
- [ ] smoke 必须通过安全硬门、reliability gate、provenance 完整性和 p95 latency 门，才扩展到
  3 seed x 40 集 paired development。
- [ ] 使用同一 episode index/scene manifest 配对；按 seed 计算 safe-capture、paired delta、
  improved/degraded/tied、bootstrap 95% CI 和 exact McNemar。
- [ ] 通过标准：安全保留变体 collision/boundary/pairwise=0、raw=0、timeout=0 或有验证 fallback、
  p95<=100 ms；若要写“JEPA 有任务收益”，还需 M3 平均 paired delta >= 0 且至少 2/3 seed 非负。
- [ ] 若安全通过但 delta 仍为负，归档为 `prediction_signal_no_control_gain` 或
  `useful_safety_fallback_only`，不再扩大 episode 数量。

### P15：locked test readiness（仅在用户明确授权后）

- [ ] 只在 P14 通过、结果分类不是 `insufficient_evidence_or_reject`、所有 provenance 双向一致
  后，生成 locked-test readiness memo。
- [ ] locked split 必须是新鲜、未见、只读的场景；不能用 development 失败片段调参后直接打开。
- [ ] 预先冻结主要比较、统计方法、安全门和停止规则；执行期间禁止查看并据此调参。
- [ ] 未收到明确授权前始终保持 `locked_test_opened=false`。

## 14. 推荐时间盒和交付清单

| 时间盒 | 任务 | 必交付物 |
|---|---|---|
| Day 0--1 | P8 失败重放 | failure replay JSONL、因果表、双次 trace hash |
| Day 0--2 | P9 CBF timeout/reliability | solver audit、fault matrix、更新测试 |
| Day 2--6 | P10 JEPA 多任务训练 | 新 archive、三 seed checkpoint、prediction/calibration report |
| Day 5--8 | P11 candidate/ranker | settled rank audit、权重合同、oscillation report |
| Day 7--10 | P12 ledger 校准 | ledger v4、OOD/stale/credit report |
| Day 9--11 | P13 闭环回归 | interface/zero-perturbation/replay audit |
| Day 12--16 | P14 smoke + paired development | 新矩阵、aggregate、bootstrap/McNemar、最终分类 |
| Day 17+ | P15 readiness | 仅在获得授权后生成 locked-test memo |

## 15. 下一轮 Definition of Done

下一轮只有同时满足以下条件才算完成：

1. M3 相对 M0 的所有 degraded episode 都有可解释的 replay 结论或明确 `unresolved`，且
   replay hash 可重复。
2. reliability observability gate 通过；timeout、OOD、stale、non-finite、QP infeasible/timeout
   均有显式安全回退，raw/unverified execution 永远为 0。
3. 新 JEPA 的 interaction-aware/action-conditioned safety heads 在独立 calibration split
   通过 prediction/calibration gate，并绑定 checkpoint/ledger/archive hash。
4. candidate ranking 在 settled counterfactual label 上可校准，switch/oscillation 不再造成
   可观测 safe-capture 回归。
5. 滚动时域每周期只执行第一步，nominal/candidate/fallback 全部经过 Joint CBF-QP，step trace
   可重放且 p95 latency <= 100 ms。
6. 三 seed paired development 完整运行；安全保留变体零碰撞、零越界、零 pairwise，safe-capture
   以配对统计报告；不把 95% 或 mean capture time 作为必要条件。
7. 若未达到非负 paired delta，诚实归类为安全基础设施或 prediction signal，不打开 locked test。
