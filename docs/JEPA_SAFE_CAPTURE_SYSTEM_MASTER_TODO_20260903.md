# 无人机集群安全增强围捕系统
# Interaction-Aware Action-Conditioned JEPA + Reliability Ledger + CBF
# 下一步详细 TodoList 与验收计划

**版本：** v1.0
**日期：** 2026-09-03
**适用环境：** Windows、Conda `uav-encirclement-gpu`、NVIDIA RTX 5050
**实验性质：** development-first；未经单独授权不得打开新的 locked test
**首要指标：** `safe_capture`，而不是 `mean capture time`

---

## 0. 目标和结论边界

### 0.1 系统目标

在无人机集群对抗围捕任务中，构建一条可解释、可回退、可审计的安全闭环：

```text
观测/通信历史
    -> interaction-aware belief state
    -> 传统规划器生成候选 action chunks
    -> action-conditioned JEPA 预测目标与安全相关未来
    -> reliability ledger 可信度门控
    -> 候选轨迹排序
    -> 联合多机 CBF-QP 安全过滤
    -> 只执行第一控制步
    -> 重新观测、重新预测、重新规划
```

系统必须满足以下边界：

- JEPA 是**轨迹评价器**，不直接生成最终控制动作。
- Reliability ledger 是**可信度门控**，不是安全证明。
- CBF-QP 是最终执行安全层，不能被 JEPA 分数、uncertainty 或预测的“安全”覆盖。
- 每次只执行候选 action chunk 的第一步，下一周期必须重新观测和 replan。
- `safe_capture` 是首要任务指标；`mean capture time` 只报告，不单独否决方案。
- 不要求绝对捕获率固定达到 `95%`；首先要求安全不退化、结果可复现、失败可解释。

### 0.2 当前证据能说什么

| 阶段 | 当前状态 | 可以声称 | 不能声称 |
|---|---|---|---|
| P2 | 三 seed 离线预测完成 | JEPA 多 horizon 预测优于 constant velocity | 闭环捕获提升 |
| P3 | 三 checkpoint ledger 完成 | 可执行可信度门控和回退 | QP 可行性已校准、安全已证明 |
| P4 | 候选接口和 synthetic audit 完成 | 候选会因 action 改变，低信用可回退 | 碰撞为零或 safe-capture 提升 |
| P5 | 联合 CBF-QP 开发审计通过 | 13 tests、三 seed audit、zero-perturbation、显式 fallback、TensorBoard | 尚无随机闭环 safe-capture 结论；SLSQP 仍需部署评估 |
| P6 | 待开始 | P5 安全过滤硬门已通过，可冻结 paired development | 三 seed paired development 结论 |

在 P5 安全硬门通过前，不得运行 P6 的闭环主实验；在 P6 和 P7 完成前，不得申请新的 locked test。

---

## 1. 不可变安全合同

### 1.1 Safe-capture 定义

一个 episode 只有在以下条件全部满足时才算 `Cooperative Safe Capture`：

1. 至少一个 defender 进入目标 `0.80 m` capture radius；
2. 在 episode time limit 内完成；
3. 终止前没有 obstacle collision、defender-target collision 或 defender-defender collision；
4. 没有 defender 离开 3-D world boundary；
5. 任何 CBF-QP infeasible、controlled abort 或安全硬门失败均不能被记为 safe capture。

### 1.2 信息边界

- 在线模块只能使用 observation、communication history、action history、obstacle geometry 和 observation/message age。
- 在线禁止读取 target ground truth；ground truth 仅用于 offline settled-label 结算。
- train、validation、calibration、development 和 locked episode/seed 不得交叉。
- 训练、评估、聚合脚本必须写入并检查 `locked_test_opened=false`。
- 不允许用 development episode 重新调 ledger threshold、CBF margin、候选权重或 episode seed。

### 1.3 动作和滚动时域

- 候选数固定 `K=5`，顺序固定为 `nominal`、`intercept`、`lateral_clearance`、`formation_clearance`、`visibility_hold`。
- 第一版 action chunk 固定 3 个 control steps；只执行第一步。
- 下一周期必须重新观测、更新 belief、重新预测、重新排序和重新过滤。
- 所有候选先通过 finite、speed、acceleration、action slew 检查。
- baseline 与所有 candidate 必须使用相同的 CBF-QP solver、margin、tolerance 和 timeout。

### 1.4 安全硬门优先级

硬门按以下顺序处理：

1. non-finite、invalid shape、过期观测和协议/哈希不一致；
2. obstacle collision、boundary violation、pairwise separation violation；
3. CBF-QP infeasible、solver timeout、fallback 未被执行；
4. safe-capture non-inferiority；
5. transit、fallback 率、可见性和路径代价；
6. mean capture time、路径长度和显存/算力成本。

任何安全硬门失败时，当前变体停止并回退到冻结 nominal + CBF，不通过“提高捕获率”来抵消。

---

## 2. 目标系统接口

### 2.1 输入接口

每个控制周期保存：

- `defender_positions`, `defender_velocities`；
- target belief position/velocity 及其 observation/message age；
- obstacle geometry 和 world bounds；
- 最近 `H` 个 observation/action/communication tokens；
- 上一周期 CBF action、solver status、minimum slack、fallback mode；
- layout signature、motion mode 和 episode provenance。

### 2.2 JEPA 预测头

主 latent prediction 保留 action-conditioned target state 预测，并增加与安全直接相关的辅助头：

- target relative displacement / velocity；
- obstacle clearance；
- inter-agent clearance；
- pairwise TTC；
- target visibility / observation quality；
- CBF intervention risk；
- predictive uncertainty 和 candidate disagreement。

辅助头只用于候选排序、ledger 校准和诊断，不用于替代真实 CBF 约束。

### 2.3 Reliability ledger 状态机

| 状态 | 进入条件 | 允许行为 |
|---|---|---|
| `trusted` | context、horizon、uncertainty、历史兑现率均在可信区间 | 允许 JEPA 候选排序 |
| `fallback_nominal` | credit 低、bucket 缺失、预测分歧过大或风险异常 | 选择 nominal，仍经过 CBF-QP |
| `safe_hold` | OOD、stale observation、模型/ledger hash 不一致或连续失败 | hold/nominal safe action，仍经过 CBF-QP |

状态切换必须写入逐步 trace，禁止静默回退。

### 2.4 CBF-QP 输出接口

每次调用必须返回：

- filtered action；
- solver name/version/status/message；
- `verified_feasible`、`infeasible`、`timed_out`；
- objective、action correction norm、minimum constraint value；
- 每条 constraint 的 slack 和 active/inactive 状态；
- fallback mode 和 fallback reason；
- solve latency；
- task diagnostic（target approach progress），但不得削弱安全约束。

---

## 3. 阶段总路线和依赖

```text
P0 合同冻结 [完成]
  -> P1 反事实 archive [完成]
  -> P2 三 seed JEPA [完成]
  -> P3 ledger v2 [完成]
  -> P4 候选 action-chunk ranker [完成接口]
  -> P5 联合 CBF-QP [当前]
  -> P6 三 seed paired development
  -> P7 审计、统计和 preregistration 决策
  -> P8 可选 SIL/HIL / 部署准备
```

当前执行顺序不可跳过：

1. P5 代码、测试、audit 和报告已完成；
2. P5 安全硬门已通过，下一步冻结 P6 运行矩阵；
3. 先做 P6 smoke，再做三 seed 全量 paired development；
4. P6 结束后只做一次 P7 审计和是否新建 locked block 的决策；
5. 没有用户明确授权，不运行新的 locked test，也不修改历史 V4/V5 locked 结果。

---

## 4. P5：联合多机 CBF-QP 与安全回退（开发审计已完成）

### 4.1 实现 Todo

- [x] 检查 `src/encirclement3d/cbf_qp.py` 是否能正确读取当前环境的 `agents/task/pursuit/obstacles` 配置键。
- [x] 运行 `py_compile` 和最小 import，修复 solver 异常路径中未定义变量、非 finite candidate 和空 constraint 的问题。
- [x] 为所有 defender 联合构造一个 decision vector，不允许各机独立过滤后再拼接。
- [x] 实现 obstacle clearance barrier；几何计算统一复用环境的 cylinder/box clearance helper。
- [x] 实现每一对 defender 的 pairwise separation barrier，并验证两机同时靠近时的符号和单位。
- [x] 实现 x/y/z world-boundary 约束；z 方向单独记录 altitude lower/upper。
- [x] 实现 speed、acceleration 和 action slew 的运动学约束。
- [x] 固定 solver、`ftol`、最大迭代、tolerance、timeout 和确定性配置；全部写入 audit。
- [x] 明确区分 `success`、`feasible_nonconverged`、`infeasible`、`timeout`、`solver_exception`、`nonfinite_request`。
- [x] 完成 fallback ladder：`primary solve -> safe_hold solve -> nominal through CBF -> controlled_abort`。
- [x] 失败时禁止返回或执行未过滤的原始 desired action。
- [x] controlled abort 返回可记录的安全状态，并明确不声称 safety proof。
- [x] 为候选和 nominal 实现 zero-perturbation identity 检查。
- [x] 记录 SciPy SLSQP 局限；OSQP/CVXPy 作为后续部署工程门，不改变当前安全合同。

### 4.2 P5 单元测试 Todo

- [x] 正常可行的联合约束；重复调用输出确定。
- [x] 单个 obstacle constraint 激活。
- [x] 两机 pairwise separation 同时激活。
- [x] boundary 与 altitude lower/upper 激活。
- [x] speed、acceleration 和 action slew 限制。
- [x] 多类约束同时激活时的最小 slack 和 active list。
- [x] 明确构造 infeasible 场景，确认不会执行原始动作。
- [x] non-finite request、wrong shape 和 solver exception。
- [x] primary 失败时 safe-hold / nominal-CBF / controlled-abort 的顺序和诊断。
- [x] latency 统计、timeout 路由和 TensorBoard provenance。
- [x] 与旧 `PursuitCBFSafetyFilter`、`DiscreteTimeCBFSafetyFilter` 的接口不冲突；未修改用户已有 E1 execution-aware 模块。

### 4.3 P5 通过标准

- [ ] 所有测试通过，至少包含正常、激活、不可行、非 finite 和 deterministic repeated solve。
- [ ] 任一 infeasible/timeout 都不会静默执行未过滤动作。
- [ ] 所有输出 finite，constraint residual 在 tolerance 内可复核。
- [ ] zero-perturbation 时 baseline 与 candidate 的 filtered action 逐字段一致。
- [ ] p95 `CBF-QP` 延迟不超过 100 ms；超时必须可观测并路由到 fallback。
- [ ] audit JSON、Markdown 报告和 TensorBoard 均记录 `locked_test_opened=false`。

### 4.4 P5 产物

```text
src/encirclement3d/cbf_qp.py
tests/test_cbf_qp.py
scripts/audit_jepa_safe_capture_v2_cbf_qp.py
docs/JEPA_SAFE_CAPTURE_P5_CBF_QP_AUDIT_20260903.md
results/jepa_safe_capture_v2_p5_cbf_qp_audit_*/
results/jepa_safe_capture_v2_tensorboard/p5_*/
```

P5 完成后单独提交，例如：

```text
feat(jepa): add verified joint cbf qp and fallback audit
```

---

## 5. P6：三 seed paired safe-capture development

### 5.1 运行矩阵

| ID | 执行栈 | 目的 |
|---|---|---|
| M0 | frozen V5 nominal + 同一 CBF-QP | 主基线 |
| M1 | JEPA + nominal fallback + CBF-QP | 分离模型作用 |
| M2 | JEPA + ledger v2 + CBF-QP | 可信度门控 |
| M3 | JEPA + ledger + candidate ranker + CBF-QP | 最终候选系统 |
| A1 | M3 去掉 ledger | 漂移/幻觉消融 |
| A2 | M3 去掉 clearance/visibility heads | 辅助任务消融 |
| A3 | raw action/no CBF | 仅故障诊断，不进入安全主结论 |

M0--M3 和 A1--A3 必须使用同一 episode seed、layout、初始状态、target motion、observation schedule 和 transit reference。

### 5.2 运行顺序

- [x] 创建全新 results namespace，不覆盖任何已有 checkpoint、TensorBoard 或报告。
- [ ] 写入 protocol、环境配置、代码 revision、checkpoint hash、ledger hash 和 Conda/PyTorch/CUDA 信息。
- [ ] 每个变体和 seed 先运行 20 paired smoke episodes。
- [ ] smoke 出现 collision、boundary、pairwise violation、non-finite、unpaired 或 zero-regression 失败，立即停止该变体。
- [ ] smoke 通过后冻结所有模型、权重、chunk、阈值、solver 参数和 seed，不再调参。
- [ ] 对三 seed 运行每个最终变体 60 episodes；主 M3 总量为 `3 x 60 = 180` paired episodes。
- [ ] 保存逐 episode summary、逐 step trace、scene manifest、候选 trace、ledger state 和 CBF diagnostics。
- [ ] 每次运行 TensorBoard scalar、histogram、text provenance 和命令行配置。

### 5.3 场景覆盖

- [ ] nominal flee；
- [ ] delayed/noisy observation 和 message age；
- [ ] S-curve、速度突变、频繁随机转向；
- [ ] 3--5 个 cylinder/box/wall 混合障碍；
- [ ] narrow-channel 低净空；
- [ ] 高拥挤队形、低 pairwise TTC；
- [ ] 左右起始侧、不同 visibility 和 communication delay。

### 5.4 主指标

每个变体、每个 seed、每个场景 bucket 都必须报告：

- safe capture count/rate；
- collision count/rate；
- boundary violation count/rate；
- pairwise minimum separation 和违规数；
- CBF-QP infeasible、timeout、controlled-abort 数；
- transit success；
- safe-hold、fallback_nominal、candidate selection rate；
- non-finite、stale、OOD 路由计数。

`mean capture time`、path length、CBF correction、candidate switch rate、clearance、inference latency 和显存占用作为次指标完整报告。

### 5.5 P6 决策门

**G1 安全硬门**

- candidate collision = 0；
- candidate boundary violation = 0；
- 无新的 pairwise separation 违规；
- infeasible/timeout 全部进入显式 fallback；
- zero-perturbation 与 M0 逐 episode 一致。

任一失败即拒绝该变体，不得用捕获率抵消。

**G2 Safe-capture non-inferiority**

- 三 seed 平均 paired safe-capture delta `>= 0` 个百分点；
- 至少 2/3 seed 的 paired delta 非负；
- 不要求绝对 safe-capture 达到 95%。

**G3 正向候选**

- 平均 paired delta 严格为正；
- 至少 2/3 seed 非负；
- 使用预先固定的 bootstrap CI 或 McNemar/exact paired test；
- 否则只写 safety-preserving 或 non-inferior，不写提升。

**G4 Reliability**

- high-credit 失败率不高于 low-credit；
- OOD、stale、high-uncertainty 必须触发 fallback；
- ledger 不得降低 CBF 安全约束。

**G5 Realtime**

- p95 总控制延迟不超过 100 ms；
- timeout 有可观测的 nominal-CBF 或 safe-hold 路由。

**G6 Provenance**

- 每个 episode 均可追溯到 seed、scene、checkpoint、ledger、代码 revision 和环境 hash；
- 所有报告、JSON、TensorBoard 都标记 development-only 和 `locked_test_opened=false`。

---

## 6. P7：统计、复盘与是否申请 locked test

### 6.1 复盘任务

- [ ] 聚合三 seed safe capture、失败率、paired delta 和置信区间。
- [ ] 按 target motion、obstacle layout、clearance、visibility、observation age、ledger state 和 CBF active constraint 分桶。
- [ ] 对每个失败恢复完整因果链：候选 -> JEPA 预测 -> ledger decision -> rank -> CBF solver -> executed action -> failure。
- [ ] 复查 high-credit failure 是否集中于某个 motion/layout/horizon bucket。
- [ ] 对候选切换过多、CBF correction 过大、safe-hold 连续触发和 visibility 丢失做专门诊断。
- [ ] 用 TensorBoard 和静态 JSON 双重核对结果，防止只看汇总均值。

### 6.2 决策输出

只能输出以下三类之一：

1. `positive_development_evidence`：安全硬门通过且 safe-capture 有可重复正向证据；
2. `safety_preserving_non_inferiority`：安全硬门通过，捕获率不劣但没有稳定提升；
3. `insufficient_evidence_or_reject`：安全、可靠性、延迟或可复现性未通过。

只有第 1 类或稳定的第 2 类，并且所有审计资料齐全，才可以起草新的 preregistration。是否真正打开 locked test 必须另行取得授权。

### 6.3 P7 产物

```text
docs/JEPA_SAFE_CAPTURE_P6_THREE_SEED_PAIRED_DEVELOPMENT_*.md
docs/JEPA_SAFE_CAPTURE_P7_READINESS_AUDIT_*.md
results/jepa_safe_capture_v2_p6_*/summary.json
results/jepa_safe_capture_v2_p7_readiness_*/
```

---

## 7. P8：可选 SIL/HIL 和部署准备

P8 不是主仿真结论的前置条件，也不自动转化为真实飞行结论。只有 P5--P7 通过后才考虑：

- [ ] CPU-only 和 RTX 5050 两条推理路径的一致性审计；
- [ ] solver watchdog、动作限幅、safe hover 和通信中断演练；
- [ ] SITL/PyBullet 接口回放，不改变主 benchmark 的统计合同；
- [ ] 传感器延迟、丢包、时间戳错误和 target belief stale 演练；
- [ ] 部署 artifact、版本锁定、回滚包和审计日志完整性检查；
- [ ] 明确仿真安全证据不能替代飞控、硬件和实飞安全认证。

---

## 8. 实验记录和可复现性清单

每次运行前：

- [ ] 输出目录不存在或为空；
- [ ] checkpoint、ledger、dataset、config、代码 revision 和协议 hash 已解析；
- [ ] 当前 GPU、PyTorch、CUDA、Conda 环境已记录；
- [ ] `locked_test_opened=false`；
- [ ] 没有把 tmp archive 当作历史 warm-start checkpoint；
- [ ] 未修改用户已有 E1/execution-aware 文件。

每次运行后：

- [ ] `summary.json`、`episodes.csv`、`scenes.jsonl`、逐步 trace 已生成；
- [ ] TensorBoard scalar/text/histogram 有对应 provenance；
- [ ] 输出文件 SHA-256 已写入报告；
- [ ] 失败 episode 能被 deterministic replay；
- [ ] 报告明确区分 prediction-only、safety-preserving、non-inferiority 和 positive claim；
- [ ] 阶段代码、测试、报告和 audit 独立 commit。

推荐的最小环境核验命令：

```powershell
Set-Location D:\\uav-capture\\uav_capture
conda activate uav-encirclement-gpu
$env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
python -m pytest -q
```

长任务必须使用新的、空的 results 目录，并在 TensorBoard 中保留命令、配置、输入 hash 和环境信息。

---

## 9. 停止、回退和否决规则

- 新 collision、boundary 或 pairwise violation：停止当前变体，保存 trace，回退 nominal + CBF。
- CBF-QP infeasible 没有 safe-hold：P5 不通过，不能进入 P6。
- solver timeout 没有显式 fallback：P5 不通过。
- zero-perturbation 不一致：停止 ranker 调试，先修复执行链路。
- high-credit 比 low-credit 更危险：冻结并拒绝当前 ledger。
- checkpoint、ledger、dataset 或 protocol hash 不一致：停止并重建输入。
- p95 延迟超预算：保留 watchdog 证据，不能隐藏 timeout 或直接放宽安全约束。
- capture time 变差但 safe-capture 和所有硬安全门通过：保留并报告，不单独否决。
- 任何通过放宽 CBF margin、绕过 fallback、泄漏 ground truth 或回灌 validation/development 数据换取捕获率的修改，均视为实验失败。

---

## 10. 完成定义（Definition of Done）

本系统只有在以下条件全部满足时才算完成本阶段目标：

1. P5 联合 CBF-QP、P6 三 seed paired development、P7 readiness audit 均有独立报告；
2. 所有代码、配置、测试、命令、checkpoint、ledger、scene 和数据 hash 可追溯；
3. JEPA 永远只评价候选，最终动作永远经过 CBF-QP；
4. 所有 infeasible、timeout、OOD、stale 和 non-finite 情况都有显式 fallback；
5. collision、boundary、pairwise separation 和 CBF-QP infeasible 按 episode 审计；
6. TensorBoard 可从空目录重放，且日志含环境和 provenance；
7. safe-capture 结果至少 safety-preserving/non-inferior，或明确给出 reject/insufficient evidence；
8. mean capture time 只作为次指标，不能掩盖安全失败；
9. 没有用户明确授权前，新的 locked test 保持关闭；
10. 每阶段均有独立 Git commit，且不覆盖用户已有未提交 E1/V5 修改。

**最终研究判断：** 不是追求某一个 seed 的最高捕获率，而是证明 JEPA 的反事实评价、reliability ledger 的拒答机制、CBF-QP 的硬安全约束和滚动时域 replan 能在同一冻结协议下组成一条可重复、可解释、不会静默失效的安全闭环。

---

## 11. 冻结输入、立即执行命令和时间盒

### 11.1 冻结输入

P5/P6 默认只使用以下已审计输入；若任一文件 hash 改变，必须重新生成 provenance，不能直接继续：

| 输入 | 路径 |
|---|---|
| P2 seed 20260911 checkpoint | `results/jepa_safe_capture_v2_p2_seed20260911/checkpoint.pt` |
| P2 seed 20260912 checkpoint | `results/jepa_safe_capture_v2_p2_seed20260912/checkpoint.pt` |
| P2 seed 20260913 checkpoint | `results/jepa_safe_capture_v2_p2_seed20260913/checkpoint.pt` |
| P3 seed 20260911 ledger | `results/jepa_safe_capture_v2_p3_rerun_ledger_seed20260911/reliability_ledger.json` |
| P3 seed 20260912 ledger | `results/jepa_safe_capture_v2_p3_rerun_ledger_seed20260912/reliability_ledger.json` |
| P3 seed 20260913 ledger | `results/jepa_safe_capture_v2_p3_rerun_ledger_seed20260913/reliability_ledger.json` |
| 运行协议 | `configs/jepa_safe_capture_v2_training.yaml` 及对应评估配置 |
| 候选接口 | `src/encirclement3d/jepa_safe_capture_candidates.py`、`jepa_safe_capture_ranker.py` |

历史 `tmp` archive 只作为数据恢复和审计输入；没有找到的历史 retained-BC warm-start checkpoint 不得伪造或用后生成的 recovery checkpoint 代替。

### 11.2 当前机器的最小执行顺序

```powershell
Set-Location D:\\uav-capture\\uav_capture
conda activate uav-encirclement-gpu
$env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"

# 1) 先验证 P5 代码，不执行闭环实验
python -m py_compile src/encirclement3d/cbf_qp.py
python -c "from encirclement3d.cbf_qp import JointCBFQPSafetyFilter; print('cbf_qp import ok')"
python -m pytest -q tests/test_cbf_qp.py

# 2) P5 通过后才运行安全过滤 audit
python scripts/audit_jepa_safe_capture_v2_cbf_qp.py `
  --output-dir results/jepa_safe_capture_v2_p5_cbf_qp_audit_seed20260911 `
  --tensorboard-dir results/jepa_safe_capture_v2_tensorboard/p5_cbf_qp_seed20260911

# 3) P5 audit 通过后，先做 P6 paired smoke；禁止直接打开 locked test
python scripts/evaluate_jepa_safe_capture_v2.py --help
```

实际评估命令必须以脚本的 `--help` 和冻结 protocol 为准；若脚本接口尚未存在，先完成实现和测试，不得用临时命令绕过审计字段。

### 11.3 推荐时间盒

| 时间盒 | 工作 | 退出条件 |
|---|---|---|
| T0（半天） | P5 import、配置键、旧 filter 兼容性检查 | py_compile/import 通过 |
| T1（1--2 天） | P5 约束、solver status、fallback、测试 | P5 单测和 zero-perturbation 通过 |
| T2（半天） | P5 audit、延迟、TensorBoard、报告和独立 commit | P5 G1 全部通过 |
| T3（半天） | P6 protocol/seed/scene manifest 冻结 | 无 hash、split、paired 漂移 |
| T4（1 天） | 每变体 20 paired smoke | 无安全硬门失败 |
| T5（2--4 天） | M0--M3 三 seed 全量 60 episode | 180 个 M3 episode 可复盘 |
| T6（1--2 天） | A1--A3 消融和失败 replay | 消融不改安全合同 |
| T7（1--2 天） | P7 统计、审计、报告和决策 | 只输出三类决策之一 |

时间盒是工程安排，不是对 safe-capture 数值的承诺；任何安全硬门失败都优先于进度。

### 11.4 每一阶段的提交边界

- P5：只提交 `cbf_qp.py`、P5 测试、audit 脚本、P5 报告和必要的协议更新。
- P6：只提交 paired evaluator、aggregate、逐步 trace schema、P6 报告和测试。
- P7：只提交统计/审计脚本、报告、README 索引和 provenance manifest。
- 不暂存、不覆盖用户已有 E1/V5 工作区修改；不要把 `tmp/` 或大体积生成结果强行加入 Git。
- 每次提交前运行 `git diff --check`、相关 pytest 和 `git status --short`，确认新增文件属于当前阶段。
