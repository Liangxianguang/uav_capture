# Interaction-Aware Action-Conditioned JEPA + CBF
# P5--P7 详细续执行计划

**版本：** v1.0

**制定日期：** 2026-09-03

**实验性质：** development-only；不打开、不重跑、不替代 V4/V5 的 locked test

**执行环境：** Windows 10、NVIDIA RTX 5050、CUDA、Conda 环境 `uav-encirclement-gpu`

**项目根目录：** `D:\uav-capture\uav_capture`
**本计划替代范围：** 更新 P5--P7 的当前执行状态；旧版 P4--P7 计划保留为历史决策记录，不删除、不改写。

---

## 1. 目标、问题与最终判定边界

本计划要回答的不是“JEPA 的单次最高捕获率是多少”，而是：在冻结 V5 actor、相同冻结 development 场景、始终启用 CBF 的前提下，**interaction-aware action-conditioned JEPA 是否能可靠地改善候选动作的选择，并带来可重复的闭环围捕收益**。

运行时链路固定为：

```text
冻结 V5 nominal action
  -> 生成 5 个物理可行的常值 desired-action chunk
  -> action-conditioned JEPA 预测目标/净空/可见性/CBF 风险
  -> reliability ledger 判断该上下文是否可信
  -> 低信用时回退 nominal action，否则仅重排序候选
  -> CBF 作为最终安全过滤器
  -> 仅执行所选 chunk 的第一步
  -> 重新观测并重新规划
```

这里的 JEPA 只能改变 CBF 前的候选排序，不能直接声明安全，也不能绕过 CBF、修改环境状态、读取在线 target truth 或更新冻结 actor。

本计划**不设置 `95%` 的绝对捕获率门槛**。最终判断使用三 seed、逐 episode 配对的相对证据：安全失败、捕获差异、路径/时间代价、CBF 干预、动作平滑性与回退比例必须同时报告。单 seed、20 episode smoke 或 prediction gate 均不是正式提升证据。

---

## 2. 当前已冻结的事实与决策

### 2.1 不可变实验合同

| 项目 | 冻结值 |
| --- | --- |
| 基线 actor | `models/v5_development_exact_reactive_seed661606.pt` |
| actor SHA-256 | `535098773be05687e147043435649378532362d479bdc0375842970370ba40ba` |
| actor action scale | `5.0` |
| 观测 / 动作维度 | `63 / 3` |
| 历史长度 / 预测时域 | `8` steps / `1, 2, 3, 5` steps |
| 控制周期 | `0.1 s` |
| JEPA 候选数 | `K = 5` |
| P5 chunk 语义 | 常值 desired action 持续 3 steps 的反事实标签；实际只执行第 1 step 后重新规划 |
| 候选扰动 | `0.10 m/s` |
| 安全机制 | 所有基线和候选均启用 CBF；CBF 最后执行 |
| final development 场景 | 原冻结 S3 development 场景，60 episode，逐 episode 配对 |
| locked test | `locked_test_opened: false`，后续也不得通过本计划打开 |

P5 使用的配置是 `configs/jepa_v3_development_protocol.yaml`。其中动作块只允许 `1/3/5` steps；本轮首个且唯一已经准入的开发块长度为 `3`。在 P6 结束前，不得因 S3 结果切换到 1 或 5 steps、增加候选数、改变扰动、改变 scorer 权重或调节 ledger 阈值。

### 2.2 已完成的前序证据

| 阶段 | 状态 | 已成立的结论 | 对后续的影响 |
| --- | --- | --- | --- |
| P0 输入和协议冻结 | 完成 | 开发、validation 和 locked 数据边界已定义 | 后续只能使用规定的 train/validation 数据训练与建账本 |
| P1 反事实数据 | 完成 | 原 P1 数据有 train/validation 各 146,400 样本，5 candidates/state-agent | 作为 P2/P3/P4 基线证据保留 |
| P2 多任务 JEPA | 完成 | 三 seed 在 `0.2/0.3/0.5 s` target MAE 均优于 constant velocity | 仅说明离线预测存在信号，不说明围捕提升 |
| P3 reliability ledger | 完成 | 账本与 checkpoint 哈希绑定；零扰动回归和单 seed 20-episode non-zero smoke 通过 | 保留作 P5 的保守回退机制 |
| P4 hard-example replay | 完成，拒绝进入控制 | replay-on 在 held-out hard 与 non-hard 数据上系统性损害 target/inter-agent clearance 指标 | **P5/P6 必须使用未重放的多任务 JEPA，不得使用 replay-on checkpoint** |

P4 的负结论不是整个 JEPA 假设的否定，而是当前 `50% uniform + 50% weighted, hard weight=3.0` 的 replay 策略没有通过准入门。P5 不再为 replay 追加 closed-loop 实验。

### 2.3 P5 已经完成的准备工作

以下状态已形成，不应重复采集或覆盖：

| 项目 | 当前状态 / 不变量 |
| --- | --- |
| P5 runtime 实现 | 已实现常值 chunk 生成、chunk reranker、只执行首步、CBF 后置、chunk 合同校验、locked-test 禁止 |
| P5 单元与集成测试 | `18 passed`：多任务、反事实数据、prediction、zero-perturbation 相关测试 |
| chunk-3 train 数据 | `results/jepa_v3_chunk3_counterfactual_train/`，146,400 samples，29,280 个完整五候选组 |
| chunk-3 validation 数据 | `results/jepa_v3_chunk3_counterfactual_validation/`，146,400 samples，29,280 个完整五候选组 |
| train NPZ SHA-256 | `74BC9071CB70D3A9AD626E0D5262083095AD329C3E232FD7EB924921FF1C85E1` |
| validation NPZ SHA-256 | `AC36FD149CA3EF9A6A90D36107DD436069CF71DF486459141C588EE6BFE3C10A` |
| train / validation metadata SHA-256 | `0453AEC807BF92BEED458B342178FE397F6665E4F4CF0BC54F8A2377B85BB1B6` / `8D34A8EE0611AB2DEDB06B3A4E475A64B62A50D6DFA7AD3819D651E616D2CCF2` |
| seed `20260911` | 40 epoch CUDA 训练完成；best epoch `8`，best validation loss `-3.44105641547448` |
| seed `20260912` | 正在以与 seed `20260911` 相同的固定参数训练 |
| seed `20260913` | 尚未启动；只能在 `20260912` 的完整训练审计完成后启动 |

完整候选组在 NPZ 中不保证连续。任何 ranking 或 paired bootstrap 实现必须以 `(episode_seed, time_index, agent_id, candidate_index)` 进行 lexsort/groupby，不得直接 reshape 原始行数组。

---

## 3. 总体执行顺序与冻结点

| 顺序 | 阶段 | 目的 | 完成输出 | 未通过时的动作 |
| ---: | --- | --- | --- | --- |
| 1 | P5-A：完成训练与 provenance 审计 | 得到三个可追溯 chunk-3 checkpoint | 3 个 checkpoint、3 个完整 TensorBoard run、run metadata | 停止相应 seed，不以不完整 run 代替 |
| 2 | P5-B：离线预测与 action-following 准入 | 确认模型可用且真正响应候选动作 | 3 个 prediction gate、3 个 action audit | 不进入 ledger/闭环 |
| 3 | P5-C：建立三个 checkpoint-bound ledger | 对低信用上下文执行确定性 nominal fallback | 3 个 ledger、3 个 ledger report | 不进入闭环 |
| 4 | P5-D：零扰动回归 | 验证引入 chunk/ranker 不改变 nominal + CBF 行为 | paired comparison JSON，全部 0 差异 | 定位回归，修复后回到新版本的 P5-A |
| 5 | P5-E：non-zero smoke 与单 seed diagnostic | 发现接口、安全或严重效率问题；冻结设计 | 20 + 60 paired episodes、P5 report | 判定拒绝或进入 P6；不得据此调参 |
| 6 | P6：三 seed 最终 paired development | 形成方法是否有效的完整开发证据 | 3 x 60 paired episodes、统计与失败审计 | 负结果或证据不足，停在 development |
| 7 | P7：可复现性与 locked-readiness 审计 | 判断是否值得另写新的 preregistration | 审计报告与明确 decision | 不自动运行 locked test |

`P5-E` 开始时，P5 的设计、三个 seed、数据、扰动、chunk 长度、候选数、ledger 参数和评分函数均视为冻结。S3 development 只用于诊断和最终评估，不得反向参与超参数选择。

---

## 4. P5-A：完成 chunk-3 三 seed 训练与训练审计

### 4.1 固定训练矩阵

三个训练 run 仅改变随机 seed：

| seed | output | TensorBoard logdir | 状态 |
| ---: | --- | --- | --- |
| `20260911` | `results/jepa_v3_multitask_chunk3_seed20260911/` | `results/jepa_v3_tensorboard/multitask_chunk3_seed20260911/` | 完成，待统一离线 gate |
| `20260912` | `results/jepa_v3_multitask_chunk3_seed20260912/` | `results/jepa_v3_tensorboard/multitask_chunk3_seed20260912/` | 训练中 |
| `20260913` | `results/jepa_v3_multitask_chunk3_seed20260913/` | `results/jepa_v3_tensorboard/multitask_chunk3_seed20260913/` | 待启动 |

训练参数固定为：`epochs=40`、`batch_size=512`、`device=cuda`、learning rate `0.001`、weight decay `1e-5`、隐藏维度 `128`、latent 维度 `64`。不使用 P4 replay weights。

seed `20260913` 的命令模板如下。启动前必须确认 output 与 TensorBoard 目录不存在或为空，避免覆盖任何已完成证据：

```powershell
$py = 'D:\miniconda3\envs\uav-encirclement-gpu\python.exe'

& $py scripts\train_interaction_aware_jepa_multitask.py `
  --protocol configs\jepa_v3_development_protocol.yaml `
  --train-dataset results\jepa_v3_chunk3_counterfactual_train\counterfactual_multitask_dataset.npz `
  --train-metadata results\jepa_v3_chunk3_counterfactual_train\metadata.json `
  --validation-dataset results\jepa_v3_chunk3_counterfactual_validation\counterfactual_multitask_dataset.npz `
  --validation-metadata results\jepa_v3_chunk3_counterfactual_validation\metadata.json `
  --output results\jepa_v3_multitask_chunk3_seed20260913 `
  --tensorboard-logdir results\jepa_v3_tensorboard\multitask_chunk3_seed20260913 `
  --seed 20260913 --epochs 40 --batch-size 512 --device cuda
```

### 4.2 每个 run 的验收清单

每个 seed 结束后，在启动下一个 seed 前完成以下审计：

- [ ] `checkpoint.pt`、`run_metadata.json` 和 TensorBoard event 文件存在；checkpoint 不为空。
- [ ] metadata 记录 CUDA device、PyTorch/Python 版本、source hashes、输入 metadata、seed、最佳 epoch、最佳 validation loss 和 wall-clock time。
- [ ] checkpoint SHA-256 已写入阶段日志；不可用最佳 loss 代替 checkpoint 身份。
- [ ] TensorBoard 有各 40 个 `Loss/train` 与 `Loss/validation` 点，所需 `Target/*`、`Clearance/*`、`Visibility/*`、`Risk/*`、`Calibration/*`、`Optimization/learning_rate` scalar 与 protocol/source-hash text artifact 均存在。
- [ ] histogram 审计完整，且没有 NaN/Inf 或意外提前结束。
- [ ] 训练目录、TensorBoard 目录与其他 seed 的目录不同；任何冲突均停止而不是覆盖。

建议使用现有 `_tensorboard_audit` 例程记录机器可读审计结果：

```powershell
& $py -c "import json; from pathlib import Path; from scripts.aggregate_jepa_v3_multitask import _tensorboard_audit; print(json.dumps(_tensorboard_audit(Path('results/jepa_v3_tensorboard/multitask_chunk3_seed20260912')), indent=2))"
```

### 4.3 P5-A 准入标准

只有三个 run 都完成、各自 provenance/TensorBoard 审计通过，才可进入 P5-B。某个训练 seed 无法通过时，保留其失败输出和错误日志；不得挑选剩余两个 seed 直接声称三 seed 结果。

---

## 5. P5-B：离线 prediction gate 与 action-following 审计

### 5.1 Held-out prediction gate

每个 checkpoint 只能在 chunk-3 **validation** archive 上评估。至少报告四个时域的 target MAE、constant-velocity 对比、obstacle/inter-agent clearance MAE、visibility Brier/AUROC、CBF intervention Brier/AUROC、coverage 和全有限性。输出必须归属各自 seed：

```powershell
& $py scripts\evaluate_jepa_v3_multitask.py `
  --checkpoint results\jepa_v3_multitask_chunk3_seed<seed>\checkpoint.pt `
  --dataset results\jepa_v3_chunk3_counterfactual_validation\counterfactual_multitask_dataset.npz `
  --metadata results\jepa_v3_chunk3_counterfactual_validation\metadata.json `
  --output results\jepa_v3_multitask_chunk3_seed<seed>\prediction_gate.json `
  --device cuda
```

准入条件为：全部输出有限、所有辅助任务存在、至少一个预定义时域的 target prediction 优于 constant velocity。通过该 gate 只表示可以进入 development control smoke；它不等价于闭环性能提升。

### 5.2 Action-following audit

该审计固定使用 validation 数据与小幅 final-action 扰动，验证模型在相同 state/history 下能区分候选 action，而不是只复读 observation。每个 checkpoint 输出 JSON 和 Markdown：

```powershell
& $py scripts\audit_jepa_action_following.py `
  --checkpoint results\jepa_v3_multitask_chunk3_seed<seed>\checkpoint.pt `
  --dataset results\jepa_v3_chunk3_counterfactual_validation\counterfactual_multitask_dataset.npz `
  --metadata results\jepa_v3_chunk3_counterfactual_validation\metadata.json `
  --sample-count 4096 --perturbation 0.02 --device cuda `
  --output-json results\jepa_v3_multitask_chunk3_seed<seed>\action_following_audit.json `
  --output-md results\jepa_v3_multitask_chunk3_seed<seed>\action_following_audit.md
```

验收条件：输出有限、候选 separation 非零且有合理的 antisymmetry 行为。该结果是作用机制检查，不是安全证书。若某个 seed 近似 action-insensitive 或数值不稳定，则该 seed 不能进入 P5-C/P6。

---

## 6. P5-C：为每个 checkpoint 建立独立 reliability ledger

每个 checkpoint 只能使用 chunk-3 validation 的**已结算 cloned-rollout 标签**建立 ledger；不得使用 train 数据、S3 development episode、locked data 或在线执行结果更新 ledger。ledger 必须记录并验证所绑定的 checkpoint SHA-256。

固定 ledger 参数：horizon index `3` (`0.5 s`)、minimum sample count `128`、minimum credit `0.65`、`K=5`、候选扰动 `0.10 m/s`。低信用、缺失 bucket 或 OOD context 均回退到冻结 V5 nominal action，然后仍通过 CBF。

```powershell
& $py scripts\build_jepa_v3_reliability_ledger.py `
  --checkpoint results\jepa_v3_multitask_chunk3_seed<seed>\checkpoint.pt `
  --dataset results\jepa_v3_chunk3_counterfactual_validation\counterfactual_multitask_dataset.npz `
  --metadata results\jepa_v3_chunk3_counterfactual_validation\metadata.json `
  --minimum-sample-count 128 --minimum-credit 0.65 --device cuda `
  --output results\jepa_v3_multitask_chunk3_seed<seed>\reliability_ledger.json `
  --report results\jepa_v3_multitask_chunk3_seed<seed>\reliability_ledger_report.md
```

验收时确认：

- [ ] ledger source split 为 `validation`，不是 train/development/locked；
- [ ] 所有 source dataset、metadata 和 checkpoint 哈希一致；
- [ ] ledger 运行时无法与不同 checkpoint 配对加载；
- [ ] 报告分别给出 local、global 与 OOD fallback 预测比例，不将低 fallback 误写为安全；
- [ ] 账本文件完成后只读，不在 P5/P6 控制中在线更新。

---

## 7. P5-D：零扰动配对回归

### 7.1 目的与原则

在 `perturbation=0.0` 时，五个候选 chunk 的首步均应与冻结 V5 nominal action 相同。因此加入 JEPA、chunk 与 ledger 后，除了 JEPA 专属日志字段外，控制结果必须逐字段完全等价。该门先于任何 non-zero 场景结果，任何不一致都视为实现回归而不是“模型带来的差异”。

### 7.2 运行设计

使用同一批 20 个冻结 development scenes：

1. 生成或复用已存在的 V5 + CBF baseline 20-episode 结果；仅当 `scenes.jsonl` 与 reference episodes 的哈希经验证完全一致时可复用。
2. 用 P5 chunk-3 checkpoint + 与它 hash-bound 的 ledger，`K=5`、`perturbation=0.0`、`chunk=3`、CBF enabled，在完全相同的 reference scenes 和 reference episodes 上运行 candidate。
3. 执行 `compare_jepa_v3_zero_perturbation.py`，比较 paired episode identity、场景字节、以及全部非 JEPA 字段。

```powershell
& $py scripts\compare_jepa_v3_zero_perturbation.py `
  --baseline-dir results\<paired-baseline20> `
  --candidate-dir results\jepa_v3_p5_chunk3_seed20260911_zero20 `
  --output results\jepa_v3_p5_chunk3_seed20260911_zero20\zero_perturbation_comparison.json
```

通过门：20/20 episode identity 配对、`scenes.jsonl` byte-identical、全部非 JEPA episode 字段差异数为 0、CBF/collision/boundary/capture/path/time 结果完全一致。失败时不得进入 smoke 或调参；先记录差异字段，再修复实现，并以新的版本号重走 P5-A 至 P5-D。

---

## 8. P5-E：non-zero smoke 与单 seed 60-episode 诊断

### 8.1 20-episode smoke

仅在零扰动回归通过后，先对 seed `20260911` 执行固定的 20 episode non-zero smoke：`K=5`、`perturbation=0.10 m/s`、`chunk=3`、matching ledger、CBF enabled、同一 reference scenes/episodes。输出目录必须是新的空目录。

示意命令如下，实际 baseline checkpoint、reference 文件和 output 名称写入 run metadata：

```powershell
& $py scripts\evaluate_random_central_mixed_obstacles.py `
  --method f2 `
  --checkpoint models\v5_development_exact_reactive_seed661606.pt `
  --protocol configs\central_random_mixed_obstacle_s3_v5_protocol.yaml `
  --split validation --episodes 20 --use-cbf --device cuda `
  --action-conditioned-jepa-checkpoint results\jepa_v3_multitask_chunk3_seed20260911\checkpoint.pt `
  --jepa-reliability-ledger results\jepa_v3_multitask_chunk3_seed20260911\reliability_ledger.json `
  --jepa-candidate-count 5 --jepa-perturbation-mps 0.10 --jepa-action-chunk-length 3 `
  --reference-scenes results\<paired-baseline20>\scenes.jsonl `
  --reference-episodes results\<paired-baseline20>\episodes.csv `
  --output-dir results\jepa_v3_p5_chunk3_seed20260911_smoke20
```

smoke 必报指标：safe capture、collision、boundary、timeout、transit、paired improved/degraded/tied、capture time、总路径、最小 obstacle/inter-agent clearance、CBF correction/intervention、selected candidate index、action-change rate、ledger credit、nominal/global fallback、平均控制时延与候选评分时延。

下列任意情况拒绝 P5，不运行 60 episode：新的或无法解释的 collision/boundary；场景/episode 不配对；非有限输出；CBF 没有最后执行；严重效率恶化（相对 baseline 的平均 path 或 capture time 增加超过协议 `10%`）；或完整性/provenance 缺失。

### 8.2 60-episode 单 seed diagnostic

smoke 通过后，继续使用**相同、已冻结**的 seed `20260911` checkpoint/ledger/configuration，在完整的 60 frozen development episodes 上运行。该 run 的职责是评估失败分桶与成本构成，不是选择更好的 chunk、扰动、权重或 seed。

必须生成：`episodes.csv`、`scenes.jsonl`、`summary.json`、`evaluation_metadata.json`、protocol 副本、逐 episode delta 表、失败根因清单和 P5 报告。每个失败按 collision、boundary、timeout、transit、ledger fallback、CBF correction 过大、候选排序反向、可见性/净空误判或其他明确类别分类。

### 8.3 P5 决策

| 结果 | 判定 | 后续 |
| --- | --- | --- |
| 零扰动完整等价，smoke/diagnostic 无新安全失败，成本未超过阈值 | `admitted_to_p6` | 固定 P5 设计，进入三 seed P6 |
| 离线 gate 通过，但 non-zero 无捕获收益且成本可接受 | `prediction_signal_no_single_seed_control_gain` | 可进入 P6 获取三 seed 反证；不得调参 |
| 有新 collision/boundary、回归失败、非确定性或显著成本退化 | `rejected_for_control` | 停止该变体，写 P5 rejection report |
| S3 证据不足或区间很宽 | `insufficient_single_seed_evidence` | 不开 locked；是否进入 P6 只按安全/完整性准入决定 |

P5 收尾时运行相关测试、完成 P5 report，并且只提交 P5 所属代码、测试、配置和文档。建议提交信息：`feat(jepa): rerank feasible action chunks with jepa`。不得把 `results/`、`tmp/`、E1-prime 或无关 V5 文件纳入提交。

---

## 9. P6：最终三 seed、逐 episode 配对 development 评估

### 9.1 评估对象与配对原则

P6 的唯一目的是判断这个已冻结的 P5 变体是否具有重复性。对每个训练 seed `20260911/20260912/20260913`，使用它自己的 chunk-3 checkpoint 与 hash-bound ledger，在相同 60 个冻结 development scenes 上与 **V5 + CBF baseline** 逐 episode 配对。baseline scene/episode 文件只生成一次，并由 SHA-256 固定；候选运行均从该 reference 还原几何与 policy-independent Transit 证据。

P6 不使用 P4 replay-on。P6 不是 V4/V5 historical locked test，且不允许设置 `--split locked_test`。

### 9.2 两层运行顺序

| 层级 | 每个 seed 的规模 | 目标 | 允许的决策 |
| --- | ---: | --- | --- |
| P6 smoke | 20 paired episodes | 确认每个 seed 没有接口或安全异常 | 只允许停止/拒绝；不允许调参 |
| P6 final | 60 paired episodes | 得到最终三 seed 开发证据 | 只能分类与报告；不允许再训练或调参 |

如果 seed `20260912` 或 `20260913` 的 P6 smoke 触发安全停止规则，整个候选应被分类为不稳定，不能只报告剩余 seed 的正向结果。

### 9.3 必报统计和证据标准

每个 seed 及三 seed 汇总均报告以下内容：

- 安全与任务：safe capture count/rate，collision、boundary、timeout、transit count/rate。
- 配对差异：每个 episode 的 baseline/candidate outcome，improved/degraded/tied，capture delta；三 seed mean +/- standard deviation；paired block bootstrap 95% CI；二元捕获指标同时给出 McNemar 或 exact paired test，并说明样本量限制。
- 效率：capture time、总 defender path length、最小 obstacle/inter-agent clearance、mean/max CBF correction、intervention fraction、action-change rate、chunk smoothness、候选评分/总控制时延。
- 可靠性：ledger mean credit、local nominal fallback、global/OOD fallback；按可见性、布局、障碍数、minimum clearance、target motion、CBF correction 分桶。
- 预测到控制：对应 checkpoint 的 prediction gate、action-following audit、ledger report 与闭环失败的链接；不得只报告最有利的时域或分桶。

### 9.4 P6 预定义最终分类

| 分类 | 最低证据要求 | 表述边界 |
| --- | --- | --- |
| `promising_development_candidate` | 三 seed 没有新且未解释的安全退化；mean paired capture delta 为正；至少 2/3 seed 非负；不是单 seed 驱动；效率代价完整且合理 | 仅可称为 development 正向证据，进入 P7 |
| `useful_safety_fallback_only` | 捕获不稳定，但账本回退能解释并避免风险，且无安全损害 | 可称为安全/诊断机制证据，不称捕获提升 |
| `prediction_improvement_no_control_gain` | 离线预测/排名有改善，闭环捕获和效率无净收益 | 明确记录负结果，不开 locked |
| `rejected_for_instability` | 任何 seed 出现未解释 collision/boundary、零扰动回归破坏或严重效率退化 | 停止该候选，不以平均值抵消 |
| `insufficient_evidence_do_not_open_locked_test` | seed 矛盾、置信区间过宽或材料不完整 | 只报告不确定性，不开 locked |

P6 的提交仅包括总结统计脚本、报告和必要测试，建议信息：`docs(jepa): record three-seed development evaluation`。

---

## 10. P7：可复现性审计与新 locked 前决策

P7 不执行任何 locked benchmark。它只判断是否值得由一份新的 preregistration 单独定义新的 locked block。

### 10.1 审计清单

- [ ] 三个最终 checkpoint、三个 ledger、两个 chunk-3 dataset、两个 metadata、protocol、代码版本、场景文件均有 SHA-256。
- [ ] 每个训练 run 的 CUDA/TensorBoard/provenance 完整；任一 run 均能在 RTX 5050 环境从空 results namespace 重放。
- [ ] prediction gate、action-following、chunk 合同、zero-perturbation、CBF order、scene/episode pairing 测试均通过。
- [ ] P6 的全部 180 candidate episodes、60 baseline scenes、完整 CSV/JSON 和 failure classifications 可读。
- [ ] `locked_test_opened` 仍为 `false`；没有改写 V4/V5 历史 checkpoint、archive、locked scenes、locked reports 或协议。
- [ ] 捕获、时间、路径、净空、CBF、fallback、控制时延和负结果均已披露。
- [ ] 最终文字没有把 development signal 写成 locked improvement，没有把单 seed 的结果写成三 seed 结论。

### 10.2 是否提议新的 preregistration

仅当 P6 被分类为 `promising_development_candidate`，并且以下条件全部满足时，才可写一份**新的** locked-test preregistration 草案：

1. 三 seed 的正向趋势可解释且不是个别场景、单个训练 seed 或统计偶然造成；
2. CBF 下没有新的、未解释的 collision/boundary；
3. 路径和时间成本没有超过已披露的 `10%` 阈值，或有明确研究理由；
4. checkpoint/ledger/数据/环境/场景的哈希和复现命令完整；
5. 主指标、次指标、样本量、seed block、停止规则、统计方法和失败处理在运行前写死；
6. 用户单独授权打开新的 locked block。

任一条件未满足时，P7 的正确输出是“不打开 locked test”，而不是继续用 development 场景调参。

建议提交信息：`docs(jepa): audit locked-test readiness decision`。

---

## 11. 结果目录、日志与提交纪律

### 11.1 推荐目录

```text
results/jepa_v3_multitask_chunk3_seed<seed>/
  checkpoint.pt
  run_metadata.json
  prediction_gate.json
  action_following_audit.json
  action_following_audit.md
  reliability_ledger.json
  reliability_ledger_report.md

results/jepa_v3_tensorboard/multitask_chunk3_seed<seed>/
results/jepa_v3_p5_chunk3_seed20260911_zero20/
results/jepa_v3_p5_chunk3_seed20260911_smoke20/
results/jepa_v3_p5_chunk3_seed20260911_dev60/
results/jepa_v3_p6_chunk3_seed<seed>_smoke20/
results/jepa_v3_p6_chunk3_seed<seed>_dev60/
docs/JEPA_V3_P5_ACTION_CHUNK_REPORT_20260903.md
docs/JEPA_V3_P6_THREE_SEED_DEVELOPMENT_REPORT_20260903.md
docs/JEPA_V3_P7_LOCKED_READINESS_AUDIT_20260903.md
```

任何结果目录在运行前必须为空；脚本遇到非空目录应中止。结果文件由 `.gitignore` 管理，报告中记录其路径和 SHA-256，但不把大规模数据、checkpoint、TensorBoard event 或 `tmp/` 加入 Git。

### 11.2 每次运行后的最小记录

每次训练或闭环评估结束立即记录：完整命令、Git commit、GPU/软件版本、输入/输出 SHA-256、随机 seed、开始/结束时间、执行结果、失败或中断原因。不要在多个阶段之后补写 provenance。

提交边界必须小而清晰：P5 实现与报告、P6 统计报告、P7 审计报告分别提交；每阶段测试和文档完成后才 `git push origin main`。当前工作区存在 E1-prime、execution dynamics、历史 V5 和 `tmp/` 的用户变更，必须保持未触碰且不混入 JEPA 提交。

---

## 12. 计划完成定义

本计划完成不等于“模型一定胜出”。P5--P7 只有在下列条件均满足时才算完成：

1. 三个 chunk-3 seed 均有可审计训练记录、离线 gate、action-following audit 和 hash-bound ledger；
2. 零扰动回归通过，non-zero P5 与 P6 控制始终 CBF 后置、场景逐 episode 配对；
3. 完成 `3 x 60` 最终 development 评估或按预定义安全规则明确停止；
4. 对正向、负向和不确定证据都给出完整指标、失败分桶和可复现路径；
5. P7 明确写出“建议新的 preregistration”或“不打开 locked test”的结论；
6. 所有相关代码、测试和文档按阶段提交并推送，且没有覆盖或改写历史 V4/V5 evidence。

这套证据链的优先级是：先确认动作块和预测器没有改变 nominal 控制或破坏 CBF，再检查其在冻结场景中的实际价值；最终无论结果是正、负还是不足，都能够追溯“在哪类互动、可见性、净空和 CBF 工况下有效或失效，以及由何种机制导致”。
