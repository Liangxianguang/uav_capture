# Interaction-Aware Action-Conditioned JEPA + CBF
# P5–P7 执行版详细计划书

**版本：** v1.1（v2 精度合同修复后的执行版）  
**制定日期：** 2026-09-03  
**实验性质：** development-only；不打开、不替代、不重写 V4/V5 historical locked test  
**项目根目录：** `D:\uav-capture\uav_capture`  
**GPU 环境：** NVIDIA RTX 5050；Conda 环境 `uav-encirclement-gpu`  
**GPU Python：** `D:\miniconda3\envs\uav-encirclement-gpu\python.exe`

> 本文件是当前执行状态的工作计划。历史计划、V4/V5 locked 报告、P4 replay-on 负结果和既有结果目录均保留。本文不授权运行任何 locked test，也不把 development 结果写成正式论文结论。

---

## 1. 研究目标与最终问题

本轮实验要回答：

> 在冻结的 V5 actor、相同冻结 development 场景、相同 CBF 安全过滤器和三组训练 seed 下，interaction-aware action-conditioned JEPA 是否能更可靠地从多个物理可行候选动作中选择有利动作，并形成可重复的闭环围捕收益？

运行链路固定为：

```text
冻结 V5 nominal action
    ↓
生成 K=5 个物理可行的常值 desired-action chunk
    ↓
JEPA 预测目标位移、障碍净空、机间净空、可见性和 CBF 干预风险
    ↓
reliability ledger 判断当前上下文的预测信用
    ↓
高信用：只重排序候选；低信用/OOD：回退 nominal
    ↓
CBF 最后进行安全投影
    ↓
只执行 chunk 的第 1 步，重新观测、重新规划
```

JEPA 的权限严格受限：它只能影响 CBF 前的候选排序，不能绕过 CBF、读取在线 target truth、改变环境动力学、更新冻结 actor 或直接声明安全。

### 1.1 不采用绝对 95% 门槛

本计划不要求捕获率必须达到预注册的 `95%`。最终决策依据为三 seed、逐 episode 配对的整体证据：

- 是否出现新的 collision/boundary 或未解释的安全退化；
- capture、timeout、transit 的配对差异；
- capture time、路径长度、最小净空和 CBF 干预代价；
- 三个训练 seed 的一致性，而不是单个最好 seed；
- fallback、可见性、布局、障碍数量和困难片段分桶；
- 结果是否具有完整的 provenance、TensorBoard 和可复现命令。

单 seed、20 episode smoke、离线 prediction gate 和 action-following audit 都不是最终闭环提升证据。

---

## 2. 当前状态快照（执行起点）

### 2.1 已冻结的基线

| 项目 | 当前固定值 |
|---|---|
| V5 development actor | `models/v5_development_exact_reactive_seed661606.pt` |
| actor SHA-256 | `535098773be05687e147043435649378532362d479bdc0375842970370ba40ba` |
| actor action scale | `5.0` |
| observation/action dimension | `63 / 3` |
| history length | `8` steps |
| prediction horizons | `1, 2, 3, 5` steps（`0.1/0.2/0.3/0.5 s`） |
| control period | `0.1 s` |
| candidate count | `K=5` |
| candidate perturbation | `0.10 m/s` |
| first accepted action chunk | constant desired-action chunk, `3` steps |
| execution semantics | 只执行第 1 步，然后重新规划 |
| safety | baseline 和 candidate 均启用 CBF，CBF 最后执行 |
| final development block | 原冻结 S3 development 场景，60 episodes，逐 episode 配对 |
| locked test | `locked_test_opened: false`，本计划全程保持 `false` |

### 2.2 已成立的前序证据

| 阶段 | 状态 | 结论边界 |
|---|---|---|
| P0 协议冻结 | 完成 | 已划分 train、validation、development 和 locked 边界 |
| P1 反事实数据 | 完成 | 已建立五候选、多时域、离线标签的数据合同 |
| P2 多任务 JEPA | 完成 | 三 seed 在部分/全部 horizon 优于 constant velocity；只证明离线预测信号 |
| P3 reliability ledger | 完成 | ledger 与 checkpoint 哈希绑定，可用于保守回退 |
| P4 hard-example replay | 完成但拒绝 | replay-on 在 hard 与 non-hard validation 上系统性损害 target/inter-agent clearance，不进入 P5/P6 |

P4 的负结论只针对当前 `50% uniform + 50% weighted、hard weight=3.0` 的 replay 策略，不否定 interaction-aware JEPA + CBF 的总体假设。

### 2.3 P5 v1 为什么作废

`results/jepa_v3_chunk3_counterfactual_*` 和对应 `jepa_v3_multitask_chunk3_seed*` 是旧版精度合同产物，不能与当前 runtime 混用。旧版存在以下问题：

1. 候选动作在生成、reranker 输出和反事实 rollout 中使用 `float32`；
2. 冻结 V5 baseline CBF 使用 `float64`；
3. 因此在 `perturbation=0.0` 时，candidate 与 nominal 的物理轨迹出现微小但可传播的差异；
4. 旧版 zero-regression 出现 `87` 个非 JEPA 字段、共 `122` 个差异，故不能进入闭环结论。

旧版 checkpoint、ledger、prediction gate 和闭环结果必须保留为审计证据，但在本计划中标记为 `superseded_invalid_precision_contract`，不得作为 P5/P6 输入。

### 2.4 当前 v2 archive 状态

精度合同修复后，v2 train/validation archive 已完成采集，下一步是审计，而不是重新采集：

| 文件 | 路径 | SHA-256 |
|---|---|---|
| train NPZ | `results/jepa_v3_chunk3v2_counterfactual_train/counterfactual_multitask_dataset.npz` | `0d165646db5f0545115fa5f8cdb2bc6fd44b9ab2db5981e8de5b96963e84787c` |
| train metadata | `results/jepa_v3_chunk3v2_counterfactual_train/metadata.json` | `6d6ae07cd74981dc9097a38cc3dc43bcac6baff82aed838e126f69bd963542c4` |
| train scenario manifest | `results/jepa_v3_chunk3v2_counterfactual_train/scenario_manifest.json` | `26a1cea0e95d3d77926232fd91863d148653a45255670d978ab20fd57b353ed1` |
| validation NPZ | `results/jepa_v3_chunk3v2_counterfactual_validation/counterfactual_multitask_dataset.npz` | `1c04b9556b95fbcc050678fc4ee3a1b62b45c9185bc928d904be18745ddfe51c` |
| validation metadata | `results/jepa_v3_chunk3v2_counterfactual_validation/metadata.json` | `28be533c4665ada49ccff4f1c1482c538b90e9f6e11a9b1ec3ed362a23e247af` |
| validation scenario manifest | `results/jepa_v3_chunk3v2_counterfactual_validation/scenario_manifest.json` | `48a3227a434e7db86e4d47a7a9521ab020b35a0bbba5892e672d3dfb7cef8737` |

两份 archive 均应为 `146,400` samples、`8×63` 输入、每个 state-agent group 恰好 `5` 个候选、`chunk_length_steps=3`、动作尺度 `5.0`。当前 metadata 已记录：

- train/validation 各四类场景，每类 30 episodes；
- target truth 仅用于离线标签；
- development/locked 数据未用于训练；
- action history 已按冻结 actor action scale 归一化；
- 候选语义为 `constant_desired_action_chunk_execute_first_step_then_replan`。

---

## 3. 实验合同：本轮禁止改变的因素

### 3.1 数据和模型合同

| 类别 | 固定内容 |
|---|---|
| train data | 仅使用 v2 train archive |
| validation data | 仅用于 prediction gate、action audit、ledger；不用于训练 |
| development data | 仅用于配对闭环评估；不进入训练或 ledger 更新 |
| locked data | 全程禁止读取/运行 |
| model | interaction-aware action-conditioned JEPA multitask |
| auxiliary tasks | target displacement、obstacle clearance、inter-agent clearance、visibility、CBF intervention risk |
| replay | P4 replay-on 禁止；训练使用 replay-off |
| candidate ranking | 只在 CBF 前进行 |
| CBF | 所有 baseline/candidate 均启用，且为最后安全过滤器 |

### 3.2 运行参数合同

- `epochs=40`，`batch_size=512`，`device=cuda`；
- hidden dimension `128`，latent dimension `64`，one recurrent/interaction layer；
- learning rate `1e-3`，weight decay `1e-5`；
- `K=5`，candidate perturbation `0.10 m/s`，chunk length `3`；
- ledger horizon index `3`（`0.5 s`），minimum sample count `128`，minimum credit `0.65`；
- 不因单 seed 结果切换 chunk length、candidate count、scorer 权重、扰动大小或 ledger 阈值；
- P5/P6 期间不进行基于 S3 结果的调参。

---

## 4. 总体阶段和依赖关系

```text
P5-0 v2 archive 审计
        ↓
P5-A 三 seed 重新训练 + TensorBoard/provenance
        ↓
P5-B prediction gate + action-following
        ↓
P5-C 三个 hash-bound reliability ledger
        ↓
P5-D zero-perturbation paired regression
        ↓
P5-E seed-11 20-episode smoke
        ↓
P5-F seed-11 60-episode diagnostic + P5 report
        ↓
P6 三 seed：20 smoke → 60 final paired development
        ↓
P7 reproducibility / locked-readiness audit
        ↓
只在满足条件且用户另行授权时，提出新的 preregistration 草案
```

任何上游 gate 失败，停止下游闭环运行；不得用较小 episode 数、较宽比较条件或更好看的 seed 替代失败证据。

---

## 5. P5-0：v2 archive 审计

### 5.1 执行命令

```powershell
$py = 'D:\miniconda3\envs\uav-encirclement-gpu\python.exe'

& $py scripts\audit_jepa_v3_counterfactual_dataset.py `
  --dataset-dir results\jepa_v3_chunk3v2_counterfactual_validation `
  --compare-dataset-dir results\jepa_v3_chunk3v2_counterfactual_train `
  --output results\jepa_v3_chunk3v2_dataset_audit.json
```

必要时补充独立 SHA-256 记录：

```powershell
Get-FileHash results\jepa_v3_chunk3v2_counterfactual_train\counterfactual_multitask_dataset.npz -Algorithm SHA256
Get-FileHash results\jepa_v3_chunk3v2_counterfactual_validation\counterfactual_multitask_dataset.npz -Algorithm SHA256
Get-FileHash results\jepa_v3_chunk3v2_counterfactual_train\metadata.json -Algorithm SHA256
Get-FileHash results\jepa_v3_chunk3v2_counterfactual_validation\metadata.json -Algorithm SHA256
```

### 5.2 必须通过的条件

- train/validation 各 `146,400` samples；
- 每个 `(episode_seed, time_index, agent_id)` 恰好五个候选；
- 候选组通过 lexsort/groupby 验证，不能假定 NPZ 行天然连续；
- train 与 validation 的 episode seed 不重叠；
- 所有输入和标签 finite，无 NaN/Inf；
- `chunk_length_steps=3`、`candidate_count=5`、`candidate_perturbation_mps=0.10`；
- action scale 与冻结 actor 均为 `5.0`，action history 已归一化；
- train metadata 明确证明 development/locked 数据未参与训练；
- 采集脚本、环境、protocol、冻结 actor 的哈希均写入审计 JSON。

失败处理：保留审计 JSON 和失败日志，不覆盖 archive；若是数据合同错误，建立新版本目录并从 P5-0 重来。

---

## 6. P5-A：三 seed v2 训练与 TensorBoard 审计

### 6.1 输出命名

必须使用新命名，避免误读为 v1：

```text
results/jepa_v3_multitask_chunk3v2_seed20260911/
results/jepa_v3_multitask_chunk3v2_seed20260912/
results/jepa_v3_multitask_chunk3v2_seed20260913/

results/jepa_v3_tensorboard/multitask_chunk3v2_seed20260911/
results/jepa_v3_tensorboard/multitask_chunk3v2_seed20260912/
results/jepa_v3_tensorboard/multitask_chunk3v2_seed20260913/
```

所有目录在运行前必须不存在或为空；脚本不得覆盖已有 v1 结果。

### 6.2 单 seed 命令模板

每个 seed 独立运行；先完成该 seed 的审计，再启动下一个 seed：

```powershell
$seed = 20260911   # 依次替换为 20260912、20260913
$py = 'D:\miniconda3\envs\uav-encirclement-gpu\python.exe'

& $py scripts\train_interaction_aware_jepa_multitask.py `
  --protocol configs\jepa_v3_development_protocol.yaml `
  --train-dataset results\jepa_v3_chunk3v2_counterfactual_train\counterfactual_multitask_dataset.npz `
  --train-metadata results\jepa_v3_chunk3v2_counterfactual_train\metadata.json `
  --validation-dataset results\jepa_v3_chunk3v2_counterfactual_validation\counterfactual_multitask_dataset.npz `
  --validation-metadata results\jepa_v3_chunk3v2_counterfactual_validation\metadata.json `
  --output results\jepa_v3_multitask_chunk3v2_seed$seed `
  --tensorboard-logdir results\jepa_v3_tensorboard\multitask_chunk3v2_seed$seed `
  --seed $seed --epochs 40 --batch-size 512 --device cuda
```

禁止提供 `--train-replay-weights` 或 `--train-replay-manifest`。

### 6.3 训练后验收

每个 run 必须包含：

- `checkpoint.pt`、`run_metadata.json`、`history.json`；
- checkpoint SHA-256；
- CUDA、PyTorch、Python、GPU、seed、epoch、optimizer、source hashes；
- train/validation NPZ 和 metadata 的绝对路径及 SHA-256；
- best epoch、best validation loss、wall-clock time；
- replay 明确为 disabled。

TensorBoard 必须包含：

- `Loss/train` 与 `Loss/validation` 各 40 个 epoch 点；
- `Target/*`、`Clearance/*`、`Visibility/*`、`Risk/*`、`Calibration/validation`、`Optimization/learning_rate`；
- protocol、train metadata、validation metadata、source hashes、model/optimizer 配置文本 artifact；
- histogram 每 5 epoch 记录，完整审计应为 `149` 个 histogram tags；
- 无 NaN/Inf，无提前结束或空 event file。

审计命令：

```powershell
$seed = 20260911
& $py -c "import json; from pathlib import Path; from scripts.aggregate_jepa_v3_multitask import _tensorboard_audit; print(json.dumps(_tensorboard_audit(Path('results/jepa_v3_tensorboard/multitask_chunk3v2_seed$seed')), indent=2))"
```

若训练中断：保留该 run、stderr 和 TensorBoard，标记 `incomplete`；不得用不完整 run 进入 P5-B。

---

## 7. P5-B：离线 prediction gate 与 action-following audit

### 7.1 Prediction gate

```powershell
$seed = 20260911
& $py scripts\evaluate_jepa_v3_multitask.py `
  --checkpoint results\jepa_v3_multitask_chunk3v2_seed$seed\checkpoint.pt `
  --dataset results\jepa_v3_chunk3v2_counterfactual_validation\counterfactual_multitask_dataset.npz `
  --metadata results\jepa_v3_chunk3v2_counterfactual_validation\metadata.json `
  --output results\jepa_v3_multitask_chunk3v2_seed$seed\prediction_gate.json `
  --device cuda
```

每个 seed 均须报告四个 horizon 的：target MAE、constant-velocity MAE、clearance MAE、visibility Brier/AUROC、CBF correction MAE、CBF intervention Brier/AUROC、coverage 和 finite 检查。

最低准入条件：

- 所有输出 finite；
- 五个辅助任务都存在；
- 至少一个预定义 horizon 的 target prediction 优于 constant velocity；
- 输入、输出、checkpoint model contract 一致。

此 gate 只允许进入 development control smoke，不代表控制提升。

### 7.2 Action-following audit

```powershell
$seed = 20260911
& $py scripts\audit_jepa_action_following.py `
  --checkpoint results\jepa_v3_multitask_chunk3v2_seed$seed\checkpoint.pt `
  --dataset results\jepa_v3_chunk3v2_counterfactual_validation\counterfactual_multitask_dataset.npz `
  --metadata results\jepa_v3_chunk3v2_counterfactual_validation\metadata.json `
  --sample-count 4096 --perturbation 0.02 --device cuda `
  --output-json results\jepa_v3_multitask_chunk3v2_seed$seed\action_following_audit.json `
  --output-md results\jepa_v3_multitask_chunk3v2_seed$seed\action_following_audit.md
```

验收：候选响应 finite、separation 非零、扰动方向具有合理 antisymmetry；若某 seed action-insensitive 或数值不稳定，该 seed 不得进入 ledger/闭环，并在最终汇总中保留失败。

---

## 8. P5-C：建立三个 checkpoint-bound reliability ledger

ledger 只使用 validation archive 的已结算 cloned-rollout 标签，不使用 train、S3 development、locked 或在线执行结果。每个 ledger 必须绑定自己的 checkpoint SHA-256。

```powershell
$seed = 20260911
& $py scripts\build_jepa_v3_reliability_ledger.py `
  --checkpoint results\jepa_v3_multitask_chunk3v2_seed$seed\checkpoint.pt `
  --dataset results\jepa_v3_chunk3v2_counterfactual_validation\counterfactual_multitask_dataset.npz `
  --metadata results\jepa_v3_chunk3v2_counterfactual_validation\metadata.json `
  --minimum-sample-count 128 --minimum-credit 0.65 --device cuda `
  --output results\jepa_v3_multitask_chunk3v2_seed$seed\reliability_ledger.json `
  --report results\jepa_v3_multitask_chunk3v2_seed$seed\reliability_ledger_report.md
```

ledger 必须记录：

- checkpoint、NPZ、metadata、protocol 的 SHA-256；
- horizon index `3`、candidate count `5`、perturbation `0.10 m/s`；
- local bucket、global bucket、OOD bucket 的 sample count 和 credit；
- low-credit/OOD 的确定性 nominal fallback 规则；
- `update_rule=offline execution-settled validation outcomes only`。

完成后 ledger 只读。P5/P6 控制运行不得在线更新、重新拟合或按 development 结果修改阈值。

---

## 9. P5-D：zero-perturbation 严格回归

### 9.1 目的

当 `perturbation=0.0` 时，五个候选的首步应与冻结 V5 nominal action 相同。因此 candidate runtime 除 JEPA 专属日志字段外，不得改变 baseline 的物理轨迹、CBF、collision、boundary、capture、path 或 time 字段。

### 9.2 运行与比较

候选目录必须使用 v2 和新 runtime 命名：

```powershell
$seed = 20260911
& $py scripts\evaluate_random_central_mixed_obstacles.py `
  --method f2 `
  --checkpoint models\v5_development_exact_reactive_seed661606.pt `
  --environment-config configs\capture_radius_pursuit_central_v4_flee.yaml `
  --protocol configs\central_random_mixed_obstacle_s3_v5_protocol.yaml `
  --split validation --episodes 20 --use-cbf --device cuda `
  --action-conditioned-jepa-checkpoint results\jepa_v3_multitask_chunk3v2_seed$seed\checkpoint.pt `
  --jepa-reliability-ledger results\jepa_v3_multitask_chunk3v2_seed$seed\reliability_ledger.json `
  --jepa-candidate-count 5 --jepa-perturbation-mps 0.0 --jepa-action-chunk-length 3 `
  --reference-scenes results\jepa_v3_p3_zero_baseline20\scenes.jsonl `
  --reference-episodes results\jepa_v3_p3_zero_baseline20\episodes.csv `
  --output-dir results\jepa_v3_p5_chunk3v2_seed$seed\zero20

& $py scripts\compare_jepa_v3_zero_perturbation.py `
  --baseline-dir results\jepa_v3_p3_zero_baseline20 `
  --candidate-dir results\jepa_v3_p5_chunk3v2_seed$seed\zero20 `
  --output results\jepa_v3_p5_chunk3v2_seed$seed\zero20\zero_perturbation_comparison.json
```

### 9.3 通过门

- scenes 文件 byte-identical；
- 20/20 episode identity 完全配对；
- 非 JEPA 字段差异数为 `0`；
- CBF correction、collision、boundary、capture、path、time 等结果逐字段一致；
- candidate/ledger provenance 完整。

如果仍出现差异：先记录差异字段和第一处传播位置，修复 runtime 后建立新的 `v3` 版本并从 P5-A 重新训练；不得放宽浮点比较、跳过该门或继续 smoke。

---

## 10. P5-E/F：non-zero smoke 与 seed-11 diagnostic

### 10.1 20 episode smoke

zero-regression 通过后，使用 seed `20260911` 的 v2 checkpoint/ledger，固定 `K=5`、`0.10 m/s`、chunk `3`、CBF enabled：

```powershell
$seed = 20260911
& $py scripts\evaluate_random_central_mixed_obstacles.py `
  --method f2 `
  --checkpoint models\v5_development_exact_reactive_seed661606.pt `
  --environment-config configs\capture_radius_pursuit_central_v4_flee.yaml `
  --protocol configs\central_random_mixed_obstacle_s3_v5_protocol.yaml `
  --split validation --episodes 20 --use-cbf --device cuda `
  --action-conditioned-jepa-checkpoint results\jepa_v3_multitask_chunk3v2_seed$seed\checkpoint.pt `
  --jepa-reliability-ledger results\jepa_v3_multitask_chunk3v2_seed$seed\reliability_ledger.json `
  --jepa-candidate-count 5 --jepa-perturbation-mps 0.10 --jepa-action-chunk-length 3 `
  --reference-scenes results\jepa_v3_p3_zero_baseline20\scenes.jsonl `
  --reference-episodes results\jepa_v3_p3_zero_baseline20\episodes.csv `
  --output-dir results\jepa_v3_p5_chunk3v2_seed$seed\smoke20
```

必须记录：safe capture、collision、boundary、timeout、transit、paired improved/degraded/tied、capture time、总路径、最小 obstacle/inter-agent clearance、CBF correction/intervention、selected candidate、action-change rate、ledger credit、fallback 比例、控制/评分时延。

立即拒绝 P5 的条件：

- 新增或无法解释的 collision/boundary；
- scene/episode 未配对；
- 非有限输出或 CBF 未最后执行；
- path、capture time、CBF correction 或 clearance 的变化需完整记录，但不单独触发停止；
- provenance、ledger hash binding 或结果文件不完整。

不要求单 seed capture 必须提升；允许“无提升但安全/成本可接受”，此时仍可进入 P6 获取三 seed 证据。

### 10.2 60 episode diagnostic

smoke 通过后，以完全相同的 v2 seed-11 checkpoint、ledger、参数和冻结场景运行 60 episodes。该 run 只做诊断和预先定义的结果记录，不调参。

建议输出：

```text
results/jepa_v3_p5_chunk3v2_seed20260911/dev60/
  episodes.csv
  scenes.jsonl
  summary.json
  evaluation_metadata.json
  protocol.yaml
  paired_episode_deltas.csv
  failure_root_causes.json
```

失败按 collision、boundary、timeout、transit、nominal/global fallback、CBF correction 过大、候选排序反向、可见性误判、净空误判、控制时延和其他明确类别分桶。

P5 报告必须给出：

- 训练/数据/代码/checkpoint/ledger 哈希；
- offline prediction 与 action-following 结果；
- zero-regression 结果；
- smoke 和 60 episode 配对统计；
- 失败片段和成本构成；
- `admitted_to_p6`、`prediction_signal_no_single_seed_control_gain`、`rejected_for_control` 或 `insufficient_single_seed_evidence` 之一。

---

## 11. P6：三 seed 最终 paired development

### 11.1 评估设计

对 `20260911/20260912/20260913` 三个训练 seed，分别使用自己的 v2 checkpoint 和 hash-bound ledger，在相同 60 个冻结 development scenes 上与同一个 V5+CBF baseline 逐 episode 配对：

```text
P6 smoke：每个 seed 20 episodes，只用于接口/安全停止
P6 final：每个 seed 60 episodes，用于最终 development 证据
```

任何 seed 的 smoke 触发安全停止规则，整个候选标记为不稳定；不能只报告剩余 seed 的正向结果。

### 11.2 推荐目录

```text
results/jepa_v3_p6_chunk3v2_seed20260911_smoke20/
results/jepa_v3_p6_chunk3v2_seed20260911_dev60/
results/jepa_v3_p6_chunk3v2_seed20260912_smoke20/
results/jepa_v3_p6_chunk3v2_seed20260912_dev60/
results/jepa_v3_p6_chunk3v2_seed20260913_smoke20/
results/jepa_v3_p6_chunk3v2_seed20260913_dev60/
```

每个 candidate 必须引用：

- `results/jepa_v2_control_baseline60/scenes.jsonl` 和 `episodes.csv` 的冻结 reference；
- `configs/capture_radius_pursuit_central_v4_flee.yaml`；
- `configs/central_random_mixed_obstacle_s3_v5_protocol.yaml`；
- 对应 seed 的 checkpoint、ledger、prediction gate、action audit。

命令结构与 P5 相同，只改变 `--episodes`、`--output-dir` 和 seed 对应的 checkpoint/ledger；始终使用 `--split validation`，不得使用 `locked_test`。

### 11.3 必报指标

每个 seed 和三 seed 汇总均报告：

1. **任务与安全：** safe capture、collision、boundary、timeout、transit；
2. **配对结果：** improved/degraded/tied、capture delta、逐 episode outcome；
3. **效率：** capture time、defender path、最小 obstacle/inter-agent clearance、CBF correction/intervention、action-change rate、chunk smoothness、候选评分延迟和总控制延迟；
4. **可靠性：** mean ledger credit、local nominal fallback、global/OOD fallback；
5. **困难分桶：** visibility、layout、obstacle count、target motion、minimum clearance、CBF correction；
6. **统计：** 三 seed mean ± standard deviation、paired block bootstrap 95% CI；二元 capture 同时给出 McNemar/exact paired test，并说明小样本限制；
7. **机制链：** prediction gate → action-following → ledger → candidate ranking → CBF → episode outcome 的可追溯对应。

bootstrap 的独立单位为完整 `(training seed, episode)` 或预先定义的 episode block，不能把同一 episode 中的时间步或五个候选当作独立训练重复。

### 11.4 P6 预定义分类

| 分类 | 判定要求 | 允许表述 |
|---|---|---|
| `promising_development_candidate` | 三 seed 无新且未解释安全退化；平均 paired capture delta 为正；至少 2/3 seed 非负；收益非单 seed 驱动；成本完整且可接受 | 仅称 development 正向候选，进入 P7 |
| `useful_safety_fallback_only` | 捕获不稳定，但 ledger fallback 能解释并避免风险，且无安全损害 | 只能称安全/诊断机制证据 |
| `prediction_improvement_no_control_gain` | 离线预测或排名改善，但闭环 capture/效率无净收益 | 明确记录“预测改善未转化为控制收益” |
| `rejected_for_instability` | 任一 seed 有未解释 collision/boundary、zero-regression 破坏或严重成本退化 | 停止该变体 |
| `insufficient_evidence_do_not_open_locked_test` | seed 矛盾、CI 过宽、数据或 provenance 不完整 | 只报告不确定性 |

---

## 12. P7：可复现性和 locked-readiness 审计

P7 不运行 locked benchmark，只审计是否值得另写新的 preregistration 草案。

### 12.1 审计清单

- [ ] 两个 v2 archive、metadata、manifest、protocol 和源代码均有 SHA-256；
- [ ] 三个 v2 checkpoint、三个 ledger 和所有评估目录均 hash-bound；
- [ ] 每个 TensorBoard run 有完整 scalar、text、histogram 和 40 epoch 记录；
- [ ] prediction gate、action-following、dataset contract、chunk contract、zero-regression、CBF order 和 pairing 测试通过；
- [ ] 3×60 candidate episodes、60 baseline episodes、CSV/JSON/Markdown 报告完整可读；
- [ ] 所有失败、fallback、CBF correction、时延和负结果均披露；
- [ ] `locked_test_opened=false`；V4/V5 historical 文件、checkpoint、archive 和报告未被修改；
- [ ] 任何文字都没有把 development signal 写成 locked improvement，也没有把单 seed 写成三 seed 结论。

### 12.2 是否建议新的 preregistration

只有 P6 分类为 `promising_development_candidate`，且同时满足以下条件，才形成新的 preregistration 草案：

1. 三 seed 趋势一致且不是单个场景/seed 驱动；
2. CBF 下没有新的未解释 collision/boundary；
3. path/time、clearance、CBF 和控制时延代价已完整披露且不掩盖安全结果；
4. checkpoint、ledger、dataset、environment、scene、代码哈希完整；
5. 主指标、次指标、样本量、seed block、统计方法、停止规则和失败处理先写死；
6. 用户另行明确授权打开新的 locked block。

任一条件不满足时，正确结论是“不打开 locked test”。

---

## 13. 测试、TensorBoard 与 provenance 纪律

### 13.1 代码测试

每个阶段结束时至少运行：

```powershell
& $py -m pytest `
  tests/test_jepa_v3_counterfactual_dataset.py `
  tests/test_jepa_v3_multitask.py `
  tests/test_prediction.py `
  tests/test_jepa_v3_zero_perturbation.py -q
```

如果修改了 runtime、evaluation 或 protocol，必须增加相应测试并记录完整 pytest 输出。

### 13.2 每次运行的 provenance 最小字段

```text
run_id
git_commit
command
start_time / end_time / elapsed_seconds
python / PyTorch / CUDA / GPU
random_seed
protocol_path + SHA-256
environment_config + SHA-256
actor_checkpoint + SHA-256
train/validation dataset + metadata + SHA-256
model and optimizer configuration
checkpoint SHA-256
TensorBoard logdir and audit result
evaluation split and episode/scenes hashes
decision status
failure/interruption reason
```

provenance 在运行结束后立即写入，不在数天后凭记忆补写。

### 13.3 提交边界

results、checkpoint、TensorBoard event、NPZ 和 `tmp/` 默认不加入 Git。代码/测试/报告按阶段小提交：

```text
P5 implementation + tests + P5 report
P6 three-seed statistics + P6 report
P7 reproducibility audit + readiness decision
```

E1-prime、execution dynamics、历史 V4/V5 文档和用户已有未提交变更必须保留，不得混入 JEPA 提交。若需要提交/推送，先检查 `git diff --name-only` 和 `git status --short`，确认只包含本阶段文件。

---

## 14. 计划完成定义

本计划完成不要求 JEPA 一定胜出；必须完成的是可审计的证据链：

1. v2 archive 审计通过；
2. 三个 v2 seed 完成 40 epoch 训练、TensorBoard 审计、prediction gate、action-following 和 hash-bound ledger；
3. zero-perturbation 严格回归通过，或按预定义规则明确记录失败并停止；
4. 完成 seed-11 P5 smoke/diagnostic；
5. 完成三 seed 的 20 smoke 和 3×60 paired development，或按安全规则停止；
6. 输出 P5、P6、P7 报告，包含正向、负向、不确定证据和失败分桶；
7. 明确写出 `promising_development_candidate` 或“不打开 locked test”等最终分类；
8. 没有覆盖、修改或重新解释 V4/V5 historical locked evidence。

最终判断的优先级是：

```text
数据/代码合同
  > zero-regression
  > CBF 安全与配对完整性
  > 三 seed 闭环收益
  > 成本、可靠性和机制解释
  > 是否值得另写新的 locked preregistration
```

无论最后结论是正向、负向还是证据不足，这条流程都应能回答：JEPA 在何种互动、可见性、净空、目标运动和 CBF 工况下有效或失效，以及失败是预测误差、排序错误、ledger 回退还是安全过滤器代价造成的。
