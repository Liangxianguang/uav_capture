# V21 S3 divergence 与通信 age 审计报告

**日期：** 2026-09-05
**阶段：** S3-A / S3-D，development-only
**主指标：** episode-level `safe_capture`
**Locked test：** `locked_test_opened=false`
**审计脚本：** `scripts/audit_jepa_safe_capture_v21_s3_divergence.py`

本报告只读取冻结的 V21 failure index 和原始 step traces，不重新模拟环境、不读取 online target ground truth、不修改 CBF margin/OOD/stale 阈值，也不改变 controlled-abort 语义。

## 1. 输入与产物

输入：

- failure index：`results/jepa_safe_capture_v21_failure_index_v4/failure_index.json`
- source runs：V21 `m0/m3/a1/a2`，三个 seed、每个变体 20 episodes
- protocol SHA-256：`278623ceb7185a6c3ce23246e8a28693f025a2977fad95059ae5b0df9a03b014`

产物：

- `results/jepa_safe_capture_v21_s3_divergence_audit_v1/s3_divergence_audit.json`
- `results/jepa_safe_capture_v21_s3_divergence_audit_v1/s3_divergence_cycles.csv`
- `results/jepa_safe_capture_v21_s3_divergence_audit_v1/s3_age_semantics.csv`
- `results/jepa_safe_capture_v21_s3_divergence_audit_v1/s3_divergence_audit.md`
- `results/jepa_safe_capture_v21_s3_divergence_audit_v1/provenance.json`
- TensorBoard：`results/jepa_safe_capture_v21_current_tensorboard/s3_divergence_audit_v1/`

## 2. 最早 divergence 结果

failure index 中的 116 个 `cbf_controlled_abort` episode 全部在原始 trace 中找到唯一的最早 CBF failure row，覆盖率为 `116/116`。

| 统计项 | 结果 |
|---|---:|
| CBF abort episodes | 116 |
| earliest-divergence rows | 116 |
| 首个 failure 含负 slack | 116/116 (100.0%) |
| requested action 与 reachable nominal 相同 | 93/116 (80.2%) |
| 首次 failure 时所有 candidate 均 ineligible | 40/116 (34.5%) |
| raw unverified executed steps | 0 |

首个负 slack 的约束类别：

| 类别 | episodes |
|---|---:|
| obstacle | 48 |
| pairwise | 42 |
| boundary/altitude | 26 |

当前证据支持：主要失败是控制周期内出现了已记录的 CBF 约束不可行状态，而不是碰撞或越界已经发生。它还不能证明“所有替代动作均不可行”，因为现有 trace 没有保存 candidate、nominal、safe-hold 三条路径的独立 QP counterfactual 结果。

## 3. 通信 age 语义结果

审计覆盖全部 240 个 V21 episode，而不是只覆盖失败样例：

| 观测量 | 结果 |
|---|---:|
| `message_age_saturated`（上限 60） | 240/240 (100.0%) |
| `target_observation_stale`（大于 45） | 0/240 (0.0%) |
| 饱和时至少有一个 target visible | 120/240 (50.0%) |

因此 `message_age_steps=60` 是当前通信/队列字段的饱和状态，不能直接解释为目标观测 stale，也不能直接作为目标预测漂移标签。后续必须修复或明确 age state machine 的初始化、无通信和可见性语义，再重新生成 provenance 和 calibration。

## 4. 排序与 anchor 复核

对三个 seed 的冻结 settled decision rows 做了只读 abstention counterfactual recheck，共 `9,984` 条 decision，其中 `5,637` 条为 multi-eligible：

| 指标 | 结果 |
|---|---:|
| 记录策略 selected-not-best | 4,087/5,637 (72.5%) |
| 仅取 eligible score argmin 的 selected-not-best | 2,561/5,637 (45.4%) |
| 记录策略与 score argmin 一致 | 51.7% |
| score argmin 与 settled-best 一致 | 54.6% |

这是 offline upper-bound 诊断，不是可部署策略结果，也不绕过 CBF。它说明 nominal anchor、abstention 和 candidate eligibility 对最终 selected-not-best 有明显影响；在修复排序合同前，不能把问题简单归因于 JEPA latent 预测误差。

## 5. S3 决策

- **安全合同：** 保持通过；collision、boundary、pairwise 和 raw-unverified 仍为零。
- **排序状态：** 仍为 `ranking_unresolved`；不能把 offline settled outcome 接入在线控制。
- **当前结论：** S3-A（最早 divergence）、S3-C 的 offline anchor/abstention 分离和 S3-D（age 语义分离）已完成第一轮证据；S3-B（候选/nominal/safe-hold 独立可行性）和 S3-C 的在线合同回归尚未完成。
- **禁止动作：** 不降低 CBF margin，不放宽 stale/OOD，不删除 controlled abort，不扩大 40/60 集，不打开 locked test，不训练新 checkpoint。

## 6. 下一步

1. 新增/运行 candidate、reachable nominal、safe-hold 三路独立 Joint CBF-QP counterfactual auditor，逐 abort 判断是全路径不可行还是选中路径不可行。
2. 对首个负 slack 前后两个 cycle 做固定输入回归，覆盖 obstacle、pairwise、boundary、action scale、horizon 和 solver 初值；输出 active set、slack、校正量和 fallback 路由。
3. 完成 candidate eligibility、nominal anchor、score direction 和 communication age state-machine 修复后，创建新 protocol/hash；旧 V21 结果只读保留。
4. 只有新 protocol 的三 seed x 20 paired smoke 满足安全硬门、aggregate `safe_capture` 不低于 M0 且至少 `2/3` seed 非负，才考虑 40/60 集 development。

## 7. 可复现命令

```powershell
$py = 'D:\download\anaconda3\envs\traj_pred_prep\python.exe'
$env:PYTHONPATH = "$PWD\src;$PWD\scripts"
& $py scripts/run_with_tensorboard_compat.py `
  scripts/audit_jepa_safe_capture_v21_s3_divergence.py `
  --input-root results `
  --failure-index results/jepa_safe_capture_v21_failure_index_v4/failure_index.json `
  --output-dir results/jepa_safe_capture_v21_s3_divergence_audit_v1 `
  --tensorboard-dir results/jepa_safe_capture_v21_current_tensorboard/s3_divergence_audit_v1 `
  --development-only
```
