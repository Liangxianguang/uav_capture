# P0 障碍物条件候选路线覆盖报告

**日期：** 2026-09-07
**状态：** `development-only`；`locked_test_opened=false`
**目的：** 在进入 hard-negative archive 和 JEPA 重训前，验证“公开障碍物几何 -> 不同路线提案 -> 可达且经 CBF 验证的选项”这一执行前置合同。

## 结论

四个固定 WP1 微场景全部通过路线覆盖 gate。路线生成器能够根据当前公开障碍物几何改变左右绕行路线，并拒绝被墙体阻塞的一侧；每个场景至少保留一条 reachable route 和一条独立 Joint CBF verified route。4097 点高密度走廊复核没有发现 geometry false accept。

这是一项候选生成和几何/CBF 探针审计，不是 safe-capture 提升证据。审计没有训练 JEPA、执行控制动作、修改 CBF margin，也没有打开 locked split；在进入下一阶段前仍需完成 earliest CBF infeasibility / stopping-distance 诊断和 hard-negative counterfactual archive。

## 场景结果

| 场景 | 候选数 | 有效路线 | 高密度有效 | CBF verified | 左侧有效 | 右侧有效 | 几何 false accept |
| --- | ---: | ---: | ---: | ---: | :---: | :---: | ---: |
| `central_single` | 12 | 8 | 8 | 8 | 是 | 是 | 0 |
| `left_blocked` | 12 | 7 | 7 | 7 | 否 | 是 | 0 |
| `right_blocked` | 12 | 7 | 7 | 7 | 是 | 否 | 0 |
| `wall_single_gap` | 12 | 10 | 10 | 10 | 是 | 是 | 0 |

## Acceptance checks

- [x] `central_single` 的左右路线几何不同，且两侧均有效。
- [x] `left_blocked` 拒绝左侧阻塞路线并保留右侧路线。
- [x] `right_blocked` 拒绝右侧阻塞路线并保留左侧路线。
- [x] `wall_single_gap` 的 nominal route 有效。
- [x] 每个场景至少有一条 reachable route。
- [x] 每个场景至少有一条 Joint CBF verified route。
- [x] 高密度走廊检查没有 geometry false accept。

**总体：** `PASS`。

## 复现实验

```powershell
$env:PYTHONPATH='src'
D:\download\anaconda3\envs\traj_pred_prep\python.exe `
  scripts/audit_jepa_safe_capture_wp1_obstacle_routes.py `
  --output-dir results/jepa_safe_capture_p0_route_coverage_20260907 `
  --tensorboard-dir results/jepa_safe_capture_p0_route_coverage_20260907_tensorboard `
  --with-cbf `
  --corridor-samples 65 `
  --verification-corridor-samples 4097
```

结果和原始候选明细：

- `results/jepa_safe_capture_p0_route_coverage_20260907/summary.json`
- `results/jepa_safe_capture_p0_route_coverage_20260907/route_candidates.json`
- `results/jepa_safe_capture_p0_route_coverage_20260907/report.md`

TensorBoard event：

- `results/jepa_safe_capture_p0_route_coverage_20260907_tensorboard/`

该审计使用 `SummaryWriter` 记录场景级候选数量、几何有效性、CBF verified 数量和 acceptance 状态；可用以下命令查看：

```powershell
D:\download\anaconda3\envs\traj_pred_prep\Scripts\tensorboard.exe `
  --logdir results/jepa_safe_capture_p0_route_coverage_20260907_tensorboard
```

## Provenance

- Git revision（运行审计时）：`1d7fd808dd427932539aaa66f07159b0c5db8de4`
- Python：`3.11.14`
- NumPy：`2.3.5`
- Corridor samples：`65`
- Verification corridor samples：`4097`
- Route module SHA-256：`293a717cd17ae2b022140728b76d54b3857f3de5f3e90dc295f092f9a7cd5ca2`
- Audit script SHA-256：`fd1311bcd70b48c6582eb1439340f644ee6361e7c3a59c9e4fcfc452d6478493`
- Environment config SHA-256：`42bd4e158c5e314e0ece6add8038b32c384a7a2ca027e9387327656fccf751ad`

## 下一步 gate

该 gate 通过后，只允许进入以下诊断顺序：

1. 在固定微场景上定位 earliest CBF infeasibility、stopping distance、pairwise 和 acceleration active constraint；
2. 建立包含可行/不可行路线的 hard-negative counterfactual archive；
3. 重新设计 calibration 和 reliability ledger 后，才可开始新的 JEPA 训练。

如果 earliest-infeasibility 诊断显示候选覆盖没有转化为可执行捕获机会，应停止扩大场景或训练规模，并记录 `prediction_signal_no_control_gain`，不得通过降低安全约束制造提升。
