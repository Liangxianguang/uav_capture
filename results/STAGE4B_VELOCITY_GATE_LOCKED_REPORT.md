# 阶段 4B 结果：目标 belief 更新规则基线比较

任务：部分可观测三维障碍环境下多无人机协同捕获半径追逃。

## 1. 协议

- 本实验只比较 estimator，不训练或执行学习策略；4 架防守机的动作固定为零。
- 每种模式在每个条件上使用完全相同的障碍、目标轨迹、检测/消息随机数和 episode seed。
- `legacy` 保持既有 P1 语义；`zero_velocity` 和 `constant_velocity` 是可解释基线；
  `time_aligned` 在测量时间戳处融合保存的本地 belief，再传播到当前时间。
- 本次 `time_aligned` 的陈旧速度衰减设置：系数 0.80，
  起始观测年龄 3 步。
- 目标真值只用于 rollout 后的误差标签，不是任何模式的输入。
- 条件：delayed_measurements, burst_occlusion；每个条件每种模式：100 个锁定 episode。

## 2. 已初始化 belief 的主要误差

| 条件 | 模式 | 位置误差 (m) | 速度误差 (m/s) | 新时间戳更新误差 (m) | 重获至更新 (steps) |
|---|---|---:|---:|---:|---:|
| delayed_measurements | legacy | 6.724 | 1.620 | 0.587 | 0.333 |
| delayed_measurements | zero_velocity | 1.894 | 1.239 | 0.667 | 0.333 |
| delayed_measurements | constant_velocity | 6.945 | 1.620 | 0.340 | 0.333 |
| delayed_measurements | time_aligned | 1.501 | 1.079 | 0.340 | 0.333 |
| burst_occlusion | legacy | 2.934 | 1.367 | 0.293 | 0.609 |
| burst_occlusion | zero_velocity | 1.267 | 2.061 | 0.437 | 0.609 |
| burst_occlusion | constant_velocity | 2.887 | 1.367 | 0.151 | 0.609 |
| burst_occlusion | time_aligned | 0.850 | 1.123 | 0.151 | 0.609 |

## 3. 观测年龄分桶

| 条件 | 模式 | Fresh (0-1) 误差 (m) | Moderate (2-4) 误差 (m) | Stale (>=5) 误差 (m) | 未初始化帧 |
|---|---|---:|---:|---:|---:|
| delayed_measurements | legacy | n/a | 0.541 | 7.896 | 12 |
| delayed_measurements | zero_velocity | n/a | 0.592 | 2.158 | 12 |
| delayed_measurements | constant_velocity | n/a | 0.248 | 8.238 | 12 |
| delayed_measurements | time_aligned | n/a | 0.247 | 1.771 | 12 |
| burst_occlusion | legacy | n/a | 0.318 | 4.438 | 8 |
| burst_occlusion | zero_velocity | n/a | 0.523 | 2.184 | 8 |
| burst_occlusion | constant_velocity | n/a | 0.196 | 4.497 | 8 |
| burst_occlusion | time_aligned | n/a | 0.194 | 1.521 | 8 |

## 4. 解释边界

本结果只验证 belief 更新规则在冻结仿真观测过程中的估计误差，不是 Safe Capture 改进声明。
Validation 在独立的 `641001` seed block 中预先选定衰减系数 0.80、起始年龄 3 步；
本报告使用新的 `644001` locked-test seed 复验。相对零速度基线，门控时间对齐在
延迟测量域将已初始化位置误差从 1.894 m 降至 1.501 m，在突发遮挡域从 1.267 m
降至 0.850 m；新时间戳更新误差也分别从 0.667 m 降至 0.340 m、从 0.437 m 降至
0.151 m。因此 estimator 层面已满足进入 F1 策略训练的前提，但仍不构成 Safe Capture
改进声明，后续必须进行新的多 seed 策略训练和独立 locked-test 评估。

## 5. 复现

```powershell
conda run --no-capture-output -n uav-encirclement-gpu python scripts/evaluate_stage4b_belief_baselines.py
```
