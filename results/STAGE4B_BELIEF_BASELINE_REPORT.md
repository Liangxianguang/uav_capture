# 阶段 4B 结果：目标 belief 更新规则基线比较

任务：部分可观测三维障碍环境下多无人机协同捕获半径追逃。

## 1. 协议

- 本实验只比较 estimator，不训练或执行学习策略；4 架防守机的动作固定为零。
- 每种模式在每个条件上使用完全相同的障碍、目标轨迹、检测/消息随机数和 episode seed。
- `legacy` 保持既有 P1 语义；`zero_velocity` 和 `constant_velocity` 是可解释基线；
  `time_aligned` 在测量时间戳处融合保存的本地 belief，再传播到当前时间。
- 目标真值只用于 rollout 后的误差标签，不是任何模式的输入。
- 条件：delayed_measurements, burst_occlusion；每个条件每种模式：100 个锁定 episode。

## 2. 已初始化 belief 的主要误差

| 条件 | 模式 | 位置误差 (m) | 速度误差 (m/s) | 新时间戳更新误差 (m) | 重获至更新 (steps) |
|---|---|---:|---:|---:|---:|
| delayed_measurements | legacy | 8.003 | 1.708 | 0.586 | 0.305 |
| delayed_measurements | zero_velocity | 1.956 | 1.238 | 0.667 | 0.305 |
| delayed_measurements | constant_velocity | 8.256 | 1.708 | 0.337 | 0.305 |
| delayed_measurements | time_aligned | 8.254 | 1.707 | 0.337 | 0.305 |
| burst_occlusion | legacy | 4.799 | 1.476 | 0.300 | 0.572 |
| burst_occlusion | zero_velocity | 1.596 | 2.044 | 0.447 | 0.572 |
| burst_occlusion | constant_velocity | 4.769 | 1.476 | 0.152 | 0.572 |
| burst_occlusion | time_aligned | 4.768 | 1.476 | 0.152 | 0.572 |

## 3. 观测年龄分桶

| 条件 | 模式 | Fresh (0-1) 误差 (m) | Moderate (2-4) 误差 (m) | Stale (>=5) 误差 (m) | 未初始化帧 |
|---|---|---:|---:|---:|---:|
| delayed_measurements | legacy | n/a | 0.539 | 9.363 | 12 |
| delayed_measurements | zero_velocity | n/a | 0.590 | 2.225 | 12 |
| delayed_measurements | constant_velocity | n/a | 0.247 | 9.741 | 12 |
| delayed_measurements | time_aligned | n/a | 0.247 | 9.738 | 12 |
| burst_occlusion | legacy | n/a | 0.324 | 7.119 | 8 |
| burst_occlusion | zero_velocity | n/a | 0.536 | 2.625 | 8 |
| burst_occlusion | constant_velocity | n/a | 0.194 | 7.205 | 8 |
| burst_occlusion | time_aligned | n/a | 0.194 | 7.204 | 8 |

## 4. 解释边界

本结果只验证 belief 更新规则在冻结仿真观测过程中的估计误差，不是 Safe Capture 改进声明。
`time_aligned` 在新时间戳包到达时降低了误差：延迟测量域从 0.586 m 降至 0.337 m，
突发遮挡域从 0.300 m 降至 0.152 m；但长陈旧观测下的常速度传播会累积误差，
其总体已初始化 belief 误差并不优于 `zero_velocity`。因此当前证据不足以启动 F1/F2
策略训练；下一步只能在独立 validation seed 上选择年龄门控速度衰减，再以新的 locked-test
seed 复验估计器收益。

## 5. 复现

```powershell
conda run --no-capture-output -n uav-encirclement-gpu python scripts/evaluate_stage4b_belief_baselines.py
```
