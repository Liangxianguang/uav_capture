# 阶段 3C-P1 结果：部分可观测压力测试

任务：部分可观测障碍环境下多无人机三维捕获半径追逃
后端：kinematic 3D simulation
统计单位：独立训练 seed；压力测试只改变已实现的仿真观测、通信、目标运动和障碍参数。

## 1. 协议

- 训练 seed：521001, 521002, 521003；locked-test seed：642001。
- 每个条件每个 seed：100 回合；每个方法/执行方式/条件共 300 回合。
- 方法：D Recurrent-MAPPO（无学习式预测）；E Recurrent-MAPPO + 冻结 GRU 预测。
- raw 与 +CBF 分开报告；CI 在训练 seed 间计算，不把 episode 计作独立训练重复。

## 2. raw action 总体结果

| 条件 | D Safe Capture | E Safe Capture | D Collision | E Collision | D Obs Age | E Obs Age | E-D Safe Capture (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| nominal_partial_observation | 91.33% +/- 6.25% | 91.00% +/- 6.57% | 8.67% +/- 6.25% | 9.00% +/- 6.57% | 0.172 +/- 0.005 | 0.172 +/- 0.003 | -0.33 +/- 9.40 |
| delayed_measurements | 63.33% +/- 11.47% | 54.67% +/- 5.17% | 36.67% +/- 11.47% | 45.33% +/- 5.17% | 3.075 +/- 0.008 | 3.073 +/- 0.012 | -8.67 +/- 8.72 |
| burst_occlusion | 61.67% +/- 15.97% | 61.67% +/- 16.16% | 38.33% +/- 15.97% | 38.33% +/- 16.16% | 2.293 +/- 0.036 | 2.289 +/- 0.015 | 0.00 +/- 6.57 |
| communication_loss | 81.67% +/- 1.43% | 86.33% +/- 14.13% | 18.33% +/- 1.43% | 13.67% +/- 14.13% | 0.190 +/- 0.004 | 0.192 +/- 0.005 | 4.67 +/- 14.56 |

## 3. CBF action 结果

| 条件 | D Safe Capture | E Safe Capture | D Collision | E Collision |
|---|---:|---:|---:|---:|
| nominal_partial_observation | 100.00% +/- 0.00% | 100.00% +/- 0.00% | 0.00% +/- 0.00% | 0.00% +/- 0.00% |
| delayed_measurements | 84.00% +/- 16.29% | 75.67% +/- 7.59% | 12.33% +/- 9.40% | 16.67% +/- 10.34% |
| burst_occlusion | 69.33% +/- 11.74% | 67.00% +/- 17.91% | 27.67% +/- 12.50% | 30.00% +/- 11.38% |
| communication_loss | 99.33% +/- 1.43% | 99.67% +/- 1.43% | 0.67% +/- 1.43% | 0.33% +/- 1.43% |

## 4. 解释边界

- `nominal_partial_observation`：GRU 相对 D 的 raw Safe Capture 差值为 -0.33 +/- 9.40 个百分点，1/3 个 seed 为正。
- `delayed_measurements`：GRU 相对 D 的 raw Safe Capture 差值为 -8.67 +/- 8.72 个百分点，0/3 个 seed 为正。
- `burst_occlusion`：GRU 相对 D 的 raw Safe Capture 差值为 0.00 +/- 6.57 个百分点，1/3 个 seed 为正。
- `communication_loss`：GRU 相对 D 的 raw Safe Capture 差值为 4.67 +/- 14.56 个百分点，2/3 个 seed 为正。

这些压力测试用于刻画收益和失败边界，不构成真实传感器数据验证。若 CI 跨 0，结论只能写成方向性趋势；
若 CBF 结果饱和，则安全收益仍归因于独立安全层，不能归因于预测或循环记忆。

## 5. 复现证据

- 运行器：`scripts/run_stage3c_p1_stress.py`
- 聚合器：`scripts/aggregate_stage3c_p1_stress.py`
- 结构化结果：`results/RECURRENT_POLICY_STAGE3C_P1_STRESS_SUMMARY.json`
- 本地逐回合结果、配置和 checkpoint 元数据：`results/stage3c_p1_stress/`。

结论仅适用于当前运动学三维仿真，不等同于实体捕获、真实感知闭环、SITL 或实飞验证。
