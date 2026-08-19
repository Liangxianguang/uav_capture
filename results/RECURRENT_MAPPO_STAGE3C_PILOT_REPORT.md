# 阶段 3C 结果报告：Recurrent-MAPPO 实现验收 pilot

日期：2026-08-19  
状态：实现验收，单训练 seed，不是最终多种子方法结论

## 实现内容

- 参数共享、每架防守无人机独立 hidden state 的 GRU actor；
- centralized critic 仅用于训练；
- rollout 保存 hidden state 与 reset mask，PPO 按 32 步有序序列更新；
- 冻结 GRU 目标预测器继续通过 52 维局部 belief-state 输入 actor；
- actor 使用 MLP behavior-cloning prior 加零初始化 GRU residual，保证循环
  模块开始时等价于可验证的无记忆策略，而不是从零退化策略开始；
- 通用评估器能从 checkpoint 的 `actor_recurrent=true` 自动恢复循环执行。

## Pilot 协议

- 训练 seed：521001；训练步数：16,384；sequence length：32。
- 任务：4 架防守无人机、1 个目标、20 m × 20 m × 10 m、捕获半径 0.80 m。
- 测试：locked seed 632001，`open`、`clutter`、`occluded` 各 30 回合，
  共 90 回合。

## Pilot 结果

| 执行方式 | Safe Capture | Capture | Collision | Boundary Violation | Mean Capture Time |
|---|---:|---:|---:|---:|---:|
| raw action | 90.00% | 93.33% | 10.00% | 1.11% | 1.373 s |
| local CBF | 100.00% | 100.00% | 0.00% | 0.00% | 1.413 s |

证据目录：

- `results/recurrent_mappo_gru_prediction_residual_pilot_seed521001`
- `results/recurrent_mappo_gru_prediction_residual_pilot_raw_locked30`
- `results/recurrent_mappo_gru_prediction_residual_pilot_cbf_locked30`

## 结论

本轮证明循环 actor 的训练、序列 replay、hidden reset、预测器输入、checkpoint
恢复、raw/CBF 评估和 TensorBoard 记录均能端到端运行。Raw-action 指标尚低于
阶段 3B 的 GRU-MAPPO 正式多种子均值，且本轮只有一个短训练 seed；因此不应
声称循环记忆已经改善围捕能力。下一步是扩展训练预算，并按 Stage 3B 相同的
三 seed、每 seed 300 locked-test 回合协议比较“无学习式预测”和“GRU 预测”。

本结果只适用于运动学三维仿真捕获半径任务，不等同于实体接触、网捕、SITL、
真实视觉闭环或实飞捕获。
