# 阶段 3C 正式结果：Recurrent-MAPPO 多种子锁定评估

日期：2026-08-19  
任务：部分可观测障碍环境下多无人机三维捕获半径追逃  
后端：kinematic 3D simulation  
统计单位：独立训练 seed；不将同一 locked-test block 的逐回合结果错误视作独立训练重复。

## 1. 固定协议

- 训练 seed：521001, 521002, 521003；每种子循环 MAPPO 训练 65,536 环境步。
- locked-test seed：632001；`open`、`clutter`、`occluded` 各 100 回合，
  共 300 回合/训练 seed，900 回合/方法/执行方式。
- D：无学习式预测的 Recurrent-MAPPO；E：Recurrent-MAPPO + 冻结 GRU 预测特征。
- 两个循环 actor 均使用 MLP behavior-cloning prior + 零初始化 GRU residual；
  因此循环模块起点等价于其对应无记忆 prior，随后只学习历史带来的残差修正。
- 每种方法分别评估 raw action 与 local CBF action；95% CI 为训练 seed 间 Student-t 区间（n=3，df=2）。

## 2. 总体结果

| 方法 / 执行 | Safe Capture | Capture | Collision | Boundary Violation | Time-to-Capture (s) | Minimum Clearance (m) |
|---|---:|---:|---:|---:|---:|---:|
| Recurrent-MAPPO，无学习式预测，raw | 93.67% +/- 5.04% | 96.22% +/- 1.26% | 6.33% +/- 5.04% | 0.56% +/- 0.48% | 1.338 +/- 0.070 | 0.532 +/- 0.147 |
| Recurrent-MAPPO，无学习式预测，+CBF | 100.00% +/- 0.00% | 100.00% +/- 0.00% | 0.00% +/- 0.00% | 0.00% +/- 0.00% | 1.409 +/- 0.066 | 0.807 +/- 0.081 |
| Recurrent-MAPPO，GRU 预测，raw | 94.33% +/- 4.14% | 96.67% +/- 1.66% | 5.67% +/- 4.14% | 0.11% +/- 0.48% | 1.319 +/- 0.023 | 0.543 +/- 0.026 |
| Recurrent-MAPPO，GRU 预测，+CBF | 100.00% +/- 0.00% | 100.00% +/- 0.00% | 0.00% +/- 0.00% | 0.00% +/- 0.00% | 1.384 +/- 0.030 | 0.820 +/- 0.020 |

## 3. 场景分桶：raw action

| 场景 | D raw Safe Capture | E raw Safe Capture | D raw Collision | E raw Collision |
|---|---:|---:|---:|---:|
| open | 93.33% +/- 6.25% | 93.67% +/- 12.25% | 6.67% +/- 6.25% | 6.33% +/- 12.25% |
| clutter | 94.33% +/- 3.79% | 93.33% +/- 2.87% | 5.67% +/- 3.79% | 6.67% +/- 2.87% |
| occluded | 93.33% +/- 5.74% | 96.00% +/- 4.97% | 6.67% +/- 5.74% | 4.00% +/- 4.97% |

## 4. 配对比较与结论边界

- E 相对 D 的 raw-action Safe Capture 配对差值为 0.67 +/- 6.25 个百分点；
  2/3 个训练 seed 为正。
- 该置信区间描述独立训练 seed 间的不确定性。若区间跨越 0，报告为方向性趋势而非稳定显著提升。
- raw 与 +CBF 的差异用于分离循环策略本身和安全过滤器的贡献。若 +CBF 饱和为 100% Safe Capture，
  不应将该增益归因于预测或循环记忆。

## 5. 与阶段 3B 非循环策略的上下文比较

| 设置 | raw Safe Capture |
|---|---:|
| Stage 3B MAPPO，无预测 | 93.56% +/- 5.51% |
| Stage 3B MAPPO，GRU 预测 | 94.89% +/- 4.25% |
| Stage 3C D，循环无预测 | 93.67% +/- 5.04% |
| Stage 3C E，循环 + GRU 预测 | 94.33% +/- 4.14% |

阶段 3B 与 3C 都遵循相同 locked-test 协议，但不是对同一训练 seed 的配对随机试验；
该表仅提供跨架构的描述性上下文，不用于声称循环结构相对非循环结构的因果增益。

## 6. 可复现证据

- 运行器：`scripts/run_stage3c_formal.py`
- 聚合脚本：`scripts/aggregate_stage3c_formal.py`
- 本报告的结构化统计：`results/RECURRENT_POLICY_STAGE3C_FORMAL_SUMMARY.json`
- 本地完整证据（checkpoint、TensorBoard、逐回合 CSV、配置、协议和轨迹）：`results/stage3c_formal/`。

结论只适用于当前运动学三维仿真中的捕获半径任务，不等同于实体接触、网捕、SITL、真实视觉闭环或实飞捕获。
