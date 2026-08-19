# 阶段 3B 正式结果：预测增强非循环 MAPPO 的多种子锁定评估

日期：2026-08-19  
任务：部分可观测障碍环境下多无人机三维捕获半径追逃  
后端：kinematic 3D simulation  
统计单位：独立训练 seed；不是把同一 locked-test block 的回合当作独立训练重复。

## 1. 固定协议

- 训练 seed：521001, 521002, 521003；每种子 MAPPO 训练步数：65,536。
- locked-test seed：632001；每个场景 100 回合，
  `open`、`clutter`、`occluded` 共 300 回合/训练 seed。
- 方法：无预测（44 维）、constant-velocity prediction（48 维）、冻结 GRU prediction（52 维）。
- 每个方法分别评估 raw action 和 local CBF action；每个汇总行均为 3 个训练 seed、900 回合。
- 95% 置信区间采用训练 seed 间 Student-t 区间（n=3，df=2）；同一锁定回合在不同 seed 下仅用于比较策略，
  不被错误计为独立训练重复。

## 2. 总体结果

| 方法 / 执行 | Safe Capture | Capture | Collision | Boundary Violation | Capture Time (s) | Minimum Clearance (m) |
|---|---:|---:|---:|---:|---:|---:|
| MAPPO，无预测，raw | 93.56% +/- 5.51% | 96.78% +/- 3.35% | 6.44% +/- 5.51% | 0.33% +/- 0.83% | 1.350 +/- 0.094 | 0.505 +/- 0.034 |
| MAPPO，无预测，+CBF | 100.00% +/- 0.00% | 100.00% +/- 0.00% | 0.00% +/- 0.00% | 0.00% +/- 0.00% | 1.406 +/- 0.069 | 0.805 +/- 0.021 |
| MAPPO，常速度预测，raw | 92.44% +/- 4.56% | 96.00% +/- 2.19% | 7.56% +/- 4.56% | 0.11% +/- 0.48% | 1.367 +/- 0.005 | 0.469 +/- 0.045 |
| MAPPO，常速度预测，+CBF | 100.00% +/- 0.00% | 100.00% +/- 0.00% | 0.00% +/- 0.00% | 0.00% +/- 0.00% | 1.430 +/- 0.025 | 0.792 +/- 0.009 |
| MAPPO，GRU 预测，raw | 94.89% +/- 4.25% | 97.67% +/- 1.66% | 5.11% +/- 4.25% | 0.22% +/- 0.48% | 1.305 +/- 0.045 | 0.545 +/- 0.061 |
| MAPPO，GRU 预测，+CBF | 100.00% +/- 0.00% | 100.00% +/- 0.00% | 0.00% +/- 0.00% | 0.00% +/- 0.00% | 1.370 +/- 0.024 | 0.827 +/- 0.029 |

## 3. 结论

- raw action 下，GRU 相对无预测的 Safe Capture 差值为 1.33 +/- 2.48 个百分点；
  3 个训练 seed 中有 3 个为正。
- CBF 下三种方法在当前 900 回合/方法锁定测试中均达到 100% Safe Capture 和 0% Collision；
  因此不能把该安全收益归因于预测模块，CBF 必须作为独立组件报告。
- 由于 GRU raw-action 改善未达到计划中预设的 5 个百分点参考门槛，本阶段结果支持“接口可用且存在
  小幅 raw-action 改善”，但不支持“GRU 已稳定显著提高最终 Safe Capture”的强结论。
- 下一步进入阶段 3C：实现 Recurrent-MAPPO，并在更强遮挡/延迟目标域复查预测与记忆的独立贡献。

## 4. 可复现证据

- 运行器：`scripts/run_stage3b_formal.py`
- 聚合脚本：`scripts/aggregate_stage3b_formal.py`
- 完整结构化统计：`results/PREDICTION_POLICY_STAGE3_FORMAL_SUMMARY.json`
- 所有按种子保存的 checkpoint、TensorBoard、逐回合 CSV、配置、协议和三维轨迹：`results/stage3b_formal/`（本地生成，未入 Git）。

结论仅适用于当前运动学三维仿真中的捕获半径追逃，不等同于实体接触、网捕、SITL、真实视觉闭环或实飞捕获。
