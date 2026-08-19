# 阶段 3B 结果报告：预测器接入非循环 MAPPO 的策略级 pilot

日期：2026-08-19  
任务：部分可观测障碍环境下多无人机三维捕获半径追逃  
后端：kinematic 3D simulation  
状态：单训练种子 pilot，不是最终论文统计结论

## 1. 实验目的

本阶段只回答一个问题：将冻结的 GRU 局部历史轨迹预测器接入 actor 的
belief-state 后，是否能够在策略闭环中改善捕获表现。为区分预测特征和
安全层的贡献，使用同一训练 seed、训练预算、环境难度和 locked-test seed
block，分别测试 raw action 与 local CBF action。

本阶段没有实现 Recurrent-MAPPO；actor 仍是 MLP。GRU 只作为冻结的在线
预测特征适配器，因此不能把本报告称为“记忆型策略”最终结果。

## 2. 固定协议

- 防守无人机：4 架；逃逸目标：1 个。
- 世界：20 m × 20 m × 10 m；捕获半径：0.80 m。
- 场景：`open`、`clutter`（3 障碍物）、`occluded`（5 障碍物）。
- 训练 seed：521001；MAPPO 训练步数：65,536。
- 训练初始化：同一规则专家行为克隆 warm-start 流程。
- locked-test seed：632001；每个场景 30 回合，共 90 回合/方法。
- 预测器：`results/target_predictor_gru_v1/checkpoint.pt`，历史长度 8，
  选择第 3 个预测时域（`horizon_index=2`）。
- 评估执行：CPU，执行阶段不读取 centralized critic 或隐藏目标真值。

离线 GRU locked-test 位置误差（来自 `PREDICTION_GRU_STAGE3_REPORT.md`）为：

| 预测时域 | Constant velocity (m) | GRU (m) | GRU 相对 CV |
|---:|---:|---:|---:|
| 0.1 s | 0.625 | 0.651 | -4.2% |
| 0.3 s | 0.770 | 0.708 | +8.0% |
| 0.5 s | 0.974 | 0.913 | +6.3% |
| 1.0 s | 1.673 | 1.502 | +10.2% |

该表是离线预测误差，不是捕获率；策略级结果见下表。

## 3. 方法配置

| 方法 | actor 输入 | 预测特征 | 安全层 |
|---|---:|---|---|
| MAPPO-no-prediction | 44 维 | 无 | raw / CBF |
| MAPPO-CV-prediction | 48 维 | constant-velocity 预测位置 | raw / CBF |
| MAPPO-GRU-prediction | 52 维 | 冻结 GRU 均值和不确定度 | raw / CBF |

GRU 版本的 `LearnedPredictionObserver` 只替换环境中固定的 4 维预测块，
不改变 centralized critic 和捕获事件定义。

## 4. 总体结果

| 方法 | 回合数 | Safe Capture | Capture | Collision | Boundary Violation | 平均捕获时间 (s) | 平均最小间距 (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAPPO，无预测，raw | 90 | 92.22% | 97.78% | 7.78% | 1.11% | 1.399 | 0.497 |
| MAPPO，无预测，+CBF | 90 | 100.00% | 100.00% | 0.00% | 0.00% | 1.419 | 0.817 |
| MAPPO，常速度预测，raw | 90 | 90.00% | 93.33% | 10.00% | 1.11% | 1.375 | 0.446 |
| MAPPO，常速度预测，+CBF | 90 | 100.00% | 100.00% | 0.00% | 0.00% | 1.431 | 0.793 |
| MAPPO，GRU 预测，raw | 90 | 95.56% | 98.89% | 4.44% | 1.11% | 1.279 | 0.586 |
| MAPPO，GRU 预测，+CBF | 90 | 100.00% | 100.00% | 0.00% | 0.00% | 1.362 | 0.847 |

完整分场景数据保存在四个被评估目录的 `summary.json` 和 `episodes.csv`：

- `results/stage3b_no_prediction_raw_locked30`
- `results/stage3b_no_prediction_cbf_locked30`
- `results/stage3b_constant_velocity_raw_locked30`
- `results/stage3b_constant_velocity_cbf_locked30`
- `results/stage3b_gru_prediction_raw_locked30`
- `results/stage3b_gru_prediction_cbf_locked30`

## 5. 结果解释

1. **常速度特征没有自动带来收益。** 相比无预测 raw action，常速度 raw
   的 Safe Capture 下降 2.22 个百分点、Collision 上升 2.22 个百分点；
   这说明增加一个预测位置字段本身并不能替代学习式预测或合理的策略训练。
2. **GRU raw-action pilot 有小幅策略级收益。** 相比无预测 raw action，
   Safe Capture 提升 3.33 个百分点，Collision 降低 3.33 个百分点；相比
   常速度 raw，Safe Capture 提升 5.56 个百分点。这说明离线预测器接入
   actor 后确实能够影响策略闭环，而不是只停留在预测误差报告。
3. **CBF 是安全收益的主要来源。** 无预测策略使用 CBF 后，Safe Capture
   从 92.22% 提升到 100%，Collision 从 7.78% 降到 0%；GRU 策略也从
   95.56%/4.44% 变为 100%/0%。因此必须同时报告 raw 和 CBF，不能把 CBF
   带来的提升归因于 GRU。
4. **在 CBF 条件下，本 pilot 没有证明 GRU 提高捕获率。** 三种方法均为
   100% Safe Capture。GRU+CBF 的平均捕获时间略短（1.362 s vs 1.419 s），
   平均最小间距略高（0.847 m vs 0.817 m），但这些差异仍只有一个训练
   seed 和 90 回合，不能作为稳定优势。
5. **阶段 3B 仅部分完成。** 在线接口、warm-start、三种预测输入、raw/CBF
   消融和 pilot 报告已完成；3 个训练 seed、至少 300 locked-test 回合和
   Recurrent-MAPPO 尚未完成。

## 6. 当前结论与下一步

当前可以严谨表述为：

> 在本运动学三维仿真的单种子 pilot 中，冻结 GRU 预测特征能够接入非循环
> MAPPO，并在 raw-action 测试中带来小幅 Safe Capture 改善；加入 CBF 后，
> 两种方法在当前 90 回合测试块上均达到 100% Safe Capture。尚不能据此
> 宣称 GRU 对最终安全捕获率具有统计稳定的提升。

下一步按优先级执行：

1. 固定协议，补跑至少 3 个独立训练 seed；
2. 每个主要测试域扩展到至少 300 个 locked-test 回合；
3. 再实现 Recurrent-MAPPO，并完成无预测/GRU 预测的记忆消融；
4. 只有在阶段 3C 稳定后，才加入动作延迟、控制噪声和轻量动力学随机化。

## 7. 可复现证据

- GRU 预测器报告：`results/PREDICTION_GRU_STAGE3_REPORT.md`
- 预测接口：`src/encirclement3d/prediction.py`
- MAPPO 训练：`scripts/train_capture_radius_mappo.py`
- 行为克隆：`scripts/train_capture_radius_behavior_cloning.py`
- 锁定评估：`scripts/evaluate_capture_radius_mappo.py`
- 结构化摘要：`results/PREDICTION_POLICY_STAGE3_PILOT_SUMMARY.json`

所有结果仍限定为运动学三维仿真中的捕获半径事件，不等同于实体接触、网
捕、SITL、真实视觉闭环或实飞捕获。
