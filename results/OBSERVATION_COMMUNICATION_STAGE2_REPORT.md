# 阶段 2 结果报告：观测与通信不确定性基线

日期：2026-08-19
任务：部分可观测三维捕获半径追逃
控制器：Pure Pursuit + local CBF
环境：kinematic 3D，4 架防守无人机，1 个逃逸目标，r_capture = 0.80 m
回合数：5 个场景 × 100 = 500 个锁定回合
结果目录：results/stage2_observation_communication_pure_cbf_v4

## 1. 实现内容

阶段 2 已将以下因素纳入统一、可配置的观测接口：

- 位置/速度观测噪声；
- 单步漏检；
- 连续失联 burst；
- 本机观测延迟；
- 队友消息延迟；
- 单条消息丢包；
- 链路级丢包；
- belief 置信度衰减；
- belief 协方差增长；
- observation timestamp、observation age 和 covariance。

冻结基线的 44 维 actor 输入没有被修改。阶段 2 配置显式开启 prediction 与 uncertainty features，新的 policy observation 维度为 52，因此不直接加载旧 44 维 checkpoint。

## 2. 场景结果

| 场景 | Safe Capture | Capture | World Violation | Physical Collision | Visible Fraction | Message Age | Obs. Confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| nominal_partial_observation | 92% | 92% | 8% | 0% | 0.899 | 0.184 | 0.832 |
| delayed_measurements | 6% | 6% | 82% | 0% | 0.898 | 6.149 | 0.743 |
| burst_occlusion | 43% | 43% | 57% | 0% | 0.358 | 2.917 | 0.766 |
| communication_loss | 41% | 41% | 59% | 0% | 0.853 | 1.521 | 0.748 |
| joint_uncertainty_high_mobility | 15% | 15% | 85% | 0% | 0.275 | 21.199 | 0.490 |

完整数值见：

- results/stage2_observation_communication_pure_cbf_v4/summary.json
- results/stage2_observation_communication_pure_cbf_v4/episodes.csv
- results/stage2_observation_communication_pure_cbf_v4/tensorboard/

TensorBoard 事件核验：包含 140 个 scalar tag，以及 Config/effective_benchmark/text_summary 配置记录。

## 3. 结果解释

1. 观测/通信不确定性接口已经产生了清晰的难度梯度。延迟、连续失联和高机动联合出现时，Safe Capture 从 92% 降至 6%–15% 区间。
2. 本轮物理碰撞率为 0%，但边界越界率显著上升。因此当前主要瓶颈不是 CBF 对障碍/队友碰撞的处理，而是陈旧目标 belief 导致的追踪方向错误和边界安全失败。
3. delayed_measurements 的可见率仍为 89.8%，但 message age 为 6.15 步且 Safe Capture 只有 6%，说明“看得到目标”不等于策略使用的是及时信息。
4. joint_uncertainty_high_mobility 的平均观测置信度最低、观测协方差最高、消息年龄最大，适合作为阶段 3 预测与记忆策略的主困难测试域。
5. Pure Pursuit + CBF 只是可解释基线，不代表阶段 3 的学习式方法性能；本报告不把规则控制器结果当作论文最终方法结果。

## 4. 可复现实验信息

- 配置：configs/capture_radius_observation_communication.yaml
- seed blocks：train=630001，validation=631001，locked_test=632001
- 本次评估使用 locked_test=632001
- 每个场景：100 回合
- episode seed：630001 + 场景索引 × 10000 + 回合索引
- 控制器：pure_cbf
- backend：kinematic
- 输出包含 effective YAML、环境版本、源码 hash、CSV、JSON、轨迹图和 TensorBoard event 文件。

命令：

    conda run --no-capture-output -n uav-encirclement-gpu python scripts/run_capture_radius_pursuit.py --config configs/capture_radius_observation_communication.yaml --controller pure_cbf --output results/stage2_observation_communication_pure_cbf_v4

## 5. 阶段结论与下一步

阶段 2 达到接口和单个 locked_test 基线验收要求：不确定性可配置、可记录、可分桶、可复现，且通过真值泄漏与延迟回归测试。多训练种子策略比较留到阶段 3，不能用本轮规则基线替代。

下一步进入阶段 3：

- 生成仅由局部历史观测构成的预测器数据集；
- 先实现 constant-velocity predictor；
- 再训练学习式预测器并输出不确定度；
- 实现 GRU/LSTM Recurrent-MAPPO；
- 在本阶段锁定的五类测试域上进行 MAPPO、预测、记忆和 CBF 消融；
- 仍然区分 kinematic 结果与后续可选动力学验证。
