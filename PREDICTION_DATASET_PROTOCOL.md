# 阶段 3（第一步）：局部历史轨迹预测数据协议

## 1. 目的

本协议定义从阶段 2 环境生成预测器数据集的方式。预测器只接收执行阶段可获得的局部信息；模拟器的目标真值只在离线生成标签时使用。

## 2. 样本定义

对每一架防守无人机，在时间 t 构造：

- 输入：最近 K=8 帧该无人机的 policy-safe observation；
- 标签：目标在 t+0.1、0.3、0.5、1.0 秒的位置；
- 标签坐标：相对于时间 t 该无人机当前位置的三维位移，按 10 m 世界尺度归一化；
- belief 辅助字段：相对 belief 位置、belief 速度、confidence、covariance 对角线和 message age；
- 样本元数据：agent id、时间索引、episode seed、场景索引。

样本数组：

- inputs: [N, 8, 52]
- labels_relative: [N, 4, 3]
- belief_relative: [N, 3]
- belief_velocity: [N, 3]
- confidence: [N]
- covariance: [N, 3]
- message_age: [N]

labels_relative 是离线监督标签，不属于执行阶段输入；NPZ 中不保存 simulator target position 作为输入特征。

## 3. 数据划分

配置文件：configs/capture_radius_observation_communication.yaml

| split | seed block | 每场景回合 | 用途 |
|---|---:|---:|---|
| train | 630001 | 20 | 预测器训练 |
| validation | 631001 | 10 | 超参数和早停 |
| locked_test | 632001 | 10 | 最终报告，不调参 |

五个场景保持相同配置结构，但使用不同场景 seed 偏移。训练、验证和锁定测试不共享 episode seed。

## 4. 生成命令

    conda run --no-capture-output -n uav-encirclement-gpu python scripts/generate_prediction_dataset.py --config configs/capture_radius_observation_communication.yaml --output results/prediction_dataset_train_v2 --split train --episodes-per-scenario 20 --controller pure_cbf --history-length 8 --horizon-steps 1 3 5 10

验证集和锁定测试集只需将 --split 改为 validation 或 locked_test，并使用对应输出目录。

## 5. 基线

- zero velocity：保持最新 belief 位置；
- constant velocity：使用最新 belief 位置和速度外推。

评估命令：

    conda run --no-capture-output -n uav-encirclement-gpu python scripts/evaluate_prediction_baselines.py --dataset results/prediction_dataset_locked_test_v2/prediction_dataset.npz --metadata results/prediction_dataset_locked_test_v2/metadata.json --output results/prediction_baselines_locked_test_v2.json

## 6. 信息边界检查

- predictor input 不包含 target_position、target_velocity 或 centralized state；
- 延迟和漏检由环境接口产生，不能用离线真值填补输入；
- 训练/验证/测试按 seed block 隔离；
- 预测误差按场景和时间 horizon 分桶；
- 预测器指标不能直接等价为捕获率提升，必须在阶段 3 接入策略后重新评估。
