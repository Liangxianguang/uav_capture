# 阶段 3 结果报告：局部历史预测数据集与常速度基线

日期：2026-08-19
任务：部分可观测三维捕获半径追逃
数据生成控制器：Pure Pursuit + local CBF
环境：kinematic 3D
数据划分：train / validation / locked_test
锁定测试样本：4304 条，5 个场景，4 个预测时域

## 1. 已完成内容

- 实现 scripts/generate_prediction_dataset.py；
- 输入严格来自 policy-safe local observation；
- 支持 history length、预测时域和 seed block 配置；
- 生成 train、validation、locked_test 三个独立数据块；
- 实现 zero-velocity 和 constant-velocity 预测基线；
- 增加数据集组装单元测试，验证历史窗口、未来标签和归一化定义。

数据集形状：

| split | episodes | samples | input shape | label shape |
|---|---:|---:|---|---|
| train | 100 | 10168 | [10168, 8, 52] | [10168, 4, 3] |
| validation | 50 | 5204 | [5204, 8, 52] | [5204, 4, 3] |
| locked_test | 50 | 4304 | [4304, 8, 52] | [4304, 4, 3] |

## 2. 锁定测试整体误差

单位：m；标签是目标相对于当前防守无人机位置的未来位移。

| 预测时域 | Zero velocity | Constant velocity | 相对降低 |
|---:|---:|---:|---:|
| 0.1 s | 0.844 | 0.625 | 26.0% |
| 0.3 s | 1.426 | 0.770 | 46.0% |
| 0.5 s | 2.015 | 0.974 | 51.7% |
| 1.0 s | 3.447 | 1.673 | 51.5% |

## 3. 按场景误差

表中每项为“zero velocity / constant velocity”的平均位置误差，单位 m。

| 场景 | 0.1 s | 0.3 s | 0.5 s | 1.0 s |
|---|---:|---:|---:|---:|
| nominal_partial_observation | 0.347 / 0.217 | 0.849 / 0.399 | 1.365 / 0.620 | 2.629 / 1.270 |
| delayed_measurements | 1.045 / 0.800 | 1.563 / 0.874 | 2.075 / 1.009 | 3.305 / 1.579 |
| burst_occlusion | 0.518 / 0.278 | 1.206 / 0.464 | 1.902 / 0.719 | 3.610 / 1.562 |
| communication_loss | 0.433 / 0.214 | 1.120 / 0.389 | 1.826 / 0.619 | 3.576 / 1.351 |
| joint_uncertainty_high_mobility | 2.094 / 1.972 | 2.864 / 2.500 | 3.689 / 3.086 | 5.723 / 4.572 |

## 4. 结论

1. 常速度预测在全部锁定时域上优于保持最新 belief 的零速度基线，说明阶段 2 的 belief velocity 具有可利用信息。
2. 高机动联合不确定性场景仍然最难：1.0 s 常速度误差为 4.572 m，远高于其他场景。这是学习式预测器最有价值的测试域。
3. delayed_measurements 在可见目标下仍有较大误差，说明测量可见性不能代替时间新鲜度；预测器需要显式使用 timestamp、message age 和 covariance。
4. 该报告只证明预测数据管线和常速度基线成立，不证明捕获策略已经因预测而提升。下一步必须把学习式预测输出接入策略，并在同一 locked_test 上进行消融。

## 5. 可复现证据

协议：PREDICTION_DATASET_PROTOCOL.md
生成脚本：scripts/generate_prediction_dataset.py
评估脚本：scripts/evaluate_prediction_baselines.py
锁定测试数据：results/prediction_dataset_locked_test_v2/
锁定测试误差：results/prediction_baselines_locked_test_v2.json

所有数据集 metadata 保存了 config、seed block、输入维度、归一化方式、环境版本和源码 hash。
