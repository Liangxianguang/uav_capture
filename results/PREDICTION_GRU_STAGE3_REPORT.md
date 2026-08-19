# 阶段 3 结果报告：GRU 学习式轨迹预测器

日期：2026-08-19
模型：HistoryTargetPredictor（GRU + heteroscedastic Gaussian head）
训练设备：CUDA（RTX 4060 环境）
训练数据：prediction_dataset_train_v2
验证数据：prediction_dataset_validation_v2
锁定测试：prediction_dataset_locked_test_v2，4304 samples

## 1. 模型与训练

- 输入：8 帧、每帧 52 维 policy-safe observation；
- 输出：4 个未来时域的三维目标相对位置均值和对角协方差；
- 时域：0.1、0.3、0.5、1.0 秒；
- hidden_dim：128；
- batch size：256；
- 最大 epoch：30；
- 最优 epoch：10；
- 选择标准：validation Gaussian NLL；
- hidden target truth 只用于离线 labels，不进入模型输入；
- TensorBoard 和 checkpoint 均保存在 results/target_predictor_gru_v1/。

## 2. locked_test 误差

单位：m。

| 时域 | Zero velocity | Constant velocity | GRU predictor | GRU 相对 CV |
|---:|---:|---:|---:|---:|
| 0.1 s | 0.844 | 0.625 | 0.651 | -4.2% |
| 0.3 s | 1.426 | 0.770 | 0.708 | +8.0% |
| 0.5 s | 2.015 | 0.974 | 0.913 | +6.3% |
| 1.0 s | 3.447 | 1.673 | 1.502 | +10.2% |

“GRU 相对 CV”正值表示 GRU 误差更低。

## 3. 主要困难场景

GRU 在 joint_uncertainty_high_mobility 场景的误差为：

- 0.1 s：1.749 m；
- 0.3 s：2.061 m；
- 0.5 s：2.491 m；
- 1.0 s：3.960 m。

相较 constant velocity（1.972 / 2.500 / 3.086 / 4.572 m），GRU 分别降低约 11.3%、17.6%、19.3% 和 13.4%。这说明历史记忆对高延迟、高失联、高机动场景有实际价值。

## 4. 结论

1. 学习式 GRU 预测器在 0.3–1.0 秒时域优于常速度基线，尤其改善联合高不确定性场景。
2. 0.1 秒短时域略差于常速度，说明模型仍需多时域损失、短期 teacher-free 约束或按置信度加权的训练改进。
3. 当前结果是离线预测误差结果，尚不能声称提升 Safe Capture。下一步必须把 GRU 的均值和不确定度接入 belief-state，并完成 MAPPO/Recurrent-MAPPO 的策略消融。
4. 不应在没有策略闭环实验前把预测误差改善直接表述为围捕成功率改善。

## 5. 可复现证据

- 模型：src/encirclement3d/prediction.py
- 训练：scripts/train_target_predictor.py
- 评估：scripts/evaluate_target_predictor.py
- checkpoint：results/target_predictor_gru_v1/checkpoint.pt
- TensorBoard：results/target_predictor_gru_v1/tensorboard/
- JSON：results/target_predictor_gru_locked_test_v1.json
