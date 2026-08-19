# 阶段 2：观测与通信不确定性实验协议

## 1. 目的

本协议把阶段 1 中的困难追逃条件扩展为可配置的观测与通信不确定性，回答：

> 在目标漏检、连续遮挡、测量延迟、通信延迟、丢包和陈旧 belief 同时存在时，捕获半径策略的性能如何变化？

本阶段仍使用运动学三维仿真。它不声称实现真实相机、RGB-D、LiDAR、SITL 或实飞感知闭环。

## 2. 信息约束

执行阶段每架无人机只能访问：

- 自身位置和速度；
- 局部障碍几何；
- 本机目标 belief；
- 目标观测 visible、confidence、timestamp、covariance；
- 队友发送且已经到达的目标消息及其 message age；
- 历史信息由上层策略自行维护。

隐藏目标真值只用于环境转移、集中式 critic 和离线评估，不得进入 actor observation。

## 3. 不确定性变量

| 变量 | 含义 | 记录字段 |
|---|---|---|
| observation_noise_std | 位置噪声标准差；速度噪声按 std/dt 缩放 | covariance |
| detection_dropout_probability | 独立漏检概率 | visible |
| detection_loss_burst_probability | 每架无人机启动连续失联的概率 | visible |
| detection_loss_burst_duration_steps | 连续失联长度 | visible |
| observation_delay_steps | 本机检测进入 belief 的延迟 | timestamp、observation age |
| message_delay_steps | 队友消息传输延迟 | message age |
| message_dropout_probability | 单条消息丢失概率 | message age |
| communication_link_dropout_probability | 链路级额外丢失概率 | message age |
| observation_confidence_decay | 失联期间置信度衰减 | confidence |
| observation_covariance_growth | 失联期间协方差增长 | covariance |

所有变量均通过 YAML 配置，随机数只来自 episode seed 对应的环境 RNG。

## 4. 实验分块

配置文件：configs/capture_radius_observation_communication.yaml

- nominal_partial_observation：中等噪声和已有遮挡；
- delayed_measurements：本机检测延迟 3 步、消息延迟 5 步；
- burst_occlusion：S 型目标、连续 5 步失联；
- communication_loss：突发目标、窄通道、消息延迟/丢包；
- joint_uncertainty_high_mobility：高机动、联合观测与通信不确定性。

每个场景 100 个锁定测试回合；配置文件显式保存 train、validation 和 locked_test seed block，当前规则基线使用 locked_test：

locked_test + scenario_index * 10000 + episode_index

训练、验证和锁定测试种子必须在后续策略训练时分离。阶段 2 的规则基线使用锁定测试集，不得据此调参。

## 5. 观测维度策略

冻结基线保持 policy_observations.shape == (4, 44)。

阶段 2 配置开启 include_prediction_features 和 include_uncertainty_features，形成新的实验接口；因此该配置不得直接加载 44 维冻结 checkpoint。后续 Recurrent-MAPPO 应显式记录新的输入维度和配置 hash。

当前新增字段只增加信息表达能力，不改变环境的捕获判据。

## 6. 必须记录的结果

每回合 CSV/JSON：

- Capture、Safe Capture、Collision、Boundary Violation；
- Capture time、minimum clearance；
- mean target visible fraction；
- mean message age；
- mean observation confidence；
- mean observation age；
- mean observation covariance trace；
- CBF correction 和最低 barrier。

TensorBoard：

- Episode/<scenario>/<metric>：逐回合曲线；
- Summary/<scenario>/<metric>：场景汇总；
- Config/effective_benchmark：有效 YAML。

## 7. 真值泄漏测试

至少验证以下不变量：

1. 在已经发布的 observation 后修改 env.target_position 和 env.target_velocity，该 observation 的 belief、confidence、timestamp、covariance 不改变。
2. policy_observations 不含 target_position、target_velocity 或全局状态。
3. 延迟消息到达前，receiver belief 不因隐藏目标真值改变。
4. 相同 config 和 seed 的两个环境，观测、消息和轨迹逐步一致。

## 8. 阶段验收

- 单元测试覆盖新字段、延迟、连续失联、链路丢包和真值泄漏；
- 完整测试集通过；
- 五个场景各 100 回合，输出 episodes.csv、summary.json、TensorBoard event 文件和 source hashes；
- 结果按场景分桶，不能用混合平均值掩盖失败条件；
- 阶段报告明确区分运动学基线结论与未来学习策略结论。

## 9. 下一阶段接口

阶段 3 的预测器必须读取本阶段生成的局部历史观测，至少使用：

position belief + velocity belief + confidence + observation age + covariance + message age

不得读取环境真值或测试集标签以外的信息。
