# 阶段 4B 结果：年龄门控速度衰减 Validation 选择

- Validation seed block：641001 起，共 100 个 episode/条件/候选。
- 所有候选均使用 `time_aligned`，当观测年龄达到 3 步后对速度乘以候选衰减系数。
- 选择规则：每个条件的新时间戳更新误差不比 `aligned_decay_1_00` 高 10%，
  再选择平均已初始化位置误差最低的候选。该选择不读取 P1 locked-test seed。

| 条件 | 候选 | 已初始化位置误差 (m) | 新时间戳更新误差 (m) |
|---|---|---:|---:|
| delayed_measurements | aligned_decay_0_00 | 1.818 | 0.333 |
| delayed_measurements | aligned_decay_0_50 | 1.785 | 0.333 |
| delayed_measurements | aligned_decay_0_80 | 1.734 | 0.333 |
| delayed_measurements | aligned_decay_1_00 | 9.348 | 0.333 |
| burst_occlusion | aligned_decay_0_00 | 1.154 | 0.152 |
| burst_occlusion | aligned_decay_0_50 | 1.103 | 0.152 |
| burst_occlusion | aligned_decay_0_80 | 1.025 | 0.152 |
| burst_occlusion | aligned_decay_1_00 | 3.745 | 0.152 |

选择：`aligned_decay_0_80`，衰减系数 0.80。
此结果仅允许进入新的 locked-test estimator 复验；尚不构成策略训练或 Safe Capture 改进结论。

```powershell
conda run --no-capture-output -n uav-encirclement-gpu python scripts/select_stage4b_velocity_gate.py
```
