# 阶段 4A 结果：时延与遮挡失败轨迹诊断

任务：部分可观测三维障碍环境下多无人机协同捕获半径追逃。

## 1. 目的与边界

本阶段不训练新策略。它对已冻结的 Stage 3C-P1 `raw` 动作失败回合进行分层选择和确定性重放，
以定位后续时间对齐 belief 应解决的观测问题。目标真值只在重放结束后用于计算评估误差，
从不进入 actor、预测器或 CBF 的输入。

## 2. 诊断协议

- 条件：delayed_measurements, burst_occlusion。
- 冻结方法：recurrent_no_prediction, recurrent_gru_prediction；执行方式：raw。
- 每个条件至多选取 20 个 Safe Capture 失败回合，按 `(method, training_seed)` 轮询，
  不按最坏最典型轨迹进行人工挑选。
- 每个候选均使用其原始训练 checkpoint、评估 seed 和 P1 场景配置重放；
  Safe Capture、碰撞、步数、终止原因和越界计数必须与原始 CSV 完全一致。
- 逐回合表、NPZ 轨迹和协议保存在本地 `results/stage4a_belief_diagnostics`。

## 3. 失败轨迹统计

| 条件 | 轨迹数 | 碰撞失败 | 超时失败 | belief 位置误差 (m) | p95 位置误差 (m) | 观测年龄 (steps) | 最大年龄 (steps) | 重获观测事件/轨迹 | belief 更新/轨迹 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| delayed_measurements | 20 | 20 | 0 | 1.537 | 5.491 | 2.928 | 4.400 | 6.900 | 47.600 |
| burst_occlusion | 20 | 20 | 0 | 0.904 | 5.712 | 2.302 | 4.800 | 9.600 | 51.300 |

## 4. belief 误差与观测年龄

| 条件 | 未初始化帧 / 误差 (m) | Fresh (0-1) 帧 / 误差 (m) | Moderate (2-4) 帧 / 误差 (m) | Stale (>=5) 帧 / 误差 (m) |
|---|---:|---:|---:|---:|
| delayed_measurements | 256 / 5.472 | 0 / n/a | 1052 / 0.555 | 20 / 1.103 |
| burst_occlusion | 160 / 5.714 | 0 / n/a | 1253 / 0.283 | 79 / 0.637 |

## 5. 可解释结论

- 所有入选回合均是原始 P1 中已记录的失败，并且重放一致；因此本报告可用于后续方法设计，
  但不以单条视频代替统计结论。
- 时延条件的 belief 仍携带陈旧时间戳；突发遮挡条件存在持续不可见与再次可见事件。
  尚未收到任何观测的初始 belief 单独统计，已初始化 belief 的位置误差再按观测年龄分桶，
  从而避免将空 belief 误解为低年龄观测。
- 这些指标是描述性证据，不将碰撞单独归因于某一模块。后续 F1 应首先检验时间对齐 belief
  是否降低时延/遮挡域的估计误差和重获观测后的恢复时间，再评估 Safe Capture。

## 6. 复现

```powershell
conda run --no-capture-output -n uav-encirclement-gpu python scripts/analyze_stage4a_belief_failures.py --device cpu
```

本次共重放 40 条失败轨迹。
