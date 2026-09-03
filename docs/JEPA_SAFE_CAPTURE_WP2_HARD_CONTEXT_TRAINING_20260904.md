# JEPA Safe-Capture WP2 Hard-Context Weighted Training

**日期：** 2026-09-04
**阶段：** development-only
**locked test：** `locked_test_opened=false`
**硬件：** NVIDIA RTX 5050，PyTorch `2.7.1+cu128`，Conda 环境 `uav-encirclement-gpu`

## 目标

WP2 在保持 v2 序列化模型类型和运行时输入形状兼容的前提下，对遮挡、过期观测、低障碍净空、低机间净空、低 pairwise TTC 和 CBF 干预样本增加有限权重。训练只读取 P1 train archive，验证只读取 P1 validation archive，calibration metadata 只用于 provenance，不把 development/locked rollout 回灌训练。

## 三 seed 训练审计

| Seed | Epochs | Best epoch | Best validation loss | Device | Checkpoint SHA-256 |
| ---: | ---: | ---: | ---: | --- | --- |
| 20260911 | 40 | 5 | -2.773418 | cuda | `f7b7f5a4b540fa634f23f6e79788dc972ec3b39f40862ae4b067be98a18b078e` |
| 20260912 | 40 | 5 | -2.751427 | cuda | `496d8d15ed493da5329bd11c4327bf364abb1546d059523dc56dcab2d837a496` |
| 20260913 | 40 | 4 | -2.758563 | cuda | `cff4c2a9cc5922791f8e2f2f9b5e08cd41c302b0e63fe642645c9ef98ebe209d` |

所有 run 均通过 `scripts/audit_jepa_safe_capture_v3_training.py`：checkpoint 参数 finite，history 为连续 40 epoch，hard-context 权重满足 `[1, 8]`，并且 TensorBoard 含 46 个 scalar、9 个 text provenance 和 227 个 histogram tags。训练配置、数据、源码和 protocol hash 记录在 `results/jepa_safe_capture_v3_wp2_training_audit.json`。

## Held-out prediction gate

三个 checkpoint 在同一 P1 validation archive（`77400` samples）上运行 `scripts/evaluate_jepa_safe_capture_v2.py`，再由 `scripts/aggregate_jepa_safe_capture_v3_prediction.py` 聚合。四个 horizon（0.1/0.2/0.3/0.5 s）的 target MAE 相对 constant-velocity 均为正改善，三 seed 在每个 horizon 均改善；所有辅助预测头输出 finite。

详细指标见 [WP2 prediction aggregate](JEPA_SAFE_CAPTURE_WP2_HARD_CONTEXT_PREDICTION_20260904.md)，机器可读结果见 `results/jepa_safe_capture_v3_wp2_prediction_aggregate.json`。

## 结论边界

- 该结果证明 hard-context weighted JEPA 具备进入 reliability-ledger calibration 的离线预测条件。
- 预测 MAE、Brier、AUROC 或 uncertainty 都不是安全证书；最终 action 仍必须经过 Joint CBF-QP。
- 当前尚未完成 ledger 绑定、候选闭环排序或 safe-capture paired development，因此不能宣称 safe-capture 提升，也不能打开 locked test。
- 下一步固定为 WP3 ledger 校准 -> WP1 hard replay -> WP4/WP5 接口审计 -> 20 集 smoke -> 三 seed paired development。
