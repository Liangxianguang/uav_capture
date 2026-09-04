# JEPA Safe-Capture P10 独立校准审计

**日期：** 2026-09-04
**状态：** development-only；`locked_test_opened=false`
**硬件：** NVIDIA RTX 5050；PyTorch 2.7.1+cu128
**模型：** `interaction_aware_action_conditioned_jepa_safe_capture_v2`
**训练变体：** `hard_context_weighted_v1`
**校准数据：** 独立 P1 calibration archive，76800 samples，64 episodes

## 1. 审计范围

本阶段没有重复添加模型头，而是对现有 action-conditioned、interaction-aware 多任务 JEPA 做独立 calibration audit。审计覆盖：

- target displacement、velocity、acceleration；
- obstacle/inter-agent clearance lower-quantile；
- pairwise TTC 和 observation age；
- visibility probability；
- CBF intervention probability、correction magnitude；
- QP feasibility probability；
- 同一 belief 下五个 candidate action chunks 的 action-conditioning spread。

target ground truth 只用于离线 settled labels；运行时不读取 target truth，也没有执行环境 rollout。

## 2. 三 seed 训练 provenance

`audit_jepa_safe_capture_v3_training.py` 对下列三份 40-epoch checkpoint 均通过：

| Seed | Checkpoint SHA-256 | Best epoch | Best validation loss | TensorBoard |
|---:|---|---:|---:|---|
| 20260911 | `f7b7f5a4b540fa634f23f6e79788dc972ec3b39f40862ae4b067be98a18b078e` | 5 | -2.7734 | 46 scalar / 9 text / 227 histogram |
| 20260912 | `496d8d15ed493da5329bd11c4327bf364abb1546d059523dc56dcab2d837a496` | 5 | -2.7514 | 46 scalar / 9 text / 227 histogram |
| 20260913 | `cff4c2a9cc5922791f8e2f2f9b5e08cd41c302b0e63fe642645c9ef98ebe209d` | 4 | -2.7586 | 46 scalar / 9 text / 227 histogram |

训练 audit 检查了 finite checkpoint、连续 40 epoch history、hard-context 权重、配置/source hash、TensorBoard scalar/text/histogram 和 locked-test 标记，三份均通过。

## 3. 独立 calibration 结果

结果文件：`results/jepa_safe_capture_v4_p10_calibration_audit_rerun/`。下表为三个 checkpoint 的均值 +/- 样本 SD。

| Horizon (s) | Target MAE (m) | Target improvement vs CV | 1-std coverage | Clearance q10 coverage | Clearance overprediction | Visibility Brier / ECE | CBF intervention Brier / ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.2822 +/- 0.0090 | 0.0936 +/- 0.0288 | 0.7958 +/- 0.0082 | 0.8966 +/- 0.0319 | 0.1034 +/- 0.0319 | 0.1908 +/- 0.0065 / 0.0897 +/- 0.0262 | 0.0694 +/- 0.0095 / 0.0275 +/- 0.0223 |
| 0.2 | 0.3268 +/- 0.0067 | 0.2579 +/- 0.0152 | 0.7397 +/- 0.0028 | 0.8002 +/- 0.1806 | 0.1998 +/- 0.1806 | 0.1911 +/- 0.0028 / 0.0784 +/- 0.0149 | 0.0861 +/- 0.0148 / 0.0361 +/- 0.0393 |
| 0.3 | 0.3960 +/- 0.0057 | 0.3366 +/- 0.0095 | 0.6672 +/- 0.0120 | 0.9331 +/- 0.0297 | 0.0669 +/- 0.0297 | 0.1954 +/- 0.0049 / 0.0803 +/- 0.0230 | 0.0974 +/- 0.0159 / 0.0440 +/- 0.0438 |
| 0.5 | 0.5333 +/- 0.0051 | 0.4380 +/- 0.0054 | 0.6537 +/- 0.0137 | 0.7769 +/- 0.1380 | 0.2231 +/- 0.1380 | 0.1920 +/- 0.0045 / 0.0722 +/- 0.0230 | 0.0893 +/- 0.0049 / 0.0458 +/- 0.0169 |

## 4. 门状态与解释

### 通过

- 三 seed 所有预测 finite。
- 每个 horizon 的 target displacement MAE 都优于 constant-velocity。
- clearance lower-quantile coverage、overprediction rate、visibility Brier/ECE 和 CBF intervention Brier/ECE 已在独立 calibration split 上计算。
- 每个 state-agent group 恰有 5 个候选，action-conditioned prediction spread 非零比例为 `1.0`。
- 三个新 ledger 均通过 OOD、stale、non-finite safe-hold fallback audit；TensorBoard 已记录配置、hash、校准 scalar 和直方图。

### 尚未通过或不可判定

- 当前 `labels_cbf_qp_feasible` 每个 horizon 都只有一个类别，QP feasibility AUC 不可定义；该 head 暂时只能作为 diagnostic，不能作为可靠性证明。
- clearance coverage 在 0.2 和 0.5 秒的 seed 间方差较大，必须在 P11/P12 中采用保守 lower bound 和 uncertainty penalty，不能直接把均值用于安全决策。
- 预测/校准通过不等于闭环 safe-capture 提升；settled counterfactual ranking、ledger routing、CBF 和 rolling-horizon 回归仍未完成。

## 5. 代码与产物

- 新增 `scripts/audit_jepa_safe_capture_v4_calibration.py`：三 seed 独立 calibration audit、ECE/AUC、lower-quantile coverage、action-conditioning 和 TensorBoard 记录。
- 新增 `tests/test_audit_jepa_safe_capture_v4_calibration.py`：AUC、ECE、五候选分组校验。
- 更新 `scripts/build_jepa_safe_capture_v2_reliability_ledger.py`：返回已有 `cbf_correction` 预测，使 correction calibration 可审计；不改变旧 ledger 的计算逻辑。
- 新增 `results/jepa_safe_capture_v4_p10_calibration_audit_rerun/audit.json` 和 `report.md`。
- 新增 `results/jepa_safe_capture_v4_p10_ledger_seed20260911/`、`...seed20260912/`、`...seed20260913/`，每份绑定 checkpoint、calibration archive 和 v3 protocol。
- TensorBoard：`results/jepa_safe_capture_v4_tensorboard/p10_calibration_aggregate_rerun/`；三份 ledger TensorBoard 位于同一根目录下的 `p10_ledger_seed*`。

## 6. P10 结论

P10 的“训练 provenance + prediction + 辅助头 calibration observability”子门通过，但 P10 不授权直接运行新的 final paired block。下一步必须先完成：

1. P11 settled counterfactual rank calibration、安全优先排序、top-two margin、hysteresis 和 conservative abstention；
2. P12 ledger 的 high-credit failure、OOD/stale/credit decay 校准；
3. P13 rolling-horizon、zero-perturbation 和全回退链路回归；
4. 通过新的 20-episode/seed smoke 后，才可运行三 seed paired development。

本阶段没有打开 locked test，也没有将 prediction/calibration 结果写成 safe-capture 控制收益。
