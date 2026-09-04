# JEPA Safe-Capture v11 corrected-frame archive 审计报告

**日期：** 2026-09-04  
**阶段：** P1 archive generation and audit  
**边界：** `development_only=true`，`locked_test_opened=false`  
**硬件/环境：** RTX 5050，Conda `uav-encirclement-gpu`，Python 3.11.16，PyTorch 2.7.1+cu128

## 1. 目的

本阶段重新生成 action-conditioned interaction-aware JEPA 的 train、validation 和 calibration archive，修复历史标签中 `target_relative` 使用动作前 defender frame 的问题。所有离线标签现在使用：

```text
target_relative = (post_action_target_position - post_action_defender_position) / world_half_extent
```

target ground truth 仅用于离线结算；在线 evaluator 不读取 target truth。

## 2. 输入合同

| 项目 | 冻结值 |
|---|---|
| dataset version | `jepa_safe_capture_v2_p1_corrected_frame` |
| target frame | `post_action_defender_position` |
| frame correction version | `1` |
| history length | `8` |
| horizon steps | `[1, 2, 3, 5]` |
| candidates | `5` |
| chunk length | `3` |
| sample stride | `4` |
| perturbation | `0.10 m/s` |
| target truth | offline labels only |

输入配置：

- `configs/jepa_safe_capture_v2_corrected_frame_v11_collection.yaml`
- `configs/jepa_safe_capture_v2_corrected_frame_v11_protocol.yaml`
- actor：`models/v5_development_exact_reactive_seed661606.pt`

## 3. Archive 结果

| split | episodes | samples | dataset SHA-256 | metadata SHA-256 | TensorBoard |
|---|---:|---:|---|---|---|
| train | 64 | 78,080 | `a11283a0ab9fa3b0857beb291e5a21c99e7c05170474f1ec07f813fe82a3412f` | `8e963be106f72dc6829f13284a58bb1778135fa215a24fb188914ad9bf571878` | `archive_train` |
| validation | 64 | 78,080 | `a61c5c92ba6d9f8ac80e13e396297eb863ea2d59434d25b7f594d637049dfbe2` | `a68ac075c1f9ea1236688c0dfc26e66a50bafddb063d9b55df5450a4b66fcf34` | `archive_validation` |
| calibration | 64 | 78,080 | `ea04eec8e255bcafa95386ef4c30e366e55723334b8d4985d6c94887b9a1a307` | `531ce966d78cc448df4868bc071e507fa64bc9a7b1ee0d121ad367bba20ec6f0` | `archive_calibration` |

本机结果目录：

```text
results/jepa_safe_capture_v2_p1_corrected_frame_train/
results/jepa_safe_capture_v2_p1_corrected_frame_validation/
results/jepa_safe_capture_v2_p1_corrected_frame_calibration/
results/jepa_safe_capture_v5_v11_corrected_frame_tensorboard/archive_train/
results/jepa_safe_capture_v5_v11_corrected_frame_tensorboard/archive_validation/
results/jepa_safe_capture_v5_v11_corrected_frame_tensorboard/archive_calibration/
```

## 4. 审计结论

- 三个 split 的 episode seed 完全不重叠。
- 每个 state-agent group 恰好包含五个 candidate；invalid group 数为 `0`，nominal fraction 为 `0.2`。
- 三个 archive 的所有 arrays 均 finite，输入 shape 为 `8 x 63`，action history shape 为 `8 x 3`。
- `labels_target_visible`、`labels_cbf_intervention`、`labels_cbf_qp_feasible`、`labels_collision` 和 `labels_boundary` 均通过二值合同。
- 所有 archive 的 obstacle/boundary/collision offline label fraction 为 `0`；CBF intervention fraction 分别为 `0.1302`、`0.1421`、`0.1233`。
- archive metadata、archive manifest、实际 collection/protocol 文件和 SHA-256 一致。
- 每份 TensorBoard 均存在事件文件、配置文本、metadata 文本、source hash 文本、finite/coverage scalar 和 label histogram。
- locked split 未被读取，`locked_test_opened=false`。

## 5. 可复核命令

```powershell
Set-Location D:\\uav-capture\\uav_capture
$py = 'D:\\miniconda3\\envs\\uav-encirclement-gpu\\python.exe'
$env:PYTHONPATH = "$PWD\\src;$PWD\\scripts"
& $py scripts/audit_jepa_safe_capture_v2_archive.py `
  --dataset-dir results/jepa_safe_capture_v2_p1_corrected_frame_train `
  --compare-dataset-dir results/jepa_safe_capture_v2_p1_corrected_frame_validation `
  --compare-dataset-dir results/jepa_safe_capture_v2_p1_corrected_frame_calibration `
  --output results/jepa_safe_capture_v2_p1_corrected_frame_archive_audit.json
```

审计结果为 `all_episode_seeds_disjoint=true`、`all_finite=true`、`locked_test_opened=false`。这些 archive 现在可以作为 P2 三 seed 训练的输入；在训练完成前不得从 development episode 反向修改 archive。
