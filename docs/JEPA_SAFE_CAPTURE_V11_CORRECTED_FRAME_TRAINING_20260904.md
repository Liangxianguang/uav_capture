# JEPA Safe-Capture v11 corrected-frame 训练审计报告

**阶段：** P2 three-seed training
**日期：** 2026-09-04
**边界：** `development_only=true`，`locked_test_opened=false`
**训练器：** `scripts/train_jepa_safe_capture_v3.py`
**training variant：** `hard_context_weighted_v1`
**label frame variant：** `corrected_post_action_frame_v1`

## 1. 固定输入

- train archive：`results/jepa_safe_capture_v2_p1_corrected_frame_train/counterfactual_safe_capture_v2.npz`
- validation archive：`results/jepa_safe_capture_v2_p1_corrected_frame_validation/counterfactual_safe_capture_v2.npz`
- calibration metadata：`results/jepa_safe_capture_v2_p1_corrected_frame_calibration/metadata.json`
- protocol：`configs/jepa_safe_capture_v2_corrected_frame_v11_protocol.yaml`
- training config：`configs/jepa_safe_capture_v2_corrected_frame_v11_training.yaml`
- model：`interaction_aware_action_conditioned_jepa_safe_capture_v2`
- epochs/batch：`40/512`
- device：RTX 5050 CUDA，PyTorch `2.7.1+cu128`

三 seed 使用完全相同的 train/validation/calibration archive，仅改变训练 seed：`20260911`、`20260912`、`20260913`。

## 2. 训练结果

| seed | epochs | best epoch | best validation loss | checkpoint SHA-256 | TensorBoard |
|---:|---:|---:|---:|---|---|
| 20260911 | 40 | 4 | `-2.7635741386` | `e638c5868a0e6047ad1cefb903973e0979ab5f47bde8b9a44889eb79775aa8d4` | `train_seed20260911` |
| 20260912 | 40 | 4 | `-2.6322110335` | `f3390bd321d6b9155570f8e8f47f4c072cd4c24c7e3c2afda61f00a34a5948a2` | `train_seed20260912` |
| 20260913 | 40 | 5 | `-2.7611511859` | `c545915ab8540e468a6d863687677a9e059c1002aae8c17fa7e320f1c59052b5` | `train_seed20260913` |

共同数据 hash：

- train archive：`a11283a0ab9fa3b0857beb291e5a21c99e7c05170474f1ec07f813fe82a3412f`
- validation archive：`a61c5c92ba6d9f8ac80e13e396297eb863ea2d59434d25b7f594d637049dfbe2`
- calibration metadata：`531ce966d78cc448df4868bc071e507fa64bc9a7b1ee0d121ad367bba20ec6f0`
- training config：`996e9567446bbe2d70edb7f1c700b8a6d20e5e684b43cfc6b43de9df6103a464`

## 3. TensorBoard 和 provenance gate

三份 run 都通过 `scripts/audit_jepa_safe_capture_v3_training.py`：

- 46 个 required/derived scalar tags；
- 9 个 config、dataset 和 source-hash text tags；
- 227 个 parameter/gradient histogram tags；
- checkpoint、run metadata、history 和 training config hash 一致；
- hard-context weight cap 为 `8.0`，每个 epoch 的权重均在声明范围内；
- checkpoint 和 run metadata 均保存 training variant、seed、数据 hash 与 TensorBoard 路径；
- 三个 run 的 locked flag 均为 `false`。

## 4. 结论

P2 训练合同已满足，三份 corrected-frame checkpoint 可以进入 P3 prediction gate 和 v3 reliability ledger 构建。该 audit 只证明训练/provenance 合同成立，不证明闭环 `safe_capture` 提升；闭环收益仍需同一 paired scene manifest 下的完整 episode 证据。
