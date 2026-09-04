# JEPA Safe-Capture v11 Hard-Replay Training Audit

**范围：** development-only；`locked_test_opened=false`
**模型：** `interaction_aware_action_conditioned_jepa_safe_capture_v2`
**训练变体：** `hard_context_weighted_v1`
**设备：** RTX 5050 / CUDA / PyTorch 2.7.1+cu128

## 结果

三个训练 seed 均完成 40 个 epoch，使用同一 corrected-frame train、validation 和 calibration metadata。训练审计脚本为 `scripts/audit_jepa_safe_capture_v3_training.py`，原始 JSON 为 `results/jepa_safe_capture_v11_hard_replay_training_audit/summary.json`。

| Seed | Best epoch | Best validation loss | Checkpoint SHA-256 | TensorBoard |
|---:|---:|---:|---|---|
| 20260911 | 4 | -2.7635741386 | `2317a9464f8001f27a5c028bb6b4c431c904af7bfc33bf43b3a1d05a5a9c6154` | 46 scalar / 9 text / 227 histogram |
| 20260912 | 4 | -2.6322110335 | `8ff2531e64571c9e57cfd78e9023a8b49191e06d1c4e4fd00adfaec90b629185` | 46 scalar / 9 text / 227 histogram |
| 20260913 | 5 | -2.7611511859 | `9fe66b66a6ea441807022c1fde71e61b578df3df6ab7265532761d70d6fab708` | 46 scalar / 9 text / 227 histogram |

## Shared provenance

- Train dataset SHA-256: `a11283a0ab9fa3b0857beb291e5a21c99e7c05170474f1ec07f813fe82a3412f`
- Validation dataset SHA-256: `a61c5c92ba6d9f8ac80e13e396297eb863ea2d59434d25b7f594d637049dfbe2`
- Calibration metadata SHA-256: `531ce966d78cc448df4868bc071e507fa64bc9a7b1ee0d121ad367bba20ec6f0`
- Training config SHA-256: `d8ff356aeebd8f1b25bda340fdb56c500df0b6568df110ede69fd76c57d07c7e`
- Hard-context weights: cap `8.0`; occlusion/stale `1.5`; obstacle/inter-agent/TTC `2.0`; CBF intervention `1.0`.
- All runs use independent TensorBoard directories under `results/jepa_safe_capture_v11_hard_replay_tensorboard/train_seed<seed>`.

## Audit gates

- [x] Checkpoint model type and training variant match.
- [x] All configured and recorded epochs are present and finite.
- [x] Checkpoint parameters are finite and seed-aligned.
- [x] TensorBoard scalar, text provenance, parameter histogram and gradient histogram tags are complete.
- [x] Source/config hashes match the current workspace.
- [x] Train/validation/calibration episode seeds are disjoint.
- [x] `locked_test_opened=false` for all runs.

## Interpretation

This proves that the hard-context weighted checkpoints are reproducible and auditable on the held-out training contract. It does not prove closed-loop `safe_capture` improvement. The next authorized evidence is held-out prediction/auxiliary-head validation, followed by calibration-bound ledger and CBF-filtered development smoke.
