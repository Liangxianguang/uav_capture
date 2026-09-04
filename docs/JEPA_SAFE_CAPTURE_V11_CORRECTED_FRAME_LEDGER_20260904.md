# v11 corrected-frame Reliability Ledger 构建报告

**日期：** 2026-09-04
**阶段：** P4 - 三 seed 离线 ledger calibration
**设备：** NVIDIA GeForce RTX 5050，CUDA 12.8，PyTorch 2.7.1+cu128
**边界：** `development_only=true`，`locked_test_opened=false`
**主指标边界：** 本报告只证明 ledger 的校准、绑定和拒答审计通过，不证明 safe-capture 控制收益。

## 1. 输入与绑定

| 输入 | 值 |
|---|---|
| corrected calibration archive | `results/jepa_safe_capture_v2_p1_corrected_frame_calibration/counterfactual_safe_capture_v2.npz` |
| calibration metadata SHA-256 | `531ce966d78cc448df4868bc071e507fa64bc9a7b1ee0d121ad367bba20ec6f0` |
| calibration dataset SHA-256 | `ea04eec8e255bcafa95386ef4c30e366e55723334b8d4985d6c94887b9a1a307` |
| v11 protocol SHA-256 | `d47e6e006ffad217f363c5b7f45163a7248a8034059e82e81df6269ff3276985` |
| corrected frame | `target_relative_frame=post_action_defender_position` |
| frame revision | `label_frame_correction_version=1` |
| policy | minimum samples `128`，minimum credit `0.65`，stale limit `45` steps |

## 2. 三 seed 产物

| training seed | checkpoint SHA-256 | ledger SHA-256 | entries | state counts |
|---:|---|---|---:|---|
| `20260911` | `e638c5868a0e6047ad1cefb903973e0979ab5f47bde8b9a44889eb79775aa8d4` | `f43ea0ab57b58d131cf9f1235539cf1cd325aaf46c4b5374b0a5361348bc87c1` | 1068 | trusted 296346; fallback 15974 |
| `20260912` | `f3390bd321d6b9155570f8e8f47f4c072cd4c24c7e3c2afda61f00a34a5948a2` | `4ba2af5497e80869b23f79305babd6a9760ad6c7d1ab44e1d61f399202a3b7da` | 1187 | trusted 292282; fallback 20038 |
| `20260913` | `c545915ab8540e468a6d863687677a9e059c1002aae8c17fa7e320f1c59052b5` | `d8c25ad788ae4078afd5d67d90afba9627b9e03e2bcb8975de69d32acaf94b42` | 1296 | trusted 291788; fallback 20532 |

Aggregate JSON SHA-256：`0ea29dffac38fd549edb34944f0df2ac5b1bde5d089704f3223ec8ae55d08736`。

## 3. Gate 结果

- `all_ledgers_runtime_valid=true`
- `all_fallback_audits_pass=true`
- `high_credit_failure_rate_not_above_low_credit=true`
- `ood_stale_nonfinite_fallback_100_percent=true`
- `eligible_for_closed_loop_smoke=true`
- `eligible_for_locked_test=false`
- 三份 ledger 的 `trusted` 与 `fallback_nominal` 离线 unsafe rate 均为 `0.0`。
- 内置 OOD、stale、non-finite fault 均返回 `safe_hold`，reason code 分别为 `ood`、`stale_observation`、`non_finite_context`。

这证明 ledger 是 checkpoint/protocol/calibration hash 绑定的只读可信度路由器；CBF 仍是唯一最终安全执行边界，ledger 不能被解释成安全证书。

## 4. TensorBoard

每个 seed 和 aggregate 都使用独立目录并生成 event file：

```text
results/jepa_safe_capture_v5_v11_corrected_frame_tensorboard/ledger_seed20260911/
results/jepa_safe_capture_v5_v11_corrected_frame_tensorboard/ledger_seed20260912/
results/jepa_safe_capture_v5_v11_corrected_frame_tensorboard/ledger_seed20260913/
results/jepa_safe_capture_v5_v11_corrected_frame_tensorboard/ledger_aggregate/
```

每个 seed 记录 17 个 scalar、4 个 text tags，并包含 fallback gate；aggregate 记录配置、输入 provenance、四类 gate 和每个 horizon 的 calibration/reliability scalars。

## 5. 下一阶段

P4 已完成。下一步运行 P5 的 Joint CBF-QP fault matrix、CPU/RTX 5050 rolling-horizon deterministic replay 和 latency audit；通过后才进行三 seed M0/M3/A1/A2 20 集 paired smoke。任何阶段都保持 `locked_test_opened=false`，并以完整 episode 的 `safe_capture` 作为唯一任务指标。
