# WP1：Non-finite JEPA -> Safe-hold 阶段报告

**日期：** 2026-09-05
**范围：** development-only fault audit
**locked 状态：** `locked_test_opened=false`
**主指标边界：** 本阶段不是 safe-capture 性能实验，不打开 locked split

## 1. 目标

验证 action-conditioned JEPA 的 clearance、uncertainty 或 auxiliary head 出现 NaN/Inf 时：

1. ranker 不抛出未处理异常；
2. 所有候选变为不可选，并返回 `execution_mode="safe_hold"`；
3. reason code 固定为 `non_finite_prediction`；
4. 输出请求经过现有 Joint CBF-QP；
5. 不执行 raw/unverified action；
6. trace 中的非 finite 数值以 JSON `null` 保存，而不是污染结果文件。

## 2. 实现变更

- `src/encirclement3d/jepa_safe_capture_ranker.py`
  - 捕获底层 prediction history 的明确 non-finite 异常；
  - 检查 target displacement、uncertainty 和全部 auxiliary heads 的 finite 状态；
  - 新增统一 safe-hold result/trace 构造；
  - 增加 `prediction_fault_fields` 观测字段；
  - 将 `SafeCaptureRankingTrace.as_dict()` 中的 NaN/Inf 序列化为 `null`。
- `tests/test_jepa_safe_capture_candidates.py`
  - 新增 NaN clearance、Inf uncertainty、NaN auxiliary 和底层 raised non-finite 四类测试。
- `scripts/audit_jepa_safe_capture_v21_nonfinite_safe_hold.py`
  - 使用真实 `CaptureRadiusPursuit3DEnv` 和 `JointCBFQPSafetyFilter` 做端到端故障注入；
  - 生成 JSON、Markdown、hash manifest 和 TensorBoard。

## 3. Fault audit 结果

权威结果目录：

```text
results/jepa_safe_capture_v21_nonfinite_safe_hold_fault_audit_v2/
results/jepa_safe_capture_v21_tensorboard/wp1_nonfinite_safe_hold_v2/
```

| 注入故障 | 记录字段 | rank mode | CBF fallback | CBF verified | raw/unverified | 结果 |
|---|---|---|---:|---:|---:|---:|
| NaN clearance | `obstacle_clearance_lower_quantile` | `safe_hold` | `safe_hold` | true | false | PASS |
| Inf uncertainty | `target_uncertainty` | `safe_hold` | `safe_hold` | true | false | PASS |
| NaN auxiliary | `target_visibility_logit` | `safe_hold` | `safe_hold` | true | false | PASS |
| raised non-finite | `prediction_output` | `safe_hold` | `safe_hold` | true | false | PASS |

四个 case 的共同 gates：

- `all_cases_pass=true`
- `all_route_to_safe_hold=true`
- `all_prediction_fault_reasons_explicit=true`
- `all_actions_finite=true`
- `raw_unverified_executed_count_zero=true`
- `all_cbf_fallbacks_explicit=true`

TensorBoard 已验证以下 tags 存在：

```text
Config/fault_audit/text_summary
Provenance/report/text_summary
Gates/status/text_summary
Ranker/safe_hold
Ranker/raw_unverified
```

## 4. 回归验证

```text
tests/test_jepa_safe_capture_candidates.py
tests/test_jepa_safe_capture_v2_reliability.py
=> 34 passed

tests/test_jepa_safe_capture_monotonic_score_suite.py
tests/test_jepa_safe_capture_protocol.py
tests/test_jepa_safe_capture_v2_paired.py
tests/test_jepa_safe_capture_latency.py
=> 35 passed

scripts/audit_jepa_safe_capture_fault_injection.py
=> all CBF/ledger gates passed; raw_unverified=0
```

## 5. 结论和边界

WP1 证明了非 finite prediction 不再通过异常退出或隐式候选路径绕过安全合同；它们会进入显式 safe-hold，并由同一个 Joint CBF-QP 验证。该结果只证明 fault routing 和可审计性，不证明 JEPA 的 safe-capture 控制收益。

下一阶段进入 WP2：固定点比较、CPU/CUDA candidate order 一致性、candidate separation 和 frozen settled replay。WP2 通过前，不扩大到 40/60 集，不训练更大的模型，不打开 locked test。
