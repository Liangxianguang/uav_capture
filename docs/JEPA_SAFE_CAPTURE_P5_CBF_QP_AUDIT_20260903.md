# JEPA Safe-Capture v2 P5 联合 CBF-QP 审计

> Development-only safety-filter evidence. This report does not open a locked
> test and does not claim closed-loop capture improvement.

## 1. P5 目标

P5 将候选动作和 nominal 动作统一送入一个联合多机安全过滤器，显式处理：

- obstacle separation；
- defender-defender pairwise separation；
- world boundary 和 altitude；
- speed、acceleration 和 action slew；
- solver failure、infeasible、timeout、non-finite request；
- `safe_hold -> nominal through CBF -> controlled_abort` 回退路径。

JEPA、reliability ledger 和 candidate ranker 不在本 audit 中生成最终动作；所有未来闭环动作都必须经过该过滤器。

## 2. 实现和审计产物

| 产物 | 作用 |
|---|---|
| `src/encirclement3d/cbf_qp.py` | 联合决策向量、CBF 线性 barrier、运动学约束、SLSQP solve、fallback 和 diagnostics |
| `tests/test_cbf_qp.py` | 13 个单元测试，覆盖正常、障碍、box、pairwise、边界/高度、运动学、异常和 timeout |
| `scripts/audit_jepa_safe_capture_v2_cbf_qp.py` | 真实环境公开观测上的多 case deterministic audit 和 TensorBoard 写入 |
| `results/jepa_safe_capture_v2_p5_cbf_qp_audit_rerun_seed20260911/audit.json` | seed 20260911 审计结果 |
| `results/jepa_safe_capture_v2_p5_cbf_qp_audit_seed20260912/audit.json` | seed 20260912 审计结果 |
| `results/jepa_safe_capture_v2_p5_cbf_qp_audit_seed20260913/audit.json` | seed 20260913 审计结果 |

当前环境没有 OSQP/CVXPy，因此实现使用 SciPy SLSQP：二次动作偏差目标、联合线性 CBF rows，以及 speed/acceleration 的凸 norm constraints。solver 版本和环境信息全部写入每次 audit 的 provenance。

## 3. 单元测试结果

```text
13 passed
```

覆盖内容：

1. 正常联合可行 solve 和 deterministic repeated solve；
2. cylinder obstacle constraint；
3. box/wall geometry 从公开 observation 解码；
4. pairwise separation 联合约束；
5. boundary lower 和 altitude lower；
6. speed、acceleration；
7. 当前物理状态已违规时不报告 `verified_feasible`；
8. non-finite request；
9. motion infeasibility；
10. fallback 的安全动作和 task diagnostic 分离；
11. timeout 可观测且不执行原始 request；
12. public observation geometry 优先于隐藏环境几何。

## 4. 三 seed deterministic audit

每个 seed 均运行 9 个 case 和 20 次重复正常 solve：

| Seed | Cases | All outputs finite | Failed request never raw-executed | Zero perturbation exact | Repeated deterministic | p95 solve latency |
|---:|---:|---:|---:|---:|---:|---:|
| 20260911 | 9 | `true` | `true` | `true` | `true` | `1.667 ms` |
| 20260912 | 9 | `true` | `true` | `true` | `true` | `1.516 ms` |
| 20260913 | 9 | `true` | `true` | `true` | `true` | `1.628 ms` |

每个 seed 的 case aggregate 均为：

- `case_fallback_count=3`；
- `case_infeasible_count=3`；
- `case_timeout_count=1`；
- `state_violation_count=0`；
- TensorBoard provenance：14 scalar tags、3 text tags，均完整。

三个修复后 audit JSON 的 SHA-256：

```text
seed 20260911: a9dfd110f783c1331feb73085113fdfe331d389b6d37593c0ba0f5e08f651721
seed 20260912: f3dc788f4859578fe07eadea32fd755171271a3b8cea298036d6a79f46f0a648
seed 20260913: f50011e5a7c2ae59852b4d5debef9bd3066f8ee1d1df2a6d8dbd6eab2c364205
```

## 5. 关键行为证据

### 5.1 可行约束

正常、cylinder、box、pairwise、boundary 和 altitude case 均返回 finite action，并由 post-solve residual 验证约束。障碍、pairwise 和 altitude case 的 active constraint 名称会进入 diagnostics，能够追溯动作为何被修正。

### 5.2 失败和回退

- non-finite request：`nonfinite_request -> controlled_abort`；不把 NaN 送进 solver。
- current velocity 与 acceleration/speed 不相容：`solver_failure -> controlled_abort`；不执行原始 request。
- 强制极小 latency budget：`timeout -> controlled_abort`；timeout 状态和 primary status 均保留。
- controlled abort 返回 motion-limited emergency stop request，并明确 `verified_feasible=false`；它不是安全证明，调用方必须继续记录和处理该状态。

### 5.3 状态安全和信息边界

过滤器在求解导数 barrier 之前检查物理状态的 obstacle、pairwise 和 boundary clearance。已经发生的物理违规不会因为后续零动作而被伪装成可行。障碍几何从公开 observation 解码；target ground truth 没有进入 CBF-QP。

## 6. P5 gate 判定

| Gate | 判定 | 依据 |
|---|---|---|
| 约束覆盖 | PASS | obstacle、pairwise、boundary、altitude、speed、acceleration rows/constraints 存在 |
| 联合求解 | PASS | 单一 flattened decision vector 和共同 solver |
| 非 finite / infeasible | PASS | 13 tests + 3 seed audit，均显式 fallback |
| zero-perturbation | PASS | 三个 audit 均 `true` |
| deterministic repeated solve | PASS | 三个 audit 均 `true` |
| latency | PASS（开发预算） | p95 `1.52–1.67 ms`，低于 100 ms budget |
| TensorBoard provenance | PASS | 每个 seed 14 scalar + 3 text |

**P5 判定：PASS，允许进入 P6 paired development。**

该判定只说明安全过滤器接口、残差检查和 fallback 机制通过开发审计，不说明整个 JEPA 闭环已经提高 safe-capture，也不替代 P6 的三 seed paired episodes。

## 7. P5 保留限制

1. SLSQP 是通用 constrained optimizer，不等同于经过独立形式化证明的实时 QP solver；部署前仍应评估 OSQP/CVXPy 或专用 QP backend。
2. 本 audit 使用构造的真实环境状态和公开 geometry，不是 180 episode 的闭环任务结果。
3. `controlled_abort` 的动作在极端状态下可能仍有未满足的 measured residual；因此 diagnostics 明确为 unverified，不能作为安全成功计数。
4. 当前 P5 尚未证明 candidate ranker 与 CBF-QP 在随机混合障碍任务上的 safe-capture non-inferiority。

## 8. 下一阶段

进入 P6 前冻结：

- P2 三个 checkpoint 和 P3 三个 ledger hash；
- P4 K=5、3-step candidate contract；
- P5 solver、margin、tolerance、timeout 和 fallback 参数；
- M0--M3 主矩阵和 A1--A3 消融矩阵；
- paired episode seeds、scene manifest、TensorBoard namespace。

P6 先运行每个变体 20 paired smoke episodes；任何 collision、boundary、pairwise separation、QP fallback 静默失败或 zero-regression 失败都会停止该变体。P6 仍是 development-only，新的 locked test 保持关闭。

## 9. TensorBoard 路径

```text
results/jepa_safe_capture_v2_tensorboard/p5_cbf_qp_rerun_seed20260911
results/jepa_safe_capture_v2_tensorboard/p5_cbf_qp_seed20260912
results/jepa_safe_capture_v2_tensorboard/p5_cbf_qp_seed20260913
```

每个目录都包含：

- `Config/audit/text_summary`；
- `Provenance/sources/text_summary`；
- `Provenance/environment/text_summary`；
- CBF feasibility、fallback、slack、active constraint 和 latency scalars。
