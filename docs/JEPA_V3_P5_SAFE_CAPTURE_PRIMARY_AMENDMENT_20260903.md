# JEPA-v3 P5 Safe-Capture-Primary Protocol Amendment

**日期：** 2026-09-03  
**适用阶段：** P5-E/F、P6、P7 的 development-only 评估  
**变更授权：** 用户明确指示“mean capture time 不是问题，最主要的是保证 safe capture”  
**性质：** post-smoke amendment；不追溯改写既有报告或把本次变更伪装为原始预注册

## 1. 变更原因

seed-11 P5 non-zero smoke 的 safe capture 为 `19/20`，高于 paired V5+CBF baseline 的 `18/20`，且 collision/boundary 均为零。原执行计划把平均 capture time 或 path 相对增加超过 `10%` 设为自动停止条件；在烟雾结果中，candidate 的成功样本数从 18 变为 19，新增的成功回合捕获时间为 `24.8 s`，使按“仅成功回合”计算的 aggregate mean capture time 达到 `6.7895 s`（baseline `6.1667 s`）。

用户随后明确将 safe capture 与 CBF 安全作为主目标，不把 mean capture time 作为自动拒绝理由。

## 2. 旧规则和新规则

| 项目 | 原 P5 执行规则 | 本次修订后 |
|---|---|---|
| 主终点 | safety/task 与效率并列 gate | **safe capture，在 CBF 安全不变量下为主终点** |
| collision/boundary | 自动停止 | 保持自动停止 |
| 场景/episode 配对、finite 输出、CBF 最后执行、provenance | 自动停止 | 保持自动停止 |
| capture time/path >10% | 自动停止 | 不再单独自动停止；完整报告 |
| clearance、CBF intervention、action changes、control latency | 报告 | 保持报告，并纳入失败分桶 |
| locked test | 不自动打开 | 保持关闭 |

## 3. 防止后验选择性解释

本次修订并不允许仅报告 safe capture：

1. 每个 seed 的 capture time、路径、minimum clearance、CBF correction/intervention、fallback、selected candidate 和 latency 仍全部报告；
2. 仍使用同一个 frozen S3 development block、同一 checkpoint/ledger 配对、同一 `K=5`、`0.10 m/s`、chunk=3 参数；
3. 不因此重新选择 seed、chunk length、candidate count、scorer 或 ledger 阈值；
4. P6 最终结论仍要求三 seed 一致，不能用 seed-11 smoke 单独宣称有效；
5. collision/boundary 或 zero-regression 失败仍立即拒绝该候选；
6. P7 若讨论新的 locked preregistration，必须同时预注册 safe capture 主终点和全部次要成本指标及其解释规则。

## 4. 影响范围

该修改只改变 development evaluation 的决策规则。它不改变 v2 train/validation archives、训练代码、三个 checkpoint、TensorBoard evidence、prediction gate、ledger、CBF、环境、frozen scenes、V4/V5 historical evidence 或任何 locked-test 状态。

P5 seed-11 smoke 因此以 `safe-capture-first` 规则通过安全准入，允许继续 60 episode diagnostic；原始 `10%` 成本结果仍保留在后续 P5 报告中。
