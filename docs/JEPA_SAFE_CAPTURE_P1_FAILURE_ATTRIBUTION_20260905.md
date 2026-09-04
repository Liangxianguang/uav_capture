# v11 Corrected-Frame P1 Failure Attribution

**日期：** 2026-09-05  
**范围：** corrected-frame P6 smoke，`development_only=true`，`locked_test_opened=false`  
**输入：** 12 个运行（3 seed x M0/M3/A1/A2），240 个完整 episode  
**工具：** `scripts/index_jepa_safe_capture_v11_failures.py`

## 1. 结果摘要

failure index 原始产物：

- `failure_index.json` SHA-256：`af5cf512ddc646e05b606e6385bf6b2721071c9d964b0e552ef40af4e578d6b3`
- `failure_index.csv` SHA-256：`afc6cfa7ddc6cc85a69fc62c29bca12f293c5e7de2ca71501782340ff3b1178f`
- `report.md` SHA-256：`f7c70539d96765d4169a4a8fd224ba88d65c79e08a4eec9d996f6e6c1a85bae3`

| 项目 | 结果 |
|---|---:|
| 总 episode | 240 |
| safe capture | 119/240 = 49.6% |
| 失败 episode | 121 |
| primary `cbf_controlled_abort` | 119 |
| primary `timeout` | 2 |
| collision / boundary / pairwise | 0 / 0 / 0 |
| raw-unverified execution | 0 |

主要结论：失败几乎全部以显式 CBF controlled abort 结束，未发现 raw desired action 绕过安全过滤。当前首要问题是候选选择、ledger fallback/abstention 与 CBF 可行域之间的组合导致任务推进不足，而不是安全约束被放行。

## 2. 按变体归因

| 变体 | safe capture | 失败 | CBF-abort episode | high-credit failure | fallback episode | mean candidate switch rate |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 30/60 (50.0%) | 30 | 30 | 0 | 0 | 0.000 |
| M3 | 28/60 (46.7%) | 32 | 31 | 32 | 27 | 0.107 |
| A1 | 31/60 (51.7%) | 29 | 28 | 29 | 0 | 0.127 |
| A2 | 30/60 (50.0%) | 30 | 30 | 30 | 24 | 0.116 |

M3 比 M0 多 1 个 CBF-abort episode、少 2 个 safe captures；A1 去掉 ledger 后 safe capture 略高，但仍不能形成因果结论。A2 与 M3 的 safe capture 相同，说明本 smoke 中 clearance/visibility auxiliary ranking 尚未显示闭环收益。

## 3. 观测和困难条件

| observation condition | episodes | safe capture | failures |
|---|---:|---:|---:|
| nominal | 156 | 46.2% | 84 |
| delayed_noisy | 84 | 56.0% | 37 |

失败 episode 的诊断标签计数如下（一个 episode 可以有多个标签）：

| 标签 | episode 数 |
|---|---:|
| stale observation | 121 |
| CBF controlled abort | 119 |
| CBF infeasible/unverified | 119 |
| high-credit failure | 91 |
| visibility degraded | 89 |
| low-credit/nominal fallback | 51 |
| candidate capture regression | 14 |
| candidate oscillation | 2 |
| timeout | 2 |

`stale_observation` 在当前 trace summarizer 中表示观测或消息年龄超过诊断阈值；它是风险信号，不等于每个失败的充分原因。`high_credit_failure` 只表示该失败 episode 曾有 trusted decision，不能解释为 ledger 失效。

## 4. 与 settled ranking 的联合证据

M3/A2 settled counterfactual 显示：

- seed `20260911` 的 selected-not-settled-best 约 `7.4%--7.8%`；
- seed `20260912` 上升到约 `30.5%--34.6%`；
- seed `20260913` 上升到约 `38.1%--38.7%`；
- 六个 run 的 Spearman 均为负（约 `-0.35` 到 `-0.62`）。

这与 M3/A2 在后两个 seed 的候选切换、abstention 和 safe-capture 退化同时出现，支持“预测排序没有稳定转化为动作选择”的工程假设。但 settled replay 是 local 3-step counterfactual，不是完整 episode 因果证明；不能直接据此声称某个权重或阈值应该改变。

## 5. 当前阻塞点与下一步

### 5.1 不能做的事

- 不打开 locked test；
- 不把 controlled abort 从失败中删除；
- 不放宽 CBF margin、stale age 或 ledger threshold 来追逐 safe capture；
- 不使用同一 validation smoke block 反复调参；
- 不把 target drift 写成已验证结论，因为在线 trace 没有 offline future target labels。

### 5.2 下一阶段动作

1. 从 train split 生成新的 hard-replay counterfactual archive，覆盖急转、速度突变、遮挡、消息延迟、候选 separation 消失和 CBF correction 增大。
2. 增加 `net_clearance`、`visibility`、`TTC`、`CBF_intervention` 多任务安全头，并输出可校准的 uncertainty/lower bound。
3. 以安全 lower bound 做词典序过滤，保留 nominal anchor；ledger 非 trusted 或 margin 不足时只允许 nominal/hold。
4. 在离线 train/calibration gate 通过后，为每个 revision 生成新的 checkpoint-bound ledger 和 protocol，再做三 seed 20 集 smoke。
5. 新 smoke 只有在安全硬门和 evidence coverage 通过、且 M3 不再系统性低于 M0 时，才扩大到 40/60 集 paired development。

P1 结论：当前架构的安全执行边界成立，但控制收益瓶颈已被定位为 **ranking/abstention/可行候选覆盖不足 + 预测安全信号校准不足**。下一步应优先修复这些机制，不应更换为更大的世界模型或绕过 CBF。
