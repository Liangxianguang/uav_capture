# DN-MPC P4 Boundary-Rescue Replay

**日期：** 2026-09-07  
**状态：** development-only；locked test remains closed  
**代码 revision：** `7c57346d7942daa27549d16c591912af19403936`  
**硬件：** NVIDIA GeForce RTX 5050

## 1. 目的和边界

本阶段针对 P3 独立布局回放中 g3 的一个边界方向 controlled abort，增加一个
显式、可选的 `boundary_rescue` 候选。该候选只使用公开的世界边界和 defender
位置，先经过 reachable-dynamics projection、候选级 CBF probe 和最终 Joint
CBF，再执行首步动作。它不改变 CBF margin，不关闭 stale/OOD/non-finite
gate，不删除 `controlled_abort`，也不执行 raw-unverified action。

本阶段仍然不接入 JEPA 或 Ledger-Lite，不能解释为学习模型提升，也不能替代
V4/V5 历史结果。

## 2. 实现

- `obstacle_route_candidates.py` 增加 opt-in `boundary_rescue` route。
- `dn_mpc.py` 在公开 boundary clearance 低于阈值时允许 rescue route 越过普通
  route hysteresis 优先级。
- `evaluate_dn_mpc_cbf_g5.py` 增加 rescue 开关、trigger 和 inward offset 参数，
  并写入 provenance/TensorBoard metadata。
- 新增 route 方向、启用条件和 planner 优先级回归测试。

默认配置仍关闭 rescue，因此历史 v7 合同行为保持兼容。正式本阶段配置为：

| 参数 | 值 |
|---|---:|
| planner horizon | 5 steps |
| CBF horizon / route probe horizon | 3 / 3 steps |
| CBF mode | `strict_buffer` |
| obstacle / pairwise / boundary margin | `0.35 m` |
| rescue enabled | `true` |
| rescue trigger | `3.0 m` |
| rescue inward offset | `2.0 m` |
| execute/replan | first step only |
| JEPA / Ledger-Lite | disabled |

## 3. 阈值筛选（同一 g3 manifest）

低阈值实验只用于选择开发配置，使用同一 g3 scene manifest，不改变场景或
checkpoint：

| trigger | safe capture | controlled abort | fallback | route switches | mean capture |
|---:|---:|---:|---:|---:|---:|
| 2.0 m | 3/4 | 1 | 1 | 82 | 12.30 s |
| 2.5 m | 3/4 | 1 | 1 | 80 | 12.43 s |
| **3.0 m** | **4/4** | **0** | **0** | **85** | **12.60 s** |

三种配置的 collision、defender boundary、pairwise 和 raw-unverified 均为 0。
较低阈值没有修复边界方向失败，因此没有继续进行无依据的参数网格搜索。

## 4. 提交后最终三组回放

三个独立 development scene manifest 使用相同 actor、环境、严格 CBF 合同和
rescue 参数；每组 4 episodes，共 12 episodes。

| 组别 | safe capture | collision | defender boundary | pairwise | raw-unverified | controlled abort | route switches | mean capture |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| g0 reference | 4/4 (100%) | 0 | 0 | 0 | 0 | 0 | 70 | 16.875 s |
| g2 independent | 4/4 (100%) | 0 | 0 | 0 | 0 | 0 | 121 | 18.575 s |
| g3 independent | 4/4 (100%) | 0 | 0 | 0 | 0 | 0 | 85 | 12.600 s |
| **combined** | **12/12 (100%)** | **0** | **0** | **0** | **0** | **0** | **276** | **16.017 s** |

最小物理净空仍高于合同要求；三组 target-boundary diagnostic 合计为 2 次
（g0 1、g3 1）。该指标不属于 defender safety gate，但已保留在 summary 和
TensorBoard 中。最终三组 provenance 的 `git_revision` 均为
`7c57346d7942daa27549d16c591912af19403936`。

## 5. TensorBoard 和可重放证据

最终运行目录：

- `results/dn_mpc_cbf_g5_dev_seed20260907_g0_boundary_rescue_v3/`
- `results/dn_mpc_cbf_g5_dev_seed20260907_g2_boundary_rescue_v2/`
- `results/dn_mpc_cbf_g5_dev_seed20260907_g3_boundary_rescue_v2/`

对应 TensorBoard 目录各有一个 event file，包含 20 个 scalar tags，包括：

- `Aggregate/safe_capture_rate`、`Aggregate/controlled_abort_steps`、
  `Aggregate/route_switch_steps`；
- `Aggregate/collision_count`、`Aggregate/boundary_violation_count`、
  `Aggregate/pairwise_violation_count`、`Aggregate/raw_unverified_executed_steps`；
- `Episode/safe_capture`、`Episode/route_switch_steps`、
  `Episode/worst_min_clearance_m`；
- `Provenance/metadata/text_summary`。

TensorBoard event、`summary.json`、`provenance.json` 和 episode step traces 均保留
在本地 `results/`，未提交到 Git。

## 6. 测试和提交

相关回归测试：

```text
25 passed
```

此外，修改文件通过 `py_compile` 和 `git diff --check`。阶段代码提交为：

```text
7c57346 feat(dn-mpc): add opt-in boundary rescue route
```

推送到 GitHub 时网络无法连接 `github.com:443`，因此当前状态是本地提交已完成、
远端尚未确认。没有使用 force push，也没有修改用户已有的其他未提交文件。

## 7. 结论和下一步门槛

在 12 个独立 development episodes 上，boundary-rescue 将此前 g3 的
`3/4`（1 次合法 controlled abort）恢复为 `4/4`，且安全硬门保持为 0。代价是
route switches 上升，combined mean capture 为 16.017 s，因此该结果证明的是
“候选覆盖/边界可行性修复有效”，不是已经证明最优路线或完整系统泛化。

按主计划，下一步可以进入 JEPA evaluator 接入前的接口审计，但必须继续保持：

1. 新的 JEPA 只做候选轨迹评价，不直接输出动作；
2. 新 checkpoint 使用独立 hard-negative archive 和 calibration archive；
3. JEPA 接入先在同一 G5 manifest 做 paired replay，不能直接扩大 L1-L3；
4. 若 safe capture 下降、controlled abort 增加或出现任一安全硬门违规，立即回放
   trace 并停在解析 DN-MPC + CBF。

