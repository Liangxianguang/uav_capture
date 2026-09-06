# P1 最早 CBF 不可行诊断

**日期：** 2026-09-07
**状态：** `development-only`；`locked_test_opened=false`
**来源：** WP2 `obstacle_route_v1` validation smoke，episode `646002`

## 结论

该 controlled-abort 不是碰撞、边界越界或求解超时。第 14 步仍有 10 条几何有效且经 primary CBF 验证的路线，然而执行选择保持在 `nominal`；到第 15 步，候选仍有 10 条几何有效路线，但 selected、nominal、safe-hold 以及其余路线的 primary CBF 均无法通过。最早失败约束同时包含障碍物屏障和加速度可达性，说明系统在高速度接近障碍后才进入不可恢复区。

这项证据支持“提前刹车/提前切换绕行路线”的设计方向，但不支持降低 CBF margin、关闭 stale/OOD gate 或删除 `controlled_abort`。

## 前一周期与 abort 周期

| 指标 | 第 14 步（仍可行） | 第 15 步（abort） |
| --- | ---: | ---: |
| 几何有效路线 | 10 | 10 |
| primary CBF 接受路线 | 10 | 0 |
| selected route | `nominal:obstacle-4` | `nominal:obstacle-4` |
| 最小物理障碍净空 (m) | 1.7691 | 1.4560 |
| 最小 pairwise 净空 (m) | 0.6629 | 0.6281 |
| 最小边界净空 (m) | 2.9241 | 3.0171 |
| CBF fallback | `none` | `controlled_abort` |
| CBF 最小约束值 | 约 0 | -0.0791 |

第 14 步可行路线包括 `nominal`、左右/上绕行、`radial_out`、`formation_contract`、`braking`、`safe_intercept`、`visibility_hold` 和 `verified_safe_hold`。因此路线生成和几何 coverage 在该时刻不是空集，关键问题是没有在可行窗口内切换到减速或绕行路线。

## 最早失败约束

第 15 步 selected primary CBF 的负约束为：

- `obstacle_0_defender_1`
- `obstacle_1_defender_3`
- `acceleration_defender_1`

三路独立 counterfactual 的结果：

| probe | verified | 最小约束值 | 主要 active constraints |
| --- | :---: | ---: | --- |
| selected | 否 | -0.0678 | obstacle + acceleration |
| nominal | 否 | -0.0678 | obstacle + acceleration |
| safe-hold | 否 | -0.0424 | obstacle + acceleration |

没有 probe timeout；失败来自约束残差无法满足，而不是外部延迟或数值超时。

## 停止距离信号

以第 14 步执行速度和配置的 `a_max=6 m/s^2` 做诊断性估计，四架无人机速度约为 `4.50/4.78/4.43/4.45 m/s`，对应单机理想停止距离约为 `1.69/1.90/1.64/1.65 m`。第 15 步前，最小物理障碍净空已经从 `1.7691 m` 降到 `1.4560 m`；在 `strict_buffer` 合同下还要预留 `0.35 m` operational obstacle buffer，导致高速度减速与障碍屏障、加速度球同时激活。

该 stopping-distance 计算只用于失败归因和训练标签，不替代 CBF 证明，也不把物理净空误写成安全保证。

## 决策

本阶段停止扩大 route 多 seed、40/60 episode 和新 JEPA 训练。下一步只做：

1. 将第 14/15 步的 nominal-vs-detour、braking、safe-hold 反事实写入 train-only route-aware hard-negative archive；
2. 在候选协议中加入 stopping-distance、TTC、最早不可行 horizon 和加速度 slack 标签；
3. 重新建立 calibration archive 和 hash-bound Ledger 后，再进行小规模 JEPA 辅助头训练。

任何后续改动都必须保持 `strict_buffer` 历史基线、reachable projection、独立三路 CBF probe 和 `controlled_abort` 语义不变。

## TensorBoard 与 provenance

TensorBoard：

- `results/jepa_safe_capture_p1_earliest_cbf_abort_audit_20260907_tensorboard/`

已记录 `abort_step`、前后几何有效路线数、前后 CBF 接受数、abort 最小约束值、分类和源 trace SHA-256。查看：

```powershell
D:\download\anaconda3\envs\traj_pred_prep\Scripts\tensorboard.exe `
  --logdir results/jepa_safe_capture_p1_earliest_cbf_abort_audit_20260907_tensorboard
```

源 trace SHA-256：`a0176a03f03d2db5e4c0d222978785c492b5d72781743d7379c5cb3ffaf92c52`
源 summary SHA-256：`ca98e3415aa61df3a89ee26a81315c62f126f1eca2a47ad129c5220d7a1966e1`
源 scene manifest SHA-256：`766ef5d87fed73b0b942c1d1ab25ffcd29ce977550518c647f22f6c5f2470056`
审计脚本 SHA-256：`200bfb86f4db1866191e75b0bf3694ab1872d7eded0d94aca3c595f8b24de1c8`

## 产物

- `results/jepa_safe_capture_p1_earliest_cbf_abort_audit_20260907/audit.md`
- `results/jepa_safe_capture_p1_earliest_cbf_abort_audit_20260907/audit.json`
- `results/jepa_safe_capture_p1_earliest_cbf_abort_audit_20260907/step_audit.jsonl`
- `scripts/audit_wp2_route_abort.py`
