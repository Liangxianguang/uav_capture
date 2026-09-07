# DN-MPC P1 解析规划器 Smoke 结果

**日期：** 2026-09-07  
**状态：** development-only；未开启 locked test

## 1. 目的和边界

本轮只验证 `public belief -> obstacle_route_v1 candidates -> DN-MPC score`
这一段解析链路。使用 synthetic 的 open、single cylinder、single wall 和
mixed 场景，生成 12 类候选并运行 `DistributedMinimaxMPC`。

本轮没有执行：

- simulator action；
- Joint CBF-QP；
- JEPA 推理；
- Reliability Ledger 或 Ledger-Lite；
- safe-capture episode 结算。

因此本报告不能证明碰撞率、安全捕获率或完整闭环提升。

## 2. 重放命令

```powershell
$env:PYTHONPATH = "$PWD/src"
python scripts/smoke_dn_mpc_route_selection.py `
  --config configs/dn_mpc_jepa_safe_capture_development.yaml `
  --output-dir results/dn_mpc_jepa_safe_capture_dev/p1_route_smoke_seed20260907_v3 `
  --tensorboard-dir results/dn_mpc_jepa_safe_capture_tensorboard/p1_route_smoke_seed20260907_v3 `
  --seed 20260907
```

输出包括：

- `summary.json`；
- `candidate_rows.jsonl`；
- `provenance.json`；
- TensorBoard event file。

## 3. 结果

| 场景 | 候选总数 | valid 候选 | 选中路线 |
| --- | ---: | ---: | --- |
| open | 12 | 8 | `safe_intercept` |
| single cylinder | 12 | 6 | `visibility_hold` |
| single wall | 12 | 6 | `visibility_hold` |
| mixed | 12 | 6 | `visibility_hold` |

所有场景都返回了 valid route decision，planner 没有出现空候选或 non-finite
内部异常。TensorBoard event 含 242 个 scalar tag，且 242 个 scalar value
全部为有限值，覆盖 candidate score、
worst-case escape、capture cost、formation cost、smoothness、validity、
selected route、route age 和 route switch。

## 4. 发现和修复

本轮 smoke 暴露并修复了两个候选合同问题：

1. 无障碍场景的 `radial_out` 和 `formation_split` 没有 fallback，导致完整
   12 候选生成失败；现在会生成 public-belief-only 候选。
2. route geometry 只检查 waypoint 段，未检查编队质心到第一个 waypoint 的
   起始段；现在 nominal/safe-intercept 直穿障碍的路线会被标为 invalid。

新增回归后相关测试为：

```text
30 passed
```

## 5. 结果解释

障碍场景中没有立即选左右切向，而是选 `visibility_hold`，并不等价于规划器
错误。当前 smoke 是单周期静态排序：`visibility_hold` 先向目标方向靠近，
不穿越障碍，短期 worst-case cost 比完整切向路线低。它说明下一版目标函数必须
显式建模：

- route phase（approach/pre-brake/tangent/encircle/intercept）；
- 障碍前沿的 stopping distance；
- 通过障碍后的 terminal progress；
- 连续多个 rolling cycles 的 route identity 和 side hold。

在这些项加入前，不应把 `visibility_hold` 的单周期选择解释为“已经学会绕障碍”，
也不应接入 JEPA 或正式评估。

## 6. P1 门判定

- [x] 12 类候选可生成或明确标记 invalid。
- [x] 候选经过 reachable dynamics projection。
- [x] planner 在 4 个 synthetic 场景返回 valid decision。
- [x] TensorBoard 和 provenance 可重放。
- [x] 无 simulator action、CBF、JEPA 或 Ledger 越权。
- [ ] 尚未证明 CBF 可行性。
- [ ] 尚未证明 safe capture 提升。

**P1 结论：通过解析 smoke，进入路线状态机和 G5 + CBF 集成前准备；不进入
JEPA 训练或 locked benchmark。**

## 7. 下一阶段

1. 把 route phase、terminal progress 和 obstacle-front stopping distance 加入
   DN-MPC score，并增加 multi-cycle synthetic replay。
2. 在固定 G5 4 场景接入 selected/nominal/safe-hold 三路独立 CBF counterfactual。
3. 只有 `safe_capture >= 4/4` 且 collision、boundary、pairwise、raw-unverified
   均为 0，才允许进入 L0-L1 或 JEPA 候选评价。
