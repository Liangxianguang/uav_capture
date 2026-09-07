# Action-Conditioned Chunk Pairwise Audit

**日期：** 2026-09-07  
**状态：** development-only offline diagnostic  
**Locked test：** not opened

## 1. 目的与边界

在已有 action-conditioned pairwise audit 之后，补充三步、已投影候选动作块的
分段几何特征。每个 `0.1 s` 步使用候选动作块对应的 focal velocity，超过动作块
长度后保持最后一步；队友速度由当前公开相对速度和 focal 当前速度重建并保持不变。

该审计只使用 archive 中的公开 observation、candidate action chunk 和离线
pairwise TTC label。它不执行动作、不改变 CBF margin、不关闭 stale/OOD/non-finite
gate、不训练 JEPA、不建立 Ledger-Lite，也不打开 locked test。

## 2. 运行与 TensorBoard

输入仍是 seed-disjoint 的 train、validation、calibration archive：

- train：`results/jepa_safe_capture_p3_hard_negative_archive_train_smoke4_v2_20260907`
- validation：`results/jepa_safe_capture_p3_hard_negative_archive_validation_smoke4_v2_20260907`
- calibration：`results/jepa_route_identity_hard_negative_actor_calibration20_seed661606`

输出：

- JSON：`results/jepa_safe_capture_pairwise_action_conditioned_audit_20260907_v4.json`
- TensorBoard：`results/jepa_safe_capture_pairwise_action_conditioned_audit_20260907_v4_tensorboard/`

环境为 `D:\download\anaconda3\envs\traj_pred_prep`，Python 3.11.14，
PyTorch 2.9.1+cu130，CUDA 可用，TensorBoard 2.19.0。

## 3. Calibration 结果

风险标签仍为未来最小 pairwise TTC `<= 1.0 s`。阈值只用于离线 F1 报告，
不是部署安全阈值。

| 特征 | AUC | precision | recall |
|---|---:|---:|---:|
| candidate min distance, 1.0 s | 0.8246 | 0.5426 | 0.7555 |
| candidate chunk min distance, 1.0 s | 0.7649 | 0.4499 | 0.8145 |
| candidate chunk endpoint distance, 1.0 s | 0.7530 | 0.4235 | 0.8846 |
| candidate chunk min distance, 0.5 s | 0.7544 | 0.4572 | 0.7640 |
| candidate chunk endpoint distance, 0.5 s | 0.7667 | 0.4495 | 0.8137 |

没有一个候选特征同时满足 `recall >= 0.80` 和 `precision >= 0.50`。原来的
1.0 s 几何特征仍然是最佳候选，说明动作块信息能提高覆盖，但也带来了较多
false positive；仅延长投影不能解决 calibration tail。

## 4. 决策

- [x] 动作块分段投影是有限、可重现的。
- [x] TensorBoard 保存了 v4 运行的 provenance、分割统计、AUC 和 gate 状态。
- [x] CBF、stale/OOD/non-finite 和 raw-unverified 合同保持不变。
- [ ] pairwise recall/precision gate 通过。
- [ ] full JEPA training 获授权。
- [ ] Ledger-Lite calibration 获授权。
- [ ] closed-loop JEPA evaluation 获授权。

因此继续执行停止规则：不扩大网络、不创建新 Ledger、不运行完整 JEPA 闭环。
下一步应采集新的、场景 seed 独立的 calibration tail，专门区分 pairwise closing、
安全的 formation contract/expand 和 topology false positive，并保留每个动作块的
projection/CBF provenance。若新 archive 仍不能同时达到门槛，应回到标签定义和
公共观测契约诊断，而不是放宽 CBF。
