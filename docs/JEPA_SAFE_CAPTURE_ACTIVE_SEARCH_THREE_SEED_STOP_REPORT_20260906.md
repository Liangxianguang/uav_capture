# JEPA Safe Capture Active-Search 三 Seed 停止报告

**日期：** 2026-09-06
**状态：** development-only；`locked_test_opened=false`
**硬件：** NVIDIA GeForce RTX 5050
**主指标：** episode-level `safe_capture`
**决策：** 停止继续训练、增加数据和扩大 L1-L3；先做离线 route-ranking / stale-ledger 归因。

## 1. 本阶段做了什么

本阶段按照 [Next Master TODO Plan](JEPA_SAFE_CAPTURE_NEXT_MASTER_TODO_PLAN_20260906.md) 完成了受限的三训练 seed development smoke：

- 相同 actor、独立 route-identity JEPA、hash-bound Reliability Ledger、`obstacle_route_v1` 12 候选、reachable projection、三路独立 CBF probe；
- `strict_buffer` CBF，`anticipatory_horizon_steps=5`，margin 保持 `0.35 m`；
- M0 与 M3 使用完全相同的 scene manifest 做配对比较；
- 所有训练、校准和闭环评估均写入独立 TensorBoard；
- 额外用 `20260911` retry checkpoint 重建 Ledger 并在同一 manifest 上复跑，检查 checkpoint 差异是否是结果波动来源。

本阶段没有打开 locked test，没有使用在线目标真值，没有执行 raw-unverified action，也没有降低 stale/OOD gate 或删除 `controlled_abort`。

## 2. 配对结果

| training seed | M0 actor + CBF | M3 JEPA + Ledger + route + CBF | paired delta | M3 终止情况 | M3 target-boundary diagnostic |
|---|---:|---:|---:|---|---:|
| `20260911` | `2/3` (`66.7%`) | `2/3` (`66.7%`) | `0.0 pp` | `1` controlled abort + `2` safe capture | `2/3` |
| `20260912` | `2/3` (`66.7%`) | `1/3` (`33.3%`) | `-33.3 pp` | `2` timeout + `1` safe capture | `0/3` |
| `20260913` | `2/3` (`66.7%`) | `0/3` (`0.0%`) | `-66.7 pp` | `3` timeout | `0/3` |
| **pooled** | **`6/9` (`66.7%`)** | **`3/9` (`33.3%`)** | **`-33.3 pp`** | **5 timeout + 1 controlled abort** | **`2/9`** |

因此本阶段没有证明 M3 的闭环控制增益。按计划书的结论标签，本阶段应标记为：

> `prediction_signal_no_control_gain`

### Checkpoint-swap 复核

`20260911` 的首次 checkpoint 和 retry checkpoint SHA-256 不同：

- 首次：`d35759ede6afa3aa205bafa64c30533fb83b3279e8203b3292a4c03c0decab9a`；
- retry：`7b2e85c8fc6822880d22d8916daa396b4ee7dac657346fd85953f06aa51d4a0f`。

retry checkpoint 在同一 `scene_manifest_sha256=d6e0e3b2dd2893e1ece8e3557e85bd88741c06d6d2f56816907661afd0ba4996` 上仍为 `2/3 safe_capture`，与首次 M3 完全一致，且同样为 `1` controlled abort、`0` UAV 安全违规。因此当前负增益不是 TensorBoard hparams 兼容修复或 checkpoint 序列化差异造成的。

## 3. 安全结果

18 个 M0/M3 episode 的 UAV 硬安全门均通过：

| gate | count |
|---|---:|
| UAV collision | `0` |
| UAV boundary violation | `0` |
| pairwise violation | `0` |
| raw-unverified execution | `0` |

M3 的 `target_boundary_violation=2/9` 是任务结算诊断，不是 UAV 安全违规；它说明当前目标逃逸/任务边界合同仍可能影响能力指标，后续必须继续单独报告，不能并入 UAV 安全结论。

## 4. 失败归因

当前证据把问题定位在“闭环捕获机会没有增加”，而不是 CBF 物理安全失效：

1. **候选 CBF 可行性不是主要瓶颈。** M3 各 seed 的 route prefilter 几乎全部接受：`2270/2277`、`5705/5705`、`7318/7318`；独立 selected/nominal/safe-hold probes 也没有普遍拒绝。
2. **seed11 的单次失败是晚期联合约束不可行。** 该 episode 触发一次 `cbf_controlled_abort`；trace 中存在 `nominal_infeasible`，但没有碰撞或越界。
3. **seed12/13 的主要失败是 timeout，而非 CBF abort。** M3 反复选择 `right_detour`、`left_detour`、`formation_split`、`safe_intercept` 等路线，却没有转化为足够的目标接近进度；M0 在完全相同 manifest 上各捕获 `2/3`。
4. **stale / reacquisition / rank fallback 仍明显。** 失败 episode 的 trace 中出现 `cautious_reacquisition`、`stale_observation`、`nominal_infeasible` 和 `nominal_anchor_tie`；`rank_fallback_steps` 在 seed13 三个 episode 分别为 `35/21/10`。这说明 Ledger 和 route ranker 需要区分“信息过期”“候选排序错误”和“确实无可行路线”，不能把它们都路由成保守动作。
5. **辅助头的离线识别正确不等于控制增益。** 当前 route identity/side 头可以在校准数据上给出很高的识别率，但闭环结果为 `33.3%`，所以本阶段只能证明预测信号存在，不能证明它改善了 safe capture。

## 5. TensorBoard 与 provenance

训练 TensorBoard：

- `results/jepa_route_identity_model_active_search_seed20260911_tb`
- `results/jepa_route_identity_model_active_search_seed20260911_retry_tb`
- `results/jepa_route_identity_model_active_search_seed20260912_tb`
- `results/jepa_route_identity_model_active_search_seed20260913_tb`

校准 TensorBoard：

- `results/jepa_route_identity_ledger_active_search_v4_seed20260911_tb`
- `results/jepa_route_identity_ledger_active_search_v4_seed20260911_retry_tb`
- `results/jepa_route_identity_ledger_active_search_v4_seed20260912_tb`
- `results/jepa_route_identity_ledger_active_search_v4_seed20260913_tb`

闭环 TensorBoard：

- `results/wp1_active_search_v4_m0_seed20260911_current_tensorboard`
- `results/wp1_active_search_v4_m3_seed20260911_tensorboard`
- `results/wp1_active_search_v4_m3_seed20260911_retry_tensorboard`
- `results/wp1_active_search_v4_m0_seed20260912_tensorboard`
- `results/wp1_active_search_v4_m3_seed20260912_tensorboard`
- `results/wp1_active_search_v4_m0_seed20260913_tensorboard`
- `results/wp1_active_search_v4_m3_seed20260913_tensorboard`

每个目录均包含 event file，评估 provenance 中 `required_provenance=true`。训练均使用 RTX 5050 的 CUDA 环境；TensorBoard 2.4.1/NumPy 2.x/protobuf 兼容性通过 `scripts/run_with_tensorboard_compat.py` 处理。

共同合同哈希：

- protocol：`5fc57476361b016f474600900de1887849a853b30ae776f3b51476be7041e119`；
- environment：`42bd4e158c5e314e0ece6add8038b32c384a7a2ca027e9387327656fccf751ad`；
- actor：`535098773be05687e147043435649378532362d479bdc0375842970370ba40ba`；
- JEPA checkpoint：seed11 `d35759ede6afa3aa205bafa64c30533fb83b3279e8203b3292a4c03c0decab9a`，seed12 `1f0b169d0b845d0c1591abba5e17b7fb520818c8bdbc0c2dc3bd0f7d48891f1a`，seed13 `a97820c8b2ed84361fe4979ca8830edfeaf79080fa8bfe4b373586c6b069bf28`；
- scene manifest：seed11 `d6e0e3b2dd2893e1ece8e3557e85bd88741c06d6d2f56816907661afd0ba4996`，seed12 `130f9602a87bcc392b10b2fa8bddaeaa88dab21bae4d932f95ddfeb7647da644`，seed13 `41b0b4db4040ea36a110e29fe1070242bbe26325e4a755632a97ca03c7ba1b48`。

## 6. 停止决定和下一步

本阶段触发计划书的停止规则：M3 没有稳定的非负 paired delta，且 pooled safe capture 比 M0 低 `33.3 pp`。因此现在不再：

- 继续训练更多 epoch 或更多随机 seed；
- 继续扩大数据集、L1-L3 或 S3 block；
- 调低 CBF margin、关闭 stale/OOD、删除 controlled abort；
- 把 target-boundary diagnostic 混入 UAV safety 结论。

下一步只做不改安全合同的离线归因：

1. 对 9 个 M3 episode 做 settled counterfactual route-regret audit：比较 JEPA 选中的 route 与离线真实执行标签，统计 `jepa_wrong_route`、`ledger_over_abstention`、`stale_or_ood` 和 `controlled_abort_no_verified_action`。
2. 单独统计 route candidate coverage、route progress、目标可见性恢复后的有效位移和 nominal anchor tie；先判断是 rank direction、candidate geometry 还是 Ledger 过度拒绝。
3. 在固定 manifest 上做小规模 replay 验证修复，只有当同一合同下 M3 至少达到 M0 且安全硬门继续全为 0，才允许重新生成困难负样本并开始下一轮训练。

本报告不构成 V4/V5 locked benchmark 结论；它只封存当前三 seed development smoke 的失败证据和后续准入条件。
