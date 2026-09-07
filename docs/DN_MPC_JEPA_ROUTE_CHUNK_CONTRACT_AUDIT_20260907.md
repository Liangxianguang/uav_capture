# DN-MPC + JEPA 路线块合同审计

本报告是 `development-only` 审计，未开启 locked test。它解决一个必须先澄清的比较问题：历史解析 DN-MPC G5 与新的 JEPA paired replay 使用了不同的可执行路线块长度。

## 结果

| 运行 | route chunk | 方法 | safe capture | 平均捕获时间 | 路线切换 | 安全硬事件 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `dn_mpc_cbf_contract_audit_seed20260907_chunk5` | 5 | DN-MPC + CBF | 4/4 = 100% | 17.1 s | 75 | 0 |
| `dn_mpc_jepa_cbf_g5_dev_seed20260907_v4` baseline | 3 | DN-MPC + CBF | 2/4 = 50% | 16.1 s | 56 | 0 |
| `dn_mpc_jepa_cbf_g5_dev_seed20260907_v4` JEPA | 3 | DN-MPC + JEPA + CBF | 2/4 = 50% | 18.1 s | 98 | 0 |

三组使用同一个 G5 scene manifest，SHA-256 为
`6ef41f43d4b37e894559a4ee78604d1f3dc0c08fe48a069e7b4140ad646d327b`，并使用相同 actor、环境和严格 CBF 参数。安全硬事件包括 collision、defender boundary、pairwise 和 raw-unverified，均为 0。

## 解释

1. 5 步解析合同复现了历史 `4/4` 结果；因此历史 G5 结果本身没有因为当前代码回归而消失。
2. JEPA checkpoint 的模型合同是 `route_chunk_length=3`，新的 paired evaluator 因此使用 3 步候选块。3 步合同下，解析 baseline 与 JEPA 都为 `2/4`，所以不能把 `2/4` 的下降归因于 JEPA 单独造成。
3. 在相同 3 步合同内，JEPA 没有提升 safe capture，且路线切换从 56 增至 98、平均捕获时间从 16.1 s 增至 18.1 s。这是停止继续训练/扩展 seed 的证据，而不是 JEPA 已经有效的证据。
4. 后续若要公平比较，必须选择一种合同：训练 route chunk=5 的 JEPA，或把 3 步合同作为新的共同 baseline。不能把 5 步历史 baseline 与 3 步 JEPA 结果直接做方法差值。

## 阶段决策

- 保留 5 步解析结果作为历史 G5 合同参考。
- 保留 3 步 paired 结果作为 JEPA 合同诊断。
- 暂停 L1-L3、三 seed JEPA 扩展和继续训练。
- 不降低 CBF margin，不关闭 stale/OOD/non-finite gate，不删除 controlled-abort，也不执行 raw-unverified action。

## 可重放产物

- 机器可读审计：`results/dn_mpc_jepa_route_chunk_contract_audit_20260907/summary.json`
- TensorBoard：`results/dn_mpc_jepa_route_chunk_contract_audit_20260907_tensorboard/`
- 生成脚本：`scripts/summarize_dn_mpc_jepa_chunk_contract_audit.py`
