# JEPA-v3 P5 Zero-Perturbation Regression

**阶段：** P5-D  
**日期：** 2026-09-03  
**最终结论：** `pass; admitted_to_nonzero_seed11_smoke`  
**实验性质：** development-only；不是 locked test 或闭环改善证据

## 1. 目的

在 `perturbation=0.0` 时，五个 constant action chunks 的第一步语义上均应为冻结 V5 actor 的 nominal desired action。因此接入 JEPA、candidate chunk 和 reliability ledger 后，除 JEPA 专属诊断字段外，结果必须与 V5+CBF baseline 严格逐字段相同。

## 2. 初始失败与根因

初次 v2 run（`results/jepa_v3_p5_chunk3v2_seed20260911_zero20/`）使用相同的 20 scenes 和 episode identities，但在 87 个非 JEPA 字段中仍有 4 个差异：均为 `mean_cbf_action_correction_norm` 或 `max_cbf_action_correction_norm` 的最后浮点位。该 run 仍判定为失败，未进入 non-zero smoke。

根因是 zero case 仍然生成 `float64` constant chunks 并经过 reranker，然后传给 CBF；baseline 则直接传入 actor 的 original nominal action。值在物理层相同，但 CBF 的中间计算因 dtype 路径不同出现末位差异。

## 3. 修复

`rollout_showcase()` 增加了只用于回归验证的 identity bypass：

- 仅在 JEPA enabled **且** `jepa_perturbation_mps == 0.0` 时启用；
- 不生成候选 chunk、不调用 JEPA 推断、不执行 reranker；
- 保留 JEPA enabled、checkpoint、ledger 和 action-chunk 的运行元数据；
- nominal actor action 直接交给与 baseline 完全相同的 CBF path；
- 非零扰动控制永远不经过这个 bypass。

新增单元测试验证 bypass 仅在 JEPA enabled 的精确零扰动情形触发；相关测试结果为 `19 passed`。该修复不改变 v2 archive、training source、checkpoint、prediction gate、ledger 或任何 non-zero reranker 语义，所以不重新训练三 seed。

## 4. Strict replay result

新的运行目录：`results/jepa_v3_p5_chunk3v2p1_seed20260911_zero20/`。

| 检查 | 结果 |
|---|---:|
| baseline directory | `results/jepa_v3_p3_zero_baseline20/` |
| candidate checkpoint | `57741bbfdffb806d14043bc8620024f602eb412f7907f81e762e3d6af5b48c4f` |
| candidate ledger | matching v2 seed-11 ledger |
| episodes | 20 / 20 paired |
| scenes SHA-256 baseline/candidate | `1402bf6429814f7638625025bc75a3b4ca04ac3c0bc107eef13ac0cdf2a18b99` / identical |
| scenes byte-identical | true |
| non-JEPA fields compared | 87 |
| field differences | 0 |
| CBF/collision/boundary/capture/path/time | exact match |
| result | pass |

Machine-readable comparison：`results/jepa_v3_p5_chunk3v2p1_seed20260911_zero20/zero_perturbation_comparison.json`。

## 5. Reproduction

```powershell
$py = 'D:\miniconda3\envs\uav-encirclement-gpu\python.exe'

& $py scripts\evaluate_random_central_mixed_obstacles.py `
  --method f2 `
  --checkpoint models\v5_development_exact_reactive_seed661606.pt `
  --environment-config configs\capture_radius_pursuit_central_v4_flee.yaml `
  --protocol configs\central_random_mixed_obstacle_s3_v5_protocol.yaml `
  --split validation --episodes 20 --use-cbf --device cuda `
  --action-conditioned-jepa-checkpoint results\jepa_v3_multitask_chunk3v2_seed20260911\checkpoint.pt `
  --jepa-reliability-ledger results\jepa_v3_multitask_chunk3v2_seed20260911\reliability_ledger.json `
  --jepa-candidate-count 5 --jepa-perturbation-mps 0.0 --jepa-action-chunk-length 3 `
  --reference-scenes results\jepa_v3_p3_zero_baseline20\scenes.jsonl `
  --reference-episodes results\jepa_v3_p3_zero_baseline20\episodes.csv `
  --output-dir results\jepa_v3_p5_chunk3v2p1_seed20260911_zero20

& $py scripts\compare_jepa_v3_zero_perturbation.py `
  --baseline-dir results\jepa_v3_p3_zero_baseline20 `
  --candidate-dir results\jepa_v3_p5_chunk3v2p1_seed20260911_zero20 `
  --output results\jepa_v3_p5_chunk3v2p1_seed20260911_zero20\zero_perturbation_comparison.json
```

## 6. Decision

P5-D now passes. The only authorized next step is the fixed non-zero seed-11 20-episode smoke (`K=5`, `0.10 m/s`, chunk `3`, matching ledger, CBF enabled). This result validates identity behavior at zero perturbation only; it does not establish that JEPA selects better non-zero actions.
