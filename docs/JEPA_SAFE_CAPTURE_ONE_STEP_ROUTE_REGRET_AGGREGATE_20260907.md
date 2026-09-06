# One-Step Settled Route-Regret Aggregate

**Date:** 2026-09-07  
**Scope:** development-only offline audit across the three-episode paired manifest  
**Locked test:** not opened

## Aggregate result

| Episode seed | Trusted steps | Selected = settled best | Mean regret (m) | Score argmin = selected |
| ---: | ---: | ---: | ---: | ---: |
| `650101` | `80` | `63.75%` | `0.0204` | `100%` |
| `650102` | `111` | `47.75%` | `0.0252` | `100%` |
| `650103` | `248` | `22.98%` | `0.0453` | `100%` |
| **Weighted total** | **439** | **36.67%** | **0.0357** | **100%** |

Maximum observed one-step regret was `0.1141 m`. Only one step across the
three episodes had no primary-CBF-accepted candidate; selected-route CBF
failure was zero in all comparable steps.

## Conclusion

This is a cross-scene score/label alignment failure, not a ranking execution
bug. The ranker consistently executes its own score argmin, but that argmin
matches the true one-step best safe route only `36.67%` of the time. The
agreement also degrades as the scene becomes more difficult (`63.75%` to
`22.98%`).

The result does **not** authorize changing an online score term, reducing CBF
margins, disabling Ledger gates, retraining, enlarging the scene matrix, or
opening a locked test. The next permitted work is offline calibration of a
settled progress/escape label on fixed micro-scenes, followed by a fresh
paired L0 gate only if the offline signal is stable.

## Reproducibility artifacts

- Aggregator: [aggregate_jepa_safe_capture_one_step_route_regret.py](../scripts/aggregate_jepa_safe_capture_one_step_route_regret.py)
- Per-episode audit: [audit_jepa_safe_capture_one_step_route_regret.py](../scripts/audit_jepa_safe_capture_one_step_route_regret.py)
- Source manifest SHA-256: `130f9602a87bcc392b10b2fa8bddaeaa88dab21bae4d932f95ddfeb7647da644`
- Aggregate JSON: `results/wp1_active_search_v5_escape_switch_one_step_route_regret_aggregate/aggregate.json`
- Aggregate JSON SHA-256: `2faa89be9fc8a8d9ba05e1c748c5f24ce16c117246a4df4da40caca477b1e27e`
- TensorBoard: `results/wp1_active_search_v5_escape_switch_one_step_route_regret_aggregate_tensorboard/`
- TensorBoard event SHA-256: `f06dc6d8e71cabe08a215df051ca8060e6912ae6ca258fc2394e72099bd38104`

All three per-episode JSON reports and TensorBoard logs remain in their
source directories and are hash-recorded by the aggregate.
