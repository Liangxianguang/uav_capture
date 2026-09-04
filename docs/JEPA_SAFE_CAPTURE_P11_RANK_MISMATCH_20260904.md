# P11 Candidate Rank Mismatch 审计归档

**日期：** 2026-09-04
**状态：** development-only；locked_test_opened=false
**输入：** WP-7 tie3 aggregate、WP-8 全量 M3 canonical replay、failure index

## 1. 结果

| 配对类别 | 集数 | 平均候选切换率 | 平均 selected-not-best | 平均 top-two margin | high-credit failure | oscillation |
|---|---:|---:|---:|---:|---:|---:|
| degraded | 30 | 0.2161 | 0.1959 | 0.001263 | 30 | 6 |
| improved | 10 | 0.1320 | 0.4094 | 0.001376 | 0 | 1 |
| tied | 80 | 0.1721 | 0.2978 | 0.001321 | 50 | 18 |

degraded 集的切换率比 improved 集高约 63.7%，且全部属于 high-credit failure。由于
degraded 集的 selected-not-best 反而较低，当前问题不是“没有按 score 选 top-1”，而是
JEPA score 本身与 settled safe-capture 结果失配：高信用、近似并列的候选被稳定地选中，
但最终更容易触发 CBF abort 或丢失捕获机会。

## 2. 工程决定

下一版 ranker 必须：

1. 先做 finite/reachability/安全下界筛选，再比较 task progress；
2. 将 predicted clearance lower quantile、visibility、CBF intervention cost、uncertainty
   和 action-change cost 纳入安全优先排序；
3. 对 top-two margin 过小、净空/可见性 gap 突增或 high-credit failure 模式执行
   conservative abstention，回退 nominal/safe-hold；
4. 增加 ranking hysteresis 和最小保持时间，抑制候选振荡；
5. 以离线 settled counterfactual label 重新校准 score，而不是按单个 seed 调权重。

该审计是 trace correlation，不是 target-drift 的因果证明。所有新权重必须进入新
protocol、重新训练/校准并通过 smoke 后才能运行三 seed paired block。

## 3. 产物

- results/wp11_rank_mismatch_tie3/rank_mismatch_audit.json
- results/wp11_rank_mismatch_tie3/rank_mismatch_episode.csv
- results/wp11_rank_mismatch_tie3/report.md
- results/jepa_safe_capture_v3_tensorboard/wp11_rank_mismatch_tie3/
- scripts/audit_jepa_safe_capture_wp11_rank_mismatch.py
- tests/test_audit_jepa_safe_capture_wp11_rank_mismatch.py
