# V5 P2 v9 三 Seed Paired Smoke 报告

**日期：** 2026-09-04
**范围：** `development_only=true`，`locked_test_opened=false`
**设备：** NVIDIA GeForce RTX 5050，CUDA 12.8，PyTorch 2.7.1+cu128
**协议：** `configs/central_random_mixed_obstacle_s3_v5_p2_ranking_audit_freeze_v9_development_protocol.yaml`
**聚合器：** `scripts/aggregate_jepa_safe_capture_v5_p2_smoke.py`

## 1. 实验设计

本轮使用三个训练 seed（`20260911`、`20260912`、`20260913`），每个 seed 先运行 M0 生成 20 集
paired scene manifest，再让 M3/A1/A2 复用同一 manifest。四个变体均使用同一 frozen actor、同一
`K=5` 候选、3-step action chunk、只执行第一步、Joint CBF-QP 和 rolling horizon。M3/A2 使用与
各自 JEPA checkpoint 绑定的 v9 Reliability Ledger；A1 为 ledger 消融；A3/no-CBF 未纳入本轮安全
矩阵。

## 2. 输入与 provenance

| 项目 | 值 |
|---|---|
| v9 protocol SHA-256 | `3c734eaba0bcf8cd44724e077d0890b671d50e5c58938e298efe0b667e22f861` |
| environment config SHA-256 | `42bd4e158c5e314e0ece6add8038b32c384a7a2ca027e9387327656fccf751ad` |
| actor SHA-256 | `535098773be05687e147043435649378532362d479bdc0375842970370ba40ba` |
| canonical scene manifest SHA-256 | `da098f35854b6e4e9834c81b3bea7031a34484579dd701b095879cc6ddd29719` |
| calibration archive SHA-256 | `2ce7131f081e096572232fe56ea33dc9459877b53f79575875be3590b66a9a73` |
| scene manifest raw SHA-256 | `6a5fa090...` / `44515c75...` / `ae2081b6...`（按 seed） |

三个 ledger 的 source protocol hash 均等于 v9 hash，source checkpoint hash 分别等于三个 JEPA
checkpoint hash；OOD、stale、non-finite fallback audit 全部通过。每个 run 均包含 `summary.json`、
`episodes.csv`、`step_traces/episode_*.jsonl`、`scene_manifest.jsonl`、`provenance.json` 和
TensorBoard event file。

## 3. Safe-capture 结果

| 变体 | seed 20260911 | seed 20260912 | seed 20260913 | 总体 |
|---|---:|---:|---:|---:|
| M0 | 10/20 = 50.0% | 10/20 = 50.0% | 10/20 = 50.0% | 30/60 = 50.0% +/- 0.0% |
| M3 | 8/20 = 40.0% | 8/20 = 40.0% | 9/20 = 45.0% | 25/60 = 41.7% +/- 2.9% |
| A1 | 8/20 = 40.0% | 8/20 = 40.0% | 9/20 = 45.0% | 25/60 = 41.7% +/- 2.9% |
| A2 | 9/20 = 45.0% | 9/20 = 45.0% | 9/20 = 45.0% | 27/60 = 45.0% +/- 0.0% |

M3 相对 M0 的每 seed delta 为 `-10/-10/-5 pp`。完整 episode 配对统计为：

- improved / degraded / tied = `3 / 8 / 49`；
- mean paired delta = `-8.3 pp`；
- bootstrap 95% CI（episode-pair 单位）=`[-10.0,-5.0] pp`；
- 每个 seed 的 exact McNemar 双侧 p 值分别为 `0.5000`、`0.6250`、`1.0000`。

这不是 JEPA safe-capture 提升证据。当前任务分类为 `useful_safety_fallback_only`，并保留为负向
development evidence。

## 4. 安全与执行审计

四个变体、三个 seed 的以下计数均为 0：

- obstacle/target collision；
- defender boundary violation；
- defender pairwise separation violation；
- CBF timeout；
- `raw_unverified_executed_steps`。

CBF infeasible/controlled-abort 仍发生并被逐步记录：M0 总计 30 steps，M3/A1/A2 各 31 steps。
这些步骤走显式 fallback/controlled-abort，不能被当作成功，也不能改写成 raw execution。seed
`20260913` 的 M3/A1 有 1 次 target-boundary diagnostic；这不是 defender boundary gate，但已保留在
episode summary 中。

所有 ranking audit 的结构 gates、三个 ledger alignment、六个 temporal ledger audit、12 个 latency
audit 和独立 CBF/ledger fault-injection audit 均通过。RTX 5050 端到端 cycle p95 约为 11--22 ms，
CBF solver p95 约为 2.7--3.8 ms，低于 100 ms contract。

## 5. 排序诊断

M3/A2 的 settled-ranking 审计均为 `no_control_gain`。M3 seed `20260911` 的 settled decisions
为 1,075，high-credit selected-not-settled-best 为 282/1,052；seed `20260912` 和 `20260913`
仍有同类排序失配。high-credit failure rate 低于 low/missing-credit bucket，说明 ledger 的拒答
路径有效，但它没有把 JEPA 预测稳定转化为更好的候选选择。

## 6. 决策与下一步

本轮安全门和 provenance 门通过，但 G-Noninferiority 未通过，因此：

1. 不运行 40 集 validation；
2. 不打开 locked test；
3. 不把 `mean_capture_time` 用来抵消 safe-capture 下降；
4. 不修改当前 v9 结果文件，不删除失败 episode；
5. 下一阶段建立新 protocol revision，优先修复 lower-quantile clearance、abstention/nominal anchor、困难片段 replay 和安全辅助头校准，然后重新生成 checkpoint、ledger、manifest、TensorBoard 和 smoke 证据。

**结论：** 当前链路已证明 `JEPA -> Reliability Ledger -> safety-first ranker -> Joint CBF-QP -> rolling horizon`
可以在 RTX 5050 上安全、可追溯地运行，但 v9 三 seed smoke 尚未证明对 safe-capture 有控制收益。
