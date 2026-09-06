# JEPA Safe-Capture Plan Index

自 2026-09-06 起，唯一执行入口是：

- [JEPA_SAFE_CAPTURE_CANONICAL_TODO_20260906.md](JEPA_SAFE_CAPTURE_CANONICAL_TODO_20260906.md)
- [WP1 obstacle-route audit](JEPA_SAFE_CAPTURE_WP1_OBSTACLE_ROUTE_AUDIT_20260906.md)
- [WP2 route runtime smoke](JEPA_SAFE_CAPTURE_WP2_ROUTE_RUNTIME_SMOKE_20260906.md)

此前按 V2/V3/V4/V5/V11/V20/V21、WP、P、T 或日期拆分的 TODO、PLAN、ROADMAP 和 EXECUTION 文件均为历史参考，状态为 `superseded / reference-only`，不得作为新的执行指令。它们保留在仓库中，是为了保留当时的实验假设、失败诊断和可复现命令。

以下内容不属于计划文件，不应删除：

- locked/development 实验报告与 summary；
- checkpoint、NPZ、calibration archive、reliability ledger；
- TensorBoard、episode CSV、step trace 和 deterministic replay；
- 测试文件、环境配置、协议 hash 和 provenance。

清理规则：新实验只引用 canonical TODO；旧计划若需查阅，只作为历史上下文，不能覆盖当前候选路线、数据标签、Ledger 或 CBF 合同。
