# WP-8 tie3 配对失败重放归档

**日期：** 2026-09-04
**状态：** development-only；`locked_test_opened=false`
**输入：** WP-7 tie3 aggregate 与 21-run failure index
**范围：** M3 全部 120 集 paired episodes；不是新的环境 rollout

## 1. 目标

对每个 M3 episode 重新检查冻结 trace 中的
`belief -> candidate ranking -> ledger -> CBF -> action -> termination` 证据，
并与相同 seed、episode index 的 M0 settled outcome 绑定。该工具是只读审计，不重新
采样场景、不访问 target future ground truth、不修改 WP-7 输出。

## 2. 结果

| 配对类别 | 集数 | 定义 |
|---|---:|---|
| `degraded` | 30 | M0 safe，M3 unsafe |
| `improved` | 10 | M0 unsafe，M3 safe |
| `tied` | 80 | M0 与 M3 的 safe-capture 结果相同 |

120/120 集均成功生成两份 canonical JSONL。两份 hash 全部一致，动作数组和关键
观测、候选、ledger、CBF 字段均为 finite，未发现 trace 级别的非确定性。

## 3. 产物

- `results/wp8_failure_replay_tie3_v2/replay_summary.json`
- `results/wp8_failure_replay_tie3_v2/replay_summary.csv`
- `results/wp8_failure_replay_tie3_v2/replays/`
- `results/wp8_failure_replay_tie3_v2/hash_manifest.json`
- `results/jepa_safe_capture_v3_tensorboard/wp8_failure_replay_tie3_v2/`

配对重放脚本为 `scripts/replay_jepa_safe_capture_wp8_tie3.py`，其单元测试为
`tests/test_replay_jepa_safe_capture_wp8_tie3.py`。

## 4. 解释边界

重放确认了 30 个 degraded episode 的输入、选择、CBF 状态和终止记录可重复，但
trace-only 证据不能单独证明目标预测漂移的因果性。下一步 P9 应把这些 episode 与
CBF timeout/controlled-abort 的 solver 计时、active constraints、slack、ledger credit
和 top-two score margin 联合分析，再决定是修改 ledger、排序器还是 CBF fallback。

当前不打开 locked test，也不以 mean capture time 替代 safe-capture 判断。
