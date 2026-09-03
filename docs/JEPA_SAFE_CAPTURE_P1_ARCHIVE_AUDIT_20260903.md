# P1 困难场景与校准归档审计

**日期：** 2026-09-03
**阶段：** P1，development-only
**安全主指标：** safe capture ；collision、boundary、机间净空和 CBF 可行性为硬约束
**locked test：** 未打开（`locked_test_opened=false`）

## 1. 交付内容

本阶段新增了独立的 v2 archive 采集器和审计器：

- `configs/jepa_safe_capture_v2_collection.yaml`
- `scripts/generate_jepa_safe_capture_v2_archive.py`
- `scripts/audit_jepa_safe_capture_v2_archive.py`
- `tests/test_jepa_safe_capture_v2_archive.py`

旧的 V3 archive 采集脚本和历史 V4/V5 文件没有被覆盖。新增 archive 只使用
P1 collection config 中声明的训练、验证和校准 seed block；development/locked
seed 仅用于 namespace 审计，采集器拒绝读取这两个 split。

## 2. 数据合同

每个样本由显式的 `(episode_seed, time_index, agent_id, candidate_index)` 索引，
每个状态-agent 组包含恰好 5 个候选，候选 0 是 nominal action。模型输入仍是
`8 x 63` policy-safe observation history 和 `8 x 3` action history，在线 target
truth 没有写入输入。

离线 settled labels 为：

- target relative position、target velocity、target acceleration；
- obstacle clearance、inter-agent clearance；
- pairwise time-to-collision（秒，裁剪到 10 秒）；
- target visibility、observation age；
- CBF correction、CBF intervention、CBF QP feasibility；
- collision、boundary。

目标速度和加速度分别按 target 最大速度/加速度归一化，净空按世界半边长归一化；
TTC、观测年龄和 CBF correction 的单位在 metadata 中冻结。

## 3. 场景覆盖

8 个 scenario family 均包含 3--5 个障碍物，并覆盖：nominal flee、延迟噪声、
s-curve、高拥挤、低可见性 burst、目标速度突变、频繁随机转向和 narrow-channel
低净空。每个 split 为 8 个 scenario family x 8 episodes，共 64 episodes。

## 4. 审计结果

| Split | 样本数 | Episode 数 | State-agent 组 | 候选组完整 | Dataset SHA-256 |
|---|---:|---:|---:|---|---|
| train | 76,700 | 64 | 15,340 | 是，5/组 | `3186b05ce145303658b3fdb87ff5c3868ac8330170ad8f24515d93d9ced2ecfd` |
| validation | 77,400 | 64 | 15,480 | 是，5/组 | `48af3ce3bd83a7aa4d068d1f25c8311df706cf892c88d51690dd595c2643ccc7` |
| calibration | 76,800 | 64 | 15,360 | 是，5/组 | `fe2c6148d09a299633397b30f55bcb75c096650d54dd401286260b9f9cd95615` |

三 split 联合审计结果：

- episode seed 交集为 `0`；
- 全部 arrays finite；
- action history 最大绝对归一化值不超过合同范围；
- binary labels 均为 0/1；
- 每个 split 的 nominal fraction 为 `0.2`；
- archive manifest、metadata、collection/protocol/source hash 均匹配；
- 每个 TensorBoard logdir 有 31 个 scalar、13 个 histogram、4 个 provenance text，必需标签齐全；
- P1 生成和审计过程均保持 `locked_test_opened=false`。

### 标签覆盖摘要

| Split | 最小障碍净空（归一化） | 最小机间净空（归一化） | 最小 TTC（秒） | CBF intervention | QP infeasible |
|---|---:|---:|---:|---:|---:|
| train | 0.0335 | 0.0393 | 0.2687 | 12.32% | 0 |
| validation | 0.0169 | 0.0476 | 0.3001 | 13.66% | 0 |
| calibration | 0.0182 | 0.0393 | 0.3003 | 11.93% | 0 |

采集标签中的少量 collision/boundary 记录是困难 counterfactual 的监督信号，
不是部署安全结论；P5 仍必须验证运行时 CBF 的 zero-regression 和 infeasible fallback。

## 5. 发现并修复的数据问题

第一次采集使用的 split base seed 相差 `10,000`，而场景偏移也使用
`scenario_index * 10,000`，因此 train 与 validation 出现了 seed namespace 重叠。
联合审计在训练开始前捕获该问题。随后将 split base seed 间隔扩大到 `100,000`，
重新采集 validation 和 calibration，并仅把修复后的 archive 作为 P1 正式输入。
旧重叠输出保留在本机作为失败审计痕迹，不进入后续训练。

## 6. 下一步准入

P1 已满足进入 P2 的数据准入条件。P2 必须只使用：

- train：`results/jepa_safe_capture_v2_p1_train_rerun/`；
- validation：`results/jepa_safe_capture_v2_p1_validation_rerun/`；
- calibration：`results/jepa_safe_capture_v2_p1_calibration/`。

P2 将新增速度/加速度、保守净空、TTC、可见性、CBF 风险和 action-consistency
heads，并把三 split metadata/hash 写入 TensorBoard。calibration split 只能用于
P3 ledger 校准，不能用于 JEPA 参数更新。
