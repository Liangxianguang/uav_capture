# 3D UAV Cooperative Capture-Radius Pursuit

本仓库实现并记录一个可重复的多无人机三维捕获半径追逃仿真：4 架防守无人机仅使用局部观测、历史信息和受限队友通信，协同追捕 1 个带障碍规避行为的逃逸目标。

## 研究边界

项目的主线后端是运动学（kinematic）三维仿真，场地为 `20 m x 20 m x 10 m`，捕获半径为 `0.80 m`。`Safe Capture` 表示首次进入捕获半径前没有障碍物碰撞、机间碰撞或边界越界。

这里的 capture-radius event 不是网捕、机械抓取、接触动力学、真实视觉闭环、SITL 或实飞结果。PyBullet/SITL 代码属于可选的 sim-to-sim 扩展，不是主线复现门槛。实验协议和停止规则见 [CENTRAL_OBSTACLE_BIDIRECTIONAL_CAPTURE_TODOLIST_V4.txt](CENTRAL_OBSTACLE_BIDIRECTIONAL_CAPTURE_TODOLIST_V4.txt) 与 [NEXT_PHASE_CAPTURE_ROBUSTNESS_TODOLIST.txt](NEXT_PHASE_CAPTURE_ROBUSTNESS_TODOLIST.txt)。

## 当前正式结论

以下数字来自已提交的冻结报告，不代表 clone 后无需训练即可重新生成同一 checkpoint。Raw policy 与 CBF execution 始终分开统计。

| 实验块 | Raw policy | Policy + local CBF | 结论 |
| --- | ---: | ---: | --- |
| Central V4 fixed S1 cylinder/box | 未作为独立安全结论 | `100.0%` Safe Capture，`0.0%` collision | 固定场景通过 |
| Central V4 fixed S1 wall | 未作为独立安全结论 | `98.7 +/- 0.6%` Safe Capture | 存在少量安全失败 |
| Central V4 fixed S2 mixed | 未作为独立安全结论 | `100.0%` Safe Capture，`0.0%` collision | 固定场景通过 |
| Central V4 random S3 locked | `2.3 +/- 1.2%`，`97.7 +/- 1.2%` collision | `75.3 +/- 6.5%` Safe Capture，`4.7 +/- 1.2%` collision | 部分成功，尚未解决 |
| Stage 3C-P1 delayed measurements | `63.3 +/- 11.5%` | `84.0 +/- 16.3%` | CBF 提供主要安全收益 |
| Stage 4 F2 burst occlusion | `63.0 +/- 10.8%` | `65.7 +/- 5.2%` | belief 特征没有稳定改善 |

F1/F2 的共同 locked-test 配对区间均跨 0，因此当前阶段停止继续堆叠 belief 模块；明确的失败边界是长测量时延和突发遮挡。完整证据：

- [CENTRAL_V4_LOCKED_TEST_REPORT.md](CENTRAL_V4_LOCKED_TEST_REPORT.md)
- [CENTRAL_V4_D1_VALIDATION_REPORT.md](CENTRAL_V4_D1_VALIDATION_REPORT.md)
- [CENTRAL_V4_S3_RETENTION_VALIDATION_REPORT.md](CENTRAL_V4_S3_RETENTION_VALIDATION_REPORT.md)
- [results/RECURRENT_POLICY_STAGE3C_P1_STRESS_REPORT.md](results/RECURRENT_POLICY_STAGE3C_P1_STRESS_REPORT.md)
- [results/STAGE4C_F1_FORMAL_REPORT.md](results/STAGE4C_F1_FORMAL_REPORT.md)
- [results/STAGE4_COMMON_LOCKED_REPORT.md](results/STAGE4_COMMON_LOCKED_REPORT.md)

## 目录

```text
src/encirclement3d/       环境、观测编码、控制器、CBF、学习模型和可视化
configs/                  主线 YAML 协议和训练配置
scripts/                  训练、评估、聚合、重放和可选 PyBullet 入口
tests/                    单元测试、环境契约和 CLI 回归测试
bench_measurements/       可选 PyBullet 标定数据模板
third_party/              仅保留主线所需的最小 PyBullet 源码和资源
results/                  只提交正式 Markdown/JSON 证据；训练生成物默认被忽略
```

## 安装环境（Windows PowerShell）

```powershell
Set-Location F:\uav_capture\three_d_encirclement
conda env create -f environment.yml
conda activate uav-encirclement-gpu
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
$env:PYTHONPATH = "$PWD\src;$PWD\scripts"
```

`environment.yml` 固定 Python 3.11、PyBullet 3.25、PyTorch 2.7.1+cu126、NumPy、SciPy、TensorBoard 和 pytest。正式训练默认使用 CUDA；没有 GPU 时可把脚本参数改为 `--device cpu`，但训练时间会明显增加。

## 复现顺序

### 1. 回归测试

```powershell
python -m pytest -q
```

### 2. Central V4 从零训练与评估

下面的输出目录是约定路径，所有生成的 checkpoint、数据集、CSV、TensorBoard 和轨迹都在 `results/` 下，且不会进入 Git。

先训练 retained recurrent behavior cloning。该配置不依赖仓库外的数据集，会直接用规则专家重新采样数据：

```powershell
python scripts/train_capture_radius_recurrent_behavior_cloning.py `
  --config configs/capture_radius_recurrent_behavior_cloning_central_v4_s3_retained.yaml `
  --output results/central_v4/bc_s3_retained_seed661201 `
  --device cuda
```

再从 BC checkpoint 训练 retained recurrent MAPPO。配置中的 BC regularizer 数据集正是上一步生成的 `expert_sequence_dataset.npz`：

```powershell
python scripts/train_capture_radius_recurrent_mappo.py `
  --config configs/capture_radius_recurrent_mappo_central_v4_s3_retained.yaml `
  --output results/central_v4/mappo_s3_retained_seed661301 `
  --initialize-from results/central_v4/bc_s3_retained_seed661201/checkpoint.pt `
  --device cuda
```

固定 S1/S2 showcase 和随机 S3 评估：

```powershell
python scripts/evaluate_mixed_obstacle_showcase.py `
  --method f2 `
  --checkpoint results/central_v4/mappo_s3_retained_seed661301/checkpoint.pt `
  --output-dir results/central_v4/eval_s2 `
  --seed 660501 --episodes 20 --scenario v4_s2 --layout mixed --use-cbf

python scripts/evaluate_random_central_mixed_obstacles.py `
  --method f2 `
  --checkpoint results/central_v4/mappo_s3_retained_seed661301/checkpoint.pt `
  --environment-config configs/capture_radius_pursuit_central_v4_flee.yaml `
  --protocol configs/central_random_mixed_obstacle_s3_protocol.yaml `
  --split validation `
  --output-dir results/central_v4/eval_s3_validation `
  --use-cbf
```

可选的规则专家基线无需 checkpoint：将前一条命令的 `--checkpoint ...` 替换为 `--baseline dynamic_encirclement`。正式评估时不要用 locked-test 结果调参；smoke run 只能检查链路，不能支持论文结论。

聚合当前输出：

```powershell
python scripts/aggregate_central_v4_validation.py `
  --results-root results/central_v4 `
  --output-json results/central_v4/validation_summary.json `
  --output-md results/central_v4/validation_report.md

python scripts/aggregate_central_v4_locked_test.py `
  --results-root results/central_v4 `
  --output-json results/central_v4/locked_summary.json `
  --output-md results/central_v4/locked_report.md
```

由于正式 locked-test 需要冻结的三训练 seed 目录和参考场景文件，干净 clone 后应先按中央 V4 计划书完成三 seed 训练，再运行 locked 聚合；仓库提交的 `CENTRAL_V4_LOCKED_TEST_REPORT.md` 是已经完成的历史证据。

### 3. Stage 3C-P1 压力测试

P1 依赖一个只使用本地历史观测训练得到的冻结 GRU，以及 Stage 3C 的两组 recurrent policy。以下步骤从干净 clone 生成这些依赖；训练集和验证集可调参，locked-test 数据只用于最后评估。

```powershell
python scripts/generate_prediction_dataset.py `
  --config configs/capture_radius_observation_communication.yaml `
  --output results/prediction_dataset_train_v2 --split train

python scripts/generate_prediction_dataset.py `
  --config configs/capture_radius_observation_communication.yaml `
  --output results/prediction_dataset_validation_v2 --split validation

python scripts/train_target_predictor.py `
  --train-dataset results/prediction_dataset_train_v2/prediction_dataset.npz `
  --train-metadata results/prediction_dataset_train_v2/metadata.json `
  --validation-dataset results/prediction_dataset_validation_v2/prediction_dataset.npz `
  --validation-metadata results/prediction_dataset_validation_v2/metadata.json `
  --output results/target_predictor_gru_v1 --device cuda

python scripts/run_stage3c_formal.py `
  --seeds 521001 521002 521003 --device cuda --evaluation-device cpu

python scripts/aggregate_stage3c_formal.py

python scripts/run_stage3c_p1_stress.py --device cuda --output-root results/stage3c_p1_stress
python scripts/aggregate_stage3c_p1_stress.py
```

### 4. Stage 4 F1/F2

Stage 4 会自行训练其 BC prior 和 recurrent policy；正式运行是长时间、多 seed 实验，推荐 GPU：

```powershell
python scripts/run_stage4c_formal.py --device cuda --evaluation-device cpu --output-root results/stage4c_formal
python scripts/aggregate_stage4c_formal.py

python scripts/run_stage4d_formal.py --device cuda --evaluation-device cpu --output-root results/stage4d_formal
python scripts/aggregate_stage4d_formal.py `
  --f1-root results/stage4c_formal `
  --f2-root results/stage4d_formal
```

Stage 4 的最后一个共同 locked-test 聚合会读取 P1 summary：

```powershell
python scripts/aggregate_stage4d_formal.py `
  --f1-root results/stage4c_formal `
  --f2-root results/stage4d_formal `
  --p1-summary results/RECURRENT_POLICY_STAGE3C_P1_STRESS_SUMMARY.json `
  --output-json results/STAGE4_COMMON_LOCKED_SUMMARY.json `
  --output-report results/STAGE4_COMMON_LOCKED_REPORT.md
```

正式命令默认每个条件 100 回合、3 个训练 seed、每个 seed 65,536 环境步，耗时可能从数小时到更久。可用 `--episodes-per-condition` 或 `--train-steps` 做 smoke run，但 smoke 输出不能替代正式统计。

### 5. 重放和可视化

训练产生 checkpoint 后，可以重放单个 F1/F2 episode：

```powershell
python scripts/replay_capture_radius_checkpoint.py `
  --method f2 `
  --checkpoint results/stage4d_formal/f2_uncertainty_features/seed521001/recurrent_mappo/checkpoint.pt `
  --condition delayed_measurements --seed 642002 `
  --use-cbf --output-dir results/stage4_visualizations/f2_delayed_seed642002_cbf
```

输出包括 `episode.json`、`trajectory.npz`、PNG/GIF；安装环境中的 FFmpeg 可用时还会生成 MP4。TensorBoard：

```powershell
tensorboard --logdir results --port 6006
```

## 结果与文件管理规则

- Git 只保留源码、测试、配置、协议、正式报告和机器可读 summary。
- checkpoint、expert dataset、TensorBoard、逐回合日志、轨迹、PNG/GIF/MP4、缓存和 smoke/pilot 目录均写入被忽略的 `results/`。
- 训练脚本会在输出目录保存 effective YAML、环境信息和 source hash，便于审计。
- 不要把单个最佳 seed 当作结论；正式统计单位是独立训练 seed。
- 生成物可安全删除后重新运行；不会覆盖非空输出目录。

## 可选 PyBullet 扩展

`src/encirclement3d/pybullet_env.py`、`scripts/evaluate_pybullet.py`、`scripts/train_dagger_pybullet.py` 和 `third_party/` 提供 sim-to-sim/标定入口。它们需要先由主线训练或外部归档提供 checkpoint，不能在没有 checkpoint 的干净 clone 上直接声称完成 PyBullet 评估；该扩展不改变本文运动学主线结论。
