# 3D Multi-UAV Cooperative Capture

This repository studies cooperative pursuit of one evasive target by four UAVs
in a partially observable three-dimensional obstacle field. The research
question is whether the team can bring at least one pursuer inside a capture
radius while avoiding obstacle collision, inter-UAV collision, and world-boundary
violation.

The main benchmark is a kinematic simulation, not a physical capture system,
flight-controller validation, vision-in-the-loop system, SITL result, or flight
test. The safety filter is part of the deployed method: results labelled
`Policy + CBF` must not be interpreted as results of the neural policy alone.

## Capture Definition

An episode is a **Cooperative Safe Capture** only if all of the following hold:

1. At least one defender enters the target's `0.80 m` capture radius.
2. The event occurs before the episode time limit.
3. No defender, target, or defender pair collides with an obstacle or each
   other before termination.
4. No defender leaves the allowed 3-D world boundary.

Entering the central obstacle region is recorded as a diagnostic, not a
precondition for capture. The target may be captured on either side of the
obstacles.

## Released Evidence

The repository distinguishes formal locked-test evidence from development-only
experiments. Do not cite a single selected development seed as a formal
improvement.

| Benchmark | Execution stack | Cooperative Safe Capture | Safety / interpretation |
| --- | --- | ---: | --- |
| V4 fixed cylinder | retained BC + CBF | `100.0% +/- 0.0%` | 3 independently trained checkpoints |
| V4 fixed box | retained BC + CBF | `100.0% +/- 0.0%` | 3 independently trained checkpoints |
| V4 fixed wall | retained BC + CBF | `98.7% +/- 0.6%` | small residual safety failures |
| V4 fixed mixed S2 | retained BC + CBF | `100.0% +/- 0.0%` | 3 independently trained checkpoints |
| V4 random mixed S3 locked test | retained BC + CBF | `75.3% +/- 6.5%` | collision `4.7% +/- 1.2%`, boundary `4.7% +/- 1.2%` |
| V4 random mixed S3 locked test | raw policy | `2.3% +/- 1.2%` | collision `97.7% +/- 1.2%`; not deployable |
| V5 random mixed S3 development | exact-reactive seed `661606` + CBF | `57/60 = 95.0%` | collision `0%`, boundary `0%`, Transit `100%`; one seed only |
| V5 E1-prime execution feasibility | rule expert + execution-aware CBF | `E0: 95.0%`; all-profile gate **FAIL** | E1-E6 fail capture and/or safety thresholds; no policy/locked evaluation opened |

The V4 locked-test report is the current formal benchmark result:

- [V4 locked-test report](CENTRAL_V4_LOCKED_TEST_REPORT.md)
- [V4 locked-test summary](CENTRAL_V4_LOCKED_TEST_SUMMARY.json)
- [V4 visualisation audit](CENTRAL_V4_VISUALIZATION_REPORT.md)
- [V4 archive-faithful RTX 5050 reproduction](docs/CENTRAL_V4_ARCHIVE_FAITHFUL_REPRODUCTION_20260830.md)
- [V5 development status](CENTRAL_V5_EXACT_REACTIVE_DEVELOPMENT_STATUS.md)
- [JEPA safe-capture v2 P7 full development report (historical pre-tie3)](docs/JEPA_SAFE_CAPTURE_P7_FULL_DEVELOPMENT_REPORT_20260904.md)
- [JEPA safe-capture P7 之后下一阶段详细目标计划书](docs/JEPA_SAFE_CAPTURE_NEXT_PHASE_TODOLIST_20260904.md)
- [JEPA safe-capture 当前执行版下一步 TODO 与验收计划](docs/JEPA_SAFE_CAPTURE_NEXT_TODO_PLAN_20260904.md)
- [JEPA safe-capture v11 corrected-frame 下一步详细 TODO 与验收计划](docs/JEPA_SAFE_CAPTURE_V11_CORRECTED_FRAME_NEXT_TODO_PLAN_20260904.md)
- [JEPA safe-capture v11 hard-replay 下一步执行版 TODO 计划书（当前入口）](docs/JEPA_SAFE_CAPTURE_NEXT_EXECUTION_TODO_20260905.md)
- [JEPA safe-capture v20 CPU deterministic 下一步详细 TODO 计划书（最新执行入口）](docs/JEPA_SAFE_CAPTURE_NEXT_TODO_PLAN_20260905_V2.md)
- [JEPA safe-capture v20 seed 20260911 CPU/CUDA replay 归档](docs/JEPA_SAFE_CAPTURE_V20_SEED20260911_DEVICE_REPLAY_20260905.md)
- [JEPA safe-capture 当前下一步执行 TODO 计划（V21 smoke 之后）](docs/JEPA_SAFE_CAPTURE_CURRENT_NEXT_TODO_PLAN_20260905.md)
- [JEPA safe-capture V21 三 seed settled ranking aggregate](docs/JEPA_SAFE_CAPTURE_V21_SETTLED_RANKING_AGGREGATE_20260905.md)
- [JEPA safe-capture V5 当前下一阶段详细 TODO 计划（P2 之后）](docs/JEPA_SAFE_CAPTURE_V5_NEXT_TODO_PLAN_20260904.md)
- [JEPA safe-capture V5 P2 v9 三 seed paired smoke 报告](docs/JEPA_SAFE_CAPTURE_V5_P2_V9_SMOKE_20260904.md)
- [JEPA safe-capture P11 之后下一步详细 TODO 与目标计划书](docs/JEPA_SAFE_CAPTURE_NEXT_DETAILED_TODO_20260904.md)
- [JEPA safe-capture P12 clearance-floor sensitivity 与 temporal ledger audit](docs/JEPA_SAFE_CAPTURE_P12_FLOOR015_SENSITIVITY_20260904.md)
- [JEPA safe-capture T2 settled counterfactual ranking audit](docs/JEPA_SAFE_CAPTURE_T2_SETTLED_COUNTERFACTUAL_20260904.md)
- [JEPA safe-capture T3 reliability ledger alignment audit](docs/JEPA_SAFE_CAPTURE_T3_LEDGER_ALIGNMENT_20260904.md)
- [JEPA safe-capture 当前主 TODO 与目标计划书（T3 证据缺口修复版）](docs/JEPA_SAFE_CAPTURE_CURRENT_MASTER_TODO_20260904.md)
- [JEPA safe-capture 下一阶段实施 TODO 与验收计划（当前入口）](docs/JEPA_SAFE_CAPTURE_NEXT_IMPLEMENTATION_TODO_20260904.md)
- [JEPA safe-capture tie2 后下一步详细执行计划](docs/JEPA_SAFE_CAPTURE_NEXT_EXECUTION_PLAN_20260904.md)
- [JEPA safe-capture WP7 tie3 完整开发实验归档](docs/JEPA_SAFE_CAPTURE_WP7_TIE3_FINAL_DEVELOPMENT_20260904.md)
- [JEPA safe-capture WP8 tie3 配对失败重放归档](docs/JEPA_SAFE_CAPTURE_WP8_TIE3_FAILURE_REPLAY_20260904.md)
- [JEPA safe-capture P9 CBF Jacobian 可靠性修复归档](docs/JEPA_SAFE_CAPTURE_P9_CBF_JACOBIAN_20260904.md)
- [JEPA safe-capture P11 Candidate Rank Mismatch 审计归档](docs/JEPA_SAFE_CAPTURE_P11_RANK_MISMATCH_20260904.md)
- [JEPA safe-capture 系统详细 TODO、实验与验收计划书](docs/JEPA_SAFE_CAPTURE_SYSTEM_DETAILED_TODO_PLAN_20260904.md)
- [JEPA safe-capture WP0 baseline freeze report](docs/JEPA_SAFE_CAPTURE_WP0_BASELINE_FREEZE_20260904.md)
- [JEPA safe-capture WP1 failure index and causal replay audit](docs/JEPA_SAFE_CAPTURE_WP1_FAILURE_REPLAY_20260904.md)
- [JEPA safe-capture WP6 smoke boundary semantics audit](docs/JEPA_SAFE_CAPTURE_WP6_SMOKE_SAFETY_AUDIT_20260904.md)
- [JEPA safe-capture WP6 boundary-fixed smoke report](docs/JEPA_SAFE_CAPTURE_WP6_SMOKE_BOUNDARYFIXED_20260904.md)
- [JEPA safe-capture WP2 hard-context weighted training report](docs/JEPA_SAFE_CAPTURE_WP2_HARD_CONTEXT_TRAINING_20260904.md)
- [JEPA safe-capture WP2 held-out prediction aggregate](docs/JEPA_SAFE_CAPTURE_WP2_HARD_CONTEXT_PREDICTION_20260904.md)
- [JEPA safe-capture WP3 v3 reliability-ledger aggregate](docs/JEPA_SAFE_CAPTURE_WP3_LEDGER_V3_20260904.md)
- [E1-prime feasibility rejection report](E1_PRIME_RULE_EXPERT_FEASIBILITY_REJECTION_REPORT.md)
- [E1-prime feasibility aggregate](E1_PRIME_RULE_EXPERT_FEASIBILITY_REPORT.md)
- [最新围捕拦截模型候选与 JEPA 研究备忘录](docs/LATEST_MODEL_CANDIDATES_FOR_PURSUIT_INTERCEPTION_20260902.md)
- [JEPA safe-capture v2 当前执行计划](docs/JEPA_SAFE_CAPTURE_V2_NEXT_EXECUTION_PLAN_20260903.md)
- [JEPA + Reliability Ledger + CBF 系统总 TodoList](docs/JEPA_SAFE_CAPTURE_SYSTEM_MASTER_TODO_20260903.md)
- [JEPA safe-capture 下一步执行 TODO 计划](docs/JEPA_SAFE_CAPTURE_NEXT_STEP_TODOLIST_20260903.md)
- [JEPA safe-capture v2 P5 联合 CBF-QP 审计](docs/JEPA_SAFE_CAPTURE_P5_CBF_QP_AUDIT_20260903.md)

The V5 `95.0%` number is included to make the best current observed run
inspectable. It has **not** opened its V5 locked block and does not meet the
three-independent-seed gate.

## Repository Layout

```text
src/encirclement3d/       Environment, observations, controller, CBF, policy, dynamics
configs/                  Versioned V4/V5 environment and evaluation contracts
scripts/                  Training, evaluation, aggregation, replay, and rendering CLIs
tests/                    Environment, protocol, CLI, and regression tests
models/                   Reviewed V5 development checkpoint (0.60 MB)
docs/media/               Representative development-only capture media and provenance
results/                  Local generated output only; ignored except formal reports/summaries
third_party/              Minimal vendored assets for optional PyBullet extensions
```

Generated checkpoints, expert archives, TensorBoard logs, episode CSV files,
trajectory arrays, and scratch media belong in `results/`. They are deliberately
not tracked. Each run writes its effective configuration and provenance next to
its output, so it can be rerun without retaining gigabytes of intermediates.

## Environment

The project was exercised on Windows with Python 3.11, PyTorch `2.7.1+cu126`,
and an NVIDIA RTX 4060. `environment.yml` also installs PyBullet, FFmpeg,
TensorBoard, and pytest.

```powershell
Set-Location F:\uav_capture\three_d_encirclement
conda env create -f environment.yml
conda activate uav-encirclement-gpu
$env:PYTHONPATH = "$PWD\src;$PWD\scripts"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If the conda environment already exists, use `conda env update -f environment.yml
--prune`. CUDA is recommended for training; evaluation and media rendering can
run on CPU.

## Quick Reproduction: Released V5 Development Checkpoint

This command recreates the 60-episode V5 **development** protocol from the
released checkpoint. It is not a locked test. The per-step recurrent reset is
read from the checkpoint automatically; specifying `1` makes the deployment
contract explicit.

```powershell
python scripts/evaluate_random_central_mixed_obstacles.py `
  --checkpoint models/v5_development_exact_reactive_seed661606.pt `
  --environment-config configs/capture_radius_pursuit_central_v4_flee.yaml `
  --protocol configs/central_random_mixed_obstacle_s3_v5_protocol.yaml `
  --split validation `
  --output-dir results/reproduce_v5_seed661606_s3_validation `
  --use-cbf --recurrent-reset-interval 1 --device cuda
```

Expected historical result: `57/60` Cooperative Safe Capture, with no
collision or boundary failure. Output includes `episodes.csv`, `scenes.jsonl`,
`summary.json`, protocol/evaluation metadata, and the generated random layouts.

To inspect the CBF contribution, repeat the same command without `--use-cbf` in
a different output directory. This is an exploratory comparison, not a basis
for tuning the locked V4 benchmark.

## Train the Formal V4 Protocol from Scratch

The formal V4 result was calculated over three independent BC training seeds.
The source and contract are included, while historical raw expert archives and
large intermediate outputs are intentionally not versioned. This reruns the
same training procedure and produces new local archives under `results/`.

```powershell
$seeds = 661201, 661202, 661203
foreach ($seed in $seeds) {
  python scripts/train_capture_radius_recurrent_behavior_cloning.py `
    --config configs/capture_radius_recurrent_behavior_cloning_central_v4_s3_retained.yaml `
    --seed $seed `
    --output "results/central_v4/bc_s3_retained_seed$seed" `
    --device cuda
}
```

Monitor training with:

```powershell
tensorboard --logdir results --port 6006
```

The training script refuses to overwrite a non-empty run directory. A complete
formal repetition needs the three frozen checkpoints, its registered seed block,
and the locked evaluator; do not use locked-test outputs to change a model,
threshold, scene sampler, or CBF parameter.

## Evaluate Fixed and Random Obstacles

Use a fresh output directory for every run. The following is a development
example using the released V5 checkpoint and the fixed mixed S2 scene:

```powershell
python scripts/evaluate_mixed_obstacle_showcase.py `
  --checkpoint models/v5_development_exact_reactive_seed661606.pt `
  --scenario v4_s2 --layout mixed --seed 660501 --episodes 20 `
  --output-dir results/reproduce_v5_seed661606_s2 `
  --use-cbf --recurrent-reset-interval 1 --device cuda
```

The random S3 command in the previous section is the recommended robustness
evaluation: it samples 3-5 central cylinder, box, and wall obstacles, uses
separate motion/layout seeds, includes nominal and delayed-noisy observations,
and records Transit independently from capture.

## Replay and Render a Capture

After the random evaluation finishes, replay a known successful scene. The
script restores its scene metadata and produces trajectory data, a top-down
PNG/GIF/MP4, and audit metadata.

```powershell
python scripts/render_random_capture_episode.py `
  --checkpoint models/v5_development_exact_reactive_seed661606.pt `
  --scenes results/reproduce_v5_seed661606_s3_validation/scenes.jsonl `
  --episode-index 0 --use-cbf --recurrent-reset-interval 1 `
  --output-dir results/reproduce_v5_seed661606_episode0 --device cuda

python scripts/render_3d_capture_animation.py `
  --trajectory results/reproduce_v5_seed661606_episode0/trajectory.npz `
  --result results/reproduce_v5_seed661606_episode0/episode.json `
  --output-dir results/reproduce_v5_seed661606_episode0/three_d
```

The 3-D renderer keeps obstacle volumes, altitude projections, capture sphere,
trajectory tails, and a frozen `CAPTURE CONFIRMED` frame. FFmpeg in the conda
environment writes H.264 MP4 in addition to GIF. A representative V5
development replay is documented in [docs/media/README.md](docs/media/README.md).

## Verification

Run the complete test suite after changing source or protocol code:

```powershell
python -m pytest -q
```

For a quick CLI smoke check:

```powershell
python scripts/render_3d_capture_animation.py --help
python scripts/render_random_capture_episode.py --help
python scripts/evaluate_random_central_mixed_obstacles.py --help
```

## Scope and Reporting Rules

- `CBF` is part of the execution stack and must be reported separately from a
  raw actor.
- A capture-radius event is not physical contact capture or net capture.
- Formal conclusions use independent training seeds and the locked protocol;
  single-seed results are development evidence only.
- The V4 locked result remains `75.3% +/- 6.5%`, not `95.0%`.
- The V5 development checkpoint is released for inspection and rerun, not as a
  substitute for a multi-seed locked comparison.
- Optional PyBullet and execution-dynamics modules are extensions; they do not
  convert the main kinematic benchmark into a real-flight result.
