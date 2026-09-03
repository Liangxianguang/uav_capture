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
- [E1-prime feasibility rejection report](E1_PRIME_RULE_EXPERT_FEASIBILITY_REJECTION_REPORT.md)
- [E1-prime feasibility aggregate](E1_PRIME_RULE_EXPERT_FEASIBILITY_REPORT.md)
- [最新围捕拦截模型候选与 JEPA 研究备忘录](docs/LATEST_MODEL_CANDIDATES_FOR_PURSUIT_INTERCEPTION_20260902.md)
- [JEPA safe-capture v2 当前执行计划](docs/JEPA_SAFE_CAPTURE_V2_NEXT_EXECUTION_PLAN_20260903.md)

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
