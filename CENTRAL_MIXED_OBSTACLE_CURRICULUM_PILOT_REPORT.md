# Central Mixed-Obstacle Curriculum Pilot

## Scope

This report records a single-seed curriculum pilot following the Phase A
distribution-shift probe. It is evidence that the proposed training interface
can improve the controlled S1 showcase; it is **not** a multi-seed formal
claim and it does not replace the existing F1/F2 locked-test conclusion.

## Training Setup

- Environment: `configs/capture_radius_pursuit_showcase_mixed_curriculum.yaml`.
- Curriculum: original random episodes are retained, while a growing fraction
  of episodes uses opposite-side starts and progresses through `open`,
  `cylinder`, `cylinder_box`, and `mixed` central layouts.
- Behavior-cloning teacher: existing Dynamic Encirclement controller with the
  local CBF filter.
- BC pilot: 160 episodes, 20 epochs, seed `640101`.
- MAPPO pilot: initialized from the BC checkpoint, 32,768 environment steps,
  seed `640101`, RTX 4060 CUDA execution.
- Showcase probe: 20 independent seeds `643001..643020`; 5.0 m side distance,
  14.0 m showcase-only detection range, target speed scale 0.55, CBF enabled.
- F2 baseline checkpoint: formal seed `521001` checkpoint. All comparison
  methods use the same fixed geometry and probe seeds.

## Results

| Method | Safe Capture | Defender zone crossing | Collision | Boundary violation | Mean capture time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen formal F2 + CBF | 0 / 20 (0%) | 100% | 0% | 0% | n/a |
| Rule expert + CBF | 20 / 20 (100%) | 100% | 0% | 0% | 3.45 s |
| Curriculum BC + CBF | 12 / 20 (60%) | 100% | 0% | 0% | 14.63 s |
| Curriculum MAPPO + CBF | 14 / 20 (70%) | 100% | 5% | 5% | 6.66 s |

The BC expert demonstration set contained 160 episodes with an expert Safe
Capture rate of 86.9%. The BC model's ordinary random validation was weak
(20% Safe Capture), so it was used only as a warm start. MAPPO then improved
the controlled S1 probe to 70%, with faster successful captures than BC.

## Interpretation

The curriculum pilot demonstrates that the original F2 failure is addressable:
the learned policy can now safely traverse the fixed central mixed obstacle
zone and capture in a majority of held-out perception/noise seeds. The gain is
not yet sufficient for a robust final model because:

- 5/20 episodes still time out;
- 1/20 episodes has a collision and boundary violation;
- only one training seed has been run;
- S2 and randomized S3 have not yet been evaluated.

The next training iteration should increase CBF-consistent behavior-cloning
coverage in late mixed-layout stages, retain original random episodes, and
select checkpoints using a held-out S1 validation block before any multi-seed
formal comparison. The model should not yet be presented as stable across
mixed obstacle layouts.

## Representative Success

The curriculum MAPPO checkpoint safely captures probe seed `643019`:

- capture time: `12.4 s`;
- capturing defender: `D3` (zero-index ID 2);
- nearest distance: `0.73 m < 0.80 m`;
- collision: false; boundary violations: 0;
- all four defenders entered the central obstacle zone;
- the exported GIF/MP4 includes the fixed physical capture ring, visual pulse,
  green capture highlight, and a 1.75-second confirmation freeze.

The local media artifact is generated at:
`results/showcase/curriculum_mappo_seed640101_success_seed643019/`.

## Reproduction

```powershell
python scripts/train_capture_radius_recurrent_behavior_cloning.py `
  --config configs/capture_radius_recurrent_behavior_cloning_showcase_curriculum_pilot.yaml `
  --output results/showcase_curriculum/bc_seed640101 `
  --seed 640101 --device cuda

python scripts/train_capture_radius_recurrent_mappo.py `
  --config configs/capture_radius_recurrent_mappo_showcase_curriculum_pilot.yaml `
  --output results/showcase_curriculum/mappo_seed640101 `
  --seed 640101 --device cuda `
  --initialize-from results/showcase_curriculum/bc_seed640101/checkpoint.pt

python scripts/evaluate_mixed_obstacle_showcase.py `
  --method f2 `
  --checkpoint results/showcase_curriculum/mappo_seed640101/checkpoint.pt `
  --output-dir results/showcase/probe_curriculum_mappo_seed640101_cbf_d5_vis14_seed643001_n20 `
  --seed 643001 --episodes 20 `
  --initial-side-distance 5.0 --detection-range 14.0 `
  --target-speed-scale 0.55 --use-cbf --device cuda
```
