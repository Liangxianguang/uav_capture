# Central V4 D1 Validation Aggregate

Date: 2026-08-21

This report combines fixed S1/S2 regression and S3 validation artifacts.
It is validation evidence only: no locked-test or multi-seed claim is made.

## Candidate and decision

- The retained S3 BC candidate is seed `661201`. Its CBF execution stack
  achieves `34/40` (`85.0%`) Cooperative Safe Capture on S3 validation.
- The retained MAPPO pilot is seed `661301`, trained for `65,536` environment
  steps from the retained BC checkpoint. Checkpoint SHA-256:
  `32c73c85f405b9bf714fce7ab56d1c7e4aaa52b733196eb8512683b8418edcd8`.
- MAPPO+CBF reaches only `30/40` (`75.0%`) on the same S3 validation block,
  ten percentage points below retained BC+CBF. MAPPO also regresses on the
  fixed S1 wall CBF block (`18/20`) and on raw cylinder/wall safety.
- The retained MAPPO pilot is therefore rejected. The retained BC+CBF stack
  remains the frozen formal candidate. Its raw actor is not accepted as a safe
  standalone policy because S3 raw collision rate is `92.5%`.
- D1 is complete. D2 must replicate the frozen retained-BC recipe with three
  independent training seeds before touching the locked-test seed block.

Independent Transit succeeds in every listed artifact. This confirms map
feasibility but is not counted as capture success.

| Group | Artifact | Episodes | Cooperative safe capture | Capture | Collision | Boundary | Transit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | `expert_s1_cylinder_raw_20` | 20 | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% |
| S1 | `expert_s1_cylinder_cbf_20` | 20 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| S1 | `expert_s1_box_raw_20` | 20 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| S1 | `expert_s1_box_cbf_20` | 20 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| S1 | `expert_s1_wall_raw_20` | 20 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| S1 | `expert_s1_wall_cbf_20` | 20 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| S1 | `bc_s3_retained_seed661201_s1_cylinder_raw_20` | 20 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| S1 | `bc_s3_retained_seed661201_s1_cylinder_cbf_20` | 20 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| S1 | `bc_s3_retained_seed661201_s1_box_raw_20` | 20 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| S1 | `bc_s3_retained_seed661201_s1_box_cbf_20` | 20 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| S1 | `bc_s3_retained_seed661201_s1_wall_raw_20` | 20 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| S1 | `bc_s3_retained_seed661201_s1_wall_cbf_20` | 20 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| S1 | `mappo_s3_retained_seed661301_pilot_s1_cylinder_raw_20` | 20 | 95.0% | 100.0% | 5.0% | 0.0% | 100.0% |
| S1 | `mappo_s3_retained_seed661301_pilot_s1_cylinder_cbf_20` | 20 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| S1 | `mappo_s3_retained_seed661301_pilot_s1_box_raw_20` | 20 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| S1 | `mappo_s3_retained_seed661301_pilot_s1_box_cbf_20` | 20 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| S1 | `mappo_s3_retained_seed661301_pilot_s1_wall_raw_20` | 20 | 95.0% | 95.0% | 5.0% | 0.0% | 100.0% |
| S1 | `mappo_s3_retained_seed661301_pilot_s1_wall_cbf_20` | 20 | 90.0% | 90.0% | 10.0% | 10.0% | 100.0% |
| S2 | `expert_s2_v4_raw_20` | 20 | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% |
| S2 | `expert_s2_v4_cbf_20_frozen` | 20 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| S2 | `bc_s3_retained_seed661201_s2_raw_20` | 20 | 50.0% | 50.0% | 50.0% | 0.0% | 100.0% |
| S2 | `bc_s3_retained_seed661201_s2_cbf_20` | 20 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| S2 | `mappo_s3_retained_seed661301_pilot_s2_raw_20` | 20 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| S2 | `mappo_s3_retained_seed661301_pilot_s2_cbf_20` | 20 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| S3 | `s3_validation_expert_raw_40` | 40 | 5.0% | 5.0% | 95.0% | 0.0% | 100.0% |
| S3 | `s3_validation_expert_cbf_40` | 40 | 87.5% | 87.5% | 5.0% | 0.0% | 100.0% |
| S3 | `s3_validation_bc_retained_raw_40` | 40 | 7.5% | 10.0% | 92.5% | 0.0% | 100.0% |
| S3 | `s3_validation_bc_retained_cbf_40` | 40 | 85.0% | 85.0% | 2.5% | 2.5% | 100.0% |
| S3 | `s3_validation_mappo_retained_raw_40` | 40 | 12.5% | 12.5% | 87.5% | 0.0% | 100.0% |
| S3 | `s3_validation_mappo_retained_cbf_40` | 40 | 75.0% | 75.0% | 0.0% | 0.0% | 100.0% |

Raw actor and CBF execution are listed as separate artifacts. CBF improvements are not attributed to the policy network.
