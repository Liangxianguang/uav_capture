# DN-MPC P0 Baseline Freeze

**Status:** development-only input freeze; locked tests remain closed

## Purpose

This stage freezes the public scene manifests, environment contract, retained
actor checkpoint, protocol, and the last verified G5 development summary before
any further DN-MPC planner changes or JEPA training. It does not rewrite V3,
V4, V5, M0, M3, or the historical Reliability Ledger.

## Reproduction

```powershell
D:\miniconda3\Scripts\conda.exe run -n uav-encirclement-gpu `
  python scripts/build_dn_mpc_baseline_manifest.py
```

The generated artifacts are intentionally local under:

- `results/dn_mpc_jepa_safe_capture_dev/baseline_manifest.json`
- `results/dn_mpc_jepa_safe_capture_tensorboard/p0_baseline_freeze_seed20260907/`

The script refuses to overwrite an existing manifest or TensorBoard run unless
`--force` is explicitly supplied for the manifest and a new TensorBoard path is
selected. `results/` remains ignored by Git.

## Frozen Inputs

| Input | Role | Records/result |
| --- | --- | ---: |
| `jepa_route_recovery_dev_g1b_seed20260911/scene_manifest.jsonl` | G5 fixed four-scene development manifest | 4 scenes |
| `jepa_safe_capture_l0_l3_paired_smoke_seed20260911_m0/scene_manifest.jsonl` | M0 paired scene manifest | 160 episodes |
| `jepa_safe_capture_l0_l3_paired_smoke_seed20260911_m3/scene_manifest.jsonl` | M3 paired scene manifest | 160 episodes |
| `dn_mpc_cbf_g5_dev_seed20260907_v2/summary.json` | last verified DN-MPC + strict CBF development result | 3/4 safe capture |
| `capture_radius_pursuit_central_v4_flee.yaml` | simulator and capture contract | `dt=0.1`, `250` steps |
| `v5_development_exact_reactive_seed661606.pt` | retained actor used by G5 evaluator | frozen checkpoint |

The exact SHA-256 values and dirty-file snapshot are written to
`baseline_manifest.json`. The manifest records the current worktree as dirty
because unrelated user changes already existed; this is intentional and makes
the provenance limitation explicit.

## TensorBoard Contract

The P0 run records:

- `Baseline/manifest_created`
- `Baseline/g5_scene_records`
- `Baseline/m0_scene_records`
- `Baseline/m3_scene_records`
- `Baseline/g5_safe_capture_rate`
- `Provenance/metadata`

Validation command:

```powershell
D:\miniconda3\Scripts\conda.exe run -n uav-encirclement-gpu `
  tensorboard --inspect `
  --logdir results/dn_mpc_jepa_safe_capture_tensorboard/p0_baseline_freeze_seed20260907
```

## Gate Status

- **Completed:** immutable input/hash manifest and independent TensorBoard run.
- **Completed:** locked-test flag remains `false`; strict CBF and controlled-abort contract is recorded.
- **Pending:** replay parity for the 160-episode M0/M3 manifests. Until that replay is completed, this is a P0 freeze artifact, not a claim that the historical M0/M3 scores have been regenerated.
- **Not started:** JEPA training or Ledger-Lite integration.

The next permitted task is to finish the P0 replay-parity check or, if the
existing G5 trace audit is accepted as the P0 baseline, implement P3 route
state persistence before any JEPA experiment. A failed route or safety gate
must stop model expansion rather than relax CBF margins.
