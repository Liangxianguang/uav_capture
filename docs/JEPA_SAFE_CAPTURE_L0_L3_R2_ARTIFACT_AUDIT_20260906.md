# JEPA Safe-Capture L0-L3 R2 Artifact Audit

This document is the reproducibility entry point for the read-only artifact
audit. Run it after the paired evaluation and aggregate have completed:

```powershell
& 'D:\download\anaconda3\envs\traj_pred_prep\python.exe' scripts/run_with_tensorboard_compat.py scripts/audit_jepa_safe_capture_l0_l3_artifacts.py `
  --runs-root results/l0_l3_r2_full_runs `
  --tensorboard-root results/l0_l3_r2_full_tensorboard `
  --aggregate-dir results/l0_l3_r2_full_aggregate `
  --output-dir results/l0_l3_r2_artifact_audit `
  --episodes-per-run 64 --development-only
```

The audit checks all 21 seed/variant run directories, 64 episode rows and
64 step traces per run, paired canonical scene-manifest SHA-256, non-empty
per-run TensorBoard events, aggregate provenance, and the hard safety gate for
all CBF-enabled variants. It never opens or modifies a locked-test split.

## Recorded audit

The audit was executed against the local R2 outputs on 2026-09-06:

- `21/21` runs and `1,344/1,344` step traces passed;
- `21/21` per-run TensorBoard directories and the aggregate event file were present;
- canonical scene-manifest SHA-256: `748d706ae6c2a064c92620200fe4c125a5be6f357c8bf85d7642235efee7a520`;
- CBF-enabled safety hard gate: `PASS` (`collision=0`, `boundary=0`,
  `pairwise=0`, `raw_unverified=0`);
- audit summary SHA-256:
  `ad0a44d8fccc6ae8e11171d34e0fa802acc0464ba6bd2ed16c2cb1120301f342`.

The machine-readable output is `results/l0_l3_r2_artifact_audit/summary.json`
and its TensorBoard log is under
`results/l0_l3_r2_artifact_audit/tensorboard/`.
