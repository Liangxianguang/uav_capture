# DN-MPC P0 Baseline Freeze: G5 Chunk-5 Contract

**Status:** development-only input freeze; locked tests remain closed

## Scope

This freeze records the latest strict DN-MPC + CBF G5 development contract
before any new planner, JEPA, or Ledger-Lite expansion. It is a new manifest
alongside the earlier P0 artifact and does not overwrite or reinterpret the
older `baseline_manifest.json`.

## Reproduction

```powershell
D:\miniconda3\envs\uav-encirclement-gpu\python.exe `
  scripts/build_dn_mpc_baseline_manifest.py `
  --output results/dn_mpc_jepa_safe_capture_dev/baseline_manifest_g5_chunk5_seed20260907.json `
  --tensorboard-dir results/dn_mpc_jepa_safe_capture_tensorboard/p0_baseline_freeze_g5_chunk5_seed20260907 `
  --g5-manifest results/jepa_route_recovery_dev_g1b_seed20260911/scene_manifest.jsonl `
  --g5-summary results/dn_mpc_cbf_g5_contract_audit_seed20260907_chunk5/summary.json `
  --m0-manifest results/jepa_safe_capture_l0_l3_paired_smoke_seed20260911_m0/scene_manifest.jsonl `
  --m3-manifest results/jepa_safe_capture_l0_l3_paired_smoke_seed20260911_m3/scene_manifest.jsonl
```

The script only records inputs and writes an independent TensorBoard run. It
does not open a locked test or execute the simulator.

## Frozen inputs and result

| Input | Role | Records/result |
| --- | --- | ---: |
| `jepa_route_recovery_dev_g1b_seed20260911/scene_manifest.jsonl` | G5 fixed scene manifest | 4 scenes |
| `dn_mpc_cbf_g5_contract_audit_seed20260907_chunk5/summary.json` | latest strict chunk-5 G5 summary | 4/4 safe capture |
| `jepa_safe_capture_l0_l3_paired_smoke_seed20260911_m0/scene_manifest.jsonl` | M0 paired manifest | 160 episodes |
| `jepa_safe_capture_l0_l3_paired_smoke_seed20260911_m3/scene_manifest.jsonl` | M3 paired manifest | 160 episodes |
| `capture_radius_pursuit_central_v4_flee.yaml` | simulator/capture contract | `dt=0.1`, `250` steps |
| `v5_development_exact_reactive_seed661606.pt` | retained actor checkpoint | frozen input |

The G5 summary reports `safe_capture_rate=1.0`, mean capture time `17.1 s`,
zero collision, boundary, pairwise, raw-unverified, controlled-abort and
fallback events, and worst minimum clearance `0.35017664699464446 m`.

The generated manifest SHA-256 is:

```text
3d1a3b1406d9092bf94d459d32b395daee59995673296f900d9a09be4871e55a
```

## Provenance

The run used `D:\miniconda3\envs\uav-encirclement-gpu` with Python 3.11.16,
PyTorch 2.7.1+cu128, CUDA 12.8, TensorBoard 2.19.0, NumPy 2.1.3, and an
NVIDIA GeForce RTX 5050. The manifest records the exact input hashes and the
current worktree dirty-file snapshot. Existing unrelated worktree changes are
preserved and are not part of this freeze.

TensorBoard output:

```text
results/dn_mpc_jepa_safe_capture_tensorboard/p0_baseline_freeze_g5_chunk5_seed20260907/
```

Required tags are `Baseline/manifest_created`, the three scene-record tags,
`Baseline/g5_safe_capture_rate`, and `Provenance/metadata`.

## Gate decision

- **Passed:** independent manifest, input hashes, environment metadata, and
  TensorBoard provenance were written.
- **Passed:** `development_only=true`, `locked_test_opened=false`, strict CBF,
  no raw-unverified actions, and controlled-abort preservation are recorded.
- **Pending:** M0/M3 replay-parity execution against these frozen manifests.
- **Blocked by evidence:** no JEPA training or L1-L3 expansion until replay
  parity and candidate-contract checks are complete.

The G5 `4/4` result is a development baseline, not a new locked benchmark.
