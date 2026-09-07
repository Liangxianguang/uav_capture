# DN-MPC S1 Replay and Artifact Inventory

**Status:** development-only; locked tests remain closed

## S1 replay

The frozen G5 scene manifest was replayed with the analytic DN-MPC planner and
strict Joint CBF, using the same actor, protocol, environment, route chunk
length, and CBF horizon as the P0 baseline. JEPA and Ledger-Lite remained
disabled.

| Metric | S1 result |
| --- | ---: |
| Episodes | 4 |
| Safe capture | 4/4 (100.0%) |
| Mean capture time | 17.1 s |
| Route switches | 75 |
| Collision | 0 |
| Defender boundary violation | 0 |
| Pairwise violation | 0 |
| Raw-unverified actions | 0 |
| Controlled abort | 0 |
| CBF fallback | 0 |
| Worst minimum clearance | 0.35017664699464446 m |
| CBF probe checks / rejected | 5663 / 139 |

The replay output is isolated at:

```text
results/dn_mpc_cbf_s1_g5_replay_parity_seed20260907_chunk5/
results/dn_mpc_jepa_safe_capture_tensorboard/s1_g5_replay_parity_seed20260907_chunk5/
```

TensorBoard inspection found aggregate and per-episode safety, capture,
route-switch, clearance, and provenance tags. The replay was executed with
Python 3.11.16, PyTorch 2.7.1+cu128, CUDA 12.8, TensorBoard 2.19.0, and the
RTX 5050. The recorded code revision is `e17932e`.

## Gate decision

S1 passes the current development gate: it reproduces the frozen `4/4` G5
result without a safety event or controlled-abort increase. This permits the
next planner-only stage, route-state/nominal L0-L1 replay. It does **not**
authorize JEPA training, Ledger-Lite integration, or a locked benchmark.

## S2 planner-only replay

The generic manifest replay entry point
(`scripts/evaluate_dn_mpc_cbf_scene_manifest.py`) was added so dedicated
development manifests can be replayed without regenerating them under the S3
protocol. It validates each scene hash and geometry, then uses the same
reachable projection, candidate-level CBF probes, independent selected/
nominal/safe-hold probes, strict final CBF, and first-step execution contract.

| Frozen slice | Episodes | Safe capture | Mean time | Route switches | Safety events | Worst clearance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L0 open, `horizon=5` | 8 | 8/8 (100.0%) | 5.900 s | 40 | 0 | 0.353859 m |
| L0 single obstacle, `horizon=5` | 8 | 8/8 (100.0%) | 8.725 s | 75 | 0 | 0.350302 m |
| L1 nominal, 3 obstacles, `horizon=5` | 8 | 8/8 (100.0%) | 11.188 s | 129 | 0 | 0.350664 m |

For all three slices, collision, defender boundary, pairwise, raw-unverified,
controlled-abort, and CBF fallback counts were zero. The route probe rejection
counts were respectively 149/3776, 689/6874, and 1135/7907; these are
candidate-level rejections and did not produce an execution safety event.

The output and TensorBoard runs are isolated under:

```text
results/dn_mpc_cbf_s2_l0_open_replay_seed20260907_h5/
results/dn_mpc_cbf_s2_l0_single_obstacle_seed20260907_h5/
results/dn_mpc_cbf_s2_l1_nominal_seed20260907_h5/
results/dn_mpc_jepa_safe_capture_tensorboard/s2_l0_open_replay_seed20260907_h5/
results/dn_mpc_jepa_safe_capture_tensorboard/s2_l0_single_obstacle_seed20260907_h5/
results/dn_mpc_jepa_safe_capture_tensorboard/s2_l1_nominal_seed20260907_h5/
```

These are development slices from the frozen M0 scene manifest, not new
locked-test claims. S2 passes the planner-only gate for these slices. The
route-switch count rises with obstacle complexity, so the next review should
inspect route identity/side persistence before any JEPA integration.

## Artifact inventory

The inventory was collected before cleanup so that deletion did not erase
provenance:

| Location | Files | Directories | Size | Policy |
| --- | ---: | ---: | ---: | --- |
| `results/` | 16,581 | 3,243 | 8.62 GB | retain; contains historical evidence and TensorBoard runs |
| `tmp/` | 1,236 | 100 | 59.74 MB | retain frozen V4/V5 archives and audit inputs; review other items individually |
| `.pytest_cache/` | 6 | 2 | 0.06 MB | disposable test cache; cleanup command blocked by host policy |
| `__pycache__/` | generated caches | 67 directories | generated | disposable Python caches; cleanup command blocked by host policy |
| `results/v12_aggregate_input/` | 0 | 0 | 0 | empty generated placeholder; cleanup command blocked by host policy |

The two frozen expert archives under `tmp/` remain untouched:

- `tmp/v4_archive_recovery_output/expert_sequence_dataset.npz`
- `tmp/v4_s3_archive_recovery_output/expert_sequence_dataset.npz`

Zero-byte logs and historical result directories were **not** deleted in this
pass. Some are referenced by prior failure audits, and their evidentiary role
has not yet been verified. They remain candidates for a later quarantine pass
after a path/reference check.

## Cleanup rule

No material artifact was deleted in this pass because the host rejected the
cleanup command before execution. No checkpoint, archive, scene manifest,
TensorBoard event, trace, report, or non-empty result directory was touched.
Future cleanup must first update this inventory and verify references from
reports and manifests; disposable caches can then be removed with an approved
local cleanup operation.
