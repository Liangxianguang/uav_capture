# DN-MPC P14 Offline Virtual-Probe Calibration

**Date:** 2026-09-08
**Status:** development-only calibration complete; online training and locked evaluation remain closed.

## Purpose

P13 established that constant-velocity pairwise TTC warnings were not the same
as an operational strict-margin violation or verified CBF infeasibility. P14
therefore collects unsafe counterfactual probes on copied mathematical states.
The requested probe action is never sent to `env.step`, and no raw-unverified
action is executed. The archive is intended to provide strict-margin positives
for a future outcome-conditioned JEPA calibration gate.

## Contract

| Item | Value |
|---|---|
| protocol | `central_random_mixed_obstacle_s3_route_v1_p14_virtual_probe` |
| split | `calibration` (development-only) |
| episodes | 20 |
| horizon / chunk | 5 / 5 steps |
| pairwise margin | 0.35 m |
| source actor | `models/v5_development_exact_reactive_seed661606.pt` |
| actor reset | every step (`sequence_length=1`) |
| CBF horizon | 5 |
| raw-unverified execution | prohibited |
| locked test | not opened |

## Results

The immutable virtual-probe archive contains `11,144` rows (`11,144 x 5`
horizon labels):

| Metric | Result |
|---|---:|
| strict-margin positive cells | `8,673 / 55,720 = 15.57%` |
| strict-margin positive rows | `40.33%` |
| CBF-infeasible cells | `42,536 / 55,720 = 76.34%` |
| `converge_pair` rows | `9,552` |
| `safe_hold` rows | `1,592` |
| `safe_hold` strict-positive row rate | `6.34%` |
| branch-failure cells | `0` (expected for virtual probes) |
| raw-unverified actions executed | `false` |

The source dataset SHA-256 is
`4e9f1977263d49ac554f1c4e7ed00d1b4832452d9b8e96c9a3cbfd996bc09105`.
After the explicit four-way label contract was materialized, the labelled
dataset SHA-256 is
`9be0e94711f0afed218799be693d7d778ce33708499d733dcbbe744d70d3feb9`.

The four labels remain disjoint in meaning:

- `labels_predicted_ttc_hazard`: a warning under constant-relative-velocity
  extrapolation;
- `labels_strict_margin_violation`: physical clearance below 0.35 m;
- `labels_cbf_infeasible`: the primary CBF feasibility result is false or
  non-finite;
- `labels_branch_failure`: an actual counterfactual branch termination.

P14 has no branch-failure positives because the virtual state probe does not
execute a branch. This is not evidence that branches are safe; it is an
explicit limitation of this calibration archive.

## Gate decision

P14 passes the **data-availability** gate for strict-margin positives and is
usable for calibration diagnostics. It does **not** yet pass the promotion
gate for online JEPA training or Ledger-Lite because the archive is dominated
by CBF-infeasible probes (`76.34%`) and has no branch-failure labels. Before
training, add a balanced, independently generated validation archive containing
verified feasible near-misses and executed/fully filtered counterfactual branch
outcomes. Keep the analytic DN-MPC + strict joint CBF path as the only
runtime-eligible controller until that gate passes.

## Artifacts and provenance

- Collector: `scripts/collect_dn_mpc_pairwise_virtual_probe_archive.py`
- Materializer: `scripts/materialize_dn_mpc_pairwise_label_contract.py`
- Protocol: `configs/central_random_mixed_obstacle_s3_route_v1_p14_virtual_probe_protocol.yaml`
- Source archive: `results/dn_mpc_jepa_safe_capture_dev/p14_pairwise_virtual_probe_calibration/pairwise_virtual_probe_counterfactual.npz`
- Label archive: `results/dn_mpc_jepa_safe_capture_dev/p14_pairwise_virtual_probe_labels/route_identity_pairwise_outcome_labels.npz`
- TensorBoard logs: `results/dn_mpc_jepa_safe_capture_tensorboard/p14_pairwise_virtual_probe_calibration` and `.../p14_pairwise_virtual_probe_labels`

Generated result files remain local and are intentionally not committed. The
source code, protocol, tests, and this audit report are the reproducible
repository artifacts.
