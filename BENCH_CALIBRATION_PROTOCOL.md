# Flexible-Net Bench Calibration Protocol

## Purpose

This protocol obtains traceable inputs for the flexible-net simulation. It
does not validate flight capture by itself. A parameter candidate is never a
hardware claim and must be validated in a separate net-sag and low-speed
target-contact experiment.

## Required CSV Measurements

Use SI units and preserve raw samples without smoothing or overwriting.

| Test | Required CSV columns | Fixture requirement |
|---|---|---|
| Representative segment pull | `extension_m,force_n` | Same cord, knot, termination, and nominal segment length as the planned mesh edge. |
| Free decay | `time_s,displacement_m` | Known moving mass, small displacement, released without continuing contact. |
| Low-speed impact | `time_s,force_n` | Instrumented target contact at a recorded low approach speed. |

Record net mass, attachment configuration, load-cell calibration, sampling
rate, ambient conditions, test operator, and each rejected run alongside the
CSV files. Determine the safe-working tension and maximum allowable strain
from separate material and attachment-strength tests, not from a single
low-speed impact.

## Fitting

Run only after raw-data review:

```powershell
cd F:\uav_capture\three_d_encirclement
conda run --no-capture-output --name uav-encirclement-gpu python scripts\fit_capture_net_calibration.py `
  --static-csv F:\measurements\static_segment_pull.csv `
  --decay-csv F:\measurements\free_decay.csv `
  --impact-csv F:\measurements\low_speed_impact.csv `
  --total-moving-net-mass-kg 0.08 `
  --measured-pretension-n 0.00 `
  --safe-working-tension-n 10.0 `
  --maximum-strain 0.15 `
  --output results\net_calibration_candidate_YYYYMMDD
```

The script writes raw-input SHA256 hashes, fit quality, impact metrics, and a
`calibration_candidate.yaml`. That YAML is intentionally incomplete and
labelled `candidate_requires_review`; merge values into a new simulation
configuration only after checking fixture equivalence and residuals.

## Acceptance Before Flexible-Net Evaluation

1. Static stiffness fit has a positive slope and reviewed residuals.
2. Free-decay trace contains at least three resolved, decaying peaks.
3. Tension and strain limits cover cords, knots, anchors, and deployment
   hardware with an explicit safety factor.
4. A held-out physical sag/impact test agrees with the selected simulator
   parameters within a predeclared tolerance.
5. Only then create a new flexible-net development split. Do not modify the
   existing 430xxx, 432xxx, or 433xxx evidence directories.
