# Capture-Net Bench Measurement Package

Copy the three CSV templates and `metadata_template.yaml` into a new,
date-stamped directory for one physical net and attachment configuration. Do
not put fabricated or simulated samples in this directory. Keep raw samples,
instrument exports, photographs, and rejected runs alongside the copied files.

Before fitting a flexible-net candidate, review the raw traces and run:

```powershell
cd F:\uav_capture\three_d_encirclement
conda run --no-capture-output --name uav-encirclement-gpu python scripts\validate_capture_net_measurements.py `
  --metadata-yaml F:\measurements\net_YYYYMMDD\metadata.yaml `
  --static-csv F:\measurements\net_YYYYMMDD\static_segment_pull.csv `
  --decay-csv F:\measurements\net_YYYYMMDD\free_decay.csv `
  --impact-csv F:\measurements\net_YYYYMMDD\low_speed_impact.csv `
  --output results\net_measurement_preflight_YYYYMMDD
```

The report proves only that the supplied data meets syntax and fitting
prerequisites. It does not establish material limits, model validity, flight
safety, or capture success. Use `BENCH_CALIBRATION_PROTOCOL.md` for the
physical fixtures and acceptance gates.
