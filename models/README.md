# Released Checkpoint

`v5_development_exact_reactive_seed661606.pt` is the best current V5
development checkpoint. It is included for reproducible evaluation and media
generation only. It must not be presented as a locked-test model or a
three-seed result.

Use `configs/central_random_mixed_obstacle_s3_v5_protocol.yaml` and the
commands in the repository README. The checkpoint embeds
`recurrent_reset_interval_steps=1`; the evaluation and replay CLIs honour that
metadata unless an explicit compatible value is supplied.

SHA-256: `535098773be05687e147043435649378532362d479bdc0375842970370ba40ba`
