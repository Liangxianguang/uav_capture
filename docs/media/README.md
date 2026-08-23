# Representative Capture Media

These files show a successful **V5 development** replay, not a formal locked
test result. The checkpoint is `models/v5_development_exact_reactive_seed661606.pt`.
It was evaluated on S3 validation episode `0` (`episode_seed=646101`,
`layout_seed=1646101`) with CBF enabled and a recurrent-state reset interval of
one control step.

The scene contains one cylinder, one wall, and one box. It terminates in a
safe capture: a defender reaches the `0.80 m` capture radius with no collision
and no boundary violation. The capture snapshot/GIF/MP4 are visual evidence of
one replay, whereas rates must be read from the development status and formal
reports.

The release verification reran the complete V5 validation block in this
repository on 2026-08-23: `57/60` safe captures (`95.0%`), collision `0%`,
boundary violation `0%`, and Transit `100%`. This remains development-only
evidence because the checkpoint is a single training seed. In the displayed
episode, capture occurs at `4.8 s` with final nearest distance `0.78284 m`.

- `v5_development_s3_episode0_capture_3d.png`: final 3-D capture frame.
- `v5_development_s3_episode0_capture_3d.gif`: rendered 3-D animation.
- `v5_development_s3_episode0_capture_3d.mp4`: H.264 version of the same
  animation.

Regenerate the media with the two commands in the repository README. The
checkpoint SHA-256 is
`535098773be05687e147043435649378532362d479bdc0375842970370ba40ba`.
