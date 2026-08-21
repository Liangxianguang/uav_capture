# E1 Execution-Dynamics Pre-registration

## Question and scope

E1 measures whether the three frozen V4 retained-BC policies retain safe cooperative capture when their defender velocity commands pass through a transparent, bounded action-execution model. It compares raw commands, the original kinematic CBF, and an execution-aware CBF. It is a sim-to-sim robustness study, **not** a real-flight, real-controller, PyBullet, SITL, visual-perception, or physical-contact capture claim.

The task remains four pursuers, one evader, a `0.80 m` capture radius, static central cylinder/box/wall obstacles, and the existing partial-observation/communication contract. Cooperative Safe Capture requires no prior obstacle/inter-agent collision or boundary violation and at least two pursuers entering the central obstacle zone. The target need not enter that zone.

## Frozen intervention

The E1 command path is:

```text
policy command -> raw/K-CBF/E-CBF -> FIFO delay -> acceleration/speed limit
               -> bounded execution noise -> defender velocity applied by V4 environment
```

Only defender command execution changes. Target motion, obstacles, task termination, capture radius, observation model and communication model do not change. The wrapper has an E0 disabled-mode regression test that forwards the original action exactly.

| Profile | Delay (steps) | Acceleration scale | Noise std (m/s) | Role |
| --- | ---: | ---: | ---: | --- |
| E0 | 0 | 1.00 | 0.00 | Nominal identity reference |
| E1 | 0 | 0.75 | 0.00 | Reduced braking |
| E2 | 0 | 0.50 | 0.00 | Stronger reduced braking |
| E3 | 1 | 1.00 | 0.00 | One-step command hold |
| E4 | 2 | 1.00 | 0.00 | Two-step command hold |
| E5 | 0 | 1.00 | 0.25 | Bounded noise (`3 sigma` row bound) |
| E6 | 1 | 0.75 | 0.25 | Primary combined stress |

`E0`–`E6`, all three execution modes, and every predeclared S3 case remain in the study regardless of observed outcomes.

## Independent splits and statistics

The smoke/development/locked case blocks are respectively `681001`, `681101`, and `681201`. Each profile uses 10 smoke, 60 development S3, and 100 locked S3 episodes per checkpoint/mode. Fixed S1/S2 scenes use 20 episodes per scene in the formal stage. Cases are generated from the frozen S3 factorial table and record a static `case_sha256`; raw, K-CBF, E-CBF also share the episode and execution-noise seeds.

The principal statistical unit is the independently trained frozen V4 checkpoint (`661201`, `661202`, `661203`), never individual episodes. The final report will give per-checkpoint results, three-seed mean ± sample standard deviation, episode-level Wilson intervals, and paired-case bootstrap intervals for E-CBF minus K-CBF.

## Preconditions and stop rules

The authoritative V4 checkpoint contract is [E1_SOURCE_MANIFEST.json](E1_SOURCE_MANIFEST.json). The checkpoint artifacts are currently absent locally, so no E1 policy development evaluation has started. They must first be restored or regenerated under the frozen V4 configuration and match the recorded SHA-256 values.

Rule expert + E-CBF is run first solely as a feasibility oracle. A profile with less than 95% Cooperative Safe Capture or more than 5% collision/boundary in its 60 development cases is reported as infeasible/over-strong; it is not silently weakened. Any replacement profile requires a new pre-registration and entirely new development/locked seed blocks.

Development outcomes may trigger code-correctness repairs only (identity mismatch, case-pairing mismatch, NaN/Inf, or protocol bug). They cannot change profiles, CBF parameters, strategy weights, source checkpoint selection, or the locked seed block. When implementation and feasibility preconditions hold, locked evaluation is run and reported even if E-CBF does not improve over K-CBF.

## Frozen implementation fingerprints

The precise hashes, source commit, task contract, metrics and exclusions are machine-readable in [E1_EXECUTION_DYNAMICS_PREREGISTRATION.json](E1_EXECUTION_DYNAMICS_PREREGISTRATION.json). Run:

```powershell
python scripts/verify_e1_preregistration.py
```

before commencing policy development evaluation and again immediately before the first locked evaluation.
