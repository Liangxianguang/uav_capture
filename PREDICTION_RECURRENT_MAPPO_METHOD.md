# Recurrent-MAPPO Method Contract

## Scope

This module implements the Stage 3C method for partial-observation 3D
capture-radius pursuit: a parameter-sharing decentralized recurrent actor,
a centralized training-only critic, optional frozen GRU target prediction, and
a local CBF execution filter.

It remains a kinematic simulation method. The capture event is the existing
geometric `r_capture = 0.80 m` condition; it is not physical contact, net
capture, SITL, or hardware flight.

## Actor and Critic Information

- Each defender runs the same actor parameters with its own local GRU hidden
  state.
- The actor receives only the published local policy observation. With the
  learned prediction configuration this is 52 dimensions, including
  confidence, covariance, message age, frozen GRU target prediction mean, and
  predicted uncertainty.
- The centralized critic receives simulator global state only while training.
  The rollout/evaluation actor never queries it.
- `LearnedPredictionObserver` is frozen and replaces only the fixed 4-column
  prediction feature block. It does not access target truth during execution.

## Residual Recurrent Actor

The actor is intentionally initialized as:

```text
mean_action = MLP_prior(local_observation) + GRU_residual(local_history)
```

The GRU residual output head is initialized to zero. A compatible non-recurrent
behavior-cloning actor can therefore initialize the MLP prior exactly, while
the recurrent state begins as a zero correction. This avoids interpreting poor
from-scratch sequence imitation as evidence against temporal memory, and it
makes the no-memory behavior a controlled starting point.

## Recurrent PPO Protocol

- `scripts/train_capture_radius_recurrent_mappo.py` stores local observation,
  centralized state, action, log probability, GAE target, pre-step hidden
  state, and an episode reset mask for every transition.
- Each PPO rollout is divided into fixed `--sequence-length` chunks. The
  configured rollout length must be divisible by that length.
- During PPO replay, each chunk starts from its stored hidden state; reset
  masks zero the hidden state before the first frame of a new episode.
- The actor loss is computed over ordered chunks, never by independently
  shuffling time frames. Critic values are still optimized per time frame from
  centralized state.
- `tests/test_prediction.py` verifies that sequence replay produces the same
  actor means as step-by-step execution under reset masks.

## Reproducible Commands

```powershell
conda run --no-capture-output -n uav-encirclement-gpu python scripts/train_capture_radius_recurrent_mappo.py `
  --config configs/capture_radius_recurrent_mappo_gru_prediction_pilot.yaml `
  --output results/recurrent_mappo_gru_prediction_pilot `
  --seed 521001 --total-steps 16384 --device cuda --sequence-length 32 `
  --initialize-from results/capture_radius_behavior_cloning_gru_prediction_dev_gpu_seed521001/checkpoint.pt `
  --prediction-checkpoint results/target_predictor_gru_v1/checkpoint.pt `
  --prediction-history-length 8 --prediction-horizon-index 2
```

Evaluate the frozen checkpoint with the same normal evaluator. The evaluator
selects the recurrent actor from `actor_recurrent=true` in the checkpoint.

## Required Final Ablation

The implementation pilot is not a final method result. The final Stage 3C
comparison must use at least three training seeds and 300 locked episodes per
seed for:

1. residual Recurrent-MAPPO without learned prediction;
2. residual Recurrent-MAPPO with the frozen GRU prediction; and
3. each method with raw action and local CBF execution.

The final report must distinguish the MLP prior, recurrent residual,
prediction feature, and CBF contributions.
