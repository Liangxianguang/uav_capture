"""Non-invasive E1 wrapper around the frozen pursuit environment."""

from __future__ import annotations

from typing import Any

import numpy as np

from .execution_dynamics import DefenderExecutionDynamics, ExecutionDynamicsConfig, ExecutionStep
from .pursuit_env import CaptureRadiusPursuit3DEnv


_EXECUTION_SEED_SALT = 0xE1_2026


class ExecutionDynamicsPursuitWrapper:
    """Apply auditable defender execution dynamics before ``env.step``.

    The wrapped environment is never patched.  In disabled mode the exact
    action array is forwarded, so a paired rollout remains byte-for-byte the
    original environment path apart from the additive ``info`` diagnostics.
    """

    def __init__(self, env: CaptureRadiusPursuit3DEnv, execution: ExecutionDynamicsConfig) -> None:
        self.env = env
        self.execution = DefenderExecutionDynamics(
            execution,
            defender_count=int(env.n_defenders),
            dt=float(env.dt),
            nominal_max_speed=float(env.agents["defender_max_speed"]),
            nominal_max_acceleration=float(env.agents["defender_max_acceleration"]),
        )
        self.execution_history: list[dict[str, Any]] = []

    def reset(self, seed: int, record_history: bool = False) -> dict[str, Any]:
        observation = self.env.reset(seed=int(seed), record_history=record_history)
        sequence = np.random.SeedSequence([int(seed), _EXECUTION_SEED_SALT])
        execution_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
        self.execution.reset(seed=execution_seed, initial_velocity=self.env.defender_velocities)
        self.execution_history = []
        return observation

    def step(
        self,
        defender_actions: np.ndarray,
        record_history: bool = False,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        transition = self.execution.execute(defender_actions)
        observation, reward, terminated, truncated, info = self.env.step(
            transition.executed_velocity,
            record_history=record_history,
        )
        audit = transition.audit_dict()
        audit["runtime"] = self.execution.runtime_dict()
        audit["requested_velocity_mps"] = transition.requested_velocity.tolist()
        audit["delayed_velocity_mps"] = transition.delayed_velocity.tolist()
        audit["executed_velocity_mps"] = transition.executed_velocity.tolist()
        audit["noise_velocity_mps"] = transition.noise_velocity.tolist()
        info = {**info, "execution_dynamics": audit}
        if record_history:
            self.execution_history.append(audit)
        return observation, reward, terminated, truncated, info

    @property
    def last_execution_step(self) -> ExecutionStep | None:
        return self.execution.last_step

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
