"""Non-learning baseline controllers for the 3D benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np

from .environment import Encirclement3DEnv, _unit
from .safety import DiscreteTimeCBFSafetyFilter, SafetyFilterDiagnostics


class TetrahedralSlotController:
    """Tracks fixed tetrahedral target slots with local obstacle repulsion."""

    def __init__(self, env: Encirclement3DEnv):
        self.env = env
        self.task = env.task
        self.max_speed = float(env.agents["defender_max_speed"])
        dynamics = env.config.get("dynamics", {})
        backend = str(dynamics.get("backend", "kinematic"))
        self.slot_tracking_gain = float(
            dynamics.get("slot_tracking_gain", self.task["slot_tracking_gain"])
        ) if backend in {"inertial", "pybullet"} else float(self.task["slot_tracking_gain"])
        self.target_velocity_feedforward = float(
            dynamics.get("target_velocity_feedforward", 0.0)
        ) if backend in {"inertial", "pybullet"} else 0.0

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        positions = observation["defender_positions"]
        slots = observation["slot_positions"]
        actions = np.zeros_like(positions)
        avoidance_distance = float(self.task["obstacle_avoidance_distance"])
        obstacle_gain = float(self.task["obstacle_avoidance_gain"])
        inter_agent_gain = float(self.task["inter_agent_avoidance_gain"])

        for index, position in enumerate(positions):
            desired = self.slot_tracking_gain * (slots[index] - position)
            desired += self._target_velocity_feedforward(observation)
            for obstacle in self.env.obstacles:
                clearance, normal = self.env._cylinder_clearance_and_normal(position, obstacle)
                if clearance < avoidance_distance:
                    desired += normal * (avoidance_distance - clearance) * obstacle_gain

            for other_index, other_position in enumerate(positions):
                if index == other_index:
                    continue
                separation = position - other_position
                distance = float(np.linalg.norm(separation))
                if distance < float(self.task["safety_distance"]):
                    desired += _unit(separation) * (float(self.task["safety_distance"]) - distance) * inter_agent_gain
            actions[index] = desired

        return self.env._clip_rows(actions, self.max_speed)

    def _target_velocity_feedforward(self, observation: dict[str, Any]) -> np.ndarray:
        """Return the target-motion compensation used by the nominal expert.

        Keeping this as a hook lets a phase-aware controller change only the
        target-motion term while preserving the collision and obstacle
        repulsion logic of the audited baseline.
        """
        return self.target_velocity_feedforward * np.asarray(observation["target_velocity"])


class HoldAwareTetrahedralSlotController(TetrahedralSlotController):
    """Use target-velocity feedforward only after entering the hold phase.

    The always-on feedforward sweep was unstable on independent PyBullet
    seeds.  This controller separates approach from containment: it first
    converges to the tetrahedral slots using the audited baseline, then adds a
    bounded target-motion term once all slot errors are inside an activation
    radius.  Hysteresis prevents rapid switching when the target moves near
    the tolerance boundary.
    """

    def __init__(self, env: Encirclement3DEnv):
        super().__init__(env)
        dynamics = env.config.get("dynamics", {})
        self.hold_activation_error = float(dynamics.get("hold_activation_error", 0.95))
        self.hold_exit_error = float(
            dynamics.get("hold_exit_error", max(self.hold_activation_error + 0.15, 1.10))
        )
        self.hold_feedforward_scale = float(
            dynamics.get("hold_feedforward_scale", self.target_velocity_feedforward)
        )
        self.hold_mode = False
        if self.hold_activation_error <= 0.0 or self.hold_exit_error < self.hold_activation_error:
            raise ValueError("hold_exit_error must be >= a positive hold_activation_error.")

    def _target_velocity_feedforward(self, observation: dict[str, Any]) -> np.ndarray:
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        slots = np.asarray(observation["slot_positions"], dtype=np.float64)
        max_slot_error = float(np.max(np.linalg.norm(positions - slots, axis=1)))
        if not self.hold_mode and max_slot_error <= self.hold_activation_error:
            self.hold_mode = True
        elif self.hold_mode and max_slot_error > self.hold_exit_error:
            self.hold_mode = False
        if not self.hold_mode:
            return np.zeros(3, dtype=np.float64)
        return self.hold_feedforward_scale * np.asarray(observation["target_velocity"])


class CaptureAwareTetrahedralSlotController(TetrahedralSlotController):
    """Rule baseline that commands closure only from a validated cage geometry.

    The velocity controller remains the audited tetrahedral slot controller.
    The additional binary command is deliberately separated from continuous
    motion, so a rejected close request cannot silently change the flight
    action.  It provides a non-learning feasibility baseline for the capture
    proxy before any policy-training change is attempted.
    """

    def __init__(self, env: Encirclement3DEnv):
        super().__init__(env)
        dynamics = env.config.get("dynamics", {})
        self.capture_hold_feedforward = float(dynamics.get("capture_hold_feedforward", 1.0))

    def should_close(self, observation: dict[str, Any]) -> bool:
        del observation  # The environment owns the authoritative geometry state.
        return self.env.capture_close_feasible()

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        actions = super().act(observation)
        if self.env.capture_closed:
            # Once the cage has accepted closure, keep the tetrahedron moving
            # with the target instead of continuing an open-phase policy.
            # This is a fixed safety controller, not a learned action head.
            actions += self.capture_hold_feedforward * np.asarray(observation["target_velocity"])[None, :]
            actions = self.env._clip_rows(actions, self.max_speed)
        return actions


class SpatialContainmentTetrahedralSlotController(CaptureAwareTetrahedralSlotController):
    """Semantic main-task name for the virtual tetrahedral closure controller.

    This controller deliberately keeps the audited behavior of
    :class:`CaptureAwareTetrahedralSlotController`: the target is a kinematic
    evader and the closure event is a radius-aware geometric constraint.  The
    separate name prevents reports for the primary research task from implying
    a physical net deployment or vehicle-to-target contact.
    """


class CBFSafetyFilteredSlotController:
    """Apply the CBF-QP filter to the same actions as the rule baseline."""

    def __init__(self, env: Encirclement3DEnv):
        self.nominal_controller = TetrahedralSlotController(env)
        self.safety_filter = DiscreteTimeCBFSafetyFilter(env)
        self.last_diagnostics = SafetyFilterDiagnostics(0.0, float("inf"), True, False)

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        nominal_actions = self.nominal_controller.act(observation)
        safe_actions, self.last_diagnostics = self.safety_filter.filter(nominal_actions, observation)
        return safe_actions
