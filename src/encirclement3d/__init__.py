"""Reproducible 3D encirclement and capture-radius pursuit benchmarks."""

__all__ = ["CaptureRadiusPursuit3DEnv", "Encirclement3DEnv", "PyBulletEncirclement3DEnv"]


def __getattr__(name: str):
    # Do not load the historical PyBullet/containment stack when a lightweight
    # pursuit-only training worker imports encirclement3d.learning.
    if name == "CaptureRadiusPursuit3DEnv":
        from .pursuit_env import CaptureRadiusPursuit3DEnv

        return CaptureRadiusPursuit3DEnv
    if name == "Encirclement3DEnv":
        from .environment import Encirclement3DEnv

        return Encirclement3DEnv
    if name == "PyBulletEncirclement3DEnv":
        from .pybullet_env import PyBulletEncirclement3DEnv

        return PyBulletEncirclement3DEnv
    raise AttributeError(name)
