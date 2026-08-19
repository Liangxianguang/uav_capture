"""PyBullet soft-body infrastructure for a tetrahedral capture net.

This module creates and reads a bending/self-collision soft body. It requires
the PyBullet client to have been reset with ``RESET_USE_DEFORMABLE_WORLD``
before any bodies are created. It does not yet connect that body to the capture
task's target collision object or to PX4; it is a separately testable backend
needed before those integrations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .flexible_net import tetrahedral_surface_mesh_topology


@dataclass(frozen=True)
class TetrahedralSoftBodyMesh:
    """Exported OBJ topology with defender-anchor vertices first."""

    path: Path
    vertices: int
    triangles: int
    face_subdivisions: int


@dataclass(frozen=True)
class PyBulletSoftBodyConfig:
    """PyBullet soft-body settings, pending physical calibration."""

    mass_kg: float
    spring_elastic_stiffness: float
    spring_damping_stiffness: float
    friction_coefficient: float = 0.5
    use_self_collision: bool = True
    use_face_contact: bool = True

    def __post_init__(self) -> None:
        for name in ("mass_kg", "spring_elastic_stiffness", "spring_damping_stiffness"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if not 0.0 <= float(self.friction_coefficient) <= 1.0:
            raise ValueError("friction_coefficient must lie in [0, 1].")


def _anchors(anchors: np.ndarray) -> np.ndarray:
    value = np.asarray(anchors, dtype=np.float64)
    if value.shape != (4, 3) or not np.all(np.isfinite(value)):
        raise ValueError("anchors must contain four finite world-frame 3D positions.")
    return value


def write_tetrahedral_softbody_obj(
    path: Path,
    anchors: np.ndarray,
    *,
    face_subdivisions: int,
) -> TetrahedralSoftBodyMesh:
    """Export a closed, shared-node triangular tetrahedral surface OBJ.

    The caller supplies a fresh artifact path. Refusing to overwrite prevents
    an experiment's generated mesh from silently changing after a run starts.
    """
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite soft-body mesh: {output}")
    anchor_positions = _anchors(anchors)
    weights, triangles, _panels = tetrahedral_surface_mesh_topology(face_subdivisions)
    vertices = np.sum(weights[:, :, None] * anchor_positions[None, :, :], axis=1)
    lines = ["# Generated closed tetrahedral capture-net surface"]
    lines.extend(f"v {point[0]:.17g} {point[1]:.17g} {point[2]:.17g}" for point in vertices)
    # OBJ uses one-based indexing.
    lines.extend(f"f {first + 1} {second + 1} {third + 1}" for first, second, third in triangles)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="ascii")
    return TetrahedralSoftBodyMesh(
        path=output,
        vertices=len(vertices),
        triangles=len(triangles),
        face_subdivisions=face_subdivisions,
    )


def load_tetrahedral_softbody(
    pybullet: Any,
    mesh: TetrahedralSoftBodyMesh,
    config: PyBulletSoftBodyConfig,
    *,
    physics_client_id: int,
) -> int:
    """Load an OBJ net with bending springs, mass springs, and self contact."""
    body_id = pybullet.loadSoftBody(
        str(mesh.path),
        basePosition=[0.0, 0.0, 0.0],
        scale=1.0,
        mass=float(config.mass_kg),
        useNeoHookean=0,
        useBendingSprings=1,
        useMassSpring=1,
        springElasticStiffness=float(config.spring_elastic_stiffness),
        springDampingStiffness=float(config.spring_damping_stiffness),
        springDampingAllDirections=1,
        useSelfCollision=int(config.use_self_collision),
        frictionCoeff=float(config.friction_coefficient),
        useFaceContact=int(config.use_face_contact),
        physicsClientId=physics_client_id,
    )
    if int(body_id) < 0:
        raise RuntimeError("PyBullet could not load the tetrahedral soft body.")
    return int(body_id)


def attach_tetrahedral_softbody_anchors(
    pybullet: Any,
    softbody_id: int,
    anchor_body_ids: np.ndarray | list[int],
    *,
    physics_client_id: int,
) -> list[int]:
    """Attach OBJ vertices 0--3 to the four defender or test-anchor bodies."""
    body_ids = np.asarray(anchor_body_ids, dtype=np.int64)
    if body_ids.shape != (4,):
        raise ValueError("anchor_body_ids must provide exactly four body IDs.")
    constraints: list[int] = []
    for vertex_index, body_id in enumerate(body_ids):
        constraint = pybullet.createSoftBodyAnchor(
            int(softbody_id),
            vertex_index,
            int(body_id),
            -1,
            [0.0, 0.0, 0.0],
            physicsClientId=physics_client_id,
        )
        constraints.append(int(constraint))
    return constraints


def softbody_vertices(pybullet: Any, softbody_id: int, *, physics_client_id: int) -> np.ndarray:
    """Read the current simulation mesh in the same vertex order as the OBJ."""
    mesh_data = pybullet.getMeshData(
        int(softbody_id),
        flags=pybullet.MESH_DATA_SIMULATION_MESH,
        physicsClientId=physics_client_id,
    )
    vertices = np.asarray(mesh_data[1], dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
        raise RuntimeError("PyBullet returned an invalid soft-body simulation mesh.")
    return vertices
