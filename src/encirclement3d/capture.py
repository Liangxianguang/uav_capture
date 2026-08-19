"""Geometry primitives for the analytical tetrahedral capture-cage proxy.

The proxy represents a deployed four-face net whose corners are carried by the
four defender UAVs.  It deliberately does not claim to model flexible net,
tether, contact, or aerodynamic dynamics.  Its purpose is to make the
intermediate capture condition explicit and auditable before a higher-fidelity
mechanism is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class TetrahedralCageMetrics:
    """Signed target-to-face clearances and edge geometry for four cage corners."""

    face_clearances: np.ndarray
    edge_lengths: np.ndarray

    @property
    def min_face_clearance(self) -> float:
        """Distance to the closest face; non-negative means inside the cage."""
        return float(np.min(self.face_clearances))

    @property
    def min_edge_length(self) -> float:
        return float(np.min(self.edge_lengths))

    @property
    def max_edge_length(self) -> float:
        return float(np.max(self.edge_lengths))

    @property
    def target_inside(self) -> bool:
        return bool(np.all(self.face_clearances >= 0.0))


@dataclass(frozen=True)
class TetrahedralContactResult:
    """Result of resolving one sphere-versus-tetrahedron motion interval.

    This is a rigid, zero-thickness planar-net proxy.  It is deliberately
    separate from a cable or soft-body model: a successful resolution proves
    only that the target sphere did not cross a closed planar face during the
    control interval.
    """

    position: np.ndarray
    contact: bool
    contact_face_index: int | None
    contact_fraction: float | None
    maximum_unresolved_penetration: float
    contained: bool


def tetrahedral_face_planes(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return one point and an inward unit normal for every tetrahedral face."""
    vertices = np.asarray(corners, dtype=np.float64)
    if vertices.shape != (4, 3):
        raise ValueError(f"Expected tetrahedral corners with shape (4, 3), got {vertices.shape}.")

    origins: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for opposite_index in range(4):
        face_indices = [index for index in range(4) if index != opposite_index]
        face = vertices[face_indices]
        face_origin = face[0]
        face_normal = np.cross(face[1] - face_origin, face[2] - face_origin)
        normal_length = float(np.linalg.norm(face_normal))
        if normal_length < 1e-9:
            raise ValueError("Degenerate tetrahedral cage: a face has zero area.")
        face_normal /= normal_length
        if float(np.dot(vertices[opposite_index] - face_origin, face_normal)) < 0.0:
            face_normal *= -1.0
        origins.append(face_origin)
        normals.append(face_normal)
    return np.asarray(origins, dtype=np.float64), np.asarray(normals, dtype=np.float64)


def tetrahedral_cage_metrics(corners: np.ndarray, target: np.ndarray) -> TetrahedralCageMetrics:
    """Measure a target with respect to the closed tetrahedron formed by corners.

    For face ``i`` (the face opposite corner ``i``), the inward-pointing normal
    is selected using corner ``i``.  Consequently every signed face clearance
    is positive for a target inside the cage and negative once it has escaped
    through that face.  The result is invariant to global translation and to
    the ordering of the three vertices within a face.
    """

    vertices = np.asarray(corners, dtype=np.float64)
    point = np.asarray(target, dtype=np.float64)
    if vertices.shape != (4, 3):
        raise ValueError(f"Expected tetrahedral corners with shape (4, 3), got {vertices.shape}.")
    if point.shape != (3,):
        raise ValueError(f"Expected target position with shape (3,), got {point.shape}.")

    face_origins, face_normals = tetrahedral_face_planes(vertices)
    face_clearances = np.sum((point[None, :] - face_origins) * face_normals, axis=1)

    edges = np.asarray(
        [np.linalg.norm(vertices[first] - vertices[second]) for first, second in combinations(range(4), 2)],
        dtype=np.float64,
    )
    return TetrahedralCageMetrics(
        face_clearances=np.asarray(face_clearances, dtype=np.float64),
        edge_lengths=edges,
    )


def resolve_sphere_tetrahedral_contact(
    previous_corners: np.ndarray,
    current_corners: np.ndarray,
    previous_center: np.ndarray,
    proposed_center: np.ndarray,
    radius: float,
    *,
    tolerance: float = 1e-6,
    max_projection_iterations: int = 96,
    bisection_steps: int = 28,
) -> TetrahedralContactResult:
    """Resolve a swept spherical target against a moving closed tetrahedron.

    Corners and target center are linearly interpolated over the control
    interval.  The first outward crossing is found by bisection, then the
    remaining target displacement is projected into the contact tangent plane
    and into the final tetrahedron's inward half spaces.  The function is a
    deterministic contact proxy, not a substitute for compliant net dynamics.
    """
    prior_vertices = np.asarray(previous_corners, dtype=np.float64)
    next_vertices = np.asarray(current_corners, dtype=np.float64)
    start = np.asarray(previous_center, dtype=np.float64)
    proposed = np.asarray(proposed_center, dtype=np.float64)
    if prior_vertices.shape != (4, 3) or next_vertices.shape != (4, 3):
        raise ValueError("previous_corners and current_corners must both have shape (4, 3).")
    if start.shape != (3,) or proposed.shape != (3,):
        raise ValueError("previous_center and proposed_center must both have shape (3,).")
    if radius < 0.0:
        raise ValueError("radius must be non-negative.")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")
    if max_projection_iterations <= 0 or bisection_steps <= 0:
        raise ValueError("Contact iteration counts must be positive.")

    def margin(fraction: float) -> tuple[float, TetrahedralCageMetrics]:
        corners = prior_vertices + fraction * (next_vertices - prior_vertices)
        center = start + fraction * (proposed - start)
        metrics = tetrahedral_cage_metrics(corners, center)
        return metrics.min_face_clearance - radius, metrics

    start_margin, _ = margin(0.0)
    end_margin, end_metrics = margin(1.0)
    contact_fraction: float | None = None
    contact_face_index: int | None = None
    resolved = proposed.copy()
    contact = False

    if end_margin < -tolerance:
        contact = True
        if start_margin < -tolerance:
            contact_fraction = 0.0
        else:
            lower, upper = 0.0, 1.0
            for _ in range(bisection_steps):
                middle = 0.5 * (lower + upper)
                middle_margin, _ = margin(middle)
                if middle_margin >= 0.0:
                    lower = middle
                else:
                    upper = middle
            contact_fraction = lower

        impact_corners = prior_vertices + contact_fraction * (next_vertices - prior_vertices)
        impact_center = start + contact_fraction * (proposed - start)
        impact_metrics = tetrahedral_cage_metrics(impact_corners, impact_center)
        contact_face_index = int(np.argmin(impact_metrics.face_clearances))
        _origins, normals = tetrahedral_face_planes(impact_corners)
        inward_normal = normals[contact_face_index]
        remaining = (1.0 - contact_fraction) * (proposed - start)
        outward_component = min(float(np.dot(remaining, inward_normal)), 0.0)
        resolved = impact_center + remaining - outward_component * inward_normal

    # Project against all final inward half spaces. This also handles a cage
    # that moved into an otherwise stationary target during the interval.
    final_origins, final_normals = tetrahedral_face_planes(next_vertices)
    for _ in range(max_projection_iterations):
        clearances = np.sum((resolved[None, :] - final_origins) * final_normals, axis=1)
        deficits = radius - clearances
        violating_faces = np.flatnonzero(deficits > tolerance)
        if len(violating_faces) == 0:
            break
        contact = True
        face_index = int(violating_faces[np.argmax(deficits[violating_faces])])
        if contact_face_index is None:
            contact_face_index = face_index
        resolved += (float(deficits[face_index]) + tolerance) * final_normals[face_index]

    resolved_metrics = tetrahedral_cage_metrics(next_vertices, resolved)
    unresolved_penetration = max(0.0, radius - resolved_metrics.min_face_clearance)
    contained = bool(unresolved_penetration <= tolerance)
    if not contact and not contained:
        contact = True
        contact_face_index = int(np.argmin(resolved_metrics.face_clearances))
    return TetrahedralContactResult(
        position=resolved,
        contact=contact,
        contact_face_index=contact_face_index,
        contact_fraction=contact_fraction,
        maximum_unresolved_penetration=float(unresolved_penetration),
        contained=contained,
    )
