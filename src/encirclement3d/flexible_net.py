"""Deterministic mass-spring tetrahedral capture-net models.

``face_subdivisions=1`` retains the original one-center-per-face proxy.  A
value of two or more builds a shared-node triangular surface mesh: edge nodes
belong to both adjacent tetrahedral faces and therefore transmit cable load
across the surface.  The refined mode is still an explicit, tension-only
mass-spring approximation.  It does not claim cloth self-contact, knotting,
or aerodynamic wake resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_FACE_ANCHORS: tuple[tuple[int, int, int], ...] = (
    (1, 2, 3),
    (0, 2, 3),
    (0, 1, 3),
    (0, 1, 2),
)


@dataclass(frozen=True)
class FlexibleNetSnapshot:
    """Net geometry at one control boundary.

    ``vertices`` and its connectivity are present for the refined shared-node
    mesh.  Their absence denotes the legacy four-center model.
    """

    anchors: np.ndarray
    face_centers: np.ndarray
    vertices: np.ndarray | None = None
    triangle_nodes: np.ndarray | None = None
    triangle_panels: np.ndarray | None = None


@dataclass(frozen=True)
class FlexibleNetMetrics:
    """Containment and structural measurements for the triangulated net."""

    face_clearances: np.ndarray
    max_tension: float
    max_strain: float
    peak_contact_impulse: float

    @property
    def min_face_clearance(self) -> float:
        return float(np.min(self.face_clearances))


@dataclass(frozen=True)
class FlexibleNetContactResult:
    """Resolved target center and the contacted mesh triangle, if any."""

    position: np.ndarray
    contact: bool
    triangle_index: int | None
    panel_index: int | None
    contact_fraction: float | None
    maximum_unresolved_penetration: float
    contained: bool
    inward_normal: np.ndarray | None


def _validate_anchors(anchors: np.ndarray) -> np.ndarray:
    value = np.asarray(anchors, dtype=np.float64)
    if value.shape != (4, 3):
        raise ValueError(f"Expected four anchors with shape (4, 3), got {value.shape}.")
    return value


def _triangles(snapshot: FlexibleNetSnapshot) -> tuple[np.ndarray, np.ndarray]:
    """Return all mesh triangles and their parent tetrahedral-face indices."""
    if snapshot.vertices is not None:
        if snapshot.triangle_nodes is None or snapshot.triangle_panels is None:
            raise ValueError("Refined flexible-net snapshot is missing mesh connectivity.")
        vertices = np.asarray(snapshot.vertices, dtype=np.float64)
        triangle_nodes = np.asarray(snapshot.triangle_nodes, dtype=np.int64)
        panels = np.asarray(snapshot.triangle_panels, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("Flexible-net mesh vertices must have shape (N, 3).")
        if triangle_nodes.ndim != 2 or triangle_nodes.shape[1] != 3:
            raise ValueError("Flexible-net mesh triangles must have shape (M, 3).")
        if panels.shape != (len(triangle_nodes),):
            raise ValueError("Flexible-net triangle panels must have shape (M,).")
        if np.any(triangle_nodes < 0) or np.any(triangle_nodes >= len(vertices)):
            raise ValueError("Flexible-net triangle connectivity is out of range.")
        return vertices[triangle_nodes], panels

    anchors = _validate_anchors(snapshot.anchors)
    centers = np.asarray(snapshot.face_centers, dtype=np.float64)
    if centers.shape != (4, 3):
        raise ValueError(f"Expected four face centers with shape (4, 3), got {centers.shape}.")
    triangles: list[np.ndarray] = []
    panels: list[int] = []
    for panel, indices in enumerate(_FACE_ANCHORS):
        first, second, third = (anchors[index] for index in indices)
        center = centers[panel]
        triangles.extend(
            [
                np.asarray((first, second, center), dtype=np.float64),
                np.asarray((second, third, center), dtype=np.float64),
                np.asarray((third, first, center), dtype=np.float64),
            ]
        )
        panels.extend((panel, panel, panel))
    return np.asarray(triangles, dtype=np.float64), np.asarray(panels, dtype=np.int64)


def flexible_mesh_planes(snapshot: FlexibleNetSnapshot) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an inward half-space for every triangular micro-panel.

    Testing all local planes is intentionally conservative for a warped mesh:
    a folded or ambiguous surface shrinks the accepted interior rather than
    silently declaring a traversal safe.
    """
    triangles, panels = _triangles(snapshot)
    interior = np.mean(snapshot.anchors, axis=0)
    origins = triangles[:, 0, :]
    normals = np.cross(triangles[:, 1, :] - origins, triangles[:, 2, :] - origins)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-9):
        raise ValueError("Degenerate flexible-net triangle.")
    normals = normals / lengths[:, None]
    toward_interior = np.sum((interior[None, :] - origins) * normals, axis=1)
    normals[toward_interior < 0.0] *= -1.0
    return origins, normals, panels


def _refined_mesh_topology(face_subdivisions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build shared barycentric mesh nodes for all four tetrahedral faces."""
    if face_subdivisions < 2:
        raise ValueError("Refined mesh requires at least two face subdivisions.")
    node_lookup: dict[tuple[int, int, int, int], int] = {}
    weight_counts: list[tuple[int, int, int, int]] = []

    # The first four indices always follow the defender-anchor order.
    for anchor in range(4):
        counts = [0, 0, 0, 0]
        counts[anchor] = face_subdivisions
        key = tuple(counts)
        node_lookup[key] = anchor
        weight_counts.append(key)

    def node(face: tuple[int, int, int], u: int, v: int) -> int:
        first, second, third = face
        counts = [0, 0, 0, 0]
        counts[first] = face_subdivisions - u - v
        counts[second] = u
        counts[third] = v
        key = tuple(counts)
        index = node_lookup.get(key)
        if index is None:
            index = len(weight_counts)
            node_lookup[key] = index
            weight_counts.append(key)
        return index

    triangles: list[tuple[int, int, int]] = []
    panels: list[int] = []
    for panel, face in enumerate(_FACE_ANCHORS):
        for u in range(face_subdivisions):
            for v in range(face_subdivisions - u):
                triangles.append((node(face, u, v), node(face, u + 1, v), node(face, u, v + 1)))
                panels.append(panel)
                if u + v <= face_subdivisions - 2:
                    triangles.append(
                        (node(face, u + 1, v), node(face, u + 1, v + 1), node(face, u, v + 1))
                    )
                    panels.append(panel)
    weights = np.asarray(weight_counts, dtype=np.float64) / float(face_subdivisions)
    return weights, np.asarray(triangles, dtype=np.int64), np.asarray(panels, dtype=np.int64)


def tetrahedral_surface_mesh_topology(face_subdivisions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return shared-node tetrahedral surface weights and triangle topology.

    The first four vertices always correspond to the four defender anchors.
    This public helper is shared by the explicit mass-spring and PyBullet
    soft-body backends so their mesh discretizations are auditable.
    """
    return _refined_mesh_topology(face_subdivisions)


def _spring_edges(triangles: np.ndarray) -> np.ndarray:
    """Return the unique tension-only spring edges of a triangular mesh."""
    edges: set[tuple[int, int]] = set()
    for first, second, third in np.asarray(triangles, dtype=np.int64):
        edges.add(tuple(sorted((int(first), int(second)))))
        edges.add(tuple(sorted((int(second), int(third)))))
        edges.add(tuple(sorted((int(third), int(first)))))
    return np.asarray(sorted(edges), dtype=np.int64)


def _interpolate_snapshots(
    previous: FlexibleNetSnapshot,
    current: FlexibleNetSnapshot,
    fraction: float,
) -> FlexibleNetSnapshot:
    """Interpolate geometry while retaining immutable refined connectivity."""
    previous_refined = previous.vertices is not None
    current_refined = current.vertices is not None
    if previous_refined != current_refined:
        raise ValueError("Cannot interpolate different flexible-net topologies.")
    if not previous_refined:
        return FlexibleNetSnapshot(
            anchors=previous.anchors + fraction * (current.anchors - previous.anchors),
            face_centers=previous.face_centers + fraction * (current.face_centers - previous.face_centers),
        )
    assert previous.vertices is not None and current.vertices is not None
    assert previous.triangle_nodes is not None and current.triangle_nodes is not None
    assert previous.triangle_panels is not None and current.triangle_panels is not None
    if not np.array_equal(previous.triangle_nodes, current.triangle_nodes) or not np.array_equal(
        previous.triangle_panels, current.triangle_panels
    ):
        raise ValueError("Cannot interpolate refined flexible nets with different connectivity.")
    return FlexibleNetSnapshot(
        anchors=previous.anchors + fraction * (current.anchors - previous.anchors),
        face_centers=previous.face_centers + fraction * (current.face_centers - previous.face_centers),
        vertices=previous.vertices + fraction * (current.vertices - previous.vertices),
        triangle_nodes=previous.triangle_nodes.copy(),
        triangle_panels=previous.triangle_panels.copy(),
    )


class FlexibleTetrahedralNet:
    """Tension-only capture net attached to four moving tetrahedral anchors."""

    def __init__(
        self,
        anchors: np.ndarray,
        *,
        node_mass: float,
        spring_stiffness: float,
        spring_damping: float,
        drag_coefficient: float,
        gravity: float,
        substeps: int,
        face_subdivisions: int = 1,
        spring_pretension: float = 0.0,
    ) -> None:
        anchors = _validate_anchors(anchors)
        if node_mass <= 0.0 or spring_stiffness <= 0.0:
            raise ValueError("Flexible-net mass and spring stiffness must be positive.")
        if spring_damping < 0.0 or drag_coefficient < 0.0 or gravity < 0.0 or spring_pretension < 0.0:
            raise ValueError("Flexible-net damping, drag, gravity, and pretension must be non-negative.")
        if substeps <= 0:
            raise ValueError("Flexible-net substeps must be positive.")
        if int(face_subdivisions) != face_subdivisions or face_subdivisions <= 0:
            raise ValueError("Flexible-net face_subdivisions must be a positive integer.")
        self.node_mass = float(node_mass)
        self.spring_stiffness = float(spring_stiffness)
        self.spring_damping = float(spring_damping)
        self.drag_coefficient = float(drag_coefficient)
        self.gravity = float(gravity)
        self.substeps = int(substeps)
        self.face_subdivisions = int(face_subdivisions)
        self.spring_pretension = float(spring_pretension)
        self.anchors = anchors.copy()
        self.peak_contact_impulse = 0.0
        self._refined = self.face_subdivisions >= 2

        if self._refined:
            self.mesh_weights, self.mesh_triangles, self.mesh_panels = _refined_mesh_topology(self.face_subdivisions)
            # The mesh is small; elementwise contraction avoids dispatching
            # to a second BLAS/OpenMP runtime after Torch is imported.
            self.mesh_vertices = np.sum(self.mesh_weights[:, :, None] * anchors[None, :, :], axis=1)
            self.mesh_velocities = np.zeros_like(self.mesh_vertices)
            self.spring_edges = _spring_edges(self.mesh_triangles)
            self.spring_rest_lengths = np.linalg.norm(
                self.mesh_vertices[self.spring_edges[:, 1]] - self.mesh_vertices[self.spring_edges[:, 0]], axis=1
            )
            if np.any(self.spring_rest_lengths < 1e-9):
                raise ValueError("Flexible-net mesh contains a zero-length spring.")
            self.dynamic_node_indices = np.arange(4, len(self.mesh_vertices), dtype=np.int64)
            # Preserve the total moving mass of the legacy four-center model
            # while changing spatial resolution.
            self.dynamic_node_mass = 4.0 * self.node_mass / float(len(self.dynamic_node_indices))
            self.last_tensions = np.full(len(self.spring_edges), self.spring_pretension, dtype=np.float64)
            self.last_strains = np.zeros(len(self.spring_edges), dtype=np.float64)
            self.face_centers = np.asarray(
                [np.mean(anchors[list(indices)], axis=0) for indices in _FACE_ANCHORS], dtype=np.float64
            )
        else:
            self.face_centers = np.asarray(
                [np.mean(anchors[list(indices)], axis=0) for indices in _FACE_ANCHORS], dtype=np.float64
            )
            self.face_velocities = np.zeros((4, 3), dtype=np.float64)
            self.rest_lengths = np.asarray(
                [
                    [np.linalg.norm(anchors[index] - self.face_centers[panel]) for index in indices]
                    for panel, indices in enumerate(_FACE_ANCHORS)
                ],
                dtype=np.float64,
            )
            self.last_tensions = np.full((4, 3), self.spring_pretension, dtype=np.float64)
            self.last_strains = np.zeros((4, 3), dtype=np.float64)

    def snapshot(self) -> FlexibleNetSnapshot:
        if self._refined:
            return FlexibleNetSnapshot(
                anchors=self.anchors.copy(),
                face_centers=self.face_centers.copy(),
                vertices=self.mesh_vertices.copy(),
                triangle_nodes=self.mesh_triangles.copy(),
                triangle_panels=self.mesh_panels.copy(),
            )
        return FlexibleNetSnapshot(anchors=self.anchors.copy(), face_centers=self.face_centers.copy())

    def advance(self, next_anchors: np.ndarray, dt: float) -> FlexibleNetSnapshot:
        """Advance the net under moving attachments and tension-only springs."""
        target_anchors = _validate_anchors(next_anchors)
        if dt <= 0.0:
            raise ValueError("Flexible-net time step must be positive.")
        if self._refined:
            return self._advance_refined(target_anchors, dt)
        return self._advance_legacy(target_anchors, dt)

    def _advance_legacy(self, target_anchors: np.ndarray, dt: float) -> FlexibleNetSnapshot:
        previous = self.snapshot()
        initial_anchors = self.anchors.copy()
        anchor_velocity = (target_anchors - initial_anchors) / dt
        sub_dt = dt / self.substeps
        for substep in range(self.substeps):
            fraction = (substep + 1) / self.substeps
            anchors = initial_anchors + fraction * (target_anchors - initial_anchors)
            forces = np.zeros_like(self.face_centers)
            forces[:, 2] -= self.node_mass * self.gravity
            forces -= self.drag_coefficient * self.face_velocities
            for panel, indices in enumerate(_FACE_ANCHORS):
                center = self.face_centers[panel]
                velocity = self.face_velocities[panel]
                for local_index, anchor_index in enumerate(indices):
                    offset = anchors[anchor_index] - center
                    length = float(np.linalg.norm(offset))
                    if length < 1e-9:
                        raise ValueError("Flexible-net cable collapsed to zero length.")
                    direction = offset / length
                    extension = length - self.rest_lengths[panel, local_index]
                    relative_speed = float(np.sum((anchor_velocity[anchor_index] - velocity) * direction))
                    tension = max(
                        0.0,
                        self.spring_pretension + self.spring_stiffness * extension + self.spring_damping * relative_speed,
                    )
                    forces[panel] += tension * direction
                    self.last_tensions[panel, local_index] = tension
                    self.last_strains[panel, local_index] = extension / self.rest_lengths[panel, local_index]
            self.face_velocities += (forces / self.node_mass) * sub_dt
            self.face_centers += self.face_velocities * sub_dt
        self.anchors = target_anchors.copy()
        return previous

    def _advance_refined(self, target_anchors: np.ndarray, dt: float) -> FlexibleNetSnapshot:
        previous = self.snapshot()
        initial_anchors = self.anchors.copy()
        anchor_velocity = (target_anchors - initial_anchors) / dt
        sub_dt = dt / self.substeps
        dynamic = self.dynamic_node_indices
        for substep in range(self.substeps):
            fraction = (substep + 1) / self.substeps
            anchors = initial_anchors + fraction * (target_anchors - initial_anchors)
            self.mesh_vertices[:4] = anchors
            self.mesh_velocities[:4] = anchor_velocity
            forces = np.zeros_like(self.mesh_vertices)
            forces[dynamic, 2] -= self.dynamic_node_mass * self.gravity
            forces[dynamic] -= self.drag_coefficient * self.mesh_velocities[dynamic]
            for edge_index, (first, second) in enumerate(self.spring_edges):
                offset = self.mesh_vertices[second] - self.mesh_vertices[first]
                length = float(np.linalg.norm(offset))
                if length < 1e-9:
                    raise ValueError("Flexible-net mesh spring collapsed to zero length.")
                direction = offset / length
                extension = length - self.spring_rest_lengths[edge_index]
                relative_speed = float(
                    np.sum((self.mesh_velocities[second] - self.mesh_velocities[first]) * direction)
                )
                tension = max(
                    0.0,
                    self.spring_pretension + self.spring_stiffness * extension + self.spring_damping * relative_speed,
                )
                self.last_tensions[edge_index] = tension
                self.last_strains[edge_index] = extension / self.spring_rest_lengths[edge_index]
                if first >= 4:
                    forces[first] += tension * direction
                if second >= 4:
                    forces[second] -= tension * direction
            self.mesh_velocities[dynamic] += (forces[dynamic] / self.dynamic_node_mass) * sub_dt
            self.mesh_vertices[dynamic] += self.mesh_velocities[dynamic] * sub_dt
        self.mesh_vertices[:4] = target_anchors
        self.mesh_velocities[:4] = anchor_velocity
        self.anchors = target_anchors.copy()
        self.face_centers = np.asarray(
            [np.mean(target_anchors[list(indices)], axis=0) for indices in _FACE_ANCHORS], dtype=np.float64
        )
        return previous

    def apply_contact_impulse(
        self,
        panel_index: int,
        inward_normal: np.ndarray,
        impulse: float,
        *,
        triangle_index: int | None = None,
    ) -> None:
        """Apply a target's equal-and-opposite normal impulse to the mesh."""
        if not 0 <= panel_index < 4:
            raise ValueError("Flexible-net panel index must be in [0, 3].")
        if impulse < 0.0:
            raise ValueError("Flexible-net contact impulse must be non-negative.")
        normal = np.asarray(inward_normal, dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        if normal.shape != (3,) or norm < 1e-9:
            raise ValueError("Flexible-net contact normal must be a nonzero 3-vector.")
        direction = normal / norm
        if self._refined:
            if triangle_index is None:
                candidates = np.flatnonzero(self.mesh_panels == panel_index)
                if len(candidates) == 0:
                    raise ValueError("Flexible-net panel has no mesh triangles.")
                triangle_index = int(candidates[0])
            if not 0 <= triangle_index < len(self.mesh_triangles):
                raise ValueError("Flexible-net triangle index is out of range.")
            if int(self.mesh_panels[triangle_index]) != panel_index:
                raise ValueError("Flexible-net triangle does not belong to the reported panel.")
            dynamic_nodes = self.mesh_triangles[triangle_index][self.mesh_triangles[triangle_index] >= 4]
            if len(dynamic_nodes):
                node_impulse = impulse / float(len(dynamic_nodes))
                self.mesh_velocities[dynamic_nodes] -= (node_impulse / self.dynamic_node_mass) * direction
        else:
            self.face_velocities[panel_index] -= (impulse / self.node_mass) * direction
        self.peak_contact_impulse = max(self.peak_contact_impulse, float(impulse))

    def metrics(self, target: np.ndarray) -> FlexibleNetMetrics:
        point = np.asarray(target, dtype=np.float64)
        if point.shape != (3,):
            raise ValueError("Flexible-net target must have shape (3,).")
        origins, normals, _panels = flexible_mesh_planes(self.snapshot())
        clearances = np.sum((point[None, :] - origins) * normals, axis=1)
        return FlexibleNetMetrics(
            face_clearances=clearances,
            max_tension=float(np.max(self.last_tensions)),
            # A slack cable may be shorter than its rest length; compression
            # is not a failure mode for a tension-only cable.
            max_strain=max(0.0, float(np.max(self.last_strains))),
            peak_contact_impulse=float(self.peak_contact_impulse),
        )


def resolve_sphere_flexible_net_contact(
    previous: FlexibleNetSnapshot,
    current: FlexibleNetSnapshot,
    previous_center: np.ndarray,
    proposed_center: np.ndarray,
    radius: float,
    *,
    tolerance: float = 1e-6,
    max_projection_iterations: int = 128,
    bisection_steps: int = 28,
) -> FlexibleNetContactResult:
    """Conservatively resolve a swept sphere against a moving triangulated net."""
    start = np.asarray(previous_center, dtype=np.float64)
    proposed = np.asarray(proposed_center, dtype=np.float64)
    if start.shape != (3,) or proposed.shape != (3,):
        raise ValueError("Flexible-net target centers must have shape (3,).")
    if radius < 0.0 or tolerance < 0.0:
        raise ValueError("Flexible-net radius and tolerance must be non-negative.")
    if max_projection_iterations <= 0 or bisection_steps <= 0:
        raise ValueError("Flexible-net contact iteration counts must be positive.")

    def margin(fraction: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        snapshot = _interpolate_snapshots(previous, current, fraction)
        origins, normals, panels = flexible_mesh_planes(snapshot)
        point = start + fraction * (proposed - start)
        clearances = np.sum((point[None, :] - origins) * normals, axis=1)
        return float(np.min(clearances) - radius), origins, normals, panels

    start_margin, _start_origins, _start_normals, _start_panels = margin(0.0)
    end_margin, end_origins, end_normals, end_panels = margin(1.0)
    resolved = proposed.copy()
    contact = False
    triangle_index: int | None = None
    panel_index: int | None = None
    contact_fraction: float | None = None
    inward_normal: np.ndarray | None = None

    if end_margin < -tolerance:
        contact = True
        if start_margin < -tolerance:
            contact_fraction = 0.0
        else:
            lower, upper = 0.0, 1.0
            for _ in range(bisection_steps):
                middle = 0.5 * (lower + upper)
                middle_margin, _origins, _normals, _panels = margin(middle)
                if middle_margin >= 0.0:
                    lower = middle
                else:
                    upper = middle
            contact_fraction = lower
        _impact_margin, impact_origins, impact_normals, impact_panels = margin(contact_fraction)
        impact_center = start + contact_fraction * (proposed - start)
        impact_clearances = np.sum((impact_center[None, :] - impact_origins) * impact_normals, axis=1)
        triangle_index = int(np.argmin(impact_clearances))
        panel_index = int(impact_panels[triangle_index])
        inward_normal = impact_normals[triangle_index]
        remaining = (1.0 - contact_fraction) * (proposed - start)
        outward_component = min(float(np.sum(remaining * inward_normal)), 0.0)
        resolved = impact_center + remaining - outward_component * inward_normal

    for _ in range(max_projection_iterations):
        clearances = np.sum((resolved[None, :] - end_origins) * end_normals, axis=1)
        deficits = radius - clearances
        violating = np.flatnonzero(deficits > tolerance)
        if len(violating) == 0:
            break
        contact = True
        face = int(violating[np.argmax(deficits[violating])])
        if triangle_index is None:
            triangle_index = face
            panel_index = int(end_panels[face])
            inward_normal = end_normals[face]
        resolved += (float(deficits[face]) + tolerance) * end_normals[face]

    final_clearances = np.sum((resolved[None, :] - end_origins) * end_normals, axis=1)
    unresolved_penetration = max(0.0, radius - float(np.min(final_clearances)))
    contained = bool(unresolved_penetration <= tolerance)
    if not contact and not contained:
        triangle_index = int(np.argmin(final_clearances))
        panel_index = int(end_panels[triangle_index])
        inward_normal = end_normals[triangle_index]
        contact = True
    return FlexibleNetContactResult(
        position=resolved,
        contact=contact,
        triangle_index=triangle_index,
        panel_index=panel_index,
        contact_fraction=contact_fraction,
        maximum_unresolved_penetration=float(unresolved_penetration),
        contained=contained,
        inward_normal=None if inward_normal is None else inward_normal.copy(),
    )
