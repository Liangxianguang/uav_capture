from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet

from encirclement3d.environment import TETRAHEDRON_DIRECTIONS
from encirclement3d.pybullet_softbody_net import (
    PyBulletSoftBodyConfig,
    attach_tetrahedral_softbody_anchors,
    load_tetrahedral_softbody,
    softbody_vertices,
    write_tetrahedral_softbody_obj,
)


def test_pybullet_softbody_net_exports_mesh_and_keeps_four_anchor_vertices(tmp_path: Path) -> None:
    anchors = TETRAHEDRON_DIRECTIONS * 2.8 + np.array([0.0, 0.0, 5.0])
    mesh = write_tetrahedral_softbody_obj(tmp_path / "tetrahedral_net.obj", anchors, face_subdivisions=3)
    assert mesh.vertices == 20
    assert mesh.triangles == 36
    obj_lines = mesh.path.read_text(encoding="ascii").splitlines()
    assert sum(line.startswith("v ") for line in obj_lines) == 20
    assert sum(line.startswith("f ") for line in obj_lines) == 36

    client = pybullet.connect(pybullet.DIRECT)
    try:
        pybullet.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD, physicsClientId=client)
        pybullet.setGravity(0.0, 0.0, -9.81, physicsClientId=client)
        sphere = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=0.01, physicsClientId=client)
        anchor_ids = [
            pybullet.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=sphere,
                basePosition=position,
                physicsClientId=client,
            )
            for position in anchors
        ]
        softbody = load_tetrahedral_softbody(
            pybullet,
            mesh,
            PyBulletSoftBodyConfig(
                mass_kg=0.08,
                spring_elastic_stiffness=40.0,
                spring_damping_stiffness=0.40,
            ),
            physics_client_id=client,
        )
        constraints = attach_tetrahedral_softbody_anchors(
            pybullet, softbody, anchor_ids, physics_client_id=client
        )
        assert len(constraints) == 4
        for _ in range(60):
            pybullet.stepSimulation(physicsClientId=client)
        vertices = softbody_vertices(pybullet, softbody, physics_client_id=client)
        assert vertices.shape == (20, 3)
        np.testing.assert_allclose(vertices[:4], anchors, atol=1e-6)
    finally:
        pybullet.disconnect(client)


def test_pybullet_softbody_net_reports_rigid_sphere_contact(tmp_path: Path) -> None:
    anchors = TETRAHEDRON_DIRECTIONS * 2.8 + np.array([0.0, 0.0, 5.0])
    mesh = write_tetrahedral_softbody_obj(tmp_path / "contact_tetrahedral_net.obj", anchors, face_subdivisions=3)
    client = pybullet.connect(pybullet.DIRECT)
    try:
        pybullet.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD, physicsClientId=client)
        pybullet.setGravity(0.0, 0.0, 0.0, physicsClientId=client)
        anchor_shape = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=0.01, physicsClientId=client)
        anchor_ids = [
            pybullet.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=anchor_shape,
                basePosition=position,
                physicsClientId=client,
            )
            for position in anchors
        ]
        softbody = load_tetrahedral_softbody(
            pybullet,
            mesh,
            PyBulletSoftBodyConfig(
                mass_kg=0.08,
                spring_elastic_stiffness=80.0,
                spring_damping_stiffness=1.0,
            ),
            physics_client_id=client,
        )
        attach_tetrahedral_softbody_anchors(pybullet, softbody, anchor_ids, physics_client_id=client)
        target_shape = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=0.25, physicsClientId=client)
        target = pybullet.createMultiBody(
            baseMass=0.03,
            baseCollisionShapeIndex=target_shape,
            basePosition=[0.0, 0.0, 5.0],
            physicsClientId=client,
        )
        pybullet.resetBaseVelocity(target, linearVelocity=TETRAHEDRON_DIRECTIONS[0] * 3.0, physicsClientId=client)
        contacted = False
        for _ in range(360):
            pybullet.stepSimulation(physicsClientId=client)
            contacted = contacted or bool(
                pybullet.getContactPoints(target, softbody, physicsClientId=client)
            )
        assert contacted
        assert np.isfinite(softbody_vertices(pybullet, softbody, physics_client_id=client)).all()
    finally:
        pybullet.disconnect(client)
