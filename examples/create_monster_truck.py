"""Create a stylized, textured-free monster truck matching the reference image.

The output is a portable GLB assembled from simple watertight primitives.  It
is intended as a deterministic fallback when running the diffusion pipeline is
not practical (for example, on a CPU-only machine).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from trimesh.creation import box, cylinder


GREEN = (0.65, 0.95, 0.02, 1.0)
GREEN_DARK = (0.25, 0.55, 0.01, 1.0)
BLACK = (0.015, 0.02, 0.035, 1.0)
TIRE = (0.035, 0.045, 0.065, 1.0)
RUBBER_HIGHLIGHT = (0.12, 0.15, 0.18, 1.0)
WINDOW = (0.08, 0.22, 0.23, 1.0)
METAL = (0.48, 0.52, 0.55, 1.0)
RED = (0.9, 0.03, 0.02, 1.0)


def material(color):
    return trimesh.visual.material.PBRMaterial(
        baseColorFactor=np.asarray(color, dtype=np.float32),
        metallicFactor=0.15,
        roughnessFactor=0.55,
    )


def part(mesh, color, name):
    mesh.visual.material = material(color)
    mesh.metadata["name"] = name
    return mesh


def add_box(scene, size, location, color, name, rotation=None):
    mesh = box(extents=size)
    mesh.apply_translation(location)
    if rotation is not None:
        mesh.apply_transform(rotation)
    scene.add_geometry(part(mesh, color, name))


def add_cylinder(scene, radius, height, location, color, name, axis="z", sections=32):
    mesh = cylinder(radius=radius, height=height, sections=sections)
    if axis == "y":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    elif axis == "x":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    mesh.apply_translation(location)
    scene.add_geometry(part(mesh, color, name))


def add_flame(scene, x, y, z, side):
    """Extrude a flame silhouette and place it on a door side."""
    points = [
        (0.0, 0.0), (0.34, 0.08), (0.18, 0.22), (0.68, 0.18),
        (0.42, 0.42), (0.98, 0.39), (0.53, 0.70), (0.70, 1.02),
        (0.18, 0.78), (-0.12, 1.15), (-0.05, 0.61), (-0.38, 0.72),
        (-0.20, 0.33), (-0.52, 0.22), (-0.10, 0.12),
    ]
    vertices = np.asarray([(px, py, pz) for pz in (0.0, 0.035) for px, py in points])
    count = len(points)
    faces = [[0, index, index + 1] for index in range(1, count - 1)]
    faces += [[count, count + index + 1, count + index] for index in range(1, count - 1)]
    faces.extend(
        [face
         for index in range(count)
         for face in (
             [index, (index + 1) % count, count + index],
             [(index + 1) % count, count + (index + 1) % count, count + index],
         )]
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    mesh.apply_scale([0.62, 0.62, side])
    mesh.apply_translation([x, y, z])
    scene.add_geometry(part(mesh, GREEN_DARK, "door flame graphic"))


def build_truck():
    scene = trimesh.Scene()

    # Raised chassis, pickup bed, hood, and cab.
    add_box(scene, (3.45, 1.52, 0.28), (0.0, 0.0, 1.55), BLACK, "ladder chassis")
    add_box(scene, (1.45, 1.45, 0.68), (-0.88, 0.0, 2.02), GREEN, "hood")
    add_box(scene, (1.28, 1.42, 1.12), (0.35, 0.0, 2.23), GREEN, "cab")
    add_box(scene, (1.48, 1.40, 0.45), (1.55, 0.0, 2.02), GREEN, "pickup bed")
    add_box(scene, (1.55, 1.46, 0.12), (1.55, 0.0, 2.30), GREEN_DARK, "bed rail")
    add_box(scene, (0.42, 1.58, 0.16), (-1.72, 0.0, 1.68), METAL, "front bumper")
    add_box(scene, (0.12, 1.48, 0.16), (2.30, 0.0, 1.78), METAL, "rear bumper")

    # Slanted windscreen and side windows.
    windshield = box(extents=(0.06, 1.12, 0.72))
    windshield.apply_transform(trimesh.transformations.rotation_matrix(-0.18, [0, 1, 0]))
    windshield.apply_translation((-0.06, 0.0, 2.38))
    scene.add_geometry(part(windshield, WINDOW, "front windshield"))
    for y in (-0.72, 0.72):
        add_box(scene, (0.58, 0.045, 0.62), (0.46, y, 2.40), WINDOW, "side window")
        add_box(scene, (0.07, 0.05, 0.92), (0.10, y, 2.30), BLACK, "window pillar")

    # Grille, headlights, mirrors, and tail lamps.
    add_box(scene, (0.04, 1.05, 0.65), (-1.64, 0.0, 1.98), BLACK, "front grille")
    for y in (-0.43, 0.43):
        add_cylinder(scene, 0.23, 0.07, (-1.68, y, 2.08), (0.82, 0.92, 0.88, 1), "round headlight", axis="x")
        add_cylinder(scene, 0.13, 0.08, (-1.70, y, 1.78), RED, "front indicator", axis="x")
    for y in (-0.82, 0.82):
        add_box(scene, (0.18, 0.12, 0.25), (0.22, y, 2.82), BLACK, "side mirror")
    add_box(scene, (0.06, 0.12, 0.13), (2.28, -0.68, 2.04), RED, "tail lamp")

    # Flames on both door sides.
    add_flame(scene, 0.25, -0.76, 2.02, 1)
    add_flame(scene, 0.25, 0.76, 2.02, -1)

    # Four giant wheels, rims, and simple angular tread lugs.
    for x in (-1.18, 1.30):
        for y in (-1.02, 1.02):
            add_cylinder(scene, 0.78, 0.42, (x, y, 0.82), TIRE, "oversized tire", axis="y", sections=40)
            add_cylinder(scene, 0.42, 0.45, (x, y, 0.82), GREEN_DARK, "lime wheel rim", axis="y", sections=32)
            add_cylinder(scene, 0.22, 0.47, (x, y, 0.82), BLACK, "wheel hub", axis="y", sections=24)
            for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                lug = box(extents=(0.22, 0.48, 0.13))
                lug.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 1, 0]))
                lug.apply_translation((x + 0.70 * np.cos(angle), y, 0.82 + 0.70 * np.sin(angle)))
                scene.add_geometry(part(lug, RUBBER_HIGHLIGHT, "tire tread lug"))

    # Visible lifted suspension: axles, links, and coil springs.
    for x in (-1.18, 1.30):
        add_cylinder(scene, 0.12, 2.0, (x, 0.0, 1.08), METAL, "solid axle", axis="y")
        for y in (-0.72, 0.72):
            add_cylinder(scene, 0.14, 0.62, (x, y, 1.30), BLACK, "shock absorber", axis="z")
            add_cylinder(scene, 0.22, 0.58, (x, y, 1.38), METAL, "coil spring", axis="z", sections=20)
            link = box(extents=(1.35, 0.08, 0.10))
            link.apply_transform(trimesh.transformations.rotation_matrix(0.28, [0, 1, 0]))
            link.apply_translation((0.0, y, 1.22))
            scene.add_geometry(part(link, BLACK, "suspension link"))

    # Ground shadow-like platform keeps the model pleasant in GLB viewers.
    add_box(scene, (5.0, 3.0, 0.04), (0.0, 0.0, 0.02), (0.18, 0.12, 0.08, 1), "ground")
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="assets/monster_truck.glb")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_truck().export(output)
    print(f"Exported {output}")


if __name__ == "__main__":
    main()
