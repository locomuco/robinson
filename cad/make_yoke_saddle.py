"""Saddle for the CRT's deflection yoke: sits loose on the plate, no holes needed.

Run:  uv run --with trimesh --with manifold3d --with scipy python cad/make_yoke_saddle.py [gap_mm] [tilt_deg]

    gap_mm    height from the plate up to the underside of the yoke (default 19:
              glass neck top at 50 mm, 20 mm neck, 42 mm yoke -> 50 - 10 - 21)
    tilt_deg  how much the neck rises towards the back (default 0); the cradle is short
              along the neck, so a few degrees off do not bind

The yoke lies in a half shell (inner Ø 43, i.e. 0.5 mm clearance; add felt tape if
it rattles) up to its centre line, so the tube lifts straight out. A Velcro strap
(up to 20 mm) goes over the yoke and through an eyelet on each side wall, around its
bar - well above the foot, so the strap never gets between saddle and plate. Print in
PETG or ASA (the yoke gets warm), flat side down, no supports.
"""

import os
import sys

import numpy as np
import trimesh

GAP = float(sys.argv[1]) if len(sys.argv) > 1 else 19.0
TILT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
YOKE_D, CLEAR = 42.0, 0.5
R = (YOKE_D + 2 * CLEAR) / 2               # inner radius of the shell
WALL = 4.0
LEN = 30.0                                 # along the neck (room for a 20 mm strap)
BASE_W, BASE_D, BASE_T = 80.0, 40.0, 3.0   # foot: across x along x height
SLOT_W, SLOT_L = 4.0, 22.0                 # strap eyelets (Velcro up to 20 mm)
BAR = 4.0                                  # eyelet bar the strap wraps around
EYE_Z0 = 10.0                              # eyelet underside above the plate: strap room under the bar

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yoke-saddle.stl")

axis_z = GAP + R - CLEAR                   # yoke centre height (yoke rests on the shell bottom)
# x = across the neck, y = along the neck, z = up; plate surface at z = 0
body = trimesh.creation.box((2 * (R + WALL), LEN, axis_z), trimesh.transformations.translation_matrix((0, 0, axis_z / 2)))
foot = trimesh.creation.box((BASE_W, BASE_D, BASE_T), trimesh.transformations.translation_matrix((0, 0, BASE_T / 2)))
shape = body.union(foot, engine="manifold")

bore = trimesh.creation.cylinder(radius=R, height=LEN + 20, sections=160)
rot = trimesh.transformations.rotation_matrix(np.pi / 2, (1, 0, 0))            # cylinder along y
tilt = trimesh.transformations.rotation_matrix(np.radians(TILT), (1, 0, 0))     # neck rises to +y
bore.apply_transform(trimesh.transformations.translation_matrix((0, 0, axis_z)) @ tilt @ rot)
shape = shape.difference(bore, engine="manifold")

# eyelets on the outside of both side walls: a frame around a vertical slot, open below
eye_h = min(axis_z - 4, 14.0)
for sx in (-1, 1):
    x_in = R + WALL                                                  # outer face of the wall
    ring = trimesh.creation.box((SLOT_W + BAR, SLOT_L + 2 * BAR, eye_h))
    ring.apply_translation((sx * (x_in + (SLOT_W + BAR) / 2), 0, EYE_Z0 + eye_h / 2))
    hole = trimesh.creation.box((SLOT_W, SLOT_L, eye_h + 2))
    hole.apply_translation((sx * (x_in + SLOT_W / 2), 0, EYE_Z0 + eye_h / 2))
    shape = shape.union(ring, engine="manifold").difference(hole, engine="manifold")

assert shape.is_watertight
shape.export(OUT)
print(f"gap {GAP} mm, tilt {TILT}°: yoke centre at {axis_z:.1f} mm, saddle {BASE_W} x {BASE_D} x "
      f"{shape.bounds[1, 2]:.1f} mm, shell length {LEN} mm -> {os.path.basename(OUT)}")
