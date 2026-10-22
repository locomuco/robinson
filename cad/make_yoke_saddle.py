"""Saddle for the CRT's deflection yoke: sits loose on the plate, no holes needed.

Run:  uv run --with trimesh --with manifold3d --with shapely --with mapbox-earcut --with scipy \
          python cad/make_yoke_saddle.py [gap_mm] [tilt_deg]

    gap_mm    height from the plate up to the underside of the yoke (default 19:
              glass neck top at 50 mm, 20 mm neck, 42 mm yoke -> 50 - 10 - 21)
    tilt_deg  how much the neck rises towards the back (default 0); the cradle is short
              along the neck, so a few degrees off do not bind

Trapezoid in cross-section (wide foot, sloped sides). The yoke lies in a half shell
(inner Ø 43, i.e. 0.5 mm clearance; add felt tape if it rattles) up to its centre line,
so the tube lifts straight out. A Velcro strap (up to 20 mm) goes over the yoke, into
the upper window in each sloped side, down the channel behind the bar, out of the lower
window and back up over the bar - it never gets between saddle and plate. Print in
PETG or ASA (the yoke gets warm), flat side down, no supports.
"""

import math
import os
import sys

import numpy as np
import trimesh
from shapely.geometry import Polygon

GAP = float(sys.argv[1]) if len(sys.argv) > 1 else 19.0
TILT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
YOKE_D, CLEAR = 42.0, 0.5
R = (YOKE_D + 2 * CLEAR) / 2               # inner radius of the shell
WALL = 4.0                                 # at the top of the sides
LEN = 34.0                                 # along the neck (room for a 20 mm strap)
FOOT_W = 80.0                              # width at the plate
STRAP_W, STRAP_T = 22.0, 4.0               # channel for a Velcro strap up to 20 mm wide
BAR = 4.0                                  # skin in front of the channel = bar the strap wraps
WIN = 4.5                                  # window height
CH_Z0, CH_Z1 = 6.0, 26.0                   # channel from/to height above the plate

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yoke-saddle.stl")

axis_z = GAP + R - CLEAR                   # yoke centre height (yoke rests on the shell bottom)
top_x, foot_x = R + WALL, FOOT_W / 2

# x = across the neck, y = along the neck, z = up; plate surface at z = 0
dx, dz = top_x - foot_x, axis_z            # direction of the right sloped face, bottom -> top
nlen = math.hypot(dx, dz)
nx, nz = dz / nlen, -dx / nlen             # its outward unit normal


def face_x(z: float) -> float:
    return foot_x + dx * z / axis_z


def inset(z: float, d: float) -> tuple[float, float]:
    """Point on the right sloped face at height z, moved d mm inwards (perpendicular)."""
    return face_x(z) - d * nx, z - d * nz


def prism(poly: Polygon, length: float) -> trimesh.Trimesh:
    """Extrude an x/z polygon symmetrically along y."""
    m = trimesh.creation.extrude_polygon(poly, length)
    m.apply_transform(np.array([[1, 0, 0, 0], [0, 0, 1, -length / 2], [0, 1, 0, 0], [0, 0, 0, 1]], float))
    if m.volume < 0:
        m.invert()
    return m


body = prism(Polygon([(-foot_x, 0), (foot_x, 0), (top_x, axis_z), (-top_x, axis_z)]), LEN)

bore = trimesh.creation.cylinder(radius=R, height=LEN + 20, sections=160)
rot = trimesh.transformations.rotation_matrix(np.pi / 2, (1, 0, 0))            # cylinder along y
tilt = trimesh.transformations.rotation_matrix(np.radians(TILT), (1, 0, 0))     # neck rises to +y
bore.apply_transform(trimesh.transformations.translation_matrix((0, 0, axis_z)) @ tilt @ rot)
shape = body.difference(bore, engine="manifold")

for sx in (-1, 1):
    def mirror(p, sx=sx):
        return Polygon([(sx * x, z) for x, z in p.exterior.coords])
    channel = Polygon([inset(CH_Z0, BAR), inset(CH_Z1, BAR), inset(CH_Z1, BAR + STRAP_T), inset(CH_Z0, BAR + STRAP_T)])
    win_lo = Polygon([inset(CH_Z0, -2), inset(CH_Z0 + WIN, -2), inset(CH_Z0 + WIN, BAR + 0.5), inset(CH_Z0, BAR + 0.5)])
    win_hi = Polygon([inset(CH_Z1 - WIN, -2), inset(CH_Z1, -2), inset(CH_Z1, BAR + 0.5), inset(CH_Z1 - WIN, BAR + 0.5)])
    for p in (channel, win_lo, win_hi):
        shape = shape.difference(prism(mirror(p), STRAP_W), engine="manifold")

# drop the sliver triangles the booleans leave, so the file reloads as one closed body
shape.merge_vertices(digits_vertex=4)
shape.update_faces(shape.nondegenerate_faces(height=1e-6))
shape.remove_unreferenced_vertices()
shape.export(OUT)
check = trimesh.load(OUT)
assert check.is_watertight and len(check.split()) == 1, "saddle mesh is not one closed body"
cx, cz = inset(CH_Z1, BAR + STRAP_T)
print(f"gap {GAP} mm, tilt {TILT}°: yoke centre at {axis_z:.1f} mm; {FOOT_W:.0f} x {LEN:.0f} mm foot, "
      f"{2 * top_x:.0f} mm wide at the top, {axis_z:.0f} mm high -> {os.path.basename(OUT)}")
print(f"thinnest wall between strap channel and shell: {math.hypot(cx, cz - axis_z) - R:.1f} mm")
