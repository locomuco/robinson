"""Saddle for the CRT's deflection yoke: sits loose on the plate, no holes needed.

Run:  uv run --with trimesh --with manifold3d --with shapely --with mapbox-earcut --with scipy \
          python cad/make_yoke_saddle.py [gap_mm] [tilt_deg] [length_mm]

    gap_mm    height from the plate up to the underside of the yoke (default 19:
              glass neck top at 50 mm, 20 mm neck, 42 mm yoke -> 50 - 10 - 21)
    tilt_deg  how much the neck rises towards the back (default 0); the cradle is short
              along the neck, so a few degrees off do not bind; turn the saddle around
              to tilt the other way
    length_mm along the neck (default 12); from 30 mm on there is room for a Velcro strap

Trapezoid in cross-section (wide foot, sloped sides). The yoke lies in a half shell
(inner Ø 43, i.e. 0.5 mm clearance; add felt tape if it rattles) up to its centre line,
so the tube lifts straight out. A Velcro strap (up to 20 mm) goes over the yoke, into
the upper window in each sloped side, down the channel behind the bar, out of the lower
window and back up over the bar - it never gets between saddle and plate. Or a rubber
band / O-ring over the yoke between the hex-head pegs on both sides.

Steampunk dress: rivet rows along the sloped sides, a gear on both end faces, hex bolt
heads as band hooks. Print in PETG or ASA (the yoke gets warm; copper or brass coloured
matches the yoke), flat side down; the peg heads overhang a little, supports optional.
"""

import math
import os
import sys

import numpy as np
import trimesh
from shapely.geometry import Polygon

GAP = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
TILT = float(sys.argv[2]) if len(sys.argv) > 2 else 7.0
YOKE_D, CLEAR = 42.0, 0.5
R = (YOKE_D + 2 * CLEAR) / 2               # inner radius of the shell
WALL = 4.0                                 # at the top of the sides
LEN = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0   # along the neck
VELCRO = LEN >= 30                         # a 20 mm strap channel needs the length
FOOT_W = 80.0                              # width at the plate
STRAP_W, STRAP_T = 22.0, 4.0               # channel for a Velcro strap up to 20 mm wide
BAR = 4.0                                  # skin in front of the channel = bar the strap wraps
WIN = 4.5                                  # window height
CH_Z0, CH_Z1 = 6.0, 26.0                   # channel from/to height above the plate
PEG_YS = (-14.5, 14.5) if VELCRO else (0.0,)   # rubber band pegs: beside the strap, or centred
PEG_R, PEG_L, HEAD_R, HEAD_T = 2.3, 4.5, 4.3, 2.4
RIVET_R, RIVET_Y = 1.4, LEN / 2 - 2.6      # rivet rows along both edges of each slope
GEAR_TEETH, GEAR_H = 10, 0.8

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yoke-saddle.stl")

axis_z = GAP + R - CLEAR                   # yoke centre height (yoke rests on the shell bottom)
top_x, foot_x = R + WALL, FOOT_W / 2
PEG_Z = axis_z - 9.0                       # band pegs on the upper part of the slope
GEAR_RO = min(7.5, (GAP - 3.0) / 2)        # gear below the shell (also where the tilt lowers it)
GEAR_RI, GEAR_Z = GEAR_RO * 0.78, 1.0 + GEAR_RO

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

for sx in ((-1, 1) if VELCRO else ()):
    def mirror(p, sx=sx):
        return Polygon([(sx * x, z) for x, z in p.exterior.coords])
    channel = Polygon([inset(CH_Z0, BAR), inset(CH_Z1, BAR), inset(CH_Z1, BAR + STRAP_T), inset(CH_Z0, BAR + STRAP_T)])
    win_lo = Polygon([inset(CH_Z0, -2), inset(CH_Z0 + WIN, -2), inset(CH_Z0 + WIN, BAR + 0.5), inset(CH_Z0, BAR + 0.5)])
    win_hi = Polygon([inset(CH_Z1 - WIN, -2), inset(CH_Z1, -2), inset(CH_Z1, BAR + 0.5), inset(CH_Z1 - WIN, BAR + 0.5)])
    for p in (channel, win_lo, win_hi):
        shape = shape.difference(prism(mirror(p), STRAP_W), engine="manifold")



def along(p, v, length):
    """Transform placing a z-axis primitive from point p along unit vector v (x/z plane)."""
    ang = math.atan2(v[0], v[2])                                   # rotate +z towards v
    return (trimesh.transformations.translation_matrix(p) @
            trimesh.transformations.rotation_matrix(ang, (0, 1, 0)) @
            trimesh.transformations.translation_matrix((0, 0, length / 2)))


adds = []
for sx in (-1, 1):
    n = (sx * nx, 0.0, nz)
    fx = sx * face_x(PEG_Z)
    for py in PEG_YS:                                              # hex bolt band hooks
        shaft = trimesh.creation.cylinder(radius=PEG_R, height=PEG_L + 1, sections=32)
        shaft.apply_transform(along((fx - n[0], py, PEG_Z - n[2]), n, PEG_L + 1))
        head = trimesh.creation.cylinder(radius=HEAD_R, height=HEAD_T, sections=6)
        head.apply_transform(along((fx + n[0] * PEG_L, py, PEG_Z + n[2] * PEG_L), n, HEAD_T))
        adds += [shaft, head]
    for ry in (-RIVET_Y, RIVET_Y):                                 # rivets, half sunk into the slope
        for z in np.arange(4.0, axis_z - 2.5, 5.0):
            if abs(z - PEG_Z) < HEAD_R + RIVET_R + 1 and any(abs(ry - py) < HEAD_R + RIVET_R for py in PEG_YS):
                continue
            rivet = trimesh.creation.icosphere(subdivisions=2, radius=RIVET_R)
            rivet.apply_translation((sx * face_x(z), ry, z))
            adds.append(rivet)

# gear relief on both end faces, below the shell
t = np.linspace(0, 2 * np.pi, GEAR_TEETH * 8, endpoint=False)
rr = np.where((t * GEAR_TEETH / (2 * np.pi)) % 1 < 0.5, GEAR_RO, GEAR_RI)
gear2d = Polygon(np.c_[rr * np.sin(t), GEAR_Z + rr * np.cos(t)]).difference(
    Polygon([(1.8 * math.sin(a), GEAR_Z + 1.8 * math.cos(a)) for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]))
for sy in (-1, 1):
    g = trimesh.creation.extrude_polygon(gear2d, GEAR_H + 0.4)
    g.apply_transform(np.array([[1, 0, 0, 0], [0, 0, sy, sy * (LEN / 2 - 0.4)], [0, 1, 0, 0], [0, 0, 0, 1]], float))
    if g.volume < 0:
        g.invert()
    adds.append(g)

shape = trimesh.boolean.union([shape] + adds, engine="manifold")

# drop the sliver triangles the booleans leave, so the file reloads as one closed body
shape.merge_vertices(digits_vertex=4)
shape.update_faces(shape.nondegenerate_faces(height=1e-6))
shape.remove_unreferenced_vertices()
shape.export(OUT)
check = trimesh.load(OUT)
assert check.is_watertight and len(check.split()) == 1, "saddle mesh is not one closed body"
cx, cz = inset(CH_Z1, BAR + STRAP_T)
print("hold-down:", "Velcro strap or rubber band" if VELCRO else "rubber band (too short for a Velcro strap)")
print(f"gap {GAP} mm, tilt {TILT}°: yoke centre at {axis_z:.1f} mm; {FOOT_W:.0f} x {LEN:.0f} mm foot, "
      f"{2 * top_x:.0f} mm wide at the top, {axis_z:.0f} mm high -> {os.path.basename(OUT)}")
if VELCRO:
    print(f"thinnest wall between strap channel and shell: {math.hypot(cx, cz - axis_z) - R:.1f} mm")
