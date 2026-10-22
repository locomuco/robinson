"""Build top-plate-eyes-pi5-pca9685.stl from top-plate-original-side-notches.stl.

Run:  uv run --with trimesh --with shapely --with mapbox-earcut --with scipy python cad/make_top_plate.py

Coordinates below are mm in the FreeCAD front view of the plate: X to the right (160),
Z up (210); the top edge (Z = 210) is Robinson's front, where the eyes look.
"""

import os

import numpy as np
import trimesh
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Point, Polygon, box

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "top-plate-original-side-notches.stl")
OUT = os.path.join(HERE, "top-plate-eyes-pi5-pca9685.stl")

src = trimesh.load(SRC)
(X0, Y0, Z0), (X1, Y1, Z1) = src.bounds
W, H, T = X1 - X0, Z1 - Z0, Y1 - Y0          # 160 x 210 x 4
R = 7.0                                        # corner radius of the original
RODS = [(10, 10), (25, 10), (135, 10), (150, 10), (10, 200), (25, 200), (135, 200), (150, 200),
        (10, 100), (150, 100)]                 # M4 threaded rods, Ø 4.3


def circle(x, z, d):
    return Point(x, z).buffer(d / 2, quad_segs=24)


def slot(x, z, d, length):                     # vertical stadium
    return LineString([(x, z - length / 2), (x, z + length / 2)]).buffer(d / 2, quad_segs=24)


def original_outline() -> Polygon:
    s = src.section(plane_origin=[0, (Y0 + Y1) / 2, 0], plane_normal=[0, 1, 0])
    return max((Polygon(np.c_[e[:, 0] - X0, e[:, 2] - Z0]) for e in s.discrete), key=lambda p: p.area)


full = box(R, R, W - R, H - R).buffer(R, quad_segs=32)
holes = [circle(x, z, 4.3) for x, z in RODS]

# EyeMech ε3.2 base (145.4 x 95, eyes towards +Z), centred, front edge at Z = 190; 4 x M4
EYE_X0, EYE_FRONT = (W - 145.4) / 2, 190.0
eye_holes = [(EYE_X0 + bx, EYE_FRONT - by) for bx in (49.2, 96.2) for by in (12.04, 69.25)]
holes += [circle(x, z, 4.5) for x, z in eye_holes]

# Raspberry Pi 5 flat, USB/Ethernet towards the left edge; M2.5 in 58 x 49
PI_X, PI_Z = 20.0, 32.0
holes += [circle(PI_X + hx, PI_Z + hz, 2.7) for hx in (85 - 3.5, 85 - 61.5) for hz in (3.5, 52.5)]

# HW-170 / PCA9685 upright, holes ~19 x 56 mm, slotted ±1.5 mm
PCA_X, PCA_Z, PCA_W, PCA_H = 112.0, 28.0, 25.4, 62.3
holes += [slot(PCA_X + PCA_W / 2 + dx, PCA_Z + PCA_H / 2 + dz, 3.2, 3.0)
          for dx in (-9.5, 9.5) for dz in (-28.0, 28.0)]

# the original 50 mm notch (left side of the original plate), turned to the front edge, centred
notch = max((g for g in full.difference(original_outline()).geoms if g.bounds[0] < 1),
            key=lambda g: g.area)
notch = notch.union(box(-10, notch.bounds[1], 0.5, notch.bounds[3]))   # open past the edge
notch = rotate(notch, -90, origin=(0, 0))                               # mouth faces +Z
nb = notch.bounds
notch = translate(notch, W / 2 - (nb[0] + nb[2]) / 2, H + 10 - nb[3])

plate = full
for h in holes:
    plate = plate.difference(h)
plate = plate.difference(notch)

b = notch.intersection(full).bounds
print("front cutout X %.1f-%.1f, Z %.1f-%.1f: %.1f wide at the edge, %.1f deep"
      % (b[0], b[2], b[1], b[3], b[2] - b[0], b[3] - b[1]))
print("cutout to nearest rod hole centre: %.1f mm" % min(notch.distance(Point(p)) for p in RODS))
for x, z in eye_holes:
    pt = Point(x, z)
    print("eye base hole (%.1f, %.1f): %s" % (x, z, "in the cutout" if notch.contains(pt)
                                              else "on the plate, %.1f mm from the cutout" % notch.distance(pt)))

solid = trimesh.creation.extrude_polygon(plate, T)
solid.apply_transform(np.array([[1, 0, 0, X0], [0, 0, 1, Y0], [0, 1, 0, Z0], [0, 0, 0, 1]], float))
if solid.volume < 0:
    solid.invert()
assert solid.is_watertight
solid.export(OUT)
print("wrote %s: %.1f x %.1f x %.1f mm, %d holes/slots" % (os.path.basename(OUT), W, H, T, len(plate.interiors)))
