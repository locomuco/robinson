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
CUTOUT_SRC = os.path.join(HERE, "plate-160x120-trapezoid-cutout.stl")   # its front cutout is reused
OUT = os.path.join(HERE, "top-plate-eyes-pi5-pca9685.stl")

src = trimesh.load(SRC)
(X0, Y0, Z0), (X1, Y1, Z1) = src.bounds
W, H, T = X1 - X0, Z1 - Z0, Y1 - Y0          # 160 x 210 x 4
R = 7.0                                        # corner radius of the original
# M4 threaded rods, Ø 4.3; the inner front pair (25/135, 200) is dropped: the front cutout runs
# through it, as on the 160 x 120 plate the cutout comes from
RODS = [(10, 10), (25, 10), (135, 10), (150, 10), (10, 200), (150, 200), (10, 100), (150, 100)]


def circle(x, z, d):
    return Point(x, z).buffer(d / 2, quad_segs=24)


def stadium(x, z, w, h):                       # rounded rectangle cut, w along X, h along Z
    r = min(w, h) / 2
    return box(x - w / 2 + r, z - h / 2 + r, x + w / 2 - r, z + h / 2 - r).buffer(r, quad_segs=24)


def slot(x, z, d, length):                     # vertical stadium
    return LineString([(x, z - length / 2), (x, z + length / 2)]).buffer(d / 2, quad_segs=24)


def outline_of(mesh) -> Polygon:
    (x0, y0, z0), (_, y1, _) = mesh.bounds
    s = mesh.section(plane_origin=[0, (y0 + y1) / 2, 0], plane_normal=[0, 1, 0])
    return max((Polygon(np.c_[e[:, 0] - x0, e[:, 2] - z0]) for e in s.discrete), key=lambda p: p.area)


full = box(R, R, W - R, H - R).buffer(R, quad_segs=32)
holes = [circle(x, z, 4.3) for x, z in RODS]

# EyeMech ε3.2 base (145.4 x 95, eyes towards +Z), centred, front edge at Z = 190; 4 x M4
EYE_X0, EYE_FRONT = (W - 145.4) / 2, 190.0
eye_holes = [(EYE_X0 + bx, EYE_FRONT - by) for bx in (49.2, 96.2) for by in (12.04, 69.25)]
holes += [circle(x, z, 4.5) for x, z in eye_holes]

# Raspberry Pi 5 flat, back right: USB-A/Ethernet towards the right edge (+X), USB-C and
# micro-HDMI on the long edge towards the back (-Z), GPIO/HAT towards the eyes; M2.5 in 58 x 49
PI_X, PI_Z = 55.0, 32.0
holes += [circle(PI_X + hx, PI_Z + hz, 2.7) for hx in (3.5, 61.5) for hz in (3.5, 52.5)]
USBC_X = PI_X + 11.2                           # USB-C centre, 11.2 mm from the SD-card end
holes.append(stadium(USBC_X, 19.0, 18.0, 10.0))  # the USB-C plug passes through (cable from below)

# HW-170 / PCA9685 upright, back left; holes ~19 x 56 mm, slotted ±1.5 mm; the V+ screw
# terminal sits in the middle of a long side - its cable passes through next to it
PCA_X, PCA_Z, PCA_W, PCA_H = 20.0, 28.0, 25.4, 62.3
holes += [slot(PCA_X + PCA_W / 2 + dx, PCA_Z + PCA_H / 2 + dz, 3.2, 3.0)
          for dx in (-9.5, 9.5) for dz in (-28.0, 28.0)]
holes.append(stadium(12.0, PCA_Z + PCA_H / 2, 8.0, 14.0))   # power cable, outside the board edge

# front cutout: the trapezoid of the 160 x 120 plate (126 mm wide at the edge, 20 mm deep),
# turned from its bottom edge to our front edge
cut_src = trimesh.load(CUTOUT_SRC)
cw, ch = cut_src.extents[0], cut_src.extents[2]
cut_full = box(R, R, cw - R, ch - R).buffer(R, quad_segs=32)
notch = max(cut_full.difference(outline_of(cut_src)).geoms, key=lambda g: g.area)
nb = notch.bounds
notch = notch.union(box(nb[0], -10, nb[2], 0.5))                       # open past the edge
notch = translate(rotate(notch, 180, origin=(cw / 2, 0)), W / 2 - cw / 2, H)

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

parts = {"EyeMech": None, "Pi 5": box(PI_X, PI_Z, PI_X + 85, PI_Z + 56),
         "HW-170": box(PCA_X, PCA_Z, PCA_X + PCA_W, PCA_Z + PCA_H)}
print("Pi 5 X %.1f-%.1f, HW-170 X %.1f-%.1f, gap %.1f mm" % (PI_X, PI_X + 85, PCA_X, PCA_X + PCA_W, PI_X - PCA_X - PCA_W))
for name in ("Pi 5", "HW-170"):
    print("%s: nearest rod hole centre %.1f mm" % (name, min(parts[name].distance(Point(p)) for p in RODS)))
for name, h in (("USB-C hole", stadium(USBC_X, 19.0, 18.0, 10.0)), ("PCA power hole", stadium(12.0, PCA_Z + PCA_H / 2, 8.0, 14.0))):
    print("%s: bounds %s, to nearest rod hole centre %.1f mm, to plate edge %.1f mm"
          % (name, [round(v, 1) for v in h.bounds], min(h.distance(Point(p)) for p in RODS), h.distance(full.exterior)))

solid = trimesh.creation.extrude_polygon(plate, T)
solid.apply_transform(np.array([[1, 0, 0, X0], [0, 0, 1, Y0], [0, 1, 0, Z0], [0, 0, 0, 1]], float))
if solid.volume < 0:
    solid.invert()
assert solid.is_watertight
solid.export(OUT)
print("wrote %s: %.1f x %.1f x %.1f mm, %d holes/slots" % (os.path.basename(OUT), W, H, T, len(plate.interiors)))
