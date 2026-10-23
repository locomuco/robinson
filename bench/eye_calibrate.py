#!/usr/bin/env python3
"""Interactive end-position calibration of the eyes (Robinson must run with eyes=sim).

    ssh -t robinson ~/robinson/.venv/bin/python ~/robinson/bench/eye_calibrate.py

Walks through every servo position Robinson needs, starting from the 90° assembly pose:
    ← / →  (or a / d)   1° steps        A / D   5° steps
    Enter              save this position and go on
    s                  skip (keep the current value)
    q                  quit; positions saved so far are written

Rule of thumb: move until the part *just* reaches its end, then 3-5° back - a servo
pressed against a stop gets hot and strips its gears.

Writes ~/robinson-data/eyes.json (old file kept as eyes.json.bak-<time>), using the
PCA9685 channels from settings.json (eye_channels).
"""

import json
import os
import sys
import termios
import time
import tty

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from robinson.eyes import SERVO_PULSE_US  # noqa: E402

DATA = os.path.expanduser("~/robinson-data")
CAL = os.path.join(DATA, "eyes.json")
SETTINGS = os.path.join(DATA, "settings.json")

# logical servo numbers as in robinson/eyecal.py
LR, UD, L_BOT, L_TOP, R_BOT, R_TOP = range(6)

STEPS = [  # (logical servo, key in eyes.json, what to do)
    (LR, ("lr", "mitte"), "eyes look STRAIGHT AHEAD (left/right)"),
    (LR, ("lr", "links"), "eyes look as far as they go to Robinson's LEFT"),
    (LR, ("lr", "rechts"), "eyes look as far as they go to Robinson's RIGHT"),
    (UD, ("ud", "mitte"), "eyes look STRAIGHT AHEAD (up/down)"),
    (UD, ("ud", "oben"), "eyes look as far UP as they go"),
    (UD, ("ud", "unten"), "eyes look as far DOWN as they go"),
    (L_TOP, ("lids", L_TOP, 1), "LEFT TOP lid CLOSED (just touching the bottom lid)"),
    (L_TOP, ("lids", L_TOP, 0), "LEFT TOP lid fully OPEN"),
    (L_BOT, ("lids", L_BOT, 1), "LEFT BOTTOM lid CLOSED (just touching the top lid)"),
    (L_BOT, ("lids", L_BOT, 0), "LEFT BOTTOM lid fully OPEN"),
    (R_TOP, ("lids", R_TOP, 1), "RIGHT TOP lid CLOSED"),
    (R_TOP, ("lids", R_TOP, 0), "RIGHT TOP lid fully OPEN"),
    (R_BOT, ("lids", R_BOT, 1), "RIGHT BOTTOM lid CLOSED"),
    (R_BOT, ("lids", R_BOT, 0), "RIGHT BOTTOM lid fully OPEN"),
]
NAMES = {LR: "left/right", UD: "up/down", L_BOT: "left bottom lid", L_TOP: "left top lid",
         R_BOT: "right bottom lid", R_TOP: "right top lid"}


def read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        c = sys.stdin.read(1)
        if c == "\x1b":                      # arrow keys: ESC [ C / D (shift: ESC [ 1 ; 2 C)
            seq = sys.stdin.read(2)
            if seq == "[1":
                seq = "[" + sys.stdin.read(3)[-1]
                return {"[C": "D", "[D": "A"}.get(seq, "")
            return {"[C": "d", "[D": "a"}.get(seq, "")
        return c
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main() -> None:
    from adafruit_servokit import ServoKit
    with open(SETTINGS) as f:
        settings = json.load(f)
    if settings.get("eyes") != "sim":
        sys.exit("Robinson drives the eyes right now - set eyes to sim first.")
    chans = settings.get("eye_channels", list(range(6)))
    kit = ServoKit(channels=16, frequency=60)
    for ch in chans:
        kit.servo[ch].set_pulse_width_range(*SERVO_PULSE_US)
    with open(CAL) as f:
        cal = json.load(f)
    pos = {s: 90.0 for s in range(6)}       # assembly pose: everything at 90°

    def go(servo: int, target: float) -> None:
        target = max(0.0, min(180.0, target))
        a = pos[servo]
        while abs(target - a) > 0.5:
            a += 1.0 if target > a else -1.0
            kit.servo[chans[servo]].angle = a
            time.sleep(0.02)
        kit.servo[chans[servo]].angle = target
        pos[servo] = target

    for s in range(6):
        go(s, 90)
    print("All servos at 90° (assembly pose).\n")
    done = 0
    for servo, key, what in STEPS:
        if key[0] == "lids":                # lids start from the closed value once known
            pair = cal["lids"].setdefault(str(key[1]), [90, 90])
            go(servo, pair[1] if key[2] == 0 else 90)
        print(f"[{NAMES[servo]}, PCA channel {chans[servo]}]  {what}")
        while True:
            sys.stdout.write(f"\r  angle {pos[servo]:5.1f}°   (←/→ 1°, A/D 5°, Enter save, s skip, q quit) ")
            sys.stdout.flush()
            k = read_key()
            step = {"a": -1, "d": 1, "A": -5, "D": 5}.get(k)
            if step:
                go(servo, pos[servo] + step)
            elif k in ("\r", "\n"):
                if key[0] == "lids":
                    cal["lids"][str(key[1])][key[2]] = round(pos[servo])
                else:
                    cal[key[0]][key[1]] = round(pos[servo])
                print(f"\n  saved {round(pos[servo])}°\n")
                done += 1
                break
            elif k == "s":
                print("\n  skipped\n")
                break
            elif k in ("q", "\x03"):
                print("\n")
                break
        else:
            continue
        if k in ("q", "\x03"):
            break
    if done:
        backup = f"{CAL}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        os.replace(CAL, backup)
        with open(CAL, "w") as f:
            json.dump(cal, f, indent=1)
        print(f"{done} positions written to {CAL} (old file: {os.path.basename(backup)})")
        print(json.dumps({k: cal[k] for k in ("lr", "ud", "lids")}))
    for s in range(6):                      # back to the assembly pose
        go(s, 90)


if __name__ == "__main__":
    main()
