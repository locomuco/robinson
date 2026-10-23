#!/usr/bin/env python3
"""Step-by-step servo check for a newly built eye mechanism (Robinson must run with eyes=sim).

    eye_check.py center                 all channels 0-5 to 90° (EyeMech assembly pose), one by one
    eye_check.py move <ch> <angle>      move one channel there slowly (1° steps) from its last angle
    eye_check.py release                stop the pulses: servos go limp

Same driver settings as robinson/eyes.py (ServoKit, 60 Hz). The last angles are kept in
/tmp/eye_check.json so `move` can ramp instead of jumping.
"""

import json
import os
import sys
import time

STATE = "/tmp/eye_check.json"


def load() -> dict:
    try:
        with open(STATE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save(state: dict) -> None:
    with open(STATE, "w") as f:
        json.dump(state, f)


def kit():
    from adafruit_servokit import ServoKit
    return ServoKit(channels=16, frequency=60)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    state = load()
    if cmd == "center":
        k = kit()
        for ch in range(6):
            print(f"channel {ch} -> 90°", flush=True)
            k.servo[ch].angle = 90
            state[str(ch)] = 90
            save(state)
            time.sleep(2.0)
    elif cmd == "move" and len(sys.argv) == 4:
        ch, target = int(sys.argv[2]), float(sys.argv[3])
        if not 0 <= target <= 180:
            sys.exit("angle must be 0-180")
        if str(ch) not in state:
            sys.exit(f"channel {ch} has no known position yet - run `center` first")
        k = kit()
        a = float(state[str(ch)])
        step = 1.0 if target > a else -1.0
        while abs(target - a) > 0.5:
            a += step
            k.servo[ch].angle = a
            time.sleep(0.03)
        k.servo[ch].angle = target
        state[str(ch)] = target
        save(state)
        print(f"channel {ch} at {target:.0f}°")
    elif cmd == "release":
        k = kit()
        for ch in range(6):
            k.servo[ch].angle = None
        if os.path.exists(STATE):
            os.remove(STATE)
        print("outputs off, servos limp")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
