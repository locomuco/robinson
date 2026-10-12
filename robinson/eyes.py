"""Eye controller: six servos on a PCA9685 (I2C), driven by a background thread.

The voice loop never moves servos directly. It sets a *state*
(idle/listening/thinking/speaking/sleep), a *mood* and fires one-shot
*actions* (look left, wink, …); the controller turns that into gaze,
saccades, blinks and lid shapes at ~25 Hz.

Channel map and angle limits come from the original eyes.py:
  ch0 left/right  30 (right) … 140 (left), center 85
  ch1 up/down      0 (down)  …  90 (up),   center 45
  ch2/3 left lids  bottom open 90 / closed 20, top open 0 / closed 50
  ch4/5 right lids bottom open 0  / closed 90, top open 50 / closed 0
"""

import logging
import random
import threading
import time

log = logging.getLogger("robinson.eyes")

LR, UD, L_BOT, L_TOP, R_BOT, R_TOP = range(6)

LR_RIGHT, LR_LEFT, LR_CENTER = 30, 140, 85
UD_DOWN, UD_UP, UD_CENTER = 0, 90, 45

# (open angle, closed angle) per lid channel
LIDS = {L_BOT: (90, 20), L_TOP: (0, 50), R_BOT: (0, 90), R_TOP: (50, 0)}

# mood → (lid openness 0..1, gaze wander scale, blink rate per second)
MOODS = {
    "neutral":      (0.85, 1.0, 0.25),
    "fröhlich":     (0.80, 1.2, 0.30),
    "überrascht":   (1.00, 0.4, 0.10),
    "nachdenklich": (0.65, 0.6, 0.15),
    "traurig":      (0.50, 0.3, 0.15),
    "verärgert":    (0.45, 0.5, 0.10),
    "müde":         (0.35, 0.3, 0.35),
    "flüsternd":    (0.70, 0.5, 0.20),
}

ACTIONS = {"links", "rechts", "oben", "unten", "geradeaus", "zwinkern", "augen zu", "augen auf"}


class ServoBackend:
    def __init__(self):
        from adafruit_servokit import ServoKit  # needs I2C + Blinka
        self.kit = ServoKit(channels=16, frequency=60)

    def set(self, ch: int, angle: float) -> None:
        self.kit.servo[ch].angle = max(0.0, min(180.0, angle))


class SimBackend:
    """Logs servo moves instead of driving hardware (no PCA9685 attached)."""

    def __init__(self):
        self.angles: dict[int, float] = {}

    def set(self, ch: int, angle: float) -> None:
        if abs(self.angles.get(ch, -999) - angle) >= 1:
            self.angles[ch] = angle
            log.debug("servo %d → %.0f°", ch, angle)


def make_backend(kind: str):
    if kind == "off":
        return None
    if kind in ("servo", "auto"):
        try:
            return ServoBackend()
        except Exception as e:
            if kind == "servo":
                raise
            log.info("no servo hardware (%s) — simulating eyes", e)
    return SimBackend()


class Eyes(threading.Thread):
    def __init__(self, backend, sleep_after_s: float = 300.0):
        super().__init__(name="eyes", daemon=True)
        self.b = backend
        self.sleep_after_s = sleep_after_s
        self._lock = threading.Lock()
        self._state = "idle"
        self._mood = "neutral"
        self._actions: list[str] = []
        self._stop = threading.Event()
        self._last_activity = time.monotonic()
        self._hold_gaze_until = 0.0
        self._lids_closed_by_user = False

    # ── public API (thread-safe, non-blocking) ───────────────────────────────
    def set_state(self, state: str) -> None:
        with self._lock:
            if state != "idle":
                self._last_activity = time.monotonic()
            self._state = state

    def set_mood(self, mood: str) -> None:
        if mood in MOODS:
            with self._lock:
                self._mood = mood

    def action(self, name: str) -> None:
        if name in ACTIONS:
            with self._lock:
                self._actions.append(name)

    def stop(self) -> None:
        self._stop.set()

    # ── servo primitives ─────────────────────────────────────────────────────
    def _lids(self, openness: float, left: bool = True, right: bool = True) -> None:
        for ch, (o, c) in LIDS.items():
            if (ch in (L_BOT, L_TOP) and left) or (ch in (R_BOT, R_TOP) and right):
                self.b.set(ch, c + (o - c) * openness)

    def _gaze(self, lr: float, ud: float) -> None:
        self.b.set(LR, max(LR_RIGHT, min(LR_LEFT, lr)))
        self.b.set(UD, max(UD_DOWN, min(UD_UP, ud)))

    def _blink(self, openness: float, dur: float | None = None) -> None:
        self._lids(0.0)
        time.sleep(dur or random.uniform(0.06, 0.1))
        self._lids(openness)

    # ── behaviour loop ───────────────────────────────────────────────────────
    def run(self) -> None:
        if self.b is None:
            return
        self._gaze(LR_CENTER, UD_CENTER)
        self._lids(MOODS["neutral"][0])
        next_saccade = next_blink = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            with self._lock:
                state, mood = self._state, self._mood
                actions, self._actions = self._actions, []
                idle_for = now - self._last_activity
            openness, wander, blink_rate = MOODS.get(mood, MOODS["neutral"])

            if state == "idle" and idle_for > self.sleep_after_s:
                state = "sleep"

            for a in actions:
                openness = self._do_action(a, openness)
            if self._lids_closed_by_user:
                openness = 0.0

            if state == "sleep":
                self._gaze(LR_CENTER, UD_DOWN + 10)
                self._lids(0.0)
                self._stop.wait(0.2)
                continue
            if state == "listening":
                self._lids_closed_by_user = False  # being talked to wakes it up
                openness = max(openness, 0.95)     # attentive: wide open
                wander *= 0.25
            elif state == "thinking":
                wander *= 0.5

            if now >= next_saccade and now >= self._hold_gaze_until:
                if state == "thinking":
                    # classic "searching memory" glance: up and to one side
                    lr = random.choice((LR_CENTER - 35, LR_CENTER + 35))
                    ud = UD_CENTER + random.uniform(20, 35)
                    next_saccade = now + random.uniform(0.8, 1.6)
                else:
                    lr = LR_CENTER + random.uniform(-30, 30) * wander
                    ud = UD_CENTER + random.uniform(-15, 15) * wander
                    hold = (0.3, 1.2) if state == "speaking" else (0.6, 3.0)
                    next_saccade = now + random.uniform(*hold)
                self._gaze(lr, ud)
            self._lids(openness)

            if now >= next_blink and openness > 0:
                self._blink(openness)
                next_blink = now + random.expovariate(blink_rate) + 0.5

            self._stop.wait(0.04)

    def _do_action(self, a: str, openness: float) -> float:
        hold = time.monotonic() + 2.5
        if a == "links":
            self._gaze(LR_LEFT - 10, UD_CENTER)
        elif a == "rechts":
            self._gaze(LR_RIGHT + 10, UD_CENTER)
        elif a == "oben":
            self._gaze(LR_CENTER, UD_UP - 5)
        elif a == "unten":
            self._gaze(LR_CENTER, UD_DOWN + 5)
        elif a == "geradeaus":
            self._gaze(LR_CENTER, UD_CENTER)
        elif a == "zwinkern":
            self._lids(0.0, left=False)
            time.sleep(0.25)
            self._lids(openness, left=False)
            hold = 0.0
        elif a == "augen zu":
            self._lids_closed_by_user = True
            hold = 0.0
        elif a == "augen auf":
            self._lids_closed_by_user = False
            hold = 0.0
        self._hold_gaze_until = max(self._hold_gaze_until, hold)
        return openness
