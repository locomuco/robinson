"""Eye controller: six servos on a PCA9685 (I2C), driven by a background thread.

The voice loop never moves servos directly. It sets a *state*
(idle/listening/thinking/speaking/sleep), a *mood* and fires one-shot
*actions* (look left, wink, …); the controller turns that into gaze,
saccades, blinks and lid shapes at ~25 Hz.

Channels: 0 left/right, 1 up/down, 2/3 left lid bottom/top, 4/5 right lid
bottom/top. Limits and behaviour factors come from EyeCalibration
(eyes.json, adjustable by voice); after a change the eyes show the result.
"""

import logging
import random
import threading
import time

from .eyecal import L_BOT, L_TOP, LR, R_BOT, R_TOP, UD, EyeCalibration

log = logging.getLogger("robinson.eyes")

LID_CHANNELS = (L_BOT, L_TOP, R_BOT, R_TOP)

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

# one-shot movements (no calibration change); see normalize_action for wording
ACTIONS = {"links", "rechts", "oben", "unten", "geradeaus", "zwinkern", "zwinkern links",
           "zwinkern rechts", "blinzeln", "augen rollen", "augen zu", "augen auf", "schlafen",
           "aufwachen", "große augen", "skeptisch"}

_ACTION_SYNONYMS = {
    "zwinker": "zwinkern", "blinzel": "blinzeln", "roll": "augen rollen", "verdreh": "augen rollen",
    "schlaf": "schlafen", "einschlaf": "schlafen", "wach": "aufwachen", "aufwach": "aufwachen",
    "gross": "große augen", "groß": "große augen", "staun": "große augen", "weit auf": "große augen",
    "skept": "skeptisch", "misstrau": "skeptisch", "zu": "augen zu", "schließ": "augen zu",
    "auf": "augen auf", "öffne": "augen auf", "gerade": "geradeaus", "mitte": "geradeaus",
    "hoch": "oben", "runter": "unten", "links": "links", "rechts": "rechts",
    "oben": "oben", "unten": "unten",
}


def normalize_action(aktion: str, auge: str = "") -> str | None:
    """Model/user wording → an ACTIONS name ("mit dem linken Auge zwinkern" → "zwinkern links")."""
    a, e = aktion.lower().strip(), auge.lower().strip()
    if a in ACTIONS:
        name = a
    else:
        name = next((v for k, v in _ACTION_SYNONYMS.items() if k in a), None)
    if name is None:
        return None
    side = "links" if "link" in (e or a) else "rechts" if "recht" in (e or a) else ""
    if name in ("zwinkern", "blinzeln") and side and (e or name == "zwinkern"):
        return f"zwinkern {side}"     # blinking with one eye *is* winking
    return name


class ServoBackend:
    def __init__(self, channels=range(6)):
        from adafruit_servokit import ServoKit  # needs I2C + Blinka
        self.kit = ServoKit(channels=16, frequency=60)
        self.map = list(channels)      # logical servo (LR, UD, L_BOT, L_TOP, R_BOT, R_TOP) -> PCA channel

    def set(self, ch: int, angle: float) -> None:
        self.kit.servo[self.map[ch]].angle = max(0.0, min(180.0, angle))


class SimBackend:
    """Logs servo moves instead of driving hardware (no PCA9685 attached)."""

    def __init__(self):
        self.angles: dict[int, float] = {}

    def set(self, ch: int, angle: float) -> None:
        if abs(self.angles.get(ch, -999) - angle) >= 1:
            self.angles[ch] = angle
            log.debug("servo %d → %.0f°", ch, angle)


def make_backend(kind: str, channels=range(6)):
    if kind == "off":
        return None
    if kind in ("servo", "auto"):
        try:
            backend = ServoBackend(channels)
            log.info("eyes: PCA9685 servo driver found, channels %s", list(channels))
            return backend
        except Exception as e:
            if kind == "servo":
                raise
            log.info("no servo hardware (%s) — simulating eyes", e)
    return SimBackend()


class Eyes(threading.Thread):
    def __init__(self, backend, cal: EyeCalibration):
        super().__init__(name="eyes", daemon=True)
        self.b = backend
        self.cal = cal
        self._pos: dict[int, float] = {}
        self._lock = threading.Lock()
        self._state = "idle"
        self._mood = "neutral"
        self._actions: list[str] = []
        self._stop = threading.Event()
        self._last_activity = time.monotonic()
        self._hold_gaze_until = 0.0
        self._lids_closed_by_user = False
        self._sleep_now = False            # "schlaf ein": sleep until spoken to
        self._expr_until = 0.0             # temporary lid expression (große Augen …)
        self._expr_openness = 1.0
        self._look: float | None = None    # room azimuth of a voice/face (+ = left), see look_at
        self._look_el: float | None = None # elevation of a face (+ = up), if known
        self._look_t = 0.0

    # ── public API (thread-safe, non-blocking) ───────────────────────────────
    def set_state(self, state: str) -> None:
        with self._lock:
            if state != "idle":
                self._last_activity = time.monotonic()
            self._state = state

    @property
    def state(self) -> str:
        return self._state

    def look_at(self, azimuth: float, elevation: float | None = None) -> None:
        """Turn the eyes towards a voice or face: degrees, 0 = ahead, + = left / up."""
        with self._lock:
            self._look, self._look_el, self._look_t = azimuth, elevation, time.monotonic()

    def attention(self) -> None:
        """Someone is there (e.g. a face): don't fall asleep, wake up."""
        with self._lock:
            self._last_activity = time.monotonic()

    def set_mood(self, mood: str) -> None:
        if mood in MOODS:
            with self._lock:
                self._mood = mood

    def action(self, name: str) -> None:
        if name in ACTIONS:
            with self._lock:
                if name in ("aufwachen", "augen auf"):
                    self._last_activity = time.monotonic()
                self._actions.append(name)

    def stop(self) -> None:
        self._stop.set()

    def calibration_changed(self) -> None:
        """eyes.json was modified (possibly by another process): reload it."""
        self.cal.reload()

    # ── servo primitives ─────────────────────────────────────────────────────
    def _set(self, ch: int, angle: float) -> None:
        self._pos[ch] = angle
        self.b.set(ch, angle)

    def _lids(self, openness: float, left: bool = True, right: bool = True) -> None:
        for ch in LID_CHANNELS:
            if (ch in (L_BOT, L_TOP) and left) or (ch in (R_BOT, R_TOP) and right):
                o, c = self.cal.lid(ch)
                self._set(ch, c + (o - c) * openness)

    def _gaze(self, lr: float, ud: float) -> None:
        g_lr, g_ud = self.cal.gaze("lr"), self.cal.gaze("ud")
        lr = max(min(g_lr["links"], g_lr["rechts"]), min(max(g_lr["links"], g_lr["rechts"]), lr))
        ud = max(min(g_ud["oben"], g_ud["unten"]), min(max(g_ud["oben"], g_ud["unten"]), ud))
        tempo = self.cal.b("tempo")
        glide = 0.0 if tempo >= 1.0 else (1.0 / tempo - 1.0) * 0.25   # slower = visible glide
        start = (self._pos.get(LR, lr), self._pos.get(UD, ud))
        steps = max(1, int(glide / 0.02))
        for i in range(1, steps + 1):
            f = i / steps
            self._set(LR, start[0] + (lr - start[0]) * f)
            self._set(UD, start[1] + (ud - start[1]) * f)
            if steps > 1:
                time.sleep(0.02)

    def _centre(self) -> tuple[float, float]:
        return self.cal.gaze("lr")["mitte"], self.cal.gaze("ud")["mitte"]

    def _show_preview(self, p: dict, openness: float) -> None:
        """Hold the adjusted part in the adjusted position so it can be judged."""
        kind = p.get("kind")
        if kind == "lid":
            for ch in p["channels"]:
                self._set(ch, self.cal.lid(ch)[p["index"]])
            time.sleep(2.0)
            self._lids(openness)
        elif kind == "gaze":
            g = self.cal.gaze(p["axis"])
            lr, ud = self._centre()
            if p["axis"] == "lr":
                lr = g[p["key"]]
            else:
                ud = g[p["key"]]
            self._gaze(lr, ud)
            time.sleep(2.0)
            self._gaze(*self._centre())
        else:
            self._gaze(*self._centre())
            self._blink(openness)
        self._hold_gaze_until = time.monotonic() + 0.5

    def _blink(self, openness: float, dur: float | None = None) -> None:
        self._lids(0.0)
        time.sleep(dur or random.uniform(0.06, 0.1))
        self._lids(openness)

    # ── behaviour loop ───────────────────────────────────────────────────────
    def run(self) -> None:
        if self.b is None:
            return
        self._gaze(*self._centre())
        self._lids(MOODS["neutral"][0])
        next_saccade = next_blink = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            with self._lock:
                state, mood = self._state, self._mood
                actions, self._actions = self._actions, []
                idle_for = now - self._last_activity
            openness, wander, blink_rate = MOODS.get(mood, MOODS["neutral"])
            openness = min(1.0, openness * self.cal.b("wachheit"))
            wander *= self.cal.b("umherschauen")
            blink_rate *= self.cal.b("blinzeln")
            lr_c, ud_c = self._centre()

            if state == "idle" and idle_for > self.cal.b("einschlafen_nach_s"):
                state = "sleep"

            if (p := self.cal.pop_preview()) is not None:
                self._show_preview(p, openness)

            for a in actions:
                openness = self._do_action(a, openness)
            if self._sleep_now:
                if state == "listening":
                    self._sleep_now = False        # talking to it wakes it up
                else:
                    state = "sleep"
            if time.monotonic() < self._expr_until:
                openness = self._expr_openness
            if self._lids_closed_by_user:
                openness = 0.0

            if state == "sleep":
                self._gaze(lr_c, self.cal.gaze("ud")["unten"] + 10)
                self._lids(0.0)
                self._stop.wait(0.2)
                continue
            if state == "listening":
                self._lids_closed_by_user = False  # being talked to wakes it up
                openness = max(openness, 0.95)     # attentive: wide open
                wander *= 0.25
            elif state == "thinking":
                wander *= 0.5

            look = self._look if now - self._look_t < 2.5 else None
            if look is not None and state in ("idle", "listening") \
                    and now >= self._hold_gaze_until:
                # someone is talking: look their way (sides beyond ±90° = full turn)
                g = self.cal.gaze("lr")
                edge = g["links"] if look > 0 else g["rechts"]
                lr = lr_c + (edge - lr_c) * min(1.0, abs(look) / 90) * 0.9
                ud = ud_c
                if (el := self._look_el) is not None:
                    gu = self.cal.gaze("ud")
                    edge = gu["oben"] if el > 0 else gu["unten"]
                    ud = ud_c + (edge - ud_c) * min(1.0, abs(el) / 45) * 0.9
                if abs(self._pos.get(LR, lr_c) - lr) > 3 or abs(self._pos.get(UD, ud_c) - ud) > 3:
                    self._gaze(lr, ud)
                next_saccade = now + random.uniform(1.0, 2.5)
            elif now >= next_saccade and now >= self._hold_gaze_until:
                if state == "thinking":
                    # classic "searching memory" glance: up and to one side
                    lr = random.choice((lr_c - 35, lr_c + 35))
                    ud = ud_c + random.uniform(20, 35)
                    next_saccade = now + random.uniform(0.8, 1.6)
                else:
                    lr = lr_c + random.uniform(-30, 30) * wander
                    ud = ud_c + random.uniform(-15, 15) * wander
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
        g_lr, g_ud = self.cal.gaze("lr"), self.cal.gaze("ud")
        lr_c, ud_c = self._centre()
        if a in ("links", "rechts"):
            edge = g_lr[a]
            self._gaze(edge - (10 if edge > lr_c else -10), ud_c)
        elif a in ("oben", "unten"):
            edge = g_ud[a]
            self._gaze(lr_c, edge - (5 if edge > ud_c else -5))
        elif a == "geradeaus":
            self._gaze(lr_c, ud_c)
        elif a.startswith("zwinkern"):
            left = a.endswith("links")             # default: right eye
            self._lids(0.0, left=left, right=not left)
            time.sleep(0.35)
            self._lids(openness, left=left, right=not left)
            hold = 0.0
        elif a == "blinzeln":
            self._blink(openness)
            hold = 0.0
        elif a == "augen rollen":
            # half circle over the top: right → up-right → up → up-left → left → centre
            r, l, up = g_lr["rechts"], g_lr["links"], g_ud["oben"]
            for lr, ud in ((r, ud_c), ((r + lr_c) / 2, up - 10), (lr_c, up), ((l + lr_c) / 2, up - 10),
                           (l, ud_c), (lr_c, ud_c)):
                self._gaze(lr, ud)
                time.sleep(0.15)
            hold = time.monotonic() + 1.0
        elif a == "schlafen":
            self._sleep_now = True
            hold = 0.0
        elif a == "aufwachen":
            self._sleep_now = self._lids_closed_by_user = False
            self._expr_until, self._expr_openness = time.monotonic() + 1.5, 1.0
            self._gaze(lr_c, ud_c)
        elif a == "große augen":
            self._expr_until, self._expr_openness = time.monotonic() + 2.5, 1.0
            self._gaze(lr_c, ud_c)
        elif a == "skeptisch":
            self._expr_until, self._expr_openness = time.monotonic() + 2.5, 0.35
            self._gaze(lr_c + random.choice((-25, 25)), ud_c)
        elif a == "augen zu":
            self._lids_closed_by_user = True
            hold = 0.0
        elif a == "augen auf":
            self._lids_closed_by_user = False
            hold = 0.0
        self._hold_gaze_until = max(self._hold_gaze_until, hold)
        return openness
