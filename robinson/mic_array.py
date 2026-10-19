"""ReSpeaker 4-Mic Array: direction of arrival + delay-and-sum beamforming.

Model: four mics on a circle, mic i at angle 90°·i in the array's own frame
(azimuth counter-clockwise). A plane wave from azimuth θ reaches mic i at
    t_i = -r·cos(θ - 90°·i) / c.
The opposite pairs (0,2) and (1,3) give, via GCC-PHAT,
    t0 - t2 = -(d/c)·cos θ,   t1 - t3 = -(d/c)·sin θ   →   θ = atan2(-(t1-t3), -(t0-t2)).
DOA and beamformer use the same model, so if the board numbers its mics
clockwise both are mirrored consistently; the mapping to the room (where is
"front", mirrored or not) is configuration (mic_front_deg, mic_mirror).

Beamforming delays every channel so a wave from the steering direction lines
up, then averages. Unaligned averaging (what a plain mono mix does) cancels
treble for anyone not straight above the board: the pairs are up to 3.8
samples apart at 16 kHz, so the first comb notch sits near 2 kHz.
"""

import logging
import math
import threading
import time

import numpy as np

log = logging.getLogger("robinson.array")

SR = 16_000
C = 343.0                      # speed of sound, m/s
DIAG = 0.0813                  # distance between opposite mics (ReSpeaker 4-Mic), m
TAPS = 33                      # fractional-delay FIR length
DOA_N = 1024                   # GCC-PHAT window (64 ms)
DOA_HOP = 512
INTERP = 8                     # GCC upsampling for sub-sample delays


def gcc_phat(a: np.ndarray, b: np.ndarray, max_tau: float) -> tuple[float, float]:
    """Delay of a relative to b in seconds (positive: a lags) and peak strength."""
    n = 2 * len(a)
    A, B = np.fft.rfft(a, n), np.fft.rfft(b, n)
    R = A * np.conj(B)
    f = np.fft.rfftfreq(n, 1 / SR)
    R[(f < 200) | (f > 5000)] = 0                 # speech band only
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, n * INTERP)
    m = int(INTERP * SR * max_tau) + 1
    cc = np.concatenate((cc[-m:], cc[:m + 1]))
    k = int(np.argmax(cc))
    strength = float(cc[k]) * INTERP * n / max(1, 2 * np.count_nonzero(R))   # 1 = identical
    return (k - m) / (INTERP * SR), strength


def _frac_delay(d: float) -> np.ndarray:
    """Windowed-sinc FIR delaying by TAPS//2 + d samples (|d| < TAPS//2)."""
    k = np.arange(TAPS) - TAPS // 2 - d
    h = np.sinc(k) * np.hamming(TAPS)
    return (h / h.sum()).astype(np.float32)


class MicArray:
    def __init__(self, beamform: bool = True, min_strength: float = 0.25):
        self.beamform = beamform
        self.min_strength = min_strength
        self.active = False            # set by the listener: speech, and not our own voice
        self.on_doa = None             # callback(azimuth_deg_array_frame, strength)
        self.azimuth: float | None = None
        self.az_time = 0.0
        self._lock = threading.Lock()
        self._hist = np.zeros((TAPS - 1, 4), np.float32)
        self._doa_buf = np.zeros((DOA_N, 4), np.float32)
        self._since = 0
        self._recent: list[tuple[float, float]] = []   # (time, azimuth) of good estimates
        self._steer(None)

    # ── beamformer ───────────────────────────────────────────────────────────
    def _steer(self, az: float | None) -> None:
        if az is None:
            delays = np.zeros(4)
        else:
            th = math.radians(az)
            t = np.array([-(DIAG / 2) * math.cos(th - math.pi / 2 * i) / C for i in range(4)])
            delays = (t.max() - t) * SR             # delay the early mics
            delays -= delays.mean()
        self._filters = np.stack([_frac_delay(d) for d in delays])
        self.steering = az

    def process(self, x: np.ndarray) -> np.ndarray:
        """(n, 4) float32 block → mono float32 (beamformed if steered)."""
        buf = np.concatenate([self._hist, x])
        self._hist = buf[-(TAPS - 1):]
        if self.beamform and self.steering is not None:
            y = sum(np.convolve(buf[:, i], self._filters[i], "valid") for i in range(4)) / 4
        else:
            y = buf[TAPS // 2:TAPS // 2 + len(x)].mean(axis=1)   # same latency, plain mix
        self._feed_doa(x)
        return y.astype(np.float32)

    # ── direction of arrival ─────────────────────────────────────────────────
    def _feed_doa(self, x: np.ndarray) -> None:
        n = len(x)
        self._doa_buf = np.concatenate([self._doa_buf[n:], x]) if n < DOA_N else x[-DOA_N:]
        self._since += n
        if self._since < DOA_HOP:
            return
        self._since = 0
        if not self.active:
            return
        res = self.estimate(self._doa_buf)
        if res is None:
            return
        az, strength = res
        now = time.monotonic()
        with self._lock:
            self._recent = [(t, a) for t, a in self._recent if now - t < 1.0] + [(now, az)]
            # circular mean of the last second: one wrong frame doesn't move the beam
            s = sum(math.sin(math.radians(a)) for _, a in self._recent)
            c = sum(math.cos(math.radians(a)) for _, a in self._recent)
            self.azimuth, self.az_time = math.degrees(math.atan2(s, c)) % 360, now
        if self.steering is None or _angle_diff(self.azimuth, self.steering) > 10:
            self._steer(self.azimuth)
        if self.on_doa:
            self.on_doa(self.azimuth, strength)

    def estimate(self, x: np.ndarray) -> tuple[float, float] | None:
        """(azimuth in the array frame, strength) of the dominant source, or None."""
        max_tau = DIAG / C
        t02, s02 = gcc_phat(x[:, 0], x[:, 2], max_tau)
        t13, s13 = gcc_phat(x[:, 1], x[:, 3], max_tau)
        strength = min(s02, s13)
        if strength < self.min_strength:
            return None
        return math.degrees(math.atan2(-t13, -t02)) % 360, strength


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def to_room(az_array: float, front_deg: float, mirror: bool) -> float:
    """Array-frame azimuth → room azimuth: 0 = straight ahead, + = to Robinson's left."""
    a = (az_array - front_deg) % 360
    a = -a if mirror else a
    return (a + 180) % 360 - 180
