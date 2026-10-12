"""Robot voice effect applied to piper's output.

ring modulation (metallic timbre) → comb filter (tin-can resonance) →
gentle limiter. Presets go from "aus" (plain piper) to "stark".
"""

import numpy as np
from scipy.signal import lfilter

# ring_hz, ring_mix, comb_ms, comb_fb, monotone (piper noise scale factor)
PRESETS = {
    "aus":    None,
    "leicht": (55.0, 0.35, 6.0, 0.35, 0.7),
    "mittel": (45.0, 0.6, 8.0, 0.5, 0.45),
    "stark":  (35.0, 0.9, 10.0, 0.65, 0.25),
}


def monotone_factor(preset: str) -> float:
    p = PRESETS.get(preset)
    return 1.0 if p is None else p[4]


def apply(audio: np.ndarray, rate: int, preset: str) -> np.ndarray:
    p = PRESETS.get(preset)
    if p is None or audio.size == 0:
        return audio
    ring_hz, ring_mix, comb_ms, comb_fb, _ = p
    t = np.arange(len(audio)) / rate
    x = audio * ((1 - ring_mix) + ring_mix * np.sin(2 * np.pi * ring_hz * t))
    d = max(1, int(rate * comb_ms / 1000))
    a = np.zeros(d + 1)
    a[0], a[d] = 1.0, -comb_fb             # y[n] = x[n] + fb * y[n-d]
    x = lfilter([1.0 - comb_fb], a, x)
    # soft limiter keeps the resonance from clipping, then restore loudness
    x = np.tanh(1.5 * x / (np.abs(x).max() + 1e-9)) / np.tanh(1.5)
    return x.astype(np.float32)
