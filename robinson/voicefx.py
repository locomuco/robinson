"""Voice colour for piper output: pitch/formant shift and a tone EQ.

Pitch: the audio is resampled as if recorded at rate*pitch, which shifts
pitch *and* formants (sounds like a different, larger/smaller speaker);
piper's length_scale is multiplied by the same factor so the speaking tempo
stays unchanged. Tone: RBJ shelving EQ — "warm" (more body, softer highs),
"hell" (brighter, clearer on small speakers) or "neutral".
"""

from fractions import Fraction

import numpy as np
from scipy.signal import lfilter, resample_poly

# tone → [(type, freq Hz, gain dB)]
TONES = {
    "neutral": [],
    "warm": [("low", 250, 4.0), ("high", 4000, -3.0)],
    "hell": [("low", 200, -2.0), ("high", 3000, 4.0)],
}


def _shelf(kind: str, f0: float, gain_db: float, rate: int):
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / rate
    cos, alpha = np.cos(w0), np.sin(w0) / 2 * np.sqrt(2)   # shelf slope S = 1
    sa = 2 * np.sqrt(a) * alpha
    if kind == "low":
        b = [a * ((a + 1) - (a - 1) * cos + sa), 2 * a * ((a - 1) - (a + 1) * cos),
             a * ((a + 1) - (a - 1) * cos - sa)]
        d = [(a + 1) + (a - 1) * cos + sa, -2 * ((a - 1) + (a + 1) * cos),
             (a + 1) + (a - 1) * cos - sa]
    else:
        b = [a * ((a + 1) + (a - 1) * cos + sa), -2 * a * ((a - 1) + (a + 1) * cos),
             a * ((a + 1) + (a - 1) * cos - sa)]
        d = [(a + 1) - (a - 1) * cos + sa, 2 * ((a - 1) - (a + 1) * cos),
             (a + 1) - (a - 1) * cos - sa]
    return np.array(b) / d[0], np.array(d) / d[0]


def resample(audio: np.ndarray, src: int, dst: int, pitch: float = 1.0) -> np.ndarray:
    """src→dst sample rate; pitch > 1 raises voice (plays the src audio faster)."""
    ratio = Fraction(dst, 1) / (Fraction(src, 1) * Fraction(pitch).limit_denominator(200))
    ratio = ratio.limit_denominator(2000)
    if ratio == 1:
        return audio
    return resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)


def tone(audio: np.ndarray, rate: int, name: str) -> np.ndarray:
    for kind, f0, gain in TONES.get(name, []):
        b, a = _shelf(kind, f0, gain, rate)
        audio = lfilter(b, a, audio)
    peak = np.abs(audio).max() if audio.size else 0
    if peak > 1.0:                       # EQ boost must not clip
        audio = audio / peak
    return audio.astype(np.float32)
