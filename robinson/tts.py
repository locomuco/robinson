"""Piper TTS in-process (piper-tts 1.8, OHF-Voice/piper1-gpl).

Mood tags select a speaker of the multi-speaker thorsten_emotional voice;
single-speaker voices ignore the mood.
"""

import logging
import re
from math import gcd

import numpy as np
from piper import PiperVoice, SynthesisConfig
from scipy.signal import resample_poly

from . import robotfx

log = logging.getLogger("robinson.tts")

# thorsten_emotional speakers: amused=0 angry=1 disgusted=2 drunk=3 neutral=4
# sleepy=5 surprised=6 whisper=7
MOOD_SPEAKER = {
    "neutral": 4, "fröhlich": 0, "überrascht": 6, "nachdenklich": 4,
    "traurig": 5, "verärgert": 1, "müde": 5, "flüsternd": 7,
}

_CLEAN = re.compile(r"[*_#`~>|]+")


class Tts:
    def __init__(self, voice_path: str, out_rate: int, length_scale=0.9,
                 noise_scale=0.8, noise_w_scale=0.9, robot: str = "aus"):
        self.voice = PiperVoice.load(voice_path)
        self.multi = self.voice.config.num_speakers > 1
        self.out_rate = out_rate
        self.robot = robot
        mono = robotfx.monotone_factor(robot)   # robots speak flatter
        self.kw = dict(length_scale=length_scale, noise_scale=noise_scale * mono,
                       noise_w_scale=noise_w_scale * mono)
        log.info("TTS voice %s (%d speakers)", voice_path.rsplit("/", 1)[-1],
                 self.voice.config.num_speakers)

    def synth(self, text: str, mood: str = "neutral") -> np.ndarray | None:
        """Text → mono float32 at the output rate (None if nothing to say)."""
        text = _CLEAN.sub("", text).strip()
        if not re.search(r"\w", text):
            return None
        cfg = SynthesisConfig(
            speaker_id=MOOD_SPEAKER.get(mood, 4) if self.multi else None, **self.kw)
        chunks = list(self.voice.synthesize(text, syn_config=cfg))
        if not chunks:
            return None
        audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        src = chunks[0].sample_rate
        if src != self.out_rate:
            g = gcd(src, self.out_rate)
            audio = resample_poly(audio, self.out_rate // g, src // g).astype(np.float32)
        return robotfx.apply(audio, self.out_rate, self.robot)
