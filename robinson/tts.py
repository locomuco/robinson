"""Piper TTS in-process (piper-tts 1.8, OHF-Voice/piper1-gpl).

Mood tags select a speaker of the multi-speaker thorsten_emotional voice;
single-speaker voices ignore the mood.
"""

import logging
import re
import numpy as np
from piper import PiperVoice, SynthesisConfig

from . import robotfx, voicefx

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
                 noise_scale=0.8, noise_w_scale=0.9, robot: str = "aus",
                 pitch: float = 1.0, tone: str = "neutral"):
        self.voice = PiperVoice.load(voice_path)
        self.multi = self.voice.config.num_speakers > 1
        self.out_rate = out_rate
        self.robot = robot
        self.pitch, self.tone = pitch, tone
        mono = robotfx.monotone_factor(robot)   # robots speak flatter
        # pitch shift speeds the audio up by `pitch`; synthesize slower to compensate
        self.kw = dict(length_scale=length_scale * pitch, noise_scale=noise_scale * mono,
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
        audio = voicefx.resample(audio, chunks[0].sample_rate, self.out_rate, self.pitch)
        audio = voicefx.tone(audio, self.out_rate, self.tone)
        return robotfx.apply(audio, self.out_rate, self.robot)
