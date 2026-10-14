"""Piper TTS in-process (piper-tts 1.8, OHF-Voice/piper1-gpl).

Mood tags select a speaker of the multi-speaker thorsten_emotional voice;
single-speaker voices ignore the mood.
"""

import logging
import os
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
_EN = frozenset("the and is are you your i it to of that this with for was what how can do not have be "
                "hello hi yes no please thanks thank sure well my me we they he she in on at as but "
                "will would could should there here just so very good great".split())
_DE = frozenset("der die das und ist sind du dein ich es zu von dass mit für war was wie kann nicht "
                "habe hallo ja nein bitte danke klar mein mir wir sie er in auf an als aber wird "
                "würde könnte soll da hier nur so sehr gut".split())


def is_english(text: str) -> bool:
    words = re.findall(r"[a-zäöüß']+", text.lower())
    if re.search(r"[äöüß]", text.lower()):
        return False
    en = sum(w in _EN for w in words)
    de = sum(w in _DE for w in words)
    return en >= 2 and en > de


def detect_language(text: str, previous: str = "de") -> str:
    """'en' or 'de' for what the human said. Ambiguous one-worders ("Hello?" —
    Parakeet also writes that for a German "Hallo") keep the previous language."""
    t = text.lower()
    words = re.findall(r"[a-zäöüß']+", t)
    if re.search(r"[äöüß]", t):
        return "de"
    en = sum(w in _EN for w in words)
    de = sum(w in _DE for w in words)
    if en > de and (en >= 2 or len(words) <= 3 and en == len(words) and len(words) > 1):
        return "en"
    if de > en or (de and de == en):
        return "de"
    return previous


class Tts:
    def __init__(self, voice_path: str, out_rate: int, length_scale=0.9,
                 noise_scale=0.8, noise_w_scale=0.9, robot: str = "aus",
                 pitch: float = 1.0, tone: str = "neutral", en_voice_path: str | None = None):
        self.voice = PiperVoice.load(voice_path)
        self.en_voice = None
        if en_voice_path and os.path.exists(en_voice_path):
            try:
                self.en_voice = PiperVoice.load(en_voice_path)
            except Exception:
                log.exception("English voice not loadable")
        self.multi = self.voice.config.num_speakers > 1
        self.out_rate = out_rate
        self.base = dict(length_scale=length_scale, noise_scale=noise_scale,
                         noise_w_scale=noise_w_scale)
        self.configure(robot=robot, pitch=pitch, tone=tone)
        log.info("TTS voice %s (%d speakers)", voice_path.rsplit("/", 1)[-1],
                 self.voice.config.num_speakers)

    def configure(self, robot: str | None = None, pitch: float | None = None,
                  tone: str | None = None, length_scale: float | None = None) -> None:
        """Change the voice at runtime (settings by voice); None keeps a value."""
        if length_scale is not None:
            self.base["length_scale"] = length_scale
        self.robot = robot if robot is not None else getattr(self, "robot", "aus")
        self.pitch = pitch if pitch is not None else getattr(self, "pitch", 1.0)
        self.tone = tone if tone is not None else getattr(self, "tone", "neutral")
        mono = robotfx.monotone_factor(self.robot)   # robots speak flatter
        # pitch shift speeds the audio up by `pitch`; synthesize slower to compensate
        self.kw = dict(length_scale=self.base["length_scale"] * self.pitch,
                       noise_scale=self.base["noise_scale"] * mono,
                       noise_w_scale=self.base["noise_w_scale"] * mono)

    def synth(self, text: str, mood: str = "neutral", lang: str | None = None) -> np.ndarray | None:
        """Text → mono float32 at the output rate (None if nothing to say)."""
        text = _CLEAN.sub("", text).strip()
        if not re.search(r"\w", text):
            return None
        voice, multi = self.voice, self.multi
        english = lang == "en" if lang else is_english(text)
        if self.en_voice and english:              # English reply → native English voice
            voice, multi = self.en_voice, False
        cfg = SynthesisConfig(
            speaker_id=MOOD_SPEAKER.get(mood, 4) if multi else None, **self.kw)
        chunks = list(voice.synthesize(text, syn_config=cfg))
        if not chunks:
            return None
        audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        audio = voicefx.resample(audio, chunks[0].sample_rate, self.out_rate, self.pitch)
        audio = voicefx.tone(audio, self.out_rate, self.tone)
        return robotfx.apply(audio, self.out_rate, self.robot)
