"""Piper TTS in-process (piper-tts 1.8, OHF-Voice/piper1-gpl), or a natural
cloud voice (tts_cloud.py) in front of it while that is reachable.

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
_EN_WORDS = """
the a an and or but if then so not no yes is are was were be been am do does did have has had
i me my mine you your yours he him his she her it its we us our they them their this that these
those there here what when where who whom why how which can could will would shall should may
might must to of for with about from by at on up down out over under after before again now
today tomorrow tonight time weather please thanks thank sorry hello hi hey okay sure well very
really just only also too much many more most some any all every each one two three tell say
said show give make let get got go going come know think want need like love look see hear
talk speak sing play stop start help joke story funny short long little big good great nice
bad new old right left please name call robot robots television tv eyes again morning evening
night day week year "i'm" "it's" "don't" "can't" "you're" "what's" "that's" "let's" "isn't"
""".replace('"', "").split()
_DE_WORDS = """
der die das den dem des ein eine einen einem einer und oder aber wenn dann nicht kein keine ja
nein ist sind war waren sein bin bist hat habe hast haben ich mich mir mein meine du dich dir
dein deine er ihn ihm sie ihr es wir uns unser euch wer wo wann warum wieso weshalb wie welche
welcher kann kannst können könntest wird würde soll sollst muss darf zu von für mit über aus
bei nach vor heute morgen abend jetzt gleich bitte danke hallo tschüss gern gerne klar gut sehr
nur auch noch schon mal doch denn weil viel mehr alle etwas nichts erzähl erzähle erkläre sag
sage mach mache gib zeig schau hör spiel witz geschichte kurz kurzen lang wetter zeit uhr
roboter fernseher augen tag woche jahr wieder hier dort heißt heiße wohl eigentlich ganz
gute guten guter nacht lied musik singen spielen sprich sprechen licht tür spaß
""".split()
# words that exist in both languages carry no signal
_EN = frozenset(_EN_WORDS) - frozenset(_DE_WORDS) - {"in", "so", "was", "also", "will", "hat"}
_DE = frozenset(_DE_WORDS) - frozenset(_EN_WORDS) - {"in", "so", "was", "also", "will", "hat"}


def is_english(text: str) -> bool:
    words = re.findall(r"[a-zäöüß']+", text.lower())
    if re.search(r"[äöüß]", text.lower()):
        return False
    en = sum(w in _EN for w in words)
    de = sum(w in _DE for w in words)
    return en >= 2 and en > de


def detect_language(text: str, previous: str = "de") -> str:
    """'en' or 'de' for what the human said. Ambiguous utterances ("Hello?" —
    Parakeet also writes that for a German "Hallo", "OK.") keep the previous
    turn's language."""
    t = text.lower()
    if re.search(r"[äöüß]", t):
        return "de"
    words = re.findall(r"[a-z']+", t)
    en = sum(w in _EN for w in words)
    de = sum(w in _DE for w in words)
    if en >= 2 and en > de:
        return "en"
    if de >= 1 and de >= en:
        return "de"
    return previous


class Tts:
    def __init__(self, voice_path: str, out_rate: int, length_scale=0.9,
                 noise_scale=0.8, noise_w_scale=0.9, robot: str = "aus",
                 pitch: float = 1.0, tone: str = "neutral", en_voice_path: str | None = None,
                 cloud=None):
        self.cloud = cloud                 # e.g. tts_cloud.ElevenLabs; None = Piper only
        self.use_cloud = cloud is not None
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
        if self.cloud is not None:         # Piper's 0.9 length scale = normal cloud speed
            self.cloud.speed = round(min(1.2, max(0.7, 0.9 / self.base["length_scale"])), 2)
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
        if self.use_cloud and (audio := self.cloud.synth(text, "en" if english else "de")) is not None:
            from .tts_cloud import RATE        # pitch shift is Piper's; tone/robot apply to both
            audio = voicefx.resample(audio, RATE, self.out_rate, 1.0)
            audio = voicefx.tone(audio, self.out_rate, self.tone)
            return robotfx.apply(audio, self.out_rate, self.robot)
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
