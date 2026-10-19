"""Natural voice from ElevenLabs while online; Piper (tts.py) is the fallback.

One HTTPS request per sentence, PCM back (~0.3 s to the first byte with
eleven_flash_v2_5). Short fixed phrases ("Ja?", fillers) are cached on disk,
so they cost no characters and no latency the second time. After a failure
the cloud is skipped for a minute and Piper speaks.
"""

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request

import numpy as np

log = logging.getLogger("robinson.tts")

RATE = 24_000                  # pcm_24000 works on every ElevenLabs plan


class ElevenLabs:
    def __init__(self, api_key: str, voice_id: str, model: str, cache_dir: str):
        self.key, self.voice_id, self.model = api_key, voice_id, model
        self.cache_dir = cache_dir
        self.speed = 1.0
        self.down_until = 0.0
        os.makedirs(cache_dir, exist_ok=True)

    @property
    def available(self) -> bool:
        return time.monotonic() >= self.down_until

    def synth(self, text: str, lang: str) -> np.ndarray | None:
        """Text → mono float32 at RATE, or None (then Piper speaks)."""
        if not self.available:
            return None
        cached = len(text) <= 40
        path = os.path.join(self.cache_dir, hashlib.sha1(
            f"{self.voice_id}|{self.model}|{lang}|{self.speed}|{text}".encode()).hexdigest() + ".npy")
        if cached and os.path.exists(path):
            return np.load(path)
        body = json.dumps({
            "text": text, "model_id": self.model, "language_code": lang,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "speed": self.speed},
        }).encode()
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}?output_format=pcm_{RATE}",
            data=body, headers={"xi-api-key": self.key, "Content-Type": "application/json"})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=4) as resp:
                pcm = resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            detail = e.read()[:200].decode(errors="replace") if isinstance(e, urllib.error.HTTPError) else e
            log.warning("ElevenLabs failed (%s) - Piper speaks for a minute", detail)
            self.down_until = time.monotonic() + 60
            return None
        audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
        log.debug("elevenlabs %.2fs: %s", time.monotonic() - t0, text[:40])
        if cached and len(audio):
            np.save(path, audio)
        return audio if len(audio) else None
