"""Speaker recognition: tell people apart by voice (sherpa-onnx embeddings).

A person is enrolled by voice ("Merk dir meine Stimme, ich bin Anna"); each
enrollment adds one voiceprint. Later utterances are matched by cosine
similarity, and the name is handed to the LLM.
"""

import json
import logging
import os
import re
import threading

import numpy as np

log = logging.getLogger("robinson.speakers")
SR = 16_000
MIN_SECONDS = 1.0

_ENROLL = re.compile(
    r"\b(merk|lern|speicher)\w*\s+(dir\s+)?(bitte\s+)?(meine|die)\s+stimme\b|"
    r"\berkenn\w*\s+(bitte\s+)?(mich|meine\s+stimme)\b|\bmeine\s+stimme\s+(merken|lernen)\b", re.I)
_NAME = re.compile(r"\b(?:ich bin|ich heiße|mein name ist|das ist|hier ist)\s+(?:der |die )?"
                   r"(?P<name>[A-ZÄÖÜ][\wäöüß-]+)")


def enroll_request(text: str) -> str | None:
    """Name from "Merk dir meine Stimme, ich bin Anna" (None = no enrollment request).
    Returns "" if the request has no recognisable name."""
    if not _ENROLL.search(text):
        return None
    m = _NAME.search(text)
    return m.group("name") if m else ""


class SpeakerId:
    def __init__(self, model_path: str, store: str, threshold: float = 0.5, threads: int = 2):
        import sherpa_onnx
        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model_path, num_threads=threads))
        self.store = store
        self.threshold = threshold
        self._lock = threading.Lock()
        self.voices: dict[str, list[list[float]]] = {}
        try:
            with open(store) as f:
                self.voices = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
        log.info("speaker id: %d known people", len(self.voices))

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        if len(audio) < MIN_SECONDS * SR:
            return None
        s = self.ex.create_stream()
        s.accept_waveform(SR, audio.astype(np.float32))
        s.input_finished()
        if not self.ex.is_ready(s):
            return None
        e = np.asarray(self.ex.compute(s), np.float32)
        return e / (np.linalg.norm(e) + 1e-9)

    def identify(self, audio: np.ndarray) -> str | None:
        e = self.embed(audio)
        if e is None or not self.voices:
            return None
        with self._lock:
            scores = {n: max(float(np.dot(e, np.asarray(v, np.float32))) for v in vs)
                      for n, vs in self.voices.items()}
        name = max(scores, key=scores.get)
        log.info("speaker scores: %s", {n: round(s, 2) for n, s in scores.items()})
        return name if scores[name] >= self.threshold else None

    def enroll(self, name: str, audio: np.ndarray) -> bool:
        e = self.embed(audio)
        if e is None:
            return False
        with self._lock:
            self.voices.setdefault(name, []).append(e.tolist())
            self.voices[name] = self.voices[name][-8:]
            os.makedirs(os.path.dirname(self.store), exist_ok=True)
            with open(self.store + ".tmp", "w") as f:
                json.dump(self.voices, f)
            os.replace(self.store + ".tmp", self.store)
        log.info("enrolled voice of %s", name)
        return True
