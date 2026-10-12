"""Speech-to-text engines, loaded once and kept in memory.

parakeet: NVIDIA Parakeet TDT 0.6B v3 (int8) via sherpa-onnx — multilingual
          incl. German, punctuated output.
whisper:  whisper.cpp via pywhispercpp (build with GGML_NATIVE=ON on the Pi).
"""

import logging
import time

import numpy as np

log = logging.getLogger("robinson.stt")
SR = 16_000


class ParakeetStt:
    def __init__(self, model_dir: str, threads: int = 4):
        import sherpa_onnx
        d = model_dir
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=f"{d}/encoder.int8.onnx", decoder=f"{d}/decoder.int8.onnx",
            joiner=f"{d}/joiner.int8.onnx", tokens=f"{d}/tokens.txt",
            num_threads=threads, model_type="nemo_transducer",
            decoding_method="greedy_search",
        )

    def __call__(self, audio: np.ndarray) -> str:
        s = self.rec.create_stream()
        s.accept_waveform(SR, audio)
        self.rec.decode_stream(s)
        return s.result.text.strip()


class WhisperStt:
    def __init__(self, model_path: str, threads: int = 4, language: str = "de"):
        from pywhispercpp.model import Model
        self.m = Model(model_path, n_threads=threads, language=language,
                       print_progress=False, print_realtime=False,
                       no_context=True, single_segment=True)

    def __call__(self, audio: np.ndarray) -> str:
        return " ".join(s.text for s in self.m.transcribe(audio)).strip()


def make_stt(cfg):
    t0 = time.monotonic()
    if cfg.stt_engine == "parakeet":
        stt = ParakeetStt(cfg.path(cfg.parakeet_dir), cfg.stt_threads)
    elif cfg.stt_engine == "whisper":
        stt = WhisperStt(cfg.path(cfg.whisper_model), cfg.stt_threads, cfg.language)
    else:
        raise ValueError(f"unknown stt_engine {cfg.stt_engine!r}")
    log.info("STT %s ready in %.1fs", cfg.stt_engine, time.monotonic() - t0)
    return stt


def is_hallucination(text: str) -> bool:
    """Drop typical non-speech transcripts (noise, coughs, subtitles credits)."""
    t = text.strip().lower().strip(".!?, ")
    return not t or t in {"äh", "ähm", "hm", "mhm"} or "untertitel" in t
