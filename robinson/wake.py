"""Wake word "Robinson" by keyword spotting (sherpa-onnx open-vocabulary KWS).

Speech recognition alone is unreliable for a lone "Robinson?" (heard as
"Robin's on", "Romanza", ...); a streaming keyword spotter listens for the
sound of the word itself. The transcript check in app.py stays as a second
way in.
"""

import logging
import os

import numpy as np

log = logging.getLogger("robinson.wake")

# BPE pieces of the model's 500-token vocabulary; "@" names the keyword
KEYWORDS = "▁RO B IN S ON @ROBINSON\n"


class WakeWord:
    def __init__(self, model_dir: str, threshold: float = 0.2, score: float = 1.5):
        import sherpa_onnx
        f = lambda name: os.path.join(model_dir, name)  # noqa: E731
        kw_file = os.path.join(model_dir, "robinson-keywords.txt")
        with open(kw_file, "w") as fh:
            fh.write(KEYWORDS)
        self.kws = sherpa_onnx.KeywordSpotter(
            tokens=f("tokens.txt"),
            encoder=f("encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
            decoder=f("decoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
            joiner=f("joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
            num_threads=1, keywords_file=kw_file, keywords_score=score,
            keywords_threshold=threshold, max_active_paths=4, provider="cpu")
        self.stream = self.kws.create_stream()

    def __call__(self, frame: np.ndarray) -> bool:
        """Feed 16 kHz mono float32; True when "Robinson" was just heard."""
        self.stream.accept_waveform(16_000, frame)
        hit = False
        while self.kws.is_ready(self.stream):
            self.kws.decode_stream(self.stream)
            if self.kws.get_result(self.stream):
                hit = True
                self.kws.reset_stream(self.stream)
        return hit
