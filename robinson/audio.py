"""Full-duplex audio on the USB adapter with optional echo cancellation.

One PortAudio stream drives both directions at the device rate, so every mic
block comes paired with exactly the samples played at that moment — the
reference WebRTC AEC needs to subtract Robinson's own voice from the mic.
The callback only moves buffers; decimation, AEC and VAD run in a worker.

Output: play() appends mono float32 at the device rate; flush() drops
everything queued (barge-in). Input: 16 kHz mono float32 frames of 10 ms on
`mic` (a queue), after AEC.
"""

import collections
import logging
import queue
import threading

import numpy as np
import sounddevice as sd
from scipy.signal import firwin, lfilter, lfilter_zi

log = logging.getLogger("robinson.audio")

SR = 16_000
FRAME = SR // 100           # 10 ms at 16 kHz


class _Decimator:
    """Stateful anti-aliased integer decimation (48 kHz → 16 kHz)."""

    def __init__(self, factor: int):
        self.f = factor
        self.taps = firwin(63, 0.9 / factor)
        self.zi = lfilter_zi(self.taps, 1.0) * 0.0
        self.phase = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = lfilter(self.taps, 1.0, x, zi=self.zi)
        out = y[self.phase::self.f]
        self.phase = (self.phase - len(x)) % self.f
        return out.astype(np.float32)


class _Aec:
    """WebRTC audio processing (AEC + noise suppression) via livekit.rtc."""

    def __init__(self, delay_ms: int):
        from livekit import rtc
        self.rtc = rtc
        self.apm = rtc.AudioProcessingModule(
            echo_cancellation=True, noise_suppression=True,
            high_pass_filter=True, auto_gain_control=False)
        self.apm.set_stream_delay_ms(delay_ms)

    def _frame(self, x: np.ndarray):
        pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        return self.rtc.AudioFrame(pcm.tobytes(), SR, 1, len(pcm))

    def __call__(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        self.apm.process_reverse_stream(self._frame(ref))
        f = self._frame(mic)
        self.apm.process_stream(f)
        return np.frombuffer(f.data, np.int16).astype(np.float32) / 32768


def find_device(name: str) -> int:
    for i, d in enumerate(sd.query_devices()):
        if name.lower() in d["name"].lower() and d["max_input_channels"] > 0 \
                and d["max_output_channels"] > 0:
            return i
    raise RuntimeError(f"no duplex audio device matching {name!r}:\n{sd.query_devices()}")


class AudioIO:
    def __init__(self, device: str, rate: int = 48_000, aec: bool = True,
                 aec_delay_ms: int = 60, gain: float = 1.0):
        self.rate = rate
        self.block = rate // 50                   # 20 ms device blocks
        self.gain = gain
        self.mic: queue.Queue[np.ndarray] = queue.Queue(maxsize=3000)  # 30 s
        self._raw: queue.Queue = queue.Queue(maxsize=500)
        self._out = collections.deque()           # np arrays at device rate
        self._out_pos = 0
        self._out_lock = threading.Lock()
        self.drained = threading.Event()
        self.drained.set()
        self.aec = None
        if aec:
            try:
                self.aec = _Aec(aec_delay_ms)
                log.info("echo cancellation on (WebRTC)")
            except Exception as e:
                log.warning("echo cancellation unavailable: %s", e)
        dev = find_device(device)
        out_ch = min(2, sd.query_devices(dev)["max_output_channels"])
        self.stream = sd.Stream(
            device=(dev, dev), samplerate=rate, blocksize=self.block,
            channels=(1, out_ch), dtype="float32", callback=self._callback,
            latency="high")              # survive LLM prefill bursts on all cores
        self._dec_mic, self._dec_ref = _Decimator(rate // SR), _Decimator(rate // SR)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._process, name="audio", daemon=True)

    def start(self) -> None:
        self._worker.start()
        self.stream.start()
        log.info("audio running on %s @ %d Hz", self.stream.device, self.rate)

    def close(self) -> None:
        self._stop.set()
        self.stream.close()

    # ── output ───────────────────────────────────────────────────────────────
    def play(self, audio: np.ndarray) -> None:
        with self._out_lock:
            self._out.append(audio.astype(np.float32) * self.gain)
            self.drained.clear()

    def flush(self) -> None:
        with self._out_lock:
            self._out.clear()
            self._out_pos = 0
            self.drained.set()

    @property
    def playing(self) -> bool:
        return not self.drained.is_set()

    def _pull(self, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        filled = 0
        with self._out_lock:
            while filled < n and self._out:
                cur = self._out[0]
                take = min(n - filled, len(cur) - self._out_pos)
                out[filled:filled + take] = cur[self._out_pos:self._out_pos + take]
                filled += take
                self._out_pos += take
                if self._out_pos >= len(cur):
                    self._out.popleft()
                    self._out_pos = 0
            if not self._out:
                self.drained.set()
        return out

    # ── realtime callback: copy only ─────────────────────────────────────────
    def _callback(self, indata, outdata, frames, _time, status):
        if status:
            log.debug("audio status: %s", status)
        ref = self._pull(frames)
        outdata[:] = ref[:, None]
        try:
            self._raw.put_nowait((indata[:, 0].copy(), ref))
        except queue.Full:
            pass

    # ── worker: decimate, AEC, publish 10 ms frames ─────────────────────────
    def _process(self) -> None:
        mic_buf = np.zeros(0, np.float32)
        ref_buf = np.zeros(0, np.float32)
        while not self._stop.is_set():
            try:
                mic, ref = self._raw.get(timeout=0.5)
            except queue.Empty:
                continue
            mic_buf = np.concatenate([mic_buf, self._dec_mic(mic)])
            ref_buf = np.concatenate([ref_buf, self._dec_ref(ref)])
            while len(mic_buf) >= FRAME and len(ref_buf) >= FRAME:
                m, mic_buf = mic_buf[:FRAME], mic_buf[FRAME:]
                r, ref_buf = ref_buf[:FRAME], ref_buf[FRAME:]
                if self.aec is not None:
                    m = self.aec(m, r)
                if self.mic.full():
                    self.mic.get_nowait()
                self.mic.put_nowait(m)
