"""Audio I/O with optional echo cancellation.

Duplex mode (USB adapter only): one PortAudio stream drives both directions
at the device rate, so every mic block comes paired with exactly the samples
played at that moment — the reference WebRTC AEC needs to subtract Robinson's
own voice from the mic.

Split mode (mic_device found, e.g. the ReSpeaker 4-Mic on the PIO card): the
mic array records 16 kHz multichannel on its own stream and is beamformed to
mono (mic_array.py, which also estimates where speech comes from);
the speaker plays on the USB adapter, or nowhere if it is missing. Mic and
reference are paired by arrival; WebRTC's delay estimation covers the rest.
The callbacks only move buffers; decimation, AEC and VAD run in a worker.

Output: play() appends mono float32 at the device rate; flush() drops
everything queued (barge-in). Input: 16 kHz mono float32 frames of 10 ms on
`mic` (a queue), after AEC.
"""

import collections
import logging
import queue
import threading
import time

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


def find_device(name: str, need_in: bool = True, need_out: bool = True) -> int:
    for i, d in enumerate(sd.query_devices()):
        if name and name.lower() in d["name"].lower() \
                and (not need_in or d["max_input_channels"] > 0) \
                and (not need_out or d["max_output_channels"] > 0):
            return i
    kind = "duplex" if need_in and need_out else "input" if need_in else "output"
    raise RuntimeError(f"no {kind} audio device matching {name!r}:\n{sd.query_devices()}")


def _try_device(name: str, need_in: bool, need_out: bool) -> int | None:
    try:
        return find_device(name, need_in, need_out)
    except RuntimeError:
        return None


class AudioIO:
    def __init__(self, device: str, rate: int = 48_000, aec: bool = True,
                 aec_delay_ms: int = 60, gain: float = 1.0, mic_device: str = "",
                 mic_channels: int = 4, beamform: bool = True):
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
        self.streams: list = []
        self.array = None                         # MicArray when recording from the 4-mic board
        self._mic_dev = None                      # split mode: PortAudio index of the mic array
        self._null_out = False
        mic = _try_device(mic_device, True, False) if mic_device else None
        if mic is None:
            if mic_device:
                log.warning("mic device %r not found, using the mic of %r", mic_device, device)
            dev = find_device(device)
            out_ch = min(2, sd.query_devices(dev)["max_output_channels"])
            self.streams.append(sd.Stream(
                device=(dev, dev), samplerate=rate, blocksize=self.block,
                channels=(1, out_ch), dtype="float32", callback=self._callback,
                latency="high"))         # survive LLM prefill bursts on all cores
        else:
            self.mic_channels = mic_channels
            if mic_channels == 4:
                from .mic_array import MicArray
                self.array = MicArray(beamform)
            self._mic_dev = mic
            self.streams.append(self._open_mic())
            out = _try_device(device, False, True)
            if out is None:
                log.warning("no speaker matching %r - playing into the void", device)
                self._null_out = True
            else:
                out_ch = min(2, sd.query_devices(out)["max_output_channels"])
                self.streams.append(sd.OutputStream(
                    device=out, samplerate=rate, blocksize=self.block, channels=out_ch,
                    dtype="float32", callback=self._out_callback, latency="high"))
        self._dec_mic, self._dec_ref = _Decimator(rate // SR), _Decimator(rate // SR)
        self._raw_peak = 0.0
        self._last_mic = time.monotonic()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._process, name="audio", daemon=True)

    def _open_mic(self):
        return sd.InputStream(
            device=self._mic_dev, samplerate=SR, blocksize=SR // 50, channels=self.mic_channels,
            dtype="int32", callback=self._in_callback, latency="high")

    def _revive_mic(self) -> None:
        """The mic stream went silent (PortAudio gives up after a failed xrun
        recovery, e.g. while the codec clocks restart): reopen it."""
        log.warning("no mic audio for %.0f s - reopening the mic stream",
                    time.monotonic() - self._last_mic)
        self._last_mic = time.monotonic()           # next attempt in a few seconds
        try:
            self.streams[0].close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.streams[0] = self._open_mic()
            self.streams[0].start()
        except Exception as e:  # noqa: BLE001
            log.warning("mic stream not reopened yet: %s", e)

    def start(self) -> None:
        self._last_mic = time.monotonic()
        self._worker.start()
        for st in self.streams:
            st.start()
        if self._null_out:
            threading.Thread(target=self._null_output, name="null-out", daemon=True).start()
        log.info("audio running on %s, output @ %d Hz",
                 ", ".join(str(st.device) for st in self.streams), self.rate)

    def close(self) -> None:
        self._stop.set()
        for st in self.streams:
            st.close()

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

    def raw_peak(self) -> float:
        """Max block RMS of the raw mic (before AEC) since the last call."""
        p, self._raw_peak = self._raw_peak, 0.0
        return p

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
        self._put(("duplex", indata[:, 0].copy(), ref))

    def _in_callback(self, indata, frames, _time, status):
        if status:
            log.debug("mic status: %s", status)
        self._last_mic = time.monotonic()
        self._put(("mic", indata.copy()))

    def _out_callback(self, outdata, frames, _time, status):
        if status:
            log.debug("speaker status: %s", status)
        ref = self._pull(frames)
        outdata[:] = ref[:, None]
        self._put(("ref", ref))

    def _null_output(self) -> None:
        """No speaker: consume the output queue in real time so `playing` still ends."""
        t = time.monotonic()
        while not self._stop.is_set():
            self._put(("ref", self._pull(self.block)))
            t += self.block / self.rate
            time.sleep(max(0.0, t - time.monotonic()))

    def _put(self, item) -> None:
        try:
            self._raw.put_nowait(item)
        except queue.Full:
            pass

    # ── worker: decimate, AEC, publish 10 ms frames ─────────────────────────
    def _process(self) -> None:
        mic_buf = np.zeros(0, np.float32)
        ref_buf = np.zeros(0, np.float32)
        while not self._stop.is_set():
            try:
                kind, *data = self._raw.get(timeout=0.5)
            except queue.Empty:
                kind = None
            if self._mic_dev is not None and time.monotonic() - self._last_mic > 3:
                self._revive_mic()
            if kind is None:
                continue
            if kind == "ref":
                ref_buf = np.concatenate([ref_buf, self._dec_ref(data[0])])[-SR // 4:]
                continue
            if kind == "duplex":
                mic = data[0]
                ref_buf = np.concatenate([ref_buf, self._dec_ref(data[1])])
                mic_buf = np.concatenate([mic_buf, self._dec_mic(mic)])
            else:                                 # array frames: int32 → mono float
                x = data[0].astype(np.float32) / 2**31
                mic = self.array.process(x) if self.array else x.mean(axis=1)
                mic_buf = np.concatenate([mic_buf, mic])
            self._raw_peak = max(self._raw_peak, float(np.sqrt(np.mean(mic ** 2))))
            while len(mic_buf) >= FRAME:
                m, mic_buf = mic_buf[:FRAME], mic_buf[FRAME:]
                if len(ref_buf) >= FRAME:
                    r, ref_buf = ref_buf[:FRAME], ref_buf[FRAME:]
                elif kind == "duplex":
                    mic_buf = np.concatenate([m, mic_buf])
                    break
                else:
                    r = np.zeros(FRAME, np.float32)
                if self.aec is not None:
                    m = self.aec(m, r)
                if self.mic.full():
                    self.mic.get_nowait()
                self.mic.put_nowait(m)
