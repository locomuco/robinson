#!/usr/bin/env python3
"""
Robinson — local voice assistant
Mic → VAD → whisper.cpp → ollama → piper → speaker

Pipeline:
  - TTS synthesis (piper) and playback run in separate threads so sentence N+1
    is synthesised while sentence N plays — no inter-sentence gap.
  - When the player starts the LAST audio segment it signals the main thread to
    stop discarding mic frames and start accumulating them. By the time the bot
    finishes, those frames are already in the queue and capture_utterance picks
    them up immediately with flush=False — zero dead time after the last word.
  - Barge-in: sustained user speech above a higher threshold interrupts the bot.
"""

import collections
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave

import alsaaudio
import httpx
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from math import gcd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT          = os.path.dirname(os.path.abspath(__file__))
WHISPER_BIN   = os.path.join(ROOT, "whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL = os.path.join(ROOT, "whisper.cpp/models/ggml-base.bin")
PIPER_BIN     = os.path.join(ROOT, "piper/piper")
PIPER_VOICE   = os.path.join(ROOT, "piper/de_DE-thorsten_emotional-medium.onnx")
# Available speakers: amused=0, angry=1, disgusted=2, drunk=3,
#                     neutral=4, sleepy=5, surprised=6, whisper=7
PIPER_SPEAKER = 0  # 0=amused (cheerful), 6=surprised (also lively)

# ── Ollama ───────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"
SYSTEM_MSG   = {
    "role": "system",
    "content": (
        "you are robinson, a robot built into a retro TV"
    ),
}

# ── Audio input ──────────────────────────────────────────────────────────────
SAMPLE_RATE        = 16_000
CHANNELS           = 1
FRAME_MS           = 30
FRAME_SAMPLES      = SAMPLE_RATE * FRAME_MS // 1000  # 480 samples = 30 ms
PRE_ROLL_FRAMES    = 10
MIN_SPEECH_FRAMES  = 8
SILENCE_FRAMES     = 25
ENERGY_THRESHOLD   = 0.015

BARGE_IN_THRESHOLD = ENERGY_THRESHOLD * 3  # higher — survives speaker bleed-through
BARGE_IN_FRAMES    = 8                      # ~240 ms of voiced audio to trigger
BARGE_IN_GRACE     = 0.4                    # seconds before monitoring starts
BARGE_IN_SAVE      = PRE_ROLL_FRAMES + BARGE_IN_FRAMES + 4

# ── Audio output ─────────────────────────────────────────────────────────────
OUTPUT_CARD        = 0
OUTPUT_SAMPLE_RATE = 48_000
OUTPUT_PERIOD      = 1024
OUTPUT_GAIN        = 1.0   # 1.0 = no clipping; raise only if the amp is too quiet

# ── TTS prosody ──────────────────────────────────────────────────────────────
# length_scale < 1.0  → faster / more energetic   (0.9 = 10 % faster)
# noise_scale   > 0.667 → more expressive pitch variation
# noise_w       > 0.8   → more variation in phoneme durations
PIPER_LENGTH_SCALE = 0.9    # speed up slightly for a peppier feel
PIPER_NOISE_SCALE  = 0.9    # more pitch expressiveness (default 0.667)
PIPER_NOISE_W      = 0.9    # more duration variation   (default 0.8)

# ── Shared mic queue ─────────────────────────────────────────────────────────
_audio_q: queue.Queue = queue.Queue(maxsize=300)  # ~9 s of 30-ms frames


def _is_speech(frame: np.ndarray, threshold: float = ENERGY_THRESHOLD) -> bool:
    return float(np.sqrt(np.mean(frame ** 2))) > threshold


def _load_piper_rate() -> int:
    try:
        with open(PIPER_VOICE + ".json") as f:
            return json.load(f)["audio"]["sample_rate"]
    except Exception:
        return 22_050


PIPER_RATE = _load_piper_rate()


# ─────────────────────────────────────────────────────────────────────────────
# 0. Background mic reader  (always running)
# ─────────────────────────────────────────────────────────────────────────────

def audio_reader(stop: threading.Event) -> None:
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS,
        dtype="float32", blocksize=FRAME_SAMPLES,
    ) as stream:
        while not stop.is_set():
            data, _ = stream.read(FRAME_SAMPLES)
            frame = data.flatten()
            if _audio_q.full():
                _audio_q.get_nowait()
            _audio_q.put_nowait(frame)


# ─────────────────────────────────────────────────────────────────────────────
# 1. VAD / utterance capture
# ─────────────────────────────────────────────────────────────────────────────

def capture_utterance(flush: bool = True) -> np.ndarray:
    """
    Block until a complete utterance is ready.
    flush=False: keep queued frames — used right after the bot's last sentence
    starts so any speech during that window is not lost.
    """
    if flush:
        while not _audio_q.empty():
            try:
                _audio_q.get_nowait()
            except queue.Empty:
                break

    pre_roll: collections.deque = collections.deque(maxlen=PRE_ROLL_FRAMES)
    speech_frames: list[np.ndarray] = []
    silence_count = 0
    speaking = False

    while True:
        try:
            frame = _audio_q.get(timeout=1.0)
        except queue.Empty:
            continue

        if _is_speech(frame):
            if not speaking:
                speaking = True
                speech_frames.extend(pre_roll)
            speech_frames.append(frame)
            silence_count = 0
        else:
            if speaking:
                speech_frames.append(frame)
                silence_count += 1
                if silence_count >= SILENCE_FRAMES:
                    if len(speech_frames) >= MIN_SPEECH_FRAMES:
                        return np.concatenate(speech_frames)
                    speaking = False
                    speech_frames = []
                    silence_count = 0
            else:
                pre_roll.append(frame)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Barge-in monitor
# ─────────────────────────────────────────────────────────────────────────────

def barge_in_monitor(stop: threading.Event, interrupted: threading.Event) -> None:
    """
    Runs while the bot is speaking earlier sentences.
    Consumes (discards) mic frames; fires `interrupted` on sustained user speech.
    Replays the last N frames back so capture_utterance doesn't lose the onset.
    """
    time.sleep(BARGE_IN_GRACE)
    recent: collections.deque = collections.deque(maxlen=BARGE_IN_SAVE)
    voiced = 0

    while not stop.is_set():
        try:
            frame = _audio_q.get(timeout=0.05)
        except queue.Empty:
            continue

        recent.append(frame)

        if _is_speech(frame, threshold=BARGE_IN_THRESHOLD):
            voiced += 1
            if voiced >= BARGE_IN_FRAMES:
                interrupted.set()
                for f in recent:
                    try:
                        _audio_q.put_nowait(f)
                    except queue.Full:
                        break
                return
        else:
            voiced = 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. STT
# ─────────────────────────────────────────────────────────────────────────────

def transcribe(audio: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())
        result = subprocess.run(
            [WHISPER_BIN, "-m", WHISPER_MODEL, "-f", wav_path,
             "-nt", "-l", "de", "--no-prints", "-t", "4"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30,
        )
        text = result.stdout.decode(errors="ignore").strip()
        return re.sub(r"\[.*?\]", "", text).strip()
    except subprocess.TimeoutExpired:
        print("[STT] timeout")
        return ""
    except Exception as e:
        print(f"[STT] {e}")
        return ""
    finally:
        os.unlink(wav_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. LLM
# ─────────────────────────────────────────────────────────────────────────────

def stream_llm(history: list[dict], token_queue: queue.Queue) -> str:
    messages = [SYSTEM_MSG] + history
    full: list[str] = []
    try:
        with httpx.Client(timeout=120) as client:
            with client.stream(
                "POST", OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "messages": messages, "stream": True},
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data  = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        print(token, end="", flush=True)
                        token_queue.put(token)
                        full.append(token)
                    if data.get("done"):
                        break
    except httpx.ConnectError:
        print("\n[LLM] cannot reach ollama — run: ollama serve")
    except httpx.HTTPStatusError as e:
        print(f"\n[LLM] HTTP {e.response.status_code}")
    except Exception as e:
        print(f"\n[LLM] {type(e).__name__}: {e}")
    finally:
        token_queue.put(None)
    return "".join(full)


# ─────────────────────────────────────────────────────────────────────────────
# 5. TTS — pipelined synthesis + playback
# ─────────────────────────────────────────────────────────────────────────────

def _synthesize(text: str) -> tuple[bytes, float] | None:
    """Run piper → (stereo_pcm_bytes, duration_sec), or None on failure."""
    text = text.strip()
    if not text:
        return None
    try:
        r = subprocess.run(
            [
                PIPER_BIN, "--model", PIPER_VOICE, "--output-raw",
                "--speaker",      str(PIPER_SPEAKER),
                "--length_scale", str(PIPER_LENGTH_SCALE),
                "--noise_scale",  str(PIPER_NOISE_SCALE),
                "--noise_w",      str(PIPER_NOISE_W),
            ],
            input=text.encode(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        if r.returncode != 0 or not r.stdout:
            return None

        audio = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32)
        if PIPER_RATE != OUTPUT_SAMPLE_RATE:
            # Anti-aliased rational resampling — much better than np.interp
            g   = gcd(OUTPUT_SAMPLE_RATE, PIPER_RATE)
            up  = OUTPUT_SAMPLE_RATE // g
            down = PIPER_RATE // g
            audio = resample_poly(audio, up, down).astype(np.float32)

        pcm    = np.clip(audio * OUTPUT_GAIN, -32768, 32767).astype(np.int16)
        stereo = np.empty(len(pcm) * 2, dtype=np.int16)
        stereo[0::2] = pcm
        stereo[1::2] = pcm
        return stereo.tobytes(), len(pcm) / OUTPUT_SAMPLE_RATE

    except Exception as e:
        print(f"\n[TTS synth] {e}")
        return None


def _play(raw: bytes, duration: float, interrupted: threading.Event) -> None:
    """Write pre-synthesised stereo PCM to ALSA and drain."""
    try:
        pcm_out = alsaaudio.PCM(
            alsaaudio.PCM_PLAYBACK, alsaaudio.PCM_NORMAL,
            channels=2, rate=OUTPUT_SAMPLE_RATE,
            format=alsaaudio.PCM_FORMAT_S16_LE,
            periodsize=OUTPUT_PERIOD, cardindex=OUTPUT_CARD,
        )
        chunk = OUTPUT_PERIOD * 2 * 2
        for i in range(0, len(raw), chunk):
            if interrupted.is_set():
                return
            pcm_out.write(raw[i : i + chunk])

        elapsed, step = 0.0, 0.05
        while elapsed < duration and not interrupted.is_set():
            time.sleep(step)
            elapsed += step
    except Exception as e:
        print(f"\n[TTS play] {e}")


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def tts_pipeline(
    token_queue:     queue.Queue,
    interrupted:     threading.Event,
    on_last_segment: threading.Event,
) -> None:
    """
    Synthesiser thread:  tokens → sentences → piper → audio_q
    Player thread:       audio_q → ALSA

    on_last_segment is set the moment the player STARTS the final audio chunk.

    The trick for knowing which chunk is last: the synthesiser increments
    `chunks_queued` before each put; the player increments `chunks_played`
    after each get. When synth_done is set AND chunks_played == chunks_queued,
    the player is on the last chunk. No sentinel confusion.
    """
    audio_q      = queue.Queue(maxsize=4)
    synth_done   = threading.Event()
    chunks_queued  = [0]   # written only by synthesiser
    chunks_played  = [0]   # written only by player

    # ── Synthesiser ──────────────────────────────────────────────────────────
    def synthesiser() -> None:
        buffer = ""
        while True:
            token = token_queue.get()
            done  = (token is None) or interrupted.is_set()

            if not done:
                buffer += token

            while True:
                if done:
                    if buffer.strip() and not interrupted.is_set():
                        chunk = _synthesize(buffer)
                        if chunk:
                            chunks_queued[0] += 1
                            audio_q.put(chunk)
                    break
                m = _SENTENCE_END.search(buffer)
                if not m:
                    break
                sentence, buffer = buffer[: m.start() + 1], buffer[m.end():]
                if not interrupted.is_set():
                    chunk = _synthesize(sentence)
                    if chunk:
                        chunks_queued[0] += 1
                        audio_q.put(chunk)

            if done:
                break

        audio_q.put(None)   # sentinel — no more audio
        synth_done.set()

    # ── Player ───────────────────────────────────────────────────────────────
    def player() -> None:
        while True:
            item = audio_q.get()
            if item is None:
                # Edge case: LLM produced no speakable text
                on_last_segment.set()
                break
            if interrupted.is_set():
                break

            raw, duration = item
            chunks_played[0] += 1

            # Is this the last chunk? Synth is done and we've taken all chunks.
            if synth_done.is_set() and chunks_played[0] == chunks_queued[0]:
                on_last_segment.set()

            _play(raw, duration, interrupted)

            # Synth may have finished while we were playing — check again.
            if not on_last_segment.is_set() \
                    and synth_done.is_set() \
                    and chunks_played[0] == chunks_queued[0]:
                on_last_segment.set()

    synth_t = threading.Thread(target=synthesiser, daemon=True)
    play_t  = threading.Thread(target=player,      daemon=True)
    synth_t.start()
    play_t.start()
    synth_t.join()
    play_t.join()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Startup checks
# ─────────────────────────────────────────────────────────────────────────────

def _check_setup() -> None:
    missing = []
    for label, path in [
        ("whisper-cli",   WHISPER_BIN),
        ("whisper model", WHISPER_MODEL),
        ("piper",         PIPER_BIN),
        ("piper voice",   PIPER_VOICE),
    ]:
        if not os.path.exists(path):
            missing.append(f"  ✗ {label}: {path}")
    if missing:
        print("Setup incomplete — run ./setup.sh first.\nMissing:\n" + "\n".join(missing))
        sys.exit(1)
    try:
        resp   = httpx.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(OLLAMA_MODEL in m for m in models):
            print(f"⚠  Model '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}")
            sys.exit(1)
    except Exception:
        print("✗  Cannot reach ollama. Run: ollama serve")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _check_setup()
    print("🤖 Robinson ready.  Speak any time — interrupt me mid-sentence.  Ctrl-C to quit.\n")

    history:     list[dict]      = []
    stop_reader: threading.Event = threading.Event()
    flush_next:  bool            = True

    reader_t = threading.Thread(target=audio_reader, args=(stop_reader,), daemon=True)
    reader_t.start()

    try:
        while True:
            # ── LISTEN ───────────────────────────────────────────────────────
            # flush=False after an interrupt or after the bot's last sentence
            # started: any speech the user produced during that window is
            # already in _audio_q and should not be thrown away.
            print("🎙  listening …", end="\r", flush=True)
            audio      = capture_utterance(flush=flush_next)
            flush_next = True

            # ── TRANSCRIBE ───────────────────────────────────────────────────
            print("⏳ transcribing …", end="\r", flush=True)
            text = transcribe(audio)
            if not text:
                continue

            print(f"\nYou : {text}")
            history.append({"role": "user", "content": text})

            # ── SPEAK ────────────────────────────────────────────────────────
            interrupted      = threading.Event()
            on_last_segment  = threading.Event()
            barge_stop       = threading.Event()

            token_q = queue.Queue()

            tts_t = threading.Thread(
                target=tts_pipeline,
                args=(token_q, interrupted, on_last_segment),
                daemon=True,
            )
            barge_t = threading.Thread(
                target=barge_in_monitor,
                args=(barge_stop, interrupted),
                daemon=True,
            )

            tts_t.start()
            barge_t.start()

            print("Bot : ", end="", flush=True)
            full_response = stream_llm(history, token_q)
            print()

            # ── HAND-OFF ─────────────────────────────────────────────────────
            # Wait until the player starts the last audio chunk.
            # Then stop the barge-in monitor: it was discarding mic frames.
            # From this point _audio_q accumulates freely — any speech the user
            # produces during the bot's last sentence will be there waiting.
            on_last_segment.wait(timeout=15)
            barge_stop.set()   # barge_in_monitor exits within one 50 ms timeout
            barge_t.join()

            tts_t.join()       # last sentence finishes playing

            if interrupted.is_set():
                print("      [interrupted]")
                # Barge-in replayed frames back into _audio_q → don't flush
                flush_next = False
            else:
                # Don't flush: speech during the last sentence is in the queue
                flush_next = False

            # ── HISTORY ──────────────────────────────────────────────────────
            if full_response:
                history.append({"role": "assistant", "content": full_response})
            if len(history) > 20:
                history = history[-20:]

    except KeyboardInterrupt:
        print("\nBye!")
    finally:
        stop_reader.set()


if __name__ == "__main__":
    main()
