"""Robinson main loop.

    mic ─► AEC ─► Silero VAD ─► utterance ─► STT ─► Gemma (stream) ─► sentences ─► piper ─► speaker
                     │                                   │ tags
                     └── speech while replying = barge-in └──► eyes (mood / actions)

Threads: audio callback + audio worker (audio.py), eyes (eyes.py), the
listener (this module, runs VAD and barge-in) and the responder (STT → LLM →
TTS for one utterance at a time).
"""

import argparse
import logging
import os
import queue
import random
import re
import signal
import sys
import threading
import time

import numpy as np

from .config import Config
from .eyes import ACTIONS, MOODS, Eyes, make_backend
from .memory import forget_request, personal_fact

log = logging.getLogger("robinson")

# sentence end; the first chunk may also end at a comma to start talking sooner
_SENT_END = re.compile(r"(?<=[.!?…])\s+")

# eye commands recognised directly in what the user said — reliable and
# instant, independent of whether the model remembers to emit an action tag
_USER_ACTIONS = [
    (re.compile(r"\bnach links\b|\blinks (schauen|gucken|sehen)"), "links"),
    (re.compile(r"\bnach rechts\b|\brechts (schauen|gucken|sehen)"), "rechts"),
    (re.compile(r"\bnach oben\b"), "oben"),
    (re.compile(r"\bnach unten\b"), "unten"),
    (re.compile(r"\b(geradeaus|mich an)\b"), "geradeaus"),
    (re.compile(r"\bzwinker"), "zwinkern"),
    (re.compile(r"\baugen (zu|schließen)|\bschließ\w* (die|deine) augen"), "augen zu"),
    (re.compile(r"\baugen (auf|öffnen)|\bmach\w* (die|deine) augen auf|\bwach auf"), "augen auf"),
]
_FIRST_CUT = re.compile(r"(?<=[,;:])\s+")


class Robinson:
    def __init__(self, cfg: Config, use_audio: bool = True):
        self.cfg = cfg
        from .eyecal import EyeCalibration
        self.eye_cal = EyeCalibration(cfg.eyes_file)
        self.eyes = Eyes(make_backend(cfg.eyes), self.eye_cal)

        from .memory import Memory
        from .tts import Tts
        self.memory = Memory(cfg.memory_file, on_forget=lambda: self.llm.start_fresh())
        self.llm = self._make_llm()
        self.audio = None
        if use_audio:
            from .audio import AudioIO
            from .stt import make_stt
            self.stt = make_stt(cfg)
            self.audio = AudioIO(cfg.audio_device, cfg.audio_rate, cfg.aec,
                                 cfg.aec_delay_ms, cfg.output_gain)
            self.vad = self._make_vad()
        self.tts = Tts(cfg.path(cfg.tts_voice), self.audio.rate if self.audio else 48_000,
                       cfg.tts_length_scale, cfg.tts_noise_scale, cfg.tts_noise_w_scale,
                       cfg.tts_robot, cfg.tts_pitch, cfg.tts_tone)

        self.fillers = {}
        if cfg.fillers and self.audio:
            self.fillers = {k: [(p, self.tts.synth(p)) for p in v] for k, v in FILLERS.items()}
        self._last_filler = None
        self.utterances: queue.Queue = queue.Queue()
        self.busy = threading.Event()        # a reply is being produced/played
        self.cancel = threading.Event()      # barge-in: abandon current reply
        self.started_speaking = threading.Event()  # first audio of the reply queued
        self.merge_next = False              # user kept talking before we answered
        self._prev_seg = np.zeros(0, np.float32)
        self.stop = threading.Event()

    def _make_llm(self):
        """Selected backend; online falls back to local if it can't start."""
        cfg = self.cfg
        prompt = lambda: cfg.system_prompt + self.memory.prompt_section()  # noqa: E731
        if cfg.llm == "online":
            try:
                from .llm_online import ClaudeCodeLlm
                llm = ClaudeCodeLlm(prompt, cfg.online_model, cfg.memory_file,
                                    on_tool=self._on_tool, eyes_file=cfg.eyes_file,
                                    thinking=cfg.online_thinking)
                # first turn proves auth + network; doubles as connection warm-up
                if not "".join(v for k, v in llm.chat("Antworte nur mit: ok") if k == "text"):
                    raise RuntimeError("empty reply")
                llm.reset()
                return llm
            except Exception as e:
                log.error("online LLM unavailable (%s) — falling back to local", e)
        from .llm import LocalLlm
        from .tools import make_tools
        return LocalLlm(cfg.path(cfg.llm_model), prompt, cfg.llm_threads,
                        cfg.llm_max_tokens, cfg.llm_cache_dir,
                        tools=make_tools(self.memory, self.eye_cal) if cfg.tools else None,
                        on_tool=self._on_tool)

    def _on_tool(self, name: str, _args: dict) -> None:
        """After any tool call (in-process or in the MCP server): pick up file changes."""
        from .eyecal import EYE_TOOL_NAMES
        if name in ("merken", "vergessen"):
            self.memory.reload()
        if name == "vergessen":
            self.llm.start_fresh()
        if name in EYE_TOOL_NAMES:
            self.eyes.calibration_changed()   # reloads eyes.json and shows the change

    def _make_vad(self):
        import sherpa_onnx
        c = self.cfg
        vc = sherpa_onnx.VadModelConfig()
        vc.silero_vad.model = c.path(c.vad_model)
        vc.silero_vad.threshold = c.vad_threshold
        vc.silero_vad.min_silence_duration = c.vad_min_silence_s
        vc.silero_vad.min_speech_duration = c.vad_min_speech_s
        vc.silero_vad.max_speech_duration = c.vad_max_speech_s
        vc.silero_vad.window_size = 512
        vc.sample_rate = 16_000
        vc.num_threads = 1
        return sherpa_onnx.VoiceActivityDetector(vc, buffer_size_in_seconds=60)

    # ── listener: VAD + barge-in ─────────────────────────────────────────────
    def listen(self) -> None:
        speech_since = None
        meter_t, meter_peak, meter_speech = time.monotonic(), 0.0, 0
        while not self.stop.is_set():
            try:
                frame = self.audio.mic.get(timeout=0.5)
            except queue.Empty:
                continue
            self.vad.accept_waveform(frame)
            speaking = self.vad.is_speech_detected()

            if log.isEnabledFor(logging.DEBUG):
                meter_peak = max(meter_peak, float(np.sqrt(np.mean(frame ** 2))))
                meter_speech += speaking
                if time.monotonic() - meter_t >= 2.0:
                    log.debug("mic rms max %.4f (raw %.4f), speech frames %d",
                              meter_peak, self.audio.raw_peak(), meter_speech)
                    meter_t, meter_peak, meter_speech = time.monotonic(), 0.0, 0

            if not self.busy.is_set():
                self.eyes.set_state("listening" if speaking else "idle")
            elif speaking and self.cfg.barge_in and self._loud_enough(frame):
                speech_since = speech_since or time.monotonic()
                if time.monotonic() - speech_since >= self.cfg.barge_in_s \
                        and not self.cancel.is_set():
                    if self.started_speaking.is_set():
                        log.info("barge-in")
                    else:
                        # only a pause mid-sentence: answer both parts together
                        log.info("user continues — merging with previous utterance")
                        self.merge_next = True
                    self.cancel.set()
                    self.llm.cancel()
                    self.audio.flush()
                    self.eyes.set_state("listening")
            else:
                speech_since = None

            while not self.vad.empty():
                seg = np.asarray(self.vad.front.samples, np.float32)
                self.vad.pop()
                # without barge-in, speech heard while talking is our own echo
                if self.busy.is_set() and not self.cancel.is_set():
                    log.debug("dropping %.1fs segment heard while speaking", len(seg) / 16e3)
                    continue
                self.utterances.put(seg)

    def _loud_enough(self, frame: np.ndarray) -> bool:
        if self.audio.aec is not None:
            return True
        # no AEC: only sustained speech clearly louder than the speaker bleed
        return float(np.sqrt(np.mean(frame ** 2))) > self.cfg.barge_in_rms

    # ── responder: STT → LLM → TTS ───────────────────────────────────────────
    def respond_loop(self) -> None:
        from .stt import is_hallucination
        while not self.stop.is_set():
            try:
                seg = self.utterances.get(timeout=0.5)
            except queue.Empty:
                continue
            if self.merge_next:
                seg = np.concatenate([self._prev_seg, np.zeros(1600, np.float32), seg])
                self.merge_next = False
            self._prev_seg = seg
            self.busy.set()
            self.cancel.clear()
            self.started_speaking.clear()
            self.eyes.set_state("thinking")
            t0 = time.monotonic()
            text = self.stt(seg)
            log.info("heard (%.1fs audio, %.2fs STT): %s", len(seg) / 16e3,
                     time.monotonic() - t0, text)
            if is_hallucination(text):
                self.busy.clear()
                self.eyes.set_state("idle")
                continue
            self._play_filler(text)
            self._update_memory(text)
            self.reply(text, t0)
            self.busy.clear()
            if not self.cancel.is_set():
                self.eyes.set_state("idle")

    def _update_memory(self, text: str) -> None:
        """Deterministic remember/forget from the user's words (E2B often only
        *claims* to have called merken/vergessen)."""
        if self.llm.name != "local":
            return   # Claude calls merken/vergessen reliably; only Gemma needs this net
        if what := forget_request(text):
            self.memory.forget(what)
        elif fact := personal_fact(text):
            self.memory.add(fact)

    def _play_filler(self, text: str) -> None:
        kind = filler_kind(text)
        if not kind or not self.fillers:
            return
        choices = [f for f in self.fillers[kind] if f[0] != self._last_filler] or self.fillers[kind]
        phrase, audio = random.choice(choices)
        self._last_filler = phrase
        log.debug("filler: %s", phrase)
        self.audio.play(audio)

    def reply(self, text: str, t0: float | None = None) -> str:
        t0 = t0 or time.monotonic()
        mood, buf, spoken, first = "neutral", "", [], True
        tagged_mood = False   # an explicit tag/emoji from the model wins over wording
        heard = text.lower()
        user_actions = {a for rx, a in _USER_ACTIONS if rx.search(heard)}
        for a in user_actions:
            self.eyes.action(a)

        def speak(sentence: str) -> None:
            nonlocal first, mood
            if not tagged_mood:
                mood = mood_of(sentence)
                self.eyes.set_mood(mood)
            ts = time.monotonic()
            audio = self.tts.synth(sentence, mood)
            log.debug("tts %.2fs: %s", time.monotonic() - ts, sentence.strip()[:40])
            if audio is None or self.cancel.is_set():
                return
            if first:
                log.info("first audio after %.2fs", time.monotonic() - t0)
                self.started_speaking.set()
                self.eyes.set_state("speaking")
                first = False
            spoken.append(sentence.strip())
            if self.audio:
                self.audio.play(audio)

        stream = self.llm.chat(text)
        t_tok = None
        for kind, val in stream:
            if t_tok is None:
                t_tok = time.monotonic()
                log.debug("first LLM token after %.2fs", t_tok - t0)
            if self.cancel.is_set():
                for _ in stream:  # let the cancelled generation wind down cleanly
                    pass
                break
            if kind == "tag":
                log.debug("tag [%s]", val)
                if val in MOODS:
                    mood, tagged_mood = val, True
                    self.eyes.set_mood(val)
                elif val in ACTIONS and val not in user_actions:
                    self.eyes.action(val)
                continue
            buf += val
            cut = _SENT_END if not first or len(buf.split()) < 5 else _ANY_CUT
            while (m := cut.search(buf)):
                speak(buf[:m.start()])
                buf = buf[m.end():]
                cut = _SENT_END
        if buf.strip() and not self.cancel.is_set():
            speak(buf)
        if self.audio:
            while self.audio.playing and not self.cancel.is_set():
                time.sleep(0.02)
        reply = " ".join(spoken)
        log.info("said [%s]: %s", mood, reply)
        return reply

    # ── lifecycle ────────────────────────────────────────────────────────────
    def run(self) -> None:
        self.eyes.start()
        self.audio.start()
        if self.cfg.boot_greeting:
            # also prefills the system prompt, so the first real answer is fast
            self.busy.set()
            self.reply("(Du wurdest gerade eingeschaltet. Begrüße kurz in einem Satz, "
                       "ohne Werkzeuge.)")
            self.busy.clear()
        threading.Thread(target=self.respond_loop, name="responder", daemon=True).start()
        log.info("Robinson ready — sprich mit mir.")
        self.listen()

    def run_text(self) -> None:
        """Typed chat (debugging over ssh): no mic/STT, TTS only if audio is on."""
        self.eyes.start()
        if self.audio:
            self.audio.start()
        for line in sys.stdin:
            self._update_memory(line)
            if line.strip():
                self.eyes.set_state("thinking")
                print("Robinson:", self.reply(line.strip()), flush=True)
                self.eyes.set_state("idle")

    def close(self) -> None:
        self.stop.set()
        if hasattr(self.llm, "close"):
            self.llm.close()
        self.eyes.stop()
        if self.audio:
            self.audio.close()


# short sounds played right after STT while Gemma prefills (~1 s); never added
# to the conversation, and they don't count as "speaking" for barge-in
FILLERS = {
    "question": ["Hmm.", "Hm, mal sehen.", "Moment.", "Hmm, lass mich überlegen.", "Also."],
    "statement": ["Mhm.", "Aha.", "Okay.", "Ah.", "Verstehe."],
}
_GREETING = re.compile(r"^(hallo|hi|hey|servus|moin|guten (morgen|tag|abend)|tschüss|gute nacht)\b")
_SHORT_REPLY = re.compile(r"^(danke|ja|nein|okay|ok|stopp?|genau|super)\b")
_QUESTION = re.compile(r"\?\s*$|^(wer|wie|was|wo|wann|warum|wieso|weshalb|welche\w*|"
                       r"kannst|könntest|hast|bist|weißt|erzähl|erklär|sag)\b")


# mood per spoken sentence, from its wording (picks eye shape + piper speaker)
_MOOD_RULES = [
    ("traurig",      re.compile(r"\b(leider|schade|traurig|tut mir leid|bedauer)")),
    ("überrascht",   re.compile(r"\b(oh|wow|wirklich\?|echt\?|ach was|tatsächlich)\b")),
    ("müde",         re.compile(r"\b(müde|gähn|schlaf|gute nacht)")),
    ("flüsternd",    re.compile(r"\b(psst|pst|geheimnis|flüster)")),
    ("verärgert",    re.compile(r"\b(ärger|frechheit|unverschämt|grr)")),
    ("nachdenklich", re.compile(r"\b(hmm|vielleicht|überleg|ich glaube|keine ahnung)")),
    ("fröhlich",     re.compile(r"!|\b(haha|super|toll|klar|gerne|prima|juhu|cool|schön)")),
]


def mood_of(sentence: str) -> str:
    t = sentence.lower()
    return next((m for m, rx in _MOOD_RULES if rx.search(t)), "neutral")


def filler_kind(text: str) -> str | None:
    t = text.lower().strip()
    if _GREETING.search(t) or (_SHORT_REPLY.search(t) and len(t.split()) <= 3):
        return None   # the model's own "Hallo!" / "Gern!" is the right first sound
    return "question" if _QUESTION.search(t) else "statement"


_ANY_CUT = re.compile(f"{_SENT_END.pattern}|{_FIRST_CUT.pattern}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="robinson")
    ap.add_argument("--text", action="store_true", help="type instead of talk")
    ap.add_argument("--no-audio", action="store_true", help="with --text: print only")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    verbose = args.verbose or os.environ.get("ROBINSON_DEBUG", "") not in ("", "0")
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = Config.from_env()
    bot = Robinson(cfg, use_audio=not args.no_audio)
    signal.signal(signal.SIGTERM, lambda *_: (bot.close(), sys.exit(0)))
    try:
        bot.run_text() if args.text else bot.run()
    except KeyboardInterrupt:
        pass
    finally:
        bot.close()
