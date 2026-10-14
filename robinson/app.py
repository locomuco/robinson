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
import subprocess
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
    (re.compile(r"\bzwinker|(linke|rechte)n? auge\w* (\w+ )?blinzel|blinzel\w* (\w+ )*(linke|rechte)n? auge"),
     "zwinkern"),
    (re.compile(r"augen (\w+ )?(roll|verdreh)|(roll|verdreh)\w* (\w+ )?(die|deine) augen"), "augen rollen"),
    (re.compile(r"\bschlaf (\w+ ){0,2}ein\b"), "schlafen"),   # imperative only
    (re.compile(r"\bwach (\w+ )?auf\b"), "aufwachen"),
    (re.compile(r"\baugen (zu|schließen)|\bschließ\w* (die|deine) augen"), "augen zu"),
    (re.compile(r"\baugen (auf|öffnen)|\bmach\w* (die|deine) augen auf"), "augen auf"),
]
_FIRST_CUT = re.compile(r"(?<=[,;:])\s+")


class Robinson:
    def __init__(self, cfg: Config, use_audio: bool = True):
        self.cfg = cfg
        from .settings import Settings, apply_mixer
        self.settings = Settings(cfg.settings_file)
        self.settings.apply_to(cfg)          # values changed by voice win over defaults
        if use_audio:
            apply_mixer(self.settings)
        self._vad_rebuild = False
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
            self.speakers = None
            if cfg.speaker_id and os.path.exists(cfg.speaker_model):
                try:
                    from .speakers import SpeakerId
                    self.speakers = SpeakerId(cfg.speaker_model, cfg.speaker_file,
                                              cfg.speaker_threshold)
                except Exception as e:  # noqa: BLE001
                    log.warning("speaker id unavailable: %s", e)
            self.audio = AudioIO(cfg.audio_device, cfg.audio_rate, cfg.aec,
                                 cfg.aec_delay_ms, cfg.output_gain)
            self.vad = self._make_vad()
        self.tts = Tts(cfg.path(cfg.tts_voice), self.audio.rate if self.audio else 48_000,
                       cfg.tts_length_scale, cfg.tts_noise_scale, cfg.tts_noise_w_scale,
                       cfg.tts_robot, cfg.tts_pitch, cfg.tts_tone,
                       en_voice_path=cfg.path(cfg.tts_voice_en))

        self.fillers = {}
        if self.audio:
            self._render_fillers()
        self._last_filler = None
        from .maintenance import Maintenance
        self.maint = Maintenance(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 os.path.dirname(cfg.state_file), self._maint_done,
                                 model=cfg.maint_model)
        self._pending: dict | None = None    # action waiting for "Ja" (+ optional PIN)
        self._announcements: queue.Queue = queue.Queue()
        self.text_mode = False
        self.speaker: str | None = None      # who spoke last (by voice), if known
        self.lang = "de"                     # language of the last human utterance
        self._turn_user_actions: set[str] = set()
        self._turn_tools: list[str] = []
        self.utterances: queue.Queue = queue.Queue()
        self.busy = threading.Event()        # a reply is being produced/played
        self.cancel = threading.Event()      # barge-in: abandon current reply
        self.started_speaking = threading.Event()  # first audio of the reply queued
        self.merge_next = False              # user kept talking before we answered
        self._prev_seg = np.zeros(0, np.float32)
        self.stop = threading.Event()

    def _make_llm(self):
        """Router over the local model and (if a token is set up) Claude online."""
        from .llm import LocalLlm
        from .llm_router import LlmRouter
        from .tools import make_tools
        cfg = self.cfg
        prompt = lambda: cfg.system_prompt + self.memory.prompt_section()  # noqa: E731

        def make_local(key: str):
            path = cfg.path(cfg.local_model_files.get(key, cfg.local_model_files["e2b"]))
            return LocalLlm(path, prompt, cfg.llm_threads, cfg.llm_max_tokens, cfg.llm_cache_dir,
                            tools=make_tools(self.memory, self.eye_cal, self._status,
                                             self.settings, cfg.state_file,
                                             cfg.maintenance, bool(cfg.maint_pin))
                            if cfg.tools else None,
                            on_tool=self._on_tool, model_key=key)

        def make_online(model: str):
            from .llm_online import ClaudeCodeLlm
            return ClaudeCodeLlm(prompt, model, cfg.memory_file,
                                 on_tool=self._on_tool, eyes_file=cfg.eyes_file,
                                 state_file=cfg.state_file, settings_file=cfg.settings_file,
                                 thinking=cfg.online_thinking)

        online_ready = bool(os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")) or \
            os.path.exists(os.path.expanduser("~/.claude/.credentials.json"))
        if cfg.llm == "online" and not online_ready:
            log.warning("ROBINSON_LLM=online but no Claude Code login found — using local")
        return LlmRouter(make_local, make_online if online_ready else None, cfg.llm,
                         cfg.state_file, cfg.local_model, cfg.online_model)

    def _status(self) -> str:
        return self.llm.status if hasattr(self, "llm") else ""

    def _render_fillers(self) -> None:
        self.fillers = {k: [(p, self.tts.synth(p)) for p in v] for k, v in FILLERS.items()}

    def _apply_settings(self) -> None:
        """Make settings.json (just changed by a tool) take effect immediately."""
        from .settings import apply_mixer
        s, cfg = self.settings, self.cfg
        s.reload()
        old_silence = cfg.vad_min_silence_s
        for k in ("fillers", "barge_in", "boot_greeting", "vad_min_silence_s", "tts_pitch",
                  "tts_tone", "tts_robot", "tts_length_scale"):
            setattr(cfg, k, s.get(k, getattr(cfg, k)))
        self.tts.configure(robot=cfg.tts_robot, pitch=cfg.tts_pitch, tone=cfg.tts_tone,
                           length_scale=cfg.tts_length_scale)
        if self.audio:
            apply_mixer(s)
            threading.Thread(target=self._render_fillers, daemon=True).start()
            if cfg.vad_min_silence_s != old_silence:
                self._vad_rebuild = True

    def _on_tool(self, name: str, _args: dict) -> None:
        """After any tool call (in-process or in the MCP server): pick up file changes."""
        from .eyecal import EYE_TOOL_NAMES
        from .settings import SETTINGS_TOOL_NAMES
        self._turn_tools.append(name)
        if name in SETTINGS_TOOL_NAMES:
            self._apply_settings()
        if name in ("merken", "vergessen"):
            self.memory.reload()
        if name == "vergessen":
            self.llm.start_fresh()
        if name in EYE_TOOL_NAMES:
            self.eyes.calibration_changed()   # reloads eyes.json and shows the change
        if name == "neustarten":
            w = str(_args.get("was", "")).lower()
            self._request_restart("ausschalten" if "aus" in w else
                                  "pi" if ("pi" in w or "rechner" in w) else "robinson")
        if name == "wartung" and _args.get("auftrag"):
            self._request_maint(str(_args["auftrag"]))
        if name == "wartung_rueckgaengig":
            self._request_maint_undo()
        if name == "modell_wechseln":
            self.llm.set_mode(_args.get("ziel", ""))
        if name == "augen_bewegen":
            from .eyes import normalize_action
            action = normalize_action(_args.get("aktion", ""), _args.get("auge", ""))
            if action and action not in self._turn_user_actions:   # not twice per turn
                self.eyes.action(action)

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
            if self._vad_rebuild:               # pause length changed by voice
                self._vad_rebuild = False
                self.vad = self._make_vad()
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
                self._flush_announcements()
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
            self.speaker = None
            if self.audio and getattr(self, "speakers", None):
                if (who := self._enroll_voice(text, seg)) is not None:
                    self.say(who)
                    self.busy.clear()
                    self.eyes.set_state("idle")
                    continue
                try:
                    self.speaker = self.speakers.identify(seg)
                except Exception as e:  # noqa: BLE001
                    log.warning("speaker id failed: %s", e)
                if self.speaker:
                    log.info("speaker: %s", self.speaker)
            if self._handle_direct(text):
                self.busy.clear()
                self.eyes.set_state("idle")
                continue
            self._play_filler(text)
            self._update_memory(text)
            self.reply(text, t0)
            self.busy.clear()
            if not self.cancel.is_set():
                self.eyes.set_state("idle")

    def _enroll_voice(self, text: str, seg: np.ndarray) -> str | None:
        """Answer to a "remember my voice" request, or None if it is none."""
        from .speakers import enroll_request
        name = enroll_request(text)
        if name is None:
            return None
        if not name:
            return "Wie heißt du? Sag zum Beispiel: Merk dir meine Stimme, ich bin Anna."
        if self.speakers.enroll(name, seg):
            self.memory.add(f"{name} ist ein Mensch, den ich an der Stimme erkenne")
            return f"Alles klar {name}, ich erkenne dich jetzt an deiner Stimme."
        return "Das war zu kurz. Sag es bitte nochmal etwas ausführlicher."

    def say(self, text: str) -> None:
        """Speak a fixed sentence (no LLM), e.g. a mode-switch confirmation."""
        log.info("said: %s", text)
        if self.text_mode:
            print("Robinson:", text, flush=True)
        if self.audio and (audio := self.tts.synth(text)) is not None:
            self.audio.play(audio)
            while self.audio.playing:
                time.sleep(0.02)

    # ── things handled without the LLM ───────────────────────────────────────
    def _handle_direct(self, text: str) -> bool:
        """Confirmations, maintenance requests and model switches. True = handled."""
        from .control import (RESTART_TARGETS, confirmation, maintenance_request,
                              maintenance_undo_request, restart_request)
        if self._pending is not None:
            p, self._pending = self._pending, None
            answer = confirmation(text)
            if time.monotonic() > p["until"]:
                self.say("Die Frage ist abgelaufen, ich habe nichts gemacht.")
            elif answer == "yes":
                if p["pin"] and not self._pin_ok(text):
                    self.say("Die PIN stimmt nicht. Ich mache nichts.")
                else:
                    log.info("confirmed: %s", p["desc"])
                    self.say(p["ok"])
                    p["action"]()
                return True
            elif answer == "no":
                self.say("Okay, abgebrochen.")
                return True
            else:
                self.say("Gut, dann lasse ich das.")   # moved on → answer normally below
        if what := restart_request(text):
            self._request_restart(what)
            self.say(f"Soll ich wirklich {RESTART_TARGETS[what]}? Sag Ja.")
            return True
        if maintenance_undo_request(text):
            self._request_maint_undo()
            self.say("Soll ich den letzten Wartungsauftrag zurücknehmen? "
                     f"Sag {self._confirm_words()}.")
            return True
        if task := maintenance_request(text):
            if self._request_maint(task):
                self.say(f"Wartungsauftrag: {task} Sag {self._confirm_words()}, "
                         "dann lege ich los.")
            else:
                self.say("Der Wartungsmodus ist ausgeschaltet oder Claude Code fehlt.")
            return True
        if target := switch_request(text):
            self.say(self.llm.set_mode(target))
            return True
        return False

    def _pin_ok(self, text: str) -> bool:
        from .control import spoken_digits
        return self.cfg.maint_pin in spoken_digits(text)

    def _confirm_words(self) -> str:
        return "Ja und deine PIN" if self.cfg.maint_pin else "Ja"

    def _set_pending(self, desc: str, action, ok: str, pin: bool = False) -> None:
        self._pending = {"desc": desc, "action": action, "ok": ok, "pin": pin,
                         "until": time.monotonic() + 90}
        log.info("waiting for confirmation: %s", desc)

    def _request_restart(self, what: str) -> None:
        from .control import RESTART_TARGETS
        self._set_pending(RESTART_TARGETS[what], lambda: self.restart(what),
                          {"robinson": "Bis gleich!", "pi": "Ich starte den Pi neu. Bis gleich!",
                           "ausschalten": "Gute Nacht!"}[what])

    def _request_maint(self, task: str) -> bool:
        if not self.cfg.maintenance or not self.maint.available:
            return False
        self._set_pending(f"Wartung: {task}", lambda: self.maint.start(task),
                          "Alles klar, ich gebe den Auftrag an Claude Code. Das dauert ein paar "
                          "Minuten, ich sage Bescheid, wenn ich fertig bin.",
                          pin=bool(self.cfg.maint_pin))
        return True

    def _request_maint_undo(self) -> None:
        self._set_pending("Wartung rückgängig", self.maint.rollback,
                          "Okay, ich nehme den letzten Wartungsauftrag zurück.",
                          pin=bool(self.cfg.maint_pin))

    def _maint_done(self, summary: str, code_changed: bool) -> None:
        self._announcements.put((summary, code_changed))

    def _flush_announcements(self) -> None:
        while not self.busy.is_set():
            try:
                text, restart = self._announcements.get_nowait()
            except queue.Empty:
                return
            self.say(text)
            if restart:
                self.say("Ich starte kurz neu, um die Änderung zu laden.")
                self.restart("robinson")

    def restart(self, what: str) -> None:
        cmd = {"robinson": ["systemctl", "restart", "robinson"], "pi": ["reboot"],
               "ausschalten": ["poweroff"]}[what]
        if not os.environ.get("INVOCATION_ID"):     # not under systemd (e.g. --text tests)
            log.info("would run: sudo %s", " ".join(cmd))
            return
        subprocess.Popen(["sudo", "-n", *cmd])

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
        if not kind or not self.fillers or not self.cfg.fillers:
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
        if "zwinkern" in user_actions and (side := re.search(r"\b(linke|rechte)n? auge", heard)):
            user_actions = (user_actions - {"zwinkern"}) | {f"zwinkern {side.group(1)[:-1]}s"}
        self._turn_user_actions = user_actions
        self._turn_tools = []
        for a in user_actions:
            self.eyes.action(a)
        from .tts import detect_language
        self.lang = lang = detect_language(text, self.lang)

        def speak(sentence: str) -> None:
            nonlocal first, mood
            if not tagged_mood:
                mood = mood_of(sentence)
                self.eyes.set_mood(mood)
            ts = time.monotonic()
            audio = self.tts.synth(sentence, mood, lang)
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

        # the reply language is decided here, not left to the model
        hint = "(Please answer in English.) " if lang == "en" else "(Antworte auf Deutsch.) "
        if self.speaker:
            hint += f"(Es spricht gerade {self.speaker}.) "
        stream = self.llm.chat(hint + text)
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
        if not spoken and self._turn_tools and not self.cancel.is_set():
            # Gemma sometimes ends a tool turn without any words — don't stay mute
            speak(random.choice(["Bitte sehr!", "So!", "Voilà!"])
                  if self._turn_tools[-1] == "augen_bewegen" else "Erledigt.")
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
        self.text_mode = True
        for line in sys.stdin:
            self._flush_announcements()
            if not line.strip() or self._handle_direct(line.strip()):
                continue
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
    "question": ["Hmm.", "Hm.", "Hmm, mal sehen."],
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


_SWITCH = re.compile(r"\b(geh|gehe|wechsel\w*|schalt\w*|nimm|nutz\w*|benutz\w*|zurück zu|bleib\w*)\b"
                     r"(?P<rest>.{0,40})$")


def switch_request(text: str) -> str | None:
    """'Geh online' → 'online', 'Nimm Sonnet' → 'sonnet', 'Wechsel aufs große Gemma'
    → 'gemma groß' … (returned text is parsed by LlmRouter.set_mode)."""
    from .llm_router import parse_target
    t = text.lower().strip()
    if "?" in t:
        return None
    m = _SWITCH.search(t)
    if not m or parse_target(m.group("rest")) is None:
        return None
    rest = m.group("rest")
    if "modell" not in rest and not re.search(r"online|offline|lokal|claude|haiku|heiku|sonnet|"
                                             r"sonett|opus|gemma|e[24]b", rest):
        return None
    return rest.strip(" .!")


def filler_kind(text: str) -> str | None:
    t = text.lower().strip()
    if _GREETING.search(t) or (_SHORT_REPLY.search(t) and len(t.split()) <= 3):
        return None   # the model's own "Hallo!" / "Gern!" is the right first sound
    # only clear questions get a neutral "Hmm."; "Aha"/"Okay" on statements or commands often misfit
    return "question" if _QUESTION.search(t) else None


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
