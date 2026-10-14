"""Local ⇄ online LLM routing with automatic fallback.

Robinson talks to the router like to any Llm. It owns the canonical
conversation history; whichever backend answers a turn is first synced to
that history (so switching never loses context), and its history becomes
the canonical one afterwards.

Mode "online" uses Claude while it is healthy. A failed turn (no network,
timeout, auth, crash) is answered by the always-loaded local Gemma in the
same turn; a background check re-enables online once api.anthropic.com is
reachable again. The chosen mode is persisted, so a voice "geh online"
survives restarts.
"""

import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterator

from .llm import Llm

log = logging.getLogger("robinson.llm")

MODES = ("local", "online")

# spoken/model wording → (mode, model key). Parakeet spellings included.
LOCAL_MODELS = {"e2b": "Gemma 4 E2B", "e4b": "Gemma 4 E4B"}
ONLINE_MODELS = {"haiku": "Claude Haiku", "sonnet": "Claude Sonnet", "opus": "Claude Opus"}
_ALIASES = [
    (r"haiku|heiku|haiko", ("online", "haiku")),
    (r"sonnet|sonett|sonet", ("online", "sonnet")),
    (r"opus", ("online", "opus")),
    (r"e ?4 ?b|(groß|gross|größ|gröss)\w*\s+(\w+\s+)?(lokal|gemma)|gemma\s+(groß|gross)", ("local", "e4b")),
    (r"e ?2 ?b|klein\w*\s+(\w+\s+)?(lokal|gemma)|gemma\s+klein|\bgemma\b", ("local", "e2b")),
    (r"online|claude|cloud|internet", ("online", None)),
    (r"lokal|local|offline", ("local", None)),
]


def parse_target(text: str) -> tuple[str, str | None] | None:
    t = " ".join(text.lower().replace("-", " ").split())
    for pattern, target in _ALIASES:
        if re.search(pattern, t):
            return target
    return None


class LlmRouter:
    health_interval_s = 20.0
    min_offline_s = 30.0        # after a failure, don't retry online sooner than this

    def __init__(self, make_local: Callable[[str], Llm], make_online: Callable[[str], Llm] | None,
                 mode: str, state_file: str, local_model: str = "e2b",
                 online_model: str = "haiku"):
        self._make_local, self._make_online = make_local, make_online
        self.make_online = make_online           # None = online not set up
        self.state_file = state_file
        state = self._load_state()
        self.mode = state.get("llm", mode) if state.get("llm", mode) in MODES else "local"
        self.local_model = state.get("local_model", local_model)
        self.online_model = state.get("online_model", online_model)
        self._lock = threading.Lock()
        self._save_mode()   # the MCP tool server reads the current choice from this file
        self.history: list[dict] = []
        self.local: Llm | None = None
        self.online: Llm | None = None
        # how much of self.history each backend has seen; behind → hand over the rest
        self._seen = {"local": 0, "online": 0}
        self._must_reset: set[str] = set()      # e.g. after forgetting something
        self._online_ok = True
        self._offline_since = 0.0
        self._announce: str | None = None
        self._active: Llm | None = None
        self._local_ready = threading.Event()
        self._stop = threading.Event()

        if self.mode == "online" and make_online:
            self._start_online()
            # Gemma loads in the background so a fallback never waits for it
            threading.Thread(target=self._load_local, name="local-llm", daemon=True).start()
        else:
            self._load_local()
        threading.Thread(target=self._health_loop, name="llm-health", daemon=True).start()

    # ── setup / state ────────────────────────────────────────────────────────
    def _load_state(self) -> dict:
        try:
            with open(self.state_file) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_mode(self) -> None:
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        state = {}
        try:
            with open(self.state_file) as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
        state.update(llm=self.mode, local_model=self.local_model, online_model=self.online_model)
        with open(self.state_file, "w") as f:
            json.dump(state, f)

    def _load_local(self) -> None:
        local = self._make_local(self.local_model)
        if self.mode == "online" and hasattr(local, "warm_up"):
            local.warm_up()          # stand-by: makes a fallback turn ~5 s faster
        self.local = local
        self._local_ready.set()

    def _start_online(self) -> None:
        try:
            self.online = self._make_online(self.online_model)
            self._seen["online"] = 0
        except Exception as e:
            log.error("online LLM could not start: %s", e)
            self._mark_offline()

    def _mark_offline(self) -> None:
        if self._online_ok:
            self._announce = "Ich bin gerade offline und antworte mit meinem eigenen Kopf."
        self._online_ok = False
        self._offline_since = time.monotonic()

    def _health_loop(self) -> None:
        from .llm_online import internet_ok
        while not self._stop.wait(self.health_interval_s):
            if self.mode != "online" or self._online_ok:
                continue
            if time.monotonic() - self._offline_since < self.min_offline_s:
                continue
            if internet_ok():
                log.info("online LLM reachable again")
                if self.online is None and self.make_online:
                    self._start_online()
                if self.online is not None:
                    self._online_ok = True
                    self._announce = "Ich bin wieder online."

    # ── public API (same shape as Llm) ───────────────────────────────────────
    @property
    def name(self) -> str:
        b = self._active or self._pick()
        return b.name if b else "local"

    @property
    def status(self) -> str:
        local = LOCAL_MODELS.get(self.local_model, self.local_model)
        online = ONLINE_MODELS.get(self.online_model, self.online_model)
        if self.mode == "local":
            return f"Ich denke lokal mit {local}."
        if self._online_ok:
            return f"Ich denke online mit {online}. Lokal steht {local} bereit."
        return f"{online} ist gewählt, aber ich bin offline und antworte mit {local}."

    def set_mode(self, target: str) -> str:
        """Switch mode and/or model ("online", "lokal", "sonnet", "gemma groß" …).
        Safe to call while a turn is running: a new local engine is only loaded
        once that turn is over (see chat)."""
        parsed = parse_target(target)
        if parsed is None:
            return ("Das kenne ich nicht. Möglich sind lokal, online, Haiku, Sonnet, Opus, "
                    "kleines oder großes Gemma.")
        mode, model = parsed
        if mode == "online" and not self.make_online:
            return "Online ist nicht eingerichtet (kein Claude-Code-Login)."
        with self._lock:
            before = (self.mode, self.local_model, self.online_model)
            self.mode = mode
            if model and mode == "online":
                self.online_model = model
            elif model:
                self.local_model = model
            self._save_mode()
            after = (self.mode, self.local_model, self.online_model)
        if mode == "online":
            if self.online is None:
                self._start_online()
            elif getattr(self.online, "model", None) != self.online_model:
                self.online.model = self.online_model
                self.online.close()          # restarts with the new model on the next turn
            self._online_ok = self.online is not None
        log.info("LLM → %s (local %s, online %s)", self.mode, self.local_model, self.online_model)
        if before == after:
            return "Schon erledigt. " + self.status
        if mode == "local" and self.local is not None and \
                getattr(self.local, "model_key", self.local_model) != self.local_model:
            if self._active is None:
                self._swap_local()
            else:
                return f"Gleich: nach dieser Antwort lade ich {LOCAL_MODELS[self.local_model]}."
        return "Gewechselt. " + self.status

    def _swap_local(self) -> None:
        """Load the selected Gemma variant and drop the old one (blocking, ~10 s)."""
        old = self.local
        if old is not None and getattr(old, "model_key", None) == self.local_model:
            return
        log.info("loading local model %s", self.local_model)
        self._local_ready.clear()
        self.local = None
        if old is not None and hasattr(old, "close"):
            old.close()
        self._load_local()
        self._seen["local"] = 0

    def _pick(self) -> Llm | None:
        if self.mode == "online" and self._online_ok and self.online is not None:
            return self.online
        self._local_ready.wait()
        return self.local

    max_handover = 10           # missed messages passed as text; more → rebuild

    def _alive(self, b: Llm) -> bool:
        if b is self.online:
            return getattr(b, "proc", None) is not None and b.proc.poll() is None
        return True

    def _sync(self, b: Llm) -> str:
        """Bring `b` up to the canonical history. Usually the messages it missed are
        handed over as a short text block in front of the next user message, which
        keeps its warm context (Gemma: no multi-second re-prefill). Returns that
        prefix ("" if nothing is missing or the backend was rebuilt)."""
        missed = self.history[self._seen[b.name]:]
        if b.name in self._must_reset or not self._alive(b) or len(missed) > self.max_handover:
            self._must_reset.discard(b.name)
            b.history = [dict(m) for m in self.history[-6:]]
            b.reset(keep=len(b.history))
            return ""
        if not missed:
            return ""
        b.history = [dict(m) for m in self.history]
        lines = "\n".join(("Mensch: " if m["role"] == "user" else "Du: ") + m["content"]
                          for m in missed)
        return f"(Was inzwischen gesagt wurde:\n{lines})\n\n"

    def chat(self, text: str) -> Iterator[tuple[str, str]]:
        b = self._pick()
        if self._announce:
            yield "text", self._announce + " "
            self._announce = None
        self._active = b
        prefix = self._sync(b)
        got_text = False
        handed_over = bool(prefix)
        try:
            for ev in b.chat(prefix + text):
                got_text = got_text or (ev[0] == "text" and ev[1].strip() != "")
                yield ev
        except Exception as e:
            if b is not self.online:
                raise
            log.warning("online LLM failed (%s) — answering locally", e)
            self._mark_offline()
            if got_text:   # half an answer was already spoken; don't start over
                yield "text", " … Moment, meine Verbindung ist gerade weg."
                return
            yield "text", (self._announce or "") + " "
            self._announce = None
            b = self._pick()
            self._active = b
            prefix = self._sync(b)
            handed_over = bool(prefix)
            yield from b.chat(prefix + text)
        finally:
            if handed_over:
                # store what the human said, not the context block we prepended
                for m in reversed(b.history):
                    if m["role"] == "user":
                        m["content"] = text
                        break
            self.history = [dict(m) for m in b.history]
            self._seen[b.name] = len(self.history)
            self._active = None
            if self.mode == "local" and self.local is not None and \
                    getattr(self.local, "model_key", self.local_model) != self.local_model:
                self._swap_local()           # requested by a tool call during this turn

    def cancel(self) -> None:
        if self._active is not None:
            self._active.cancel()

    def start_fresh(self) -> None:
        """Forget the conversation everywhere (e.g. after deleting a memory)."""
        self.history = []
        self._seen = {"local": 0, "online": 0}
        self._must_reset = {"local", "online"}

    def close(self) -> None:
        self._stop.set()
        for b in (self.online, self.local):
            if b is not None and hasattr(b, "close"):
                b.close()
