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
import threading
import time
from collections.abc import Callable, Iterator

from .llm import Llm

log = logging.getLogger("robinson.llm")

MODES = ("local", "online")


class LlmRouter:
    health_interval_s = 20.0
    min_offline_s = 30.0        # after a failure, don't retry online sooner than this

    def __init__(self, make_local: Callable[[], Llm], make_online: Callable[[], Llm] | None,
                 mode: str, state_file: str):
        self.make_local, self.make_online = make_local, make_online
        self.state_file = state_file
        self.mode = self._load_mode(mode)
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
    def _load_mode(self, default: str) -> str:
        try:
            with open(self.state_file) as f:
                mode = json.load(f).get("llm", default)
        except (OSError, json.JSONDecodeError):
            mode = default
        return mode if mode in MODES else "local"

    def _save_mode(self) -> None:
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        state = {}
        try:
            with open(self.state_file) as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
        state["llm"] = self.mode
        with open(self.state_file, "w") as f:
            json.dump(state, f)

    def _load_local(self) -> None:
        local = self.make_local()
        if self.mode == "online" and hasattr(local, "warm_up"):
            local.warm_up()          # stand-by: makes a fallback turn ~5 s faster
        self.local = local
        self._local_ready.set()

    def _start_online(self) -> None:
        try:
            self.online = self.make_online()
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
        if self.mode == "local":
            return "lokal"
        return "online" if self._online_ok else "online gewählt, aber offline (antwortet lokal)"

    def set_mode(self, mode: str) -> str:
        mode = {"lokal": "local", "offline": "local", "local": "local",
                "online": "online", "claude": "online", "cloud": "online"}.get(mode.lower().strip())
        if mode is None:
            return "Ziel muss 'lokal' oder 'online' sein."
        if mode == "online" and not self.make_online:
            return "Online ist nicht eingerichtet (kein Claude-Code-Token)."
        changed = mode != self.mode
        self.mode = mode
        self._save_mode()
        if mode == "online":
            if self.online is None:
                self._start_online()
            self._online_ok = self.online is not None
        log.info("LLM mode → %s", mode)
        if not changed:
            return f"Ich bin schon {('online' if mode == 'online' else 'lokal')}."
        return ("Ab jetzt denke ich online mit Claude." if mode == "online"
                else "Ab jetzt denke ich lokal, ganz ohne Internet.")

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
