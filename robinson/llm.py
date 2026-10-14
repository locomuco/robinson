"""LLM backends behind one interface.

Robinson talks to `Llm.chat(text)`, which streams ("text", chunk) and
("tag", name) events. Everything model-independent lives in `Llm`: the
shared history (so a backend can be swapped and rebuild the conversation),
tool wrapping/logging, the sentence break after a tool call, tag/emoji
filtering and the cancel/forget rebuild rules. Backends only implement
`_open` (start a conversation from history), `_stream` (raw text pieces for
one user turn, running tools themselves), `_cancel` and `_context_full`.

Backends: LocalLlm (Gemma 4 E2B on LiteRT-LM). Online backends live in
llm_online.py and are selected with ROBINSON_LLM.
"""

import functools
import logging
import re
import threading
import time
from collections.abc import Callable, Iterator

log = logging.getLogger("robinson.llm")

_TAG = re.compile(r"\[([^\[\]]{1,20})\]")
EMOJI_MOOD = {"😄": "fröhlich", "😐": "neutral", "😮": "überrascht", "🤔": "nachdenklich",
              "😢": "traurig", "😠": "verärgert", "😴": "müde", "🤫": "flüsternd"}
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿️‍]")


class Llm:
    name = "llm"

    def __init__(self, system_prompt: Callable[[], str] | str, tools: list | None = None,
                 on_tool: Callable[[str, dict], None] | None = None):
        # str, or a callable so the memory section is fresh on every reset
        self.system_prompt = system_prompt if callable(system_prompt) else (lambda: system_prompt)
        self.on_tool = on_tool                # callback(name, args) after each tool call
        self.tools = [self._wrap_tool(t) for t in tools or []]
        self.history: list[dict] = []     # {"role": "user"|"assistant", "content": str}
        self._lock = threading.Lock()
        self._tool_ran = False
        self._cancelled = False
        self._fresh = False

    # ── backend hooks ────────────────────────────────────────────────────────
    def _open(self, history: list[dict]) -> None:
        """Start a conversation with the current system prompt + history."""
        raise NotImplementedError

    def _stream(self, text: str) -> Iterator[str]:
        """Raw reply text for one user turn; tools are executed inside."""
        raise NotImplementedError

    def _cancel(self) -> None:
        pass

    def _context_full(self) -> bool:
        return False

    # ── shared behaviour ─────────────────────────────────────────────────────
    def reset(self, keep: int = 0) -> None:
        """New conversation, replaying the last `keep` messages of history."""
        self.history = self.history[-keep:] if keep else []
        self._open(self.history)
        self._cancelled = False

    def start_fresh(self) -> None:
        """Drop the conversation before the next turn (e.g. something was
        forgotten and must not linger in the context or history)."""
        self._fresh = True

    def cancel(self) -> None:
        """Abort the reply being generated (barge-in). Safe from any thread."""
        self._cancelled = True
        try:
            self._cancel()
        except Exception as e:
            log.debug("cancel: %s", e)

    def _wrap_tool(self, fn):
        @functools.wraps(fn)   # keeps signature + docstring = the schema the model sees
        def wrapper(*args, **kwargs):
            t0 = time.monotonic()
            result = fn(*args, **kwargs)
            log.info("tool %s(%s) → %s (%.2fs)", fn.__name__,
                     ", ".join([*map(repr, args), *(f"{k}={v!r}" for k, v in kwargs.items())]),
                     result, time.monotonic() - t0)
            self._tool_ran = True
            if self.on_tool:
                self.on_tool(fn.__name__, kwargs)
            return result
        return wrapper

    def chat(self, text: str) -> Iterator[tuple[str, str]]:
        """Stream a reply as ("text", chunk) and ("tag", name) events."""
        with self._lock:
            if self._fresh:
                self._fresh = False
                self.reset(keep=0)
            elif self._cancelled:
                # cancelled conversations are rebuilt from history (LiteRT's stay dead)
                self.reset(keep=len(self.history))
            elif self._context_full():
                log.info("context full — starting a new conversation")
                self.reset(keep=4)
            self.history.append({"role": "user", "content": text})
            self._tool_ran = False
            reply = ""
            buf = ""
            try:
                for piece in self._stream(text):
                    if piece and self._tool_ran:
                        # text resumes after a tool call: keep the sentences apart
                        self._tool_ran = False
                        if reply and not reply[-1].isspace():
                            piece = (" " if reply[-1] in ".!?" else ". ") + piece.lstrip()
                    reply += piece
                    buf += piece
                    # hold back a possibly unfinished "[..." so tags are never split
                    cut = buf.rfind("[")
                    if cut != -1 and "]" not in buf[cut:] and len(buf) - cut <= 20:
                        ready, buf = buf[:cut], buf[cut:]
                    else:
                        ready, buf = buf, ""
                    yield from _split_tags(ready)
                yield from _split_tags(buf)
            finally:
                # also on barge-in: the model should know what it already said
                self.history.append({"role": "assistant", "content": reply.strip() or "…"})


class LocalLlm(Llm):
    """Gemma 4 E2B on LiteRT-LM (CPU + YNNPACK). The conversation lives in the
    KV cache, so each turn only prefills the new message (~0.8 s floor)."""

    name = "local"

    def __init__(self, model_path: str, system_prompt, threads: int = 4,
                 max_tokens: int = 4096, cache_dir: str | None = None,
                 tools: list | None = None, on_tool=None, model_key: str = "e2b"):
        import litert_lm as lm
        self.model_key = model_key
        super().__init__(system_prompt, tools, on_tool)
        lm.set_min_log_severity(lm.LogSeverity.ERROR)
        t0 = time.monotonic()
        self.engine = lm.Engine(
            model_path,
            backend=lm.Backend.CPU(thread_count=threads),
            max_num_tokens=max_tokens,
            enable_ynnpack=True,          # 2.5x prefill, 1.8x decode on the Pi 5
            cache_dir=cache_dir,
        )
        self.max_tokens = max_tokens
        self.conv = None
        self.warm = False             # KV holds system prompt + tools, no real turns yet
        self.reset()
        log.info("local LLM ready in %.1fs", time.monotonic() - t0)

    def close(self) -> None:
        if self.conv is not None:
            self.conv.close()
            self.conv = None
        self.engine.close()

    def warm_up(self) -> None:
        """Prefill system prompt + tool schemas now (several seconds) with a hidden
        turn, so a later fallback only pays for the new message."""
        t0 = time.monotonic()
        # neutral content, 1 output token: nothing here should colour later answers
        for _ in self.conv.send_message_async("(Robinson ist im Bereitschaftsmodus.)",
                                              max_output_tokens=1):
            pass
        self.warm = True
        log.info("local LLM warmed up in %.1fs", time.monotonic() - t0)

    def reset(self, keep: int = 0) -> None:
        self.warm = False
        super().reset(keep)

    def _open(self, history):
        if self.conv is not None:
            self.conv.close()
        self.conv = self.engine.create_conversation(
            system_message=self.system_prompt(), messages=history or None,
            tools=self.tools or None)

    def _stream(self, text):
        for chunk in self.conv.send_message_async(text):
            yield "".join(c.get("text", "") for c in chunk.get("content", [])
                          if c.get("type", "text") == "text")

    def _cancel(self):
        self.conv.cancel_process()

    def _context_full(self):
        # leave room for the next exchange
        return self.conv.token_count > self.max_tokens - 512


def _split_tags(s: str) -> Iterator[tuple[str, str]]:
    for e in _EMOJI.findall(s):
        if e in EMOJI_MOOD:
            yield "tag", EMOJI_MOOD[e]
    s = _EMOJI.sub("", s)    # never let piper see emojis
    pos = 0
    for m in _TAG.finditer(s):
        if m.start() > pos:
            yield "text", s[pos:m.start()]
        yield "tag", m.group(1).strip().lower()
        pos = m.end()
    if pos < len(s):
        yield "text", s[pos:]
