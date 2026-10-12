"""Gemma 4 E2B on LiteRT-LM: streaming chat with tools, cancel and inline tags.

The model keeps the whole conversation in its KV cache, so each turn only
prefills the new user message. Tools run automatically inside the stream;
a tool call costs one extra generation pass (~2-3 s). Bracket tags and
emojis are split off the text so they never reach TTS.
"""

import functools
import logging
import re
import threading
import time
from collections.abc import Iterator

import litert_lm as lm

log = logging.getLogger("robinson.llm")

_TAG = re.compile(r"\[([^\[\]]{1,20})\]")
# mood emojis are one token each (vs 4-6 for "[fröhlich]") — ~0.4 s earlier speech
EMOJI_MOOD = {"😄": "fröhlich", "😐": "neutral", "😮": "überrascht", "🤔": "nachdenklich",
              "😢": "traurig", "😠": "verärgert", "😴": "müde", "🤫": "flüsternd"}
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")


class Llm:
    def __init__(self, model_path: str, system_prompt, threads: int = 4,
                 max_tokens: int = 4096, cache_dir: str | None = None,
                 tools: list | None = None):
        lm.set_min_log_severity(lm.LogSeverity.ERROR)
        t0 = time.monotonic()
        self.engine = lm.Engine(
            model_path,
            backend=lm.Backend.CPU(thread_count=threads),
            max_num_tokens=max_tokens,
            enable_ynnpack=True,          # 2.5x prefill, 1.8x decode on the Pi 5
            cache_dir=cache_dir,
        )
        # str, or a callable so the memory section is fresh on every reset
        self.system_prompt = system_prompt if callable(system_prompt) else (lambda: system_prompt)
        self.tools = [self._wrap_tool(t) for t in tools or []]
        self._tool_ran = False
        self.max_tokens = max_tokens
        self._lock = threading.Lock()
        self.history: list[dict] = []     # mirror of the conversation, for rebuilds
        self._cancelled = False
        self._fresh = False               # rebuild without history (after forgetting)
        self.conv = None
        self.reset()
        log.info("LLM ready in %.1fs", time.monotonic() - t0)

    def reset(self, keep: int = 0) -> None:
        """New conversation, replaying the last `keep` messages of history."""
        if self.conv is not None:
            self.conv.close()
        self.history = self.history[-keep:] if keep else []
        self.conv = self.engine.create_conversation(
            system_message=self.system_prompt(), messages=self.history or None,
            tools=self.tools or None)
        self._cancelled = False

    def _wrap_tool(self, fn):
        @functools.wraps(fn)   # keeps signature + docstring = the schema Gemma sees
        def wrapper(*args, **kwargs):
            t0 = time.monotonic()
            result = fn(*args, **kwargs)
            log.info("tool %s(%s) → %s (%.2fs)", fn.__name__,
                     ", ".join([*map(repr, args), *(f"{k}={v!r}" for k, v in kwargs.items())]),
                     result, time.monotonic() - t0)
            self._tool_ran = True
            return result
        return wrapper

    def start_fresh(self) -> None:
        """Drop the conversation before the next turn (e.g. something was
        forgotten and must not linger in the KV cache or history)."""
        self._fresh = True

    def cancel(self) -> None:
        """Abort the reply being generated (barge-in). Safe from any thread."""
        self._cancelled = True
        try:
            self.conv.cancel_process()
        except Exception as e:
            log.debug("cancel: %s", e)

    def chat(self, text: str) -> Iterator[tuple[str, str]]:
        """Stream a reply as ("text", chunk) and ("tag", name) events."""
        with self._lock:
            if self._fresh:
                self._fresh = False
                self.reset(keep=0)
            elif self._cancelled:
                # a cancelled LiteRT conversation stays dead: rebuild it from history
                self.reset(keep=len(self.history))
            elif self.conv.token_count > self.max_tokens - 512:
                # leave room for the next exchange; keep only the recent turns
                log.info("context full — starting a new conversation")
                self.reset(keep=4)
            self.history.append({"role": "user", "content": text})
            self._tool_ran = False
            reply = ""
            buf = ""
            try:
                for chunk in self.conv.send_message_async(text):
                    piece = "".join(c.get("text", "") for c in chunk.get("content", [])
                                    if c.get("type", "text") == "text")
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
