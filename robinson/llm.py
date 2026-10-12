"""Gemma 4 E2B on LiteRT-LM: streaming chat with cancel and inline tags.

The model keeps the whole conversation in its KV cache, so each turn only
prefills the new user message. Replies start with a mood tag like
"[fröhlich]" and may contain eye actions like "[links]"; those are split off
the text stream and delivered as events so neither ends up in TTS.
"""

import logging
import re
import threading
import time
from collections.abc import Iterator

import litert_lm as lm

log = logging.getLogger("robinson.llm")

_TAG = re.compile(r"\[([^\[\]]{1,20})\]")


class Llm:
    def __init__(self, model_path: str, system_prompt: str, threads: int = 4,
                 max_tokens: int = 4096, cache_dir: str | None = None):
        lm.set_min_log_severity(lm.LogSeverity.ERROR)
        t0 = time.monotonic()
        self.engine = lm.Engine(
            model_path,
            backend=lm.Backend.CPU(thread_count=threads),
            max_num_tokens=max_tokens,
            enable_ynnpack=True,          # 2.5x prefill, 1.8x decode on the Pi 5
            cache_dir=cache_dir,
        )
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self._lock = threading.Lock()
        self.history: list[dict] = []     # mirror of the conversation, for rebuilds
        self._cancelled = False
        self.conv = None
        self.reset()
        log.info("LLM ready in %.1fs", time.monotonic() - t0)

    def reset(self, keep: int = 0) -> None:
        """New conversation, replaying the last `keep` messages of history."""
        if self.conv is not None:
            self.conv.close()
        self.history = self.history[-keep:] if keep else []
        self.conv = self.engine.create_conversation(
            system_message=self.system_prompt, messages=self.history or None)
        self._cancelled = False

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
            if self._cancelled:
                # a cancelled LiteRT conversation stays dead: rebuild it from history
                self.reset(keep=len(self.history))
            elif self.conv.token_count > self.max_tokens - 512:
                # leave room for the next exchange; keep only the recent turns
                log.info("context full — starting a new conversation")
                self.reset(keep=4)
            self.history.append({"role": "user", "content": text})
            reply = ""
            buf = ""
            try:
                for chunk in self.conv.send_message_async(text):
                    piece = "".join(c.get("text", "") for c in chunk.get("content", [])
                                    if c.get("type", "text") == "text")
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
    pos = 0
    for m in _TAG.finditer(s):
        if m.start() > pos:
            yield "text", s[pos:m.start()]
        yield "tag", m.group(1).strip().lower()
        pos = m.end()
    if pos < len(s):
        yield "text", s[pos:]
