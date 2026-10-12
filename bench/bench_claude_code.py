#!/usr/bin/env python3
"""Latency probe: Claude Code (subscription) as Robinson's online LLM.

Keeps ONE `claude -p` process alive in stream-json mode and measures, per
German turn, time to first text delta, to the first sentence end and total.
Auth comes from CLAUDE_CODE_OAUTH_TOKEN in the environment.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robinson.config import Config  # noqa: E402

TURNS = ["Hallo Robinson, wie geht es dir?",
         "Erzähl mir einen kurzen Witz über Roboter.",
         "Was sollte ich heute Abend kochen, wenn ich nur Kartoffeln, Eier und Zwiebeln habe?"]


def main(model: str) -> None:
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    workdir = tempfile.mkdtemp(prefix="robinson-claude-")   # no CLAUDE.md etc.
    t0 = time.monotonic()
    p = subprocess.Popen(
        [os.path.expanduser("~/.local/bin/claude"), "-p",
         "--input-format", "stream-json", "--output-format", "stream-json",
         "--include-partial-messages", "--verbose", "--safe-mode",
         "--no-session-persistence", "--tools", "", "--model", model,
         "--system-prompt", Config().system_prompt],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, cwd=workdir, env=env)
    for i, turn in enumerate(TURNS):
        start = time.monotonic()
        p.stdin.write(json.dumps({"type": "user", "message": {"role": "user", "content": turn}}) + "\n")
        p.stdin.flush()
        first = sentence = None
        text = ""
        for line in p.stdout:
            ev = json.loads(line)
            if ev.get("type") == "stream_event":
                d = ev["event"].get("delta", {})
                if d.get("type") == "text_delta":
                    now = time.monotonic() - start
                    first = first or now
                    text += d["text"]
                    if sentence is None and re.search(r"[.!?](\s|$)", text):
                        sentence = now
            elif ev.get("type") == "result":
                if ev.get("is_error"):
                    print("ERROR:", ev.get("result"), file=sys.stderr)
                break
        total = time.monotonic() - start
        extra = f" (process start {start - t0:.1f}s)" if i == 0 else ""
        print(f"[{model}] first {first or 0:.2f}s  sentence {sentence or 0:.2f}s  "
              f"total {total:.2f}s{extra}\n    {text.strip()[:150]!r}", flush=True)
    p.stdin.close()
    p.wait(timeout=10)
    err = p.stderr.read().strip()
    if err:
        print("stderr:", err[:300])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "haiku")
