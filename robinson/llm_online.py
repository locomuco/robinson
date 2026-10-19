"""Online backend: Claude via the Claude Code CLI and the user's own subscription.

One `claude -p` process stays alive in stream-json mode (a fresh process per
turn would add seconds). Built-in Claude Code tools are disabled; Robinson's
own tools come from the robinson.mcp_tools stdio server, which shares
memory.json with the local backend. Auth: CLAUDE_CODE_OAUTH_TOKEN (from
`claude setup-token`) — personal use of the unmodified Claude Code binary.

History is plain text, so a rebuilt process (after barge-in, forgetting or
switching backends) gets earlier turns appended to its system prompt.
"""

import json
import logging
import os
import select
import socket
import subprocess
import sys
import tempfile
import threading
import time

from .llm import Llm

log = logging.getLogger("robinson.llm")

MCP_SERVER = "robinson"
_TOOL_PREFIX = f"mcp__{MCP_SERVER}__"


class OnlineUnavailable(RuntimeError):
    """No usable answer from Claude (network, auth, timeout, process died)."""


def internet_ok(host: str = "api.anthropic.com", timeout: float = 2.0) -> bool:
    try:
        socket.create_connection((host, 443), timeout=timeout).close()
        return True
    except OSError:
        return False


class ClaudeCodeLlm(Llm):
    name = "online"
    first_timeout_s = 5.0      # no API event this long after sending → give up (normal: < 1 s)
    idle_timeout_s = 20.0      # stream stalls this long mid-reply → give up

    def __init__(self, system_prompt, model: str = "haiku", memory_file: str | None = None,
                 on_tool=None, claude_bin: str | None = None, max_turns_kept: int = 20,
                 thinking: bool = False, eyes_file: str | None = None,
                 state_file: str | None = None, settings_file: str | None = None):
        super().__init__(system_prompt, tools=None, on_tool=on_tool)  # tools run in the MCP server
        self.model = model
        self.thinking = thinking
        self.claude = claude_bin or os.path.expanduser("~/.local/bin/claude")
        self.max_turns_kept = max_turns_kept
        # empty cwd: no project CLAUDE.md/settings. (--safe-mode would also drop MCP servers)
        self.workdir = tempfile.mkdtemp(prefix="robinson-claude-")
        mcp_cfg = {"mcpServers": {MCP_SERVER: {
            "command": sys.executable, "args": ["-m", "robinson.mcp_tools"],
            "env": {"ROBINSON_MEMORY_FILE": memory_file or "",
                    "ROBINSON_EYES_FILE": eyes_file or "",
                    "ROBINSON_STATE_FILE": state_file or "",
                    "ROBINSON_SETTINGS_FILE": settings_file or "",
                    "PYTHONPATH": os.path.dirname(os.path.dirname(os.path.abspath(__file__)))},
        }}}
        self.mcp_path = os.path.join(self.workdir, "mcp.json")
        with open(self.mcp_path, "w") as f:
            json.dump(mcp_cfg, f)
        self.proc: subprocess.Popen | None = None
        self._stderr: list[str] = []
        t0 = time.monotonic()
        self.reset()
        log.info("online LLM (Claude Code, %s) started in %.1fs", model, time.monotonic() - t0)

    # ── process management ───────────────────────────────────────────────────
    def _open(self, history):
        self._kill()
        prompt = self.system_prompt()
        if history:
            turns = history[-self.max_turns_kept:]
            lines = "\n".join(("Mensch: " if m["role"] == "user" else "Robinson: ") + m["content"]
                              for m in turns)
            prompt += f"\n\nBisheriger Gesprächsverlauf (setze ihn nahtlos fort):\n{lines}"
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}  # use the subscription
        if not self.thinking:
            env["MAX_THINKING_TOKENS"] = "0"   # short spoken replies: speed over deliberation
        self.proc = subprocess.Popen(
            [self.claude, "-p", "--input-format", "stream-json",
             "--output-format", "stream-json", "--include-partial-messages", "--verbose",
             "--no-session-persistence", "--model", self.model,
             "--tools", "", "--mcp-config", self.mcp_path, "--strict-mcp-config",
             "--allowedTools", f"{_TOOL_PREFIX}*", "--system-prompt", prompt],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, cwd=self.workdir, env=env)   # binary: read with select + timeouts
        self._stderr = []
        threading.Thread(target=self._drain_stderr, args=(self.proc,), daemon=True).start()
        log.info("claude process started (model %s, %d history messages)", self.model, len(history))

    def _drain_stderr(self, proc) -> None:
        for line in proc.stderr:
            self._stderr.append(line.decode(errors="replace").rstrip())
            del self._stderr[:-20]

    def _kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        self.proc = None

    def _cancel(self):
        # stop generating now; chat() rebuilds the process from history next turn
        self._kill()

    def close(self) -> None:
        self._kill()

    # ── one turn ─────────────────────────────────────────────────────────────
    def _events(self):
        """JSON events from claude's stdout, raising OnlineUnavailable on timeouts.

        The deadline only advances on real progress (text, tool use, result) —
        an offline claude still prints its init line and then retries forever."""
        fd = self.proc.stdout.fileno()
        buf = b""
        deadline = time.monotonic() + self.first_timeout_s
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise OnlineUnavailable("timeout waiting for Claude")
            ready, _, _ = select.select([fd], [], [], left)
            if not ready:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                if self._cancelled:
                    return
                raise OnlineUnavailable(f"claude process ended: {self._stderr[-3:]}")
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") in ("stream_event", "assistant", "result"):
                    deadline = time.monotonic() + self.idle_timeout_s
                yield ev

    def _stream(self, text):
        if self.proc is None or self.proc.poll() is not None:
            self._open(self.history[:-1])
        content = text
        if (img := getattr(self, "pending_image", None)) is not None:   # camera photo
            import base64
            self.pending_image = None
            content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.b64encode(img).decode()}},
                       {"type": "text", "text": text}]
        msg = {"type": "user", "message": {"role": "user", "content": content}}
        try:
            self.proc.stdin.write((json.dumps(msg, ensure_ascii=False) + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise OnlineUnavailable(f"claude process gone: {e}; {self._stderr[-3:]}") from e
        try:
            for ev in self._events():
                kind = ev.get("type")
                if kind == "stream_event":
                    delta = ev.get("event", {}).get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta["text"]
                elif kind == "assistant":
                    for block in ev.get("message", {}).get("content", []):
                        if block.get("type") == "tool_use":
                            name = block.get("name", "").removeprefix(_TOOL_PREFIX)
                            log.info("tool %s(%s)", name, json.dumps(block.get("input", {}),
                                                                    ensure_ascii=False))
                            self._tool_ran = True
                            if self.on_tool:
                                self.on_tool(name, block.get("input", {}))
                elif kind == "result":
                    if ev.get("is_error"):
                        raise OnlineUnavailable(f"claude: {ev.get('result')}")
                    return
        except OnlineUnavailable:
            self._kill()          # a stuck process is useless; rebuild next time
            raise
