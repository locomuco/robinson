"""Wartungsmodus: free-form changes to Robinson by voice, done by Claude Code.

After a spoken confirmation + PIN, a `claude -p` job (Sonnet) works on the
Robinson repo with file and shell tools (no sudo). Every job ends in a git
commit, so "Wartung rückgängig" reverts it. Results are read out; if code
changed, Robinson restarts itself to load it.
"""

import json
import logging
import os
import re
import subprocess
import threading
import time

log = logging.getLogger("robinson.maint")

RULES = """\
Du arbeitest per Sprachauftrag am Code des Sprach-Roboters "Robinson" in diesem
Git-Repository auf einem Raspberry Pi 5 (Python-Paket robinson/, venv in .venv,
Einstellungen und Daten in ~/robinson-data). Der Auftrag kommt aus einer
Spracherkennung und kann ungenau formuliert sein.

Regeln:
- Ändere nur, was für den Auftrag nötig ist. Kein sudo, nichts außerhalb dieses
  Repos und ~/robinson-data anfassen, keine Secrets (robinson.env) lesen oder ausgeben.
- Prüfe deine Änderung mindestens mit `.venv/bin/python -m py_compile robinson/*.py`.
- Committe am Ende mit git (Commit-Nachricht beginnt mit "voice: ").
- Starte Robinson nicht neu, das passiert danach automatisch.
- Deine letzte Antwort wird vorgelesen: ein bis zwei kurze deutsche Sätze, was du
  geändert hast oder warum es nicht ging. Kein Markdown.
"""


def _git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout.strip()


class Maintenance:
    def __init__(self, repo: str, state_dir: str, on_done, model: str = "sonnet",
                 claude_bin: str | None = None, timeout_s: int = 900):
        self.repo = repo
        self.state_path = os.path.join(state_dir, "maintenance.json")
        self.on_done = on_done                      # callback(spoken_text, code_changed)
        self.model = model
        self.claude = claude_bin or os.path.expanduser("~/.local/bin/claude")
        self.timeout_s = timeout_s
        self.running: str | None = None
        self.started = 0.0

    @property
    def available(self) -> bool:
        return os.path.exists(self.claude) and os.path.isdir(os.path.join(self.repo, ".git"))

    def status(self) -> str:
        if self.running:
            return (f"Ich arbeite seit {int(time.monotonic() - self.started)} Sekunden "
                    f"am Wartungsauftrag: {self.running}.")
        last = self._last()
        return f"Kein Wartungsauftrag läuft. Zuletzt: {last.get('task', 'keiner')}."

    def _last(self) -> dict:
        try:
            with open(self.state_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def start(self, task: str) -> None:
        if self.running:
            self.on_done("Ich arbeite schon an einem Wartungsauftrag.", False)
            return
        self.running, self.started = task, time.monotonic()
        last = self._last()
        last["running"] = task                      # visible to the MCP status tool
        with open(self.state_path, "w") as f:
            json.dump(last, f, ensure_ascii=False)
        threading.Thread(target=self._run, args=(task,), name="maintenance", daemon=True).start()

    def _untracked(self) -> set[str]:
        return set(_git(self.repo, "ls-files", "--others", "--exclude-standard").splitlines())

    def _code_changed(self, a: str, b: str) -> bool:
        return a != b and bool(_git(self.repo, "diff", "--name-only", a, b, "--",
                                    "robinson", "pyproject.toml"))

    def _run(self, task: str) -> None:
        before = _git(self.repo, "rev-parse", "HEAD")
        untracked_before = self._untracked()
        log.info("maintenance job started: %s", task)
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        cmd = [self.claude, "-p", task, "--model", self.model, "--output-format", "json",
               "--permission-mode", "acceptEdits", "--no-session-persistence",
               "--allowedTools", "Read", "Edit", "Write", "Glob", "Grep", "Bash",
               "--disallowedTools", "Bash(sudo:*)", "Bash(*robinson.env*)",
               "Read(robinson.env)", "Read(./robinson.env)", "Read(**/robinson.env)",
               "--append-system-prompt", RULES]
        summary, ok = "", False
        try:
            r = subprocess.run(cmd, cwd=self.repo, env=env, capture_output=True, text=True,
                               timeout=self.timeout_s)
            try:
                out = json.loads(r.stdout)
                summary, ok = str(out.get("result", "")).strip(), not out.get("is_error")
            except json.JSONDecodeError:
                summary = (r.stdout or r.stderr).strip()[-300:]
        except subprocess.TimeoutExpired:
            summary = "Der Wartungsauftrag hat zu lange gedauert und wurde abgebrochen."
        except OSError as e:
            summary = f"Claude Code ließ sich nicht starten: {e}"
        # commit what the job changed but left uncommitted (only its own new files,
        # never unrelated untracked ones), so the job can always be reverted
        _git(self.repo, "add", "-u")
        for f in sorted(self._untracked() - untracked_before):
            _git(self.repo, "add", "--", f)
        if _git(self.repo, "diff", "--cached", "--name-only"):
            _git(self.repo, "commit", "-q", "-m", f"voice: {task[:60]}")
        after = _git(self.repo, "rev-parse", "HEAD")
        changed = after != before
        code_changed = self._code_changed(before, after)
        with open(self.state_path, "w") as f:
            json.dump({"task": task, "before": before, "after": after, "ok": ok,
                       "summary": summary, "running": None}, f, ensure_ascii=False)
        log.info("maintenance job done (changed=%s): %s", changed, summary[:200])
        self.running = None
        summary = re.sub(r"[*`#]", "", summary)[:400] or "Fertig."
        self.on_done(summary, code_changed)

    def rollback(self) -> None:
        last = self._last()
        before, after = last.get("before"), last.get("after")
        if not before or before == after:
            self.on_done("Es gibt keinen Wartungsauftrag zum Zurücknehmen.", False)
            return
        r = subprocess.run(["git", "-C", self.repo, "revert", "--no-edit", f"{before}..{after}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            _git(self.repo, "revert", "--abort")
            self.on_done("Das Zurücknehmen hat nicht geklappt.", False)
            return
        code_changed = self._code_changed(before, after)
        last["after"] = last["before"]
        with open(self.state_path, "w") as f:
            json.dump(last, f, ensure_ascii=False)
        self.on_done(f"Ich habe den Wartungsauftrag zurückgenommen: {last.get('task')}.",
                     code_changed)
