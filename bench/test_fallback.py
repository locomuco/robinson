#!/usr/bin/env python3
"""End-to-end check of the online → local fallback and mode switching.

Drives `robinson --text --no-audio` and simulates an internet outage by
blackholing api.anthropic.com (needs sudo; routes are always removed again).
"""

import os
import queue
import subprocess
import sys
import threading
import time

ROUTES = ["160.79.104.10", "2607:6bc0::10"]
SUDO_PW = os.environ.get("SUDO_PW", "")


def sudo(*cmd):
    subprocess.run(["sudo", "-S", "-p", "", *cmd], input=SUDO_PW + "\n", text=True,
                   capture_output=True)


def net(block: bool):
    for r in ROUTES:
        fam = ["-6"] if ":" in r else []
        sudo("ip", *fam, "route", "add" if block else "del", "blackhole", r)
    print(f"--- internet {'BLOCKED' if block else 'restored'}", flush=True)


def main():
    env = dict(os.environ, ROBINSON_LLM="online", ROBINSON_EYES="off", ROBINSON_DEBUG="0",
               ROBINSON_BOOT_GREETING="false", ROBINSON_STATE_FILE="/tmp/state-test.json",
               ROBINSON_MEMORY_FILE="/tmp/mem-test.json", PYTHONUNBUFFERED="1")
    for f in ("/tmp/state-test.json", "/tmp/mem-test.json"):
        if os.path.exists(f):
            os.remove(f)
    p = subprocess.Popen([".venv/bin/robinson", "--text", "--no-audio"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env=env)

    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [lines.put(ln) for ln in p.stdout], daemon=True).start()

    def ask(q, timeout=90):
        t0 = time.monotonic()
        print(f">>> {q}", flush=True)
        p.stdin.write(q + "\n")
        p.stdin.flush()
        while time.monotonic() - t0 < timeout:
            try:
                line = lines.get(timeout=1)
            except queue.Empty:
                continue
            if any(k in line for k in ("online LLM", "LLM mode", "reachable", "tool ", "warmed")):
                print("    log:", line.strip()[24:160], flush=True)
            if line.startswith("Robinson:"):
                print(f"    [{time.monotonic() - t0:4.1f}s] {line.strip()[:170]}", flush=True)
                return
        print("    TIMEOUT", flush=True)

    try:
        time.sleep(12)                       # let Gemma finish loading in the background
        ask("Hallo, ich heiße Matthias.")
        net(True)
        ask("Wie spät ist es?")
        ask("Wie heiße ich?")
        net(False)
        print("--- waiting 55 s for the health check", flush=True)
        time.sleep(55)
        ask("Bist du wieder da? Antworte kurz.")
        ask("Wechsel bitte auf das lokale Modell.")
        ask("Wer bist du? Ein Satz.")
        ask("Geh online.")
        ask("Und wer bist du jetzt? Ein Satz.")
        print("--- state file:", open("/tmp/state-test.json").read(), flush=True)
    finally:
        net(False)
        p.stdin.close()
        p.wait(timeout=30)


if __name__ == "__main__":
    sys.exit(main())
