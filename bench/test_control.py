#!/usr/bin/env python3
"""End-to-end check of confirmations, restart requests and the Wartungsmodus."""

import os
import queue
import subprocess
import threading
import time


def main():
    os.makedirs("/tmp/rtest", exist_ok=True)
    env = dict(os.environ, ROBINSON_EYES="off", ROBINSON_BOOT_GREETING="false",
               ROBINSON_STATE_FILE="/tmp/rtest/state.json", ROBINSON_MEMORY_FILE="/tmp/rtest/mem.json",
               ROBINSON_SETTINGS_FILE="/tmp/rtest/settings.json", ROBINSON_MAINT_PIN="9999",
               PYTHONUNBUFFERED="1")
    p = subprocess.Popen([".venv/bin/robinson", "--text", "--no-audio"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env=env)
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [lines.put(ln) for ln in p.stdout], daemon=True).start()

    def show(line):
        if line.startswith("Robinson:"):
            print("   ", line.strip()[:220], flush=True)
        elif any(k in line for k in ("tool ", "confirmed", "waiting for confirmation",
                                     "would run", "maintenance job", "Traceback", "Error")):
            print("    log:", line.strip()[24:200], flush=True)

    def send(q, wait_for="Robinson:", timeout=90):
        print(f">>> {q}", flush=True)
        p.stdin.write(q + "\n")
        p.stdin.flush()
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            try:
                line = lines.get(timeout=1)
            except queue.Empty:
                continue
            show(line)
            if wait_for in line:
                return True
        return False

    time.sleep(10)
    send("Starte bitte den Pi neu.")
    send("Nein, lieber nicht.")
    send("Starte Robinson neu.")
    send("Ja.", wait_for="would run")
    task = ("Wartungsmodus: Ergänze in der README im Abschnitt Configuration eine Zeile, dass "
            "ROBINSON_MAINT_PIN den Wartungsmodus mit einer gesprochenen PIN aktiviert.")
    send(task)
    send("Ja, eins zwei drei vier.")
    send(task)
    send("Ja, neun neun neun neun.", wait_for="ich sage Bescheid")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 480:      # job runs in the background; poll with blank lines
        p.stdin.write("\n")
        p.stdin.flush()
        try:
            line = lines.get(timeout=10)
        except queue.Empty:
            continue
        show(line)
        if "maintenance job done" in line:
            time.sleep(1)
            p.stdin.write("\n")
            p.stdin.flush()
            time.sleep(2)
            while not lines.empty():
                show(lines.get())
            break
    send("Wartung rückgängig.")
    send("Ja 9999.", wait_for="zurückgenommen", timeout=60)
    p.stdin.close()
    p.wait(timeout=30)


if __name__ == "__main__":
    main()
