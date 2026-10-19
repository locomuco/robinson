"""Runtime settings changeable by voice, persisted in settings.json.

Values override the matching Config fields at startup and are applied live
when a tool changes them (the tool may run in the MCP server process, so the
file is the hand-over; the main process reloads it in its on_tool callback).
Mixer levels are set with amixer (no root needed) and re-applied at startup.
"""

import json
import logging
import os
import re
import socket
import subprocess
import threading

log = logging.getLogger("robinson.settings")

# key: (kind, default, step, min, max)
SPEC = {
    "tts_pitch":         ("float", 1.04, 0.02, 0.85, 1.2),
    "tts_length_scale":  ("float", 0.9, 0.05, 0.7, 1.3),
    "tts_tone":          ("choice", "neutral", ("neutral", "warm", "hell"), None, None),
    "tts_robot":         ("choice", "aus", ("aus", "leicht", "mittel", "stark"), None, None),
    "volume":            ("int", 100, 10, 10, 100),
    "mic_gain":          ("int", 80, 10, 20, 100),
    "fillers":           ("bool", True, None, None, None),
    "barge_in":          ("bool", True, None, None, None),
    "boot_greeting":     ("bool", True, None, None, None),
    "vad_min_silence_s": ("float", 0.45, 0.1, 0.25, 1.2),
}
SETTINGS_TOOL_NAMES = {"stimme_einstellen", "lautstaerke_einstellen", "mikrofon_einstellen",
                       "gespraech_einstellen"}


class Settings:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self.values: dict = {}
        self.reload()

    def reload(self) -> None:
        try:
            with open(self.path) as f:
                self.values = json.load(f)
        except (OSError, json.JSONDecodeError):
            self.values = {}

    def get(self, key: str, default=None):
        return self.values.get(key, default if default is not None else SPEC[key][1])

    def set(self, key: str, value) -> None:
        with self._lock:
            self.reload()
            self.values[key] = value
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.values, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)
        log.info("setting %s = %r", key, value)

    def apply_to(self, cfg) -> None:
        """Startup: saved values override the Config defaults."""
        for k, v in self.values.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    def step(self, key: str, up: bool, cfg=None) -> float:
        kind, default, step, lo, hi = SPEC[key]
        cur = self.get(key, getattr(cfg, key, default) if cfg else None)
        new = max(lo, min(hi, cur + (step if up else -step)))
        new = int(new) if kind == "int" else round(new, 3)
        self.set(key, new)
        return new


# ── mixer (USB audio adapter) ────────────────────────────────────────────────
def _usb_card() -> str:
    try:
        with open("/proc/asound/cards") as f:
            for line in f:
                if "USB" in line and (m := re.match(r"\s*(\d+)", line)):
                    return m.group(1)
    except OSError:
        pass
    return "0"


def set_mixer(control: str, percent: int, capture: bool = False, card: str | None = None) -> bool:
    cmd = ["amixer", "-q", "-c", card or _usb_card(), "sset", control, f"{percent}%"]
    if capture:
        cmd.append("capture")
    ok = subprocess.run(cmd, capture_output=True).returncode == 0
    if not ok:
        log.warning("amixer %s failed", control)
    return ok


RESPEAKER_CARD = "seeed4micvoicec"     # AC108 control card of the ReSpeaker 4-Mic Array


def apply_mixer(settings: Settings) -> None:
    set_mixer("Speaker", int(settings.get("volume")))
    gain = int(settings.get("mic_gain"))
    if os.path.exists(f"/proc/asound/{RESPEAKER_CARD}"):
        for ch in range(1, 5):          # AC108 PGA 0..31 dB
            set_mixer(f"ADC{ch} PGA gain", gain, card=RESPEAKER_CARD)
    else:
        set_mixer("Mic", gain, capture=True)


# ── system status ────────────────────────────────────────────────────────────
def _sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=3).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def system_status(state_file: str | None = None) -> str:
    temp = _sh("vcgencmd", "measure_temp").replace("temp=", "").replace("'C", " Grad")
    thr = _sh("vcgencmd", "get_throttled").split("=")[-1]
    throttled = "ja, wegen Hitze oder Spannung" if thr and int(thr, 16) & 0x6 else "nein"
    try:
        ip = socket.gethostbyname(socket.gethostname() + ".local")
    except OSError:
        ip = _sh("hostname", "-I").split(" ")[0]
    wifi = _sh("iwgetid", "-r") or _sh("nmcli", "-t", "-f", "active,ssid", "dev", "wifi").split(
        "yes:")[-1].split("\n")[0]
    with open("/proc/uptime") as f:
        up_h = float(f.read().split()[0]) / 3600
    disk = _sh("df", "-h", "--output=avail", "/").split("\n")[-1].strip()
    mem = _sh("free", "-g").split("\n")[1].split()
    model = ""
    if state_file and os.path.exists(state_file):
        try:
            with open(state_file) as f:
                st = json.load(f)
            model = (f" Modell: {st.get('llm')}, online {st.get('online_model')}, "
                     f"lokal {st.get('local_model')}.")
        except (OSError, json.JSONDecodeError):
            pass
    version = _sh("git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "log", "-1", "--format=%h %s")
    return (f"Temperatur {temp}, gedrosselt: {throttled}. IP {ip}, WLAN {wifi or 'unbekannt'}. "
            f"Läuft seit {up_h:.1f} Stunden. Freier Speicher {disk}, RAM {mem[6] if len(mem) > 6 else '?'} "
            f"von {mem[1] if len(mem) > 1 else '?'} GB frei.{model}"
            + (f" Version {version}." if version else ""))


# ── tools ────────────────────────────────────────────────────────────────────
def _up(w: str) -> bool | None:
    w = w.lower()
    if re.search(r"höher|hoeher|heller|schneller|lauter|mehr|empfindlich(er)?$|länger|laenger|an\b|ein\b", w) \
            and not re.search(r"weniger|nicht", w):
        return True
    if re.search(r"tiefer|langsamer|leiser|weniger|kürzer|kuerzer|aus\b", w):
        return False
    return None


def make_settings_tools(settings: Settings, state_file: str | None = None) -> list:
    def stimme_einstellen(eigenschaft: str, wert: str) -> str:
        """Ändert deine Stimme dauerhaft.

        Args:
          eigenschaft: "tonhoehe", "klang", "roboter" oder "tempo".
          wert: tonhoehe: "höher"/"tiefer"/"normal"; klang: "neutral"/"warm"/"hell"; roboter: "aus"/"leicht"/"mittel"/"stark"; tempo: "schneller"/"langsamer"/"normal".
        """
        e, w = eigenschaft.lower(), wert.lower().strip()
        if e.startswith("ton"):
            if "normal" in w:
                settings.set("tts_pitch", 1.0)
                return "Tonhöhe ist wieder normal."
            up = _up(w)
            if up is None:
                return "Wert muss höher, tiefer oder normal sein."
            return f"Tonhöhe jetzt {settings.step('tts_pitch', up):.2f}."
        if e.startswith("klang"):
            if w not in SPEC["tts_tone"][2]:
                return "Klang muss neutral, warm oder hell sein."
            settings.set("tts_tone", w)
            return f"Klang ist jetzt {w}."
        if e.startswith("robot"):
            if w not in SPEC["tts_robot"][2]:
                return "Roboter-Effekt muss aus, leicht, mittel oder stark sein."
            settings.set("tts_robot", w)
            return f"Roboter-Effekt ist jetzt {w}."
        if e.startswith("tempo"):
            if "normal" in w:
                settings.set("tts_length_scale", 0.9)
                return "Sprechtempo ist wieder normal."
            up = _up(w)
            if up is None:
                return "Wert muss schneller, langsamer oder normal sein."
            # length_scale: smaller = faster
            return f"Sprechtempo eingestellt ({settings.step('tts_length_scale', not up):.2f})."
        return "Eigenschaft muss tonhoehe, klang, roboter oder tempo sein."

    def lautstaerke_einstellen(wert: str) -> str:
        """Ändert die Lautstärke deines Lautsprechers.

        Args:
          wert: "lauter", "leiser", "maximal" oder eine Prozentzahl wie "70".
        """
        w = wert.lower()
        if m := re.search(r"\d+", w):
            v = max(10, min(100, int(m.group())))
            settings.set("volume", v)
        elif "max" in w:
            settings.set("volume", v := 100)
        elif (up := _up(w)) is not None:
            v = settings.step("volume", up)
        else:
            return "Wert muss lauter, leiser, maximal oder eine Prozentzahl sein."
        return f"Lautstärke jetzt {v} Prozent."

    def mikrofon_einstellen(wert: str) -> str:
        """Ändert, wie empfindlich dein Mikrofon ist.

        Args:
          wert: "empfindlicher", "weniger empfindlich" oder eine Prozentzahl.
        """
        w = wert.lower()
        if m := re.search(r"\d+", w):
            settings.set("mic_gain", v := max(20, min(100, int(m.group()))))
        elif (up := _up(w)) is not None:
            v = settings.step("mic_gain", up)
        else:
            return "Wert muss empfindlicher, weniger empfindlich oder eine Prozentzahl sein."
        return f"Mikrofon jetzt {v} Prozent."

    def gespraech_einstellen(eigenschaft: str, wert: str) -> str:
        """Ändert dein Gesprächsverhalten.

        Args:
          eigenschaft: "fuellwoerter" (Hmm beim Nachdenken), "unterbrechen" (darf man dich unterbrechen), "pause" (wie lange du wartest, bis der Mensch fertig ist) oder "begruessung" (Begrüßung beim Start).
          wert: "an"/"aus", bei pause "länger"/"kürzer".
        """
        e, w = eigenschaft.lower(), wert.lower()
        key = {"fuell": "fillers", "füll": "fillers", "unterb": "barge_in",
               "pause": "vad_min_silence_s", "begr": "boot_greeting"}
        k = next((v for p, v in key.items() if e.startswith(p)), None)
        if k is None:
            return "Eigenschaft muss fuellwoerter, unterbrechen, pause oder begruessung sein."
        up = _up(w)
        if up is None:
            return "Wert muss an/aus bzw. länger/kürzer sein."
        if k == "vad_min_silence_s":
            return f"Ich warte jetzt {settings.step(k, up):.2f} Sekunden Stille ab."
        settings.set(k, up)
        return f"{eigenschaft} ist jetzt {'an' if up else 'aus'}."

    def systemstatus() -> str:
        """Gibt deinen technischen Zustand zurück: Temperatur, Drosselung, IP, WLAN, Laufzeit, Speicher, Modell, Version."""
        return system_status(state_file)

    return [stimme_einstellen, lautstaerke_einstellen, mikrofon_einstellen,
            gespraech_einstellen, systemstatus]
