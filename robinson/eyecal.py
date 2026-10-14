"""Eye calibration: servo limits + behaviour, adjustable by voice.

Stored in eyes.json and shared by both LLM backends — the tools may run in
the MCP server process, so every change is written to disk together with a
`preview` hint; the main process reloads the file and the eye thread shows
the result (e.g. closes the adjusted lid) for a moment.

Angles are always clamped to 0..180°. The last 20 states are kept for undo.
"""

import copy
import json
import logging
import os
import re
import threading

log = logging.getLogger("robinson.eyes")

LR, UD, L_BOT, L_TOP, R_BOT, R_TOP = range(6)

DEFAULTS = {
    # gaze channels: limits and centre (values from the original eyes.py)
    "lr": {"rechts": 30, "links": 140, "mitte": 85},
    "ud": {"unten": 0, "oben": 90, "mitte": 45},
    # lid channel → [open angle, closed angle]
    "lids": {str(L_BOT): [90, 20], str(L_TOP): [0, 50], str(R_BOT): [0, 90], str(R_TOP): [50, 0]},
    "behaviour": {"blinzeln": 1.0, "umherschauen": 1.0, "tempo": 1.0, "wachheit": 1.0,
                  "einschlafen_nach_s": 300.0},
}

LID_NAMES = {"oben links": L_TOP, "unten links": L_BOT, "oben rechts": R_TOP, "unten rechts": R_BOT}
BEHAVIOUR_STEP = 1.3   # "mehr" multiplies, "weniger" divides


def _norm(s: str) -> str:
    s = s.lower().replace("ß", "ss").replace("-", " ").replace("_", " ")
    for a, b in (("obere", "oben"), ("oberes", "oben"), ("untere", "unten"), ("unteres", "unten"),
                 ("linke", "links"), ("linkes", "links"), ("rechte", "rechts"), ("rechtes", "rechts"),
                 ("lid", ""), ("augen", ""), ("auge", "")):
        s = re.sub(rf"\b{a}\b", b, s)
    return " ".join(s.split())


class EyeCalibration:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self.data = copy.deepcopy(DEFAULTS)
        self.undo: list[dict] = []
        self.preview: dict | None = None
        self.reload()
        self.preview = None   # don't replay a stale preview at startup

    # ── persistence ──────────────────────────────────────────────────────────
    def reload(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("could not read %s: %s", self.path, e)
            return
        with self._lock:
            data = copy.deepcopy(DEFAULTS)
            for k in DEFAULTS:
                data[k].update(raw.get(k, {}))
            self.data = data
            self.undo = raw.get("_undo", [])
            self.preview = raw.get("_preview")

    def _commit(self, preview: dict | None) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.preview = preview
        out = {**self.data, "_undo": self.undo[-20:], "_preview": preview}
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def _snapshot(self) -> None:
        self.undo.append(copy.deepcopy(self.data))
        self.undo = self.undo[-20:]

    def pop_preview(self) -> dict | None:
        with self._lock:
            p, self.preview = self.preview, None
            return p

    # ── accessors for the eye thread ─────────────────────────────────────────
    def lid(self, ch: int) -> tuple[float, float]:
        o, c = self.data["lids"][str(ch)]
        return float(o), float(c)

    def gaze(self, axis: str) -> dict:
        return self.data[axis]

    def b(self, key: str) -> float:
        return float(self.data["behaviour"][key])

    # ── adjustments (called by the tools) ────────────────────────────────────
    def adjust_lid(self, lid: str, position: str, aenderung: str, grad: int = 5) -> str:
        name = _norm(lid)
        chans = list(LID_NAMES.values()) if name in ("alle", "", "beide") else \
            [ch for n, ch in LID_NAMES.items() if n == name or set(n.split()) == set(name.split())]
        if not chans:
            return f"Unbekanntes Lid '{lid}'. Möglich: {', '.join(LID_NAMES)} oder alle."
        pos = _norm(position)
        if pos.startswith("geschl") or pos in ("zu",):
            idx = 1
        elif pos.startswith("off") or pos in ("auf",):
            idx = 0
        else:
            return "Position muss 'offen' oder 'geschlossen' sein."
        sign = 1 if _norm(aenderung).startswith(("weiter", "mehr")) else -1
        grad = max(1, min(30, abs(int(grad))))
        with self._lock:
            self._snapshot()
            stuck = []
            for ch in chans:
                lid_cfg = self.data["lids"][str(ch)]
                # "weiter" moves the end point further away from the other end
                away = 1 if lid_cfg[idx] >= lid_cfg[1 - idx] else -1
                new = max(0, min(180, lid_cfg[idx] + sign * away * grad))
                if new == lid_cfg[idx]:
                    stuck.append(ch)
                lid_cfg[idx] = new
            if len(stuck) == len(chans):
                self.undo.pop()
                return (f"Geht nicht: Kanal {', '.join(map(str, stuck))} steht schon am "
                        "Servo-Anschlag (0 oder 180 Grad). Das Servohorn muss mechanisch "
                        "umgesteckt werden.")
            self._commit({"kind": "lid", "channels": chans, "index": idx})
        what = "Geschlossene" if idx else "Offene"
        detail = ", ".join(f"Kanal {c}: {self.data['lids'][str(c)]}" for c in chans)
        note = (f" Kanal {', '.join(map(str, stuck))} steht am Servo-Anschlag."
                if stuck else "")
        return (f"{what} Position {'weiter' if sign > 0 else 'weniger weit'} um {grad} Grad "
                f"verstellt ({detail}).{note}")

    def adjust_gaze(self, richtung: str, aenderung: str, grad: int = 5) -> str:
        r, a = _norm(richtung), _norm(aenderung)
        grad = max(1, min(30, abs(int(grad))))
        axis = "lr" if r in ("links", "rechts") or (r == "mitte" and a in ("links", "rechts")) else "ud"
        cfg = self.data[axis]
        with self._lock:
            self._snapshot()
            if r == "mitte":
                toward = {"links": "links", "rechts": "rechts", "hoch": "oben", "oben": "oben",
                          "runter": "unten", "unten": "unten"}.get(a)
                if not toward or toward not in cfg:
                    self.undo.pop()
                    return "Die Mitte kann nach links, rechts, oben oder unten verschoben werden."
                step = grad if cfg[toward] > cfg["mitte"] else -grad
                cfg["mitte"] = max(0, min(180, cfg["mitte"] + step))
                preview = {"kind": "gaze", "axis": axis, "key": "mitte"}
            elif r in cfg:
                sign = 1 if a.startswith(("weiter", "mehr")) else -1
                away = 1 if cfg[r] > cfg["mitte"] else -1
                new = max(0, min(180, cfg[r] + sign * away * grad))
                if new == cfg[r]:
                    self.undo.pop()
                    return f"Geht nicht: {r} steht schon am Servo-Anschlag ({cfg[r]} Grad)."
                cfg[r] = new
                preview = {"kind": "gaze", "axis": axis, "key": r}
            else:
                self.undo.pop()
                return "Richtung muss links, rechts, oben, unten oder mitte sein."
            self._commit(preview)
        return f"Blick {r} angepasst: {cfg}."

    def adjust_behaviour(self, eigenschaft: str, aenderung: str) -> str:
        e, a = _norm(eigenschaft), _norm(aenderung)
        key = next((k for k in self.data["behaviour"] if k.startswith(e[:5])), None)
        if e.startswith(("schlaf", "einschl")):
            key = "einschlafen_nach_s"
        if key is None:
            return ("Eigenschaft muss blinzeln, umherschauen, tempo, wachheit oder "
                    "einschlafen sein.")
        more = a.startswith(("mehr", "öfter", "oefter", "schneller", "weiter", "später", "spaeter"))
        with self._lock:
            self._snapshot()
            v = self.data["behaviour"][key]
            v = v * BEHAVIOUR_STEP if more else v / BEHAVIOUR_STEP
            limits = {"einschlafen_nach_s": (30, 3600), "wachheit": (0.5, 1.2)}.get(key, (0.2, 4.0))
            self.data["behaviour"][key] = round(max(limits[0], min(limits[1], v)), 2)
            self._commit({"kind": "behaviour", "key": key})
        return f"{key} ist jetzt {self.data['behaviour'][key]}."

    def reset(self, was: str = "alles") -> str:
        w = _norm(was)
        parts = {"lider": ["lids"], "lid": ["lids"], "blick": ["lr", "ud"],
                 "verhalten": ["behaviour"]}.get(w, list(DEFAULTS))
        with self._lock:
            self._snapshot()
            for p in parts:
                self.data[p] = copy.deepcopy(DEFAULTS[p])
            self._commit({"kind": "all"})
        return f"Zurückgesetzt: {was}."

    def undo_last(self) -> str:
        with self._lock:
            if not self.undo:
                return "Es gibt nichts rückgängig zu machen."
            before, self.data = self.data, self.undo.pop()
            self._commit({"kind": "all"})
        changes = _describe_diff(before, self.data)
        return "Rückgängig gemacht: " + (changes or "keine Änderung") + "."


_CHANNEL_NAMES = {str(L_BOT): "unteres linkes Lid", str(L_TOP): "oberes linkes Lid",
                  str(R_BOT): "unteres rechtes Lid", str(R_TOP): "oberes rechtes Lid"}


def _describe_diff(old: dict, new: dict) -> str:
    """Human-readable 'what changed' between two calibrations (old → new)."""
    out = []
    for axis, label in (("lr", "Blick"), ("ud", "Blick")):
        for k, v in new[axis].items():
            if old[axis].get(k) != v:
                out.append(f"{label} {k} {old[axis].get(k)} → {v} Grad")
    for ch, (o, c) in new["lids"].items():
        oo, oc = old["lids"][ch]
        name = _CHANNEL_NAMES.get(ch, f"Kanal {ch}")
        if oo != o:
            out.append(f"{name} offen {oo} → {o} Grad")
        if oc != c:
            out.append(f"{name} geschlossen {oc} → {c} Grad")
    for k, v in new["behaviour"].items():
        if old["behaviour"].get(k) != v:
            out.append(f"{k} {old['behaviour'].get(k)} → {v}")
    return ", ".join(out)


def make_eye_tools(cal: EyeCalibration) -> list:
    def lid_einstellen(lid: str, position: str, aenderung: str, grad: int = 5) -> str:
        """Kalibriert ein Augenlid. Nutze das, wenn dein Mensch die Mechanik der Lider einstellen will.

        Args:
          lid: "oben links", "unten links", "oben rechts", "unten rechts" oder "alle".
          position: "geschlossen" (Stellung beim Schließen) oder "offen" (Stellung beim Öffnen).
          aenderung: "weiter" (z.B. weiter schließen / weiter öffnen) oder "weniger".
          grad: Schrittweite in Grad, normalerweise 5.
        """
        return cal.adjust_lid(lid, position, aenderung, grad)

    def blick_einstellen(richtung: str, aenderung: str, grad: int = 5) -> str:
        """Kalibriert, wie weit die Augen schauen oder wo die Mitte ist.

        Args:
          richtung: "links", "rechts", "oben", "unten" oder "mitte".
          aenderung: Bei links/rechts/oben/unten "weiter" oder "weniger"; bei mitte die Richtung, in die die Mitte verschoben wird ("links", "rechts", "hoch", "runter").
          grad: Schrittweite in Grad, normalerweise 5.
        """
        return cal.adjust_gaze(richtung, aenderung, grad)

    def verhalten_einstellen(eigenschaft: str, aenderung: str) -> str:
        """Ändert das Verhalten der Augen.

        Args:
          eigenschaft: "blinzeln" (wie oft), "umherschauen" (wie viel), "tempo" (Augenbewegung), "wachheit" (wie weit offen) oder "einschlafen" (wann die Augen bei Stille zugehen).
          aenderung: "mehr" (öfter, schneller, weiter offen, später einschlafen) oder "weniger".
        """
        return cal.adjust_behaviour(eigenschaft, aenderung)

    def augen_zuruecksetzen(was: str = "alles") -> str:
        """Setzt die Augen-Kalibrierung auf die Werkseinstellung zurück.

        Args:
          was: "alles", "lider", "blick" oder "verhalten".
        """
        return cal.reset(was)

    def augen_rueckgaengig() -> str:
        """Macht die letzte Änderung an der Augen-Kalibrierung rückgängig."""
        return cal.undo_last()

    return [lid_einstellen, blick_einstellen, verhalten_einstellen, augen_zuruecksetzen,
            augen_rueckgaengig]


EYE_TOOL_NAMES = {"lid_einstellen", "blick_einstellen", "verhalten_einstellen",
                  "augen_zuruecksetzen", "augen_rueckgaengig"}
