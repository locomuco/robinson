"""Risky actions by voice: restart/reboot/power off and Wartungsmodus.

The tools only *request* an action; the main process registers it as
pending (on_tool callback) and runs it after the human confirms with "Ja"
(+ a PIN for maintenance, if one is configured) in their next utterance.
"""

import json
import os
import re

_DIGIT_WORDS = {"null": "0", "eins": "1", "ein": "1", "eine": "1", "zwei": "2", "zwo": "2",
                "drei": "3", "vier": "4", "fünf": "5", "fuenf": "5", "sechs": "6",
                "sieben": "7", "acht": "8", "neun": "9"}
_YES = re.compile(r"\b(ja|jawohl|jep|genau|bestätigt|bestätige|ausführen|mach (es|das)|"
                  r"los|okay|ok|yeah|yes|yep)\b")
_NO = re.compile(r"\b(nein|nö|abbrechen|abbruch|stopp?|lieber nicht|doch nicht)\b")
_MAINT = re.compile(r"^(robinson\W*)?wartung(smodus|sauftrag)?\W+(?P<task>\w.{5,})$", re.I)
_RESTART = [
    # power off needs an explicit object: "schalt die Füllwörter aus" must not match
    (re.compile(r"\b(schalt\w*|fahr\w*) (\w+ )?(dich|den pi|den rechner|robinson) (\w+ )?"
                r"(aus|runter|herunter)\b"), "ausschalten"),
    (re.compile(r"\b(start\w*|boot\w*) (\w+ ){0,2}(pi|rechner|raspberry|computer) (\w+ )?neu\b"
                r"|\breboot"), "pi"),
    (re.compile(r"\b(start\w*) (\w+ ){0,2}(dich|robinson|die software) (\w+ )?neu\b"), "robinson"),
]
_MAINT_UNDO = re.compile(r"\bwartung\w*\s+(\w+\s+)?(rückgängig|zurück\w*)", re.I)

RESTART_TARGETS = {"robinson": "Robinson neu starten", "pi": "den Pi neu starten",
                   "ausschalten": "den Pi ausschalten"}


def spoken_digits(text: str) -> str:
    """All digits in an utterance, also spoken ones: 'eins zwei drei vier' → '1234'."""
    out = []
    for tok in re.findall(r"\d+|[a-zäöüß]+", text.lower()):
        out.append(tok if tok.isdigit() else _DIGIT_WORDS.get(tok, ""))
    return "".join(out)


def confirmation(text: str) -> str:
    """'yes', 'no' or 'other' for the answer to a confirmation question."""
    t = text.lower()
    if _NO.search(t):
        return "no"
    return "yes" if _YES.search(t) else "other"


def maintenance_request(text: str) -> str | None:
    m = _MAINT.match(text.strip())
    return m.group("task").strip() if m else None


def restart_request(text: str) -> str | None:
    t = text.lower()
    if "?" in t:
        return None
    return next((what for rx, what in _RESTART if rx.search(t)), None)


def maintenance_undo_request(text: str) -> bool:
    return bool(_MAINT_UNDO.search(text))


def make_control_tools(state_dir: str, enabled: bool, pin_set: bool = False) -> list:
    confirm = "ein Ja zusammen mit der PIN" if pin_set else "ein Ja"

    def neustarten(was: str) -> str:
        """Startet Robinson oder den Raspberry Pi neu oder schaltet ihn aus. Ruf es sofort auf, ohne vorher selbst nachzufragen – es fragt selbst nach einem "Ja".

        Args:
          was: "robinson" (nur die Software), "pi" (ganzer Rechner) oder "ausschalten".
        """
        w = was.lower()
        key = "ausschalten" if "aus" in w else "pi" if ("pi" in w or "rechner" in w) else "robinson"
        return (f"Bestätigung nötig. Frag den Menschen kurz, ob du wirklich "
                f"{RESTART_TARGETS[key]} sollst, und bitte um ein Ja.")

    def wartung(auftrag: str) -> str:
        """Gibt einen Wartungsauftrag an Claude Code, das deinen eigenen Programmcode ändern kann (neue Funktionen, Verhalten ändern, Fehler beheben). Ruf es sofort auf, ohne vorher selbst nachzufragen – danach braucht es ein "Ja".

        Args:
          auftrag: Der Auftrag in den Worten des Menschen, möglichst vollständig.
        """
        if not enabled:
            return "Der Wartungsmodus ist ausgeschaltet."
        return ("Bestätigung nötig. Wiederhole den Auftrag in einem Satz und bitte den "
                f"Menschen um {confirm}.")

    def wartung_rueckgaengig() -> str:
        """Nimmt den letzten Wartungsauftrag zurück. Braucht danach ein "Ja"."""
        if not enabled:
            return "Der Wartungsmodus ist ausgeschaltet."
        return f"Bestätigung nötig. Bitte den Menschen um {confirm}."

    def wartung_status() -> str:
        """Sagt, ob gerade ein Wartungsauftrag läuft und welcher zuletzt lief."""
        try:
            with open(os.path.join(state_dir, "maintenance.json")) as f:
                st = json.load(f)
        except (OSError, json.JSONDecodeError):
            return "Es gab noch keinen Wartungsauftrag."
        if st.get("running"):
            return f"Ich arbeite gerade am Wartungsauftrag: {st['running']}."
        return f"Letzter Auftrag: {st.get('task')}. Ergebnis: {st.get('summary', '')[:200]}"

    return [neustarten, wartung, wartung_rueckgaengig, wartung_status]
