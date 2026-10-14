"""Tools Gemma can call (LiteRT-LM automatic tool calling).

Signatures and docstrings are the tool schema the model sees, so they are
German and short. Note: tool calls only work when the system prompt does
not force a fixed first token (e.g. "start with an emoji").
"""

import datetime
import json
import os

from .eyecal import EyeCalibration, make_eye_tools
from .eyes import ACTIONS, normalize_action
from .memory import Memory

_WOCHENTAG = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
_MONAT = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
          "September", "Oktober", "November", "Dezember"]


def make_tools(memory: Memory, eyes: EyeCalibration | None = None, status=None) -> list:
    def uhrzeit() -> str:
        """Gibt die aktuelle Uhrzeit, den Wochentag und das Datum zurück."""
        n = datetime.datetime.now()
        return (f"{_WOCHENTAG[n.weekday()]}, {n.day}. {_MONAT[n.month - 1]} {n.year}, "
                f"{n.hour}:{n.minute:02d} Uhr")

    def merken(fakt: str) -> str:
        """Speichert dauerhaft etwas, das dir ein Mensch über sich erzählt hat (Name, Vorliebe, Geburtstag). Nicht für Fragen.

        Args:
          fakt: Der Fakt als kurzer Satz in der dritten Person, z.B. "Anna mag keine Pilze".
        """
        return memory.add(fakt)

    def vergessen(stichwort: str) -> str:
        """Löscht gespeicherte Erinnerungen, die zum Stichwort passen.

        Args:
          stichwort: Worum es geht, z.B. "Pilze".
        """
        return memory.forget(stichwort)

    def augen_bewegen(aktion: str, auge: str = "") -> str:
        """Führt sofort eine einzelne Augenbewegung aus, ohne etwas dauerhaft zu verstellen.

        Args:
          aktion: "zwinkern", "blinzeln", "augen rollen", "schlafen", "aufwachen", "große augen", "skeptisch", "augen zu", "augen auf", "links", "rechts", "oben", "unten" oder "geradeaus".
          auge: Beim Zwinkern "links" oder "rechts", sonst leer.
        """
        name = normalize_action(aktion, auge)
        if name is None:
            return f"Unbekannte Bewegung. Möglich: {', '.join(sorted(ACTIONS))}."
        return f"Ausgeführt: {name}."   # the main process performs it (on_tool callback)

    def modell_wechseln(ziel: str) -> str:
        """Wechselt das Modell, mit dem du denkst. Gilt ab der nächsten Antwort.

        Args:
          ziel: "lokal", "online", "haiku" (schnell), "sonnet" (klüger), "opus" (am stärksten, langsam), "gemma klein" (lokal, schnell) oder "gemma groß" (lokal, langsam).
        """
        from .llm_router import parse_target
        if parse_target(ziel) is None:
            return "Unbekanntes Ziel. Möglich: lokal, online, haiku, sonnet, opus, gemma klein, gemma groß."
        return f"Wechsel auf {ziel} angefordert."

    def aktuelles_modell() -> str:
        """Sagt, mit welchem Modell du gerade denkst (lokal oder online, welches Modell)."""
        if status is not None:
            return status()
        try:   # MCP server process: report the saved choice
            with open(os.environ.get("ROBINSON_STATE_FILE", "")) as f:
                st = json.load(f)
            return f"Modus {st.get('llm')}, online {st.get('online_model')}, lokal {st.get('local_model')}."
        except (OSError, json.JSONDecodeError):
            return "Unbekannt."

    return [uhrzeit, merken, vergessen, augen_bewegen, modell_wechseln, aktuelles_modell,
            *(make_eye_tools(eyes) if eyes else [])]
