"""Tools Gemma can call (LiteRT-LM automatic tool calling).

Signatures and docstrings are the tool schema the model sees, so they are
German and short. Note: tool calls only work when the system prompt does
not force a fixed first token (e.g. "start with an emoji").
"""

import datetime

from .eyecal import EyeCalibration, make_eye_tools
from .eyes import ACTIONS, normalize_action
from .memory import Memory

_WOCHENTAG = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
_MONAT = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
          "September", "Oktober", "November", "Dezember"]


def make_tools(memory: Memory, eyes: EyeCalibration | None = None) -> list:
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

    return [uhrzeit, merken, vergessen, augen_bewegen, *(make_eye_tools(eyes) if eyes else [])]
