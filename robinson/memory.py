"""Long-term memory: small facts Robinson learns about its people.

Facts live in a JSON file and are injected into the system prompt at
startup (prefilled during boot, so recalling them costs no latency). New
facts arrive through the `merken` tool and are also in the running
conversation, so they're usable immediately.
"""

import datetime
import json
import logging
import os
import re
import threading

log = logging.getLogger("robinson.memory")

# first-person statements worth keeping, recognised directly in what was said —
# Gemma 4 E2B often claims "ich habe es notiert" without calling the tool
_PERSONAL = re.compile(
    r"\b(ich heiße|mein name ist|ich bin \d+ jahre|ich wohne|ich arbeite|ich mag|"
    r"ich liebe|ich hasse|ich esse (gern|kein)|mein(e|en)? lieblings\w*|"
    r"(hat|habe) am .{1,20} geburtstag|meine? (frau|mann|tochter|sohn|kinder|"
    r"freundin|freund|katze|hund) heißt)", re.I)
_MERK_DIR = re.compile(r"\bmerk(e)? dir[\s,:]*(bitte[\s,:]*)?(?P<dass>dass\s+)?(?P<fact>.+)", re.I)
_VERGISS = re.compile(r"\bvergiss[\s,]*(bitte[\s,]*)?(alles\s+)?(das|was)?\s*(mit|über|von)?\s*"
                      r"(den|dem|der|die|das|meine[mnr]?|mein)?\s*(?P<what>[\wäöüß ]+)", re.I)
# meta statements / questions the model sometimes tries to store
_JUNK = re.compile(r"\?$|\b(benutzer|nutzer) (möchte|fragt|will)|\bwissen\b.*\bname", re.I)


def personal_fact(utterance: str) -> str | None:
    """A fact to store verbatim from the user's own words, or None."""
    u = utterance.strip()
    if m := _MERK_DIR.search(u):
        fact = m.group("fact").strip().rstrip(".!")
        return (f"Ich soll mir merken, dass {fact}" if m.group("dass")
                else f"Ich soll mir merken: {fact}")
    if "?" in u:
        return None
    for sentence in re.split(r"(?<=[.!])\s+", u):
        if _PERSONAL.search(sentence):
            return f"Der Mensch sagte: „{sentence.strip().rstrip('.!')}“"
    return None


def forget_request(utterance: str) -> str | None:
    """Keyword(s) of what to forget ("Vergiss das mit den Pilzen" → "Pilzen")."""
    m = _VERGISS.search(utterance)
    what = m.group("what").strip() if m else ""
    if not what or what.lower() in {"das", "es", "alles", "was", "dies", "ihn", "sie"}:
        return None   # too vague to delete anything safely
    return what


class Memory:
    def __init__(self, path: str, max_facts: int = 60, on_forget=None):
        self.path = path
        self.on_forget = on_forget        # called after facts were deleted
        self.max_facts = max_facts
        self._lock = threading.Lock()
        self.facts: list[dict] = []
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self.facts = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                log.warning("could not read %s: %s", path, e)
        log.info("memory: %d facts", len(self.facts))

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.facts, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def add(self, fact: str) -> str:
        fact = fact.strip().rstrip(".")
        if not fact or _JUNK.search(fact):
            return "Das ist kein Fakt zum Merken."
        with self._lock:
            if any(f["fakt"].lower() == fact.lower() for f in self.facts):
                return "Das weiß ich schon."
            self.facts.append({"fakt": fact, "seit": datetime.date.today().isoformat()})
            self.facts = self.facts[-self.max_facts:]
            self._save()
        log.info("remembered: %s", fact)
        return f"Gemerkt: {fact}."

    def forget(self, keyword: str) -> str:
        words = [w[:max(4, len(w) - 2)] for w in re.findall(r"\w+", keyword.lower()) if len(w) > 2]
        if not words:
            return "Was genau soll ich vergessen?"

        def matches(fact: str) -> bool:
            f = fact.lower()
            # "Namen" should also hit "ich heiße …"
            return all(w in f or (w.startswith("name") and "heiße" in f) for w in words)

        with self._lock:
            keep = [f for f in self.facts if not matches(f["fakt"])]
            gone = len(self.facts) - len(keep)
            if gone:
                self.facts = keep
                self._save()
        log.info("forgot %d fact(s) matching %r", gone, keyword)
        if gone and self.on_forget:
            self.on_forget()
        return f"{gone} Erinnerung(en) gelöscht." if gone else "Dazu habe ich nichts gespeichert."

    def prompt_section(self) -> str:
        with self._lock:
            if not self.facts:
                return ""
            lines = "\n".join(f"- {f['fakt']}" for f in self.facts)
        return f"\n\nWas du dir gemerkt hast:\n{lines}"
