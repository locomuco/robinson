"""Weather forecast via Open-Meteo (free, no API key)."""

import datetime
import json
import urllib.parse
import urllib.request

_WOCHENTAG = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
_CODES = {0: "klar", 1: "überwiegend klar", 2: "teilweise bewölkt", 3: "bedeckt",
          45: "Nebel", 48: "Reifnebel", 51: "leichter Nieselregen", 53: "Nieselregen",
          55: "starker Nieselregen", 56: "gefrierender Nieselregen",
          57: "gefrierender Nieselregen", 61: "leichter Regen", 63: "Regen",
          65: "starker Regen", 66: "gefrierender Regen", 67: "gefrierender Regen",
          71: "leichter Schneefall", 73: "Schneefall", 75: "starker Schneefall",
          77: "Schneegriesel", 80: "leichte Regenschauer", 81: "Regenschauer",
          82: "heftige Regenschauer", 85: "Schneeschauer", 86: "starke Schneeschauer",
          95: "Gewitter", 96: "Gewitter mit Hagel", 99: "Gewitter mit starkem Hagel"}


def _get(url: str, params: dict) -> dict:
    with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(params)}", timeout=8) as r:
        return json.load(r)


def forecast(ort: str, tage: int) -> str:
    """Forecast text for `ort`, `tage` days from today (0 = today)."""
    try:
        geo = _get("https://geocoding-api.open-meteo.com/v1/search",
                   {"name": ort, "count": 1, "language": "de", "format": "json"})
        if not geo.get("results"):
            return f"Ort {ort} nicht gefunden."
        g = geo["results"][0]
        d = _get("https://api.open-meteo.com/v1/forecast", {
            "latitude": g["latitude"], "longitude": g["longitude"], "timezone": "auto",
            "forecast_days": 8,
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                     "precipitation_probability_max,precipitation_sum,wind_speed_10m_max"})["daily"]
    except (OSError, ValueError, KeyError) as e:
        return f"Wetterdienst nicht erreichbar ({type(e).__name__})."
    if not 0 <= tage < len(d["time"]):
        return "Nur Vorhersagen für heute und die nächsten 7 Tage möglich."
    day = datetime.date.fromisoformat(d["time"][tage])
    name = {0: "heute", 1: "morgen", 2: "übermorgen"}.get(tage, _WOCHENTAG[day.weekday()])
    return (f"Wetter {name} in {g['name']}: {_CODES.get(d['weather_code'][tage], 'unbekannt')}, "
            f"{round(d['temperature_2m_min'][tage])} bis {round(d['temperature_2m_max'][tage])} Grad, "
            f"Regenwahrscheinlichkeit {d['precipitation_probability_max'][tage] or 0} Prozent, "
            f"Niederschlag {d['precipitation_sum'][tage]} mm, "
            f"Wind bis {round(d['wind_speed_10m_max'][tage])} km/h.")
