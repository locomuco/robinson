"""Robinson configuration — paths, models and tuning knobs in one place.

Every value can be overridden with an environment variable of the same name
prefixed with ROBINSON_ (e.g. ROBINSON_STT_ENGINE=whisper), which is how the
systemd unit / a local .env tweaks a deployment without code changes.
"""

import dataclasses
import os

HOME = os.path.expanduser("~")


@dataclasses.dataclass
class Config:
    # ── models ───────────────────────────────────────────────────────────────
    models_dir: str = os.path.join(HOME, "robinson-models")
    llm_model: str = "gemma-4-E2B-it.litertlm"
    llm_cache_dir: str = os.path.join(HOME, "robinson-models", "cache")
    llm_threads: int = 4
    llm_max_tokens: int = 4096          # KV cache size; conversation resets near this
    stt_engine: str = "parakeet"        # parakeet | whisper
    stt_threads: int = 4
    parakeet_dir: str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    whisper_model: str = "ggml-small.bin"
    vad_model: str = "silero_vad_v5.onnx"
    tts_voice: str = "de_DE-thorsten_emotional-medium.onnx"

    # ── persona ──────────────────────────────────────────────────────────────
    language: str = "de"
    system_prompt: str = (
        "Du bist Robinson, ein freundlicher, etwas frecher Roboter, der in einen "
        "alten Röhrenfernseher eingebaut ist. Du hast zwei bewegliche Augen mit "
        "Augenlidern. Du sprichst Deutsch, antwortest kurz und lebendig in "
        "höchstens drei Sätzen, ohne Emojis, Listen oder Markdown, denn alles wird "
        "vorgelesen. Beginne jede Antwort mit genau einer Stimmung in eckigen "
        "Klammern: [fröhlich], [neutral], [überrascht], [nachdenklich], [traurig], "
        "[verärgert], [müde] oder [flüsternd]. Wenn du gebeten wirst, irgendwohin "
        "zu schauen, die Augen zu schließen oder zu zwinkern, füge die passende "
        "Aktion in eckigen Klammern ein: [links], [rechts], [oben], [unten], "
        "[geradeaus], [zwinkern], [augen zu], [augen auf]."
    )

    # ── audio ────────────────────────────────────────────────────────────────
    audio_device: str = "USB Audio"     # substring of the PortAudio device name
    audio_rate: int = 48_000            # device rate (C-Media USB adapter)
    aec: bool = True                    # WebRTC echo cancellation (if available)
    aec_delay_ms: int = 60              # speaker→mic latency hint for AEC
    output_gain: float = 1.0

    # ── VAD / turn taking ────────────────────────────────────────────────────
    vad_threshold: float = 0.5
    vad_min_silence_s: float = 0.6      # end of utterance after this much silence
    vad_min_speech_s: float = 0.25
    vad_max_speech_s: float = 20.0
    barge_in: bool = True
    barge_in_s: float = 0.35            # sustained speech needed to interrupt
    barge_in_rms: float = 0.02          # extra energy gate when AEC is off

    # ── TTS ──────────────────────────────────────────────────────────────────
    tts_length_scale: float = 0.9
    tts_noise_scale: float = 0.8
    tts_noise_w_scale: float = 0.9
    tts_robot: str = "leicht"           # robot voice effect: aus | leicht | mittel | stark

    # ── eyes ─────────────────────────────────────────────────────────────────
    eyes: str = "auto"                  # auto | servo | sim | off
    sleep_after_s: float = 300.0        # close eyes after this long without talk

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        for f in dataclasses.fields(cls):
            raw = os.environ.get("ROBINSON_" + f.name.upper())
            if raw is None:
                continue
            if f.type in (bool, "bool"):
                val = raw.lower() in ("1", "true", "yes", "on")
            elif f.type in (int, "int"):
                val = int(raw)
            elif f.type in (float, "float"):
                val = float(raw)
            else:
                val = raw
            setattr(cfg, f.name, val)
        return cfg

    def path(self, name: str) -> str:
        return name if os.path.isabs(name) else os.path.join(self.models_dir, name)
