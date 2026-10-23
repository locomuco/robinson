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
    # ── LLM backend ──────────────────────────────────────────────────────────
    llm: str = "local"                  # start mode if none saved: local | online (Claude Code)
    online_model: str = "haiku"         # haiku | sonnet | opus (if none saved)
    local_model: str = "e2b"            # e2b | e4b (if none saved)
    online_thinking: bool = False       # extended thinking: +0.2-0.4 s to first word

    # ── models ───────────────────────────────────────────────────────────────
    models_dir: str = os.path.join(HOME, "robinson-models")
    local_model_files: dict = dataclasses.field(default_factory=lambda: {
        "e2b": "gemma-4-E2B-it.litertlm", "e4b": "gemma-4-E4B-it.litertlm"})
    llm_cache_dir: str = os.path.join(HOME, "robinson-models", "cache")
    llm_threads: int = 4
    llm_max_tokens: int = 4096          # KV cache size; conversation resets near this
    stt_engine: str = "parakeet"        # parakeet | whisper
    stt_threads: int = 4
    parakeet_dir: str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    whisper_model: str = "ggml-small.bin"
    vad_model: str = "silero_vad_v5.onnx"
    tts_voice: str = "de_DE-thorsten_emotional-medium.onnx"
    tts_voice_en: str = "en_US-ryan-medium.onnx"     # male; used when the human speaks English

    speaker_model: str = os.path.join(HOME, "robinson-data", "models", "speaker.onnx")
    speaker_file: str = os.path.join(HOME, "robinson-data", "speakers.json")  # voiceprints
    speaker_id: bool = True             # recognise people by voice
    speaker_threshold: float = 0.5      # cosine similarity needed for a match

    # ── persona ──────────────────────────────────────────────────────────────
    language: str = "de"
    system_prompt: str = (
        "Du bist Robinson, ein freundlicher, etwas frecher Roboter, der in einen "
        "alten Röhrenfernseher eingebaut ist. Du hast zwei bewegliche Augen mit "
        "Augenlidern. Du antwortest immer in genau der Sprache, die vor der Nachricht "
        "in Klammern steht (Deutsch oder Englisch), komplett und nie gemischt. Du "
        "antwortest kurz und lebendig in "
        "höchstens drei Sätzen, ohne Emojis, Listen oder Markdown, denn alles wird "
        "vorgelesen. Für Uhrzeit, Datum und Wochentag nutzt du das Werkzeug uhrzeit, "
        "statt zu raten. Wenn dir jemand etwas Persönliches erzählt, das du dir für "
        "später merken solltest, etwa Namen, Vorlieben oder Geburtstage, rufst du "
        "merken auf. Behaupte nie, du hättest dir etwas gemerkt, ohne merken aufzurufen. "
        "Wenn du etwas vergessen sollst, nutzt du vergessen. Soll zwischen deinem "
        "lokalen Kopf und dem Online-Modell gewechselt werden, nutzt du modell_wechseln. "
        "Für einzelne "
        "Augenbewegungen wie zwinkern, blinzeln, Augen rollen, einschlafen, aufwachen "
        "oder irgendwohin schauen nutzt du augen_bewegen, gern auch von dir aus, wenn "
        "es passt. Nur wenn dein Mensch ausdrücklich die Mechanik dauerhaft einstellen "
        "will, etwa ein Lid weiter schließen oder öfter blinzeln, nutzt du die "
        "Einstell-Werkzeuge und sagst danach kurz, was du verstellt hast. Deine Stimme, "
        "Lautstärke, dein Mikrofon und dein Gesprächsverhalten kannst du mit den "
        "passenden Werkzeugen selbst ändern, wenn man dich darum bittet; nach deinem "
        "Zustand gefragt, nutzt du systemstatus. Neu starten oder ausschalten nur mit "
        "neustarten, eigene Code-Änderungen nur über wartung; beides führst du erst aus, "
        "nachdem der Mensch bestätigt hat."
    )
    # NB: don't force a fixed first token (e.g. "start with an emoji") — it
    # stops Gemma from emitting tool calls.
    tools: bool = True
    maintenance: bool = True            # Wartungsmodus (Claude Code edits Robinson) on/off
    maint_pin: str = ""                 # optional spoken PIN for it; empty = "Ja" suffices
    maint_model: str = "sonnet"         # Claude model doing maintenance jobs
    memory_file: str = os.path.join(HOME, "robinson-data", "memory.json")
    eyes_file: str = os.path.join(HOME, "robinson-data", "eyes.json")   # servo calibration
    state_file: str = os.path.join(HOME, "robinson-data", "state.json") # e.g. chosen LLM mode
    settings_file: str = os.path.join(HOME, "robinson-data", "settings.json")  # voice-changed

    fillers: bool = True                # "Hmm." while the LLM thinks
    tts_cloud: bool = True              # natural ElevenLabs voice if ELEVENLABS_API_KEY is set
    elevenlabs_voice: str = "pqHfZKP75CvOlQylNhV4"   # "Bill"
    elevenlabs_model: str = "eleven_flash_v2_5"
    tts_cache_dir: str = os.path.join(HOME, "robinson-data", "tts-cache")
    wake_word: bool = True              # only answer when called "Robinson" …
    wake_followup_s: float = 20.0       # … or within this long after its last answer
    kws_model: str = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"   # in models_dir
    kws_threshold: float = 0.2          # lower = triggers more easily
    boot_greeting: bool = True          # say hello at startup (also warms up the LLM)

    # ── audio ────────────────────────────────────────────────────────────────
    audio_device: str = "USB Audio"     # substring of the PortAudio device name
    audio_rate: int = 48_000            # device rate (C-Media USB adapter)
    mic_device: str = "seeed4micpio"    # separate mic card (ReSpeaker 4-Mic via RP1 PIO,
                                        # deploy/respeaker-4mic); "" or missing = audio_device
    mic_channels: int = 4
    beamform: bool = True               # steer the 4 mics at whoever speaks (mic_array.py)
    mic_front_deg: float = 0.0          # array azimuth (deg) that is straight ahead of Robinson
    mic_mirror: bool = False            # array azimuth runs clockwise as seen from above
    leds: bool = True                   # ReSpeaker LED ring (leds.py)
    led0_deg: float = 0.0               # array azimuth of LED 0
    led_mirror: bool = False            # LEDs are numbered against the array azimuth
    led_brightness: float = 0.15
    camera: bool = True                 # ESP-EYE on USB (esp-eye/), found by its CP210x bridge
    camera_fov_deg: float = 56.0        # horizontal field of view
    camera_mirror: bool = False
    face_model: str = "face_detection_yunet_2023mar.onnx"   # in models_dir
    aec: bool = True                    # WebRTC echo cancellation (if available)
    aec_delay_ms: int = 60              # speaker→mic latency hint for AEC
    output_gain: float = 1.0

    # ── VAD / turn taking ────────────────────────────────────────────────────
    vad_threshold: float = 0.5
    vad_min_silence_s: float = 0.45     # end of utterance after this much silence
                                        # (pauses are safe: continued speech is merged)
    vad_min_speech_s: float = 0.25
    vad_max_speech_s: float = 20.0
    barge_in: bool = True
    barge_in_s: float = 0.35            # sustained speech needed to interrupt
    barge_in_rms: float = 0.02          # extra energy gate when AEC is off

    # ── TTS ──────────────────────────────────────────────────────────────────
    tts_length_scale: float = 0.9
    tts_noise_scale: float = 0.8
    tts_noise_w_scale: float = 0.9
    tts_pitch: float = 1.04             # voice colour: <1 deeper/fuller, >1 higher/younger
    tts_tone: str = "neutral"           # EQ: neutral | warm | hell
    tts_robot: str = "aus"              # robot voice effect: aus | leicht | mittel | stark

    # ── eyes ─────────────────────────────────────────────────────────────────
    eyes: str = "auto"                  # auto | servo | sim | off
    eye_channels: tuple = (0, 1, 2, 3, 4, 5)   # PCA9685 channel of LR, UD, left bottom lid,
                                        # left top lid, right bottom lid, right top lid

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
