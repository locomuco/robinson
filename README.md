# Robinson 🤖

Minimal local voice assistant for Raspberry Pi 5.

```
Mic → VAD → whisper.cpp (STT) → ollama (LLM) → piper (TTS) → Speaker
```

## Quick start

```bash
./setup.sh        # ~10 min first run (builds whisper.cpp, downloads models)
uv run python main.py
```

## Pipeline

| Stage | Tool | Model / Notes |
|-------|------|---------------|
| Voice Activity Detection | energy RMS | no extra deps |
| Speech-to-Text | whisper.cpp | `base.en` (~150 MB) |
| LLM | ollama | `qwen2.5:1.5b` (~1 GB, fast on Pi 5) |
| Text-to-Speech | piper | `en_US-lessac-medium` (~60 MB) |

## Tuning

Edit the constants at the top of `main.py`:

| Constant | Default | Effect |
|----------|---------|--------|
| `ENERGY_THRESHOLD` | `0.015` | Raise if background noise triggers false STT |
| `SILENCE_FRAMES` | `25` | Frames of quiet before utterance ends (~750 ms) |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Swap to `llama3.2:3b` for better quality |
| `WHISPER_MODEL` | `base.en` | `tiny.en` for speed, `small.en` for accuracy |

## Swapping models

```bash
# Faster whisper (less accurate)
bash whisper.cpp/models/download-ggml-model.sh tiny.en
# Change WHISPER_MODEL in main.py to "whisper.cpp/models/ggml-tiny.en.bin"

# Better LLM
ollama pull llama3.2:3b
# Change OLLAMA_MODEL in main.py
```

## Troubleshooting

```bash
# List audio devices
uv run python -c "import sounddevice; print(sounddevice.query_devices())"

# Test microphone
uv run python -c "
import sounddevice as sd, numpy as np
print('Recording 3s …')
a = sd.rec(48000, samplerate=16000, channels=1)
sd.wait()
print('Peak:', np.abs(a).max())
"

# Test piper
echo 'Hello from Robinson' | ./piper/piper --model piper/en_US-lessac-medium.onnx --output-raw | aplay -r 22050 -f S16_LE -c 1

# Test ollama
curl http://localhost:11434/api/tags
```
