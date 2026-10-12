# Robinson 🤖📺

A friendly retro robot built into an old CRT TV, inspired by
https://hackaday.io/project/189041-a-workbench-companion-from-an-amazon-echo-dot.
Robinson talks German, runs **fully offline** on a Raspberry Pi 5 and moves
its eyes (6 servos on a PCA9685).

```
mic ─► WebRTC AEC ─► Silero VAD ─► Parakeet v3 (STT) ─► Gemma 4 E2B (LiteRT-LM) ─► piper ─► speaker
                        │                                     │ [mood] / [action] tags
                        └── speech while talking = barge-in   └──► eyes thread (PCA9685)
```

| Stage | Choice | Pi 5 (16 GB) measured |
|---|---|---|
| VAD | Silero v5 via sherpa-onnx | < 1 ms / frame |
| STT | Parakeet TDT 0.6B v3 int8 (sherpa-onnx), in-process | 0.58 s per utterance, 7 % WER (synthetic German set) |
| LLM | Gemma 4 E2B, LiteRT-LM CPU + YNNPACK | 166 tok/s prefill, 10.5 tok/s decode, ~1.1 s to first token |
| TTS | piper-tts 1.8 (thorsten_emotional), in-process | RTF 0.13 |
| AEC | WebRTC APM (livekit.rtc) on the duplex stream | |
| Eyes | adafruit-circuitpython-servokit on Blinka / rpi-lgpio | |

Hardware: Raspberry Pi 5 (8 GB+), NVMe SSD, USB audio adapter (mic + speaker),
PCA9685 servo board on I2C, **active cooling** — without it the SoC hits
85 °C+ within minutes of LLM load and drops to 1.5 GHz.

## Install

On a fresh Raspberry Pi OS Lite (trixie, 64-bit):

```bash
git clone https://github.com/locomuco/robinson && cd robinson
./setup.sh                      # deps, uv env, ~3.5 GB of models, I2C, systemd unit
sudo systemctl start robinson   # starts on boot from now on
journalctl -u robinson -f
```

Debug without talking: `uv run robinson --text` (type, Robinson speaks) or
`uv run robinson --text --no-audio` (pure chat in the terminal).

## How it behaves

- Gemma starts each reply with a mood tag — `[fröhlich]`, `[überrascht]`,
  `[müde]` … — that sets the eye lids / gaze style **and** the piper speaker
  of the emotional Thorsten voice. Tags never reach the TTS.
- Asking it to look somewhere ("schau nach links", "zwinker mal", "mach die
  Augen zu") makes it emit action tags that the eye thread executes.
- Eye states follow the conversation: idle wandering + blinking, wide-open
  attentive while you speak, glancing up while thinking, lively while talking,
  eyes closed after 5 min of silence.
- Talking over Robinson interrupts it (barge-in); echo cancellation keeps its
  own voice from triggering that.

## Local or online brain

Robinson runs fully offline on Gemma 4 E2B by default. It can instead use
Claude through **Claude Code with your own Claude subscription** — same
persona, same tools (`uhrzeit`, `merken`, `vergessen`), same `memory.json`:

```bash
curl -fsSL https://claude.ai/install.sh | bash        # on the Pi
claude setup-token                                    # on a machine with a browser
# robinson.env on the Pi (chmod 600):
#   CLAUDE_CODE_OAUTH_TOKEN=…
#   ROBINSON_LLM=online
#   ROBINSON_ONLINE_MODEL=haiku                       # or sonnet
sudo systemctl restart robinson
```

One `claude -p` process stays alive in stream-json mode; Robinson's tools are
served to it by `python -m robinson.mcp_tools`. If the online backend can't
start (no network, bad token) Robinson falls back to the local model.
Measured on the Pi: Haiku ~0.65 s to first word (local Gemma ~1.1 s).

This is personal use of the unmodified Claude Code binary with your own
subscription — don't share the robot's token with others; usage counts
against your normal Claude limits. For anything beyond personal use, use an
API key instead.

## Configuration

All knobs live in `robinson/config.py`; override any of them with
`ROBINSON_<NAME>` in `robinson.env` next to this README, e.g.

```
ROBINSON_STT_ENGINE=whisper       # needs: uv sync --extra whisper
ROBINSON_EYES=sim                 # no servos attached
ROBINSON_AEC=false
```

## Benchmarks

`bench/bench_litert.py` (LLM configs, thermal log) and `bench/bench_stt.py`
(German STT bake-off: Parakeet vs whisper.cpp vs Gemma audio).
