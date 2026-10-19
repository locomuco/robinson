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

- Each spoken sentence gets a mood from its wording ("leider" → traurig,
  "Oh, wirklich?" → überrascht, "Super!" → fröhlich …) that sets the eye
  lids / gaze style **and** the piper speaker of the emotional Thorsten voice.
- Asking it to look somewhere ("schau nach links", "zwinker mal", "mach die
  Augen zu") is recognised in what you said and executed by the eye thread.
- Eye states follow the conversation: idle wandering + blinking, wide-open
  attentive while you speak, glancing up while thinking, lively while talking,
  eyes closed after 5 min of silence.
- Talking over Robinson interrupts it (barge-in); echo cancellation keeps its
  own voice from triggering that.
- **Wake word:** it only answers when called "Robinson" (keyword spotter +
  fuzzy match in the transcript), within 20 s after its last answer, or while
  it waits for a "Ja". Other speech in the room is transcribed locally and
  dropped. "Robinson, schalte das Aktivierungswort aus" turns that off.
- **Voice:** a natural ElevenLabs voice while online (set
  `ELEVENLABS_API_KEY` in `robinson.env`; voice/model in `config.py`), piper
  otherwise or when the cloud fails. Short phrases are cached in
  `~/robinson-data/tts-cache`. "Robinson, schalte die natürliche Stimme aus"
  switches to piper.

## ReSpeaker 4-Mic Array (optional)

With the ReSpeaker 4-Mic Array HAT, Robinson records from its four mics
(USB adapter = speaker only), steers a delay-and-sum beam at the voice,
turns its eyes towards whoever speaks and shows on the 12-LED ring where it
hears speech (white: not addressed, blue: listening, purple: thinking, cyan:
speaking). RP1's I2S on the Pi 5 has no TDM, so the board's audio is read by
an RP1 PIO state machine: run `deploy/respeaker-4mic/install.sh` once and
after every kernel upgrade, then reboot. Room orientation of mics and LEDs:
`mic_front_deg`, `mic_mirror`, `led0_deg`, `led_mirror` (settings.json or
`ROBINSON_*`).

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
served to it by `python -m robinson.mcp_tools`.

**Fallback and switching.** Gemma stays loaded (and pre-warmed) in online mode.
If Claude doesn't respond within 5 s, the stream stalls, or the login fails,
that turn is answered locally ("Ich bin gerade offline …"); a background check
re-enables online once api.anthropic.com is reachable ("Ich bin wieder
online."). Both models share one conversation — a backend that fell behind
gets the turns it missed handed over, so nothing is lost.

Switch by voice (or let the model call `modell_wechseln`): "Geh online",
"Bleib offline", "Nimm Sonnet", "Wechsel auf Opus", "Zurück zu Haiku",
"Nimm das große lokale Modell" (Gemma 4 E4B, ~3.6 tok/s) / "das kleine
Gemma" (E2B, default). "Welches Modell nutzt du?" answers via
`aktuelles_modell`. The choice is saved in `~/robinson-data/state.json` and
survives restarts.
`ROBINSON_LLM` only sets the initial default. Test: `bench/test_fallback.py`
(simulates an outage by blackholing the API address).
Measured on the Pi: Haiku ~0.65 s to first word (local Gemma ~1.1 s).

This is personal use of the unmodified Claude Code binary with your own
subscription — don't share the robot's token with others; usage counts
against your normal Claude limits. For anything beyond personal use, use an
API key instead.

## Changing Robinson by voice

Settings (voice, volume, microphone, pauses, fillers, eyes, model, memory)
have their own tools and apply immediately. Restart / power off ask for a
spoken "Ja". Anything else goes through the **Wartungsmodus**:
"Robinson, Wartungsmodus: …" → Robinson repeats the job → "Ja" → a Claude
Code job (Sonnet) edits `~/robinson` on the Pi, commits only what it touched,
reads out a summary and restarts if code changed. "Wartung rückgängig" + "Ja"
reverts the last job. Optional: `ROBINSON_MAINT_PIN=…` additionally requires a
spoken PIN; `ROBINSON_MAINTENANCE=false` disables the mode.

`~/robinson` on the Pi is a git checkout the laptop pushes to
(`git push pi offline`). Voice changes are commits there — always
`git pull pi offline` before deploying so they aren't lost.

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
