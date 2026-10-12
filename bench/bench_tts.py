#!/usr/bin/env python3
"""German TTS comparison on the Pi: piper voices vs Supertonic 3.

Measures real-time factor (synthesis time / audio duration; < 0.5 keeps
sentence N+1 ready while N plays) and writes one sample WAV per voice that
introduces itself, for listening on Robinson's speaker (`--play`).
"""

import argparse
import os
import subprocess
import time
import wave

import numpy as np

TEXT = ("Hallo, ich bin Robinson, der Roboter in deinem alten Fernseher. "
        "Heute Abend gibt es Bratkartoffeln mit Spiegelei, und danach erzähle ich dir einen Witz.")
VOICES = os.path.expanduser("~/stt/voices")
OUT = os.path.expanduser("~/stt/tts-samples")


def write_wav(path: str, audio: np.ndarray, rate: int) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def piper_voice(name: str, speaker: int | None = None):
    from piper import PiperVoice, SynthesisConfig
    v = PiperVoice.load(os.path.join(VOICES, name + ".onnx"))
    cfg = SynthesisConfig(speaker_id=speaker) if speaker is not None else None

    def synth(text):
        chunks = list(v.synthesize(text, syn_config=cfg))
        return np.concatenate([c.audio_float_array for c in chunks]), chunks[0].sample_rate
    return synth


def supertonic_voice(tts, style: str, steps: int):
    st = tts.get_voice_style(style)

    def synth(text):
        wav, _dur = tts.synthesize(text, voice_style=st, total_steps=steps, lang="de")
        return np.asarray(wav, np.float32).reshape(-1), tts.sample_rate
    return synth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    candidates = [
        ("Piper Thorsten emotional (aktuell)", piper_voice("de_DE-thorsten_emotional-medium", 4)),
        ("Piper Thorsten high", piper_voice("de_DE-thorsten-high")),
    ]
    import supertonic
    tts = supertonic.TTS(intra_op_num_threads=4)
    for s in ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]:
        candidates.append((f"Supertonic {s}", supertonic_voice(tts, s, args.steps)))

    for i, (label, synth) in enumerate(candidates, 1):
        synth("Warmlaufen.")
        intro = f"Stimme {i}: {label.replace('(aktuell)', '')}. "
        t0 = time.monotonic()
        audio, rate = synth(TEXT)
        dt = time.monotonic() - t0
        dur = len(audio) / rate
        print(f"{i:2d} {label:38s} RTF {dt / dur:.2f}  ({dt:.2f}s for {dur:.1f}s audio)", flush=True)
        a_intro, r_intro = synth(intro)
        assert r_intro == rate
        path = os.path.join(OUT, f"{i:02d}.wav")
        write_wav(path, np.concatenate([a_intro, np.zeros(int(0.3 * rate)), audio,
                                        np.zeros(int(0.8 * rate))]).astype(np.float32), rate)
        if args.play:
            subprocess.run(["aplay", "-q", "-D", "plughw:0,0", path])


if __name__ == "__main__":
    main()
