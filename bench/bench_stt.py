#!/usr/bin/env python3
"""German STT bake-off for Robinson on the Pi 5.

Generates a German test set with several piper voices (+ light noise), then
transcribes it with each engine and reports WER and latency. Engines are
loaded once (as they would be in the robot) and the load time is reported
separately from per-utterance latency.

Engines: parakeet (sherpa-onnx, Parakeet TDT 0.6B v3 int8), whisper-small /
whisper-base (pywhispercpp), gemma (Gemma 4 E2B native audio via LiteRT-LM).
"""

import argparse
import glob
import json
import os
import re
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 16_000

SENTENCES = [
    "Hallo Robinson, wie geht es dir heute?",
    "Kannst du mir sagen, wie spät es ist?",
    "Erzähl mir bitte einen Witz über Roboter.",
    "Wie wird das Wetter morgen in München?",
    "Ich habe heute Abend keine Lust zu kochen.",
    "Mach bitte die Augen zu und schlaf ein bisschen.",
    "Wie viele Einwohner hat Berlin ungefähr?",
    "Stell einen Timer auf fünfzehn Minuten.",
    "Was ist der Unterschied zwischen einem Kühlschrank und einer Gefriertruhe?",
    "Meine Tochter heißt Lena und wird nächste Woche sieben Jahre alt.",
    "Schau mal nach links, da steht jemand.",
    "Kannst du mir ein Rezept mit Kartoffeln und Zwiebeln empfehlen?",
    "Wer hat eigentlich das Fernsehen erfunden?",
    "Ich finde deinen alten Röhrenfernseher wirklich schön.",
    "Bitte sprich etwas langsamer, ich habe dich nicht verstanden.",
    "Wie lange braucht man mit dem Fahrrad von Hamburg nach Bremen?",
    "Erkläre mir in zwei Sätzen, wie ein Elektromotor funktioniert.",
    "Das war lustig, erzähl noch einen.",
    "Gute Nacht Robinson, bis morgen früh.",
    "Wie viel ist zweihundertdreiundvierzig plus achtzehn?",
]

VOICES = [
    ("de_DE-thorsten_emotional-medium", 4),  # neutral speaker of the emotional set
    ("de_DE-kerstin-low", None),
    ("de_DE-eva_k-x_low", None),
    ("de_DE-ramona-low", None),
    ("de_DE-karlsson-low", None),
]


# ── test set ─────────────────────────────────────────────────────────────────

def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(src, dst)
    return resample_poly(x, dst // g, src // g)


def make_testset(voice_dir: str, out_dir: str, snr_db: float) -> list[dict]:
    from piper import PiperVoice, SynthesisConfig

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    items, tts_stats = [], []
    voices = {}
    for i, text in enumerate(SENTENCES):
        name, speaker = VOICES[i % len(VOICES)]
        if name not in voices:
            voices[name] = PiperVoice.load(os.path.join(voice_dir, name + ".onnx"))
        voice = voices[name]
        cfg = SynthesisConfig(speaker_id=speaker) if speaker is not None else None
        t0 = time.time()
        chunks = list(voice.synthesize(text, syn_config=cfg))
        synth_s = time.time() - t0
        rate = chunks[0].sample_rate
        audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        tts_stats.append({"voice": name, "rtf": synth_s / (len(audio) / rate)})
        audio = _resample(audio, rate, SR).astype(np.float32)
        # 300 ms silence padding + white noise at the requested SNR
        pad = np.zeros(int(0.3 * SR), np.float32)
        audio = np.concatenate([pad, audio, pad])
        p_sig = np.mean(audio ** 2)
        audio += rng.normal(0, np.sqrt(p_sig / 10 ** (snr_db / 10)), audio.shape).astype(np.float32)
        path = os.path.join(out_dir, f"{i:02d}.wav")
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SR)
            wf.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        items.append({"wav": path, "ref": text, "voice": name, "dur": len(audio) / SR})
    rtf = {}
    for s in tts_stats:
        rtf.setdefault(s["voice"], []).append(s["rtf"])
    print(json.dumps({"piper_rtf": {k: round(float(np.mean(v)), 3) for k, v in rtf.items()}}))
    with open(os.path.join(out_dir, "testset.json"), "w") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    return items


def read_wav(path: str) -> np.ndarray:
    with wave.open(path) as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768


# ── WER ──────────────────────────────────────────────────────────────────────

_NUM = {"fünfzehn": "15", "sieben": "7", "achtzehn": "18", "zwei": "2",
        "zweihundertdreiundvierzig": "243"}


def _norm(s: str) -> list[str]:
    s = s.lower().replace("ß", "ss")
    s = re.sub(r"[^\wäöü ]+", " ", s)
    return [_NUM.get(w, w) for w in s.split()]


def wer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = _norm(ref), _norm(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return int(d[len(h)]), len(r)


# ── engines ──────────────────────────────────────────────────────────────────

def engine_parakeet(models: str, threads: int):
    import sherpa_onnx
    d = os.path.join(models, "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8")
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f"{d}/encoder.int8.onnx", decoder=f"{d}/decoder.int8.onnx",
        joiner=f"{d}/joiner.int8.onnx", tokens=f"{d}/tokens.txt",
        num_threads=threads, model_type="nemo_transducer", decoding_method="greedy_search",
    )

    def run(wav):
        s = rec.create_stream()
        s.accept_waveform(SR, read_wav(wav))
        rec.decode_stream(s)
        return s.result.text
    return run


def engine_whisper(models: str, threads: int, size: str):
    from pywhispercpp.model import Model
    m = Model(os.path.join(models, f"ggml-{size}.bin"), n_threads=threads,
              language="de", print_progress=False, print_realtime=False,
              no_context=True, single_segment=True)

    def run(wav):
        return " ".join(seg.text for seg in m.transcribe(read_wav(wav)))
    return run


def engine_gemma(model: str, threads: int):
    import litert_lm as lm
    lm.set_min_log_severity(lm.LogSeverity.ERROR)
    eng = lm.Engine(model, backend=lm.Backend.CPU(thread_count=threads),
                    audio_backend=lm.Backend.CPU(thread_count=threads),
                    enable_ynnpack=True, cache_dir=os.path.expanduser("~/models/cache"))
    prompt = ("Transkribiere diese deutsche Sprachaufnahme wörtlich. "
              "Gib nur den gesprochenen Text aus, ohne Kommentar.")

    def run(wav):
        conv = eng.create_conversation()
        try:
            r = conv.send_message({"role": "user", "content": [
                {"type": "audio", "path": wav}, {"type": "text", "text": prompt}]})
            return "".join(c.get("text", "") for c in r.get("content", []))
        finally:
            conv.close()
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.expanduser("~/stt/models"))
    ap.add_argument("--voices", default=os.path.expanduser("~/stt/voices"))
    ap.add_argument("--gemma", default=os.path.expanduser("~/models/gemma-4-E2B-it.litertlm"))
    ap.add_argument("--data", default=os.path.expanduser("~/stt/testset"))
    ap.add_argument("--snr", type=float, default=25.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--engines", default="parakeet,whisper-small,whisper-base,gemma")
    ap.add_argument("--out")
    args = ap.parse_args()

    tsf = os.path.join(args.data, "testset.json")
    items = json.load(open(tsf)) if os.path.exists(tsf) else make_testset(args.voices, args.data, args.snr)
    total_audio = sum(i["dur"] for i in items)

    factories = {
        "parakeet": lambda: engine_parakeet(args.models, args.threads),
        "whisper-small": lambda: engine_whisper(args.models, args.threads, "small"),
        "whisper-base": lambda: engine_whisper(args.models, args.threads, "base"),
        "gemma": lambda: engine_gemma(args.gemma, args.threads),
    }
    results = {}
    for name in args.engines.split(","):
        t0 = time.time()
        try:
            run = factories[name]()
        except Exception as e:
            print(json.dumps({"engine": name, "error": f"{type(e).__name__}: {e}"[:300]}), flush=True)
            continue
        load_s = time.time() - t0
        run(items[0]["wav"])  # warm-up
        errs = words = 0
        lat, rows = [], []
        for it in items:
            t = time.time()
            hyp = run(it["wav"]).strip()
            dt = time.time() - t
            e, n = wer(it["ref"], hyp)
            errs, words = errs + e, words + n
            lat.append(dt)
            rows.append({"ref": it["ref"], "hyp": hyp, "err": e, "s": round(dt, 2)})
        res = {
            "engine": name, "load_s": round(load_s, 1), "wer_pct": round(100 * errs / words, 1),
            "lat_mean_s": round(float(np.mean(lat)), 2), "lat_max_s": round(float(np.max(lat)), 2),
            "rtf": round(sum(lat) / total_audio, 3),
        }
        print(json.dumps(res), flush=True)
        for r in rows:
            if r["err"]:
                print("   ", json.dumps(r, ensure_ascii=False), flush=True)
        results[name] = {**res, "rows": rows}
    if args.out:
        json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
