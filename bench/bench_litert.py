#!/usr/bin/env python3
"""Benchmark Gemma 4 on LiteRT-LM for Robinson (Raspberry Pi 5).

1. Synthetic sweep via litert_lm.Benchmark: prefill/decode tok/s per config.
2. Conversational test: German robot-style turns, streamed, measuring
   time-to-first-token, time-to-first-sentence (what TTS waits for) and decode
   rate, while sampling SoC temperature and throttle flags.
"""

import argparse
import json
import re
import subprocess
import threading
import time

import litert_lm as lm

SYSTEM = (
    "Du bist Robinson, ein freundlicher Roboter, eingebaut in einen alten "
    "Röhrenfernseher. Antworte immer auf Deutsch, kurz und lebendig, in "
    "höchstens zwei oder drei Sätzen. Keine Emojis, kein Markdown."
)
TURNS = [
    "Hallo Robinson, wie geht es dir heute?",
    "Erzähl mir einen kurzen Witz über Roboter.",
    "Was sollte ich heute Abend kochen, wenn ich nur Kartoffeln, Eier und Zwiebeln habe?",
    "Erkläre in einfachen Worten, wie ein Kühlschrank funktioniert.",
]
SENTENCE_END = re.compile(r"[.!?](\s|$)")


def vc(*args):
    return subprocess.run(["vcgencmd", *args], capture_output=True, text=True).stdout.strip()


class ThermalLog(threading.Thread):
    def __init__(self, period=1.0):
        super().__init__(daemon=True)
        self.period, self.samples, self.stop = period, [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            t = float(vc("measure_temp").split("=")[1].split("'")[0])
            f = int(vc("measure_clock", "arm").split("=")[1]) // 1_000_000
            thr = int(vc("get_throttled").split("=")[1], 16)
            self.samples.append((time.time(), t, f, thr))
            self.stop.wait(self.period)

    def summary(self):
        if not self.samples:
            return {}
        temps = [s[1] for s in self.samples]
        freqs = [s[2] for s in self.samples]
        return {
            "temp_max_c": max(temps),
            "arm_mhz_min": min(freqs),
            "throttled_flags": hex(max(s[3] for s in self.samples)),
        }


def sweep(model, configs, prefill, decode):
    rows = []
    for name, kw in configs:
        b = lm.Benchmark(
            model_path=model,
            backend=lm.Backend.CPU(thread_count=kw.get("threads")),
            prefill_tokens=prefill,
            decode_tokens=decode,
            enable_ynnpack=kw.get("ynnpack", False),
            enable_speculative_decoding=kw.get("spec"),
            cache_dir="/home/matthias/models/cache",
        )
        try:
            r = b.run()
            row = {
                "config": name,
                "init_s": round(r.init_time_in_second, 2),
                "prefill_tps": round(r.last_prefill_tokens_per_second, 1),
                "decode_tps": round(r.last_decode_tokens_per_second, 2),
            }
        except Exception as e:  # unsupported combos shouldn't kill the run
            row = {"config": name, "error": f"{type(e).__name__}: {e}"[:200]}
        row["temp_c"] = vc("measure_temp")
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def conversation(model, kw):
    t0 = time.time()
    engine = lm.Engine(
        model,
        backend=lm.Backend.CPU(thread_count=kw.get("threads")),
        enable_ynnpack=kw.get("ynnpack", False),
        cache_dir="/home/matthias/models/cache",
    )
    conv = engine.create_conversation(system_message=SYSTEM)
    load_s = time.time() - t0
    print(json.dumps({"engine_load_s": round(load_s, 2)}), flush=True)

    out = []
    for turn in TURNS:
        start = time.time()
        first_tok = first_sentence = None
        text, chunks = "", 0
        for chunk in conv.send_message_async(turn):
            piece = "".join(
                c.get("text", "") for c in chunk.get("content", []) if isinstance(c, dict)
            ) if isinstance(chunk, dict) else str(chunk)
            if not piece:
                continue
            now = time.time()
            if first_tok is None:
                first_tok = now - start
            chunks += 1
            text += piece
            if first_sentence is None and SENTENCE_END.search(text):
                first_sentence = now - start
        total = time.time() - start
        ntok = len(engine.tokenize(text))
        row = {
            "prompt": turn[:40],
            "ttft_s": round(first_tok or 0, 2),
            "first_sentence_s": round(first_sentence or total, 2),
            "total_s": round(total, 2),
            "out_tokens": ntok,
            "decode_tps": round(ntok / max(total - (first_tok or 0), 1e-6), 2),
            "reply": text.strip(),
        }
        print(json.dumps(row, ensure_ascii=False), flush=True)
        out.append(row)
    conv.close()
    engine.close()
    return {"engine_load_s": round(load_s, 2), "turns": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prefill", type=int, default=256)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    configs = [
        ("cpu4", {"threads": 4}),
        ("cpu4+ynnpack", {"threads": 4, "ynnpack": True}),
        ("cpu3", {"threads": 3}),
        ("cpu4+spec", {"threads": 4, "spec": True}),
        ("cpu4+ynnpack+spec", {"threads": 4, "ynnpack": True, "spec": True}),
    ]
    therm = ThermalLog()
    therm.start()
    result = {"model": args.model}
    if not args.skip_sweep:
        result["sweep"] = sweep(args.model, configs, args.prefill, args.decode)
        ok = [r for r in result["sweep"] if "decode_tps" in r]
        best = max(ok, key=lambda r: r["decode_tps"])["config"] if ok else "cpu4"
    else:
        best = "cpu4"
    print(json.dumps({"conversation_config": best}), flush=True)
    result["conversation"] = conversation(args.model, dict(configs)[best])
    therm.stop.set()
    therm.join()
    result["thermal"] = therm.summary()
    print(json.dumps({"thermal": result["thermal"]}), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
