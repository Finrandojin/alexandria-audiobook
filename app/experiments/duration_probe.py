"""Duration fidelity (goal 2.4) at a usable sample size.

WHY RE-MEASURE. The goal records LoRA 0.92/0.95/0.95 and clone 0.97/0.76/0.90
against a 0.90-1.10 target, on twelve clips per language. The Japanese clone
cell, 0.76, is the only failing one. Three separate twelve-clip findings have
already dissolved under proper sampling today - English pitch range, Chinese
jitter, Chinese HNR - so a lone outlier at that sample size is a hypothesis,
not a result.

WHAT IS MEASURED. Generated duration divided by the human's on the same
sentence. Read from the audio itself rather than the manifest's recorded
`human_seconds`, so the number describes the files on disk rather than what a
previous run believed about them.

Ratios are summarised by median and by how many clips fall outside the target
band, because a median near 1.0 can hide an arm that is half too fast and half
too slow - which is a different defect from one that is uniformly rushed, and
the median alone cannot tell them apart.
"""
import argparse
import json
import os
import statistics
import sys

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(APP)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pitch_quality_probe import resolve  # noqa: E402

BAND = (0.90, 1.10)
LANGUAGES = {"en": "ljspeech_generate.json",
             "ja": "kokoro_generate.json",
             "ja_same_speaker": "kokoro_same_speaker_generate.json",
             "zh": "aishell3_generate.json"}


def seconds(path):
    import soundfile as sf
    full = resolve(path)
    if not full or not os.path.exists(full):
        return None
    try:
        info = sf.info(full)
        return info.frames / info.samplerate
    except Exception:                                       # noqa: BLE001
        return None


def run(rows, arm, limit):
    ratios, dropped = [], 0
    for row in rows:
        if len(ratios) >= limit:
            break
        h, g = seconds(row.get("human_wav")), seconds(row.get(arm))
        if not h or not g:
            dropped += 1
            continue
        ratios.append(g / h)
    if not ratios:
        raise SystemExit(f"no usable pairs for {arm}")
    outside = [r for r in ratios if not BAND[0] <= r <= BAND[1]]
    return {
        "n": len(ratios), "dropped": dropped,
        "median_ratio": round(statistics.median(ratios), 4),
        "mean_ratio": round(statistics.fmean(ratios), 4),
        "p10": round(statistics.quantiles(ratios, n=10)[0], 4),
        "p90": round(statistics.quantiles(ratios, n=10)[-1], 4),
        "outside_band": len(outside),
        "outside_band_pct": round(100 * len(outside) / len(ratios), 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lines", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "duration_probe.json"))
    args = ap.parse_args()

    result = {"scope": "generated duration / human duration, same line, "
                       "measured from the audio not the manifest",
              "band": list(BAND), "languages": {}}
    for lang, manifest in LANGUAGES.items():
        path = os.path.join(REPO, "ab_test_runtime", "experiments", manifest)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            rows = json.load(handle)["rows"]
        result["languages"][lang] = {}
        for arm in ("lora_wav", "clone_wav"):
            if not any(row.get(arm) for row in rows):
                continue
            result["languages"][lang][arm] = run(rows, arm, args.lines)
            print(f"  {lang} {arm} done", flush=True)

    from utils import atomic_json_write
    atomic_json_write(result, args.out)

    print(f"\n=== duration ratio, target {BAND[0]}-{BAND[1]} ===")
    for lang, arms in result["languages"].items():
        for arm, d in arms.items():
            flag = "" if BAND[0] <= d["median_ratio"] <= BAND[1] else "  OUT"
            print(f"  {lang} {arm:10} n={d['n']:<4} median {d['median_ratio']:.4f}"
                  f"  p10-p90 {d['p10']:.2f}-{d['p90']:.2f}"
                  f"  outside {d['outside_band_pct']}%{flag}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
