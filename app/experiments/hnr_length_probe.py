"""Is the Chinese HNR gap a property of the voice, or of 3-second clips?

THE SUSPICION. Chinese LoRA measures HNR 1.17x the human's - the generated
voice more periodic, less noisy, than the person it copies - and it stays there
at 100 clips, so it is not sample size. But the Chinese clips are ~3.2 s where
English is ~7 s, and clip length has already been proven to break one
instrument in this project: the ECAPA anchor scored the narrator worse against
herself than a machine imitation, entirely because the clips were too short
(goal 2.2). An estimator fooled by short audio once is worth suspecting twice.

WHY LENGTH COULD DO THIS. Praat's harmonicity is an autocorrelation measure. It
needs enough periods to separate harmonic energy from noise, and on short
windows the estimate is both noisier and biased - and the two sides here are
not symmetric. A synthetic clip is uniformly voiced; a human clip of the same
sentence carries breath, onset and release. Shorten both and the human loses
proportionally more of what makes it look noisy, which would inflate the ratio
without either voice changing at all.

THE TEST IS THE ONE THAT SETTLED 2.2, RUN IN BOTH DIRECTIONS.

    lengthen CHINESE (join same-speaker clips to >= 7 s) -> does 1.17x fall in?
    shorten ENGLISH (truncate to the Chinese median, 3.2 s) -> does it inflate?

One direction alone proves nothing: Chinese could improve for any reason, and
English could be robust for any reason. Both moving together is the signature
of a length artifact. Neither moving means the gap is real and belongs to the
model, which is the outcome that keeps goal 2.6 open.

Joining is legitimate here for the same reason it was for the anchor: HNR is a
property of the phonation, not of sentence continuity, and every clip in a set
is the same speaker.
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
from voice_compare_view import voice_quality  # noqa: E402

TARGET_SECONDS = 7.0        # same knee ANCHOR_MIN_SECONDS was chosen from
SHORT_SECONDS = 3.2         # the Chinese median, per goal 2.2
SR = 22050


def _load(path):
    import librosa
    y, _ = librosa.load(resolve(path), sr=SR, mono=True)
    return y


def _measure_array(y, tmp_path):
    import soundfile as sf
    sf.write(tmp_path, y, SR)
    return voice_quality(tmp_path)


def hnr_ratio(rows, arm, mode, limit, tmp_dir):
    """Median HNR of generated ÷ human, with clip length manipulated.

    mode "join": concatenate consecutive clips until each side reaches
    TARGET_SECONDS. mode "cut": truncate each side to SHORT_SECONDS. mode
    "asis": no manipulation, to reproduce the existing number as a control.
    """
    import numpy as np
    pairs, buf_h, buf_g, dropped = [], [], [], 0
    for row in rows:
        if len(pairs) >= limit:
            break
        hp, gp = row.get("human_wav"), row.get(arm)
        if not hp or not gp:
            dropped += 1
            continue
        if not (os.path.exists(resolve(hp)) and os.path.exists(resolve(gp))):
            dropped += 1
            continue
        h, g = _load(hp), _load(gp)
        if mode == "cut":
            n = int(SHORT_SECONDS * SR)
            h, g = h[:n], g[:n]
        elif mode == "join":
            buf_h.append(h)
            buf_g.append(g)
            if (sum(len(x) for x in buf_h) < TARGET_SECONDS * SR or
                    sum(len(x) for x in buf_g) < TARGET_SECONDS * SR):
                continue
            h = np.concatenate(buf_h)
            g = np.concatenate(buf_g)
            buf_h, buf_g = [], []
        mh = _measure_array(h, os.path.join(tmp_dir, "h.wav"))
        mg = _measure_array(g, os.path.join(tmp_dir, "g.wav"))
        if mh.get("hnr_db") is None or mg.get("hnr_db") is None:
            dropped += 1
            continue
        pairs.append((mh["hnr_db"], mg["hnr_db"],
                      len(h) / SR, len(g) / SR))
        print(f"    pair {len(pairs)}  human {mh['hnr_db']:.2f} dB  "
              f"gen {mg['hnr_db']:.2f} dB  ({len(h)/SR:.1f}s)", flush=True)
    if not pairs:
        raise SystemExit(f"no usable pairs for {arm} mode={mode}")
    hm = statistics.median(p[0] for p in pairs)
    gm = statistics.median(p[1] for p in pairs)
    return {"n": len(pairs), "dropped": dropped,
            "median_seconds": round(statistics.median(p[2] for p in pairs), 2),
            "human_hnr_db": round(hm, 4), "generated_hnr_db": round(gm, 4),
            "ratio": round(gm / hm, 4) if hm else None,
            "difference_db": round(gm - hm, 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lines", type=int, default=100)
    ap.add_argument("--arm", default="lora_wav")
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "hnr_length_probe.json"))
    args = ap.parse_args()

    tmp_dir = os.path.join(REPO, "ab_test_runtime", "hnr_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    plan = [("zh", "aishell3_generate.json", "asis"),
            ("zh", "aishell3_generate.json", "join"),
            ("en", "ljspeech_generate.json", "asis"),
            ("en", "ljspeech_generate.json", "cut")]

    result = {"scope": "HNR ratio with clip length manipulated, both "
                       "directions; the test that settled goal 2.2",
              "arm": args.arm, "target_seconds": TARGET_SECONDS,
              "short_seconds": SHORT_SECONDS, "conditions": {}}
    for lang, manifest, mode in plan:
        path = os.path.join(REPO, "ab_test_runtime", "experiments", manifest)
        with open(path, encoding="utf-8") as handle:
            rows = json.load(handle)["rows"]
        print(f"\n=== {lang} {mode} ===", flush=True)
        result["conditions"][f"{lang}_{mode}"] = hnr_ratio(
            rows, args.arm, mode, args.lines, tmp_dir)

    from utils import atomic_json_write
    atomic_json_write(result, args.out)

    print("\n=== HNR ratio (generated / human) ===")
    for name, d in result["conditions"].items():
        band = "in band" if 0.85 <= d["ratio"] <= 1.15 else "OUT of band"
        print(f"  {name:10} {d['median_seconds']:>5.1f}s  n={d['n']:<4} "
              f"{d['ratio']:.4f}x  {band}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
