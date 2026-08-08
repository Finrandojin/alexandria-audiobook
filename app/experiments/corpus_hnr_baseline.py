"""Is AISHELL-3 itself noisy, or is SSB1585's synthetic copy too clean?

THE REMAINING EXPLANATION. Chinese LoRA measures HNR 1.17x its narrator, and
the obvious instrument suspect is gone: clip length moves it 0.0025
(`hnr_length_probe.py`), where it destroyed the ECAPA anchor completely. Two
explanations survive. Either the adapter genuinely synthesises cleaner
phonation, or the human denominator is unusually low - AISHELL-3 is 218
speakers recorded in ordinary rooms on consumer hardware, where LJSpeech and
the Japanese set are studio-grade single speakers. A depressed denominator
inflates the ratio without the generated side doing anything at all.

WHAT SEPARATES THEM. The generated side exists for exactly one speaker,
SSB1585, so the corpus question has to be asked of the human recordings alone:

  - if AISHELL-3's speakers sit far below the other corpora's humans, and
    SSB1585 sits with her peers, the ratio is a property of the corpus and the
    Chinese cell of goal 2.6 is measuring recording conditions.
  - if AISHELL-3's speakers sit near the other corpora, or SSB1585 is unusually
    noisy even for AISHELL-3, the corpus does not explain it and the adapter is
    the remaining candidate.

SSB1585 IS REPORTED SEPARATELY AND EXCLUDED FROM THE BASELINE. She is the one
speaker the adapter was trained to imitate; leaving her in the corpus median
would let the thing being tested contribute to the number it is tested against.

This measures human recordings only. Nothing is generated and no adapter is
loaded.
"""
import argparse
import glob
import json
import os
import random
import statistics
import sys

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(APP)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voice_compare_view import voice_quality  # noqa: E402

CORPORA = os.path.join(REPO, "ab_test_runtime", "corpora")
EVAL_SPEAKER = "SSB1585"
SEED = 1234


def aishell_speakers():
    """-> {speaker: [wav paths]} across both AISHELL-3 splits."""
    out = {}
    for split in ("test", "train"):
        root = os.path.join(CORPORA, "aishell3", split, "wav")
        if not os.path.isdir(root):
            continue
        for speaker in sorted(os.listdir(root)):
            wavs = sorted(glob.glob(os.path.join(root, speaker, "*.wav")))
            if wavs:
                out.setdefault(speaker, []).extend(wavs)
    return out


def other_corpus_wavs(manifest, limit):
    """Human wavs from an eval manifest.

    Read from the manifests rather than the corpus trees: these are the exact
    recordings goal 2.6's ratios were computed against, so the baseline is
    comparable to the number it explains. The Japanese corpus is also still
    zipped on disk, where its manifest paths are extracted and present.
    """
    path = os.path.join(REPO, "ab_test_runtime", "experiments", manifest)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)["rows"]
    wavs = []
    for row in rows:
        human = row.get("human_wav")
        if not human:
            continue
        full = human if os.path.isabs(human) else os.path.join(REPO, human)
        if os.path.exists(full):
            wavs.append(full)
    random.Random(SEED).shuffle(wavs)
    return wavs[:limit]


def median_hnr(wavs, label):
    values = []
    for path in wavs:
        q = voice_quality(path)
        hnr = q.get("hnr_db")
        if hnr is not None:
            values.append(hnr)
    if not values:
        return None
    med = statistics.median(values)
    print(f"    {label:14} n={len(values):<4} median {med:.3f} dB", flush=True)
    return med


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--speakers", type=int, default=40,
                    help="AISHELL-3 speakers to sample")
    ap.add_argument("--clips", type=int, default=15,
                    help="clips per speaker")
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "corpus_hnr_baseline.json"))
    args = ap.parse_args()

    by_speaker = aishell_speakers()
    if EVAL_SPEAKER not in by_speaker:
        raise SystemExit(f"{EVAL_SPEAKER} not found in the corpus")

    rng = random.Random(SEED)
    others = sorted(s for s in by_speaker if s != EVAL_SPEAKER)
    rng.shuffle(others)
    chosen = others[:args.speakers]

    print("=== AISHELL-3, speakers the adapter never saw ===", flush=True)
    per_speaker = {}
    for speaker in chosen:
        wavs = by_speaker[speaker][:args.clips]
        med = median_hnr(wavs, speaker)
        if med is not None:
            per_speaker[speaker] = round(med, 4)

    print(f"\n=== {EVAL_SPEAKER} (the eval speaker, excluded above) ===",
          flush=True)
    eval_med = median_hnr(by_speaker[EVAL_SPEAKER][:args.clips * 3],
                          EVAL_SPEAKER)

    print("\n=== other corpora, human recordings ===", flush=True)
    cross = {}
    for name, manifest in (("ljspeech (en)", "ljspeech_generate.json"),
                           ("kokoro (ja)", "kokoro_generate.json")):
        wavs = other_corpus_wavs(manifest, args.speakers * args.clips // 4)
        if wavs:
            cross[name] = median_hnr(wavs, name)

    values = sorted(per_speaker.values())
    corpus_median = statistics.median(values) if values else None
    result = {
        "scope": "human recordings only; no generation, no adapter loaded",
        "eval_speaker": EVAL_SPEAKER,
        "eval_speaker_median_hnr_db": eval_med,
        "aishell3_speakers_sampled": len(per_speaker),
        "aishell3_median_hnr_db": corpus_median,
        "aishell3_p10_p90": [values[len(values) // 10],
                             values[-max(1, len(values) // 10)]]
        if len(values) >= 10 else None,
        "other_corpora_median_hnr_db": cross,
        "per_speaker": per_speaker,
    }

    from utils import atomic_json_write
    atomic_json_write(result, args.out)

    print("\n=== VERDICT INPUTS ===")
    print(f"  AISHELL-3 median (n={len(per_speaker)} speakers): "
          f"{corpus_median:.3f} dB")
    for name, med in cross.items():
        if med is not None:
            print(f"  {name:22} {med:.3f} dB  "
                  f"({corpus_median - med:+.3f} vs AISHELL-3)")
    if eval_med is not None and corpus_median is not None:
        print(f"  {EVAL_SPEAKER} sits {eval_med - corpus_median:+.3f} dB "
              f"against her own corpus")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
