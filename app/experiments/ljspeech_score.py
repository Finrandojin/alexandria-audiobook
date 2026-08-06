"""Score generated audio against the human reading of the same line.

Deliberately NOT mean_f0. Measured 2026-08-04: declared pitch carries 12.9 Hz
mean error against a 32.4 Hz within-voice spread, so a mean over one line
cannot separate two voices. The pitch comparison here is the CONTOUR, aligned
in time - two readings of a sentence differ in pacing, and averaging throws
away the shape that carries the prosody.

METRICS

    ecapa       speaker-embedding cosine. Runs under the sibling interpreter,
                which has speechbrain; app/env does not, and its absence
                silently degrades to acoustic distance - the substitution the
                whole test plan forbids. Handled by scoring in a separate
                process rather than importing.
    f0_corr     Pearson correlation of log-f0 contours after DTW alignment.
    dur_ratio   generated / human seconds. Instruction wording alone moved
                duration 1.36-1.43x, so pacing is known to be movable.
    mcd         mel-cepstral distortion, the standard TTS timbre metric, so
                results are comparable to published work and not only to us.

ANCHORS ARE THE POINT. `human_vs_human` pairs each held-out line with a
DIFFERENT held-out line by the same narrator: the ceiling, how similar audio
gets when it genuinely is one person. Without it, "cosine 0.66" means nothing -
which is exactly the state clone_vs_lora was left in.
"""
import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

# The sibling repo sits beside this one, so derive rather than hard-code - and
# allow an override, because "the interpreter that has speechbrain" is a
# machine fact, not a repository fact. The path guard caught the literal form
# of this line, which is what it exists for.
SIBLING_PY = os.environ.get(
    "ALEXANDRIA_SIBLING_PYTHON",
    os.path.join(os.path.dirname(REPO), "alexandria-audiobook.git",
                 "app", "env", "bin", "python"))


def f0_contour_correlation(a_path, b_path):
    """Pearson r of log-f0 after DTW alignment, or None if too little voicing."""
    import numpy as np
    import librosa

    def contour(path):
        y, sr = librosa.load(path, sr=16000, mono=True)
        f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=350, sr=sr)
        f0 = f0[voiced & np.isfinite(f0)]
        return np.log(f0) if len(f0) > 20 else None

    a, b = contour(a_path), contour(b_path)
    if a is None or b is None:
        return None
    # Align before correlating. Without warping, a slower reading shifts every
    # sample and the correlation measures pacing rather than pitch shape.
    try:
        from librosa.sequence import dtw
        cost, path = dtw(a.reshape(1, -1), b.reshape(1, -1), subseq=False)
        pairs = np.array(path)
        av, bv = a[pairs[:, 0]], b[pairs[:, 1]]
    except Exception:                                   # noqa: BLE001
        n = min(len(a), len(b))
        av, bv = a[:n], b[:n]
    if len(av) < 10 or np.std(av) == 0 or np.std(bv) == 0:
        return None
    return float(np.corrcoef(av, bv)[0, 1])


def mel_cepstral_distortion(a_path, b_path):
    """MCD in dB over DTW-aligned MFCCs. Lower is closer."""
    import numpy as np
    import librosa

    def mfcc(path):
        y, sr = librosa.load(path, sr=22050, mono=True)
        return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=25)[1:]      # drop c0

    a, b = mfcc(a_path), mfcc(b_path)
    if a.shape[1] < 5 or b.shape[1] < 5:
        return None
    try:
        from librosa.sequence import dtw
        _, path = dtw(a, b)
        pairs = np.array(path)
        diff = a[:, pairs[:, 0]] - b[:, pairs[:, 1]]
    except Exception:                                   # noqa: BLE001
        n = min(a.shape[1], b.shape[1])
        diff = a[:, :n] - b[:, :n]
    k = 10.0 / np.log(10) * np.sqrt(2.0)
    return float(k * np.mean(np.sqrt(np.sum(diff ** 2, axis=0))))


def duration(path):
    import soundfile as sf
    info = sf.info(path)
    return info.frames / float(info.samplerate)


def ecapa_scores(pairs):
    """Cosine similarity for (a, b) paths, computed in the sibling env.

    A subprocess rather than an import: app/env has no speechbrain, and
    `voice_data_saturation.embedder()` silently falls back to acoustic distance
    when it is missing. Failing loudly is the point.
    """
    if not os.path.exists(SIBLING_PY):
        return None, f"sibling interpreter not found at {SIBLING_PY}"
    script = os.path.join(APP, "experiments", "_ecapa_batch.py")
    payload = json.dumps([[a, b] for a, b in pairs])
    try:
        out = subprocess.run([SIBLING_PY, script], input=payload,
                             capture_output=True, text=True, timeout=3600,
                             cwd=APP)
    except subprocess.SubprocessError as exc:
        return None, f"ecapa subprocess failed: {exc}"
    if out.returncode != 0:
        return None, f"ecapa exited {out.returncode}: {out.stderr[-300:]}"
    try:
        return json.loads(out.stdout.strip().splitlines()[-1]), None
    except Exception as exc:                            # noqa: BLE001
        return None, f"unparsable ecapa output: {exc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--generated", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "ljspeech_generate.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "ljspeech_score.json"))
    args = ap.parse_args()

    doc = json.load(open(args.generated, encoding="utf-8"))
    rows = doc["rows"][:args.limit] if args.limit else doc["rows"]
    arms = doc["arms"]
    print(f"{len(rows)} lines, arms {arms} + anchors\n")

    ap_ = lambda p: os.path.join(REPO, p)
    scored, pairs, index = [], [], []

    for i, r in enumerate(rows):
        human = ap_(r["human_wav"])
        rec = {"id": r["id"], "book": r["book"],
               "human_seconds": r["human_seconds"]}
        for arm in arms:
            gen = ap_(r[f"{arm}_wav"])
            rec[arm] = {
                "f0_corr": f0_contour_correlation(human, gen),
                "mcd": mel_cepstral_distortion(human, gen),
                "dur_ratio": duration(gen) / max(r["human_seconds"], 1e-6),
            }
            pairs.append((human, gen)); index.append((i, arm))
        # CEILING: the same narrator on a different held-out line. Costs no
        # generation and is what makes every other number readable.
        other = rows[(i + 1) % len(rows)]
        if other["id"] != r["id"]:
            o = ap_(other["human_wav"])
            rec["human_vs_human"] = {
                "f0_corr": f0_contour_correlation(human, o),
                "mcd": mel_cepstral_distortion(human, o),
                "dur_ratio": other["human_seconds"] / max(r["human_seconds"], 1e-6),
            }
            pairs.append((human, o)); index.append((i, "human_vs_human"))
        scored.append(rec)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(rows)} acoustic")

    print("\n  ECAPA in the sibling interpreter...")
    cos, err = ecapa_scores(pairs)
    if err:
        print(f"  WARNING: {err}")
        print("  Acoustic metrics stand; speaker similarity is NOT reported "
              "rather than\n  silently substituted with a weaker metric.")
    else:
        for (i, arm), value in zip(index, cos):
            scored[i][arm]["ecapa"] = value

    import statistics
    print(f"\n  {'arm':16}{'ecapa':>9}{'f0_corr':>9}{'mcd':>8}{'dur':>7}   n")
    summary = {}
    for arm in arms + ["human_vs_human"]:
        vals = [r[arm] for r in scored if arm in r]
        if not vals:
            continue
        def mean(key):
            xs = [v[key] for v in vals if v.get(key) is not None]
            return statistics.mean(xs) if xs else None
        s = {k: mean(k) for k in ("ecapa", "f0_corr", "mcd", "dur_ratio")}
        s["n"] = len(vals)
        summary[arm] = s
        fmt = lambda x, w, p: (f"{x:{w}.{p}f}" if x is not None else " " * (w - 1) + "-")
        print(f"  {arm:16}{fmt(s['ecapa'],9,3)}{fmt(s['f0_corr'],9,3)}"
              f"{fmt(s['mcd'],8,2)}{fmt(s['dur_ratio'],7,2)}{s['n']:4}")

    print("\n  human_vs_human is the CEILING: the same narrator on different")
    print("  material. A generated arm is only readable against it - an ecapa")
    print("  of 0.66 says nothing until you know what one person scores.")

    out = {"arms": arms, "summary": summary, "rows": scored,
           "ecapa_error": err, "source": os.path.relpath(args.generated, REPO)}
    try:
        from experiments.provenance import provenance
        out["provenance"] = provenance(__file__, args)
    except Exception as exc:                            # noqa: BLE001
        out["provenance"] = {"error": str(exc)[:120]}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
