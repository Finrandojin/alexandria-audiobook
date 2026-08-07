"""Does clip length alone break the human-vs-human anchor?

THE QUESTION THIS SETTLES, BEFORE ANY DATA IS FETCHED. The Chinese eval set's
anchor is invalid: the narrator matched herself at 0.691 while synthetic arms
reached 0.720 and 0.765. Chinese also has by far the shortest clips - 3.17s
median against English 7.33s - so the obvious theory is that ECAPA embeddings
degrade on short audio.

Obvious, and untested. Acting on it means acquiring a longer Chinese corpus,
which is real work, and if length is NOT the cause that work buys nothing.

THE CONTROL THAT COSTS NOTHING. Take the ENGLISH clips, whose anchor is sound
at 0.809, and truncate them to the Chinese median. Same speaker, same
recordings, same everything - only duration changes.

    anchor collapses  -> length is sufficient to cause it, and longer Chinese
                         audio is the fix
    anchor holds      -> length is NOT the cause. Something else is wrong with
                         that eval set, and fetching more of the same would not
                         have helped

No generation, no GPU training, no new data: this re-pairs audio that is
already on disk.

PAIRS ARE SAME-SPEAKER, DIFFERENT LINE - the same construction ljspeech_score
uses for `human_vs_human`, so the number is comparable to the one in the score
artifacts rather than to a fresh convention.
"""
import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

# Derived, not hardcoded: the sibling repo sits beside this one, and "the
# interpreter that has speechbrain" is a machine fact rather than a repository
# fact. Same form ljspeech_score.py uses. The machine-path guard exists to
# catch exactly the literal this replaced, and did.
SIBLING_PY = os.environ.get(
    "ALEXANDRIA_SIBLING_PYTHON",
    os.path.join(os.path.dirname(REPO), "alexandria-audiobook.git",
                 "app", "env", "bin", "python"))


def truncate(src, dest, seconds):
    """Head-truncate to `seconds`. Returns the achieved duration, or None if
    the clip is already shorter - padding it would invent audio."""
    import soundfile as sf
    y, sr = sf.read(src, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    want = int(seconds * sr)
    if len(y) <= want:
        return None
    sf.write(dest, y[:want], sr)
    return want / sr


def ecapa(pairs):
    if not pairs:
        return None, "no pairs"
    if not os.path.exists(SIBLING_PY):
        return None, "sibling interpreter missing"
    script = os.path.join(APP, "experiments", "_ecapa_batch.py")
    try:
        out = subprocess.run([SIBLING_PY, script],
                             input=json.dumps([[a, b] for a, b in pairs]),
                             capture_output=True, text=True, timeout=3600,
                             cwd=APP)
    except subprocess.SubprocessError as exc:
        return None, str(exc)[:140]
    if out.returncode != 0:
        return None, f"rc={out.returncode} {out.stderr[-160:]}"
    try:
        return json.loads(out.stdout.strip().splitlines()[-1]), None
    except Exception as exc:                                # noqa: BLE001
        return None, f"unparsable: {exc}"


def anchor_pairs(rows, n, rng):
    """Same speaker, different line — the human_vs_human construction."""
    usable = [r for r in rows if os.path.exists(os.path.join(REPO, r["human_wav"]))]
    rng.shuffle(usable)
    out = []
    for i in range(0, min(len(usable) - 1, n * 2), 2):
        out.append((os.path.join(REPO, usable[i]["human_wav"]),
                    os.path.join(REPO, usable[i + 1]["human_wav"])))
    return out[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sets", nargs="+",
                    default=["ljspeech", "kokoro", "aishell3"])
    ap.add_argument("--truncate-to", type=float, default=3.17,
                    help="seconds; the Chinese set's median clip length")
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "anchor_length_probe.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    work = tempfile.mkdtemp(prefix="anchor_len_")
    results = {}
    print(f"  truncating to {args.truncate_to}s, {args.pairs} pairs per set\n")
    for name in args.sets:
        gen = os.path.join(REPO, "ab_test_runtime", "experiments",
                           f"{name}_generate.json")
        if not os.path.exists(gen):
            print(f"  {name}: no generate artifact"); continue
        rows = json.load(open(gen, encoding="utf-8")).get("rows") or []
        pairs = anchor_pairs(rows, args.pairs, rng)
        if not pairs:
            print(f"  {name}: no usable pairs"); continue

        full, ferr = ecapa(pairs)
        # Truncated: BOTH sides of every pair, so the comparison stays
        # like-for-like. Cutting only one side would measure a mismatch.
        tdir = os.path.join(work, name)
        os.makedirs(tdir, exist_ok=True)
        tpairs, skipped = [], 0
        for i, (a, b) in enumerate(pairs):
            ta = os.path.join(tdir, f"{i}_a.wav")
            tb = os.path.join(tdir, f"{i}_b.wav")
            da = truncate(a, ta, args.truncate_to)
            db = truncate(b, tb, args.truncate_to)
            if da is None or db is None:
                skipped += 1
                continue
            tpairs.append((ta, tb))
        trunc, terr = ecapa(tpairs) if tpairs else (None, "nothing long enough")

        def med(v):
            vals = [x for x in (v or []) if x is not None]
            return round(statistics.median(vals), 4) if vals else None
        rec = {"pairs": len(pairs), "full_anchor": med(full),
               "truncated_pairs": len(tpairs), "truncated_anchor": med(trunc),
               "skipped_already_short": skipped,
               "full_error": ferr, "truncated_error": terr}
        if rec["full_anchor"] and rec["truncated_anchor"]:
            rec["drop"] = round(rec["full_anchor"] - rec["truncated_anchor"], 4)
        results[name] = rec
        d = rec.get("drop")
        print(f"  {name:10} anchor full {rec['full_anchor']}  "
              f"truncated {rec['truncated_anchor']}  "
              f"drop {d if d is not None else '--'}  "
              f"(n={len(tpairs)}, {skipped} already shorter)")

    print("\n  READING THIS: the English anchor is 0.809 at full length. If it")
    print("  falls below its own arms (0.757 clone) when cut to 3.17s, length")
    print("  alone is sufficient to invalidate an anchor, and longer Chinese")
    print("  audio is the fix. If it holds, length is not the cause and more")
    print("  Chinese data of the same kind would not repair anything.")

    doc = {"truncate_to": args.truncate_to, "pairs": args.pairs,
           "seed": args.seed, "results": results}
    try:
        from experiments.provenance import provenance
        doc["provenance"] = provenance(__file__, args)
    except Exception as exc:                                # noqa: BLE001
        doc["provenance"] = {"error": str(exc)[:120]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print(f"\nwrote {args.out}")
    if not any(r.get("truncated_anchor") for r in results.values()):
        sys.exit(3)


if __name__ == "__main__":
    main()
