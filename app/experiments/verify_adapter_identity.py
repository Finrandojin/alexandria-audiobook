"""Does a freshly trained adapter actually sound like its narrator?

THE FAILURE THIS GATES. Five adapters in the shipped library resembled nobody -
0.027, 0.061, 0.099, 0.397, 0.404 speaker similarity against the narrators they
were trained on, where a working voice reaches ~0.73. All five recovered on a
retrain, two of them to ~0.67. The voices were recoverable the whole time and
nothing said otherwise.

WHY LOSS CANNOT DO THIS JOB. Final loss on the 0.027 adapter was 4.07. Across
the entire 75-adapter library loss sits at ~4.1 regardless of whether the voice
works. It is blind to this failure, so training "succeeding" means nothing.

The sibling gate `verify_adapter_stops` catches a different defect - adapters
that never stop generating - and the same shape works here: generate a handful
of lines, measure, refuse. Neither is visible without generating audio.

WHY HELD-OUT CLIPS. Scoring against training clips measures memorisation. The
dataset builder writes a 180/20 split and `train_lora.py` now honours it, so
the val clips are genuinely unseen and this is a real test rather than a
recital.

THRESHOLD. Default 0.45. Measured reference points from the library:

    working adapters        0.65 - 0.74
    the five failures       0.027 - 0.404
    human vs human ceiling  ~0.83

0.45 sits in the empty band between the failures and the working ones. It is
deliberately generous: the purpose is catching a voice that resembles NOBODY,
not ranking good voices against each other.
"""
import argparse
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(APP, "experiments"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter", required=True,
                    help="trained adapter directory")
    ap.add_argument("--dataset", required=True,
                    help="dataset directory or zip holding val/")
    ap.add_argument("--lines", type=int, default=6)
    ap.add_argument("--min-ecapa", type=float, default=0.45,
                    help="below this the adapter resembles nobody; the library "
                         "band between failures (<=0.404) and working (>=0.65)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from library_voice_fidelity import extract_val, ecapa_pairs
    work = os.path.join(args.adapter, "identity_check")
    os.makedirs(work, exist_ok=True)

    if args.dataset.endswith(".zip"):
        clips = extract_val(args.dataset, work, args.lines)
    else:
        meta = os.path.join(args.dataset, "val", "metadata.jsonl")
        if not os.path.exists(meta):
            sys.exit(f"no val split in {args.dataset}; nothing held out to "
                     f"test against")
        clips = []
        with open(meta, encoding="utf-8") as fh:
            for line in list(fh)[:args.lines]:
                if not line.strip():
                    continue
                e = json.loads(line)
                p = os.path.join(args.dataset, e["audio_filepath"])
                if os.path.exists(p):
                    clips.append((p, e.get("text") or ""))
    if not clips:
        sys.exit("no held-out clips available")

    from tts import TTSEngine
    from experiments.generation import render, GenerationFailed
    engine = TTSEngine(json.load(open(os.path.join(APP, "config.json"),
                                      encoding="utf-8")))
    entry = {"type": "lora",
             "adapter_path": os.path.relpath(args.adapter, REPO),
             "seed": str(args.seed)}
    pairs, failed = [], 0
    print(f"{len(clips)} held-out lines through "
          f"{os.path.basename(args.adapter.rstrip('/'))}\n")
    for i, (human_wav, text) in enumerate(clips):
        gen = os.path.join(work, f"check_{i}.wav")
        try:
            render(engine, text, "", "SPEAKER", {"SPEAKER": entry}, entry, gen)
        except GenerationFailed as exc:
            failed += 1
            print(f"  line {i}: GENERATION FAILED {str(exc)[:60]}")
            continue
        pairs.append([human_wav, gen])

    sib = os.environ.get(
        "ALEXANDRIA_SIBLING_PYTHON",
        os.path.join(os.path.dirname(REPO), "alexandria-audiobook.git",
                     "app", "env", "bin", "python"))
    cos, err = ecapa_pairs(pairs, sib)
    vals = [c for c in (cos or []) if c is not None]
    if not vals:
        # Not measured is not the same as failed. Refusing here would block a
        # good adapter over a missing interpreter; passing would defeat the
        # gate. Exit 2 marks it as unrun.
        print(f"\n  NOT MEASURED: {err}")
        print("  Speaker similarity was not computed, so this adapter is "
              "neither passed nor refused.")
        sys.exit(2)

    median = statistics.median(vals)
    ok = median >= args.min_ecapa
    for i, c in enumerate(vals):
        print(f"  line {i}: {c:.3f}")
    print(f"\n  median {median:.3f}, threshold {args.min_ecapa:.2f}")
    verdict = (f"PASS - the adapter sounds like its narrator ({median:.3f})"
               if ok else
               f"FAIL - {median:.3f} is below {args.min_ecapa:.2f}. Working "
               f"adapters reach 0.65-0.74; the five known-broken ones scored "
               f"0.027-0.404. Retrain before shipping: on 2026-08-07 all five "
               f"failures recovered on a rerun, two of them to ~0.67.")
    print(f"  {verdict}")

    doc = {"adapter": os.path.relpath(args.adapter, REPO),
           "median_ecapa": round(median, 4), "lines": len(vals),
           "generation_failures": failed, "threshold": args.min_ecapa,
           "passed": ok, "verdict": verdict}
    out = args.out or os.path.join(args.adapter, "identity_check.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
    print(f"\nwrote {out}")
    # Non-zero so a training chain refuses to promote a voice that resembles
    # nobody, the same contract verify_adapter_stops uses.
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
