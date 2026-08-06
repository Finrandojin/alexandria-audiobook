"""Generate the held-out lines the human already read, in every arm.

Consumes `ljspeech_build.py`'s output. For each held-out line there is human
audio of that exact sentence, so this is a paired comparison rather than a
similarity score against the generation's own prompt.

ARMS, and what each one is for:

    lora        the trained adapter
    clone       zero-shot from the same reference, no training
    ceiling     NOT generated - the human reading a DIFFERENT held-out line
    floor       NOT generated - a different narrator entirely

The last two are the reason this experiment is worth running. `clone_vs_lora`
reported cosine 0.664 against 0.634 and nobody could say whether 0.664 was
good, because there was no scale. The ceiling is how similar audio gets when it
genuinely IS the same person on different material; the floor is what
"different voice" scores. A generated number only means something between
them, and the anchors cost no GPU time - they are pairings of audio that
already exists.

Everything is seeded. Unseeded, one adapter's pitch moves across a 32.4 Hz
median band, which is wider than the gap between many distinct voices - so an
unseeded arm would be measuring the draw, not the method.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", default=os.path.join(
        REPO, "ab_test_runtime", "ljspeech_eval", "build.json"))
    ap.add_argument("--adapter", default=os.path.join(
        REPO, "ab_test_runtime", "ljspeech_eval", "adapter"))
    ap.add_argument("--config", default=os.path.join(APP, "config.json"))
    ap.add_argument("--out-dir", default=os.path.join(
        REPO, "ab_test_runtime", "ljspeech_eval", "generated"))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = every held-out line")
    ap.add_argument("--arms", nargs="+", default=["lora", "clone"])
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "ljspeech_generate.json"))
    args = ap.parse_args()

    build = json.load(open(args.build, encoding="utf-8"))
    rows = build["test"][:args.limit] if args.limit else build["test"]
    ref = os.path.join(REPO, build["ref_sample"])
    if not os.path.exists(ref):
        sys.exit(f"no reference clip at {ref}")
    if "lora" in args.arms and not os.path.isdir(args.adapter):
        sys.exit(f"no adapter at {args.adapter} - train it first")

    os.makedirs(args.out_dir, exist_ok=True)
    from tts import TTSEngine
    from experiments.generation import render, GenerationFailed
    engine = TTSEngine(json.load(open(args.config, encoding="utf-8")))

    # The clone arm points at the SAME reference the adapter was built around,
    # so the only difference between arms is whether the fine-tuned weights are
    # applied. A different reference would confound method with prompt.
    entries = {
        "lora": {"type": "lora", "adapter_path": os.path.relpath(args.adapter, REPO),
                 "seed": str(args.seed)},
        "clone": {"type": "clone", "ref_audio": build["ref_sample"],
                  "ref_text": build["ref_text"], "seed": str(args.seed)},
    }

    print(f"{len(rows)} held-out lines x {len(args.arms)} arms, seed {args.seed}")
    print(f"  reference: {build['ref_source_id']} (training material)\n")

    produced, failures = [], []
    for i, row in enumerate(rows, 1):
        record = {"id": row["id"], "book": row["book"], "text": row["text"],
                  "human_wav": row["human_wav"], "human_seconds": row["seconds"]}
        ok = True
        for arm in args.arms:
            entry = entries[arm]
            wav = os.path.join(args.out_dir, f"{row['id']}__{arm}.wav")
            try:
                render(engine, row["text"], "", "SPEAKER",
                       {"SPEAKER": entry}, entry, wav)
            except GenerationFailed as exc:
                failures.append({"id": row["id"], "arm": arm,
                                 "error": str(exc)[:140]})
                print(f"  [{i}/{len(rows)}] {arm} FAILED: {str(exc)[:60]}")
                ok = False
                break
            record[f"{arm}_wav"] = os.path.relpath(wav, REPO)
        # Paired comparison: a line missing an arm is dropped entirely rather
        # than scored asymmetrically.
        if ok:
            produced.append(record)
        if i % 25 == 0:
            print(f"  [{i}/{len(rows)}] {len(produced)} complete, "
                  f"{len(failures)} dropped")

    doc = {"seed": args.seed, "arms": args.arms,
           "reference_id": build["ref_source_id"],
           "test_books": build["test_books"],
           "rows": produced, "failures": failures}
    try:
        from experiments.provenance import provenance
        doc["provenance"] = provenance(__file__, args)
    except Exception as exc:                            # noqa: BLE001
        doc["provenance"] = {"error": str(exc)[:120]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)

    print(f"\n  {len(produced)} lines complete in all arms, "
          f"{len(failures)} dropped")
    print(f"wrote {args.out}")
    if failures:
        sys.exit(3)


if __name__ == "__main__":
    main()
