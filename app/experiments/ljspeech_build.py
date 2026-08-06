"""Build the LoRA training set and the human reference audio, at one rate.

Consumes `ljspeech_prepare.py`'s split and produces:

    <out>/train/metadata.jsonl + wavs   what train_lora.py consumes
    <out>/human/<id>.wav                the held-out lines, human-read
    <out>/ref_sample.wav                the clone prompt, from TRAIN only
    <out>/build.json                    manifest + provenance

THE RESAMPLING IS THE POINT, not housekeeping. LJSpeech is 22.05 kHz; this
project generates at 24 kHz. Comparing a 22.05 kHz human recording against a
24 kHz generation measures the resampler alongside the model, and every
spectral metric - MCD especially - would carry that difference silently. Both
sides are therefore written at one rate, once, here.

Resampling the HUMAN audio rather than downsampling the generated audio is
deliberate: the model's output rate is a property of the system under test and
should not be altered to suit the yardstick.

THE REFERENCE CLIP COMES FROM TRAINING MATERIAL ONLY. It is the clone prompt,
so drawing it from a held-out book would hand the zero-shot arm a sample of the
very recording session it is about to be scored on - the clip-level leak the
split exists to prevent, reintroduced through the back door.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

TARGET_RATE = 24000          # what tts.py writes; see _save_wav


def resample_to(src, dst, rate=TARGET_RATE):
    """Write `src` at `rate`, mono. Returns duration in seconds."""
    import librosa
    import soundfile as sf
    audio, _ = librosa.load(src, sr=rate, mono=True)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    sf.write(dst, audio, rate)
    from audio_validation import validate_generated_audio
    validate_generated_audio(dst, f"resample of {os.path.basename(src)}")
    return len(audio) / float(rate)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default=os.path.join(
        REPO, "ab_test_runtime", "corpora", "ljspeech", "split.json"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "ljspeech_eval"))
    ap.add_argument("--ref-min-seconds", type=float, default=4.0,
                    help="a clone prompt too short carries little of the voice")
    ap.add_argument("--ref-max-seconds", type=float, default=12.0)
    args = ap.parse_args()

    doc = json.load(open(args.split, encoding="utf-8"))
    root = os.path.join(REPO, doc["root"])
    wavs = os.path.join(root, "wavs")
    if not os.path.isdir(wavs):
        sys.exit(f"no wavs/ under {root}")

    train_dir = os.path.join(args.out, "train")
    human_dir = os.path.join(args.out, "human")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(human_dir, exist_ok=True)

    print(f"resampling {doc['sample_rate_native']} -> {TARGET_RATE} Hz "
          f"(both sides share one rate)\n")

    # ── training set ────────────────────────────────────────────────────
    meta_path = os.path.join(train_dir, "metadata.jsonl")
    train_rows, ref_candidates = [], []
    with open(meta_path, "w", encoding="utf-8") as fh:
        for i, row in enumerate(doc["train"], 1):
            src = os.path.join(wavs, row["id"] + ".wav")
            if not os.path.exists(src):
                continue
            rel = os.path.join("train", row["id"] + ".wav")
            secs = resample_to(src, os.path.join(args.out, rel))
            # `normalized` is what a reader SAYS ("nineteen forty-two", not
            # "1942"). Training on the raw column would teach the model to
            # pronounce digits it will never be handed at inference.
            fh.write(json.dumps({"audio_filepath": rel,
                                 "text": row["normalized"]},
                                ensure_ascii=False) + "\n")
            train_rows.append({"id": row["id"], "seconds": secs})
            if args.ref_min_seconds <= secs <= args.ref_max_seconds:
                ref_candidates.append((row, rel, secs))
            if i % 50 == 0:
                print(f"  train {i}/{len(doc['train'])}")

    # ── reference clip, from TRAIN only ─────────────────────────────────
    if not ref_candidates:
        sys.exit("no training clip of usable length for a clone prompt")
    ref_row, ref_rel, ref_secs = ref_candidates[len(ref_candidates) // 2]
    ref_path = os.path.join(args.out, "ref_sample.wav")
    import shutil
    shutil.copy2(os.path.join(args.out, ref_rel), ref_path)

    # ── held-out human audio ────────────────────────────────────────────
    test_rows = []
    for i, row in enumerate(doc["test"], 1):
        src = os.path.join(wavs, row["id"] + ".wav")
        if not os.path.exists(src):
            continue
        dst = os.path.join(human_dir, row["id"] + ".wav")
        secs = resample_to(src, dst)
        test_rows.append({"id": row["id"], "book": row["book"],
                          "text": row["normalized"], "seconds": secs,
                          "human_wav": os.path.relpath(dst, REPO)})
        if i % 50 == 0:
            print(f"  human {i}/{len(doc['test'])}")

    # The leak check the whole design rests on, asserted rather than assumed.
    train_books = {r["id"].split("-")[0] for r in train_rows}
    test_books = {r["book"] for r in test_rows}
    assert not (train_books & test_books), \
        f"source work on both sides: {sorted(train_books & test_books)}"

    build = {"corpus": doc["corpus"], "licence": doc["licence"],
             "target_rate": TARGET_RATE,
             "native_rate": doc["sample_rate_native"],
             "train_dir": os.path.relpath(train_dir, REPO),
             "metadata": os.path.relpath(meta_path, REPO),
             "ref_sample": os.path.relpath(ref_path, REPO),
             "ref_source_id": ref_row["id"], "ref_seconds": round(ref_secs, 2),
             "ref_text": ref_row["normalized"],
             "train_books": sorted(train_books),
             "test_books": sorted(test_books),
             "train": train_rows, "test": test_rows}
    try:
        from experiments.provenance import provenance
        build["provenance"] = provenance(__file__, args)
    except Exception as exc:                            # noqa: BLE001
        build["provenance"] = {"error": str(exc)[:120]}
    out_json = os.path.join(args.out, "build.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(build, fh, indent=1)

    total = sum(r["seconds"] for r in train_rows) / 60.0
    print(f"\n  train {len(train_rows)} clips ({total:.1f} min) -> {meta_path}")
    print(f"  ref   {ref_row['id']} ({ref_secs:.1f}s), from TRAINING material")
    print(f"  human {len(test_rows)} held-out lines -> {human_dir}")
    print(f"  books: train {len(train_books)}, test {sorted(test_books)}, "
          f"overlap none")
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
