"""Resample AISHELL-3 clips to the rate this project generates at.

Simpler than the LJSpeech and Kokoro builds because AISHELL-3 ships individual
per-utterance wavs - there is nothing to cut, only to resample and lay out in
the shape `train_lora.py` and `ljspeech_generate.py` expect.

The rate conversion is the same point as in the other arms and not
housekeeping: AISHELL-3 is 44.1 kHz, this project generates at 24 kHz, and
comparing across rates measures the resampler alongside the model. MCD would
carry that silently. The HUMAN side is converted, never the generated side,
whose rate is a property of the system under test.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

TARGET_RATE = 24000


def resample_to(src, dst, rate=TARGET_RATE):
    import librosa
    import soundfile as sf
    from audio_validation import validate_generated_audio
    audio, _ = librosa.load(src, sr=rate, mono=True)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    sf.write(dst, audio, rate)
    validate_generated_audio(dst, f"resample of {os.path.basename(src)}")
    return len(audio) / float(rate)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default=os.path.join(
        REPO, "ab_test_runtime", "aishell3_eval", "split.json"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "aishell3_eval"))
    ap.add_argument("--ref-min-seconds", type=float, default=3.0)
    ap.add_argument("--ref-max-seconds", type=float, default=12.0)
    args = ap.parse_args()

    doc = json.load(open(args.split, encoding="utf-8"))
    wav_root = os.path.join(REPO, doc["root"], "wav")
    if not os.path.isdir(wav_root):
        sys.exit(f"no wav/ under {os.path.join(REPO, doc['root'])}")

    train_dir = os.path.join(args.out, "train")
    human_dir = os.path.join(args.out, "human")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(human_dir, exist_ok=True)

    def src_of(row):
        return os.path.join(wav_root, row["speaker"], row["wav"])

    print(f"resampling -> {TARGET_RATE} Hz (both sides share one rate)\n")

    meta = os.path.join(train_dir, "metadata.jsonl")
    train_rows, ref_candidates = [], []
    with open(meta, "w", encoding="utf-8") as fh:
        for i, row in enumerate(doc["train"], 1):
            src = src_of(row)
            if not os.path.exists(src):
                continue
            dst = os.path.join(train_dir, row["id"] + ".wav")
            secs = resample_to(src, dst)
            # Bare filename: train_lora resolves against --data_dir.
            fh.write(json.dumps({"audio_filepath": row["id"] + ".wav",
                                 "text": row["text"]},
                                ensure_ascii=False) + "\n")
            train_rows.append({"id": row["id"], "seconds": secs})
            if args.ref_min_seconds <= secs <= args.ref_max_seconds:
                ref_candidates.append((row, dst, secs))
            if i % 50 == 0:
                print(f"  train {i}/{len(doc['train'])}")

    if not ref_candidates:
        sys.exit("no training clip of usable length for a clone prompt")
    ref_row, ref_src, ref_secs = ref_candidates[len(ref_candidates) // 2]
    import shutil
    ref_path = os.path.join(args.out, "ref_sample.wav")
    shutil.copy2(ref_src, ref_path)
    shutil.copy2(ref_src, os.path.join(train_dir, "ref.wav"))
    with open(os.path.join(train_dir, "ref_text.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(ref_row["text"])

    test_rows = []
    for i, row in enumerate(doc["test"], 1):
        src = src_of(row)
        if not os.path.exists(src):
            continue
        dst = os.path.join(human_dir, row["id"] + ".wav")
        secs = resample_to(src, dst)
        test_rows.append({"id": row["id"], "book": doc["speaker"],
                          "text": row["text"], "seconds": secs,
                          "human_wav": os.path.relpath(dst, REPO)})
        if i % 50 == 0:
            print(f"  human {i}/{len(doc['test'])}")

    ids_train = {r["id"] for r in train_rows}
    ids_test = {r["id"] for r in test_rows}
    assert not (ids_train & ids_test), "an utterance appears on both sides"

    build = {"corpus": doc["corpus"], "licence": doc["licence"],
             "language": doc["language"], "speaker": doc["speaker"],
             "split_note": doc["split"],
             "target_rate": TARGET_RATE,
             "train_dir": os.path.relpath(train_dir, REPO),
             "metadata": os.path.relpath(meta, REPO),
             "ref_sample": os.path.relpath(ref_path, REPO),
             "ref_source_id": ref_row["id"], "ref_seconds": round(ref_secs, 2),
             "ref_text": ref_row["text"],
             "train_books": [doc["speaker"]], "test_books": [doc["speaker"]],
             "train": train_rows, "test": test_rows}
    try:
        from experiments.provenance import provenance
        build["provenance"] = provenance(__file__, args)
    except Exception as exc:                            # noqa: BLE001
        build["provenance"] = {"error": str(exc)[:120]}
    out_json = os.path.join(args.out, "build.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(build, fh, indent=1, ensure_ascii=False)

    mins = sum(r["seconds"] for r in train_rows) / 60.0
    print(f"\n  train {len(train_rows)} clips ({mins:.1f} min), speaker "
          f"{doc['speaker']}")
    print(f"  human {len(test_rows)} held-out utterances")
    print(f"  ref   {ref_row['id']} ({ref_secs:.1f}s), training material")
    print(f"  NOTE: {doc['split']}")
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
