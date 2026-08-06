"""Build Kokoro Speech Dataset audio from LibriVox, for the Japanese arm.

WHY THIS EXISTS AS A SEPARATE STEP. Unlike LJSpeech, the Kokoro release is
metadata only - 3.6 MB of alignments plus `index.json` pointing at the LibriVox
recordings on archive.org. The audio is reconstructed by downloading each
novel's MP3s and cutting them at the sample offsets in the metadata. That is
the price of a corpus nobody has to relicense: the alignment is the
contribution, the audio stays where it was always free.

WHY JAPANESE MATTERS HERE. The LJSpeech arm measures the METHOD - English
non-fiction, measured audiobook register. This project ships Japanese light
novels. A clone that reproduces an English narrator well may not reproduce an
anime-style performance well, and the reverse; nothing so far distinguishes
those cases. Kokoro is single-speaker Japanese narration with public-domain
audio and Aozora text, so it tests the same pipeline on the language the
product actually generates.

METADATA COLUMNS, and a trap. The format is deliberately LJSpeech-like:

    id | source_mp3 | start_sample | end_sample | transcript | reading

but column 6 is ROMAJI, where LJSpeech's third column is normalised English.
Feeding column 6 to a Japanese TTS would train it on romanised text it will
never be given. Column 5 - the kanji-kana transcript - is the one to use, and
its tokens are space-separated by the aligner, so the spaces are stripped.
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

ROOT = os.path.join(REPO, "ab_test_runtime", "corpora", "kokoro")
NATIVE_RATE = 22050          # the metadata's sample offsets are at this rate


def load_metadata(novel):
    """-> [{id, mp3, start, end, text}] for one novel."""
    path = os.path.join(ROOT, f"{novel}.metadata.txt")
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 5:
                continue
            # Column 5, not 6. Six is romaji; see the module docstring.
            text = parts[4].replace(" ", "").strip()
            if not text:
                continue
            rows.append({"id": parts[0], "mp3": parts[1],
                         "start": int(parts[2]), "end": int(parts[3]),
                         "text": text, "book": novel})
    return rows


def fetch_novel_audio(entry, dest):
    """Download and unpack one novel's LibriVox MP3s."""
    url = entry.get("archive_url")
    if not url:
        return False, "no archive_url"
    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, os.path.basename(url))
    if not os.path.exists(zip_path):
        r = subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", zip_path, url],
                           capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            return False, f"download failed: {r.stderr[-200:]}"
    r = subprocess.run(["unzip", "-o", "-q", zip_path, "-d", dest],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        return False, f"unzip failed: {r.stderr[-200:]}"
    return True, None


def cut(src_mp3, start, end, dst, rate):
    """Cut [start, end) samples out of an MP3 into a WAV at `rate`."""
    ss = start / float(NATIVE_RATE)
    dur = (end - start) / float(NATIVE_RATE)
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{ss:.4f}", "-t", f"{dur:.4f}",
         "-i", src_mp3, "-ac", "1", "-ar", str(rate), dst, "-y"],
        capture_output=True, text=True, timeout=180)
    return r.returncode == 0, r.stderr[-200:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-novels", nargs="+",
                    default=["kusamakura-by-soseki-natsume",
                             "botchan-by-soseki-natsume-2",
                             "gan-by-ogai-mori"])
    ap.add_argument("--test-novel", default="kouyahijiri-by-kyoka-izumi",
                    help="held out ENTIRELY - a different novel, a different "
                         "recording session, never seen in training")
    ap.add_argument("--train-limit", type=int, default=200)
    ap.add_argument("--test-limit", type=int, default=150)
    ap.add_argument("--min-chars", type=int, default=18,
                    help="Japanese carries far more per character than English, "
                         "so the 60-220 window used for LJSpeech would select "
                         "only very long sentences")
    ap.add_argument("--max-chars", type=int, default=70)
    ap.add_argument("--rate", type=int, default=24000,
                    help="what tts.py writes; both sides share one rate")
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "kokoro_eval"))
    args = ap.parse_args()

    index = {e["id"]: e for e in json.load(
        open(os.path.join(ROOT, "index.json"), encoding="utf-8"))}
    wanted = list(args.train_novels) + [args.test_novel]
    missing = [n for n in wanted if n not in index]
    if missing:
        sys.exit(f"unknown novel id(s): {missing}")
    if args.test_novel in args.train_novels:
        sys.exit("the held-out novel is also in training - that is the leak "
                 "this design exists to prevent")

    audio_root = os.path.join(ROOT, "librivox")
    train_dir = os.path.join(args.out, "train")
    human_dir = os.path.join(args.out, "human")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(human_dir, exist_ok=True)

    def spread(rows, limit):
        """Round-robin across source MP3s, deterministically.

        Taking the first N by id would draw every clip from the opening
        chapters - the same flaw caught in the LJSpeech split, where 165 of 200
        came from one book.
        """
        import collections
        buckets = collections.OrderedDict()
        for r in sorted(rows, key=lambda x: x["id"]):
            buckets.setdefault(r["mp3"], []).append(r)
        picked, done = [], False
        while len(picked) < limit and not done:
            done = True
            for k in list(buckets):
                if buckets[k]:
                    picked.append(buckets[k].pop(0)); done = False
                    if len(picked) >= limit:
                        break
        return sorted(picked, key=lambda x: x["id"])

    from audio_validation import validate_generated_audio

    def build(novels, limit, dest, label):
        rows = []
        for n in novels:
            rows += [r for r in load_metadata(n)
                     if args.min_chars <= len(r["text"]) <= args.max_chars]
        chosen = spread(rows, limit)
        print(f"  {label}: {len(chosen)} of {len(rows)} usable clips")
        out = []
        for i, r in enumerate(chosen, 1):
            src = os.path.join(audio_root, r["book"], r["mp3"])
            if not os.path.exists(src):
                continue
            dst = os.path.join(dest, r["id"] + ".wav")
            ok, err = cut(src, r["start"], r["end"], dst, args.rate)
            if not ok:
                print(f"    cut failed {r['id']}: {err[:70]}")
                continue
            try:
                validate_generated_audio(dst, f"kokoro cut {r['id']}")
            except Exception as exc:                    # noqa: BLE001
                print(f"    invalid {r['id']}: {str(exc)[:70]}")
                os.remove(dst)
                continue
            import soundfile as sf
            info = sf.info(dst)
            out.append({"id": r["id"], "book": r["book"], "text": r["text"],
                        "seconds": info.frames / float(info.samplerate),
                        "wav": os.path.relpath(dst, REPO)})
            if i % 50 == 0:
                print(f"    {label} {i}/{len(chosen)}")
        return out

    print(f"fetching LibriVox audio for {len(wanted)} novels\n")
    for n in wanted:
        dest = os.path.join(audio_root, n)
        if os.path.isdir(dest) and any(f.endswith(".mp3") for f in os.listdir(dest)):
            print(f"  {n}: already present")
            continue
        print(f"  {n}: downloading {index[n].get('totaltime')} ...")
        ok, err = fetch_novel_audio(index[n], dest)
        if not ok:
            sys.exit(f"{n}: {err}")

    print("\ncutting clips")
    train = build(args.train_novels, args.train_limit, train_dir, "train")
    test = build([args.test_novel], args.test_limit, human_dir, "human")
    if not train or not test:
        sys.exit("nothing built - check the audio downloads")

    assert not ({r["book"] for r in train} & {r["book"] for r in test}), \
        "a novel appears on both sides"

    # metadata.jsonl for train_lora: bare filenames, resolved against --data_dir.
    meta = os.path.join(train_dir, "metadata.jsonl")
    with open(meta, "w", encoding="utf-8") as fh:
        for r in train:
            fh.write(json.dumps({"audio_filepath": r["id"] + ".wav",
                                 "text": r["text"]}, ensure_ascii=False) + "\n")

    # Reference clip from TRAINING material only, mid-length.
    usable = sorted([r for r in train if 4.0 <= r["seconds"] <= 12.0],
                    key=lambda r: r["seconds"])
    if not usable:
        sys.exit("no training clip of usable length for a clone prompt")
    ref = usable[len(usable) // 2]
    import shutil
    ref_src = os.path.join(REPO, ref["wav"])
    shutil.copy2(ref_src, os.path.join(args.out, "ref_sample.wav"))
    shutil.copy2(ref_src, os.path.join(train_dir, "ref.wav"))
    with open(os.path.join(train_dir, "ref_text.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(ref["text"])

    build_doc = {
        "corpus": "Kokoro Speech Dataset v1.3",
        "licence": "public domain (Aozora Bunko texts, LibriVox recordings)",
        "language": "ja",
        "target_rate": args.rate, "native_rate": NATIVE_RATE,
        "train_dir": os.path.relpath(train_dir, REPO),
        "metadata": os.path.relpath(meta, REPO),
        "ref_sample": os.path.relpath(os.path.join(args.out, "ref_sample.wav"), REPO),
        "ref_source_id": ref["id"], "ref_seconds": round(ref["seconds"], 2),
        "ref_text": ref["text"],
        "train_books": sorted({r["book"] for r in train}),
        "test_books": sorted({r["book"] for r in test}),
        "train": [{"id": r["id"], "seconds": r["seconds"]} for r in train],
        "test": [{"id": r["id"], "book": r["book"], "text": r["text"],
                  "seconds": r["seconds"], "human_wav": r["wav"]} for r in test],
    }
    try:
        from experiments.provenance import provenance
        build_doc["provenance"] = provenance(__file__, args)
    except Exception as exc:                            # noqa: BLE001
        build_doc["provenance"] = {"error": str(exc)[:120]}
    out_json = os.path.join(args.out, "build.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(build_doc, fh, indent=1, ensure_ascii=False)

    mins = sum(r["seconds"] for r in train) / 60.0
    print(f"\n  train {len(train)} clips ({mins:.1f} min) from "
          f"{len(build_doc['train_books'])} novels")
    print(f"  held out: {args.test_novel} - {len(test)} lines")
    print(f"  ref: {ref['id']} ({ref['seconds']:.1f}s), training material")
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
