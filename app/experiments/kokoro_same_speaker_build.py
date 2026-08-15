"""Build a narrator-controlled Japanese duration evaluation manifest.

The broader Kokoro evaluation holds out a novel, but LibriVox novels can have
different readers.  This manifest instead uses only *Kokoro*, whose complete
LibriVox recording is read by ekzemplaro, and reserves one clip solely as the
clone prompt.  Every evaluation clip therefore has the same reader and comes
from the same audiobook as the prompt.
"""
import argparse
import glob
import json
import os
import shutil


REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
CORPUS = os.path.join(REPO, "ab_test_runtime", "corpora", "kokoro")
NOVEL = "kokoro-by-soseki-natsume"


def load_rows(min_chars, max_chars):
    texts = {}
    metadata = os.path.join(CORPUS, NOVEL + ".metadata.txt")
    with open(metadata, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) >= 5:
                texts[parts[0]] = parts[4].replace(" ", "").strip()

    import soundfile as sf
    rows = []
    pattern = os.path.join(CORPUS, "wavs", NOVEL + "-*.flac")
    for path in sorted(glob.glob(pattern)):
        sample_id = os.path.splitext(os.path.basename(path))[0]
        text = texts.get(sample_id, "")
        if not min_chars <= len(text) <= max_chars:
            continue
        info = sf.info(path)
        rows.append({
            "id": sample_id,
            "book": NOVEL,
            "text": text,
            "seconds": info.frames / float(info.samplerate),
            "human_wav": os.path.relpath(path, REPO),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--min-chars", type=int, default=18)
    parser.add_argument("--max-chars", type=int, default=70)
    parser.add_argument("--out-dir", default=os.path.join(
        REPO, "ab_test_runtime", "kokoro_same_speaker_eval"))
    args = parser.parse_args()

    rows = load_rows(args.min_chars, args.max_chars)
    references = [row for row in rows if 5.0 <= row["seconds"] <= 8.0]
    if len(rows) < 20 or not references:
        raise SystemExit("not enough eligible local clips for a controlled test")
    reference = references[len(references) // 2]
    test = [row for row in rows if row["id"] != reference["id"]]

    os.makedirs(args.out_dir, exist_ok=True)
    ref_path = os.path.join(args.out_dir, "ref_sample.flac")
    shutil.copy2(os.path.join(REPO, reference["human_wav"]), ref_path)
    document = {
        "corpus": "Kokoro Speech Dataset v1.3 / LibriVox Kokoro",
        "language": "ja",
        "speaker": "ekzemplaro",
        "speaker_evidence": "https://librivox.org/kokoro-by-soseki-natsume/",
        "design": "same speaker, same audiobook; reference excluded from test",
        "ref_sample": os.path.relpath(ref_path, REPO),
        "ref_source_id": reference["id"],
        "ref_text": reference["text"],
        "ref_seconds": reference["seconds"],
        "test_books": [NOVEL],
        "test": test,
    }
    out = os.path.join(args.out_dir, "build.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=1, ensure_ascii=False)
    print(f"reference: {reference['id']} ({reference['seconds']:.2f}s)")
    print(f"test: {len(test)} clips, all {document['speaker']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
