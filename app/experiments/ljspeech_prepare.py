"""Split a public-domain audiobook into a LoRA training set and a held-out test.

WHAT THIS UNLOCKS. Every voice-similarity number in this repo compares
generated audio to `ref_sample.wav` - the same clip that was also the
generation prompt. That is close to circular, and it cannot answer whether the
output sounds like the character SHOULD sound, because nothing knows what the
line should sound like.

LJSpeech does: a human read every line. So for any held-out line there is a
paired comparison - same text, same speaker, human against generated - which is
a different kind of evidence from anything else in results_index.csv.

THE SPLIT IS BY SOURCE WORK, NOT BY CLIP, and that is the whole design.
LJSpeech clip ids look like `LJ042-0153`: book 042, utterance 0153. A random
clip-level split puts LJ042-0152 in train and LJ042-0153 in test - consecutive
sentences from one recording session, same paragraph, same breath. The model
would be scored on material adjacent to what it memorised, and every number
would come out too high with nothing to indicate it.

Splitting whole books apart means the test lines were recorded in a different
session, on a different subject, and the narrator's rendering of them was never
seen.

WHAT THIS SCRIPT DOES NOT DO. It does not resample. LJSpeech is 22.05 kHz and
this project generates at 24 kHz, so a comparison would be measuring the
resampler as much as the model. `ljspeech_build.py` handles that once, for
both sides, so that human and generated audio always share a rate.
"""
import argparse
import collections
import csv
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

DEFAULT_ROOT = os.path.join(REPO, "ab_test_runtime", "corpora", "ljspeech",
                            "LJSpeech-1.1")


def load_metadata(root):
    """-> [{id, book, text, normalized}] from LJSpeech's metadata.csv.

    The file is pipe-delimited with no header: id|raw|normalized. The
    normalized column is what a reader actually says ("nineteen forty-two"
    rather than "1942"), so it is the one to score a transcript against -
    scoring against the raw column would charge every number as an error.
    """
    path = os.path.join(root, "metadata.csv")
    rows = []
    with open(path, encoding="utf-8") as fh:
        for parts in csv.reader(fh, delimiter="|", quoting=csv.QUOTE_NONE):
            if len(parts) < 3:
                continue
            clip_id = parts[0].strip()
            rows.append({"id": clip_id,
                         "book": clip_id.split("-")[0],
                         "text": parts[1].strip(),
                         "normalized": parts[2].strip()})
    return rows


def split_by_book(rows, test_books, min_chars, max_chars):
    """Hold out whole source works, never individual clips."""
    usable = [r for r in rows
              if min_chars <= len(r["normalized"]) <= max_chars]
    by_book = collections.Counter(r["book"] for r in usable)
    if not test_books:
        # Smallest books that still carry enough lines, so the training set
        # keeps as much material as possible.
        ranked = sorted(by_book.items(), key=lambda kv: kv[1])
        test_books = [b for b, n in ranked if n >= 40][:2]
    test_books = set(test_books)
    unknown = test_books - set(by_book)
    if unknown:
        sys.exit(f"unknown book id(s): {sorted(unknown)}")
    train = [r for r in usable if r["book"] not in test_books]
    test = [r for r in usable if r["book"] in test_books]
    return train, test, sorted(test_books), by_book


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--test-books", nargs="*", default=None,
                    help="LJSpeech book ids to hold out, e.g. LJ042 LJ045. "
                         "Default picks the two smallest with >=40 usable "
                         "lines.")
    ap.add_argument("--min-chars", type=int, default=60,
                    help="very short lines carry too little prosody to "
                         "compare, and this repo already measured that short "
                         "fragments behave differently")
    ap.add_argument("--max-chars", type=int, default=220)
    ap.add_argument("--train-limit", type=int, default=200,
                    help="clips used to train the adapter; the library caps "
                         "voice training at 200 samples")
    ap.add_argument("--test-limit", type=int, default=150)
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "corpora", "ljspeech", "split.json"))
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.root, "metadata.csv")):
        sys.exit(f"no metadata.csv under {args.root} - is LJSpeech extracted?")

    rows = load_metadata(args.root)
    train, test, test_books, by_book = split_by_book(
        rows, args.test_books, args.min_chars, args.max_chars)

    print(f"{len(rows)} clips, {len(by_book)} source works")
    print(f"  usable at {args.min_chars}-{args.max_chars} chars: "
          f"{len(train) + len(test)}")
    print(f"  held out : {', '.join(test_books)}  ({len(test)} lines)")
    print(f"  training : {len(train)} lines from "
          f"{len(by_book) - len(test_books)} works")

    # Deterministic AND spread across source works. Sorting by id and taking
    # the first N looked deterministic and was quietly wrong: it drew 165 of
    # 200 training clips from LJ001 and the rest from LJ002, so the adapter
    # would have learned two recording sessions rather than the narrator. The
    # split was still clean - no leak - but the training material was not
    # representative of the voice being cloned.
    #
    # Round-robin over books, each book's clips in id order, so a rerun picks
    # exactly the same clips (the seed lesson applied to SELECTION) while the
    # set spans the corpus.
    def spread(rows, limit):
        buckets = collections.OrderedDict()
        for r in sorted(rows, key=lambda x: x["id"]):
            buckets.setdefault(r["book"], []).append(r)
        picked, exhausted = [], False
        while len(picked) < limit and not exhausted:
            exhausted = True
            for book in list(buckets):
                if buckets[book]:
                    picked.append(buckets[book].pop(0))
                    exhausted = False
                    if len(picked) >= limit:
                        break
        return sorted(picked, key=lambda x: x["id"])

    train = spread(train, args.train_limit)
    test = spread(test, args.test_limit)

    # No clip may appear on both sides. Cheap to assert, and the failure would
    # be invisible in every downstream number.
    overlap = {r["id"] for r in train} & {r["id"] for r in test}
    assert not overlap, f"leak: {sorted(overlap)[:5]}"
    assert not ({r["book"] for r in train} & {r["book"] for r in test}), \
        "a source work appears on both sides"

    doc = {"corpus": "LJSpeech-1.1",
           "licence": "public domain (LibriVox recordings, Gutenberg texts)",
           "root": os.path.relpath(args.root, REPO),
           "sample_rate_native": 22050,
           "test_books": test_books,
           "selection": {"min_chars": args.min_chars,
                         "max_chars": args.max_chars,
                         "train_limit": args.train_limit,
                         "test_limit": args.test_limit},
           "train": train, "test": test}
    try:
        from experiments.provenance import provenance
        doc["provenance"] = provenance(__file__, args)
    except Exception as exc:                            # noqa: BLE001
        doc["provenance"] = {"error": str(exc)[:120]}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print(f"\n  train {len(train)}  test {len(test)}  -> {args.out}")
    print("  Split is by SOURCE WORK: no test line shares a recording session "
          "with\n  any training line.")


if __name__ == "__main__":
    main()
