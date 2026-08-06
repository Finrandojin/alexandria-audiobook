"""Prepare the Mandarin arm from AISHELL-3.

Third language for the ground-truth voice evaluation: English (LJSpeech),
Japanese (Kokoro), Mandarin (here). The same pipeline, the same metrics, the
same ceiling anchor, so the three are directly comparable - which is the
actual question. Does voice cloning behave the same way in the languages this
project ships as it does in English?

WHY AISHELL-3 RATHER THAN CSS10's CHINESE SET. CSS10 has Mandarin from
LibriVox, which would have matched the other two arms for register. It was
rejected on two grounds checked rather than assumed: its distribution is a
Kaggle download requiring an account, and the repository states no licence.
AISHELL-3 is Apache 2.0, downloads without credentials from openslr, and ships
transcripts at BOTH character and pinyin level.

That last point matters. Kokoro's column 6 is romaji and CSS10 romanises
Chinese to pinyin; training a TTS on either would teach it a written form it
is never handed at inference. AISHELL-3 gives Chinese characters, so the trap
does not arise.

TWO HONEST DIFFERENCES FROM THE OTHER ARMS, stated because they limit what a
comparison across the three can claim:

1. REGISTER. AISHELL-3 is read sentences, not audiobook narration. LJSpeech
   and Kokoro are people performing a novel. A cloning result here is about
   read speech, and the gap to narration is unmeasured.

2. THE SPLIT IS WEAKER. The other arms hold out a whole novel, so a test line
   shares no recording session with any training line. AISHELL-3 has no books
   to split on - only utterances by one speaker - so this holds out a
   contiguous BLOCK of utterance ids rather than interleaving them. That is
   better than a random split and weaker than a book split, and the difference
   should be remembered when reading the numbers.

WHAT THE MULTI-SPEAKER CORPUS BUYS. 218 speakers under one recording setup
gives a FLOOR anchor the other arms lack: a different person, same microphone,
same conditions. Comparing across corpora would confound speaker identity with
recording chain; here it does not.
"""
import argparse
import collections
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

ROOT = os.path.join(REPO, "ab_test_runtime", "corpora", "aishell3")
TARGET_RATE = 24000


def load_content(train_root):
    """-> {speaker: [{id, wav, text}]} from AISHELL-3's content.txt.

    content.txt is TSV-ish: `<wav_name>\\t<char1> <pinyin1> <char2> <pinyin2>...`
    - characters and pinyin interleaved token by token. The characters are the
    even-indexed tokens; taking the odd ones would give pinyin, which is the
    form the model never sees at inference.
    """
    path = os.path.join(train_root, "content.txt")
    by_speaker = collections.defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            tokens = parts[1].split()
            text = "".join(tokens[0::2]).strip()
            if not name.endswith(".wav") or not text:
                continue
            speaker = name[:7]                  # SSB0005xxxx.wav -> SSB0005
            by_speaker[speaker].append(
                {"id": name[:-4], "wav": name, "text": text,
                 "speaker": speaker})
    return by_speaker


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.path.join(ROOT, "train"))
    ap.add_argument("--speaker", default="",
                    help="target speaker; default is the one with the most "
                         "usable utterances")
    ap.add_argument("--floor-speakers", type=int, default=3,
                    help="other speakers sampled for the different-voice floor")
    ap.add_argument("--train-limit", type=int, default=200)
    ap.add_argument("--test-limit", type=int, default=150)
    ap.add_argument("--min-chars", type=int, default=10,
                    help="Mandarin is denser again than Japanese; AISHELL-3 "
                         "utterances are short read sentences")
    ap.add_argument("--max-chars", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "aishell3_eval"))
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.root, "content.txt")):
        sys.exit(f"no content.txt under {args.root} - is AISHELL-3 extracted?")

    by_speaker = load_content(args.root)
    fits = lambda r: args.min_chars <= len(r["text"]) <= args.max_chars
    usable = {s: [r for r in rs if fits(r)] for s, rs in by_speaker.items()}
    usable = {s: rs for s, rs in usable.items() if rs}
    if not usable:
        sys.exit("no utterances in the length window")

    speaker = args.speaker or max(usable, key=lambda s: len(usable[s]))
    rows = sorted(usable[speaker], key=lambda r: r["id"])
    need = args.train_limit + args.test_limit
    if len(rows) < need:
        print(f"  WARNING: {speaker} has {len(rows)} usable utterances, "
              f"{need} wanted; limits will be reduced")

    # Contiguous block held out, not interleaved. See the module docstring:
    # weaker than the book split the other arms use, and deliberately so
    # documented rather than presented as equivalent.
    test = rows[:args.test_limit]
    train = rows[args.test_limit:args.test_limit + args.train_limit]
    overlap = {r["id"] for r in train} & {r["id"] for r in test}
    assert not overlap, f"leak: {sorted(overlap)[:5]}"

    floor = []
    for other in sorted(s for s in usable if s != speaker)[:args.floor_speakers]:
        floor += sorted(usable[other], key=lambda r: r["id"])[:20]

    print(f"{len(by_speaker)} speakers; using {speaker}")
    print(f"  usable utterances for {speaker}: {len(rows)}")
    print(f"  train {len(train)}  test {len(test)}  floor {len(floor)} "
          f"from {args.floor_speakers} other speakers")

    doc = {"corpus": "AISHELL-3", "licence": "Apache-2.0", "language": "zh",
           "speaker": speaker, "target_rate": TARGET_RATE,
           "root": os.path.relpath(args.root, REPO),
           "split": "contiguous utterance block (no book boundary available)",
           "selection": {"min_chars": args.min_chars,
                         "max_chars": args.max_chars},
           "train": train, "test": test, "floor": floor}
    try:
        from experiments.provenance import provenance
        doc["provenance"] = provenance(__file__, args)
    except Exception as exc:                            # noqa: BLE001
        doc["provenance"] = {"error": str(exc)[:120]}
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "split.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
    print(f"\nwrote {out}")
    if train:
        print(f"  sample: {train[0]['id']}  {train[0]['text'][:30]}")


if __name__ == "__main__":
    main()
