"""Build training data from PDNC's HUMAN labels, not a 70B's guesses.

Every training row so far came from `distill_collect`: two cheap passes
disagree, the 70B answers, that answer becomes the label. PDNC carries 35,978
quotations annotated by people. That is a different kind of signal, and it also
covers registers the adapter has never seen - which is what `pdnc_eval` showed
it needs, swinging +10.0 on Austen and -12.5 on Doyle.

TWO CONSTRAINTS THE OUTPUT RESPECTS.

  held out    the three novels already evaluated - Pride and Prejudice, The
              Sign of the Four, The Awakening - are EXCLUDED. Training and
              testing on one novel would produce an impressive meaningless
              number, and it is the easiest mistake to make here.
  capped      quotations per novel are capped, so a long novel cannot dominate
              the mix. The learning curve saturated near 800 rows WITHIN one
              register; the bet here is breadth across registers, not volume
              within one.

The emitted rows use the same JSONL shape `distill_train` already reads, with
the human label written into the `teacher` field, so the trainer needs no
change. `cheap_a`/`cheap_b` are absent because no cheap pass was run - nothing
downstream requires them, and inventing values would misrepresent where the
label came from.
"""
import argparse, ast, collections, csv, json, os, random, sys

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
HELD_OUT = {"PrideAndPrejudice", "TheSignOfTheFour", "TheAwakening"}
SPECIAL = {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE"}


def convert(folder, name, per_novel, context_chars, rng):
    q = os.path.join(folder, f"{name}_quotes.csv")
    c = os.path.join(folder, f"{name}_chars.csv")
    t = os.path.join(folder, f"{name}.txt")
    if not all(os.path.exists(p) for p in (q, c, t)):
        return []
    text = open(t, encoding="utf-8").read()
    roster = []
    for row in csv.DictReader(open(c, encoding="utf-8")):
        main = (row.get("Main Name") or "").strip()
        if main:
            roster.append(main.upper())
    roster = sorted(set(roster))
    rows = []
    for n, row in enumerate(csv.DictReader(open(q, encoding="utf-8"))):
        line = (row.get("quoteText") or "").strip()
        speaker = (row.get("speaker") or "").strip().upper()
        if not line or not speaker or speaker in SPECIAL or speaker not in roster:
            continue
        try:
            spans = ast.literal_eval(row.get("quoteByteSpans") or "[]")
            start = min(s[0] for s in spans)
            end = max(s[1] for s in spans)
        except Exception:
            continue
        rows.append({
            "book": f"pdnc_{name}",
            "segment_index": n,
            "roster": roster,
            "context": [
                {"type": "NARRATOR",
                 "text": text[max(0, start - context_chars):start].strip()},
                {"type": "SPOKEN", "text": line, "target": True},
                {"type": "NARRATOR", "text": text[end:end + context_chars].strip()},
            ],
            "line": line,
            "teacher": speaker,
        })
    rng.shuffle(rows)
    return rows[:per_novel]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folder", required=True)
    ap.add_argument("--per_novel", type=int, default=200)
    ap.add_argument("--context_chars", type=int, default=400)
    ap.add_argument("--out_dir", default=REPO + "/ab_test_runtime/distill")
    args = ap.parse_args()

    rng = random.Random(20260803)
    novels = sorted({f.rsplit("_quotes.csv", 1)[0]
                     for f in os.listdir(args.folder) if f.endswith("_quotes.csv")})
    total, used, skipped = 0, [], []
    for name in novels:
        if name in HELD_OUT:
            skipped.append(name)
            continue
        rows = convert(args.folder, name, args.per_novel, args.context_chars, rng)
        if not rows:
            continue
        out = os.path.join(args.out_dir, f"train__pdnc_{name}.jsonl")
        with open(out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        used.append((name, len(rows)))
        total += len(rows)

    print(f"{len(used)} novels -> {total} training rows "
          f"(cap {args.per_novel}/novel, human labels)")
    for name, n in used[:8]:
        print(f"   {name:28}{n:5}")
    if len(used) > 8:
        print(f"   ... and {len(used)-8} more")
    print(f"\n  HELD OUT, never trained on: {', '.join(sorted(skipped))}")
    print("  Those are the three novels pdnc_eval scores, so the mixed adapter")
    print("  can be measured on registers it has genuinely not seen.")


if __name__ == "__main__":
    main()
