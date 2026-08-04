"""Turn PDNC novels into fixtures, so generalisation stops needing a person.

Every result in this investigation rests on 772 rows across four light novels,
all translated, all contemporary. Whether the adapter's +5.4 is a property of
the adapter or of that corpus has been the largest open risk, and I repeatedly
described testing it as blocked on someone hand-labelling a fifth book. That
was wrong: the Project Dialogism Novel Corpus already carries speaker-attributed
quotations for 28 public-domain novels.

WHAT PDNC GIVES THAT OUR OWN GOLD DOES NOT.

  quoteType     Explicit / Implicit / Anaphoric per quotation. Pride and
                Prejudice is 26% explicit and 50% IMPLICIT, so this is not the
                soft benchmark classic prose sounds like - and it allows
                stratifying results by how much attribution cue exists, which
                our own fixtures never labelled.
  Aliases       curated per character, replacing alias groups we assembled by
                hand and repeatedly got wrong.
  Category      major / intermediate / minor, replacing the 5%-of-lines
                frequency proxy used for "lead character".

WHAT IT DOES NOT TEST. PDNC annotates quotations in raw novel text, so this
measures ATTRIBUTION only and bypasses segmentation entirely. It says nothing
about the misfiling problem, and a book where every quotation is already
delimited is an easier world than the segmenter's output.

LICENCE. The novels are public domain; the repository declares no licence for
the ANNOTATIONS. Fine for internal evaluation, unresolved for anything shipped.
"""
import argparse, ast, collections, csv, json, os, re, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
SPECIAL = {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE"}


def load_novel(folder, name):
    quotes = list(csv.DictReader(open(os.path.join(folder, f"{name}_quotes.csv"),
                                      encoding="utf-8")))
    chars = list(csv.DictReader(open(os.path.join(folder, f"{name}_chars.csv"),
                                     encoding="utf-8")))
    text = open(os.path.join(folder, f"{name}.txt"), encoding="utf-8").read()
    return quotes, chars, text


def build(folder, name, context_chars=400):
    quotes, chars, text = load_novel(folder, name)
    aliases, category, roster = [], {}, []
    for c in chars:
        main = (c.get("Main Name") or "").strip()
        if not main:
            continue
        roster.append(main.upper())
        category[main.upper()] = (c.get("Category") or "").strip()
        try:
            alt = ast.literal_eval(c.get("Aliases") or "set()")
        except Exception:
            alt = set()
        group = {main.upper()} | {str(a).upper() for a in alt if str(a).strip()}
        if len(group) > 1:
            aliases.append(sorted(group))

    entries, skipped = [], collections.Counter()
    for n, q in enumerate(quotes):
        line = (q.get("quoteText") or "").strip()
        speaker = (q.get("speaker") or "").strip().upper()
        if not line or not speaker or speaker in SPECIAL:
            skipped["no speaker"] += 1
            continue
        # Context comes from the byte spans, so the model sees the same
        # surroundings a reader would - not a reconstruction.
        try:
            spans = ast.literal_eval(q.get("quoteByteSpans") or "[]")
            start = min(s[0] for s in spans)
            end = max(s[1] for s in spans)
        except Exception:
            skipped["no span"] += 1
            continue
        entries.append({
            "id": f"{name}-{n:05d}",
            "line": line,
            "expected_speaker": speaker,
            "quote_type": (q.get("quoteType") or "").strip(),
            "category": category.get(speaker, "unknown"),
            "prev_context": text[max(0, start - context_chars):start].strip(),
            "next_context": text[end:end + context_chars].strip(),
        })
    return {"book": name, "source": "PDNC", "entries": entries,
            "aliases": aliases, "roster": sorted(set(roster)),
            "skipped": dict(skipped)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folder", required=True, help="directory of PDNC downloads")
    ap.add_argument("--novels", nargs="+", required=True)
    ap.add_argument("--out_dir", default=REPO + "/app/fixtures")
    args = ap.parse_args()

    for name in args.novels:
        fx = build(args.folder, name)
        by_type = collections.Counter(e["quote_type"] for e in fx["entries"])
        by_cat = collections.Counter(e["category"] for e in fx["entries"])
        out = os.path.join(args.out_dir, f"attribution_gold_pdnc_{name.lower()}.json")
        json.dump(fx, open(out, "w"), ensure_ascii=False, indent=1)
        print(f"{name}: {len(fx['entries'])} quotations, "
              f"{len(fx['roster'])} characters, {len(fx['aliases'])} alias groups")
        print(f"   by type: {dict(by_type)}")
        print(f"   by category: {dict(by_cat)}")
        if fx["skipped"]:
            print(f"   skipped: {fx['skipped']}")
        print(f"   wrote {out}")


if __name__ == "__main__":
    main()
