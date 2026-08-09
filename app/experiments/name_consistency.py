"""Goal 5.2: is a character's name spelled one way, or several, in a book?

WHY SPELLING IS THE RIGHT PROXY. Qwen3-TTS pronounces what it is given. Two
spellings of the same name are two pronunciations, and the listener hears the
same character called two different things - the defect 5.2 is about. Nothing
in the app currently counts it: `apply_pronunciation` can respell a name it has
a lexicon entry for, but no measurement says how often a book needs one.

WHAT COUNTS AS THE SAME NAME. Variants are grouped by a fold that removes only
what does not change identity: case, accents, and the separators between parts
of a name. So `Zoe`/`Zoë`, `Haruhiro`/`haruhiro`, and `Bri-chan`/`Bri chan`
group together, while `Merry` and `Mary` stay apart - they are different names,
not variant spellings, and merging them would invent a defect.

WHY THE TEXT AND NOT THE SPEAKER LABEL. The first version of this counted
variant spellings among `speaker` labels and found 0 of 777 - because those
labels are upper-cased and canonicalised upstream, so they cannot vary by
construction. It was measuring a surface that is protected, the same error as
auditing TTS input instead of TTS output. What the engine actually speaks is
the name inside the line text, so that is what is scanned.

WHY THE SPEAKER LIST IS THE VOCABULARY. The names that matter are the ones the
app assigns voices to; an inconsistently spelled word that is not a character
is a typo, not a pronunciation defect. Every book's `speaker` field gives that
list for free, and the same fold is applied to it.

WHAT THIS DOES NOT MEASURE. Whether the TTS pronounces a *single* spelling
correctly. That needs a listener or an ASR pass over generated audio, and it is
a different goal - this counts only the case where the app itself is
inconsistent, which is the half that can be fixed by text.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
import unicodedata

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if not os.path.isdir(os.path.join(REPO, "scripts")):
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "app"))

SEPARATORS = re.compile(r"[\s\-'’_.]+")


def fold(name):
    """Identity-preserving key: case, accents and separators removed."""
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return SEPARATORS.sub("", text).casefold()


def script_entries(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("entries") or payload.get("lines") or []
    return payload if isinstance(payload, list) else []


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scripts", default=os.path.join(REPO, "scripts"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "name_consistency.json"))
    args = ap.parse_args()

    paths = sorted(p for p in glob.glob(os.path.join(args.scripts, "*.json"))
                   if not p.endswith(".generation_quality.json"))
    per_book, total_names, inconsistent_names = {}, 0, 0
    lexicon = set()
    lex_path = os.path.join(REPO, "pronunciation.json")
    if os.path.exists(lex_path):
        try:
            with open(lex_path, encoding="utf-8") as handle:
                lexicon = {fold(k) for k in json.load(handle)}
        except (ValueError, TypeError):
            lexicon = set()

    for path in paths:
        book = os.path.basename(path)[:-len(".json")]
        entries = script_entries(path)
        if not entries:
            continue
        speakers = collections.Counter()
        for entry in entries:
            speaker = (entry or {}).get("speaker")
            if speaker:
                speakers[speaker] += 1
        # Group the speaker labels themselves by folded key. Two labels folding
        # together are the same character written two ways, which is the
        # defect: the app has already assigned them separate rows.
        # Surface forms of each character's name AS WRITTEN IN THE PROSE,
        # which is the string handed to the engine.
        wanted = {}
        for name in speakers:
            key = fold(name)
            if len(key) >= 3:
                wanted[key] = name
        groups = collections.defaultdict(collections.Counter)
        token = re.compile(r"[^\W\d_]+(?:[\-'\u2019][^\W\d_]+)*", re.UNICODE)
        for entry in entries:
            for word in token.findall((entry or {}).get("text") or ""):
                key = fold(word)
                if key in wanted:
                    groups[key][word] += 1
        # Case alone is NOT a defect: the engine says "Felt" and "felt"
        # identically, and sentence-initial capitals would otherwise flag
        # every character whose name is also a common word. A group counts
        # only when the forms still differ once case is removed - an accent,
        # a hyphen, or a different letter.
        varied = {}
        for key, forms in groups.items():
            shapes = {f.casefold() for f in forms}
            if len(shapes) > 1:
                varied[key] = dict(forms)
        total_names += len(groups)
        inconsistent_names += len(varied)
        if varied:
            per_book[book] = {
                "distinct_characters": len(groups),
                "inconsistent": len(varied),
                "covered_by_lexicon": sum(1 for k in varied if k in lexicon),
                "examples": {k: v for k, v in list(varied.items())[:6]},
            }

    result = {
        "scope": "character names as written in the prose; groups differing "
                 "only by capitalisation are excluded, since the engine "
                 "speaks those identically",
        "books": len(paths),
        "distinct_characters": total_names,
        "characters_with_multiple_spellings": inconsistent_names,
        "inconsistency_pct": round(
            100.0 * inconsistent_names / total_names, 2) if total_names else None,
        "lexicon_entries": len(lexicon),
        "per_book": per_book,
    }

    from utils import atomic_json_write
    atomic_json_write(result, args.out)

    print("=== goal 5.2: name spelling consistency ===")
    print(f"  {len(paths)} books, {total_names} distinct characters")
    print(f"  characters written more than one way: {inconsistent_names} "
          f"({result['inconsistency_pct']}%)")
    print(f"  pronunciation lexicon entries: {len(lexicon)}")
    if per_book:
        print("\n  worst books:")
        for book, data in sorted(per_book.items(),
                                 key=lambda kv: -kv[1]["inconsistent"])[:8]:
            print(f"    {book[:36]:38} {data['inconsistent']:3} of "
                  f"{data['distinct_characters']:3}  lexicon-covered "
                  f"{data['covered_by_lexicon']}")
        first = next(iter(per_book.values()))["examples"]
        print("\n  example variant groups:")
        for key, forms in list(first.items())[:4]:
            print(f"    {key[:22]:24} {forms}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
