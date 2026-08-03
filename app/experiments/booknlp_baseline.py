"""Does a 14B LLM beat the field-standard attribution tool, or lose to it?

Every accuracy in this ledger is measured against OUR OWN base model. The
adapter's +5.4, the 91 residual lead confusions, the PDNC spread from +10.0 to
-12.5 - all of it is self-referential. None of it says whether the whole
approach beats a purpose-built tool that has existed the entire time.

BookNLP (Bamman et al., the LitBank group) has a quotation-attribution stage
trained on annotated fiction: BERT-scale, no prompting, no adapter, seconds per
chapter instead of minutes. VoxNovel - which solves the same per-character-voice
problem this project does - uses it and no LLM at all.

WHAT EACH OUTCOME MEANS.

  BookNLP clearly worse    The LLM approach is justified against the standard
                           baseline, which has never been shown before.
  BookNLP comparable       A far cheaper method matches months of work, and
                           the honest move is a hybrid or a switch.
  BookNLP clearly better   The ledger has been optimising the wrong method,
                           and that is worth knowing however unwelcome.

A REASON IT MAY DO BADLY, STATED IN ADVANCE so it is not an excuse invented
after a bad number: BookNLP is trained on English-original literature. Half this
corpus is translated Japanese light novels with dense honorific and alias
structure - the thing character_aliases.json exists to handle. The PDNC books
(Austen, Doyle, Chopin) are exactly its training distribution, so the split
between PDNC and light-novel books is the informative comparison, not the pooled
figure.

Scoring is alias-aware via experiments.scoring, the same as every other harness,
so BookNLP is not penalised for saying ELIZABETH where gold says MISS BENNET.

Idea credit: DrewThomasson/VoxNovel (MIT) for identifying BookNLP as the
baseline. No code taken; see THIRD_PARTY_NOTICES.md.

USAGE
    pip install booknlp        # pulls its own models on first run
    python booknlp_baseline.py --book pdnc_prideandprejudice --text <path.txt>
    python booknlp_baseline.py --book grimgar03 --booknlp-dir <existing output>
"""
import argparse, collections, csv, json, os, re, sys

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
APP = REPO + "/app/"
sys.path.insert(0, APP)
from experiments.scoring import alias_groups, same_speaker
from experiments.stats import exact_mcnemar

LEDGER = REPO + "/ab_test_runtime/experiments"
SPECIAL = {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE"}


def norm(text):
    """Same normalisation offbyone_turns uses to match lines across sources."""
    return re.sub(r"\W+", "", text or "").lower()


def read_tsv(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def character_names(entity_rows):
    """BookNLP character id -> its most frequent PROPER name.

    The .entities file gives every mention of every entity. A character is
    referred to as "Elizabeth", "she", "Miss Bennet", "her sister"; only proper
    mentions are usable as a speaker label, and the most common one is the name
    a reader would use. Falls back to the commonest mention of any kind so a
    character with no proper mention still gets a stable label rather than being
    silently dropped - dropping it would flatter BookNLP by removing its hard
    cases from the denominator.
    """
    proper = collections.defaultdict(collections.Counter)
    any_kind = collections.defaultdict(collections.Counter)
    for row in entity_rows:
        cid = row.get("COREF")
        text = (row.get("text") or "").strip()
        if not cid or not text:
            continue
        any_kind[cid][text] += 1
        if (row.get("prop") or "").upper() == "PROP":
            proper[cid][text] += 1
    names = {}
    for cid in any_kind:
        source = proper.get(cid) or any_kind[cid]
        names[cid] = source.most_common(1)[0][0]
    return names


def parse_booknlp(quotes_rows, entity_rows):
    """-> [(normalised quote text, predicted speaker name)].

    A quote whose char_id is -1 (or absent) is one BookNLP declined to attribute.
    Those are kept with an empty speaker: they are wrong answers, not absent
    rows, and discarding them would inflate its accuracy.
    """
    names = character_names(entity_rows)
    out = []
    for row in quotes_rows:
        text = row.get("quote") or ""
        cid = (row.get("char_id") or "").strip()
        speaker = names.get(cid, "") if cid not in ("", "-1") else ""
        out.append((norm(text), speaker.upper()))
    return out


# Normalisation strips whitespace, so substring matching has no word
# boundaries: "no" is inside "nothing". A floor is therefore needed against
# spurious hits, but it must stay below the length of a real interrupted first
# half - "My dear Mr. Bennet," normalises to 14 characters, and a floor of 20
# discarded it, which hid the speaker conflicts this is meant to count.
MIN_FRAGMENT = 10      # normalised chars
MIN_COVERAGE = 0.6     # of the gold line


def align_to_gold(booknlp_rows, gold):
    """Match BookNLP quotes to gold lines by normalised text.

    Only lines appearing exactly ONCE in each source are matched. A repeated
    line ("Yes." appears forty times) cannot be aligned by text alone, and
    guessing which occurrence is which would fabricate agreement.

    INTERRUPTED QUOTES are why this is not a dict lookup. PDNC records

        "My dear Mr. Bennet, have you heard that Netherfield Park is let?"

    as ONE quotation, but the novel splits it around `," said his lady, "` and
    BookNLP emits TWO quotes. On Pride and Prejudice that is 420 of 1270 gold
    lines - a third of the book, and disproportionately the harder cases, since
    an interrupted quote is exactly where attribution gets difficult. Requiring
    exact text would drop every one of them and flatter the baseline.

    So a gold line also matches a SET of BookNLP quotes that are each substrings
    of it, subject to two guards: each fragment must be long enough not to be an
    incidental phrase, and the fragments together must cover most of the gold
    line, so a passing mention inside a long speech cannot claim it.

    The prediction is the speaker of the LONGEST fragment. When BookNLP gives
    the halves different speakers it is counted as answering with one of them,
    not excused - `conflicts` reports how often that happened.
    """
    bn_counts = collections.Counter(t for t, _ in booknlp_rows)
    bn = {t: s for t, s in booknlp_rows if bn_counts[t] == 1 and t}
    gold_entries = [g for g in gold["entries"]
                    if g["expected_speaker"].upper() not in SPECIAL]
    g_counts = collections.Counter(norm(g["line"]) for g in gold_entries)

    matched, unmatched, conflicts = [], 0, 0
    for g in gold_entries:
        key = norm(g["line"])
        if g_counts[key] != 1:
            unmatched += 1
            continue
        if key in bn:
            matched.append({"id": g["id"],
                            "expected": g["expected_speaker"].upper(),
                            "predicted": bn[key], "split": False})
            continue
        frags = [(t, s) for t, s in bn.items()
                 if len(t) >= MIN_FRAGMENT and t in key]
        if not frags or sum(len(t) for t, _ in frags) < len(key) * MIN_COVERAGE:
            unmatched += 1
            continue
        frags.sort(key=lambda ts: len(ts[0]), reverse=True)
        if len({s for _, s in frags}) > 1:
            conflicts += 1
        matched.append({"id": g["id"],
                        "expected": g["expected_speaker"].upper(),
                        "predicted": frags[0][1], "split": True})
    return matched, unmatched, conflicts


def run_booknlp(text_path, out_dir, book_id):
    """Invoke BookNLP. Imported lazily so this module loads without it."""
    from booknlp.booknlp import BookNLP
    os.makedirs(out_dir, exist_ok=True)
    BookNLP("en", {"pipeline": "entity,quote,supersense,event,coref",
                   "model": "big"}).process(text_path, out_dir, book_id)
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--book", required=True, help="gold fixture name")
    ap.add_argument("--text", help="plain-text book to run BookNLP over")
    ap.add_argument("--booknlp-dir", help="reuse an existing BookNLP output dir")
    ap.add_argument("--book-id", help="BookNLP file stem (defaults to --book)")
    ap.add_argument("--out")
    args = ap.parse_args()

    gold_path = APP + f"fixtures/attribution_gold_{args.book}.json"
    if not os.path.exists(gold_path):
        sys.exit(f"no gold fixture: {gold_path}")
    gold = json.load(open(gold_path))
    book_id = args.book_id or args.book

    out_dir = args.booknlp_dir
    if not out_dir:
        if not args.text:
            sys.exit("need --text to run BookNLP, or --booknlp-dir to reuse a run")
        out_dir = run_booknlp(args.text, LEDGER + f"/booknlp_{args.book}", book_id)

    quotes_path = os.path.join(out_dir, f"{book_id}.quotes")
    entities_path = os.path.join(out_dir, f"{book_id}.entities")
    for p in (quotes_path, entities_path):
        if not os.path.exists(p):
            sys.exit(f"missing BookNLP output: {p}")

    rows = parse_booknlp(read_tsv(quotes_path), read_tsv(entities_path))
    matched, unmatched, conflicts = align_to_gold(rows, gold)
    if not matched:
        sys.exit("no gold line matched a BookNLP quote - check the text source "
                 "is the same edition the gold was built from")

    groups = alias_groups(gold)
    for m in matched:
        m["correct"] = bool(m["predicted"]) and same_speaker(
            m["expected"], m["predicted"], groups)
    correct = sum(m["correct"] for m in matched)
    declined = sum(1 for m in matched if not m["predicted"])

    print(f"BookNLP on {args.book}\n")
    split = sum(1 for m in matched if m.get("split"))
    print(f"  {len(matched)} gold lines aligned, {unmatched} unaligned "
          f"(repeated or absent text)")
    print(f"  {split} matched as interrupted quotes, {conflicts} of those "
          f"given conflicting speakers by BookNLP")
    print(f"  accuracy {correct / len(matched) * 100:.1f}%  "
          f"({correct}/{len(matched)})")
    print(f"  declined to attribute: {declined} "
          f"({declined / len(matched) * 100:.1f}%)")

    llm = {}
    for path in sorted(__import__("glob").glob(
            LEDGER + "/lora_serving_eval__*.json")):
        for row in json.load(open(path))["rows"]:
            if row["arm"] == "lora":
                llm[row["id"]] = bool(row.get("correct"))
    paired = [m for m in matched if f"{args.book}:{m['id']}" in llm]
    if paired:
        b = sum(llm[f"{args.book}:{m['id']}"] for m in paired)
        n = sum(m["correct"] for m in paired)
        gained = sum(1 for m in paired
                     if m["correct"] and not llm[f"{args.book}:{m['id']}"])
        lost = sum(1 for m in paired
                   if llm[f"{args.book}:{m['id']}"] and not m["correct"])
        p = exact_mcnemar(lost, gained)[0]
        print(f"\n  head to head on {len(paired)} shared lines")
        print(f"    our LoRA stack  {b / len(paired) * 100:.1f}%")
        print(f"    BookNLP         {n / len(paired) * 100:.1f}%  "
              f"({(n - b) / len(paired) * 100:+.1f})")
        print(f"    +{gained}/-{lost}  p={p:.4g}")
    else:
        print("\n  no shared rows with a lora_serving_eval artifact; "
              "reporting BookNLP alone")

    out = args.out or LEDGER + f"/booknlp_baseline__{args.book}.json"
    json.dump({"book": args.book, "n": len(matched), "unaligned": unmatched,
               "accuracy": correct / len(matched), "declined": declined,
               "split_matched": split, "split_conflicts": conflicts,
               "rows": matched}, open(out, "w"), indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
