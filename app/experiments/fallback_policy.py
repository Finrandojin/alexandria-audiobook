"""When the two passes disagree, is guessing better than falling back?

`abstention_and_rules` established the trigger: where two cheap passes agree the
model is right 78.1% of the time, where they disagree only 30.4% - a 47.7-point
lift over 36% of lines. That says the system KNOWS which lines it is likely to
get wrong. It does not say what to do about them, and today it does nothing:
UNKNOWN appears on 0-1.1% of lines, so every one of those uncertain rows becomes
a confident wrong guess.

This scores the alternatives on the rows where the passes disagree:

    keep        use the first pass's answer - what production does today
    previous    reuse the speaker assigned to the previous scored line, under
                the same policy, so the substitution is one production could
                actually make
    narrator    hand the line to the narrator voice
    second      use the SECOND pass's answer instead of the first, as a control:
                if this matches `keep`, the two passes are interchangeable and
                the disagreement carries no directional information

NO GPU AND NO NEW INFERENCE. Both passes were already run and stored in
`cascade_state`, so every policy here is a re-scoring of answers on disk. That
is also the point: a policy that needs no extra inference is free to ship.

WHAT WOULD MAKE THIS WORTH SHIPPING. `previous` or `narrator` beating `keep` on
the disagreement rows, by enough to survive the whole-book dilution - these are
36% of lines, so a 5-point gain there is under 2 points overall.

THE HONEST FAILURE MODE. `narrator` can look good on a book whose narrator
speaks often, because it collects rows that were narration all along. The
per-book narrator share is printed next to the result so that is visible rather
than mistaken for a policy win.
"""
import argparse, collections, glob, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = REPO + "/app/"
sys.path.insert(0, APP)
from experiments.scoring import alias_groups, same_speaker
from experiments.stats import clopper_pearson

LEDGER = REPO + "/ab_test_runtime/experiments"
BOOKS = ("grimgar03", "index18", "mushoku16", "owarimonogatari3")
SPECIAL = {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE"}
POLICIES = ("keep", "second", "previous", "narrator")


def gold_for(book):
    path = APP + f"fixtures/attribution_gold_{book}.json"
    if not os.path.exists(path):
        return None, None, None
    doc = json.load(open(path))
    ordered = [(g["id"], g["expected_speaker"].upper()) for g in doc["entries"]
               if g["expected_speaker"].upper() not in SPECIAL]
    counts = collections.Counter(s for _, s in ordered)
    return ordered, alias_groups(doc), counts


def short_id(gid):
    """cascade_state stores 'grimgar03-00001' as '1'."""
    return str(int(gid.rsplit("-", 1)[-1])) if "-" in gid else gid


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=LEDGER + "/fallback_policy.json")
    args = ap.parse_args()

    seen_books, results = set(), {}
    totals = {p: [0, 0] for p in POLICIES}
    whole = {p: [0, 0] for p in POLICIES}
    print(f"  {'book':18}{'disagree':>9}{'keep':>8}{'second':>8}"
          f"{'previous':>10}{'narrator':>10}{'narr share':>12}")
    for path in sorted(glob.glob(LEDGER + "/cascade_state__*.json")):
        book = next((b for b in BOOKS if b in os.path.basename(path)), None)
        # One artifact per book: several runs share a book and re-counting them
        # would weight that book more heavily for no reason.
        if not book or book in seen_books:
            continue
        state = json.load(open(path))
        w1, w4 = state.get("w1") or {}, state.get("w4") or {}
        ordered, groups, counts = gold_for(book)
        if not ordered or not w1 or not w4:
            continue
        seen_books.add(book)
        narrator_share = counts.get("NARRATOR", 0) / max(sum(counts.values()), 1)

        per_policy = {p: [0, 0] for p in POLICIES}
        whole_book = {p: [0, 0] for p in POLICIES}
        previous_answer = {p: None for p in POLICIES}
        for gid, expected in ordered:
            key = short_id(gid)
            a = (w1.get(key) or "").upper()
            b = (w4.get(key) or "").upper()
            if not a or not b:
                continue
            disagreed = not same_speaker(a, b, groups)
            for policy in POLICIES:
                if not disagreed:
                    chosen = a
                elif policy == "keep":
                    chosen = a
                elif policy == "second":
                    chosen = b
                elif policy == "previous":
                    chosen = previous_answer[policy] or a
                else:
                    chosen = "NARRATOR"
                right = same_speaker(expected, chosen, groups)
                whole_book[policy][0] += 1
                whole_book[policy][1] += right
                if disagreed:
                    per_policy[policy][0] += 1
                    per_policy[policy][1] += right
                previous_answer[policy] = chosen

        n = per_policy["keep"][0]
        if not n:
            continue
        row = {p: per_policy[p][1] / n for p in POLICIES}
        results[book] = {"disagree_n": n, "narrator_share": narrator_share,
                         "on_disagreement": row,
                         "whole_book": {p: whole_book[p][1] / max(whole_book[p][0], 1)
                                        for p in POLICIES}}
        for p in POLICIES:
            totals[p][0] += n
            totals[p][1] += per_policy[p][1]
            whole[p][0] += whole_book[p][0]
            whole[p][1] += whole_book[p][1]
        print(f"  {book:18}{n:9}{row['keep']*100:7.1f}%{row['second']*100:7.1f}%"
              f"{row['previous']*100:9.1f}%{row['narrator']*100:9.1f}%"
              f"{narrator_share*100:11.1f}%")

    if not totals["keep"][0]:
        print("\nNo cascade_state artifact carries both passes.")
        return

    print("\n  pooled, ON THE DISAGREEMENT ROWS ONLY")
    base = totals["keep"][1] / totals["keep"][0]
    for p in POLICIES:
        acc = totals[p][1] / totals[p][0]
        lo, hi = clopper_pearson(totals[p][1], totals[p][0])
        mark = "" if p == "keep" else f"  {(acc-base)*100:+.1f} vs keep"
        print(f"    {p:9} {totals[p][1]:4}/{totals[p][0]:<5} = {acc*100:5.1f}% "
              f"[{lo:.1f}-{hi:.1f}]{mark}")

    print("\n  pooled, WHOLE BOOK (what a listener gets)")
    wbase = whole["keep"][1] / whole["keep"][0]
    for p in POLICIES:
        acc = whole[p][1] / whole[p][0]
        mark = "" if p == "keep" else f"  {(acc-wbase)*100:+.1f} vs keep"
        print(f"    {p:9} {acc*100:5.1f}%{mark}")

    # `second` is the WIDER-CONTEXT pass (w4 against w1's width 1), so its
    # win largely restates what context_width already showed and is not a
    # finding about uncertainty. `previous` is the informative one: it uses no
    # model at all on the rows it changes.
    # A narrator arm is meaningless on a fixture with no NARRATOR lines, and
    # these gold sets have none - it can only score 0.
    narr_possible = any(r["narrator_share"] > 0 for r in results.values())
    if not narr_possible:
        print("\n  NOTE: no book here has NARRATOR gold lines, so the narrator "
              "policy could\n  only ever score 0. That arm is vacuous on this "
              "fixture, not refuted.")
    candidates = [p for p in POLICIES if p not in ("keep", "narrator")]
    best = max(candidates, key=lambda p: whole[p][1] / whole[p][0])
    gain = whole[best][1] / whole[best][0] - wbase
    print(f"\n  Best alternative: {best}, {gain*100:+.1f} points whole-book.")
    if gain > 0.01:
        print("  Worth shipping: it needs no extra inference, because both "
              "passes are\n  already run. Check the narrator share above before "
              "believing a\n  narrator win - on a narrator-heavy book that "
              "policy collects rows that\n  were narration all along.")
    else:
        print("  Not worth shipping. The trigger is real - agreement predicts "
              "correctness\n  by 47.7 points - but none of these fallbacks "
              "beats simply guessing, so\n  the value of the signal is in "
              "deciding what to ESCALATE, not what to\n  substitute.")

    json.dump({"per_book": results,
               "pooled_disagreement": {p: totals[p][1] / totals[p][0]
                                       for p in POLICIES},
               "pooled_whole_book": {p: whole[p][1] / whole[p][0]
                                     for p in POLICIES}},
              open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
