"""Are the hard errors off-by-one turns, or 91 independent hard judgements?

`residual_errors` found the remaining core is 91 LEAD CONFUSIONS - two major
characters swapped, 11.8% of all rows, and the failure a listener notices most.
The question is whether they share a structure.

`committed_history` found something specific that suggests they might: on
unanchored alternating dialogue the models do not guess randomly, they converge
on the WRONG TURN. All nine runs answered MOGUZO where the answer was RANTA;
eight answered RON for RANTA. That is a parity error - the model has the
conversation right and its phase wrong.

If the residual lead confusions are mostly the model assigning line N the
speaker of an ADJACENT line, they are one systematic error rather than 91
independent ones, and a parity correction could address many at once. If they
are not, no structural fix exists and the remaining core is genuinely hard.

FOUR BUCKETS per wrong prediction:

    prev speaker     the prediction is the true speaker of the previous
                     scored line - the model is one turn behind
    next speaker     the true speaker of the following line - one turn ahead
    nearby           the true speaker of some line within +-3, but not
                     immediately adjacent
    unrelated        a character who does not speak nearby at all

A BASELINE IS REQUIRED, because in a two-hander the previous speaker is often
the ONLY other candidate, so "matches the previous line" happens by chance at a
high rate. The same buckets are therefore computed for the CORRECT predictions
too: if wrong answers match the previous speaker no more often than right ones
do, the pattern is the book's structure and not the model's error.
"""
import argparse, collections, glob, json, os, re, sys

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
APP = REPO + "/app/"
sys.path.insert(0, APP)
from experiments.scoring import alias_groups, same_speaker

M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
LEDGER = REPO + "/ab_test_runtime/experiments"
BOOKS = ("grimgar03", "index18", "mushoku16", "owarimonogatari3")
SPECIAL = {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE"}


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


def book_context(book):
    """Scored lines in reading order, with their gold speakers."""
    path = APP + f"fixtures/attribution_gold_{book}.json"
    if not os.path.exists(path):
        return None
    gold = json.load(open(path))
    seg = json.load(open(
        M + INPUT_RUN + f"/{book}/result.json.threepass_checkpoint.json"))["segmented"]
    occ = collections.Counter(norm(e.get("text")) for e in seg)
    pos = {}
    for index, entry in enumerate(seg):
        key = norm(entry.get("text"))
        if occ[key] == 1:
            pos[key] = index
    rows = []
    for g in gold["entries"]:
        if g["expected_speaker"].upper() in SPECIAL:
            continue
        key = norm(g["line"])
        if key in pos:
            rows.append((pos[key], g["id"], g["expected_speaker"].upper()))
    rows.sort()
    order = {gid: i for i, (_, gid, _) in enumerate(rows)}
    truth = [spk for _, _, spk in rows]
    return {"order": order, "truth": truth, "groups": alias_groups(gold)}


def bucket(index, predicted, ctx):
    truth, groups = ctx["truth"], ctx["groups"]
    if index > 0 and same_speaker(truth[index - 1], predicted, groups):
        return "prev speaker"
    if index + 1 < len(truth) and same_speaker(truth[index + 1], predicted, groups):
        return "next speaker"
    lo, hi = max(0, index - 3), min(len(truth), index + 4)
    for j in range(lo, hi):
        if j != index and same_speaker(truth[j], predicted, groups):
            return "nearby"
    return "unrelated"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", default="lora")
    ap.add_argument("--out", default=LEDGER + "/offbyone_turns.json")
    args = ap.parse_args()

    ctxs = {b: book_context(b) for b in BOOKS}
    ctxs = {b: c for b, c in ctxs.items() if c}
    wrong = collections.Counter()
    right = collections.Counter()
    per_book = collections.defaultdict(collections.Counter)

    paths = sorted(glob.glob(LEDGER + "/lora_serving_eval__*.json"))
    if not paths:
        print("no lora_serving_eval artifact yet")
        return
    for path in paths:
        doc = json.load(open(path))
        for row in doc["rows"]:
            if row["arm"] != args.arm:
                continue
            book, _, gid = row["id"].partition(":")
            ctx = ctxs.get(book)
            predicted = (row.get("predicted") or "").upper()
            if not ctx or not predicted or gid not in ctx["order"]:
                continue
            index = ctx["order"][gid]
            b = bucket(index, predicted, ctx)
            if row.get("correct"):
                right[b] += 1
            else:
                wrong[b] += 1
                per_book[book][b] += 1

    total_wrong = sum(wrong.values())
    total_right = sum(right.values())
    if not total_wrong:
        print("no wrong answers found")
        return

    print(f"{args.arm} arm: {total_wrong} wrong predictions with a speaker, "
          f"{total_right} right\n")
    print(f"  {'bucket':16}{'wrong':>8}{'of wrong':>11}"
          f"{'right':>8}{'of right':>11}{'lift':>8}")
    order = ["prev speaker", "next speaker", "nearby", "unrelated"]
    for b in order:
        w = wrong[b] / total_wrong
        r = right[b] / max(total_right, 1)
        print(f"  {b:16}{wrong[b]:8}{w*100:10.1f}%{right[b]:8}{r*100:10.1f}%"
              f"{(w-r)*100:+8.1f}")

    adjacent = (wrong["prev speaker"] + wrong["next speaker"]) / total_wrong
    adjacent_right = (right["prev speaker"] + right["next speaker"]) / max(total_right, 1)
    print(f"\n  adjacent-turn share: wrong {adjacent*100:.1f}% vs "
          f"right {adjacent_right*100:.1f}%  ({(adjacent-adjacent_right)*100:+.1f})")

    print("\n  per book, wrong answers only")
    print(f"  {'book':18}" + "".join(f"{b[:11]:>13}" for b in order))
    for book in sorted(per_book):
        print(f"  {book:18}" + "".join(f"{per_book[book][b]:13}" for b in order))

    if adjacent - adjacent_right > 0.10:
        print("\n  -> Wrong answers land on an adjacent turn far more often "
              "than right ones\n     do. That is a PARITY error: the model has "
              "the conversation and the\n     wrong phase. A correction that "
              "detects alternation and shifts it is\n     worth building, and "
              "it would address many errors at once.")
    else:
        print("\n  -> Wrong answers are no more adjacent than right ones, so "
              "the pattern is\n     the book's two-hander structure rather "
              "than a systematic phase error.\n     No parity fix exists; the "
              "remaining core is genuinely hard.")

    json.dump({"arm": args.arm, "wrong": dict(wrong), "right": dict(right),
               "per_book": {b: dict(c) for b, c in per_book.items()}},
              open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
