"""Does the adapter leave anything for the fallback policy to fix?

Two independent gains are on the table: the adapter is worth +5.4 pooled in the
shippable stack, and `fallback_policy` found gated substitution worth +3.3 more
with no extra inference. Whether they ADD is not obvious, because they are not
independent mechanisms:

  the fallback triggers on two cheap passes DISAGREEING
  the adapter changes what both passes say

If the tuned model agrees with itself more often, the fallback fires on fewer
rows and its contribution shrinks. If the rows where it still disagrees are
also the rows it now gets right, the fallback has nothing left to repair. Both
would mean the +3.3 does not survive on top of the +5.4.

This compares untuned and tuned runs of the SAME two-pass setup, over the same
gold, from artifacts already on disk:

  disagreement rate    how much the fallback has to work with at all
  agree accuracy       is the confident set still trustworthy?
  disagree accuracy    is the uncertain set still bad enough to be worth
                       replacing?
  headroom             disagreement share x (1 - disagree accuracy), the
                       fraction of the book a perfect fallback could fix

A shrinking headroom means the adapter has already eaten the fallback's
opportunity, and stacking the two would disappoint. A stable one means they
address different failures and the gains should add.
"""
import argparse, collections, glob, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = REPO + "/app/"
sys.path.insert(0, APP)
from experiments.scoring import alias_groups, same_speaker

LEDGER = REPO + "/ab_test_runtime/experiments"
BOOKS = ("grimgar03", "index18", "mushoku16", "owarimonogatari3")
SPECIAL = {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE"}


def gold_for(book):
    path = APP + f"fixtures/attribution_gold_{book}.json"
    if not os.path.exists(path):
        return None, None
    doc = json.load(open(path))
    want = {g["id"]: g["expected_speaker"].upper() for g in doc["entries"]
            if g["expected_speaker"].upper() not in SPECIAL}
    return want, alias_groups(doc)


def short_id(gid):
    return str(int(gid.rsplit("-", 1)[-1])) if "-" in gid else gid


def measure(path, book):
    """Disagreement rate and the accuracy on each side of it."""
    state = json.load(open(path))
    w1, w4 = state.get("w1") or {}, state.get("w4") or {}
    want, groups = gold_for(book)
    if not want or not w1 or not w4:
        return None
    agree = [0, 0]
    disagree = [0, 0]
    for gid, expected in want.items():
        key = short_id(gid)
        a = (w1.get(key) or "").upper()
        b = (w4.get(key) or "").upper()
        if not a or not b:
            continue
        bucket = agree if same_speaker(a, b, groups) else disagree
        bucket[0] += 1
        bucket[1] += same_speaker(expected, a, groups)
    total = agree[0] + disagree[0]
    if not total or not disagree[0]:
        return None
    return {
        "n": total,
        "disagree_share": disagree[0] / total,
        "agree_acc": agree[1] / agree[0] if agree[0] else float("nan"),
        "disagree_acc": disagree[1] / disagree[0],
        # What a PERFECT fallback could recover: the disagreement rows it
        # currently gets wrong, as a share of the whole book.
        "headroom": disagree[0] / total * (1 - disagree[1] / disagree[0]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=LEDGER + "/tuned_disagreement.json")
    args = ap.parse_args()

    rows = {}
    for book in BOOKS:
        untuned = None
        for path in sorted(glob.glob(LEDGER + f"/cascade_state__{book}__*.json")):
            if "tuned-cheap-arm" in path:
                continue
            untuned = measure(path, book)
            if untuned:
                break
        tuned_path = LEDGER + f"/cascade_state__{book}__tuned-cheap-arm.json"
        tuned = measure(tuned_path, book) if os.path.exists(tuned_path) else None
        if untuned and tuned:
            rows[book] = {"untuned": untuned, "tuned": tuned}

    if not rows:
        print("Need both an untuned and a tuned cascade_state per book.")
        return

    print("Two cheap passes, before and after the adapter\n")
    print(f"  {'book':18}{'arm':9}{'disagree%':>11}{'agree acc':>11}"
          f"{'disagree acc':>14}{'headroom':>10}")
    for book, pair in rows.items():
        for arm in ("untuned", "tuned"):
            m = pair[arm]
            print(f"  {book:18}{arm:9}{m['disagree_share']*100:10.1f}%"
                  f"{m['agree_acc']*100:10.1f}%{m['disagree_acc']*100:13.1f}%"
                  f"{m['headroom']*100:9.1f}%")

    def pooled(arm, key):
        num = sum(pair[arm][key] * pair[arm]["n"] for pair in rows.values())
        den = sum(pair[arm]["n"] for pair in rows.values())
        return num / den

    print(f"\n  pooled")
    for arm in ("untuned", "tuned"):
        print(f"    {arm:9} disagree {pooled(arm,'disagree_share')*100:5.1f}%   "
              f"agree acc {pooled(arm,'agree_acc')*100:5.1f}%   "
              f"disagree acc {pooled(arm,'disagree_acc')*100:5.1f}%   "
              f"headroom {pooled(arm,'headroom')*100:5.1f}%")

    delta = pooled("tuned", "headroom") - pooled("untuned", "headroom")
    print(f"\n  headroom change {delta*100:+.1f} points")
    if delta < -0.02:
        print("  -> the adapter has absorbed most of what a fallback could "
              "fix. Stacking\n     the two would disappoint, and the +3.3 "
              "measured on the untuned model\n     should not be added to the "
              "+5.4.")
    else:
        print("  -> the opportunity survives the adapter, so the two address "
              "different\n     failures and the gains should broadly add. "
              "Worth measuring live rather\n     than assuming - this bounds "
              "it, it does not prove it.")

    json.dump(rows, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
