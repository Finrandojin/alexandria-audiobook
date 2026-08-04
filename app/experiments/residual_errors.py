"""What is left in the quarter of lines the adapter still gets wrong?

The shippable configuration reaches 74.2% pooled. Nobody has characterised the
remaining 25.8%, so every proposal for what to do next is a guess. This sorts
those rows into kinds, because the kinds want different work:

  roster miss        the true speaker is not on the roster the model was given.
                     Unwinnable by any model: the answer was never offered.
                     Fixing it is extraction work, not attribution work.
  alias confusion    the prediction and the truth are the same character under
                     different names, and the alias groups did not catch it.
                     This is a FIXTURE defect - it counts as an error and is
                     not one.
  unanswered         the model returned nothing. A generation or format
                     failure, not a reasoning failure.
  lead confusion     both names are major characters. The listener hears this
                     most, and it is the hardest genuine case.
  minor confusion    at least one side is a walk-on. Real, but nearly
                     inaudible.

WHY THIS AND NOT MORE MODELLING. The three interventions with the best stories
this week - roster quality, scene cast, closed sets - all failed, and they were
chosen from intuition about what the model lacked. This measures what it
actually gets wrong first.

A large roster-miss share would redirect effort upstream to extraction. A large
alias share would mean the ledger's accuracy is understated. A large lead-
confusion share would mean the remaining problem is genuinely hard and no
cheap fix exists.
"""
import argparse, collections, glob, json, os, re, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = REPO + "/app/"
sys.path.insert(0, APP)
from experiments.scoring import alias_groups, normalize, same_speaker
from three_pass_generate import build_roster

M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
LEDGER = REPO + "/ab_test_runtime/experiments"
BOOKS = ("grimgar03", "index18", "mushoku16", "owarimonogatari3")
SPECIAL = {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE"}
LEAD_SHARE = 0.05


def book_context(book):
    gold = json.load(open(APP + f"fixtures/attribution_gold_{book}.json"))
    src = open(M + f"inputs/{book}.txt", encoding="utf-8").read()
    cp = json.load(open(
        M + INPUT_RUN + f"/{book}/result.json.threepass_checkpoint.json"))
    roster = [r.upper() for r in
              build_roster([e for e in (cp.get("named") or []) if e], src)]
    roster = sorted(set(roster) | {n.upper() for n in
                                   gold.get("roster_additions", {}).get("names", [])})
    counts = collections.Counter(e["expected_speaker"].upper()
                                 for e in gold["entries"])
    total = sum(counts.values())
    leads = {s for s, c in counts.items()
             if c / total >= LEAD_SHARE and s not in SPECIAL}
    return {"roster": {normalize(r) for r in roster},
            "groups": alias_groups(gold), "leads": leads, "src": src}


def classify(expected, predicted, ctx):
    exp, pred = (expected or "").upper(), (predicted or "").upper()
    if not pred:
        return "unanswered"
    if normalize(exp) not in ctx["roster"]:
        return "roster miss"
    # The scorer already applies alias groups, so anything reaching here is
    # unmatched. A shared surname or a containment relation is the signature of
    # an alias the fixture never declared.
    a, b = normalize(exp), normalize(pred)
    if a and b and (a in b or b in a or
                    (set(a.split()) & set(b.split()))):
        return "alias confusion"
    if exp in ctx["leads"] and pred in ctx["leads"]:
        return "lead confusion"
    return "minor confusion"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--artifact",
                    default=LEDGER + "/lora_serving_eval__local-rocm-lora.json")
    ap.add_argument("--arm", default="lora")
    ap.add_argument("--out", default=LEDGER + "/residual_errors.json")
    args = ap.parse_args()

    paths = sorted(glob.glob(LEDGER + "/lora_serving_eval__*.json"))
    if not paths:
        print("no lora_serving_eval artifact yet")
        return

    ctxs = {b: book_context(b) for b in BOOKS
            if os.path.exists(APP + f"fixtures/attribution_gold_{b}.json")}
    kinds = collections.Counter()
    per_book = collections.defaultdict(collections.Counter)
    examples = collections.defaultdict(list)
    scored = 0
    for path in paths:
        doc = json.load(open(path))
        for row in doc["rows"]:
            if row["arm"] != args.arm:
                continue
            book = row["id"].split(":", 1)[0]
            ctx = ctxs.get(book)
            if not ctx:
                continue
            scored += 1
            if row.get("correct"):
                continue
            kind = classify(row.get("expected"), row.get("predicted"), ctx)
            kinds[kind] += 1
            per_book[book][kind] += 1
            if len(examples[kind]) < 3:
                examples[kind].append(
                    f"[{book}] expected {row.get('expected')} / got "
                    f"{row.get('predicted') or '(nothing)'}")

    errors = sum(kinds.values())
    if not errors:
        print("no errors found in", args.arm)
        return
    print(f"{args.arm} arm: {errors} errors out of {scored} scored rows "
          f"= {errors/scored*100:.1f}%\n")
    print(f"  {'kind':18}{'count':>7}{'of errors':>11}{'of all rows':>13}")
    for kind, count in kinds.most_common():
        print(f"  {kind:18}{count:7}{count/errors*100:10.1f}%"
              f"{count/scored*100:12.1f}%")

    print("\n  per book")
    order = [k for k, _ in kinds.most_common()]
    print(f"  {'book':18}" + "".join(f"{k[:11]:>13}" for k in order))
    for book in sorted(per_book):
        print(f"  {book:18}" + "".join(f"{per_book[book][k]:13}" for k in order))

    print("\n  examples")
    for kind in order:
        for line in examples[kind][:2]:
            print(f"    {kind:18} {line}")

    unwinnable = kinds["roster miss"] + kinds["alias confusion"]
    print(f"\n  {unwinnable} of {errors} errors ({unwinnable/errors*100:.0f}%) "
          f"are not attribution failures:")
    print(f"    roster miss      the answer was never on the list - extraction "
          f"work\n    alias confusion  right character, undeclared alias - a "
          f"fixture defect")
    print(f"  The genuinely hard remainder is "
          f"{kinds['lead confusion']} lead confusions "
          f"({kinds['lead confusion']/scored*100:.1f}% of all rows).")

    json.dump({"arm": args.arm, "scored": scored, "errors": errors,
               "kinds": dict(kinds),
               "per_book": {b: dict(c) for b, c in per_book.items()}},
              open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
