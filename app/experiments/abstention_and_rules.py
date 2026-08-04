"""Two questions about knowing when NOT to ask the model.

**1. Does two-pass agreement predict CORRECTNESS, or only agreement?**

The cascade routes on disagreement between two cheap passes, and that routing
demonstrably works - escalating those rows buys points. But nobody has checked
the thing that makes an abstention policy possible: when the two passes AGREE,
how often are they right? The model answers UNKNOWN on 0-1.1% of lines, so
today every uncertain line becomes a confident wrong guess and the pipeline has
no way to say "leave this one alone".

If agreement is strongly predictive, then on disagreement the system could fall
back to something safer - the previous speaker, or the narrator - instead of
guessing. If agreement predicts nothing, abstention is dead and the disagreement
signal is only useful for deciding what to escalate.

**2. How far do rules alone get you?**

`get_deterministic_named_entry` already resolves some lines with no LLM at all.
The fraction has never been reported. Every line a rule can settle is faster,
free, and cannot hallucinate a name - and the precision of those rules against
gold is the number that says whether leaning on them harder is safe.

Both are computed from artifacts and fixtures already on disk. No GPU, no
model, no new inference.
"""
import argparse, collections, glob, json, os, re, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = REPO + "/app/"
sys.path.insert(0, APP)
from experiments.scoring import alias_groups, same_speaker
from experiments.stats import clopper_pearson
from three_pass_generate import get_deterministic_named_entry

M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
LEDGER = REPO + "/ab_test_runtime/experiments"
BOOKS = ("grimgar03", "index18", "mushoku16", "owarimonogatari3")
SPECIAL = {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE"}


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


def gold_map(book):
    """gold id -> expected speaker, plus the alias groups for the book."""
    path = APP + f"fixtures/attribution_gold_{book}.json"
    if not os.path.exists(path):
        return None, None
    doc = json.load(open(path))
    want = {g["id"]: g["expected_speaker"].upper() for g in doc["entries"]
            if g["expected_speaker"].upper() not in SPECIAL}
    return want, alias_groups(doc)


def agreement_analysis():
    print("=" * 72)
    print("1. Does two-pass agreement predict correctness?")
    print("=" * 72)
    print(f"\n  {'book':18}{'agree n':>9}{'agree acc':>11}"
          f"{'disagree n':>12}{'disagree acc':>14}{'lift':>8}")
    totals = collections.Counter()
    for path in sorted(glob.glob(LEDGER + "/cascade_state__*.json")):
        book = next((b for b in BOOKS if b in os.path.basename(path)), None)
        if not book:
            continue
        state = json.load(open(path))
        w1, w4 = state.get("w1") or {}, state.get("w4") or {}
        want, groups = gold_map(book)
        if not want or not w1 or not w4:
            continue
        agree = [0, 0]
        disagree = [0, 0]
        for gid, expected in want.items():
            # cascade_state keys are the numeric part of the gold id:
            # "grimgar03-00001" is stored as "1".
            short = str(int(gid.rsplit("-", 1)[-1])) if "-" in gid else gid
            a = (w1.get(short) or w1.get(gid) or "").upper()
            b = (w4.get(short) or w4.get(gid) or "").upper()
            if not a or not b:
                continue
            bucket = agree if same_speaker(a, b, groups) else disagree
            bucket[0] += 1
            bucket[1] += same_speaker(expected, a, groups)
        if not agree[0] or not disagree[0]:
            continue

        aa = agree[1] / agree[0]
        da = disagree[1] / disagree[0]
        totals.update({"an": agree[0], "ac": agree[1],
                       "dn": disagree[0], "dc": disagree[1]})
        print(f"  {book:18}{agree[0]:9}{aa*100:10.1f}%{disagree[0]:12}"
              f"{da*100:13.1f}%{(aa-da)*100:+8.1f}")

    if not totals["an"]:
        print("  No cascade_state artifact carries both passes.")
        return
    aa = totals["ac"] / totals["an"]
    da = totals["dc"] / totals["dn"]
    alo, ahi = clopper_pearson(totals["ac"], totals["an"])
    dlo, dhi = clopper_pearson(totals["dc"], totals["dn"])
    print(f"\n  pooled  agree {totals['ac']}/{totals['an']} = {aa*100:.1f}% "
          f"[{alo:.1f}-{ahi:.1f}]")
    print(f"          disagree {totals['dc']}/{totals['dn']} = {da*100:.1f}% "
          f"[{dlo:.1f}-{dhi:.1f}]")
    print(f"          lift {(aa-da)*100:+.1f} points")
    print(f"\n  Disagreement covers {totals['dn']/(totals['an']+totals['dn'])*100:.0f}% "
          f"of lines.")
    if aa - da > 0.15:
        print("  -> Agreement is strongly predictive. An abstention policy is "
              "buildable:\n     on disagreement, fall back rather than guess. "
              "Whether the fallback\n     beats guessing is a separate "
              "measurement - this only says the\n     signal exists.")
    else:
        print("  -> Agreement barely separates right from wrong, so abstention "
              "has\n     nothing to trigger on and the signal is only useful "
              "for escalation.")


def rule_analysis():
    print("\n" + "=" * 72)
    print("2. How many lines can rules settle with no model at all?")
    print("=" * 72)
    print(f"\n  {'book':18}{'gold lines':>11}{'rule-settled':>14}"
          f"{'coverage':>10}{'precision':>11}")
    tot_n = tot_cov = tot_ok = 0
    for book in BOOKS:
        want, groups = gold_map(book)
        cp_path = M + INPUT_RUN + f"/{book}/result.json.threepass_checkpoint.json"
        if not want or not os.path.exists(cp_path):
            continue
        seg = json.load(open(cp_path))["segmented"]
        occ = collections.Counter(norm(e.get("text")) for e in seg)
        gold_doc = json.load(open(APP + f"fixtures/attribution_gold_{book}.json"))
        by_line = {norm(g["line"]): g["expected_speaker"].upper()
                   for g in gold_doc["entries"]
                   if g["expected_speaker"].upper() not in SPECIAL
                   and occ[norm(g["line"])] == 1}
        # Coverage is measured over EVERY spoken segment, not over gold. The
        # gold set was sampled from the non-deterministic lines by
        # construction, so its rule coverage is 0 by definition and says
        # nothing about how much work rules do in production.
        spoken = [e for e in seg if e.get("type") != "NARRATOR"
                  and (e.get("text") or "").strip()]
        covered = sum(1 for e in spoken
                      if get_deterministic_named_entry(e) is not None)
        correct = 0   # unmeasurable: gold carries no rule-settled line
        n = len(spoken)
        tot_n += n
        tot_cov += covered
        tot_ok += correct
        prec = "  no gold" 
        print(f"  {book:18}{n:11}{covered:14}{covered/max(n,1)*100:9.1f}%{prec}")
    print(f"\n  pooled  {tot_cov}/{tot_n} SPOKEN SEGMENTS settled by rule "
          f"= {tot_cov/max(tot_n,1)*100:.1f}%")
    print("  Gold cannot score these: it was sampled from the lines rules do "
          "NOT\n  settle, so precision here is unmeasurable on this fixture "
          "and would\n  need a separate labelled sample of rule-settled "
          "lines.")
    print(f"\n  So the model is asked about "
          f"{(tot_n-tot_cov)/max(tot_n,1)*100:.0f}% of spoken segments, and "
          f"every accuracy\n  in this ledger describes only that portion of "
          f"the book.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args()
    agreement_analysis()
    rule_analysis()


if __name__ == "__main__":
    main()
