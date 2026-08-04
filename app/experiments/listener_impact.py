"""Does any of this reach the listener?

Every number in this investigation is accuracy over gold LINES, with each line
counting 1.0. A listener does not hear lines uniformly. They hear a wrong voice,
and how much that matters depends on who was supposed to be speaking:

  a lead's line          the two characters carrying a 400-line chapter are on
                         screen constantly, so an error there is heard
                         immediately and repeatedly
  a walk-on's line       a character with three lines in the book can be wrong
                         without a listener ever noticing
  narrator confusion     the narration voice suddenly speaks dialogue, or a
                         character narrates. This is the most jarring failure
                         and is invisible in a flat accuracy number.

So a 4-point accuracy gain concentrated on walk-ons may be inaudible, and a
2-point gain on the leads may be obvious. Nothing so far distinguishes those.

WHAT THIS IS NOT. There is no per-book cast on disk, so "same voice" is
MODELLED by speaking frequency: leads carry their own voice, the long tail may
share. That is an assumption, stated here rather than buried, and a real
`voice_config` would replace it. The narrator boundary is not modelled - it is
read directly from the labels.

The question it answers: when the adapter gains N points, WHERE do the points
land?
"""
import argparse, collections, glob, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = REPO + "/app/"
sys.path.insert(0, APP)

BOOKS = ("grimgar03", "index18", "mushoku16", "owarimonogatari3")
LEAD_SHARE = 0.05          # a speaker with >=5% of a book's lines is a lead


def gold_profile(book):
    """Who speaks how much, from gold alone."""
    path = APP + f"fixtures/attribution_gold_{book}.json"
    if not os.path.exists(path):
        return None
    entries = json.load(open(path))["entries"]
    counts = collections.Counter(e["expected_speaker"].upper() for e in entries)
    total = sum(counts.values())
    leads = {s for s, c in counts.items() if c / total >= LEAD_SHARE
             and s not in {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE"}}
    return {"counts": counts, "total": total, "leads": leads}


def classify(expected, predicted, profile):
    """What KIND of error is this, from a listener's point of view."""
    exp = (expected or "").upper()
    pred = (predicted or "").upper()
    if pred == "":
        return "unanswered"
    if "NARRATOR" in (exp, pred) and exp != pred:
        return "narrator_confusion"
    if exp in profile["leads"]:
        return "lead_line"
    return "minor_line"


def arm_rows(path, strip_book=False):
    doc = json.load(open(path))
    out = collections.defaultdict(list)
    for row in doc.get("rows") or []:
        rid = row["id"].split(":", 1)[1] if strip_book and ":" in row["id"] else row["id"]
        book = row["id"].split(":", 1)[0] if ":" in row["id"] else None
        out[row["arm"]].append((book, rid, row.get("expected"),
                                row.get("predicted"), bool(row.get("correct"))))
    return out


def report(name, arms, default_book=None):
    print(f"\n=== {name} ===")
    profiles = {b: gold_profile(b) for b in BOOKS}
    print(f"  {'arm':7}{'lines':>7}{'acc':>8}{'LEAD acc':>10}"
          f"{'lead err':>10}{'minor err':>11}{'narr conf':>11}")
    summary = {}
    for arm, rows in sorted(arms.items()):
        lead_n = lead_ok = 0
        kinds = collections.Counter()
        ok = 0
        for book, _, expected, predicted, correct in rows:
            prof = profiles.get(book or default_book)
            if not prof:
                continue
            ok += correct
            exp = (expected or "").upper()
            if exp in prof["leads"]:
                lead_n += 1
                lead_ok += correct
            if not correct:
                kinds[classify(expected, predicted, prof)] += 1
        n = len(rows)
        summary[arm] = {"n": n, "acc": ok / max(n, 1),
                        "lead_acc": lead_ok / max(lead_n, 1),
                        "lead_n": lead_n, **kinds}
        print(f"  {arm:7}{n:7}{ok/max(n,1)*100:7.1f}%{lead_ok/max(lead_n,1)*100:9.1f}%"
              f"{kinds.get('lead_line', 0):10}{kinds.get('minor_line', 0):11}"
              f"{kinds.get('narrator_confusion', 0):11}")

    names = sorted(summary)
    if len(names) == 2:
        a, b = names
        # Which arm is the improvement? Order arms so the better one is second.
        if summary[a]["acc"] > summary[b]["acc"]:
            a, b = b, a
        d_all = (summary[b]["acc"] - summary[a]["acc"]) * 100
        d_lead = (summary[b]["lead_acc"] - summary[a]["lead_acc"]) * 100
        print(f"\n  {b} - {a}:  overall {d_all:+.1f} points, "
              f"LEAD LINES {d_lead:+.1f} points")
        # Counter keys are absent when a category never occurred, and a book
        # with no narrator confusions is the good case, not a missing one.
        get = lambda arm, key: summary[arm].get(key, 0)
        fixed_lead = get(a, "lead_line") - get(b, "lead_line")
        fixed_minor = get(a, "minor_line") - get(b, "minor_line")
        fixed_narr = get(a, "narrator_confusion") - get(b, "narrator_confusion")
        print(f"  errors removed: {fixed_lead} on lead lines, "
              f"{fixed_minor} on minor lines, {fixed_narr} narrator confusions")
        if d_lead > d_all + 1:
            print("  -> the gain is concentrated on the characters a listener "
                  "hears most.\n     Audible.")
        elif d_lead < d_all - 1:
            print("  -> the gain is concentrated on the long tail. A listener "
                  "may not\n     notice it, and the flat accuracy number "
                  "overstates the benefit.")
        else:
            print("  -> the gain is spread evenly; flat accuracy is a fair "
                  "summary here.")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=REPO + "/ab_test_runtime/experiments/listener_impact.json")
    args = ap.parse_args()

    print("Lead = a speaker with >= 5% of a book's gold lines. "
          "Voices are MODELLED\nby frequency; a real voice_config would "
          "replace this assumption.")
    for book in BOOKS:
        prof = gold_profile(book)
        if prof:
            print(f"  {book:18} {len(prof['leads'])} leads of "
                  f"{len(prof['counts'])} speakers, "
                  f"{sum(prof['counts'][s] for s in prof['leads'])/prof['total']*100:.0f}% "
                  f"of lines")

    collected = {}
    d = LEDGER = REPO + "/ab_test_runtime/experiments"
    p = LEDGER + "/distill_eval__thunder-a6000-distill.json"
    if os.path.exists(p):
        collected["distillation (bf16 transformers)"] = report(
            "distillation, bf16 through transformers", arm_rows(p))
    for p in sorted(glob.glob(LEDGER + "/lora_serving_eval__*.json")):
        collected[os.path.basename(p)] = report(
            "adapter served through llama.cpp Q4 — " + os.path.basename(p),
            arm_rows(p))
    for p in sorted(glob.glob(LEDGER + "/cascade__*tuned-cheap-arm.json")):
        book = next((b for b in BOOKS if b in os.path.basename(p)), None)
        collected[os.path.basename(p)] = report(
            "cascade with tuned cheap arm — " + os.path.basename(p),
            arm_rows(p), default_book=book)

    if not collected:
        print("\nNo scored artifacts found yet.")
        return
    json.dump(collected, open(args.out, "w"), indent=1, default=str)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
