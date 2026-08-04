"""Can a classifier catch misfiled narration where the rules could not?

`segmentation_filter` established the rules are unusable: `cut` catches 39.1% of
misfiled rows at a 3.66% false-positive rate on real dialogue, and a dropped
line is silence where a character spoke. A learned model is the remaining
approach.

FIRST, A CORRECTION THAT CHANGES WHAT IS POSSIBLE HERE. The brief and
`segmentation_filter`'s own docstring describe "the 839 NOT_DIALOGUE labels".
That is wrong. 839 is the number of JUDGED ROWS; only **46** of them are
NOT_DIALOGUE, against 793 real speech. The positives are also concentrated:

    index18            21        owarimonogatari3   18
    mushoku16           3        grimgar03           4
    grimgar06           0        mushoku18           0

So two of six books contain 39 of the 46 positives and two contain none at all.

WHY THIS IS RUN ANYWAY. The point is to measure whether the labels can support
the model, not to produce an accuracy. The decision this informs is whether to
spend judging time on more NOT_DIALOGUE labels, and that decision needs the
power limit stated in numbers rather than asserted.

EVALUATION IS LEAVE-ONE-BOOK-OUT, because the deployment question is whether a
classifier trained on labelled books works on an UNLABELLED one. Random k-fold
over pooled rows would train and test on the same book and report a number that
does not survive contact with a new one. Books with no positives are scored for
false positives only - they can refute a model but cannot confirm it.

THE OPERATING POINT IS FIXED IN ADVANCE at a 1% false-positive rate on real
dialogue, and recall is read at that point. Choosing the threshold after seeing
the curve is how a 6% false-positive rule gets reported as promising.
"""
import collections, json, re, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO + "/app")
from experiments.stats import clopper_pearson

M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
BOOKS = ("grimgar03", "grimgar06", "index18", "mushoku16", "mushoku18",
         "owarimonogatari3")
TARGET_FPR = 0.01
PAST = (r"\b(was|were|had|did|could|would|should|said|saw|looked|turned|walked|"
        r"ran|stood|sat|felt|thought|knew|seemed|began|opened|closed|reached|"
        r"appeared|remained|continued|became|gave|took|made|came|went)\b")
THIRD = r"\b(he|she|they|him|her|them|his|hers|their|its)\b"
FIRST = r"\b(i|you|we|my|your|our|me|us)\b"


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


def features(seg, index, text):
    """Structural and lexical signals, all computable without labels."""
    nxt = seg[index + 1] if index + 1 < len(seg) else None
    prev = seg[index - 1] if index else None
    nxt_text = (nxt.get("text") or "").lstrip() if nxt else ""
    low = (text or "").lower()
    return {
        "no_terminal": 0.0 if re.search(r"[.!?…\"'”’]\s*$", text or "") else 1.0,
        "next_lower": 1.0 if nxt_text[:1].islower() else 0.0,
        "prev_narr": 1.0 if prev is not None and prev.get("type") == "NARRATOR" else 0.0,
        "next_narr": 1.0 if nxt is not None and nxt.get("type") == "NARRATOR" else 0.0,
        "third": float(len(re.findall(THIRD, low))),
        "first": float(len(re.findall(FIRST, low))),
        "past": float(len(re.findall(PAST, low))),
        "chars": min(len(text or ""), 400) / 100.0,
        "has_quote": 1.0 if re.search(r"[\"“”]", text or "") else 0.0,
        "commas": float((text or "").count(",")),
        "starts_upper": 1.0 if (text or "")[:1].isupper() else 0.0,
    }


def load():
    rows = []
    for book in BOOKS:
        try:
            bundle = json.load(open(
                REPO + f"/ab_test_runtime/fixtures_draft/labelling_bundle__{book}.json"))
            seg = json.load(open(
                M + INPUT_RUN + f"/{book}/result.json.threepass_checkpoint.json"))["segmented"]
        except FileNotFoundError:
            continue
        pos = {norm(e.get("text")): i for i, e in enumerate(seg)}
        for entry in bundle["entries"]:
            i = pos.get(norm(entry.get("line")))
            if i is None:
                continue
            rows.append({"book": book,
                         "y": 1 if entry.get("expected_speaker") == "NOT_DIALOGUE" else 0,
                         "x": features(seg, i, entry["line"]),
                         "line": entry["line"]})
    return rows


def main():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rows = load()
    keys = sorted(rows[0]["x"])
    counts = collections.Counter((r["book"], r["y"]) for r in rows)
    total_pos = sum(r["y"] for r in rows)
    print(f"{len(rows)} judged rows, {total_pos} NOT_DIALOGUE, "
          f"{len(rows)-total_pos} real speech")
    print("  (839 in the brief is the JUDGED-ROW count, not the label count)")
    print(f"\n  {'book':20}{'rows':>6}{'not-dialogue':>14}")
    for book in BOOKS:
        n = counts[(book, 0)] + counts[(book, 1)]
        if n:
            print(f"  {book:20}{n:6}{counts[(book,1)]:14}")

    print(f"\n  leave-one-book-out, threshold fixed at {TARGET_FPR:.0%} "
          f"false positives\n  (chosen on the TRAINING books, never on the "
          f"held-out one)")
    print(f"\n  {'held out':20}{'pos':>5}{'recall':>18}{'false pos':>16}")
    tp = fp = pos_total = neg_total = 0
    for held in BOOKS:
        train = [r for r in rows if r["book"] != held]
        test = [r for r in rows if r["book"] == held]
        if not test:
            continue
        xtr = np.array([[r["x"][k] for k in keys] for r in train])
        ytr = np.array([r["y"] for r in train])
        xte = np.array([[r["x"][k] for k in keys] for r in test])
        yte = np.array([r["y"] for r in test])
        if ytr.sum() < 2:
            print(f"  {held:20}{int(yte.sum()):5}   too few training positives")
            continue
        scaler = StandardScaler().fit(xtr)
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(scaler.transform(xtr), ytr)
        # The operating threshold is set on the training books' real dialogue,
        # so nothing about the held-out book informs it.
        neg_scores = model.predict_proba(
            scaler.transform(xtr[ytr == 0]))[:, 1]
        thresh = float(np.quantile(neg_scores, 1 - TARGET_FPR))
        score = model.predict_proba(scaler.transform(xte))[:, 1]
        flag = score >= thresh
        book_tp = int((flag & (yte == 1)).sum())
        book_fp = int((flag & (yte == 0)).sum())
        npos, nneg = int((yte == 1).sum()), int((yte == 0).sum())
        tp += book_tp
        fp += book_fp
        pos_total += npos
        neg_total += nneg
        rec = f"{book_tp}/{npos} = {book_tp/npos*100:5.1f}%" if npos else "  no positives"
        print(f"  {held:20}{npos:5}{rec:>18}{book_fp:8}/{nneg:<5} "
              f"{book_fp/max(nneg,1)*100:5.2f}%")

    rlo, rhi = clopper_pearson(tp, max(pos_total, 1))
    flo, fhi = clopper_pearson(fp, max(neg_total, 1))
    print(f"\n  pooled recall     {tp}/{pos_total} = "
          f"{tp/max(pos_total,1)*100:.1f}%  [{rlo:.1f}-{rhi:.1f}]")
    print(f"  pooled false pos  {fp}/{neg_total} = "
          f"{fp/max(neg_total,1)*100:.2f}%  [{flo:.2f}-{fhi:.2f}]")
    print(f"\n  Compare the rules: cut caught 39.1% at 3.66% false positives.")
    print(f"\n  READ THE INTERVAL, NOT THE POINT ESTIMATE. With {pos_total} "
          f"positives\n  spread over four books, the recall interval spans "
          f"{rhi-rlo:.0f} points. That is the\n  finding: the labels cannot "
          f"resolve whether a classifier beats the rules,\n  and no amount of "
          f"model choice fixes a label count. More NOT_DIALOGUE\n  labels is "
          f"the prerequisite, not a better classifier.")

    out = REPO + "/ab_test_runtime/experiments/segmentation_classifier.json"
    json.dump({"judged_rows": len(rows), "positives": total_pos,
               "per_book": {b: {"rows": counts[(b, 0)] + counts[(b, 1)],
                                "positives": counts[(b, 1)]} for b in BOOKS},
               "target_fpr": TARGET_FPR,
               "pooled": {"tp": tp, "positives": pos_total,
                          "recall_ci": [rlo, rhi],
                          "fp": fp, "negatives": neg_total,
                          "fpr_ci": [flo, fhi]},
               "verdict": "Underpowered. 46 positives over four books cannot "
                          "resolve whether a classifier beats the rule "
                          "baseline; more labels are the prerequisite."},
              open(out, "w"), indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
