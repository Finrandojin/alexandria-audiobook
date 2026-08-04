"""Label the whole corpus from noisy rules, without ground truth.

Segmentation is blocked on labels: 46 NOT_DIALOGUE examples is what made the
classifier underpowered (21.7% recall, a 25-point interval). Hand-labelling
scales linearly with a person's time.

Snorkel's insight is that it does not have to. Write several noisy LABELLING
FUNCTIONS, estimate each one's accuracy from HOW THEY AGREE WITH EACH OTHER -
which needs no labels at all - then combine them into a probabilistic label per
item. A function that agrees with the consensus often is trusted more; one that
fires alone is trusted less.

This implements that directly rather than taking the dependency: the estimator
is about forty lines, and a library would bring a modelling framework for a
problem with eight functions and two classes.

THE LABELLING FUNCTIONS, each returning +1 (narration misfiled), -1 (real
speech), or 0 (abstain). They must be genuinely different sources of evidence,
because the accuracy estimate comes from disagreement - eight copies of one
rule would look unanimous and be confidently wrong.

VALIDATION IS HONEST HERE. The 839 judged rows exist already, so the weak
labels can be scored against real judgements the model never saw. Weak
supervision is a way to spend a person's time better, not a way to avoid
measurement: if the probabilistic labels are wrong, this says so.
"""
import argparse, collections, glob, json, os, re, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = REPO + "/app/"
sys.path.insert(0, APP)

M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
BOOKS = ("grimgar03", "grimgar06", "index18", "mushoku16", "mushoku18",
         "owarimonogatari3")
PAST = (r"\b(was|were|had|did|could|would|should|said|saw|looked|turned|walked|"
        r"ran|stood|sat|felt|thought|knew|seemed|began|opened|closed|reached)\b")
THIRD = r"\b(he|she|they|him|her|them|his|hers|their|its)\b"
FIRST = r"\b(i|you|we|my|your|our|me|us)\b"
ABSTAIN, NARR, SPEECH = 0, 1, -1


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


# --- labelling functions ---------------------------------------------------
# Each is a different KIND of evidence. Overlapping copies of one idea would
# agree by construction and the accuracy estimate would believe them.

def lf_unterminated(seg, i, t):
    """Ends mid-clause and the next segment continues it in lower case."""
    nxt = (seg[i + 1].get("text") or "").lstrip() if i + 1 < len(seg) else ""
    if not nxt:
        return ABSTAIN
    cut = (nxt[:1].islower() or nxt[:1] in ".,;:") and not re.search(r"[.!?…\"'”’]\s*$", t)
    return NARR if cut else ABSTAIN


def lf_third_person_past(seg, i, t):
    low = t.lower()
    if len(t) < 40:
        return ABSTAIN
    if re.search(THIRD, low) and re.search(PAST, low) and not re.search(FIRST, low):
        return NARR
    return ABSTAIN


def lf_island(seg, i, t):
    """A short fragment with narration both sides and no terminal punctuation."""
    if i == 0 or i + 1 >= len(seg):
        return ABSTAIN
    if (seg[i - 1].get("type") == "NARRATOR" and seg[i + 1].get("type") == "NARRATOR"
            and len(t) < 40 and not re.search(r"[.!?…]\s*$", t)):
        return NARR
    return ABSTAIN


def lf_quoted(seg, i, t):
    """Explicit quotation marks are strong evidence of real speech."""
    return SPEECH if re.search(r'[“"”]', t) else ABSTAIN


def lf_first_person_address(seg, i, t):
    """Second-person address and contractions are dialogue register."""
    low = t.lower()
    if re.search(r"\b(you|your|you're|don't|can't|won't|i'm|let's)\b", low):
        return SPEECH
    return ABSTAIN


def lf_speech_verb_neighbour(seg, i, t):
    """An adjacent narration carrying a speech verb attributes THIS line."""
    for j in (i - 1, i + 1):
        if 0 <= j < len(seg) and seg[j].get("type") == "NARRATOR":
            if re.search(r"\b(said|asked|replied|shouted|whispered|muttered|"
                         r"called|answered|cried|added)\b",
                         (seg[j].get("text") or "").lower()):
                return SPEECH
    return ABSTAIN


def lf_exclamatory(seg, i, t):
    """Short exclamations and questions are overwhelmingly speech."""
    if len(t) < 90 and re.search(r"[!?]\s*$", t):
        return SPEECH
    return ABSTAIN


def lf_long_prose(seg, i, t):
    """Very long segments with several sentences read as narration."""
    if len(t) > 240 and t.count(".") >= 3 and not re.search(r'[“"”]', t):
        return NARR
    return ABSTAIN


LFS = [lf_unterminated, lf_third_person_past, lf_island, lf_quoted,
       lf_first_person_address, lf_speech_verb_neighbour, lf_exclamatory,
       lf_long_prose]


def fit_label_model(L, iters=60):
    """Estimate each function's accuracy from agreement alone, then vote.

    Starts by trusting every function equally, forms a consensus, re-scores each
    function against that consensus, and repeats. This is the practical core of
    what a label model does: no ground truth enters anywhere.
    """
    n, m = L.shape
    acc = np.full(m, 0.7)
    probs = np.zeros(n)
    for _ in range(iters):
        w = np.log(np.clip(acc, .5 + 1e-6, .99) / (1 - np.clip(acc, .5 + 1e-6, .99)))
        score = (L * w).sum(axis=1)
        probs = 1 / (1 + np.exp(-score))              # P(narration)
        consensus = np.where(probs >= .5, 1, -1)
        for j in range(m):
            voted = L[:, j] != 0
            if voted.sum() < 5:
                acc[j] = .5
                continue
            agree = (L[voted, j] == consensus[voted]).mean()
            acc[j] = np.clip(agree, .5, .95)
    return probs, acc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=REPO + "/ab_test_runtime/experiments/weak_supervision.json")
    args = ap.parse_args()

    truth = {}
    for p in glob.glob(REPO + "/ab_test_runtime/fixtures_draft/labelling_bundle__*.json"):
        for e in json.load(open(p))["entries"]:
            truth[norm(e.get("line"))] = 1 if e.get("expected_speaker") == "NOT_DIALOGUE" else -1

    rows, votes = [], []
    for book in BOOKS:
        cp = M + INPUT_RUN + f"/{book}/result.json.threepass_checkpoint.json"
        if not os.path.exists(cp):
            continue
        seg = json.load(open(cp))["segmented"]
        for i, e in enumerate(seg):
            if e.get("type") == "NARRATOR":
                continue
            t = (e.get("text") or "").strip()
            if not t:
                continue
            v = [f(seg, i, t) for f in LFS]
            if not any(v):
                continue                      # every function abstained
            rows.append({"book": book, "segment_index": i, "line": t,
                         "truth": truth.get(norm(t))})
            votes.append(v)

    L = np.array(votes, dtype=float)
    probs, acc = fit_label_model(L)
    print(f"{len(rows)} segments where at least one function fired, "
          f"{L.shape[1]} functions\n")
    print(f"  {'labelling function':28}{'fires':>8}{'est. acc':>10}")
    for j, f in enumerate(LFS):
        print(f"  {f.__name__:28}{int((L[:,j]!=0).sum()):8}{acc[j]:10.2f}")

    labelled = [(p, r["truth"]) for p, r in zip(probs, rows) if r["truth"] is not None]
    print(f"\n  {len(labelled)} of these carry a real judgement to score against")
    if labelled:
        p = np.array([x[0] for x in labelled])
        y = np.array([x[1] for x in labelled])
        pred = np.where(p >= args.threshold, 1, -1)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == -1)).sum())
        fn = int(((pred == -1) & (y == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        print(f"  narration precision {prec*100:5.1f}%   recall {rec*100:5.1f}%   "
              f"({tp} true, {fp} false, {fn} missed)")
        print(f"  the rule baseline was 39.1% recall at 3.66% false positives "
              f"(cut alone)")
        fpr = fp / max(int((y == -1).sum()), 1)
        print(f"  false-positive rate on real speech {fpr*100:.2f}%")
        print("\n  A dropped line is silence where a character spoke, so the "
              "false-positive\n  rate is the number that decides whether this "
              "is usable, not recall.")

    high = [(r, float(p)) for r, p in zip(rows, probs) if p >= .8]
    print(f"\n  {len(high)} segments labelled narration with confidence >= 0.8, "
          f"across {len(BOOKS)} books")
    print(f"  against the 46 hand-labelled NOT_DIALOGUE examples that exist today")

    json.dump({"n_scored": len(rows), "lf_accuracy": dict(zip(
        [f.__name__ for f in LFS], [float(a) for a in acc])),
        "confident_narration": len(high),
        "rows": [{**r, "p_narration": float(p)} for r, p in zip(rows, probs)][:4000]},
        open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
