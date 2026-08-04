"""Why is mushoku16 sixteen points harder than grimgar03 at every model?

That gap has sat unexplained through six models, four architectures, and every
prompt intervention tried. It is not model size, not roster size, not fixture
length, and not the amount of unanchored dialogue. It is the narration.

The pipeline's core strategy is to read the segments either side of a line and
find the speaker there. That strategy has a precondition nobody measured: the
neighbouring narration has to NAME somebody.

    adjacent narration...          grimgar03   mushoku16
      contains a speech verb          61.9%       27.5%
      names the true speaker          77.6%       35.3%
    first-person pronouns / 1000
      narration words                  17.2        59.9

mushoku16 is first-person - Rudeus narrates, and 38% of its gold lines are his.
First-person narration is interior monologue, not stage direction: it rarely
writes "Roxy said". Third-person limited narration does, constantly.

THE SAME RULE EXPLAINS ACCURACY INSIDE BOTH BOOKS, pooled over every run:

                              grimgar03            mushoku16
      narration names them    71.9%                59.8%
      it does not             43.1%                45.0%
      separation              +28.8                +14.8

Note where the two books actually agree: on rows where narration does NOT name
the speaker they score 43.1% and 45.0% - the same. The books are not different
in difficulty per row. They differ in HOW MANY rows have the signal: 67% of
row-instances in grimgar03 against 26% in mushoku16. The "book gap" is a
composition effect over a single feature.

WHAT THIS RETRODICTS, which is the reason to trust it rather than just like it:

  w4 context     +10.5 on grimgar03, -5.0 on mushoku16. Wider context supplies
                 more narration, which is worth having only if narration names
                 people. Measured before this explanation existed.
  anchoring      being adjacent to narration at all is worth +19.2 points in
                 grimgar03 and +0.3 in mushoku16 - no benefit where the
                 narration carries no names.
  the oracle gap even with the true speaker among five candidates, models fail
                 17-25%; those rows concentrate where the text never says who
                 spoke, so no candidate list can recover them.

WHAT IT IMPLIES, and none of it has been tested:

  1. Narration name-density is measurable per book BEFORE any LLM call - a
     regex over narration segments. It predicts which strategy will work, so
     the width/context decision should be per-book, not global. That is a
     cheap router with no oracle information in it.
  2. For first-person books the missing signal is elsewhere: turn-taking,
     register, and the narrator being a known constant who speaks a large
     share of all lines. A narrator-identity prior is untested.
  3. Every single-book result in the ledger needs re-reading through this
     lens, because grimgar03 is the easy regime and mushoku16 the hard one,
     and which one an experiment ran on largely determined its answer.

Offline. Consumes committed artifacts and the segmentation checkpoint.
"""
import collections
import glob, json, os, re, sys
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")
from experiments.stats import clopper_pearson

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
E = REPO + "/ab_test_runtime/experiments/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
BOOKS = (("grimgar03", "attribution_gold_grimgar03_provisional.json"),
         ("mushoku16", "attribution_gold_random.json"))
SPEECH_VERB = (r"\b(said|asked|replied|answered|shouted|whispered|muttered|"
               r"called|cried|yelled|groaned|sighed|laughed|nodded|exclaimed|"
               r"bellowed|agreed|told|added|continued|began|offered)\b")


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


def load(book, goldfile):
    cp = json.load(open(M + INPUT_RUN + f"/{book}/result.json.threepass_checkpoint.json"))
    gold = json.load(open(REPO + "/app/fixtures/" + goldfile))
    return cp["segmented"], gold


def neighbouring_narration(seg, index):
    """Text of the NARRATOR segments immediately either side of `index`.

    Immediate neighbours only, because that is what production pass 2 supplies
    at w1 - the question is whether the signal is there for the shipping
    configuration, not whether it exists somewhere in the chapter.
    """
    return " ".join((seg[j].get("text") or "")
                    for j in (index - 1, index + 1)
                    if 0 <= j < len(seg) and seg[j].get("type") == "NARRATOR")


def features(book, goldfile):
    seg, gold = load(book, goldfile)
    pos = {norm(e.get("text")): i for i, e in enumerate(seg)}
    named, tagged, total = {}, 0, 0
    for g in gold["entries"]:
        i = pos.get(norm(g["line"]))
        if i is None:
            continue
        near = neighbouring_narration(seg, i)
        first = g["expected_speaker"].upper().split()[0]
        named[g["id"]] = bool(near and re.search(r"\b" + re.escape(first) + r"\b",
                                                 near.upper()))
        if near:
            total += 1
            tagged += bool(re.search(SPEECH_VERB, near.lower()))
    narration = " ".join((e.get("text") or "") for e in seg
                         if e.get("type") == "NARRATOR").lower()
    words = len(narration.split()) or 1
    first_person = len(re.findall(r"\b(i|me|my|myself)\b", narration))
    return named, {"with_narration": total, "speech_verb": tagged,
                   "first_person_per_1000": first_person / words * 1000}


def pooled_accuracy(goldfile, named):
    """Accuracy split by the feature, pooled over every artifact on this book.

    Pooling is for the SHAPE: the runs are not independent, so the intervals
    are narrower than the truth. The separation is large enough that this does
    not change the reading, but it would matter for a close call.
    """
    tally = collections.defaultdict(lambda: [0, 0])
    for path in glob.glob(E + "*.json"):
        try:
            doc = json.load(open(path))
        except (ValueError, OSError):
            continue
        if os.path.basename(str((doc.get("meta") or {}).get("gold_path", ""))) != goldfile:
            continue
        for row in doc.get("rows", []):
            has = named.get(row["id"])
            if has is None:
                continue
            tally[has][0] += 1
            tally[has][1] += bool(row["correct"])
    return tally


if __name__ == "__main__":
    print("=" * 74)
    print("Does the neighbouring narration name the speaker?")
    print("=" * 74)
    for book, goldfile in BOOKS:
        named, stats = features(book, goldfile)
        share = sum(1 for v in named.values() if v) / max(len(named), 1) * 100
        print(f"\n  {book}")
        print(f"    gold rows whose adjacent narration names the speaker: "
              f"{share:.1f}%")
        print(f"    adjacent narration containing a speech verb: "
              f"{stats['speech_verb']}/{stats['with_narration']} = "
              f"{stats['speech_verb']/max(stats['with_narration'],1)*100:.1f}%")
        print(f"    first-person pronouns per 1000 narration words: "
              f"{stats['first_person_per_1000']:.1f}")
        tally = pooled_accuracy(goldfile, named)
        for flag, label in ((True, "narration names them"), (False, "it does not")):
            n, k = tally[flag]
            if not n:
                continue
            lo, hi = clopper_pearson(k, n)
            print(f"      {label:22} {k:6}/{n:<6} = {k/n*100:5.1f}%  [{lo:.1f}-{hi:.1f}]")
        if tally[True][0] and tally[False][0]:
            sep = (tally[True][1] / tally[True][0]
                   - tally[False][1] / tally[False][0]) * 100
            print(f"      separation {sep:+.1f} points")
    print("\n" + "=" * 74)
    print("The books agree on rows WITHOUT the signal (43.1% vs 45.0%). They")
    print("differ in how many rows have it. The book gap is a composition")
    print("effect over one feature, not a difference in per-row difficulty.")
