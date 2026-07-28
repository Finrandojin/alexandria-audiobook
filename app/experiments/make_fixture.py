"""Sample lines from an unlabelled book and build a hand-labelling bundle.

Two books cannot establish that narration name-density predicts attribution
accuracy - they can only show the two differ. Six more books are already
segmented and none are labelled, and labelling is the bottleneck: it is human
time, not GPU time.

SEGMENTATION IS AN UNCONTROLLED UPSTREAM VARIABLE and this bundle measures it.
Spot checks found third-person narration filed as SPOKEN in index18 ("In his
case, though, his goal wasn't just to arrive at his destination") and mid-
sentence fragments in owarimonogatari3 ("recreation,", "Namishiro"). The
signature is third-person pronoun density inside SPOKEN segments: 6-8 per 1000
words in the grimgar books against 16-20 in index18, mushoku16 and
owarimonogatari3. Quote marks cannot be used to detect it because segmentation
strips them - both existing fixtures contain zero quoted lines - so a
NOT_DIALOGUE answer is the only way to quantify it, and every bundle asks for
one.

THE PREDICTION IS MADE BEFORE LABELLING, from features that need no gold at
all. Measured over the segmented corpus:

    book               adjacent narration    first-person
                       has a speech verb     pronouns/1k
    grimgar03                62.5%              17.2
    grimgar06                48.0%              10.0
    index18                  45.2%               2.2
    mushoku16                27.1%              59.9
    mushoku18                23.3%              37.1
    owarimonogatari3         16.9%              52.8

grimgar03 and mushoku16 are the two labelled books and they sit at 62.5% and
27.1%, scoring 57.0% and 49.6% at the same model. If the mechanism is real,
index18 should behave like grimgar03 and owarimonogatari3 should be worse than
mushoku16 - that is an out-of-sample prediction, registered here, testable with
about a hundred labels per book.

A NOTE ON THE PROXY. "Adjacent narration contains ANY roster name" barely
separates the labelled books (87.8% against 78.3%) while "names the TRUE
speaker" separates them hugely (77.6% against 35.3%). The first is label-free
and weak; the second needs gold. Speech-verb density is label-free AND tracks
the strong version, so it is the predictor used here. Do not substitute the
any-name number for it.

NO MODEL GUESSES ARE SHOWN TO THE LABELLER. Pre-filling with model consensus
would make labelling faster and the fixture circular - a gold set drawn from
model output cannot measure model accuracy, and even shown as a "draft" it
anchors the human toward the model's answer. The bundle carries the passage,
the line, and the roster, and nothing else.
"""
import collections
import json, os, random, re, sys
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")
from three_pass_generate import build_roster, get_deterministic_named_entry

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
BOOK = os.environ.get("EXPERIMENT_BOOK", "owarimonogatari3")
COUNT = int(os.environ.get("EXPERIMENT_COUNT", "120"))
SEED = int(os.environ.get("EXPERIMENT_SEED", "20260728"))
CONTEXT = int(os.environ.get("EXPERIMENT_CONTEXT", "4"))
OUT = os.environ.get(
    "EXPERIMENT_OUT",
    os.path.join(REPO, "ab_test_runtime", "fixtures_draft",
                 f"labelling_bundle__{BOOK}.json"))


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


cp = json.load(open(M + INPUT_RUN + f"/{BOOK}/result.json.threepass_checkpoint.json"))
seg = cp["segmented"]
src = open(M + f"inputs/{BOOK}.txt", encoding="utf-8").read()
roster = sorted({r.upper() for r in
                 build_roster([e for e in (cp.get("named") or []) if e], src)})

# Same eligibility rules the scoring harnesses apply, so the fixture measures
# what they measure: spoken, non-deterministic, and unique in the book (a line
# appearing twice cannot be located unambiguously by text).
occurrences = collections.Counter(norm(e.get("text")) for e in seg)
eligible = [i for i, e in enumerate(seg)
            if e.get("type") != "NARRATOR"
            and get_deterministic_named_entry(e) is None
            and occurrences[norm(e.get("text"))] == 1
            and (e.get("text") or "").strip()]

rng = random.Random(SEED)
picked = sorted(rng.sample(eligible, min(COUNT, len(eligible))))
print(f"{BOOK}: {len(eligible)} eligible lines, sampling {len(picked)}, "
      f"roster {len(roster)}")


def passage(index, width):
    out = []
    for j in range(max(0, index - width), min(len(seg), index + 1 + width)):
        out.append({"type": seg[j].get("type"), "text": seg[j].get("text"),
                    "target": j == index})
    return out


bundle = {
    "book": BOOK,
    "seed": SEED,
    "context_segments": CONTEXT,
    "roster": roster,
    "instructions": (
        "For each item, read the passage and say who speaks the highlighted "
        "line. Use a name from the roster where one fits, or type another name "
        "if the roster is missing the speaker. Use UNKNOWN only when the "
        "passage genuinely does not determine it - that is a real and useful "
        "answer, not a failure to decide, and those rows are the ones the "
        "models argue about most. Answer NOT_DIALOGUE when the highlighted "
        "text is not a spoken line at all: the segmenter misfiles narration as "
        "speech, and how often it does so is a measurement worth having rather "
        "than an annoyance to work around."),
    "entries": [{"id": f"{BOOK}-{i:05d}", "segment_index": i,
                 "line": seg[i].get("text"),
                 "passage": passage(i, CONTEXT),
                 "expected_speaker": ""}
                for i in picked],
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(bundle, fh, indent=1, ensure_ascii=False)
print(f"wrote {OUT}")
print("\nFill in every `expected_speaker`, then convert with finalise_fixture.py.")
