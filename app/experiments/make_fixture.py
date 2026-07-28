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
# Point this at an existing gold file to RE-JUDGE exactly those lines instead
# of drawing a fresh sample. That is what makes a second judgement comparable:
# same ids, same lines, so disagreement is between judges rather than between
# samples. The existing labels are deliberately NOT carried into the bundle -
# showing them would turn a second opinion into a confirmation exercise.
FROM_GOLD = os.environ.get("EXPERIMENT_FROM_GOLD", "")
SEED = int(os.environ.get("EXPERIMENT_SEED", "20260728"))
# THE JUDGE'S CONTEXT IS NOT THE MODEL'S CONTEXT, and conflating them was the
# mistake in the first pass. Production shows the model one segment either
# side; the first bundles showed the judge four. But the judge's job is to
# establish what the answer IS, not to reproduce the model's handicap - a label
# decided on four segments is a guess with the same information the model had,
# and it cannot be ground truth for that model.
#
# Demonstrated on mushoku16-00818: at four segments the narration reads "After
# replying in an energetic voice, Isolte made a bitter smile", which points at
# Isolte having replied. At fourteen it is clear Isolte is reacting to Eris's
# reply and is herself the one who spoke the target line. Two of the three
# corrected gold rows turned on context outside the original window.
CONTEXT = int(os.environ.get("EXPERIMENT_CONTEXT", "12"))
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

if FROM_GOLD:
    prior = json.load(open(FROM_GOLD, encoding="utf-8"))
    index_of = {norm(e.get("text")): i for i, e in enumerate(seg)}
    picked, ids, unlocatable = [], {}, 0
    for entry in prior["entries"]:
        key = norm(entry.get("line"))
        # Skip lines whose text appears more than once: they cannot be located
        # unambiguously, which is the same rule the scoring harnesses apply.
        if occurrences[key] != 1 or key not in index_of:
            unlocatable += 1
            continue
        position = index_of[key]
        picked.append(position)
        ids[position] = entry["id"]
    picked.sort()
    # Aliases are a property of the BOOK, not of a judging pass. Dropping them
    # when rebuilding the bundle silently emptied grimgar03's list - losing
    # BRI-CHAN/BRITNEY and ZODIAC/ZODIAC-KUN, the second of which had taken
    # blind adjudication to find in the first place.
    inherited_aliases = prior.get("aliases", [])
    print(f"{BOOK}: re-judging {len(picked)} lines from "
          f"{os.path.basename(FROM_GOLD)} ({unlocatable} not uniquely "
          f"locatable, skipped), roster {len(roster)}")
    pass
else:
    inherited_aliases = []
    rng = random.Random(SEED)
    picked = sorted(rng.sample(eligible, min(COUNT, len(eligible))))
    ids = {i: f"{BOOK}-{i:05d}" for i in picked}
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
    "relabel_of": os.path.basename(FROM_GOLD) or None,
    "aliases": inherited_aliases,
    "entries": [{"id": ids[i], "segment_index": i,
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
