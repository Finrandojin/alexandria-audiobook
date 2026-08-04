"""What is the segmenter actually getting wrong, and where?

Judging the four books produced a byproduct nobody planned: 839 rows carrying a
NOT_DIALOGUE label, decided by two independent frontier judges. That is a
segmentation evaluation set, and it did not exist yesterday.

It says the segmenter misfiles narration as dialogue at 1.0% on grimgar03 and
17.5% on index18. One in six "spoken" lines in index18 is not dialogue at all.
Every attribution method is being scored on rows that have no speaker, and the
shipped pipeline sends them to TTS as if a character said them.

That matters more than it sounds next to the attribution work. The best
attribution intervention of the day gained 22 points on one book by routing
40% of rows to a 70B. Segmentation is losing 17.5% on index18 outright, and
unlike attribution's ceiling this is a correctness bug with a fixable cause.

Neither label-free proxy predicts the rate - fragmentation and speech-verb
density both order the books wrongly - so this characterises the errors
directly instead of trying to guess them:

  by length        a mid-sentence fragment ("Summoning magic", "recreation,")
                   is a different failure from a whole misfiled paragraph
  by neighbour     narration on both sides suggests a split inside one
                   sentence; dialogue on both sides suggests a misread of a
                   quotation
  by continuation  whether the following segment begins lower-case or with
                   punctuation, which is the signature of a sentence cut in half
  by position      whether errors cluster, which would point at passage TYPE -
                   the diary and letter sections are the standing hypothesis

Offline. Consumes the fixtures and the segmentation checkpoint.
"""
import collections, json, re, statistics, os, sys
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
BOOKS = ("grimgar03", "index18", "mushoku16", "owarimonogatari3")


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


def load(book):
    gold = json.load(open(REPO + f"/app/fixtures/attribution_gold_{book}.json"))
    cp = json.load(open(M + INPUT_RUN + f"/{book}/result.json.threepass_checkpoint.json"))
    return gold, cp["segmented"]


def classify(seg, index, text):
    """Which kind of segmentation failure is this?"""
    prev = seg[index - 1] if index else None
    nxt = seg[index + 1] if index + 1 < len(seg) else None
    nxt_text = (nxt.get("text") or "").lstrip() if nxt else ""
    cut = bool(nxt_text) and (nxt_text[0].islower() or nxt_text[0] in ".,;:!?)")
    if len(text) < 30 and cut:
        return "sentence cut in half"
    if cut:
        return "runs into the next segment"
    if prev and nxt and prev.get("type") == "NARRATOR" and nxt.get("type") == "NARRATOR":
        return "narration island"
    if len(text) > 200:
        return "whole paragraph misfiled"
    return "other"


print("=" * 74)
print("Segmenter error, measured against the judges' NOT_DIALOGUE labels")
print("=" * 74)
allrows = []
for book in BOOKS:
    try:
        gold, seg = load(book)
    except FileNotFoundError:
        print(f"\n  {book}: not available")
        continue
    pos = {norm(e.get("text")): i for i, e in enumerate(seg)}
    # NOT_DIALOGUE rows are dropped from the fixture, so recover them from the
    # bundle, which keeps every judged row including the rejected ones.
    b = json.load(open(REPO + f"/ab_test_runtime/fixtures_draft/labelling_bundle__{book}.json"))
    bad = [e for e in b["entries"] if e.get("expected_speaker") == "NOT_DIALOGUE"]
    judged = len(b["entries"])
    kinds = collections.Counter()
    lens = []
    for e in bad:
        i = pos.get(norm(e.get("line")))
        if i is None:
            continue
        kinds[classify(seg, i, e["line"])] += 1
        lens.append(len(e["line"]))
        allrows.append((book, e["line"][:70]))
    rate = len(bad) / judged * 100 if judged else 0
    print(f"\n  {book}: {len(bad)}/{judged} judged rows are not dialogue = {rate:.1f}%")
    if lens:
        print(f"    median length {statistics.median(lens):.0f} chars "
              f"(range {min(lens)}-{max(lens)})")
    for k, v in kinds.most_common():
        print(f"    {k:26} {v:3}  ({v/max(len(bad),1)*100:.0f}%)")

print("\n" + "=" * 74)
print("A sample of what is being sent to TTS as dialogue:")
for book, line in allrows[:12]:
    print(f"  [{book[:12]:12}] {line}")
print("\nFixing the segmenter raises every attribution number for free, and")
print("unlike the attribution ceiling these rows are simply wrong rather than")
print("hard. The fragment cases look mechanical; the paragraph cases may need")
print("the passage-type signal that per-section routing is still missing.")
