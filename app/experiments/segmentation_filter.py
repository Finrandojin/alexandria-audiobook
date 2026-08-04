"""Can a rule catch the misfiled segments without eating real dialogue?

`segmentation_eval` showed the segmenter sends narration to TTS as character
speech - 17.5% of index18's judged rows, 10.0% of owarimonogatari3's - and that
the failures are two distinct shapes:

    sentence cut in half     a segment ending mid-clause whose successor starts
                             lower-case: "Summoning magic" / "and the magic
                             recorded on the Dragon Race wall art"
    paragraph misfiled       plain third-person narration marked SPOKEN:
                             "A shallow cut opened in William Orwell's side."

Both look mechanical, so this tries rules before anything heavier. The two
judges' labels make it self-scoring - no review, no adjudication, just
precision and recall against gold - across 839 judged rows, of which 46 are
NOT_DIALOGUE and 793 are real speech. (An earlier version of this docstring
called it "the 839 NOT_DIALOGUE labels", conflating the row count with the
label count; the positives are 46, and `segmentation_classifier` shows that
number is the binding constraint.)

THE TRADE-OFF IS THE WHOLE POINT. A filter that flags everything reaches 100%
recall and destroys the corpus. Real dialogue wrongly dropped is worse than
narration wrongly kept, because a dropped line is silence where a character
spoke, while a kept one is a misattributed line the cascade might still fix.
So the headline number here is FALSE POSITIVES on the 772 rows the judges gave
a real speaker, and recall is only interesting subject to that staying near
zero.

Each rule is reported alone and in combination, because a rule that only fires
where another already fired is not worth its risk.
"""
import collections, json, re, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
BOOKS = ("grimgar03", "index18", "mushoku16", "owarimonogatari3")
PAST = (r"\b(was|were|had|did|could|would|should|said|saw|looked|turned|walked|"
        r"ran|stood|sat|felt|thought|knew|seemed|began|opened|closed|reached|"
        r"appeared|remained|continued|became|gave|took|made|came|went)\b")
THIRD = r"\b(he|she|they|him|her|them|his|hers|their|its)\b"


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


def rules(seg, index, text):
    """Each rule returns True if it thinks this segment is not dialogue."""
    nxt = seg[index + 1] if index + 1 < len(seg) else None
    nxt_text = (nxt.get("text") or "").lstrip() if nxt else ""
    prev = seg[index - 1] if index else None

    # 1. The segment does not end a sentence and the next one continues it.
    #    "Summoning magic" followed by "and the magic recorded on..."
    cut = bool(nxt_text) and (nxt_text[0].islower() or nxt_text[0] in ".,;:!?)") \
        and not re.search(r"[.!?…\"'”’]\s*$", text or "")

    # 2. Third-person past-tense prose with no first- or second-person voice.
    #    Deliberately requires BOTH a third-person pronoun and a past-tense
    #    verb: either alone fires constantly inside real dialogue.
    low = (text or "").lower()
    narrative = bool(re.search(THIRD, low) and re.search(PAST, low)
                     and not re.search(r"\b(i|you|we|my|your|our)\b", low)
                     and len(text or "") > 40)

    # 3. A short fragment with narration on both sides and no terminal
    #    punctuation - an island the segmenter carved out of a paragraph.
    island = (prev is not None and nxt is not None
              and prev.get("type") == "NARRATOR" and nxt.get("type") == "NARRATOR"
              and len(text or "") < 40
              and not re.search(r"[.!?…]\s*$", text or ""))
    return {"cut": cut, "narrative": narrative, "island": island}


rows = []
for book in BOOKS:
    try:
        b = json.load(open(REPO + f"/ab_test_runtime/fixtures_draft/labelling_bundle__{book}.json"))
        seg = json.load(open(M + INPUT_RUN + f"/{book}/result.json.threepass_checkpoint.json"))["segmented"]
    except FileNotFoundError:
        continue
    pos = {norm(e.get("text")): i for i, e in enumerate(seg)}
    for e in b["entries"]:
        i = pos.get(norm(e.get("line")))
        if i is None:
            continue
        rows.append({"book": book, "bad": e.get("expected_speaker") == "NOT_DIALOGUE",
                     "line": e["line"], **rules(seg, i, e["line"])})

bad = [r for r in rows if r["bad"]]
good = [r for r in rows if not r["bad"]]
print(f"{len(rows)} judged rows: {len(bad)} not dialogue, {len(good)} real speech\n")
print(f"  {'rule':28}{'catches':>10}{'of bad':>9}{'FALSE POS':>12}{'of good':>9}")
COMBOS = [("cut",), ("narrative",), ("island",),
          ("cut", "island"), ("cut", "narrative", "island")]
for combo in COMBOS:
    name = " + ".join(combo)
    tp = sum(1 for r in bad if any(r[c] for c in combo))
    fp = sum(1 for r in good if any(r[c] for c in combo))
    print(f"  {name:28}{tp:8}  {tp/max(len(bad),1)*100:6.1f}%{fp:9}  "
          f"{fp/max(len(good),1)*100:6.2f}%")

print(f"\n  per book, using cut + island (the fragment rules only)")
print(f"  {'book':18}{'bad caught':>13}{'false pos':>12}")
for book in BOOKS:
    bb = [r for r in bad if r["book"] == book]
    gg = [r for r in good if r["book"] == book]
    if not bb and not gg:
        continue
    tp = sum(1 for r in bb if r["cut"] or r["island"])
    fp = sum(1 for r in gg if r["cut"] or r["island"])
    print(f"  {book:18}{tp:5}/{len(bb):<5} {tp/max(len(bb),1)*100:5.0f}%"
          f"{fp:6}/{len(gg):<5} {fp/max(len(gg),1)*100:5.2f}%")

miss = [r for r in bad if not (r["cut"] or r["narrative"] or r["island"])]
print(f"\n  {len(miss)} of {len(bad)} misfiled rows caught by no rule:")
for r in miss[:8]:
    print(f"    [{r['book'][:12]:12}] {r['line'][:64]}")
print("\n  A rule is only worth shipping if its false-positive rate on real")
print("  dialogue is near zero: a dropped line is silence where a character")
print("  spoke, which is worse than a misattributed one the cascade may fix.")
