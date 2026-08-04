"""Turn a labelling bundle into paste-ready prompts for a frontier judge.

The existing fixtures were labelled this way - grimgar03's rows carry
`judged_by: "openai-2026-07-26 (single judge, provisional)"` - so this matches
the established workflow rather than inventing one. `make_fixture` samples the
lines; this formats them for the judge; `ingest_judgements` reads the reply
back and `finalise_fixture` writes the gold file.

WHAT GOLD FROM A BIG MODEL ACTUALLY MEASURES, stated plainly because it bounds
every number computed against these fixtures: accuracy is agreement with the
judge, not agreement with the truth. Three consequences that have already shown
up in this investigation rather than being hypothetical:

  - The ceiling is the judge's own accuracy. Blind adjudication of grimgar03's
    26 unanimous failures found 4 contested labels and 2 undeclared-alias rows
    - roughly a quarter of that set was fixture error, not model error.
  - A judged row can be wrong in a way that correlates with the models being
    scored. Where the text genuinely does not determine the speaker, the judge
    guesses, and a model that guesses the same way is scored correct for the
    wrong reason.
  - Single-judge labels cannot distinguish "hard" from "underdetermined". That
    is why UNKNOWN and NOT_DIALOGUE are offered as first-class answers below:
    an honest abstention is worth more than a confident coin-flip, and the rate
    of each is itself a measurement.

So: a second judge on a disagreeing subset buys more than more rows from one
judge, and the fixtures stay marked `provisional` until that happens.

NOT_DIALOGUE is asked for because segmentation is broken upstream. Third-person
narration is filed as SPOKEN in index18 and mid-sentence fragments in
owarimonogatari3; the signature is third-person pronoun density inside SPOKEN
segments, 6-8 per 1000 words in the grimgar books against 16-20 elsewhere,
mushoku16 included. Quote marks cannot detect it because segmentation strips
them. Asking the judge converts that unknown into a rate.
"""
import json, math, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
BOOK = os.environ.get("EXPERIMENT_BOOK", "index18")
PER_PROMPT = int(os.environ.get("EXPERIMENT_PER_PROMPT", "30"))
BUNDLE = os.environ.get("EXPERIMENT_BUNDLE", os.path.join(
    REPO, "ab_test_runtime", "fixtures_draft", f"labelling_bundle__{BOOK}.json"))
OUTDIR = os.environ.get("EXPERIMENT_OUTDIR", os.path.join(
    REPO, "ab_test_runtime", "fixtures_draft", f"judge_prompts__{BOOK}"))
# JSON rather than prose: the judge gets the same instructions either way, but
# a structured payload is unambiguous about which text is the target and
# survives copy-paste without the >>> markers being mangled.
AS_JSON = os.environ.get("EXPERIMENT_JSON", "1") not in ("0", "", "false")

bundle = json.load(open(BUNDLE, encoding="utf-8"))
entries = bundle["entries"]
roster = bundle["roster"]

TARGET_TXT = "the line marked >>> TARGET <<<"
TARGET_JSON = ('the segment in `passage` whose `is_target` is true (its text is '
               'repeated as `target_line`)')
HEADER = """You are labelling gold data for a speaker-attribution benchmark. For
each item below, decide who speaks %%TARGET%%.

Rules:
- Answer with a name in CAPITALS. Prefer a name from the roster, but if the
  true speaker is someone the roster omits, give that name anyway and we will
  add it.
- Answer UNKNOWN only if the passage genuinely does not determine WHO is
  speaking - you cannot tell which person it was. Do not guess to be helpful:
  an item nobody could resolve is useful to us precisely because it is
  unresolvable, and a confident wrong label is worse than an honest UNKNOWN.
- Answer UNNAMED when you CAN tell who is speaking but that person has no
  name - a subordinate knight, the French president, a voice from the back of
  the plane. This is not the same as UNKNOWN and the difference matters to us:
  UNKNOWN means the text is ambiguous, UNNAMED means the text is clear and the
  character simply has no name. Put the description in `reasoning`.
- If the speaker HAS a name and it is merely missing from the roster, give the
  name. Do not answer UNKNOWN or UNNAMED for a named character; the roster is
  incomplete and we will add them.
- Answer NOT_DIALOGUE if the target text is not speech at all. Our segmenter
  sometimes files narration, or a fragment of a sentence, as dialogue. If the
  target is narration, an incomplete clause, or otherwise not something a
  character says aloud, say so.
- `confident` is true only if you would defend the answer against a careful
  reader who disagreed.
- `reasoning` is required on every item, including UNKNOWN and NOT_DIALOGUE.
  Name the actual evidence - the dialogue tag, who was addressed, whose turn
  it is - not a restatement of the answer. For UNKNOWN say what is missing;
  for NOT_DIALOGUE say what the text is instead.
- Some characters go by more than one name. If the speaker appears in the
  roster under a different name than the passage uses, give the roster form
  and note the other in `alias`.

Return ONLY a JSON array, one object per item, no markdown:
[{"id": "...", "speaker": "NAME|UNNAMED|UNKNOWN|NOT_DIALOGUE", "confident": true, "alias": null, "reasoning": "<short clause naming the evidence>"}]

ROSTER: %s

The passage shows surrounding segments for context. NARRATOR is narration,
SPOKEN is dialogue as our segmenter classified it - that classification is
what may be wrong, so judge the text itself.
""" % ", ".join(roster)
HEADER = HEADER.replace("%TARGET%", TARGET_JSON if AS_JSON else TARGET_TXT)


def render(entry):
    lines = [f'ITEM {entry["id"]}']
    for part in entry["passage"]:
        tag = part.get("type") or "?"
        text = (part.get("text") or "").replace("\n", " ").strip()
        if part.get("target"):
            lines.append(f"  >>> TARGET <<< [{tag}] {text}")
        else:
            lines.append(f"                 [{tag}] {text}")
    return "\n".join(lines)


os.makedirs(OUTDIR, exist_ok=True)
chunks = math.ceil(len(entries) / PER_PROMPT)
for number in range(chunks):
    part = entries[number * PER_PROMPT:(number + 1) * PER_PROMPT]
    stem = f"{BOOK}_prompt_{number + 1:02d}_of_{chunks:02d}"
    if AS_JSON:
        path = os.path.join(OUTDIR, stem + ".json")
        payload = {
            "task": "speaker attribution gold labelling",
            "book": BOOK,
            "part": f"{number + 1} of {chunks}",
            "instructions": HEADER.strip(),
            "roster": roster,
            "output_schema": {
                "return": "a JSON array with one object per item, nothing else",
                "object": {"id": "string, copied exactly from the item",
                           "speaker": "NAME in capitals, or UNNAMED, or UNKNOWN, or NOT_DIALOGUE",
                           "confident": "boolean",
                           "alias": "other name for this character, or null",
                           "reasoning": "short clause naming the evidence: the dialogue tag, who was addressed, or whose turn it is. Required for every item, including UNKNOWN and NOT_DIALOGUE - for those, say what makes it unresolvable or not speech."},
                "must_cover_ids": [e["id"] for e in part],
            },
            "items": [{"id": e["id"], "target_line": e["line"],
                       "passage": [{"type": p.get("type"),
                                    "text": (p.get("text") or "").strip(),
                                    "is_target": bool(p.get("target"))}
                                   for p in e["passage"]]}
                      for e in part],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, ensure_ascii=False)
    else:
        path = os.path.join(OUTDIR, stem + ".txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(HEADER)
            fh.write(f"\n{len(part)} items ({number + 1} of {chunks}).\n\n")
            fh.write("\n\n".join(render(e) for e in part))
            fh.write("\n\nReturn the JSON array now, one object per item, "
                     f"all {len(part)} ids present.\n")
    print(f"wrote {path}  ({len(part)} items)")

print(f"\n{len(entries)} items in {chunks} prompts of up to {PER_PROMPT}.")
print("Paste each into the judge, save each JSON reply next to its prompt as")
print(f"  {OUTDIR}/reply_NN_of_{chunks:02d}.json")
print("then run ingest_judgements.py.")
