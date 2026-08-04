"""Re-run four negatives under the current, corrected harness conventions.

Every one was recorded before some defect was found and fixed, and a negative
result can be produced by any single break in the chain - the scaffold arm
scored a correct idea at 7.7% on a prompt bug. Re-examining two of the eight
negatives on CPU already found instrument defects in both, so these four get
the same treatment.

  voting     KNOWN broken, not merely suspect. Its prototype called
             build_roster(named[:400]) and ran with a 5-name roster. That was
             recorded at the time as invalidating the result, and the result
             was never re-taken.
  prose      -5.4 pt. Changed the OUTPUT format, which the reasoning arms have
             since shown is the sensitive axis, and drew 2.1 retries per batch
             against 1.4 - a format-instability penalty is exactly what would
             produce a spurious negative.
  narration  -2.1 pt. Same output-format concern; narration rows had to be
             answered and validated, so mislabelling one failed a whole batch.
  narrator   no effect. Measured earliest, before the gold set had aliases and
             before repeated lines were excluded, so its denominator was wrong.

All arms share the frozen inputs, temperature 0, and the same 139 unambiguous
gold lines. Only the manipulation differs.
"""
import collections
import json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
from experiments.manifest import ExperimentRecord
from dataclasses import replace
from default_prompts import load_attribute_prompts
from generate_script import LLMGenParams
from three_pass_generate import (attribute_batch, attribute_batch_voted,
                                 build_roster, get_deterministic_named_entry)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = REPO + "/app/"
M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
MODEL = os.environ.get("EXPERIMENT_MODEL", "qwen/qwen3-14b")
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
BASE_URL = "http://localhost:1234/v1"
BATCH = 25

BOOK = os.environ.get("EXPERIMENT_BOOK", "mushoku16")
GOLD_PATH = APP + os.environ.get(
    "EXPERIMENT_GOLD", "fixtures/attribution_gold_random.json")
gold = json.load(open(GOLD_PATH))
src = open(M + f"inputs/{BOOK}.txt", encoding="utf-8").read()
cp = json.load(open(M + INPUT_RUN + "/" + BOOK + "/result.json.threepass_checkpoint.json"))
seg = cp["segmented"]
# The full attested roster, not a truncated prefix. The voting prototype's
# five-name roster is the defect being corrected here.
roster = build_roster([e for e in (cp.get("named") or []) if e], src)
AL = [{n.upper() for n in g} for g in gold.get("aliases", [])]

def same(a, b):
    a, b = (a or "").upper(), (b or "").upper()
    return a == b or any(a in g and b in g for g in AL)

def norm(t): return re.sub(r"\W+", "", t or "").lower()
_occ = collections.Counter(norm(e.get("text")) for e in seg)
want = {norm(g["line"]): g for g in gold["entries"] if _occ[norm(g["line"])] == 1}
pos = {norm(e["text"]): i for i, e in enumerate(seg)}
print(f"roster {len(roster)} names | scoring {len(want)} unambiguous lines", flush=True)

client = OpenAI(base_url=BASE_URL, api_key="local")
params = LLMGenParams(max_tokens=12000, context_length=32768, temperature=0.0,
                      attribute_temperature=0.0, top_p=0.8, reasoning_effort="none")
record = ExperimentRecord(
    "reexamine", REPO, MODEL, BASE_URL,
    GOLD_PATH,
    {"temperature": 0.0, "max_tokens": 12000, "batch": BATCH},
    notes="Four previously-negative results re-run under corrected conventions.")

windows = [list(range(s, min(s + BATCH, len(seg)))) for s in range(0, len(seg), BATCH)]
windows = [w for w in windows if any(norm(seg[i].get("text")) in want for i in w)]

NARRATOR_HINT = ("The narration is first person: the narrator says \"I\" and is "
                 "never named by the narration. When a line is spoken by the "
                 "person telling the story, the speaker is RUDEUS.")

BASE_SYSTEM, _BASE_USER = load_attribute_prompts()

# The reverted interleaved-passage format, restored only as an experiment arm.
# It was measured at -5.4 points while drawing 2.1 retries per batch against
# 1.4, so the loss may have been format instability rather than the format.
PROSE_SYSTEM = (
    "You assign speaker names to lines of a novel. Output ONLY a valid JSON "
    "array, no prose and no markdown.\n"
    "You receive a PASSAGE in reading order. Lines prefixed [n] need a speaker; "
    "unprefixed lines are surrounding narration, for reading only.\n"
    'For each [n] line return {"n": <same index>, "speaker": "NAME"}. Narration '
    "lines that are themselves numbered take NARRATOR. Use UNKNOWN when the "
    "passage does not determine the speaker.\n"
    'Example of the ONLY valid output shape: [{"n": 0, "speaker": "ROXY"}]')
PROSE_USER = ("ESTABLISHED ROSTER: {roster}\n\n"
              "Read the whole passage, then assign a speaker to each [n] line:\n\n"
              "{batch}")


def prose_passage(indexes):
    """Render the batch as prose in reading order, numbering only the targets."""
    first, last = min(indexes), max(indexes)
    targets = {index: n for n, index in enumerate(indexes)}
    lines = []
    for j in range(max(0, first - 4), min(len(seg), last + 5)):
        text = " ".join((seg[j].get("text") or "").split())
        lines.append(f"[{targets[j]}] {text}" if j in targets else text)
    return "\n".join(lines)


def prose_attribute(indexes):
    """One batch through the interleaved-passage format, direct to the model."""
    user = PROSE_USER.format(roster=", ".join(roster) or "(none yet)",
                             batch=prose_passage(indexes))
    try:
        response = client.chat.completions.create(
            model=MODEL, temperature=0.0, max_tokens=8000,
            messages=[{"role": "system", "content": PROSE_SYSTEM},
                      {"role": "user", "content": user}],
            extra_body={"reasoning_effort": "none"})
    except Exception as error:
        print(f"  prose call failed: {type(error).__name__}", flush=True)
        return []
    text = (response.choices[0].message.content or "")
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return []
    try:
        answers = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    by_index = {a.get("n"): a for a in answers if isinstance(a, dict)}
    if answers and None in by_index:
        print(f"  WARNING prose: {sum(1 for a in answers if a.get('n') is None)}"
              f"/{len(answers)} responses omitted the index", flush=True)
    return [{"speaker": (by_index.get(n) or {}).get("speaker")}
            for n in range(len(indexes))]


def contexts_for(indexes):
    return [{"previous_context": seg[i - 1] if i else None,
             "next_context": seg[i + 1] if i + 1 < len(seg) else None}
            for i in indexes]

def run(arm):
    started = time.time()
    for n, window in enumerate(windows, 1):
        if arm == "narration":
            send = window                      # narration stays in the batch
        else:
            send = [i for i in window if get_deterministic_named_entry(seg[i]) is None]
        if not send or not any(norm(seg[i].get("text")) in want for i in send):
            continue
        frozen = [{"type": seg[i]["type"], "text": seg[i]["text"]} for i in send]
        kwargs = dict(on_exhaustion="fallback", max_retries=3, source_text=src,
                      neighbor_contexts=contexts_for(send))
        this = params
        kwargs["roster"] = roster
        if arm == "narrator":
            # Appended to the system prompt, which attribute_batch honours.
            # The original probe injected it as a pseudo-roster entry, which
            # renders as a fake character name in the ESTABLISHED ROSTER line.
            this = replace(params, system_prompt=BASE_SYSTEM + "\n" + NARRATOR_HINT)
        if arm == "prose":
            this = replace(params, system_prompt=PROSE_SYSTEM,
                           user_prompt_template=PROSE_USER)
        if arm == "prose":
            # Called directly rather than through attribute_batch: the passage
            # machinery was removed from production as a measured regression,
            # and re-adding a parameter there to serve one experiment arm would
            # put experiment-only code on the shipping path.
            out = prose_attribute(send)
        elif arm == "voting":
            out, _conf = attribute_batch_voted(client, MODEL, frozen, this,
                                               votes=3, vote_temperature=0.3,
                                               **kwargs)
        else:
            out = attribute_batch(client, MODEL, frozen, this, **kwargs)
        if not out or len(out) != len(frozen):
            continue
        for i, r in zip(send, out):
            key = norm(seg[i].get("text"))
            if key not in want:
                continue
            g = want[key]
            det = get_deterministic_named_entry(seg[i])
            speaker = (det["speaker"] if det and det["speaker"] == "NARRATOR"
                       else r.get("speaker"))
            record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(),
                       speaker, same(speaker, g["expected_speaker"]),
                       provenance=arm)
        if n % 25 == 0:
            print(f"  {arm} {n}/{len(windows)} ...", flush=True)
    rows = [r for r in record.rows if r["arm"] == arm]
    hit = sum(1 for r in rows if r["correct"])
    print(f"{arm:10} {hit}/{len(rows)} = {hit/max(len(rows),1)*100:.1f}%   "
          f"{time.time()-started:.0f}s", flush=True)
    return round(time.time() - started, 1)

elapsed = {}
for arm in ("baseline", "voting", "narration", "narrator", "prose"):
    elapsed[arm] = run(arm)
record.meta["elapsed_by_arm_s"] = elapsed
print("\nwrote", record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"reexamine__{MODEL.replace('/','__')}.json"),
    contract={"expected_arms": ("baseline", "voting", "narration", "narrator",
                               "prose")}))
