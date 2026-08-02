"""Is attribution context-starved rather than model-limited?

Production pass 2 shows the model ONE segment either side of the line it is
attributing. The diagnostic harness shows four. Measured prompts are 176 tokens
median against a 16384-token window - about 1.1% utilisation.

Meanwhile the error analysis says 62.1% of errors have no character name
anywhere nearby, 12.6% name the person being addressed rather than the speaker,
and only 6.8% had a speech-verb tag the model missed. Those are the symptoms of
a model that cannot see enough text, not one that cannot reason about it.

Six models spanning 4x in parameter count and four architectures all land
between 45.6% and 55.4% with no significant separation. A plateau that flat
across model choice is more consistent with an input limit than a capability
limit - which makes context width the cheapest untested explanation.

Arms: the SAME line, the SAME roster, the same decoding, varying only how many
neighbouring segments are supplied.

    w1    one segment either side   - what production actually does
    w4    four either side          - what every diagnostic in the ledger used
    w15   fifteen either side
    w40   forty either side         - approaching whole-scene

If accuracy is flat across these, context is not the constraint and the plateau
needs another explanation. If it rises, production is leaving points on the
table for free, since none of these come close to filling the window.

Runs against llama.cpp by default: it is ~11% faster than LM Studio on identical
work here, and unlike crossover.py this harness never calls
ensure_ideal_settings, so it will not fight an externally managed server for
control of the model.
"""
import collections
import json, os, re, sys, time
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")
import openai
from openai import OpenAI
from experiments.manifest import ExperimentRecord
from three_pass_generate import build_roster

RETRYABLE = (openai.APIConnectionError, openai.APITimeoutError,
             openai.InternalServerError, openai.RateLimitError,
             openai.NotFoundError)
MAX_ATTEMPTS = 6

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
APP = REPO + "/app/"
M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"

MODEL = os.environ.get("EXPERIMENT_MODEL", "qwen/qwen3-14b")
BOOK = os.environ.get("EXPERIMENT_BOOK", "grimgar03")
# GOLD follows BOOK by default. It used to hardcode grimgar03's fixture
# while BOOK stayed settable, so setting only EXPERIMENT_BOOK scored one
# book's lines against another book's gold - three matches out of 162,
# every arm 0.0%. Two runs were lost to it before the pattern was seen.
GOLD = os.environ.get("EXPERIMENT_GOLD",
                      f"fixtures/attribution_gold_{BOOK}.json")
GOLD_PATH = APP + GOLD
BASE_URL = os.environ.get("EXPERIMENT_BASE_URL", "http://127.0.0.1:8090/v1")
TAG = os.environ.get("EXPERIMENT_TAG", "local-llamacpp")
WIDTHS = [int(w) for w in os.environ.get("EXPERIMENT_WIDTHS", "1,4,15,40").split(",")]

gold = json.load(open(GOLD_PATH))
src = open(M + f"inputs/{BOOK}.txt", encoding="utf-8").read()
cp = json.load(open(M + INPUT_RUN + f"/{BOOK}/result.json.threepass_checkpoint.json"))
seg, named = cp["segmented"], [e for e in (cp.get("named") or []) if e]
roster = [r.upper() for r in build_roster(named, src)]
AL = [{n.upper() for n in g} for g in gold.get("aliases", [])]


def same(a, b):
    a, b = (a or "").upper(), (b or "").upper()
    return a == b or any(a in g and b in g for g in AL)


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


pos = {norm(e["text"]): i for i, e in enumerate(seg)}
_occ = collections.Counter(norm(e.get("text")) for e in seg)
SCOREABLE = [g for g in gold["entries"] if _occ[norm(g["line"])] == 1]
print(f"scoring {len(SCOREABLE)} of {len(gold['entries'])} gold lines, roster "
      f"{len(roster)}, widths {WIDTHS}", flush=True)

client = OpenAI(base_url=BASE_URL, api_key="local")
SYSTEM = ("You identify who speaks a line of dialogue in a novel. Answer with "
          "the speaker's name in CAPITALS and nothing else. If the passage "
          "does not determine it, answer UNKNOWN.")


# Deliberately the same verb set the candidates module uses, so "explicit tag"
# means the same thing here as everywhere else in the ledger.
SPEECH_VERB = (r"\b(SAID|ASKED|REPLIED|ANSWERED|SHOUTED|WHISPERED|MUTTERED|"
               r"CALLED|CRIED|YELLED|GROANED|SIGHED|LAUGHED|NODDED|EXCLAIMED|"
               r"BELLOWED|AGREED|TOLD|ADDED|CONTINUED|BEGAN|OFFERED)\b")


def evidence_distance(index, truth):
    """How far from the target line is the nearest mention of the true speaker,
    and of any explicit speech tag?

    The reviewer's objection to this experiment is correct and worth recording
    per row: "62.1% of errors have no character name nearby" does NOT imply a
    wider window contains the answer. The name may be absent at any width, and
    extra prose may dilute the signal. A global average can hide "wide context
    helps distant-evidence rows and harms local dialogue", which would call for
    adaptive width or retrieval rather than shipping w40 everywhere.

    Distances are in segments, signed away from the target; None means the
    evidence does not appear within the widest window tested.
    """
    first = (truth or "").split()[0].upper()
    name_at = tag_at = None
    for d in range(1, max(WIDTHS) + 1):
        for j in (index - d, index + d):
            if not (0 <= j < len(seg)):
                continue
            text = (seg[j].get("text") or "")
            up = text.upper()
            if name_at is None and first and re.search(r"\b" + re.escape(first) + r"\b", up):
                name_at = d
            if tag_at is None and re.search(SPEECH_VERB, up):
                tag_at = d
        if name_at is not None and tag_at is not None:
            break
    return name_at, tag_at


def ask(line, index, width):
    before = " ".join((seg[j].get("text") or "")
                      for j in range(max(0, index - width), index))
    after = " ".join((seg[j].get("text") or "")
                     for j in range(index + 1, min(len(seg), index + 1 + width)))
    user = (f"PASSAGE BEFORE:\n{before}\n\nLINE:\n{line}\n\n"
            f"PASSAGE AFTER:\n{after}\n"
            f"\nThe speaker is one of: {', '.join(roster + ['UNKNOWN'])}\n\n"
            f"Who speaks the LINE?")
    last = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=[{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": user}],
                temperature=0.0, max_tokens=24,
                extra_body={"reasoning_effort": "none"})
            raw = (r.choices[0].message.content or "")
            return raw.strip().upper().strip(".'\" "), user, raw, attempt
        except RETRYABLE as exc:
            last = exc
            if attempt == MAX_ATTEMPTS - 1:
                break
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"endpoint failed {MAX_ATTEMPTS} attempts against "
                       f"{BASE_URL}: {type(last).__name__}: {last}") from last


_env = os.environ.get("EXPERIMENT_ENV")
record = ExperimentRecord(
    "context_width", REPO, MODEL, BASE_URL, GOLD_PATH,
    {"temperature": 0.0, "max_tokens": 24, "reasoning_effort": "none",
     "widths": WIDTHS},
    environment=json.loads(_env) if _env else None,
    notes="Neighbouring segments supplied to pass 2, varying only the window. "
          "Production uses w1; every diagnostic in the ledger used w4; measured "
          "prompts fill about 1.1% of the context window while 62.1% of errors "
          "have no character name nearby.")
record.enable_checkpoint(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"context_width__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json.ckpt"))

summary = {}
for width in WIDTHS:
    arm = f"w{width}"
    started, chars = time.time(), []
    for g in SCOREABLE:
        i = pos.get(norm(g["line"]))
        if i is None or record.done(arm, g["id"]):
            continue
        got, prompt, raw, retries = ask(g["line"], i, width)
        chars.append(len(prompt))
        name_at, tag_at = evidence_distance(i, g["expected_speaker"])
        record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(), got,
                   same(got, g["expected_speaker"]),
                   candidates=roster,
                   # Stratifying on evidence distance is the point: a width that
                   # helps distant-evidence rows while harming local dialogue
                   # would average out to nothing and be shipped as a null.
                   provenance=f"{arm}|name_at={name_at}|tag_at={tag_at}",
                   prompt=prompt, raw=raw, retries=retries)
    rows = [r for r in record.rows if r["arm"] == arm]
    hit = sum(1 for r in rows if r["correct"])
    med = sorted(chars)[len(chars) // 2] if chars else 0
    summary[arm] = (hit, len(rows), med, time.time() - started)
    print(f"  {arm:5} {hit}/{len(rows)} = {hit/max(len(rows),1)*100:5.1f}%   "
          f"median prompt {med:5} chars (~{med//4} tok)   "
          f"{time.time()-started:.0f}s", flush=True)

print("\n  width  accuracy   vs w1   median prompt")
base = summary[f"w{WIDTHS[0]}"]
for width in WIDTHS:
    h, n, med, _ = summary[f"w{width}"]
    print(f"  {width:5}  {h/n*100:6.1f}%  {(h-base[0])/n*100:+6.1f}   {med:6} chars")
print("\n  A flat profile means context is not the constraint and the ~50% "
      "plateau needs another explanation.")

out = record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"context_width__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json"),
    contract={"expected_arms": tuple(f"w{w}" for w in WIDTHS),
              "expected_ids": {g["id"] for g in SCOREABLE},
              "require_clean_tree": True})
print("wrote", out)
