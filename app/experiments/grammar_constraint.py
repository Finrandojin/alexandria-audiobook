"""Does constraining decoding to the roster repair off-list answers?

Measured error class this targets: answers that are not in the candidate list
the model was handed. On mushoku16 magistral-small produced 19 of them in 60
oracle errors (13.7% of all oracle rows), spelling RUDEUS as RUDIUS / RUDIEUS /
RUDUEUS and ALMANFI as ARUMANFI - the right character under a different Japanese
romanization. Other models show 5.0-5.8% on the same arm. On the OPEN arm the
same class is 0.0-1.4%.

A GBNF grammar makes those outputs structurally impossible: the sampler can only
emit one of the supplied names. The interesting question is not whether off-list
answers disappear - they must - but whether forcing a valid choice REPAIRS the
line or merely relabels one error as another. If the model wanted RUDIUS and the
grammar leaves only RUDEUS reachable, that is a repair. If it wanted RUDIUS
because it had the wrong character in mind, the constraint just picks a
different wrong name.

Production relevance: pass 2 currently generates speaker names freely and a
post-hoc is_attested_name gate REJECTS unattested ones - a gate added after
finding 279 invented speakers per book. Rejection discards the answer entirely.
A grammar moves the same check from after decoding to during it, so the model
must pick its best VALID option instead of having its answer thrown away.

What this cannot do: roster recall is 85% while accuracy is 30-60%, so most
errors are picking the WRONG ROSTER NAME, which a grammar permits. The ceiling
here is the invented/misspelled class only.

Arms, all on the same lines and the same server:

  open-free      full roster in the prompt, unconstrained decoding   (as today)
  open-grammar   full roster in the prompt AND as a grammar
  oracle-free    true speaker + 4 distractors, unconstrained         (as today)
  oracle-grammar same 5 names, as a grammar

open is the arm that resembles production; oracle is where the effect should be
largest. Running both tests the mechanism and its production relevance at once.

Requires llama.cpp - `grammar` is a llama.cpp server field, not an OpenAI one,
and is one of the capabilities LM Studio does not expose.
"""
import collections
import json, os, re, sys, random, time
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")
import openai
from openai import OpenAI
from experiments.manifest import ExperimentRecord
from three_pass_generate import build_roster

RETRYABLE = (openai.APIConnectionError, openai.APITimeoutError,
             openai.InternalServerError, openai.RateLimitError,
             openai.NotFoundError)
MAX_ATTEMPTS = 6

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = REPO + "/app/"
M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"

MODEL = os.environ.get("EXPERIMENT_MODEL", "mistralai/magistral-small")
BOOK = os.environ.get("EXPERIMENT_BOOK", "mushoku16")
GOLD = os.environ.get("EXPERIMENT_GOLD", "fixtures/attribution_gold_random.json")
GOLD_PATH = APP + GOLD
BASE_URL = os.environ.get("EXPERIMENT_BASE_URL", "http://127.0.0.1:8080/v1")
TAG = os.environ.get("EXPERIMENT_TAG", "local-llamacpp")

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
print(f"scoring {len(SCOREABLE)} of {len(gold['entries'])} gold lines, roster {len(roster)}",
      flush=True)

client = OpenAI(base_url=BASE_URL, api_key="local")
SYSTEM = ("You identify who speaks a line of dialogue in a novel. Answer with "
          "the speaker's name in CAPITALS and nothing else. If the passage "
          "does not determine it, answer UNKNOWN.")


def gbnf(choices):
    """A grammar admitting exactly these names and nothing else.

    Quotes inside a name would break the literal, so they are stripped rather
    than escaped - no roster name in either fixture contains one, and silently
    emitting a malformed grammar would be worse than dropping a character.
    """
    literals = " | ".join('"%s"' % c.replace('"', '') for c in choices + ["UNKNOWN"])
    return "root ::= " + literals + "\n"


def ask(line, index, choices, constrain):
    before = " ".join((seg[j].get("text") or "")
                      for j in range(max(0, index - 4), index))
    after = " ".join((seg[j].get("text") or "")
                     for j in range(index + 1, min(len(seg), index + 4)))
    user = (f"PASSAGE BEFORE:\n{before}\n\nLINE:\n{line}\n\n"
            f"PASSAGE AFTER:\n{after}\n"
            f"\nThe speaker is one of: {', '.join(choices + ['UNKNOWN'])}\n\n"
            f"Who speaks the LINE?")
    # The prompt is IDENTICAL in both arms - the candidate list is stated either
    # way. Only the sampler differs, so any difference is the constraint itself
    # and not a prompt change. That distinction is what made `thinking` a fair
    # test and `because` an unfair one.
    extra = {"reasoning_effort": "none"}
    if constrain:
        extra["grammar"] = gbnf(choices)
    last = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=[{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": user}],
                temperature=0.0, max_tokens=24, extra_body=extra)
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
    "grammar_constraint", REPO, MODEL, BASE_URL, GOLD_PATH,
    {"temperature": 0.0, "max_tokens": 24, "reasoning_effort": "none"},
    environment=json.loads(_env) if _env else None,
    notes="GBNF-constrained decoding vs free decoding, identical prompts. Tests "
          "whether forcing a roster-valid answer repairs off-list errors "
          "(5-14% of oracle rows, 0-1.4% of open rows) or merely relabels them.")
record.enable_checkpoint(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"grammar_constraint__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json.ckpt"))

ARMS = [("open-free", "open", False), ("open-grammar", "open", True),
        ("oracle-free", "oracle", False), ("oracle-grammar", "oracle", True)]
summary = {}
for arm, kind, constrain in ARMS:
    started = time.time()
    offlist = 0
    for g in SCOREABLE:
        i = pos.get(norm(g["line"]))
        if i is None or record.done(arm, g["id"]):
            continue
        truth = g["expected_speaker"]
        if kind == "open":
            choices = roster
        else:
            # Identical distractor draw to closed_set.py's oracle arm, so the
            # free arm here is comparable to the existing artifacts.
            distractors = [x for x in roster if not same(x, truth)]
            choices = [truth] + random.Random(i).sample(
                distractors, min(4, len(distractors)))
            random.Random(i + 1).shuffle(choices)
        got, prompt, raw, retries = ask(g["line"], i, choices, constrain)
        ok = same(got, truth)
        if got and got not in [c.upper() for c in choices] and got != "UNKNOWN":
            offlist += 1
        record.add(arm, g["id"], g["line"], truth.upper(), got, ok,
                   candidates=[c.upper() for c in choices],
                   provenance=("grammar" if constrain else "free"),
                   prompt=prompt, raw=raw, retries=retries)
    rows = [r for r in record.rows if r["arm"] == arm]
    hit = sum(1 for r in rows if r["correct"])
    summary[arm] = (hit, len(rows), offlist, time.time() - started)
    print(f"{arm:15} {hit}/{len(rows)} = {hit/max(len(rows),1)*100:5.1f}%   "
          f"off-list {offlist:3}   {time.time()-started:.0f}s", flush=True)

print("\n  arm pair            free      grammar   delta   off-list free->grammar")
for kind in ("open", "oracle"):
    f, gr = summary[f"{kind}-free"], summary[f"{kind}-grammar"]
    print(f"  {kind:18} {f[0]/f[1]*100:6.1f}%  {gr[0]/gr[1]*100:6.1f}%  "
          f"{(gr[0]-f[0])/f[1]*100:+6.1f}   {f[2]:3} -> {gr[2]:3}")
print("\n  A grammar arm with any off-list answers means the constraint did not "
      "apply - check that the server accepted the `grammar` field.")

out = record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"grammar_constraint__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json"),
    contract={"expected_arms": tuple(a for a, _, _ in ARMS),
              "expected_ids": {g["id"] for g in SCOREABLE},
              "require_clean_tree": True})
print("wrote", out)
