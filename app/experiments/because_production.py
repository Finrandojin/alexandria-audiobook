"""Does a justification field still help on the production path?

The reasoning arms measured `because` at +10.8 points, p=0.004 - the first
significant positive architectural result in this investigation. But that
harness called the model directly with a simplified prompt and scored its own
baseline at 39.6%, while the production path through attribute_batch scores
49.6% on the same lines. The gain was measured against a baseline ten points
below what production already achieves, so it may not survive.

That is the same error this investigation kept making with negatives, applied to
a positive: the instrument was not the thing being shipped.

Every arm here runs through attribute_batch with the production prompt, so the
only difference is the system prompt and the output schema.

  baseline            the shipping prompt, {n, speaker}
  because             + a one-clause justification per line
  scaffold_thinking   the judge's questions with reasoning enabled - 48.2% at
                      p=0.088 in the earlier run, plausibly real and underpowered
"""
import collections
import json, os, re, sys, time
from dataclasses import replace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
from default_prompts import load_attribute_prompts
from experiments.manifest import ExperimentRecord
from generate_script import LLMGenParams
from three_pass_generate import attribute_batch, build_roster, get_deterministic_named_entry

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = REPO + "/app/"
M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
MODEL = os.environ.get("EXPERIMENT_MODEL", "qwen/qwen3-14b")
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
BASE_URL = "http://localhost:1234/v1"
BATCH = 25

# The exploratory arms reversed between books - `because` won on mushoku16 and
# is null on grimgar03, thinking the reverse - so a production-path check has
# to name its book rather than assume one. These must precede every use: an
# identical ordering slip once let three runs exit in nine seconds while the
# loop around them logged "done".
BOOK = os.environ.get("EXPERIMENT_BOOK", "mushoku16")
GOLD = os.environ.get("EXPERIMENT_GOLD", "fixtures/attribution_gold_random.json")
TAG = os.environ.get("EXPERIMENT_TAG",
                     "local" if "localhost" in BASE_URL or "127.0.0.1"
                     in BASE_URL else "remote")

gold = json.load(open(APP + GOLD))
src = open(M + f"inputs/{BOOK}.txt", encoding="utf-8").read()
cp = json.load(open(M + INPUT_RUN + f"/{BOOK}/result.json.threepass_checkpoint.json"))
seg = cp["segmented"]
roster = build_roster([e for e in (cp.get("named") or []) if e], src)
AL = [{n.upper() for n in g} for g in gold.get("aliases", [])]

def same(a, b):
    a, b = (a or "").upper(), (b or "").upper()
    return a == b or any(a in g and b in g for g in AL)

def norm(t): return re.sub(r"\W+", "", t or "").lower()
_occ = collections.Counter(norm(e.get("text")) for e in seg)
want = {norm(g["line"]): g for g in gold["entries"] if _occ[norm(g["line"])] == 1}
pos = {norm(e["text"]): i for i, e in enumerate(seg)}

BASE_SYSTEM, _ = load_attribute_prompts()
BECAUSE_SYSTEM = BASE_SYSTEM.replace(
    'return {"n": <same index>, "speaker": "..."} where:',
    'return {"n": <same index>, "speaker": "...", "because": "<one short '
    'clause naming the evidence: a dialogue tag, who was addressed, whose '
    'turn it is>"} where:').replace(
    "Output ONLY a valid JSON array — no markdown, no explanations.",
    "Output ONLY a valid JSON array — no markdown. The explanation belongs in "
    "the \"because\" field of each entry, nowhere else.")
SCAFFOLD_SYSTEM = BECAUSE_SYSTEM.replace(
    '"because": "<one short clause naming the evidence: a dialogue tag, who '
    'was addressed, whose turn it is>"',
    '"tag": "<the dialogue tag verbatim, or null>", "addressed": "<who is '
    'spoken TO, or null>", "previous_speaker": "<who spoke the previous line, '
    'or null>"')

print(f"roster {len(roster)} | scoring {len(want)} unambiguous lines", flush=True)
client = OpenAI(base_url=BASE_URL, api_key="local")
params = LLMGenParams(max_tokens=12000, context_length=32768, temperature=0.0,
                      attribute_temperature=0.0, top_p=0.8, reasoning_effort="none")
record = ExperimentRecord(
    "because_production", REPO, MODEL, BASE_URL,
    APP + GOLD,
    {"temperature": 0.0, "max_tokens": 12000, "batch": BATCH},
    notes="Justification field on the production path, against the production "
          "baseline rather than a simplified harness.")

windows = [list(range(s, min(s + BATCH, len(seg)))) for s in range(0, len(seg), BATCH)]
windows = [w for w in windows if any(norm(seg[i].get("text")) in want for i in w)]

# "thinking" is BASE_SYSTEM with reasoning left on: the SAME prompt as
# baseline, differing only in reasoning_effort. That isolation matters. The
# `because` reversal came from an arm that changed the prompt AND the schema,
# so it was compared against a baseline weakened by a different prompt;
# thinking changes neither, so if it fails here it fails on its own terms.
#
# ARMS is overridable because the gate question is baseline vs thinking, and
# running the other two costs GPU time that answers nothing about it.
ALL_ARMS = {"baseline": (BASE_SYSTEM, False),
            "because": (BECAUSE_SYSTEM, False),
            "thinking": (BASE_SYSTEM, True),
            "scaffold_thinking": (SCAFFOLD_SYSTEM, True)}
_want = [a.strip() for a in
         os.environ.get("EXPERIMENT_ARMS", "").split(",") if a.strip()]
for _a in _want:
    if _a not in ALL_ARMS:
        raise SystemExit(f"unknown arm {_a!r}; have {sorted(ALL_ARMS)}")
ARMS = {a: ALL_ARMS[a] for a in _want} if _want else ALL_ARMS
elapsed = {}
for arm, (system, thinks) in ARMS.items():
    started = time.time()
    this = replace(params, system_prompt=system,
                   reasoning_effort=None if thinks else "none")
    for n, window in enumerate(windows, 1):
        send = [i for i in window if get_deterministic_named_entry(seg[i]) is None]
        if not send or not any(norm(seg[i].get("text")) in want for i in send):
            continue
        frozen = [{"type": seg[i]["type"], "text": seg[i]["text"]} for i in send]
        contexts = [{"previous_context": seg[i-1] if i else None,
                     "next_context": seg[i+1] if i+1 < len(seg) else None}
                    for i in send]
        out = attribute_batch(client, MODEL, frozen, this, roster=roster,
                              on_exhaustion="fallback", max_retries=3,
                              neighbor_contexts=contexts, source_text=src)
        if not out or len(out) != len(frozen):
            continue
        for i, r in zip(send, out):
            key = norm(seg[i].get("text"))
            if key not in want:
                continue
            g = want[key]
            record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(),
                       r.get("speaker"), same(r.get("speaker"),
                                              g["expected_speaker"]),
                       provenance=arm)
        if n % 25 == 0:
            print(f"  {arm} {n}/{len(windows)} ...", flush=True)
    elapsed[arm] = round(time.time() - started, 1)
    rows = [r for r in record.rows if r["arm"] == arm]
    hit = sum(1 for r in rows if r["correct"])
    print(f"{arm:18} {hit}/{len(rows)} = {hit/max(len(rows),1)*100:.1f}%   "
          f"{elapsed[arm]:.0f}s", flush=True)

record.meta["elapsed_by_arm_s"] = elapsed
print("\nwrote", record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"because_production__{BOOK}__{MODEL.replace(chr(47), chr(95)*2)}__{TAG}.json"),
    contract={"expected_arms": tuple(ARMS)}))
