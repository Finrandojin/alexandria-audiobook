"""Does telling the model who narrates rescue first-person books?

`narration_signal` found the cause of the 16-point book gap: the pipeline reads
neighbouring narration to find the speaker, and first-person narration rarely
names anyone. mushoku16's adjacent narration names the true speaker 35.3% of
the time against grimgar03's 77.6%, and both books score the same (43.1% vs
45.0%) on the rows where that signal is absent. The gap is composition over one
feature.

That diagnosis implies first-person books need a signal the pipeline does not
currently use, and it names the obvious one: the narrator is a KNOWN CONSTANT
who speaks a large share of all lines. In mushoku16, Rudeus narrates and 38% of
gold lines are his - the single most predictive fact available about the book,
and nothing in the prompt mentions it.

ARMS (production path, w1 context, batch 25, everything else the shipping
configuration):

    baseline        the shipped attribution prompt
    narrator        the same prompt plus one sentence naming the narrator and
                    noting that unattributed interior lines are usually theirs

Only the system prompt differs, which is the isolation `thinking` had and
`because` did not - `because` also changed the output schema, and that is why
its diagnostic gain reversed in production.

THE CONTROL IS THE SECOND BOOK, and the prediction is asymmetric. If the
diagnosis is right:

    mushoku16 (first-person)   should GAIN - the prior supplies what the
                               narration does not
    grimgar03 (third-person)   should be flat or slightly worse - the narration
                               already names people, and Haruhiro speaks only
                               28% of lines

A gain on BOTH books means the sentence is doing something generic (priming,
extra instruction) rather than supplying a first-person prior, and the
first-person story is not established. A gain on NEITHER retires the narrator
prior and leaves turn-taking and register as the remaining candidates for the
hard regime - both already tested and null, which would make the unanchored
rows genuinely underdetermined rather than merely under-served.

This is the first experiment in the investigation designed from a mechanism
rather than from a list of things to try, so its failure is informative in a
way the earlier prompt sweeps were not.
"""
import collections
import json, os, re, sys, time
from dataclasses import replace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
from default_prompts import load_attribute_prompts
from experiments.manifest import ExperimentRecord
from narrator_prompt import add_narrator_prior
from experiments.stats import clopper_pearson, paired
from generate_script import LLMGenParams
from three_pass_generate import (attribute_batch, build_roster,
                                 get_deterministic_named_entry)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = REPO + "/app/"
M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"

MODEL = os.environ.get("EXPERIMENT_MODEL", "qwen/qwen3-14b")
BOOK = os.environ.get("EXPERIMENT_BOOK", "mushoku16")
GOLD = os.environ.get("EXPERIMENT_GOLD", "fixtures/attribution_gold_random.json")
GOLD_PATH = APP + GOLD
BASE_URL = os.environ.get("EXPERIMENT_BASE_URL", "http://127.0.0.1:8090/v1")
TAG = os.environ.get("EXPERIMENT_TAG", "local-llamacpp")
# Named explicitly rather than inferred: guessing the narrator from the text is
# a separate problem, and getting it wrong would confound this experiment with
# the accuracy of that guess. Production would take it from book metadata or a
# one-off LLM call, neither of which is what is being measured here.
NARRATOR = os.environ.get("EXPERIMENT_NARRATOR",
                          "RUDEUS" if BOOK == "mushoku16" else "HARUHIRO")
BATCH = 25
MAX_UNATTRIBUTED = float(os.environ.get("EXPERIMENT_MAX_UNATTRIBUTED", "0.25"))

gold = json.load(open(GOLD_PATH))
src = open(M + f"inputs/{BOOK}.txt", encoding="utf-8").read()
cp = json.load(open(M + INPUT_RUN + f"/{BOOK}/result.json.threepass_checkpoint.json"))
seg = cp["segmented"]
roster = build_roster([e for e in (cp.get("named") or []) if e], src)
AL = [{n.upper() for n in g} for g in gold.get("aliases", [])]


def same(a, b):
    a, b = (a or "").upper(), (b or "").upper()
    return a == b or any(a in g and b in g for g in AL)


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


_occ = collections.Counter(norm(e.get("text")) for e in seg)
want = {norm(g["line"]): g for g in gold["entries"] if _occ[norm(g["line"])] == 1}
share = sum(1 for g in want.values()
            if same(g["expected_speaker"], NARRATOR)) / max(len(want), 1) * 100
print(f"roster {len(roster)} | {len(want)} lines | narrator {NARRATOR} "
      f"speaks {share:.0f}% of them", flush=True)

BASE_SYSTEM, _ = load_attribute_prompts()
NARRATOR_SYSTEM = add_narrator_prior(BASE_SYSTEM, NARRATOR)
ARMS = {"baseline": BASE_SYSTEM, "narrator": NARRATOR_SYSTEM}
_want = [a.strip() for a in os.environ.get("EXPERIMENT_ARMS", "").split(",") if a.strip()]
if _want:
    ARMS = {a: ARMS[a] for a in _want}

client = OpenAI(base_url=BASE_URL, api_key="local")
params = LLMGenParams(max_tokens=12000, context_length=32768, temperature=0.0,
                      attribute_temperature=0.0, top_p=0.8,
                      reasoning_effort="none")

_env = os.environ.get("EXPERIMENT_ENV")
record = ExperimentRecord(
    "narrator_prior", REPO, MODEL, BASE_URL, GOLD_PATH,
    {"temperature": 0.0, "attribute_temperature": 0.0, "max_tokens": 12000,
     "batch": BATCH, "narrator": NARRATOR, "width": 1},
    environment=json.loads(_env) if _env else None,
    notes=f"Naming the narrator ({NARRATOR}) in the system prompt. Designed "
          f"from the narration-signal diagnosis: first-person narration names "
          f"the speaker 35.3% of the time in mushoku16 against 77.6% in "
          f"grimgar03. Predicts a gain on the first-person book and none on "
          f"the third-person control.")
record.enable_checkpoint(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"narrator_prior__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json.ckpt"))

windows = [list(range(s, min(s + BATCH, len(seg)))) for s in range(0, len(seg), BATCH)]
windows = [w for w in windows if any(norm(seg[i].get("text")) in want for i in w)]

summary = {}
for arm, system in ARMS.items():
    started = time.time()
    this = replace(params, system_prompt=system)
    for n, window in enumerate(windows, 1):
        send = [i for i in window if get_deterministic_named_entry(seg[i]) is None]
        if not send or not any(norm(seg[i].get("text")) in want for i in send):
            continue
        if all(record.done(arm, want[norm(seg[i].get("text"))]["id"])
               for i in send if norm(seg[i].get("text")) in want):
            continue
        frozen = [{"type": seg[i]["type"], "text": seg[i]["text"]} for i in send]
        contexts = [{"previous_context": seg[i - 1] if i else None,
                     "next_context": seg[i + 1] if i + 1 < len(seg) else None}
                    for i in send]
        try:
            out = attribute_batch(client, MODEL, frozen, this, roster,
                                  neighbor_contexts=contexts, source_text=src)
        except Exception as exc:
            print(f"  {arm} window {n}: {type(exc).__name__}", flush=True)
            for i in send:
                key = norm(seg[i].get("text"))
                if key in want and not record.done(arm, want[key]["id"]):
                    g = want[key]
                    record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(),
                               None, False, provenance=f"{arm}|batch_failed")
            continue
        for offset, i in enumerate(send):
            key = norm(seg[i].get("text"))
            if key not in want:
                continue
            g = want[key]
            speaker = (out[offset] or {}).get("speaker") if offset < len(out) else None
            record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(),
                       speaker, same(speaker, g["expected_speaker"]),
                       provenance=f"{arm}|narrator_line="
                                  f"{same(g['expected_speaker'], NARRATOR)}")
        if n % 25 == 0:
            print(f"  {arm} {n}/{len(windows)} ...", flush=True)
    rows = [r for r in record.rows if r["arm"] == arm]
    hit = sum(1 for r in rows if r["correct"])
    unatt = sum(1 for r in rows if r["predicted"] is None)
    summary[arm] = (hit, len(rows), unatt, time.time() - started)
    lo, hi = clopper_pearson(hit, max(len(rows), 1))
    print(f"  {arm:9} {hit}/{len(rows)} = {hit/max(len(rows),1)*100:5.1f}%  "
          f"[{lo:.1f}-{hi:.1f}]  {unatt} unattributed  "
          f"{time.time()-started:.0f}s", flush=True)

for arm, (h, n, u, _) in summary.items():
    if n and u / n > MAX_UNATTRIBUTED:
        raise SystemExit(f"refusing to write: {arm} left {u}/{n} unattributed")

if len(summary) == 2:
    ans = {a: {r["id"]: r["correct"] for r in record.rows if r["arm"] == a}
           for a in ARMS}
    p, x, y, n = paired(ans["baseline"], ans["narrator"])
    h0, n0 = summary["baseline"][0], summary["baseline"][1]
    h1, n1 = summary["narrator"][0], summary["narrator"][1]
    print(f"\n  narrator vs baseline: {(h1/n1 - h0/n0)*100:+.1f} points  "
          f"+{y}/-{x}  p={p:.4g}")

    # The prior should move the NARRATOR'S OWN lines most. If the gain lands
    # somewhere else, the sentence is doing something generic and the mechanism
    # story is not supported by its own result.
    marks = {r["id"]: ("narrator_line=True" in (r.get("candidate_provenance") or ""))
             for r in record.rows if r["arm"] == "baseline"}
    print(f"\n  {'subset':22} {'baseline':>9} {'narrator':>9} {'delta':>7}  n")
    for flag, label in ((True, f"{NARRATOR}'s own lines"), (False, "everyone else")):
        ids = [i for i in ans["baseline"] if marks.get(i) is flag]
        if not ids:
            continue
        b = sum(1 for i in ids if ans["baseline"][i]) / len(ids) * 100
        v = sum(1 for i in ids if ans["narrator"][i]) / len(ids) * 100
        print(f"  {label:22} {b:8.1f}% {v:8.1f}% {v-b:+7.1f}  {len(ids)}")

out = record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"narrator_prior__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json"),
    contract={"expected_arms": tuple(ARMS)})
print("wrote", out)
