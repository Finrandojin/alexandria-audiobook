"""Is context the problem, or is joint multi-target classification the problem?

Two context experiments both lost points, but each changed several things at
once: prompt length, duplicated neighbour text, rows to read but not answer,
and 25 bindings to preserve. This varies exactly two factors.

  A batch of 25, no context      C batch of 25, +/-1 context
  B single target, no context    D single target, +/-1 context

Reading:
  A~B, C<D  -> context helps alone, harmed by batch interference
  A<B, C<D  -> joint multi-target classification is the problem
  C<A, D<B  -> the context representation is actively misleading
  D>B, C<=A -> context is useful but the batch format prevents its use

Run on an idle GPU. Temperature 0.
"""
import collections
import json, os, re, sys
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")
from openai import OpenAI
from generate_script import LLMGenParams
from three_pass_generate import (attribute_batch, build_roster,
                                 get_deterministic_named_entry)
from experiments.manifest import ExperimentRecord

M = ("/home/fakemitch/pinokio/api/alexandria-audiobook2.git/"
     "ab_test_runtime/results/matrix_20260725-115148/")
MODEL = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
APP = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app/"
BOOK = os.environ.get("EXPERIMENT_BOOK", "mushoku16")
GOLD_PATH = APP + os.environ.get(
    "EXPERIMENT_GOLD", "fixtures/attribution_gold_random.json")
gold = json.load(open(GOLD_PATH))
src = open(M + f"inputs/{BOOK}.txt", encoding="utf-8").read()
cp = json.load(open(M + MODEL + "/" + BOOK + "/result.json.threepass_checkpoint.json"))
seg, named = cp["segmented"], [e for e in (cp.get("named") or []) if e]
roster = build_roster(named, src)
AL = [{"RUDEUS", "RUDI"}, {"SYLPHY", "SYLPHIETTE"}]

def same(a, b):
    a, b = (a or "").upper(), (b or "").upper()
    return a == b or any(a in g and b in g for g in AL)

def norm(t): return re.sub(r"\W+", "", t or "").lower()
want = {norm(g["line"]): g["expected_speaker"] for g in gold["entries"]}
pos = {norm(e["text"]): i for i, e in enumerate(seg)}

# A gold line whose text occurs at several positions cannot be aligned to one of
# them, so scoring it counts one judgement two or three times. The roster
# experiment recorded 155 rows for 146 lines that way and produced three
# identical arm totals that looked like a finding. Same defect the audit found
# in build_scoring_sheet; excluded here for the same reason.
_occurrences = collections.Counter(norm(e.get("text")) for e in seg)
AMBIGUOUS = {key for key in want if _occurrences[key] > 1}
print(f"excluding {len(AMBIGUOUS)} gold lines whose text repeats in the book; "
      f"{len(want) - len(AMBIGUOUS)} scoreable", flush=True)
BASE_URL = "http://localhost:1234/v1"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
client = OpenAI(base_url=BASE_URL, api_key="local")
record = ExperimentRecord(
    "two_by_two", REPO, MODEL, BASE_URL, GOLD_PATH,
    {"temperature": 0.0, "max_tokens": 12000, "reasoning_effort": "none"},
    notes="Batch-25 vs single target x no-context vs +/-1 context. NOTE: the "
          "two factors are not independent - a batch of consecutive lines is "
          "itself context - so this prices context, not batching.")
params = LLMGenParams(max_tokens=12000, context_length=32768, temperature=0.0,
                      attribute_temperature=0.0, top_p=0.8,
                      reasoning_effort="none")

def contexts_for(indexes):
    return [{"previous_context": seg[i - 1] if i else None,
             "next_context": seg[i + 1] if i + 1 < len(seg) else None}
            for i in indexes]

def run(batched, with_context, label):
    if batched:
        windows = [list(range(s, min(s + 25, len(seg))))
                   for s in range(0, len(seg), 25)]
        windows = [w for w in windows if any(norm(seg[i].get("text")) in want for i in w)]
    else:
        windows = [[i] for i in sorted({pos[k] for k in want if k in pos})]
    got = {}
    for n, idxs in enumerate(windows, 1):
        send = [i for i in idxs if get_deterministic_named_entry(seg[i]) is None]
        if not send:
            continue
        frozen = [{"type": seg[i]["type"], "text": seg[i]["text"]} for i in send]
        out = attribute_batch(
            client, MODEL, frozen, params, roster=roster,
            on_exhaustion="fallback", max_retries=3, source_text=src,
            neighbor_contexts=contexts_for(send) if with_context else None)
        if not out or len(out) != len(frozen):
            continue
        for i, r in zip(send, out):
            key = norm(seg[i].get("text"))
            if key in want and key not in AMBIGUOUS:
                got[key] = r.get("speaker")
                record.add(label.split()[0], key, seg[i].get("text"),
                           want[key].upper(), r.get("speaker"),
                           same(r.get("speaker"), want[key]),
                           provenance=("batch" if batched else "single") +
                                      ("+context" if with_context else "+nocontext"))
        if n % 25 == 0:
            print(f"  {label} {n}/{len(windows)} ...", flush=True)
    hit = sum(1 for k, v in got.items() if same(v, want[k]))
    print(f"{label:34} {hit}/{len(got)} = {hit/max(len(got),1)*100:.1f}%", flush=True)
    return hit, len(got)

results = {}
for key, batched, ctx, label in (
        ("A", True, False, "A batch-25,  no context"),
        ("B", False, False, "B single,    no context"),
        ("C", True, True, "C batch-25,  +/-1 context"),
        ("D", False, True, "D single,    +/-1 context")):
    results[key] = run(batched, ctx, label)
print("\nbaseline (shipped pipeline): 44/147 = 29.9%")
out = record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments", "two_by_two.json"))
print("wrote", out)
