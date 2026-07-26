"""Does pass 2 lose accuracy because the roster is built as it goes?

An experimental harness that supplied the complete final roster scored 35.6%
where production scored 29.9%. Production admits characters incrementally, so
early batches cannot select someone not yet established. That is a hypothesis,
not a finding - the harness also differed in retry handling and in never
carrying partial roster state - so this measures it directly.

Arms, identical frozen segmentation and decoding:

  incremental  roster grows batch by batch, as production does
  warm         roster discovered in a first pass, complete before scoring
  oracle       every gold answer's speaker, plus the discovered roster
               (diagnostic only - an upper bound on roster availability)

Reported by book position as well as overall. A roster effect must concentrate
in the first quartile, because that is where an incremental roster is most
incomplete. A flat gain means something else differs between harness and
production, and that must be found before shipping a two-pass design.
"""
import collections
import json, os, re, sys
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")
from openai import OpenAI
from experiments.manifest import ExperimentRecord


def _safe_name(model):
    """Model keys carry a publisher prefix ('qwen/qwen3-14b'), and a slash in a
    filename silently creates a directory instead of naming the artifact."""
    return model.replace("/", "__")
from generate_script import LLMGenParams
from three_pass_generate import (attribute_batch, build_roster,
                                 attested_new_speakers,
                                 get_deterministic_named_entry)

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
APP = REPO + "/app/"
M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
# The model under test. Frozen inputs below always come from INPUT_RUN, so a
# comparison across models is a comparison of selection, not of segmentation.
MODEL = os.environ.get("EXPERIMENT_MODEL",
                       "qwen3.5-9b-uncensored-hauhaucs-aggressive")
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
BASE_URL = "http://localhost:1234/v1"
BATCH = 25

gold = json.load(open(APP + "fixtures/attribution_gold_random.json"))
src = open(M + "inputs/mushoku16.txt", encoding="utf-8").read()
cp = json.load(open(M + INPUT_RUN + "/mushoku16/result.json.threepass_checkpoint.json"))
seg = cp["segmented"]
final_named = [e for e in (cp.get("named") or []) if e]
discovered = build_roster(final_named, src)
AL = [{"RUDEUS", "RUDI"}, {"SYLPHY", "SYLPHIETTE"}]

def same(a, b):
    a, b = (a or "").upper(), (b or "").upper()
    return a == b or any(a in g and b in g for g in AL)

def norm(t): return re.sub(r"\W+", "", t or "").lower()
want = {norm(g["line"]): g for g in gold["entries"]}
pos = {norm(e["text"]): i for i, e in enumerate(seg)}

# See two_by_two.py: a gold line whose text repeats cannot be aligned to one
# position, and scoring it at each occurrence counts one judgement twice.
_occurrences = collections.Counter(norm(e.get("text")) for e in seg)
want = {key: value for key, value in want.items() if _occurrences[key] == 1}
print(f"scoring {len(want)} gold lines with unambiguous text", flush=True)

client = OpenAI(base_url=BASE_URL, api_key="local")
params = LLMGenParams(max_tokens=12000, context_length=32768, temperature=0.0,
                      attribute_temperature=0.0, top_p=0.8,
                      reasoning_effort="none")
record = ExperimentRecord(
    "roster_warmup", REPO, MODEL, BASE_URL,
    APP + "fixtures/attribution_gold_random.json",
    {"temperature": 0.0, "max_tokens": 12000, "reasoning_effort": "none"},
    notes="incremental vs warm vs oracle roster, reported by book quartile.")

windows = [list(range(s, min(s + BATCH, len(seg)))) for s in range(0, len(seg), BATCH)]
scored_windows = [w for w in windows if any(norm(seg[i].get("text")) in want for i in w)]

def run(arm):
    if arm == "incremental":
        roster, seen = [], set()
    elif arm == "warm":
        roster, seen = list(discovered), set(discovered)
    else:
        extra = [g["expected_speaker"].upper() for g in gold["entries"]]
        roster = list(dict.fromkeys(list(discovered) + extra))
        seen = set(roster)
    # Incremental must walk every window so the roster grows as production's
    # does; the others only need the windows holding a gold line.
    walk = windows if arm == "incremental" else scored_windows
    for n, idxs in enumerate(walk, 1):
        send = [i for i in idxs if get_deterministic_named_entry(seg[i]) is None]
        if not send:
            continue
        scored_here = [i for i in send if norm(seg[i].get("text")) in want]
        if arm == "incremental" and not scored_here:
            # Still needs the roster contribution, taken from the frozen run
            # rather than paying for a call that scores nothing.
            for name in attested_new_speakers(
                    [final_named[i] for i in send if i < len(final_named)], seen, src):
                seen.add(name); roster.append(name)
            continue
        frozen = [{"type": seg[i]["type"], "text": seg[i]["text"]} for i in send]
        contexts = [{"previous_context": seg[i - 1] if i else None,
                     "next_context": seg[i + 1] if i + 1 < len(seg) else None}
                    for i in send]
        out = attribute_batch(client, MODEL, frozen, params, roster=roster,
                              on_exhaustion="fallback", max_retries=3,
                              neighbor_contexts=contexts, source_text=src)
        if not out or len(out) != len(frozen):
            continue
        for i, r in zip(send, out):
            key = norm(seg[i].get("text"))
            if key in want:
                g = want[key]
                record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(),
                           r.get("speaker"), same(r.get("speaker"),
                                                  g["expected_speaker"]),
                           candidates=[x.upper() for x in roster],
                           provenance=arm)
        if arm == "incremental":
            for name in attested_new_speakers(out, seen, src):
                seen.add(name); roster.append(name)
        if n % 20 == 0:
            print(f"  {arm} {n}/{len(walk)} ...", flush=True)

for arm in ("incremental", "warm", "oracle"):
    run(arm)
    rows = [r for r in record.rows if r["arm"] == arm]
    hit = sum(1 for r in rows if r["correct"])
    print(f"{arm:12} {hit}/{len(rows)} = {hit/max(len(rows),1)*100:.1f}%", flush=True)

print("\nby book quartile (entry index of the scored line):")
for arm in ("incremental", "warm", "oracle"):
    rows = [r for r in record.rows if r["arm"] == arm]
    buckets = [[] for _ in range(4)]
    for r in rows:
        i = pos.get(norm(r["line"]), 0)
        buckets[min(3, int(i / max(len(seg), 1) * 4))].append(r["correct"])
    parts = "  ".join(f"Q{q+1} {sum(b)}/{len(b)}={sum(b)/max(len(b),1)*100:4.1f}%"
                      for q, b in enumerate(buckets))
    print(f"  {arm:12} {parts}")
print("\nbaseline (shipped pipeline): 44/147 = 29.9%")
# Declare what this run must produce, so an artifact that silently drops an arm
# or half its lines is refused rather than validated on the arithmetic of
# whatever it managed to record.
contract = {"expected_arms": ("incremental", "warm", "oracle"),
            "expected_ids": {g["id"] for g in gold["entries"]
                             if norm(g["line"]) in want},
            "require_clean_tree": True}
print("wrote", record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"roster_warmup__{_safe_name(MODEL)}.json"), contract=contract))
