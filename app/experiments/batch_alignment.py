"""If contiguity is worth 16.6 points, do batch BOUNDARIES matter?

`batch_contiguity` established the mechanism behind the largest lever in this
investigation: a batch of 25 consecutive segments beats 25 scattered ones by
16.6 points (p=1.4e-07) at the same prompt size. The model attributes a
conversation better than it attributes isolated lines.

Production cuts a new batch every 25 segments regardless of where conversations
begin and end. So a run of turn-taking that straddles a boundary is split, and
both halves get a weakened version of the scattered condition - some of their
conversation is in the other batch. Nobody has tested whether that costs
anything.

TWO ARMS:

    fixed       windows of 25 consecutive sendable segments - production today
    aligned     windows built by packing whole RUNS of consecutive spoken
                segments, never splitting one, up to the same size budget

Both arms are contiguous; both send the same kind of prompt; both score the
same rows. The only difference is where the cuts fall.

READINGS, fixed before running:

  aligned >> fixed    boundaries matter and production is losing accuracy to an
                      arbitrary cut. A cheap, shippable change.
  aligned ~ fixed     the model recovers enough from the neighbour contexts it
                      already gets, and 25-segment cuts are fine. Closes the
                      lever that batch_contiguity opened.
  aligned << fixed    packing whole runs produces less uniform batches and the
                      size variance costs more than the alignment buys - check
                      the batch-size spread before concluding anything else.

BATCH SIZE IS REPORTED PER ARM. Aligned windows cannot all be exactly 25, and
batch size is itself worth points, so a difference in mean size is a
confounder. If the aligned arm's mean size drifts far from 25, its result is
about size as much as alignment and must not be read as alignment alone.
"""
import collections, json, os, re, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = REPO + "/app/"
sys.path.insert(0, APP)
from openai import OpenAI
from experiments.manifest import ExperimentRecord
from experiments.scoring import alias_groups, same_speaker
from experiments.stats import clopper_pearson, paired
from generate_script import LLMGenParams
from three_pass_generate import (attribute_batch, build_roster,
                                 get_deterministic_named_entry)

M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"

MODEL = os.environ.get("EXPERIMENT_MODEL", "qwen/qwen3-14b")
BOOK = os.environ.get("EXPERIMENT_BOOK", "grimgar03")
GOLD_PATH = APP + os.environ.get(
    "EXPERIMENT_GOLD", f"fixtures/attribution_gold_{BOOK}.json")
BASE_URL = os.environ.get("EXPERIMENT_BASE_URL", "http://127.0.0.1:8090/v1")
TAG = os.environ.get("EXPERIMENT_TAG", "local-llamacpp")
BATCH = int(os.environ.get("EXPERIMENT_BATCH", "25"))
SPECIAL = {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE"}


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


gold = json.load(open(GOLD_PATH))
src = open(M + f"inputs/{BOOK}.txt", encoding="utf-8").read()
cp = json.load(open(M + INPUT_RUN + f"/{BOOK}/result.json.threepass_checkpoint.json"))
seg = cp["segmented"]
GROUPS = alias_groups(gold)
roster = [r.upper() for r in
          build_roster([e for e in (cp.get("named") or []) if e], src)]
roster = sorted(set(roster) | {n.upper() for n in
                               gold.get("roster_additions", {}).get("names", [])})
occ = collections.Counter(norm(e.get("text")) for e in seg)
want = {norm(g["line"]): g for g in gold["entries"]
        if occ[norm(g["line"])] == 1
        and g["expected_speaker"].upper() not in SPECIAL}

sendable = [i for i, e in enumerate(seg)
            if get_deterministic_named_entry(e) is None
            and (e.get("text") or "").strip()]
scored = set(i for i in sendable if norm(seg[i].get("text")) in want)

# Production: cut every BATCH sendable segments, wherever that lands.
fixed = [sendable[s:s + BATCH] for s in range(0, len(sendable), BATCH)]

# Aligned: a run is a maximal stretch of consecutive spoken segments with no
# narration between them - the unit batch_contiguity showed the model exploits.
# Runs are packed whole; a run longer than the budget becomes its own window
# rather than being split, since splitting it is the very thing being tested.
runs, current = [], []
for i in sendable:
    if current and (i - current[-1] > 1
                    or any(seg[j].get("type") == "NARRATOR"
                           for j in range(current[-1] + 1, i))):
        runs.append(current)
        current = []
    current.append(i)
if current:
    runs.append(current)

aligned, bucket = [], []
for run in runs:
    if bucket and len(bucket) + len(run) > BATCH:
        aligned.append(bucket)
        bucket = []
    if len(run) > BATCH:
        if bucket:
            aligned.append(bucket)
            bucket = []
        aligned.append(run)
        continue
    bucket.extend(run)
if bucket:
    aligned.append(bucket)

fixed = [w for w in fixed if any(i in scored for i in w)]
aligned = [w for w in aligned if any(i in scored for i in w)]
mean = lambda ws: sum(len(w) for w in ws) / max(len(ws), 1)
print(f"{BOOK}: {len(want)} scoreable lines, {len(sendable)} sendable, "
      f"{len(runs)} spoken runs", flush=True)
print(f"  fixed   {len(fixed):4} windows, mean {mean(fixed):.1f} entries")
print(f"  aligned {len(aligned):4} windows, mean {mean(aligned):.1f} entries")
print("  If those means differ much, the result is about size as well as "
      "alignment.", flush=True)

ARMS = ("fixed", "aligned")
client = OpenAI(base_url=BASE_URL, api_key="local")
params = LLMGenParams(max_tokens=12000, context_length=32768, temperature=0.0,
                      attribute_temperature=0.0, top_p=0.8,
                      reasoning_effort="none")
_env = os.environ.get("EXPERIMENT_ENV")
record = ExperimentRecord(
    "batch_alignment", REPO, MODEL, BASE_URL, GOLD_PATH,
    {"temperature": 0.0, "batch": BATCH},
    environment=json.loads(_env) if _env else None,
    notes="Does WHERE a batch is cut matter, given contiguity is worth 16.6 "
          "points? Production cuts every 25 segments regardless of where "
          "conversations start and end; the aligned arm packs whole runs of "
          "consecutive spoken segments instead.")
record.enable_checkpoint(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"batch_alignment__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json.ckpt"))

summary, sizes = {}, {}
for arm in ARMS:
    windows = fixed if arm == "fixed" else aligned
    sizes[arm] = mean(windows)
    started = time.time()
    for k, win in enumerate(windows, 1):
        rows = [i for i in win if i in scored]
        if not rows:
            continue
        if all(record.done(arm, want[norm(seg[i].get("text"))]["id"]) for i in rows):
            continue
        frozen = [{"type": seg[i]["type"], "text": seg[i]["text"]} for i in win]
        ctx = [{"previous_context": seg[i - 1] if i else None,
                "next_context": seg[i + 1] if i + 1 < len(seg) else None}
               for i in win]
        try:
            out = attribute_batch(client, MODEL, frozen, params, roster,
                                  neighbor_contexts=ctx, source_text=src)
        except Exception as exc:
            print(f"  {arm} window {k}: {type(exc).__name__}", flush=True)
            for i in rows:
                g = want[norm(seg[i].get("text"))]
                if not record.done(arm, g["id"]):
                    record.add(arm, g["id"], g["line"],
                               g["expected_speaker"].upper(), None, False,
                               provenance=f"{arm}|batch_failed")
            continue
        for off, i in enumerate(win):
            if i not in scored:
                continue
            g = want[norm(seg[i].get("text"))]
            sp = (out[off] or {}).get("speaker") if off < len(out) else None
            record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(),
                       sp, same_speaker(g["expected_speaker"], sp, GROUPS),
                       provenance=f"{arm}|n={len(win)}")
        if k % 25 == 0:
            print(f"  {arm} {k}/{len(windows)} ...", flush=True)
    arm_rows = [r for r in record.rows if r["arm"] == arm]
    hit = sum(1 for r in arm_rows if r["correct"])
    summary[arm] = (hit, len(arm_rows))
    lo, hi = clopper_pearson(hit, max(len(arm_rows), 1))
    print(f"  {arm:9} {hit}/{len(arm_rows)} = "
          f"{hit/max(len(arm_rows),1)*100:5.1f}%  [{lo:.1f}-{hi:.1f}]  "
          f"mean {sizes[arm]:.1f} entries  {time.time()-started:.0f}s", flush=True)

ans = {a: {r["id"]: r["correct"] for r in record.rows if r["arm"] == a}
       for a in summary}
p, x, y, n = paired(ans["fixed"], ans["aligned"])
h0, n0 = summary["fixed"]
h1, n1 = summary["aligned"]
print(f"\n  aligned - fixed  {(h1/max(n1,1) - h0/max(n0,1))*100:+.1f} points  "
      f"+{y}/-{x} of {n}  p={p:.4g}")
print("  Read it against the mean batch sizes above: if they differ much, this "
      "is\n  size as well as alignment.")

out = record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"batch_alignment__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json"),
    contract={"expected_arms": ARMS})
print("wrote", out)
