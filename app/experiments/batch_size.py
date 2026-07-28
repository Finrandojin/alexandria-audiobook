"""How many lines should one attribution call carry? Production says 25, unmeasured.

`BATCH = 25` appears in every harness in this directory because the shipped
pipeline uses it, and nobody has ever varied it. Three separate results now
point at that number being wrong:

  1. joint_scene    one line per call scored 60.5% on grimgar03; the same lines
                    attributed a scene at a time scored 57.0%. Putting more
                    lines in one call LOST points.
  2. cascade        batching 25 scattered gold rows instead of 25 consecutive
                    segments dropped the cheap baseline 17.9 points on
                    mushoku16 - far outside the +-2.3 local/cloud bound.
  3. the w4 gate    an exhausted batch costs every row in it. w4 left 37 rows
                    unattributed at batch 25; at batch 1 the same failure rate
                    would cost 37 individual rows instead of whole windows.

Those are three different experiments that were not designed to measure this,
which is exactly the sort of accidental convergence worth testing directly. If
it holds, batch size is a free accuracy parameter: no new model, no prompt
change, one config number.

ARMS: 1, 5, 10, 25 lines per call, w1 context, contiguous windows, everything
else identical to the production gate.

Readings, fixed before running:

  accuracy rises as batch shrinks   ship the smallest batch throughput allows;
                                    every result in the ledger measured at 25
                                    is then an underestimate
  flat                              the cascade drop was COMPOSITION (scattered
                                    vs consecutive), not size - test that next,
                                    and the joint_scene gap is about prompt
                                    shape rather than line count
  accuracy falls as batch shrinks   the batch supplies useful cross-line
                                    context; 25 is defensible and the cascade
                                    harness simply broke it

TWO COSTS ARE TRACKED SEPARATELY, because they move in opposite directions.
Small batches repeat the roster and instructions per call, so tokens and wall
time go UP. Small batches also lose fewer rows per failure, so unattributed
rows go DOWN. A recommendation needs both numbers, not just accuracy.
"""
import collections
import json, os, re, sys, time
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")
from openai import OpenAI
from experiments.manifest import ExperimentRecord
from experiments.stats import clopper_pearson, paired
from generate_script import LLMGenParams
from three_pass_generate import (attribute_batch, build_roster,
                                 get_deterministic_named_entry)

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
APP = REPO + "/app/"
M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"

MODEL = os.environ.get("EXPERIMENT_MODEL", "qwen/qwen3-14b")
BOOK = os.environ.get("EXPERIMENT_BOOK", "grimgar03")
GOLD = os.environ.get("EXPERIMENT_GOLD",
                      "fixtures/attribution_gold_grimgar03_provisional.json")
GOLD_PATH = APP + GOLD
BASE_URL = os.environ.get("EXPERIMENT_BASE_URL", "http://127.0.0.1:8090/v1")
TAG = os.environ.get("EXPERIMENT_TAG", "local-llamacpp")
SIZES = [int(s) for s in os.environ.get("EXPERIMENT_SIZES", "1,5,10,25").split(",")]
MAX_UNATTRIBUTED = float(os.environ.get("EXPERIMENT_MAX_UNATTRIBUTED", "0.35"))

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
print(f"roster {len(roster)} | scoring {len(want)} lines | sizes {SIZES}", flush=True)

client = OpenAI(base_url=BASE_URL, api_key="local")
params = LLMGenParams(max_tokens=12000, context_length=32768, temperature=0.0,
                      attribute_temperature=0.0, top_p=0.8,
                      reasoning_effort="none")

_env = os.environ.get("EXPERIMENT_ENV")
record = ExperimentRecord(
    "batch_size", REPO, MODEL, BASE_URL, GOLD_PATH,
    {"temperature": 0.0, "attribute_temperature": 0.0, "max_tokens": 12000,
     "sizes": SIZES, "width": 1},
    environment=json.loads(_env) if _env else None,
    notes="Lines per attribution call, the one production parameter never "
          "varied. joint_scene, the cascade baseline drop and the w4 gate's "
          "exhaustion cost all point at 25 being too many.")
record.enable_checkpoint(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"batch_size__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json.ckpt"))

summary = {}
for size in SIZES:
    arm = f"b{size}"
    started, calls, failures = time.time(), 0, 0
    # Contiguous windows, exactly like the production gate: only the number of
    # segments per call changes, so this isolates size from composition.
    windows = [list(range(s, min(s + size, len(seg))))
               for s in range(0, len(seg), size)]
    windows = [w for w in windows if any(norm(seg[i].get("text")) in want for i in w)]
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
        calls += 1
        try:
            out = attribute_batch(client, MODEL, frozen, params, roster,
                                  neighbor_contexts=contexts, source_text=src)
        except Exception as exc:
            failures += 1
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
                       speaker, same(speaker, g["expected_speaker"]), provenance=arm)
        if n % 100 == 0:
            print(f"  {arm} {n}/{len(windows)} ...", flush=True)
    rows = [r for r in record.rows if r["arm"] == arm]
    hit = sum(1 for r in rows if r["correct"])
    unatt = sum(1 for r in rows if r["predicted"] is None)
    summary[arm] = (hit, len(rows), unatt, calls, failures, time.time() - started)
    lo, hi = clopper_pearson(hit, max(len(rows), 1))
    print(f"  {arm:5} {hit}/{len(rows)} = {hit/max(len(rows),1)*100:5.1f}%  "
          f"[{lo:.1f}-{hi:.1f}]  {unatt} unattributed  {calls} calls "
          f"({failures} failed)  {time.time()-started:.0f}s", flush=True)

for arm, (h, n, u, _, _, _) in summary.items():
    if n and u / n > MAX_UNATTRIBUTED:
        raise SystemExit(f"refusing to write: {arm} left {u}/{n} unattributed")

print(f"\n  {'batch':6} {'accuracy':>9} {'unattributed':>13} {'calls':>7} {'seconds':>8}")
for size in SIZES:
    h, n, u, c, f, t = summary[f"b{size}"]
    print(f"  {size:<6} {h/n*100:8.1f}% {u:13} {c:7} {t:8.0f}")

base = f"b{SIZES[-1]}"          # 25 is the incumbent, so everything is paired against it
ans = {a: {r["id"]: r["correct"] for r in record.rows if r["arm"] == a} for a in summary}
print(f"\n  paired against {base} (production)")
for size in SIZES[:-1]:
    arm = f"b{size}"
    p, x, y, n = paired(ans[base], ans[arm])
    h0, n0 = summary[base][0], summary[base][1]
    h1, n1 = summary[arm][0], summary[arm][1]
    print(f"    {arm:5} {(h1/n1 - h0/n0)*100:+6.1f} points  +{y:3}/-{x:3}  p={p:.4g}"
          f"   {summary[arm][5]/max(summary[base][5],1):.1f}x wall time")
print("\n  Accuracy and wall time move in opposite directions here, so the")
print("  recommendation is whichever size buys the most points per unit of")
print("  throughput lost - not simply the best accuracy.")

out = record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"batch_size__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json"),
    contract={"expected_arms": tuple(f"b{s}" for s in SIZES)})
print("wrote", out)
