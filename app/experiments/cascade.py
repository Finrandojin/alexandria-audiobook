"""Live test of the disagreement cascade: cheap model twice, big model on the splits.

Priced offline from artifacts already collected, the rule looked like the best
result of the investigation:

                  cascade   w1 alone   70B everywhere   70B calls
    grimgar03      78.0%      57.0%         79.2%          40%
    mushoku16      65.5%      49.6%         66.9%          60%

Within 1.2 points of running the big model on everything, on both books, for
40-60% of its calls. The mechanism is sharp: on rows where the cheap model's w1
and w4 answers disagree, the cheap model scores 25.5% and the 70B scores 77.6%.
Disagreement does not merely mark hard rows, it marks rows where capability is
the thing missing.

WHY THIS RUN EXISTS. That table is a SIMULATION over arms that were executed
independently, at different times, on two machines. Every number in it is a
real measurement of a real model on the same row, and the routing rule uses no
gold - but nothing has ever executed the cascade as one pipeline. This does,
end to end, deciding routing from answers produced in this run only.

The comparison that matters is against the simulation, not against w1: if the
live result lands near 78%/65.5%, the offline pricing method is validated and
can be trusted for the next design. If it does not, the simulation was wrong
about something, and finding out which part is worth more than the cascade.

TWO PHASES, because a 70B at 43 GiB and a 14B cannot be resident together.
Phase `cheap` runs w1 and w4 and writes the routing decision to a state file;
phase `expensive` reloads with the big model and attributes only the routed
rows. A production batch run would do exactly this - collect the hard lines,
then make one expensive pass - so the split is faithful rather than a
concession.

Both phases go through `attribute_batch` with the shipping prompt at w1
context. w4 exists here ONLY as half of the disagreement detector: the 70B gate
showed w4's answers are worse than w1's on both books, so the cascade keeps w1's
answer wherever the two agree and never keeps a w4 answer at all.
"""
import collections
import json, os, re, sys, time
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")
from openai import OpenAI
from experiments.manifest import ExperimentRecord
from experiments.stats import clopper_pearson, exact_mcnemar, paired
from generate_script import LLMGenParams
from three_pass_generate import (attribute_batch, build_roster,
                                 get_deterministic_named_entry)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = REPO + "/app/"
M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"

PHASE = os.environ.get("EXPERIMENT_PHASE", "cheap")
if PHASE not in ("cheap", "expensive"):
    raise SystemExit("EXPERIMENT_PHASE must be 'cheap' or 'expensive'")
MODEL = os.environ.get("EXPERIMENT_MODEL", "qwen/qwen3-14b")
BIG_MODEL = os.environ.get("EXPERIMENT_BIG_MODEL", "llama-3.3-70b")
BOOK = os.environ.get("EXPERIMENT_BOOK", "grimgar03")
# GOLD follows BOOK by default. It used to hardcode grimgar03's fixture
# while BOOK stayed settable, so setting only EXPERIMENT_BOOK scored one
# book's lines against another book's gold - three matches out of 162,
# every arm 0.0%. Two runs were lost to it before the pattern was seen.
GOLD = os.environ.get("EXPERIMENT_GOLD",
                      f"fixtures/attribution_gold_{BOOK}.json")
GOLD_PATH = APP + GOLD
BASE_URL = os.environ.get("EXPERIMENT_BASE_URL", "http://127.0.0.1:8090/v1")
TAG = os.environ.get("EXPERIMENT_TAG", "cascade")
BATCH = 25
# Which disagreement defines "hard". The original trigger compares w1 against
# w4, but w4 was retracted today - it costs the 70B 2.5 points - so half the
# detector is an arm we no longer believe in. Priced offline on collected
# artifacts, batch-size disagreement separates BETTER and costs less:
#
#     trigger        grimgar03   mushoku16   coverage   cheap calls
#     w1 vs w4         +26.6         --        40%      98 + 98
#     b25 vs b50       +36.4       +23.2      37-42%    98 + 52
#
# Live confirmation is the point of running it: the offline pricing has been
# right once (cascade, predicted 78.0 got 77.8) and wrong once (the scattered
# batch bug), so it is evidence, not proof.
TRIGGER = os.environ.get("EXPERIMENT_TRIGGER", "width")   # width | batch
STATE = os.path.join(REPO, "ab_test_runtime", "experiments",
                     f"cascade_state__{BOOK}__{TAG}.json")

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
pos = {norm(e.get("text")): i for i, e in enumerate(seg)}
client = OpenAI(base_url=BASE_URL, api_key="local")
params = LLMGenParams(max_tokens=12000, context_length=32768, temperature=0.0,
                      attribute_temperature=0.0, top_p=0.8,
                      reasoning_effort="none")


def neighbours(index, width):
    if width <= 1:
        return {"previous_context": seg[index - 1] if index else None,
                "next_context": (seg[index + 1] if index + 1 < len(seg) else None)}
    lo, hi = max(0, index - width), min(len(seg), index + 1 + width)
    prev_txt = " ".join((seg[j].get("text") or "") for j in range(lo, index))
    next_txt = " ".join((seg[j].get("text") or "") for j in range(index + 1, hi))
    return {
        "previous_context": ({"type": "CONTEXT", "text": prev_txt} if prev_txt else None),
        "next_context": ({"type": "CONTEXT", "text": next_txt} if next_txt else None),
    }


def run_batches(model, indices, width, label):
    """Attribute the CONTIGUOUS WINDOWS containing `indices`.

    The first version chunked `indices` directly, so a batch was 25 scattered
    gold lines rather than 25 consecutive segments. That cost the cheap
    baseline 17.9 points on mushoku16 - far outside the +-2.3 local/cloud
    bound - and it is not what production does. Production walks the book in
    windows and sends every non-deterministic entry in each.

    So: build the same windows the gate builds, keep the ones containing a
    wanted index, and send each window whole. The expensive phase therefore
    re-attributes the WINDOW around each routed line, which is both faithful
    and the reason its cost is reported in windows as well as rows.
    """
    wanted = set(indices)
    windows = [list(range(s, min(s + BATCH, len(seg))))
               for s in range(0, len(seg), BATCH)]
    windows = [w for w in windows if any(i in wanted for i in w)]
    got, failures = {}, 0
    for number, window in enumerate(windows, 1):
        chunk = [i for i in window
                 if get_deterministic_named_entry(seg[i]) is None]
        if not chunk:
            continue
        frozen = [{"type": seg[i]["type"], "text": seg[i]["text"]} for i in chunk]
        contexts = [neighbours(i, width) for i in chunk]
        try:
            out = attribute_batch(client, model, frozen, params, roster,
                                  neighbor_contexts=contexts, source_text=src)
        except Exception as exc:
            print(f"  {label} window {number}: {type(exc).__name__}", flush=True)
            failures += len(chunk)
            for i in chunk:
                got[i] = None
            continue
        for offset, i in enumerate(chunk):
            got[i] = ((out[offset] or {}).get("speaker")
                      if offset < len(out) else None)
        if number % 20 == 0:
            print(f"  {label} {number}/{len(windows)} windows ...", flush=True)
    return got, failures, len(windows)


# Only entries the pipeline actually sends to the LLM; the deterministic ones
# never reach either model and would inflate both arms identically.
scoreable = [pos[k] for k in want if k in pos
             and get_deterministic_named_entry(seg[pos[k]]) is None]
scoreable.sort()
print(f"phase={PHASE} | {len(scoreable)} scoreable non-deterministic lines | "
      f"model={MODEL if PHASE=='cheap' else BIG_MODEL}", flush=True)

# ------------------------------------------------------------- phase: cheap
if PHASE == "cheap":
    started = time.time()
    if TRIGGER == "batch":
        # Same width, two batch sizes. The kept answer is still the b25 one -
        # b50 exists only to disagree with it, exactly as w4 did.
        BATCH = 25
        w1, f1, nw1 = run_batches(MODEL, scoreable, 1, "b25")
        BATCH = 50
        w4, f4, _ = run_batches(MODEL, scoreable, 1, "b50")
        BATCH = 25
    else:
        w1, f1, nw1 = run_batches(MODEL, scoreable, 1, "w1")
        w4, f4, _ = run_batches(MODEL, scoreable, 4, "w4")
    route = [i for i in scoreable
             if (w1.get(i) or "").upper() != (w4.get(i) or "").upper()]
    state = {"book": BOOK, "cheap_model": MODEL, "endpoint": BASE_URL,
             "elapsed_s": round(time.time() - started, 1),
             "w1": {str(i): w1.get(i) for i in scoreable},
             "w4": {str(i): w4.get(i) for i in scoreable},
             "route": [str(i) for i in route],
             "failures": {"w1": f1, "w4": f4}, "cheap_windows": nw1,
             "trigger": TRIGGER}
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=1)
    k1 = sum(1 for i in scoreable if same(w1.get(i), want[norm(seg[i]['text'])]["expected_speaker"]))
    print(f"\n  w1 alone {k1}/{len(scoreable)} = {k1/len(scoreable)*100:.1f}%")
    print(f"  routing {len(route)}/{len(scoreable)} "
          f"({len(route)/len(scoreable)*100:.0f}%) to {BIG_MODEL}")
    print(f"  wrote {STATE}")
    raise SystemExit(0)

# --------------------------------------------------------- phase: expensive
with open(STATE, encoding="utf-8") as fh:
    state = json.load(fh)
if state["book"] != BOOK:
    raise SystemExit(f"state file is for {state['book']}, not {BOOK}")
route = [int(i) for i in state["route"]]
w1 = {int(k): v for k, v in state["w1"].items()}
print(f"  routing {len(route)} rows decided by the cheap phase "
      f"({state['cheap_model']}, {state['elapsed_s']:.0f}s)", flush=True)

started = time.time()
big, fbig, nbig = run_batches(BIG_MODEL, route, 1, "big")

_env = os.environ.get("EXPERIMENT_ENV")
record = ExperimentRecord(
    "cascade", REPO, f"{state['cheap_model']} + {BIG_MODEL}", BASE_URL, GOLD_PATH,
    {"temperature": 0.0, "attribute_temperature": 0.0, "max_tokens": 12000,
     "batch": BATCH, "route_rule": "w1 != w4 on the cheap model", "width": 1},
    environment=json.loads(_env) if _env else None,
    notes="Live end-to-end cascade. Offline pricing predicted 78.0% grimgar03 "
          "and 65.5% mushoku16, against 79.2%/66.9% for the big model on every "
          "row. Routing is decided only from answers produced in this run.")

routed = set(route)
for i in scoreable:
    g = want[norm(seg[i]["text"])]
    cheap_answer = w1.get(i)
    final = big.get(i) if i in routed else cheap_answer
    record.add("cheap-w1", g["id"], g["line"], g["expected_speaker"].upper(),
               cheap_answer, same(cheap_answer, g["expected_speaker"]),
               provenance="cheap-w1")
    record.add("cascade", g["id"], g["line"], g["expected_speaker"].upper(),
               final, same(final, g["expected_speaker"]),
               provenance=f"cascade|{'big' if i in routed else 'cheap'}")

ans = {a: {r["id"]: r["correct"] for r in record.rows if r["arm"] == a}
       for a in ("cheap-w1", "cascade")}
n = len(ans["cascade"])
kc = sum(ans["cascade"].values())
kb = sum(ans["cheap-w1"].values())
p, x, y, _ = paired(ans["cheap-w1"], ans["cascade"])
print(f"\n  cheap-w1 {kb}/{n} = {kb/n*100:5.1f}%  "
      f"[{clopper_pearson(kb,n)[0]:.1f}-{clopper_pearson(kb,n)[1]:.1f}]")
print(f"  cascade  {kc}/{n} = {kc/n*100:5.1f}%  "
      f"[{clopper_pearson(kc,n)[0]:.1f}-{clopper_pearson(kc,n)[1]:.1f}]")
print(f"  delta {(kc-kb)/n*100:+.1f}   rescues {y} breaks {x}   p={p:.4g}")
cheap_windows = state.get("cheap_windows") or 0
print(f"  big-model cost: {nbig}/{cheap_windows} windows = "
      f"{nbig/max(cheap_windows,1)*100:.0f}% of the book's batches, covering "
      f"{len(route)}/{n} = {len(route)/n*100:.0f}% routed rows")
print(f"  expensive phase {time.time()-started:.0f}s, {fbig} rows failed")
print("  Cost is in WINDOWS, not rows: re-attributing the window around a "
      "routed\n  line is what production would do, and it is strictly more "
      "than the row count.")

# The routed subset is where the whole claim lives: if the cheap model were
# already fine there, routing would be wasted spend.
r_cheap = sum(1 for i in route if same(w1.get(i), want[norm(seg[i]['text'])]["expected_speaker"]))
r_big = sum(1 for i in route if same(big.get(i), want[norm(seg[i]['text'])]["expected_speaker"]))
if route:
    pr, xr, yr = exact_mcnemar(
        sum(1 for i in route if same(w1.get(i), want[norm(seg[i]['text'])]["expected_speaker"])
            and not same(big.get(i), want[norm(seg[i]['text'])]["expected_speaker"])),
        sum(1 for i in route if same(big.get(i), want[norm(seg[i]['text'])]["expected_speaker"])
            and not same(w1.get(i), want[norm(seg[i]['text'])]["expected_speaker"])))
    print(f"\n  on the routed rows only: cheap {r_cheap/len(route)*100:.1f}%  "
          f"big {r_big/len(route)*100:.1f}%  (+{yr}/-{xr}, p={pr:.4g})")

out = record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"cascade__{BOOK}__{BIG_MODEL.replace('/', '__')}__{TAG}.json"),
    contract={"expected_arms": ("cheap-w1", "cascade")})
print("wrote", out)
