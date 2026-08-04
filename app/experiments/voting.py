"""Does majority voting beat greedy - and is its confidence a routing signal?

`attribute_batch_voted` has been in the codebase the whole time and appears in
none of this investigation's results. It samples the same batch once per fixed
seed at temperature 0.3 and takes the per-entry majority, returning the winning
share as a confidence. Two separate questions, and this measures both:

  ACCURACY   does the majority beat greedy on the production path?
  SIGNAL     does a split vote mark a line the pipeline got wrong?

The second question matters as much as the first because of what was measured
offline today: routing the 40% of rows where w1 and w4 disagree to a 70B took
grimgar03 from 67.5% to 74.2% (p=0.0006) at 40% of the big model's calls. That
router needs TWO cheap runs at different widths. A split vote would be a
cheaper trigger from a single configuration - if it separates right from wrong
as well. The comparison is stated in those terms rather than as "is voting
confident", because a confidence that is high everywhere is useless for
routing no matter how well calibrated it looks.

Earlier notes in `majority_vote` cite 81% greedy/unanimous agreement against
42% on split votes, measured on mushoku16 before this fixture and harness
existed. Those numbers are not comparable to anything in the current ledger,
which is the reason to re-measure rather than cite them.

Arms:

    greedy   votes=1, byte-identical to the shipped path
    vote3    three seeds, majority
    vote5    five seeds, majority

Cost is linear in votes, so vote5 has to earn a 5x bill. Pre-registered: a gain
under 3 points is not worth 5x on a book-length run, whatever its p-value.
"""
import collections
import json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
from experiments.manifest import ExperimentRecord
from experiments.stats import clopper_pearson, paired
from generate_script import LLMGenParams
from three_pass_generate import (attribute_batch_voted, build_roster,
                                 get_deterministic_named_entry)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
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
BATCH = 25
MAX_UNATTRIBUTED = float(os.environ.get("EXPERIMENT_MAX_UNATTRIBUTED", "0.25"))
ARMS = {"greedy": 1, "vote3": 3, "vote5": 5}
_want = [a.strip() for a in os.environ.get("EXPERIMENT_ARMS", "").split(",") if a.strip()]
if _want:
    for _a in _want:
        if _a not in ARMS:
            raise SystemExit(f"unknown arm {_a!r}; have {sorted(ARMS)}")
    ARMS = {a: ARMS[a] for a in _want}

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
print(f"roster {len(roster)} | scoring {len(want)} lines | arms {sorted(ARMS)}",
      flush=True)

client = OpenAI(base_url=BASE_URL, api_key="local")
# Production width (w1). The width question is the factorial's; mixing them
# here would leave neither answerable.
params = LLMGenParams(max_tokens=12000, context_length=32768, temperature=0.0,
                      attribute_temperature=0.0, top_p=0.8,
                      reasoning_effort="none")

_env = os.environ.get("EXPERIMENT_ENV")
record = ExperimentRecord(
    "voting", REPO, MODEL, BASE_URL, GOLD_PATH,
    {"temperature": 0.0, "vote_temperature": 0.3, "max_tokens": 12000,
     "batch": BATCH, "arms": sorted(ARMS)},
    environment=json.loads(_env) if _env else None,
    notes="Majority voting on the production path, plus whether a split vote "
          "marks a wrong line well enough to route on. Compared against the "
          "w1/w4 disagreement trigger, which separated 78.2% from 51.6% at 40% "
          "coverage on grimgar03.")
record.enable_checkpoint(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"voting__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json.ckpt"))

windows = [list(range(s, min(s + BATCH, len(seg)))) for s in range(0, len(seg), BATCH)]
windows = [w for w in windows if any(norm(seg[i].get("text")) in want for i in w)]

summary = {}
for arm, votes in ARMS.items():
    started = time.time()
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
            out, conf = attribute_batch_voted(
                client, MODEL, frozen, params, roster, votes=votes,
                neighbor_contexts=contexts, source_text=src)
        except Exception as exc:
            print(f"  {arm} window {n}: {type(exc).__name__}", flush=True)
            for i in send:
                key = norm(seg[i].get("text"))
                if key in want and not record.done(arm, want[key]["id"]):
                    g = want[key]
                    record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(),
                               None, False, provenance=f"{arm}|batch_failed|conf=0.0")
            continue
        for position, i in enumerate(send):
            key = norm(seg[i].get("text"))
            if key not in want:
                continue
            g = want[key]
            speaker = (out[position] or {}).get("speaker") if position < len(out) else None
            share = conf[position] if position < len(conf) else 1.0
            record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(),
                       speaker, same(speaker, g["expected_speaker"]),
                       provenance=f"{arm}|conf={share:.3f}")
        if n % 25 == 0:
            print(f"  {arm} {n}/{len(windows)} ...", flush=True)
    rows = [r for r in record.rows if r["arm"] == arm]
    hit = sum(1 for r in rows if r["correct"])
    failed = sum(1 for r in rows if r["predicted"] is None)
    summary[arm] = (hit, len(rows), failed, time.time() - started)
    lo, hi = clopper_pearson(hit, max(len(rows), 1))
    print(f"  {arm:8} {hit}/{len(rows)} = {hit/max(len(rows),1)*100:5.1f}%  "
          f"[{lo:.1f}-{hi:.1f}]  {failed} unattributed  "
          f"{time.time()-started:.0f}s", flush=True)

for arm, (h, n, f, _) in summary.items():
    if n and f / n > MAX_UNATTRIBUTED:
        raise SystemExit(f"refusing to write: {arm} left {f}/{n} rows "
                         f"unattributed - environment failure, not a result")

base = "greedy" if "greedy" in summary else sorted(summary)[0]
ans = {a: {r["id"]: r["correct"] for r in record.rows if r["arm"] == a} for a in summary}
print(f"\n  paired against {base}   (cost is linear in votes; a gain under 3 "
      f"points does not earn 5x)")
for arm in summary:
    if arm == base:
        continue
    p, x, y, n = paired(ans[base], ans[arm])
    h0, n0, _, t0 = summary[base]
    h1, n1, _, t1 = summary[arm]
    print(f"    {arm:8} {(h1/n1 - h0/n0)*100:+6.1f} points  +{y:3}/-{x:3}  "
          f"p={p:.4g}   {t1/max(t0,1):.1f}x wall time")

# ---------------------------------------------------- is a split vote a signal?
print("\n  split votes as a routing trigger")
for arm in summary:
    if ARMS[arm] <= 1:
        continue
    rows = [r for r in record.rows if r["arm"] == arm]
    conf = {}
    for r in rows:
        m = re.search(r"conf=([0-9.]+)", r.get("candidate_provenance") or "")
        conf[r["id"]] = float(m.group(1)) if m else 1.0
    unan = [r for r in rows if conf[r["id"]] >= 0.999]
    split = [r for r in rows if conf[r["id"]] < 0.999]

    def acc(sub):
        k = sum(1 for r in sub if r["correct"])
        return k, len(sub), (k / len(sub) * 100 if sub else 0.0)

    ku, nu, au = acc(unan)
    ks, ns, as_ = acc(split)
    print(f"    {arm}: unanimous {ku}/{nu} = {au:.1f}%   "
          f"split {ks}/{ns} = {as_:.1f}%   "
          f"coverage {ns/max(len(rows),1)*100:.0f}%   separation {au-as_:+.1f}")
print("    Compare the w1/w4 disagreement trigger on grimgar03: 78.2% vs 51.6%")
print("    at 40% coverage, 26.6 points of separation. A trigger is only better")
print("    if it separates MORE at the SAME or lower coverage - a signal that")
print("    fires on everything routes everything and saves nothing.")

out = record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"voting__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json"),
    contract={"expected_arms": tuple(ARMS), "require_clean_tree": True})
print("wrote", out)
