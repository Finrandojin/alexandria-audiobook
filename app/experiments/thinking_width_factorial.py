"""Are the only two interventions that work ADDITIVE, or do they fix the same rows?

Two things have survived a production-path test in this whole investigation:

    thinking   +9.7 on qwen3-14b (p=1e-5), +5.0 on qwen3-32b (p=0.035),
               -1.0 on magistral-small (p=0.42) - real, but model-dependent
    w4 context +10.5 on grimgar03 (p=0.0002), -5.0 on mushoku16 (p=0.39) -
               real on one book, a regression on the other

Everything else - `because`, `scaffold`, closed-6 candidates, committed
history, roster warmup, adaptive width - was flat or harmful. So the question
of what to ship is really a question about these two, and nobody has run them
together. The 2x2 is the whole experiment:

              no thinking      thinking
    w1        the incumbent    +9.7 measured alone
    w4        +10.5 measured   never run

Readings, fixed before running:

  w4+thinking ~ w1 + both deltas   INDEPENDENT: they fix different rows, and
                                   the combination is the production target
  w4+thinking ~ the better single  REDUNDANT: both are proxies for the same
                                   missing evidence; ship the cheaper one and
                                   stop looking for more of the same
  w4+thinking < either alone       INTERFERENCE: worth knowing before anyone
                                   ships them together on the strength of two
                                   separate measurements
  interaction differs by book      the honest likely outcome given w4 already
                                   reverses sign between books; would mean the
                                   decision is per-book, not global

Additivity is judged on the interaction term, not on whether the top-right
cell is the biggest number. Two interventions that each add 10 points to a
57% baseline cannot both add 10 points to each other's output without
overlapping, and the test is whether the observed combination beats what
independence predicts.

Same production path as the w4 gate: `attribute_batch` with the shipping
prompt, batch 25, exhausted batches scored as failures rather than dropped.
Thinking is `reasoning_effort=None` against `"none"` - the same isolation
`because_production` used, changing the decoding and nothing about the prompt.
"""
import collections
import json, os, re, sys, time
from dataclasses import replace
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")
import openai
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

MODEL = os.environ.get("EXPERIMENT_MODEL", "qwen/qwen3-32b")
BOOK = os.environ.get("EXPERIMENT_BOOK", "grimgar03")
GOLD = os.environ.get("EXPERIMENT_GOLD",
                      "fixtures/attribution_gold_grimgar03_provisional.json")
GOLD_PATH = APP + GOLD
BASE_URL = os.environ.get("EXPERIMENT_BASE_URL", "http://127.0.0.1:8090/v1")
TAG = os.environ.get("EXPERIMENT_TAG", "cloud")
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
print(f"roster {len(roster)} | scoring {len(want)} lines | model {MODEL}", flush=True)

client = OpenAI(base_url=BASE_URL, api_key="local")
BASE = LLMGenParams(max_tokens=12000, context_length=32768, temperature=0.0,
                    attribute_temperature=0.0, top_p=0.8)


def neighbours(index, width):
    """Identical to the w4 gate's, so the width factor means the same thing."""
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


ARMS = {"w1-plain": (1, False), "w1-think": (1, True),
        "w4-plain": (4, False), "w4-think": (4, True)}
_want = [a.strip() for a in os.environ.get("EXPERIMENT_ARMS", "").split(",") if a.strip()]
for _a in _want:
    if _a not in ARMS:
        raise SystemExit(f"unknown arm {_a!r}; have {sorted(ARMS)}")
if _want:
    ARMS = {a: ARMS[a] for a in _want}

_env = os.environ.get("EXPERIMENT_ENV")
record = ExperimentRecord(
    "thinking_width_factorial", REPO, MODEL, BASE_URL, GOLD_PATH,
    {"temperature": 0.0, "attribute_temperature": 0.0, "max_tokens": 12000,
     "batch": BATCH, "arms": sorted(ARMS)},
    environment=json.loads(_env) if _env else None,
    notes="2x2 of the only two interventions with a production-path effect. "
          "Tests whether thinking and w4 context are additive or redundant; "
          "the interaction term is the result, not the top cell.")
record.enable_checkpoint(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"thinking_width_factorial__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json.ckpt"))

windows = [list(range(s, min(s + BATCH, len(seg)))) for s in range(0, len(seg), BATCH)]
windows = [w for w in windows if any(norm(seg[i].get("text")) in want for i in w)]

summary = {}
for arm, (width, thinks) in ARMS.items():
    started = time.time()
    params = replace(BASE, reasoning_effort=None if thinks else "none")
    for n, window in enumerate(windows, 1):
        send = [i for i in window if get_deterministic_named_entry(seg[i]) is None]
        if not send or not any(norm(seg[i].get("text")) in want for i in send):
            continue
        if all(record.done(arm, want[norm(seg[i].get("text"))]["id"])
               for i in send if norm(seg[i].get("text")) in want):
            continue
        frozen = [{"type": seg[i]["type"], "text": seg[i]["text"]} for i in send]
        contexts = [neighbours(i, width) for i in send]
        try:
            out = attribute_batch(client, MODEL, frozen, params, roster,
                                  neighbor_contexts=contexts, source_text=src)
        except Exception as exc:
            # Same policy as the w4 gate: an exhausted batch is a production
            # outcome (the pipeline emits UNKNOWN for it), not a row to drop.
            print(f"  {arm} window {n}: {type(exc).__name__}", flush=True)
            for i in send:
                key = norm(seg[i].get("text"))
                if key in want and not record.done(arm, want[key]["id"]):
                    g = want[key]
                    record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(),
                               None, False, provenance=f"{arm}|batch_failed")
            continue
        for position, i in enumerate(send):
            key = norm(seg[i].get("text"))
            if key not in want:
                continue
            g = want[key]
            speaker = (out[position] or {}).get("speaker") if position < len(out) else None
            record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(),
                       speaker, same(speaker, g["expected_speaker"]), provenance=arm)
        if n % 25 == 0:
            print(f"  {arm} {n}/{len(windows)} ...", flush=True)
    rows = [r for r in record.rows if r["arm"] == arm]
    hit = sum(1 for r in rows if r["correct"])
    failed = sum(1 for r in rows if r["predicted"] is None)
    summary[arm] = (hit, len(rows), failed, time.time() - started)
    print(f"  {arm:10} {hit}/{len(rows)} = {hit/max(len(rows),1)*100:5.1f}%   "
          f"{failed} unattributed   {time.time()-started:.0f}s", flush=True)

for arm, (h, n, f, _) in summary.items():
    if n and f / n > MAX_UNATTRIBUTED:
        raise SystemExit(f"refusing to write: {arm} left {f}/{n} rows "
                         f"unattributed - environment failure, not a result")

if len(summary) == 4:
    def rate(a):
        h, n, _, _ = summary[a]
        return h / n * 100

    print("\n  2x2 (accuracy)")
    print(f"    {'':10} {'no thinking':>13} {'thinking':>10}")
    print(f"    {'w1':10} {rate('w1-plain'):12.1f}% {rate('w1-think'):9.1f}%")
    print(f"    {'w4':10} {rate('w4-plain'):12.1f}% {rate('w4-think'):9.1f}%")
    d_think = rate("w1-think") - rate("w1-plain")
    d_width = rate("w4-plain") - rate("w1-plain")
    predicted = rate("w1-plain") + d_think + d_width
    observed = rate("w4-think")
    print(f"\n    thinking alone {d_think:+.1f}   w4 alone {d_width:+.1f}")
    print(f"    independence predicts {predicted:.1f}%, observed {observed:.1f}%, "
          f"interaction {observed - predicted:+.1f}")
    print("    " + ("ADDITIVE: they fix different rows" if observed - predicted > 2
                    else "REDUNDANT: they overlap; ship the cheaper one"
                    if observed - predicted < -2 else
                    "no detectable interaction at this sample size"))
    # The interaction estimate above is a difference of four proportions and
    # carries roughly twice the noise of any single arm; the paired tests are
    # what the conclusion should rest on.
    print("\n  paired transitions")
    ans = {a: {r["id"]: r["correct"] for r in record.rows if r["arm"] == a}
           for a in summary}
    for a, b in (("w1-plain", "w1-think"), ("w1-plain", "w4-plain"),
                 ("w4-plain", "w4-think"), ("w1-think", "w4-think"),
                 ("w1-plain", "w4-think")):
        p, x, y, n = paired(ans[a], ans[b])
        print(f"    {b:10} vs {a:10} +{y:3} / -{x:3}  p={p:.4g}")
    print("\n    Rows 2 and 3 are the same width change with thinking off then on;")
    print("    rows 1 and 4 are the same thinking change at w1 then w4. If those")
    print("    pairs disagree, the interaction is real whatever the cell means say.")

out = record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"thinking_width_factorial__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json"),
    contract={"expected_arms": tuple(ARMS), "require_clean_tree": True})
print("wrote", out)
