"""How much of the error is the roster, and how much is the model?

Two numbers have sat next to each other unreconciled since the start of this
investigation: the generated roster contains the right name for about 85% of
lines, and the model picks the right name about 30% of the time. That gap is
where "selection not recall" came from - the information is present and not
used. But the 85% has never been varied, so nobody knows what the model would
do with a better one.

Today put a number on the roster's failures for the first time: ten characters
across four books that `build_roster` never found, including HITOGAMI with 9
lines in mushoku16 and OUGI with 10 in owarimonogatari3. Those are not walk-on
parts, and every line they speak is unwinnable for a roster-constrained model.

ARMS, same production path, differing only in the roster handed to pass 2:

    generated   what build_roster produces - what ships today
    augmented   generated plus the names the judges found it missing, which is
                the fixture's `roster_additions`. Buildable in practice: a
                second pass over the text, or a better extractor.
    gold        exactly the speakers that appear in this book's gold, and
                nothing else. NOT shippable - it uses the answer key - and it
                exists to bound the whole roster dimension.
    inflated    gold plus twenty plausible decoys drawn from other books. Also
                not shippable; it separates "a longer list is harder" from
                "the right names being present is what matters".

READINGS, fixed before running:

  augmented ~ generated      the missing names are too rare to matter, and
                             roster recall is not where the loss is
  augmented >> generated     roster extraction is worth fixing, and the size of
                             the gain says how much
  gold >> augmented          even a perfect roster leaves most of the gap, so
                             the ceiling is selection, not recall - the
                             original hypothesis, now measured
  inflated ~ gold            list length is not the problem, presence is
  inflated << gold           the model is distracted by size, which would make
                             a SHORTER roster an intervention worth testing

The decoys are drawn from other books in the corpus rather than invented, so
they are real character names of the same kind - a made-up name might be
rejected on style alone and would flatter the inflated arm.
"""
import collections
import json, os, random, re, sys, time
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")
from openai import OpenAI
from experiments.manifest import ExperimentRecord
from experiments.scoring import alias_groups, same_speaker
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
BOOK = os.environ.get("EXPERIMENT_BOOK", "grimgar03")
GOLD = os.environ.get("EXPERIMENT_GOLD", f"fixtures/attribution_gold_{BOOK}.json")
GOLD_PATH = APP + GOLD
BASE_URL = os.environ.get("EXPERIMENT_BASE_URL", "http://127.0.0.1:8090/v1")
TAG = os.environ.get("EXPERIMENT_TAG", "local-llamacpp")
BATCH = 25
DECOYS = int(os.environ.get("EXPERIMENT_DECOYS", "20"))
SEED = int(os.environ.get("EXPERIMENT_SEED", "20260728"))

gold = json.load(open(GOLD_PATH))
src = open(M + f"inputs/{BOOK}.txt", encoding="utf-8").read()
cp = json.load(open(M + INPUT_RUN + f"/{BOOK}/result.json.threepass_checkpoint.json"))
seg = cp["segmented"]
GROUPS = alias_groups(gold)
SPECIAL = {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE"}


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


_occ = collections.Counter(norm(e.get("text")) for e in seg)
want = {norm(g["line"]): g for g in gold["entries"]
        if _occ[norm(g["line"])] == 1
        and g["expected_speaker"].upper() not in SPECIAL}

generated = [r.upper() for r in
             build_roster([e for e in (cp.get("named") or []) if e], src)]
additions = [n.upper() for n in
             gold.get("roster_additions", {}).get("names", [])]
truth = sorted({g["expected_speaker"].upper() for g in want.values()})

# Decoys are real names from the other books, so they are the same KIND of
# token as the true entries. Invented names could be dismissed on style and
# would make the inflated arm look better than it is.
others = set()
for other in ("grimgar03", "mushoku16", "index18", "owarimonogatari3"):
    if other == BOOK:
        continue
    p = APP + f"fixtures/attribution_gold_{other}.json"
    if not os.path.exists(p):
        continue
    for e in json.load(open(p))["entries"]:
        n = e["expected_speaker"].upper()
        if n not in SPECIAL:
            others.add(n)
others -= set(truth) | set(generated)
rng = random.Random(SEED)
decoys = rng.sample(sorted(others), min(DECOYS, len(others)))

ROSTERS = {
    "generated": generated,
    "augmented": sorted(set(generated) | set(additions)),
    "gold": truth,
    "inflated": sorted(set(truth) | set(decoys)),
}
print(f"{BOOK}: {len(want)} scoreable lines")
for name, r in ROSTERS.items():
    covered = sum(1 for g in want.values()
                  if any(same_speaker(g["expected_speaker"], n, GROUPS) for n in r))
    print(f"  {name:10} {len(r):3} names, covers {covered}/{len(want)} "
          f"= {covered/len(want)*100:.1f}% of scored lines", flush=True)

client = OpenAI(base_url=BASE_URL, api_key="local")
params = LLMGenParams(max_tokens=12000, context_length=32768, temperature=0.0,
                      attribute_temperature=0.0, top_p=0.8,
                      reasoning_effort="none")

_env = os.environ.get("EXPERIMENT_ENV")
record = ExperimentRecord(
    "roster_quality", REPO, MODEL, BASE_URL, GOLD_PATH,
    {"temperature": 0.0, "max_tokens": 12000, "batch": BATCH, "decoys": DECOYS},
    environment=json.loads(_env) if _env else None,
    notes="Varies only the roster handed to pass 2. The generated roster "
          "contains the right name ~85% of the time and the model picks it "
          "~30%; this measures what a better roster would actually buy, and "
          "bounds the whole dimension with a gold roster that is not "
          "shippable.")
record.enable_checkpoint(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"roster_quality__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json.ckpt"))

windows = [list(range(s, min(s + BATCH, len(seg)))) for s in range(0, len(seg), BATCH)]
windows = [w for w in windows if any(norm(seg[i].get("text")) in want for i in w)]

summary = {}
for arm, roster in ROSTERS.items():
    started = time.time()
    for n, win in enumerate(windows, 1):
        send = [i for i in win if get_deterministic_named_entry(seg[i]) is None]
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
            out = attribute_batch(client, MODEL, frozen, params, roster,
                                  neighbor_contexts=contexts, source_text=src)
        except Exception as exc:
            print(f"  {arm} window {n}: {type(exc).__name__}", flush=True)
            for i in send:
                k = norm(seg[i].get("text"))
                if k in want and not record.done(arm, want[k]["id"]):
                    g = want[k]
                    record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(),
                               None, False, provenance=f"{arm}|batch_failed")
            continue
        for off, i in enumerate(send):
            k = norm(seg[i].get("text"))
            if k not in want:
                continue
            g = want[k]
            sp = (out[off] or {}).get("speaker") if off < len(out) else None
            record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(), sp,
                       same_speaker(g["expected_speaker"], sp, GROUPS),
                       candidates=roster, provenance=arm)
        if n % 25 == 0:
            print(f"  {arm} {n}/{len(windows)} ...", flush=True)
    rows = [r for r in record.rows if r["arm"] == arm]
    hit = sum(1 for r in rows if r["correct"])
    summary[arm] = (hit, len(rows), time.time() - started)
    lo, hi = clopper_pearson(hit, max(len(rows), 1))
    print(f"  {arm:10} {hit}/{len(rows)} = {hit/max(len(rows),1)*100:5.1f}%  "
          f"[{lo:.1f}-{hi:.1f}]  {time.time()-started:.0f}s", flush=True)

ans = {a: {r["id"]: r["correct"] for r in record.rows if r["arm"] == a}
       for a in summary}
print("\n  paired against the shipped roster")
for arm in ("augmented", "gold", "inflated"):
    if arm not in ans:
        continue
    p, x, y, n = paired(ans["generated"], ans[arm])
    h0, n0 = summary["generated"][0], summary["generated"][1]
    h1, n1 = summary[arm][0], summary[arm][1]
    print(f"    {arm:10} {(h1/n1 - h0/n0)*100:+6.1f} points  +{y:3}/-{x:3}  p={p:.4g}")
print("\n  A gold roster that still leaves most of the gap means the ceiling is")
print("  selection rather than recall, and roster extraction is not where the")
print("  remaining accuracy lives.")

out = record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"roster_quality__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json"),
    contract={"expected_arms": tuple(ROSTERS)})
print("wrote", out)
