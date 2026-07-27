"""Segmentation x attribution x temperature, with repeats. Pre-registered.

WHAT THIS REPLACES
------------------
Brief section 6.3 reports the closed-set decomposition and the whole pipeline
ranking qwen3-14b and gemma in opposite orders on grimgar03 - 60.8% vs 57.2%
one way, 59.4% vs 53.9% the other - and attributes it to the decomposition
freezing segmentation. That attribution was asserted, not tested, and the
comparison has THREE unseparated differences, not one:

  1. frozen 9B segmentation      vs each model's own segmentation
  2. temperature 0.0             vs temperature 0.6
  3. one run per cell            vs one run per cell

(3) is now measured and is nearly decisive on its own: repeating the pipeline
with the same model on the same book moved 17.9% of per-line answers and gives
a single-run 95% band of about +/-4.9 points. The 6.3 gap is 5.5 points. So the
"instruments disagree" claim may be describing sampling noise.

The first version of this harness ran at temperature 0.0 only, so it could not
have addressed (2) at all - it would have separated segmentation from
attribution in a decoding configuration the pipeline never uses.

DESIGN
------
Fully crossed, one aligned row set, repeats at every cell:

    segmentation source  {gemma, qwen}      - each model's own pass-1 output
  x attribution model    {gemma, qwen}
  x temperature          {0.0, 0.6}         - 0.0 is what the decomposition
                                              used; 0.6 is what ships
  x repeats              {2 at 0.0, 3 at 0.6}

Repeats at 0.0 are not wasted. Temperature-0 determinism is a prior belief in
this project, not a measured fact for THIS harness, and the project has already
been wrong once about a "noise floor" that turned out to be GPU contention. Two
identical runs confirm it; a disagreement invalidates every single-run
decomposition result in the ledger and is worth knowing immediately. Three
repeats at 0.6 give a within-cell SD with 2 degrees of freedom - thin, but
enough to say whether the between-cell effects clear the noise.

Only gold lines whose text appears exactly once in BOTH segmentations are
scored, so every cell answers on an identical row set. Rows excluded because a
segmentation split, merged or dropped them are counted and reported: that
asymmetry is itself a pass-1 quality signal and must not be silently absorbed
into a denominator.

PRE-REGISTERED ANALYSIS - fixed before any cell was run
-------------------------------------------------------
Estimated separately at each temperature, never pooled across temperatures:

  seg main effect   = mean(seg=gemma cells) - mean(seg=qwen cells)
  attr main effect  = mean(attr=gemma cells) - mean(attr=qwen cells)
  interaction       = (gg - gq - qg + qq) / 2

Decision rules, stated in advance so the result cannot be read to taste:

  * An effect counts as resolved only if its magnitude exceeds 2 x the
    within-cell SD measured at that temperature. Below that it is reported as
    unresolved, NOT as absence of an effect.
  * If the attribution main effect dominates at both temperatures, the
    decomposition was measuring the right thing and section 6.1 stands.
  * If the segmentation main effect dominates, every frozen-input row in the
    ledger is conditional on a segmentation nothing ships.
  * If the interaction dominates, component scores cannot be mixed across
    models at all.
  * If effects at 0.0 and 0.6 disagree in sign or ordering, temperature is a
    confound in section 6.3 independent of segmentation, and no temperature-0
    harness can be used to predict pipeline behaviour.
  * If within-cell SD at 0.6 is large enough that no effect clears 2 SD, the
    honest conclusion is that this design cannot resolve the question at n=3,
    and the answer is more repeats - not a narrative.

Paired McNemar tests are computed per contrast on the row level for the
temperature-0 cells, where pairing is exact. At 0.6 the repeats are averaged
first, because pairing across a stochastic repeat is not meaningful.
"""
import collections
import json, os, re, statistics, sys, time
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")
import openai
from openai import OpenAI
from experiments.manifest import ExperimentRecord
from three_pass_generate import build_roster
from lmstudio_settings import ensure_ideal_settings

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
APP = REPO + "/app/"
M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
OVERNIGHT = REPO + "/ab_test_runtime/results/overnight_20260726-185022/"

BOOK = os.environ.get("EXPERIMENT_BOOK", "grimgar03")
GOLD = os.environ.get("EXPERIMENT_GOLD",
                      "fixtures/attribution_gold_grimgar03_provisional.json")
GOLD_PATH = APP + GOLD
BASE_URL = os.environ.get("EXPERIMENT_BASE_URL", "http://localhost:1234/v1")
LLM_MODE = os.environ.get("EXPERIMENT_LLM_MODE", "local")
TAG = os.environ.get("EXPERIMENT_TAG",
                     "local" if "localhost" in BASE_URL or "127.0.0.1"
                     in BASE_URL else "remote")

GEMMA = "gemma-4-e4b-uncensored-hauhaucs-aggressive"
QWEN = "qwen/qwen3-14b"
SEG_SOURCES = {"gemma": GEMMA, "qwen": QWEN}
ATTR_MODELS = {"gemma": GEMMA, "qwen": QWEN}
# 0.0 is the decomposition's setting, 0.6 is the shipped pipeline's.
REPEATS = {0.0: 2, 0.6: 3}

RETRYABLE = (openai.APIConnectionError, openai.APITimeoutError,
             openai.InternalServerError, openai.RateLimitError,
             openai.NotFoundError)
MAX_ATTEMPTS = 6

gold = json.load(open(GOLD_PATH))
src = open(M + f"inputs/{BOOK}.txt", encoding="utf-8").read()
AL = [{n.upper() for n in g} for g in gold.get("aliases", [])]


def same(a, b):
    a, b = (a or "").upper(), (b or "").upper()
    return a == b or any(a in g and b in g for g in AL)


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


segs, rosters = {}, {}
for key, model in SEG_SOURCES.items():
    cp = json.load(open(
        f"{OVERNIGHT}{model}/{BOOK}/result.json.threepass_checkpoint.json"))
    segs[key] = cp["segmented"]
    rosters[key] = [r.upper() for r in
                    build_roster([e for e in (cp.get("named") or []) if e], src)]

occ = {k: collections.Counter(norm(e.get("text")) for e in s)
       for k, s in segs.items()}
pos = {k: {norm(e.get("text")): i for i, e in enumerate(s)}
       for k, s in segs.items()}

shared, dropped = [], collections.Counter()
for g in gold["entries"]:
    key = norm(g["line"])
    counts = {k: occ[k].get(key, 0) for k in segs}
    if all(c == 1 for c in counts.values()):
        shared.append(g)
    else:
        for k, c in counts.items():
            if c != 1:
                dropped[f"{k}:{'absent' if c == 0 else 'repeated'}"] += 1

total_calls = len(shared) * len(SEG_SOURCES) * len(ATTR_MODELS) * sum(REPEATS.values())
print(f"{len(shared)} of {len(gold['entries'])} gold lines align uniquely in "
      f"both segmentations", flush=True)
for reason, n in sorted(dropped.items()):
    print(f"   excluded {reason}: {n}", flush=True)
for k in segs:
    print(f"   {k}: {len(segs[k])} segments, roster {len(rosters[k])}", flush=True)
print(f"   {total_calls} calls total "
      f"({len(SEG_SOURCES) * len(ATTR_MODELS)} cells x "
      f"{sum(REPEATS.values())} runs x {len(shared)} lines)", flush=True)

client = OpenAI(base_url=BASE_URL, api_key="local")
SYSTEM = ("You identify who speaks a line of dialogue in a novel. Answer with "
          "the speaker's name in CAPITALS and nothing else. If the passage "
          "does not determine it, answer UNKNOWN.")


def ask(model, seg, index, line, choices, temperature):
    before = " ".join((seg[j].get("text") or "")
                      for j in range(max(0, index - 4), index))
    after = " ".join((seg[j].get("text") or "")
                     for j in range(index + 1, min(len(seg), index + 4)))
    user = (f"PASSAGE BEFORE:\n{before}\n\nLINE:\n{line}\n\n"
            f"PASSAGE AFTER:\n{after}\n"
            f"\nThe speaker is one of: {', '.join(choices + ['UNKNOWN'])}\n\n"
            f"Who speaks the LINE?")
    last = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": user}],
                temperature=temperature, max_tokens=24,
                extra_body={"reasoning_effort": "none"})
            raw = (r.choices[0].message.content or "")
            return raw.strip().upper().strip(".'\" "), user, raw, attempt
        except RETRYABLE as exc:
            last = exc
            if attempt == MAX_ATTEMPTS - 1:
                break
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"endpoint failed {MAX_ATTEMPTS} attempts against "
                       f"{BASE_URL}: {type(last).__name__}: {last}") from last


_env = os.environ.get("EXPERIMENT_ENV")
record = ExperimentRecord(
    "segmentation_crossover", REPO, "|".join(sorted(set(ATTR_MODELS.values()))),
    BASE_URL, GOLD_PATH,
    {"temperatures": sorted(REPEATS), "repeats": {str(k): v for k, v in REPEATS.items()},
     "max_tokens": 24, "reasoning_effort": "none"},
    environment=json.loads(_env) if _env else None,
    notes="Segmentation source x attribution model x temperature, with repeats. "
          "Separates pass-1 from pass-2 from decoding temperature behind the "
          "decomposition/pipeline disagreement in brief section 6.3, which has "
          "three unseparated differences and a run-to-run noise band of about "
          "4.9 points against a 5.5 point effect. Analysis pre-registered in "
          "the module docstring before any cell was run.")
record.meta["segment_counts"] = {k: len(v) for k, v in segs.items()}
record.meta["roster_sizes"] = {k: len(v) for k, v in rosters.items()}
record.meta["excluded_rows"] = dict(dropped)
record.meta["preregistered"] = True

cells = collections.defaultdict(list)
arms = []
# Attribution model is the OUTER loop: each is loaded once and answers every
# cell it appears in, instead of swapping a 9 GB model per cell.
for attr_key, attr_model in ATTR_MODELS.items():
    _, status, message = ensure_ideal_settings(LLM_MODE, BASE_URL, attr_model)
    print(f"\n[{attr_key}] {message}", flush=True)
    if not status.get("loaded"):
        raise SystemExit(f"{attr_model} would not load; refusing to run a "
                         f"partial crossover")
    for temperature, n_rep in sorted(REPEATS.items()):
        for rep in range(1, n_rep + 1):
            for seg_key in SEG_SOURCES:
                seg, roster = segs[seg_key], rosters[seg_key]
                arm = f"seg={seg_key},attr={attr_key},t={temperature},rep={rep}"
                arms.append(arm)
                started = time.time()
                for g in shared:
                    i = pos[seg_key][norm(g["line"])]
                    got, prompt, raw, retries = ask(attr_model, seg, i,
                                                    g["line"], roster,
                                                    temperature)
                    record.add(arm, g["id"], g["line"],
                               g["expected_speaker"].upper(), got,
                               same(got, g["expected_speaker"]),
                               candidates=roster, provenance=seg_key,
                               prompt=prompt, raw=raw, retries=retries)
                rows = [r for r in record.rows if r["arm"] == arm]
                hit = sum(1 for r in rows if r["correct"])
                acc = hit / max(len(rows), 1) * 100
                cells[(seg_key, attr_key, temperature)].append(acc)
                print(f"  {arm:44} {hit}/{len(rows)} = {acc:5.1f}%   "
                      f"{time.time() - started:.0f}s", flush=True)

print("\n" + "=" * 68)
for temperature in sorted(REPEATS):
    print(f"\ntemperature {temperature} "
          f"({REPEATS[temperature]} repeats per cell)")
    print(f"  {'':12} {'attr=gemma':>18} {'attr=qwen':>18}")
    mean = {}
    for seg_key in SEG_SOURCES:
        row = []
        for attr_key in ATTR_MODELS:
            vals = cells[(seg_key, attr_key, temperature)]
            mean[(seg_key, attr_key)] = statistics.mean(vals)
            spread = (f"+/-{statistics.stdev(vals):.1f}" if len(vals) > 1
                      else "")
            row.append(f"{statistics.mean(vals):6.1f}% {spread:>6}")
        print(f"  seg={seg_key:<8} {row[0]:>18} {row[1]:>18}")
    within = [statistics.stdev(v) for v in
              (cells[(s, a, temperature)] for s in SEG_SOURCES for a in ATTR_MODELS)
              if len(v) > 1]
    sd = statistics.mean(within) if within else 0.0
    seg_e = ((mean[("gemma", "gemma")] + mean[("gemma", "qwen")]) -
             (mean[("qwen", "gemma")] + mean[("qwen", "qwen")])) / 2
    attr_e = ((mean[("gemma", "gemma")] + mean[("qwen", "gemma")]) -
              (mean[("gemma", "qwen")] + mean[("qwen", "qwen")])) / 2
    inter = (mean[("gemma", "gemma")] - mean[("gemma", "qwen")] -
             mean[("qwen", "gemma")] + mean[("qwen", "qwen")]) / 2
    print(f"    within-cell SD: {sd:.2f} pt   resolution threshold (2 SD): "
          f"{2 * sd:.2f} pt")
    for label, val in (("segmentation", seg_e), ("attribution", attr_e),
                       ("interaction", inter)):
        verdict = "RESOLVED" if abs(val) > 2 * sd else "unresolved"
        print(f"    {label:13} (gemma - qwen): {val:+6.2f} pt   {verdict}")

print("\nPer the pre-registered rules, 'unresolved' means this design could not")
print("separate the effect at this n - it is not evidence the effect is absent.")

out = record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"crossover__{BOOK}__{TAG}.json"),
    contract={"expected_arms": tuple(arms),
              "expected_ids": {g["id"] for g in shared},
              "require_clean_tree": True})
print("wrote", out)
