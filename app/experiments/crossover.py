"""Does the frozen segmentation explain why the decomposition and the pipeline
disagree about which model is better?

Brief §6.3: on grimgar03, the closed-set decomposition ranks qwen3-14b above
gemma (60.8% vs 57.2%) while whole-pipeline output ranks gemma above qwen3-14b
(59.4% vs 53.9%) - same book, same fixture. The decomposition freezes
segmentation from the 9B run and varies only pass 2; end-to-end varies pass 1
and pass 2 together. Nothing in either result separates the two.

This crosses them. Segmentation comes from each model's own end-to-end run in
the overnight 2x2, attribution is then run by each model over each
segmentation:

    seg=gemma  attr=gemma      seg=gemma  attr=qwen
    seg=qwen   attr=gemma      seg=qwen   attr=qwen

The diagonal reproduces the pipeline; the off-diagonal is what neither existing
instrument measures. Reading the four cells:

  attr main effect dominates   - pass 2 is what matters, the decomposition was
                                 measuring the right thing and §6.1 stands
  seg main effect dominates    - the ranking is a pass-1 result wearing a pass-2
                                 costume, and every frozen-input row in §4 is
                                 conditional on a segmentation nothing ships
  interaction dominates        - a model attributes best over its own
                                 segmentation, so component scores cannot be
                                 mixed across models at all

Only gold lines whose text appears exactly once in BOTH segmentations are
scored, so every cell answers on an identical row set. Lines that one
segmentation split, merged or dropped are reported separately rather than
silently scored against a different denominator - that asymmetry is itself a
pass-1 quality signal.
"""
import collections
import json, os, re, sys, time
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")
import openai
from openai import OpenAI
from experiments.manifest import ExperimentRecord
from three_pass_generate import build_roster

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
APP = REPO + "/app/"
M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
OVERNIGHT = REPO + "/ab_test_runtime/results/overnight_20260726-185022/"

BOOK = os.environ.get("EXPERIMENT_BOOK", "grimgar03")
GOLD = os.environ.get("EXPERIMENT_GOLD",
                      "fixtures/attribution_gold_grimgar03_provisional.json")
GOLD_PATH = APP + GOLD
BASE_URL = os.environ.get("EXPERIMENT_BASE_URL", "http://localhost:1234/v1")
TAG = os.environ.get("EXPERIMENT_TAG", "local")

# The two models whose end-to-end runs produced the segmentations being crossed.
# Keys are the checkpoint directory names; values are the model ids to call.
GEMMA = "gemma-4-e4b-uncensored-hauhaucs-aggressive"
QWEN = "qwen/qwen3-14b"
SEG_SOURCES = {"gemma": GEMMA, "qwen": QWEN}
ATTR_MODELS = {"gemma": GEMMA, "qwen": QWEN}

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
    seg = cp["segmented"]
    segs[key] = seg
    rosters[key] = [r.upper() for r in
                    build_roster([e for e in (cp.get("named") or []) if e], src)]

# A gold line is scoreable only where BOTH segmentations contain its text
# exactly once. Anything else cannot be aligned to one position in both, and
# scoring it would compare different rows across cells.
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

print(f"{len(shared)} of {len(gold['entries'])} gold lines align uniquely in "
      f"both segmentations", flush=True)
for reason, n in sorted(dropped.items()):
    print(f"   excluded {reason}: {n}", flush=True)
for k in segs:
    print(f"   {k}: {len(segs[k])} segments, roster {len(rosters[k])}",
          flush=True)

client = OpenAI(base_url=BASE_URL, api_key="local")
SYSTEM = ("You identify who speaks a line of dialogue in a novel. Answer with "
          "the speaker's name in CAPITALS and nothing else. If the passage "
          "does not determine it, answer UNKNOWN.")


def ask(model, seg, index, line, choices):
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
                temperature=0.0, max_tokens=24,
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
    {"temperature": 0.0, "max_tokens": 24, "reasoning_effort": "none"},
    environment=json.loads(_env) if _env else None,
    notes="Segmentation source x attribution model, 2x2 on one aligned row "
          "set. Separates the pass-1 effect from the pass-2 effect behind the "
          "decomposition/pipeline disagreement in brief section 6.3.")
record.meta["segment_counts"] = {k: len(v) for k, v in segs.items()}
record.meta["roster_sizes"] = {k: len(v) for k, v in rosters.items()}
record.meta["excluded_rows"] = dict(dropped)

cells = {}
# Attribution model is the OUTER loop on purpose: each model is then loaded once
# and answers over both segmentations, rather than alternating and forcing a
# model swap per cell. A 9 GB reload between every cell would dominate the run
# and, on a machine where VRAM is tight, is a chance to fail.
for attr_key, attr_model in ATTR_MODELS.items():
    for seg_key in SEG_SOURCES:
        seg, roster = segs[seg_key], rosters[seg_key]
        arm = f"seg={seg_key},attr={attr_key}"
        started = time.time()
        for g in shared:
            i = pos[seg_key][norm(g["line"])]
            got, prompt, raw, retries = ask(attr_model, seg, i, g["line"],
                                            roster)
            ok = same(got, g["expected_speaker"])
            record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(),
                       got, ok, candidates=roster, provenance=seg_key,
                       prompt=prompt, raw=raw, retries=retries)
        rows = [r for r in record.rows if r["arm"] == arm]
        hit = sum(1 for r in rows if r["correct"])
        cells[arm] = hit / max(len(rows), 1) * 100
        print(f"{arm:26} {hit}/{len(rows)} = {cells[arm]:5.1f}%   "
              f"{time.time() - started:.0f}s", flush=True)

print("\n2x2 (rows = segmentation source, cols = attribution model):")
print(f"  {'':12} {'attr=gemma':>12} {'attr=qwen':>12}")
for seg_key in SEG_SOURCES:
    a = cells[f"seg={seg_key},attr=gemma"]
    b = cells[f"seg={seg_key},attr=qwen"]
    print(f"  seg={seg_key:<8} {a:11.1f}% {b:11.1f}%")
seg_effect = ((cells["seg=gemma,attr=gemma"] + cells["seg=gemma,attr=qwen"]) -
              (cells["seg=qwen,attr=gemma"] + cells["seg=qwen,attr=qwen"])) / 2
attr_effect = ((cells["seg=gemma,attr=gemma"] + cells["seg=qwen,attr=gemma"]) -
               (cells["seg=gemma,attr=qwen"] + cells["seg=qwen,attr=qwen"])) / 2
inter = (cells["seg=gemma,attr=gemma"] - cells["seg=gemma,attr=qwen"] -
         cells["seg=qwen,attr=gemma"] + cells["seg=qwen,attr=qwen"]) / 2
print(f"\n  segmentation main effect (gemma - qwen): {seg_effect:+.1f} pt")
print(f"  attribution  main effect (gemma - qwen): {attr_effect:+.1f} pt")
print(f"  interaction                            : {inter:+.1f} pt")
print("  Main effects are descriptive; paired tests belong on the rows.")

out = record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"crossover__{BOOK}__{TAG}.json"),
    contract={"expected_arms": tuple(cells),
              "expected_ids": {g["id"] for g in shared},
              "require_clean_tree": True})
print("wrote", out)
