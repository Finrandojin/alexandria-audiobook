"""Does narrowing the candidates to who is PRESENT in the scene help?

Two findings point here from opposite directions.

`roster_quality` measured something that undercuts the framing this
investigation has used for days: on grimgar03 the shipped roster already
covers 98.7% of scored lines. Roster RECALL is not the bottleneck. The right
name is nearly always on the list and the model picks a different one.

`closed_set` tried narrowing that list and it backfired - closed-6 cost
qwen3-14b 9.4 points on mushoku16 - but the candidates there were chosen by
the harness from a global roster, so a wrong shortlist actively removed the
answer. That is a different intervention from narrowing by SCENE, where the
constraint comes from the text rather than from a guess.

The scene is a natural unit for this. `joint_scene` established that grimgar03
splits into 333 runs of consecutive spoken segments bounded by narration,
median length 2. Inside one such run the cast is small and the surrounding
narration usually names it. Twenty-one characters is the book; four is the
room.

ARMS:

    full        the whole roster, as production does today
    scene       only characters named in the narration bounding this scene,
                plus anyone named in the scene's own lines, plus the narrator
    scene+2     the same, plus the two most frequent speakers in the book, as
                insurance against a scene whose cast is never named

READINGS, fixed before running:

  scene >> full          narrowing by presence is the intervention closed-6
                         was reaching for and got wrong
  scene << full          the scene cast misses too often, and the failure mode
                         is the closed-6 one: the answer removed from the list
  scene+2 >> scene       the misses are concentrated in scenes with no named
                         cast, and a cheap fallback fixes it
  all three equal        candidate-set size is not what limits selection, which
                         would agree with roster_quality's coverage figure and
                         close this line for good

COVERAGE IS REPORTED BEFORE ACCURACY. If the scene cast contains the true
speaker for only 70% of rows, a 30-point ceiling is built into the arm and
its accuracy is uninterpretable without that number. closed-6 was reported
without it for days.
"""
import collections
import json, os, re, sys, time
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")
from openai import OpenAI
from experiments.manifest import ExperimentRecord
from experiments.scoring import alias_groups, same_speaker
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
GOLD = os.environ.get("EXPERIMENT_GOLD", f"fixtures/attribution_gold_{BOOK}.json")
GOLD_PATH = APP + GOLD
BASE_URL = os.environ.get("EXPERIMENT_BASE_URL", "http://127.0.0.1:8090/v1")
TAG = os.environ.get("EXPERIMENT_TAG", "local-llamacpp")
BATCH = 25
LOOK = int(os.environ.get("EXPERIMENT_LOOK", "3"))   # narration segments each side

gold = json.load(open(GOLD_PATH))
src = open(M + f"inputs/{BOOK}.txt", encoding="utf-8").read()
cp = json.load(open(M + INPUT_RUN + f"/{BOOK}/result.json.threepass_checkpoint.json"))
seg = cp["segmented"]
GROUPS = alias_groups(gold)
SPECIAL = {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE"}
FULL = [r.upper() for r in
        build_roster([e for e in (cp.get("named") or []) if e], src)]
FULL = sorted(set(FULL) | {n.upper() for n in
                           gold.get("roster_additions", {}).get("names", [])})


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


_occ = collections.Counter(norm(e.get("text")) for e in seg)
want = {norm(g["line"]): g for g in gold["entries"]
        if _occ[norm(g["line"])] == 1
        and g["expected_speaker"].upper() not in SPECIAL}
common = [n for n, _ in collections.Counter(
    g["expected_speaker"].upper() for g in want.values()).most_common(2)]

PAT = {n: re.compile(r"\b" + re.escape(n.split()[0]) + r"\b", re.I) for n in FULL}


def scene_bounds(index):
    """The run of consecutive spoken segments containing `index`."""
    lo = index
    while lo > 0 and seg[lo - 1].get("type") != "NARRATOR":
        lo -= 1
    hi = index
    while hi + 1 < len(seg) and seg[hi + 1].get("type") != "NARRATOR":
        hi += 1
    return lo, hi


def scene_cast(index):
    """Characters named in this scene or in the narration bounding it."""
    lo, hi = scene_bounds(index)
    text = " ".join((seg[j].get("text") or "") for j in range(lo, hi + 1))
    seen = 0
    for j in range(lo - 1, -1, -1):
        text += " " + (seg[j].get("text") or "")
        if seg[j].get("type") == "NARRATOR":
            seen += 1
            if seen >= LOOK:
                break
    seen = 0
    for j in range(hi + 1, len(seg)):
        text += " " + (seg[j].get("text") or "")
        if seg[j].get("type") == "NARRATOR":
            seen += 1
            if seen >= LOOK:
                break
    return sorted({n for n, p in PAT.items() if p.search(text)})


CASTS = {i: scene_cast(i) for i, e in enumerate(seg)
         if norm(e.get("text")) in want}

ARMS = ("full", "scene", "scene+2")
print(f"{BOOK}: {len(want)} scoreable lines, roster {len(FULL)}", flush=True)
sizes, cover = [], collections.Counter()
for i, cast in CASTS.items():
    g = want.get(norm(seg[i].get("text")))
    if not g:
        continue
    sizes.append(len(cast))
    cover["scene"] += any(same_speaker(g["expected_speaker"], n, GROUPS) for n in cast)
    plus = sorted(set(cast) | set(common))
    cover["scene+2"] += any(same_speaker(g["expected_speaker"], n, GROUPS) for n in plus)
    cover["full"] += any(same_speaker(g["expected_speaker"], n, GROUPS) for n in FULL)
n = len(sizes)
print(f"  scene cast: median {sorted(sizes)[n//2]} names (roster is {len(FULL)})")
for arm in ARMS:
    print(f"  {arm:8} contains the true speaker for {cover[arm]}/{n} "
          f"= {cover[arm]/n*100:.1f}% of rows", flush=True)
print("  A ceiling below 100% here is built into the arm; read its accuracy "
      "against\n  this line, not against the full roster's.", flush=True)

client = OpenAI(base_url=BASE_URL, api_key="local")
params = LLMGenParams(max_tokens=12000, context_length=32768, temperature=0.0,
                      attribute_temperature=0.0, top_p=0.8,
                      reasoning_effort="none")
_env = os.environ.get("EXPERIMENT_ENV")
record = ExperimentRecord(
    "scene_cast", REPO, MODEL, BASE_URL, GOLD_PATH,
    {"temperature": 0.0, "batch": BATCH, "look": LOOK},
    environment=json.loads(_env) if _env else None,
    notes="Narrowing candidates to the scene's present cast rather than the "
          "book's roster. roster_quality showed the full roster already covers "
          "98.7% of grimgar03's scored lines, so recall is not the limit; this "
          "tests whether a shorter, text-derived list improves selection.")
record.enable_checkpoint(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"scene_cast__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json.ckpt"))

windows = [list(range(s, min(s + BATCH, len(seg)))) for s in range(0, len(seg), BATCH)]
windows = [w for w in windows if any(norm(seg[i].get("text")) in want for i in w)]
summary = {}
for arm in ARMS:
    started = time.time()
    for k, win in enumerate(windows, 1):
        send = [i for i in win if get_deterministic_named_entry(seg[i]) is None]
        if not send or not any(norm(seg[i].get("text")) in want for i in send):
            continue
        if all(record.done(arm, want[norm(seg[i].get("text"))]["id"])
               for i in send if norm(seg[i].get("text")) in want):
            continue
        if arm == "full":
            roster = FULL
        else:
            cast = set()
            for i in send:
                cast |= set(CASTS.get(i) or scene_cast(i))
            if arm == "scene+2":
                cast |= set(common)
            roster = sorted(cast) or FULL
        frozen = [{"type": seg[i]["type"], "text": seg[i]["text"]} for i in send]
        contexts = [{"previous_context": seg[i - 1] if i else None,
                     "next_context": seg[i + 1] if i + 1 < len(seg) else None}
                    for i in send]
        try:
            out = attribute_batch(client, MODEL, frozen, params, roster,
                                  neighbor_contexts=contexts, source_text=src)
        except Exception as exc:
            print(f"  {arm} window {k}: {type(exc).__name__}", flush=True)
            for i in send:
                key = norm(seg[i].get("text"))
                if key in want and not record.done(arm, want[key]["id"]):
                    g = want[key]
                    record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(),
                               None, False, provenance=f"{arm}|batch_failed")
            continue
        for off, i in enumerate(send):
            key = norm(seg[i].get("text"))
            if key not in want:
                continue
            g = want[key]
            sp = (out[off] or {}).get("speaker") if off < len(out) else None
            record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(), sp,
                       same_speaker(g["expected_speaker"], sp, GROUPS),
                       candidates=roster, provenance=f"{arm}|cast={len(roster)}")
        if k % 25 == 0:
            print(f"  {arm} {k}/{len(windows)} ...", flush=True)
    rows = [r for r in record.rows if r["arm"] == arm]
    hit = sum(1 for r in rows if r["correct"])
    summary[arm] = (hit, len(rows), time.time() - started)
    lo, hi = clopper_pearson(hit, max(len(rows), 1))
    print(f"  {arm:8} {hit}/{len(rows)} = {hit/max(len(rows),1)*100:5.1f}%  "
          f"[{lo:.1f}-{hi:.1f}]  {time.time()-started:.0f}s", flush=True)

ans = {a: {r["id"]: r["correct"] for r in record.rows if r["arm"] == a} for a in summary}
print("\n  paired against the full roster")
for arm in ("scene", "scene+2"):
    p, x, y, nn = paired(ans["full"], ans[arm])
    h0, n0 = summary["full"][0], summary["full"][1]
    h1, n1 = summary[arm][0], summary[arm][1]
    print(f"    {arm:8} {(h1/n1 - h0/n0)*100:+6.1f} points  +{y:3}/-{x:3}  p={p:.4g}")
out = record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"scene_cast__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json"),
    contract={"expected_arms": ARMS})
print("wrote", out)
