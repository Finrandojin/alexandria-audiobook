"""Why is batch size worth 18.7 points? Context, or conversation?

Batch size is the largest single lever measured in this investigation. On
grimgar03 the 70B went from 60.5% at b1 to 79.2% at b25, and `batch_size` found
peaks at 50 on one book and 25 on another. The mechanism has never been tested,
and it matters: the two candidate explanations point at different work.

    amortised context   a bigger batch simply puts more of the book in the
                        prompt, and any 25 lines would do
    conversation        the batch happens to contain a RUN of turn-taking, and
                        the model attributes a conversation better than it
                        attributes isolated lines

TWO ARMS, IDENTICAL IN EVERY RESPECT BUT ONE:

    contiguous   25 consecutive segments, which is what production sends
    scattered    25 segments drawn from across the whole book, seeded

Both send 25 entries. Both carry the same per-entry `previous_context` and
`next_context`, so each line keeps its own immediate neighbours and the local
evidence is preserved. The ONLY difference is whether the other 24 entries in
the batch are that line's conversation or strangers.

READINGS, fixed before running:

  contiguous >> scattered   the gain is conversational structure. Batch
                            boundaries then matter, and aligning them to scene
                            or turn boundaries is a new lever worth testing.
  contiguous ~ scattered    the gain is amortised prompt context and nothing
                            more; batch size is a knob, not a phenomenon, and
                            there is nothing further here.
  scattered >> contiguous   would mean contiguity actively hurts, e.g. the
                            model over-applies alternation within a run. That
                            would make it a candidate explanation for
                            owarimonogatari3, where 50 of 63 arms score below a
                            previous-speaker baseline.

WHY THIS IS FAIR. Scattered batches are built to contain the SAME scored rows
as the contiguous ones, so the two arms are scored on an identical row set and
paired exactly. A scattered arm that answered a different subset would not be
comparable, which is the defect that made w4 read 11 points high.

The token count differs slightly between arms because different companion
lines have different lengths; the entry COUNT is what is held fixed, and mean
prompt length is reported so any imbalance is visible rather than assumed away.
"""
import collections, json, os, random, re, sys, time

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
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
SEED = int(os.environ.get("EXPERIMENT_SEED", "20260731"))
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
scored = [i for i in sendable if norm(seg[i].get("text")) in want]
print(f"{BOOK}: {len(want)} scoreable lines, {len(sendable)} sendable, "
      f"roster {len(roster)}", flush=True)

# Contiguous windows, exactly as production builds them.
contiguous = [sendable[s:s + BATCH] for s in range(0, len(sendable), BATCH)]
contiguous = [w for w in contiguous if any(i in set(scored) for i in w)]

# Scattered batches cover the SAME scored rows, each padded with companions
# drawn from across the book rather than from its neighbourhood.
rng = random.Random(SEED)
pool = list(sendable)
scattered, bucket = [], []
targets = [i for w in contiguous for i in w if norm(seg[i].get("text")) in want]
for target in targets:
    bucket.append(target)
    if len(bucket) == 1:
        companions = []
        far = [j for j in pool if abs(j - target) > 200]
        rng.shuffle(far)
        companions = far[:BATCH - 1]
        scattered.append(sorted(set([target] + companions)))
        bucket = []

ARMS = ("contiguous", "scattered")
client = OpenAI(base_url=BASE_URL, api_key="local")
params = LLMGenParams(max_tokens=12000, context_length=32768, temperature=0.0,
                      attribute_temperature=0.0, top_p=0.8,
                      reasoning_effort="none")
_env = os.environ.get("EXPERIMENT_ENV")
record = ExperimentRecord(
    "batch_contiguity", REPO, MODEL, BASE_URL, GOLD_PATH,
    {"temperature": 0.0, "batch": BATCH, "seed": SEED},
    environment=json.loads(_env) if _env else None,
    notes="Isolates why batch size helps. Both arms send the same number of "
          "entries with the same per-entry neighbour contexts; only the "
          "companions differ - the line's own conversation, or 24 strangers "
          "from elsewhere in the book.")
record.enable_checkpoint(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"batch_contiguity__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json.ckpt"))

summary, prompt_chars = {}, collections.defaultdict(list)
for arm in ARMS:
    windows = contiguous if arm == "contiguous" else scattered
    started = time.time()
    for k, win in enumerate(windows, 1):
        rows = [i for i in win if norm(seg[i].get("text")) in want]
        if not rows:
            continue
        if all(record.done(arm, want[norm(seg[i].get("text"))]["id"]) for i in rows):
            continue
        frozen = [{"type": seg[i]["type"], "text": seg[i]["text"]} for i in win]
        ctx = [{"previous_context": seg[i - 1] if i else None,
                "next_context": seg[i + 1] if i + 1 < len(seg) else None}
               for i in win]
        prompt_chars[arm].append(sum(len(e["text"] or "") for e in frozen))
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
            key = norm(seg[i].get("text"))
            if key not in want:
                continue
            g = want[key]
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
    mean_chars = (sum(prompt_chars[arm]) / len(prompt_chars[arm])
                  if prompt_chars[arm] else 0)
    print(f"  {arm:11} {hit}/{len(arm_rows)} = "
          f"{hit/max(len(arm_rows),1)*100:5.1f}%  [{lo:.1f}-{hi:.1f}]  "
          f"mean batch {mean_chars:.0f} chars  {time.time()-started:.0f}s",
          flush=True)

ans = {a: {r["id"]: r["correct"] for r in record.rows if r["arm"] == a}
       for a in summary}
p, x, y, n = paired(ans["contiguous"], ans["scattered"])
h0, n0 = summary["contiguous"]
h1, n1 = summary["scattered"]
print(f"\n  scattered - contiguous  {(h1/max(n1,1) - h0/max(n0,1))*100:+.1f} "
      f"points  +{y}/-{x} of {n}  p={p:.4g}")
print("  A null here says batch size is amortised context and nothing more.")
print("  A contiguous win says the unit of attribution is the conversation, "
      "and\n  batch boundaries are worth aligning to it.")

out = record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"batch_contiguity__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json"),
    contract={"expected_arms": ARMS})
print("wrote", out)
