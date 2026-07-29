"""Does attributing a whole exchange at once beat attributing each line alone -
and if it does, is it the ORDER that helps or just seeing the lines together?

Where this comes from. Blind adjudication of the rows every model fails found
twenty cases of unmarked alternating dialogue where the models do not guess
randomly, they converge on the WRONG TURN: all nine runs answered MOGUZO where
the answer was RANTA. 64% of grimgar03's gold lines abut another spoken segment
with no narration between them. Production attributes batches of 25 entries
independently and never sees its own decisions.

The obvious fix - hand the model the previous speaker - was tested and did
nothing. `committed_history` moved accuracy by exactly zero rows with the TRUE
previous speaker supplied free (63.5% none, 63.5% oracle). That result is what
makes this experiment worth running rather than redundant: sequential state was
not the missing ingredient, so if turn-taking structure is usable at all it has
to be used jointly - assigning a whole exchange as one interdependent set,
where "these two alternate" can constrain both ends at once.

THE CONTROL IS THE POINT. A joint arm that beats independent decoding proves
almost nothing on its own, because it changes two things simultaneously: the
model sees more text at once, AND it sees the lines in order. The shuffled arm
holds the first constant and destroys the second - identical lines, identical
scene, identical prompt shape, presented in a fixed-seed random order and
mapped back by explicit line ids.

Readings, fixed before running:

  chrono > independent AND chrono > shuffled   order carries usable signal;
                                               sequential/joint decoding is the
                                               fix and the alternation pattern
                                               is exploitable
  chrono ~ shuffled, both > independent        co-presentation helps, ORDER does
                                               not; ship joint decoding but drop
                                               the turn-taking story entirely
  both ~ independent                           co-presentation adds nothing;
                                               the alternating-error pattern is
                                               a symptom, not a lever - retire
                                               this line of attack
  shuffled > chrono                            suspect the id mapping before
                                               believing it

Scoring is restricted to gold lines, but the model attributes every line in the
scene: a joint decision that only had to cover the scored lines would not be
the joint decision production would make.
"""
import collections
import json, os, random, re, sys, time
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")
import openai
from openai import OpenAI
from experiments.manifest import ExperimentRecord
from experiments.stats import exact_mcnemar
from three_pass_generate import build_roster

RETRYABLE = (openai.APIConnectionError, openai.APITimeoutError,
             openai.InternalServerError, openai.RateLimitError,
             openai.NotFoundError)
MAX_ATTEMPTS = 6

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
# w4 on either side of the scene, matching the width the diagnostic sweep and
# the production gate both settled on, so this is not also a width experiment.
WIDTH = int(os.environ.get("EXPERIMENT_WIDTH", "4"))
SEED = int(os.environ.get("EXPERIMENT_SEED", "20260728"))

gold = json.load(open(GOLD_PATH))
src = open(M + f"inputs/{BOOK}.txt", encoding="utf-8").read()
cp = json.load(open(M + INPUT_RUN + f"/{BOOK}/result.json.threepass_checkpoint.json"))
seg, named = cp["segmented"], [e for e in (cp.get("named") or []) if e]
roster = [r.upper() for r in build_roster(named, src)]
AL = [{n.upper() for n in g} for g in gold.get("aliases", [])]


def same(a, b):
    a, b = (a or "").upper(), (b or "").upper()
    return a == b or any(a in g and b in g for g in AL)


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


pos = {norm(e["text"]): i for i, e in enumerate(seg)}
_occ = collections.Counter(norm(e.get("text")) for e in seg)
SCOREABLE = {pos[norm(g["line"])]: g for g in gold["entries"]
             if _occ[norm(g["line"])] == 1 and norm(g["line"]) in pos}


def scenes():
    """Maximal runs of consecutive spoken segments, bounded by narration.

    This is the unit the turn-taking hypothesis is about: inside a run there is
    no narration to anchor anyone, so alternation is the only structure
    available. Runs with no gold line in them are dropped - they cost tokens and
    contribute no measurement.
    """
    out, run = [], []
    for i, entry in enumerate(seg):
        if entry.get("type") == "NARRATOR":
            if run:
                out.append(run)
            run = []
        else:
            run.append(i)
    if run:
        out.append(run)
    return [r for r in out if any(i in SCOREABLE for i in r)]


SCENES = scenes()
covered = sum(1 for r in SCENES for i in r if i in SCOREABLE)
lengths = sorted(len(r) for r in SCENES)
print(f"{len(SCENES)} scenes covering {covered}/{len(SCOREABLE)} scoreable lines | "
      f"scene length median {lengths[len(lengths)//2]}, max {max(lengths)} | "
      f"roster {len(roster)}", flush=True)

client = OpenAI(base_url=BASE_URL, api_key="local")
SYSTEM = ("You identify who speaks each line of dialogue in a novel. Answer "
          "with one line per numbered item, formatted exactly as `id: NAME` "
          "with the name in CAPITALS. If a line's speaker is not determined by "
          "the passage, answer UNKNOWN. Answer every id you are given and "
          "nothing else.")


def context_around(scene):
    before = " ".join((seg[j].get("text") or "")
                      for j in range(max(0, scene[0] - WIDTH), scene[0]))
    after = " ".join((seg[j].get("text") or "")
                     for j in range(scene[-1] + 1,
                                    min(len(seg), scene[-1] + 1 + WIDTH)))
    return before, after


def ask_scene(scene, order):
    """One call for the whole scene. `order` is the presentation sequence.

    Ids are the segment indices and are attached to each line explicitly, so the
    shuffled arm can be mapped back without relying on position. That mapping is
    the one thing that could silently fake a result, which is why the ids are
    printed with the lines rather than inferred from order.
    """
    before, after = context_around(scene)
    body = "\n".join(f"{i}: {seg[i].get('text')}" for i in order)
    user = (f"PASSAGE BEFORE THE EXCHANGE:\n{before}\n\n"
            f"LINES TO ATTRIBUTE"
            + (" (in the order they appear in the book):"
               if order == scene else
               " (listed in arbitrary order, not the order they appear):")
            + f"\n{body}\n\n"
            f"PASSAGE AFTER THE EXCHANGE:\n{after}\n\n"
            f"Each speaker is one of: {', '.join(roster + ['UNKNOWN'])}\n\n"
            f"Give the speaker of every line, one `id: NAME` per line.")
    last = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=[{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": user}],
                temperature=0.0, max_tokens=16 * len(order) + 64,
                extra_body={"reasoning_effort": "none"})
            raw = (r.choices[0].message.content or "")
            # Models do not reliably echo the ids they are given: qwen3-14b
            # returned the segment indices, llama-3.3-70b renumbered the items
            # 1..N and every answer was discarded, producing an artifact that
            # read 0.0% and still validated. So accept either - the supplied
            # ids, or a 1-based sequence mapped back through `order`.
            pairs = []
            for line in raw.splitlines():
                m = re.match(r"\s*(\d+)\s*[:.\-]\s*(.+?)\s*$", line)
                if not m:
                    continue
                # The 70B answers "1: **33**: RUDI" - its own sequence number,
                # then the id it was given, then the name. Taking the first
                # capture as the name yielded "**33**: RUDI", which scored
                # wrong while looking answered: 9.4% accuracy that passed the
                # unanswered guard. The name is whatever follows the LAST
                # separator, with markdown stripped.
                tail = re.split(r"[:\-]", m.group(2))[-1]
                name = re.sub(r"[*_`]", "", tail).upper().strip(".'\" ")
                if name:
                    pairs.append((int(m.group(1)), name))
            supplied = set(order)
            got = {k: v for k, v in pairs if k in supplied}
            if not got and pairs and {k for k, _ in pairs} <= set(
                    range(1, len(order) + 1)):
                got = {order[k - 1]: v for k, v in pairs}
            return got, user, raw, attempt
        except RETRYABLE as exc:
            last = exc
            if attempt == MAX_ATTEMPTS - 1:
                break
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"endpoint failed {MAX_ATTEMPTS} attempts against "
                       f"{BASE_URL}: {type(last).__name__}: {last}") from last


def ask_single(index):
    """The incumbent: one line, w4 context, no siblings."""
    before = " ".join((seg[j].get("text") or "")
                      for j in range(max(0, index - WIDTH), index))
    after = " ".join((seg[j].get("text") or "")
                     for j in range(index + 1, min(len(seg), index + 1 + WIDTH)))
    user = (f"PASSAGE BEFORE:\n{before}\n\nLINE:\n{seg[index].get('text')}\n\n"
            f"PASSAGE AFTER:\n{after}\n"
            f"\nThe speaker is one of: {', '.join(roster + ['UNKNOWN'])}\n\n"
            f"Who speaks the LINE? Answer `{index}: NAME`.")
    last = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=[{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": user}],
                temperature=0.0, max_tokens=24,
                extra_body={"reasoning_effort": "none"})
            raw = (r.choices[0].message.content or "")
            m = re.search(r"\d+\s*[:.\-]\s*(.+?)\s*$", raw.strip().splitlines()[0]
                          if raw.strip() else "")
            got = (m.group(1) if m else raw).upper().strip(".'\"* ")
            return {index: got}, user, raw, attempt
        except RETRYABLE as exc:
            last = exc
            if attempt == MAX_ATTEMPTS - 1:
                break
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"endpoint failed {MAX_ATTEMPTS} attempts against "
                       f"{BASE_URL}: {type(last).__name__}: {last}") from last


_env = os.environ.get("EXPERIMENT_ENV")
record = ExperimentRecord(
    "joint_scene", REPO, MODEL, BASE_URL, GOLD_PATH,
    {"temperature": 0.0, "reasoning_effort": "none", "context_width": WIDTH,
     "shuffle_seed": SEED},
    environment=json.loads(_env) if _env else None,
    notes="independent vs joint-chronological vs joint-shuffled scene "
          "attribution. The shuffled arm is the control that separates seeing "
          "the lines together from seeing them in order; without it a joint "
          "gain cannot be attributed to turn-taking. Follows committed_history, "
          "where oracle previous-speaker state moved zero rows.")
record.enable_checkpoint(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"joint_scene__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json.ckpt"))

rng = random.Random(SEED)
# Drawn once, before any arm runs, so both joint arms see the same permutation
# for a given scene and a re-run reproduces it exactly.
#
# Forced to a DERANGEMENT, not a plain shuffle. The median scene here is two
# lines long, and a plain shuffle of two elements returns the original order
# half the time - which would have made half the control arm a duplicate of the
# chronological arm and biased the comparison towards "order does not matter"
# by construction. For a two-line scene the derangement is the reversal, which
# is also the strongest order perturbation available.
PERMUTED = {}
for scene in SCENES:
    order = list(scene)
    if len(order) > 1:
        while order == list(scene):
            rng.shuffle(order)
    PERMUTED[scene[0]] = order

by_arm = {}
for arm in ("independent", "joint-chrono", "joint-shuffled"):
    started, calls = time.time(), 0
    for scene in SCENES:
        wanted = [i for i in scene if i in SCOREABLE]
        if all(record.done(arm, SCOREABLE[i]["id"]) for i in wanted):
            continue
        if arm == "independent":
            got = {}
            for i in wanted:
                one, prompt, raw, retries = ask_single(i)
                got.update(one)
                calls += 1
        else:
            order = scene if arm == "joint-chrono" else PERMUTED[scene[0]]
            got, prompt, raw, retries = ask_scene(scene, order)
            calls += 1
        for i in wanted:
            g = SCOREABLE[i]
            answer = got.get(i)
            record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(),
                       answer, same(answer, g["expected_speaker"]),
                       candidates=roster,
                       provenance=f"{arm}|scene={scene[0]}|scene_len={len(scene)}",
                       prompt=prompt, raw=raw, retries=retries)
    rows = [r for r in record.rows if r["arm"] == arm]
    hit = sum(1 for r in rows if r["correct"])
    miss = sum(1 for r in rows if r["predicted"] is None)
    by_arm[arm] = (hit, len(rows))
    print(f"  {arm:16} {hit}/{len(rows)} = {hit/max(len(rows),1)*100:5.1f}%   "
          f"{calls} calls, {miss} unanswered, {time.time()-started:.0f}s",
          flush=True)

# --------------------------------------------------------------- the readings
answers = {arm: {r["id"]: r["correct"] for r in record.rows if r["arm"] == arm}
           for arm in by_arm}


# Same guard the width gate carries: an arm that answered almost nothing is an
# environment or parsing failure, not a null result, and must not be written as
# though it were one. The 70B joint run reached 0.0% and validated without it.
# Two guards, because the first one was not enough. "Unanswered" caught the
# arm that returned nothing; it did not catch the arm that returned
# "**33**: RUDI" for every row - answered, parsed, and scored 9.4%. An answer
# that is not a roster name is a parse failure wearing a result's clothes.
_ROSTER = {r.upper() for r in roster} | {"UNKNOWN"}
for _arm, (_hit, _n) in by_arm.items():
    _rows = [r for r in record.rows if r["arm"] == _arm]
    if not _rows:
        continue
    _blank = sum(1 for r in _rows if r["predicted"] is None) / len(_rows)
    _named = [r for r in _rows if r["predicted"] is not None]
    _off = (sum(1 for r in _named if (r["predicted"] or "").upper() not in _ROSTER)
            / len(_named)) if _named else 0
    if _blank > 0.25:
        raise SystemExit(f"refusing to write: {_arm} left {_blank*100:.0f}% of "
                         f"rows unanswered - check the reply format")
    if _off > 0.5:
        raise SystemExit(f"refusing to write: {_arm} produced {_off*100:.0f}% "
                         f"answers that are not roster names, e.g. "
                         f"{next(r['predicted'] for r in _named if (r['predicted'] or '').upper() not in _ROSTER)!r} "
                         f"- that is a parse failure, not a result")

MULTI = {SCOREABLE[i]["id"] for scene in SCENES if len(scene) > 1
         for i in scene if i in SCOREABLE}


def compare(a, b, restrict=None):
    shared = set(answers[a]) & set(answers[b])
    if restrict is not None:
        shared &= restrict
    x = sum(1 for i in shared if answers[a][i] and not answers[b][i])
    y = sum(1 for i in shared if answers[b][i] and not answers[a][i])
    p, _, _ = exact_mcnemar(x, y)
    print(f"  {b:16} vs {a:16} +{y:3} / -{x:3}   p={p:.4f}   (n={len(shared)})")


print("\n  paired transitions (exact McNemar, discordant pairs only)")
compare("independent", "joint-chrono")
compare("independent", "joint-shuffled")
compare("joint-shuffled", "joint-chrono")

# Declared here, before any result exists, for the same reason the bin edges in
# length_bins are: a subgroup chosen after seeing the totals is not a subgroup,
# it is a search. A one-line scene has no siblings, so all three arms are the
# same prompt shape and those rows can only dilute the contrast.
print(f"\n  restricted to multi-line scenes ({len(MULTI)} of {len(SCOREABLE)} rows),")
print("  where the hypothesis applies at all:")
compare("independent", "joint-chrono", MULTI)
compare("independent", "joint-shuffled", MULTI)
compare("joint-shuffled", "joint-chrono", MULTI)

print("\n  The chrono-vs-shuffled line is the experiment. The other two only")
print("  establish that something changed; only that one says whether ORDER was")
print("  it. Read the multi-line block as primary and the full block as the")
print("  production-weighted effect.")

out = record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"joint_scene__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json"),
    contract={"expected_arms": ("independent", "joint-chrono", "joint-shuffled"),
              "require_clean_tree": True})
print("wrote", out)
