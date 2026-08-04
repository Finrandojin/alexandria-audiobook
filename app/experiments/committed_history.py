"""Does supplying the previous speaker help, and can we supply it accurately?

Everything measured on 2026-07-27 converges here. Blind adjudication of the 26
grimgar03 rows that EVERY model run fails with the true speaker among five
candidates found three causes: four contested gold labels, two rows of an
undeclared alias, and **twenty rows of unmarked alternating dialogue**. On
those twenty the models do not guess randomly - they converge on the WRONG
TURN. All nine runs answered MOGUZO where the answer was RANTA; eight answered
RON for RANTA; eight answered SHIHORU for MOGUZO.

64% of grimgar03's gold lines sit directly against another spoken segment with
no narration between them, so turn-taking is the only available evidence on
most of the fixture. Production pass 2 attributes batches of 25 entries
INDEPENDENTLY and never sees its own prior decisions.

FOUR ARMS. A two-arm test conflates two questions, and a three-arm one leaves
the answer unbuilt:

    none        no history                    - what production does today
    oracle      TRUE previous speaker         - is the representation useful?
    predicted   this run's own prior answer   - can we supply it well enough?
    gated       own answer, only where a second pass agrees - can we supply it
                well enough SOMETIMES, and is that enough?

The gated arm was added after the first run's null was found to be a pooling
artifact. Per book the oracle is worth +9.3 on owarimonogatari3 and +3.7 on
mushoku16 while predicted costs ~3 on both: the representation works and the
state source fails. The gate supplies history only where the no-history pass
independently agrees, trading coverage for correctness with no gold, so a win
is shippable.

Readings, fixed before running:

  oracle helps, predicted does not  - the representation is useful and the state
                                      source is not; work on the state source
  both help                         - production candidate
  neither helps                     - retire simple sequential history; the next
                                      candidate is joint scene decoding, which
                                      exploits turn-taking without committing an
                                      early error as immutable state
  predicted helps more than oracle  - almost certainly a bug; investigate before
                                      believing it

The `scaffold` arm already asked the model to INFER `previous_speaker` and lost
4.0 points. That is not this experiment. Asking a model to introspect is not the
same as handing it state, and the distinction is the whole point.

ERROR PROPAGATION is measured, not assumed: the predicted arm is decoded in
book order so a wrong answer becomes the next line's history, and accuracy is
reported by distance from the last narration anchor. If early mistakes
compound, that shows up as decay with distance.
"""
import collections
import json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import openai
from openai import OpenAI
from experiments.manifest import ExperimentRecord
from three_pass_generate import build_roster

RETRYABLE = (openai.APIConnectionError, openai.APITimeoutError,
             openai.InternalServerError, openai.RateLimitError,
             openai.NotFoundError)
MAX_ATTEMPTS = 6

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
WIDTH = int(os.environ.get("EXPERIMENT_WIDTH", "4"))

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
SCOREABLE = [g for g in gold["entries"] if _occ[norm(g["line"])] == 1]
# Book order matters for the predicted arm: history must be built from answers
# already given, exactly as a sequential production pass would.
SCOREABLE.sort(key=lambda g: pos.get(norm(g["line"]), 0))
truth_at = {pos[norm(g["line"])]: g["expected_speaker"].upper()
            for g in SCOREABLE if norm(g["line"]) in pos}

print(f"scoring {len(SCOREABLE)} gold lines in book order, roster {len(roster)}",
      flush=True)

client = OpenAI(base_url=BASE_URL, api_key="local")
SYSTEM = ("You identify who speaks a line of dialogue in a novel. Answer with "
          "the speaker's name in CAPITALS and nothing else. If the passage "
          "does not determine it, answer UNKNOWN.")


def anchor_distance(index):
    """Spoken segments since the last narration segment.

    Narration is where speech tags and names live, so this is a proxy for how
    far the line has drifted from any explicit anchor. Accuracy by this value is
    how error propagation shows up: if committed history compounds mistakes,
    the predicted arm decays as distance grows while the oracle arm does not.
    """
    d = 0
    for j in range(index - 1, -1, -1):
        if seg[j].get("type") == "NARRATOR":
            return d
        d += 1
    return d


def prior_speakers(index, source, answers, k=3):
    """The last k resolved speakers before `index`, most recent last.

    source="oracle"    - gold, an upper bound on the representation's value
    source="predicted" - this run's own earlier answers, what production could
                         actually supply
    source="gated"     - this run's own answers, but ONLY where an independent
                         second pass agreed. Wrong history is what makes
                         `predicted` lose, so the gate trades coverage for
                         correctness using no gold, which means a win here is
                         shippable rather than an upper bound.
    """
    out = []
    for j in range(index - 1, -1, -1):
        if len(out) >= k:
            break
        if seg[j].get("type") == "NARRATOR":
            continue
        if source == "oracle":
            who = truth_at.get(j)
        elif source == "gated":
            # Only trust this run's answer where the confirming pass agrees.
            # STOP on a blocked entry rather than skipping past it. The first
            # two attempts continued the loop, so a blocked line was silently
            # replaced by one further back and the arm always ended up with
            # three names - the gate changed WHICH names appeared, never
            # whether any did, and fired on 99.4% of rows twice running.
            # Stopping is what "supply nothing when uncertain" actually means.
            who = answers.get(j)
            if who and not same(who, CONFIRM.get(j, "")):
                break
        else:
            who = answers.get(j)
        if who:
            out.append(who)
    return list(reversed(out))


def ask(line, index, history):
    before = " ".join((seg[j].get("text") or "")
                      for j in range(max(0, index - WIDTH), index))
    after = " ".join((seg[j].get("text") or "")
                     for j in range(index + 1, min(len(seg), index + 1 + WIDTH)))
    hist = ""
    if history:
        hist = ("\nThe previous spoken lines were said by, in order: "
                + " -> ".join(history))
    user = (f"PASSAGE BEFORE:\n{before}\n\nLINE:\n{line}\n\n"
            f"PASSAGE AFTER:\n{after}\n{hist}\n"
            f"\nThe speaker is one of: {', '.join(roster + ['UNKNOWN'])}\n\n"
            f"Who speaks the LINE?")
    last = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=[{"role": "system", "content": SYSTEM},
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
    "committed_history", REPO, MODEL, BASE_URL, GOLD_PATH,
    {"temperature": 0.0, "max_tokens": 24, "reasoning_effort": "none",
     "context_width": WIDTH, "history_depth": 3},
    environment=json.loads(_env) if _env else None,
    notes="none vs oracle vs predicted previous-speaker history. Twenty of the "
          "26 unanimous oracle failures on this book are unanchored "
          "turn-taking, and 64% of gold lines abut another spoken segment. "
          "Separates whether the representation is useful from whether the "
          "state can be supplied.")
ARMS = ("none", "oracle", "predicted", "gated")
record.enable_checkpoint(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"committed_history__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json.ckpt"))

# The gate's confirming pass. The first version reused the `none` arm, which
# fired on 99.4% of rows and made `gated` a rename of `predicted`: same model,
# temperature 0, nearly identical prompt, so it agreed with itself almost
# always. A confirming pass has to vary the EVIDENCE, not just be asked twice,
# so this one runs at a different context width. Where two different views of
# the passage agree, the answer is much more likely right - and it needs no
# gold, so a win here is shippable.
CONFIRM_WIDTH = int(os.environ.get("EXPERIMENT_CONFIRM_WIDTH", "12"))
CONFIRM = {}


def confirming_pass():
    """Answer every scored line with no history at a different context width."""
    global WIDTH
    keep, WIDTH = WIDTH, CONFIRM_WIDTH
    try:
        for g in SCOREABLE:
            i = pos.get(norm(g["line"]))
            if i is None:
                continue
            got, _, _, _ = ask(g["line"], i, [])
            CONFIRM[i] = got
    finally:
        WIDTH = keep


confirming_pass()
_agree = sum(1 for g in SCOREABLE
             if pos.get(norm(g["line"])) in CONFIRM)
print(f"  confirming pass done at width {CONFIRM_WIDTH} "
      f"({_agree} lines answered)", flush=True)

by_arm = {}
for arm in ARMS:
    started = time.time()
    answers = {}
    for g in SCOREABLE:
        i = pos.get(norm(g["line"]))
        if i is None:
            continue
        if record.done(arm, g["id"]):
            continue
        hist = [] if arm == "none" else prior_speakers(i, arm, answers)
        got, prompt, raw, retries = ask(g["line"], i, hist)
        answers[i] = got
        if arm == "none":
            # The no-history arm doubles as the gate's confirming pass; it is
            # already being run, so the gate costs no extra inference.
            CONFIRM[i] = got
        record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(), got,
                   same(got, g["expected_speaker"]), candidates=roster,
                   provenance=f"{arm}|anchor_dist={anchor_distance(i)}|hist={len(hist)}",
                   prompt=prompt, raw=raw, retries=retries)
    rows = [r for r in record.rows if r["arm"] == arm]
    hit = sum(1 for r in rows if r["correct"])
    by_arm[arm] = (hit, len(rows))
    print(f"  {arm:10} {hit}/{len(rows)} = {hit/max(len(rows),1)*100:5.1f}%   "
          f"{time.time()-started:.0f}s", flush=True)

gate_fired = sum(1 for r in record.rows if r["arm"] == "gated"
                 and "hist=0" not in (r.get("candidate_provenance") or ""))
gate_rows = sum(1 for r in record.rows if r["arm"] == "gated")
print(f"\n  gate supplied history on {gate_fired}/{gate_rows} rows "
      f"= {gate_fired/max(gate_rows,1)*100:.1f}%")
print("  A gate that fires on almost nothing cannot move the score, and one "
      "that\n  fires on almost everything is not a gate - read the arms "
      "against this.")

base = by_arm["none"]
print("\n  arm         accuracy   vs none")
for arm in ARMS:
    h, n = by_arm[arm]
    print(f"  {arm:10} {h/n*100:7.1f}%  {(h-base[0])/n*100:+6.1f}")

# Accuracy by distance from the last narration anchor: error propagation, if it
# happens, appears as the predicted arm decaying where the oracle arm does not.
print("\n  accuracy by distance from the last narration anchor:")
buckets = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
for r in record.rows:
    m = re.search(r"anchor_dist=(\d+)", r.get("candidate_provenance") or "")
    if not m:
        continue
    d = int(m.group(1))
    key = "0" if d == 0 else "1" if d == 1 else "2-3" if d <= 3 else "4+"
    b = buckets[key][r["arm"]]
    b[0] += 1
    b[1] += bool(r["correct"])
print(f"    {'distance':10} " + " ".join(f"{a:>10}" for a in ("none", "oracle", "predicted")))
for key in ("0", "1", "2-3", "4+"):
    cells = []
    for arm in ("none", "oracle", "predicted"):
        n, ok = buckets[key][arm]
        cells.append(f"{ok/n*100:9.1f}%" if n else "        -")
    n0 = buckets[key]["none"][0]
    print(f"    {key:10} " + " ".join(cells) + f"   (n={n0})")

out = record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"committed_history__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json"),
    contract={"expected_arms": ARMS,
              "expected_ids": {g["id"] for g in SCOREABLE},
              "require_clean_tree": True})
print("wrote", out)
