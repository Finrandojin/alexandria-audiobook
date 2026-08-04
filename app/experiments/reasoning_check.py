"""Does a model's own justification reveal when its answer is wrong?

The observation this comes from: judging owarimonogatari3, gemini answered
UNKNOWN on two rows while its own `reasoning` field named Ougi Oshino
outright - "the speaker (Ougi Oshino) is missing from the roster". The answer
was wrong and the reasoning was right, in the same response. Index18's LESSER
is a third instance.

That is the "selection not recall" pattern again: the roster contains the
right name 85% of the time and the model picks it 29.9%. If the reason field
carries information the answer field loses, a cheap consistency check between
them is an error detector.

WHY THIS IS NOT THE `because` EXPERIMENT. `because` asked whether justifying
improves the answer and the answer is no - it lost 7.2 points on the
production path. This asks whether the justification DETECTS a wrong answer,
which does not require it to improve anything. A model that reasons correctly
and then emits the wrong name is exactly what we want to catch, and that is
the case `because` scored as a failure.

MEASURED ON THE JUDGES FIRST, and it was weak there: across 1043 rows from two
frontier judges, accuracy was 97.9% where the reasoning confirmed the answer
and 95.5% where it named only other people - 2.4 points of separation on
overlapping intervals. But those judges are 96-99% accurate, so there is
almost no error to detect, and most mismatches were the reasoning mentioning
an ADDRESSEE ("his leap toward Hachikuji") rather than making a competing
claim. Neither of those objections applies to a 14B model at ~55%.

THE BAR IT HAS TO CLEAR. The disagreement cascade already routes on a signal
that separates 78.2% from 51.6% at 40% coverage - but it costs TWO cheap calls
to compute, because it compares w1 against w4. A reasoning-consistency flag
costs ONE. So it does not need to beat the cascade's separation, only to earn
its place at half the price:

    separation >= ~15 points at <= 40% coverage   worth routing on
    separation < 5 points                         the judges' result holds at
                                                  every capability level and
                                                  this line closes
    high separation but tiny coverage             a precision instrument with
                                                  no reach, like the confidence
                                                  signal that died at 17%

Arms: `plain` is the shipped prompt; `reasoned` adds a because-style field.
Accuracy of `reasoned` is expected to be WORSE and that is not the finding -
the finding is whether its mismatch flag sorts right from wrong.
"""
import collections
import json, os, re, sys, time
from dataclasses import replace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
from default_prompts import load_attribute_prompts
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
# GOLD follows BOOK by default. It used to hardcode grimgar03's fixture
# while BOOK stayed settable, so setting only EXPERIMENT_BOOK scored one
# book's lines against another book's gold - three matches out of 162,
# every arm 0.0%. Two runs were lost to it before the pattern was seen.
GOLD = os.environ.get("EXPERIMENT_GOLD",
                      f"fixtures/attribution_gold_{BOOK}.json")
GOLD_PATH = APP + GOLD
BASE_URL = os.environ.get("EXPERIMENT_BASE_URL", "http://127.0.0.1:8090/v1")
TAG = os.environ.get("EXPERIMENT_TAG", "local-llamacpp")
BATCH = int(os.environ.get("EXPERIMENT_BATCH", "25"))

gold = json.load(open(GOLD_PATH))
src = open(M + f"inputs/{BOOK}.txt", encoding="utf-8").read()
cp = json.load(open(M + INPUT_RUN + f"/{BOOK}/result.json.threepass_checkpoint.json"))
seg = cp["segmented"]
roster = build_roster([e for e in (cp.get("named") or []) if e], src)
GROUPS = alias_groups(gold)
ROSTER = {r.upper() for r in roster} | {
    n.upper() for n in gold.get("roster_additions", {}).get("names", [])}


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


_occ = collections.Counter(norm(e.get("text")) for e in seg)
want = {norm(g["line"]): g for g in gold["entries"] if _occ[norm(g["line"])] == 1}
print(f"roster {len(ROSTER)} | {len(want)} lines | batch {BATCH}", flush=True)

BASE_SYSTEM, _ = load_attribute_prompts()
REASONED_SYSTEM = BASE_SYSTEM.replace(
    'return {"n": <same index>, "speaker": "..."} where:',
    'return {"n": <same index>, "speaker": "...", "because": "<one short '
    'clause naming the evidence: the dialogue tag, who is addressed, or whose '
    'turn it is>"} where:').replace(
    "Output ONLY a valid JSON array — no markdown, no explanations.",
    "Output ONLY a valid JSON array — no markdown. The explanation belongs in "
    "the \"because\" field of each entry, nowhere else.")

client = OpenAI(base_url=BASE_URL, api_key="local")

# attribute_batch returns {**frozen, "speaker": ...} and DISCARDS every other
# field the model produced - which is why because_production stored
# raw_response: None and why this experiment would otherwise measure nothing.
# Rather than change the production path to suit the experiment, wrap the
# client and keep what the path throws away. The call itself is untouched, so
# this is still the shipped code being measured.
TRANSCRIPT = []
_real_create = client.chat.completions.create


def _recording_create(*args, **kwargs):
    response = _real_create(*args, **kwargs)
    try:
        TRANSCRIPT.append(response.choices[0].message.content or "")
    except (AttributeError, IndexError):
        TRANSCRIPT.append("")
    return response


client.chat.completions.create = _recording_create


def because_by_index(raw_texts):
    """Pull {n: because} out of whatever the model actually emitted.

    Keyed on the model's own `n`, not on position, because a retry inside
    attribute_batch appends another completion to the transcript and the last
    one is the one whose speakers were used.
    """
    out = {}
    for text in raw_texts:
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end < 0:
            continue
        try:
            items = json.loads(text[start:end + 1])
        except ValueError:
            continue
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict) and "n" in item and item.get("because"):
                out[item["n"]] = str(item["because"])
    return out
params = LLMGenParams(max_tokens=12000, context_length=32768, temperature=0.0,
                      attribute_temperature=0.0, top_p=0.8,
                      reasoning_effort="none")

_env = os.environ.get("EXPERIMENT_ENV")
record = ExperimentRecord(
    "reasoning_check", REPO, MODEL, BASE_URL, GOLD_PATH,
    {"temperature": 0.0, "max_tokens": 12000, "batch": BATCH, "width": 1},
    environment=json.loads(_env) if _env else None,
    notes="Whether a model's own justification flags its wrong answers. Not "
          "the `because` experiment: that asked whether justifying improves "
          "accuracy (it does not, -7.2). This asks whether the mismatch "
          "between reason and answer sorts right from wrong, which is a "
          "one-call routing trigger where the cascade needs two.")
record.enable_checkpoint(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"reasoning_check__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json.ckpt"))

windows = [list(range(s, min(s + BATCH, len(seg)))) for s in range(0, len(seg), BATCH)]
windows = [w for w in windows if any(norm(seg[i].get("text")) in want for i in w)]

NAME_PAT = {n: re.compile(r"\b" + re.escape(n.split()[0]) + r"\b", re.I)
            for n in ROSTER if n.split()}


def mentioned(text):
    return {n for n, p in NAME_PAT.items() if p.search(text or "")}


summary = {}
for arm, system in (("plain", BASE_SYSTEM), ("reasoned", REASONED_SYSTEM)):
    started = time.time()
    this = replace(params, system_prompt=system)
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
        del TRANSCRIPT[:]
        try:
            out = attribute_batch(client, MODEL, frozen, this, roster,
                                  neighbor_contexts=contexts, source_text=src)
        except Exception as exc:
            print(f"  {arm} window {n}: {type(exc).__name__}", flush=True)
            for i in send:
                key = norm(seg[i].get("text"))
                if key in want and not record.done(arm, want[key]["id"]):
                    g = want[key]
                    record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(),
                               None, False, provenance=f"{arm}|batch_failed")
            continue
        reasons = because_by_index(TRANSCRIPT) if arm == "reasoned" else {}
        for offset, i in enumerate(send):
            key = norm(seg[i].get("text"))
            if key not in want:
                continue
            g = want[key]
            item = out[offset] if offset < len(out) else {}
            speaker = (item or {}).get("speaker")
            # attribute_batch numbers entries with enumerate(frozen_batch),
            # so `n` is ZERO-based and matches the position in `send`. Checked
            # against three_pass_generate rather than assumed - an off-by-one
            # here would silently attach every reason to the wrong line and the
            # mismatch rate would be noise.
            why = reasons.get(offset, "")
            hits = mentioned(why)
            flag = "none"
            if arm == "reasoned" and why:
                if not hits:
                    flag = "no-name"
                elif speaker and any(same_speaker(speaker, h, GROUPS) for h in hits):
                    flag = "confirms"
                else:
                    flag = "MISMATCH"
            record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(),
                       speaker, same_speaker(g["expected_speaker"], speaker, GROUPS),
                       provenance=f"{arm}|flag={flag}", raw=why or None)
        if n % 25 == 0:
            print(f"  {arm} {n}/{len(windows)} ...", flush=True)
    rows = [r for r in record.rows if r["arm"] == arm]
    hit = sum(1 for r in rows if r["correct"])
    summary[arm] = (hit, len(rows), time.time() - started)
    lo, hi = clopper_pearson(hit, max(len(rows), 1))
    print(f"  {arm:9} {hit}/{len(rows)} = {hit/max(len(rows),1)*100:5.1f}%  "
          f"[{lo:.1f}-{hi:.1f}]  {time.time()-started:.0f}s", flush=True)

# ---------------------------------------------------- the actual question
rows = [r for r in record.rows if r["arm"] == "reasoned"]
by_flag = collections.defaultdict(lambda: [0, 0])
for r in rows:
    m = re.search(r"flag=(\S+)", r.get("candidate_provenance") or "")
    f = m.group(1) if m else "none"
    by_flag[f][0] += 1
    by_flag[f][1] += bool(r["correct"])
print(f"\n  {'flag':12}{'n':>6}{'accuracy':>11}   95% CI")
for f in ("confirms", "MISMATCH", "no-name", "none"):
    n, ok = by_flag[f]
    if not n:
        continue
    lo, hi = clopper_pearson(ok, n)
    print(f"  {f:12}{n:6}{ok/n*100:10.1f}%   [{lo:.0f}-{hi:.0f}]")
c, mm = by_flag["confirms"], by_flag["MISMATCH"]
if c[0] and mm[0]:
    sep = c[1] / c[0] * 100 - mm[1] / mm[0] * 100
    cov = mm[0] / len(rows) * 100
    print(f"\n  separation {sep:+.1f} points at {cov:.0f}% coverage")
    print(f"  the bar: the w1/w4 cascade trigger separates 26.6 points at 40% "
          f"coverage\n  and costs two cheap calls; this costs one.")
    print("  " + ("worth routing on" if sep >= 15 and cov <= 45 else
                  "too little separation - the judges' null holds here too"
                  if sep < 5 else
                  "precision without reach" if cov < 10 else
                  "marginal: cheaper than the cascade but weaker"))

out = record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"reasoning_check__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json"),
    contract={"expected_arms": ("plain", "reasoned")})
print("wrote", out)
