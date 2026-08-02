"""Does telling the model to trust an adjacent speech tag over everything else help?

This one comes straight out of a labelling failure rather than a hypothesis.
Adjudicating grimgar03's keep-assault scene, the wide-context judge got 19 of
19 wrong in a single way: it took names from surrounding prose instead of from
the attribution tag sitting immediately beside the line.

    "We'll do it, too!"
    -> "Mr. Pleasant, the guy who seemed to be the leader of Choco's PARTY, said"
    judged CHOCO

    "You're doing fabulous!..."
    -> "Bri-chan was shouting things like that"
    judged CHOCO

Nine of nineteen went to CHOCO, who speaks none of them. A frontier model with
twelve segments of context lost to a four-segment window because the extra
context supplied more candidate names, not more evidence.

If a judge with the whole scene makes that mistake, the production model at w1
is at least as exposed - and unlike the judge, it cannot be given more context
to fix it, because more context is what caused it. The remedy is a rule, not a
window: an adjacent tag OUTRANKS every other kind of evidence.

ARMS, differing only in the system prompt:

    baseline    the shipped attribution prompt
    tagfirst    the same, plus an explicit precedence rule - if the segment
                immediately before or after names someone with a verb of
                speech, that person is the speaker, whatever else the passage
                suggests

WHERE IT SHOULD AND SHOULD NOT WORK, registered before running. Adjacent tags
are what speech-verb density measures, and the books differ nearly fourfold:
grimgar03 62.5%, mushoku16 27.1%. So a real effect should be LARGER on
grimgar03. A uniform gain across both books would suggest the sentence is doing
something generic rather than exploiting tags, and a gain on mushoku16 alone
would contradict the mechanism outright.

The risk is a model that follows the rule too literally and attributes a line
to whoever is named nearby even when the tag belongs to a different line - the
same failure the rule is meant to prevent, relocated. Errors are therefore
reported split by whether the target actually has an adjacent tag: a gain
confined to tagged rows is the rule working, a loss on untagged rows is it
misfiring.
"""
import collections
import json, os, re, sys, time
from dataclasses import replace
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")
from openai import OpenAI
from default_prompts import load_attribute_prompts
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
MAX_UNATTRIBUTED = float(os.environ.get("EXPERIMENT_MAX_UNATTRIBUTED", "0.25"))
SPEECH_VERB = (r"\b(said|asked|replied|answered|shouted|whispered|muttered|"
               r"called|cried|yelled|groaned|sighed|laughed|nodded|exclaimed|"
               r"bellowed|agreed|told|added|continued|began|offered|roared|"
               r"declared|ordered|screamed|moaned|snorted|murmured)\b")

gold = json.load(open(GOLD_PATH))
src = open(M + f"inputs/{BOOK}.txt", encoding="utf-8").read()
cp = json.load(open(M + INPUT_RUN + f"/{BOOK}/result.json.threepass_checkpoint.json"))
seg = cp["segmented"]
roster = build_roster([e for e in (cp.get("named") or []) if e], src)
GROUPS = alias_groups(gold)


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


_occ = collections.Counter(norm(e.get("text")) for e in seg)
want = {norm(g["line"]): g for g in gold["entries"] if _occ[norm(g["line"])] == 1}
pos = {norm(e.get("text")): i for i, e in enumerate(seg)}


def has_adjacent_tag(index):
    """Is there a speech verb in the NARRATOR segment either side?

    This is the stratifier, not part of the intervention - the model is never
    told which rows have one. It exists so a gain can be located: on tagged
    rows the rule should help, on untagged rows it can only mislead.
    """
    for j in (index - 1, index + 1):
        if 0 <= j < len(seg) and seg[j].get("type") == "NARRATOR":
            if re.search(SPEECH_VERB, (seg[j].get("text") or "").lower()):
                return True
    return False


tagged = {g["id"]: has_adjacent_tag(pos[k]) for k, g in want.items() if k in pos}
share = sum(tagged.values()) / max(len(tagged), 1) * 100
print(f"roster {len(roster)} | {len(want)} lines | {share:.0f}% have an adjacent "
      f"speech tag", flush=True)

BASE_SYSTEM, _ = load_attribute_prompts()
TAGFIRST_SYSTEM = BASE_SYSTEM.rstrip() + (
    "\n\nPRECEDENCE RULE. If the entry immediately before or after a line is "
    "narration that names someone together with a verb of speech - said, "
    "asked, shouted, replied, bellowed, and so on - then that person is the "
    "speaker of the line, and you must answer with them even if other names "
    "appear nearby and even if the surrounding conversation suggests someone "
    "else. Names that appear only as part of a description, a possessive, or "
    "another character's group are NOT evidence: in \"Mr. Pleasant, the leader "
    "of Choco's party, said\", the speaker is Mr. Pleasant, not Choco. Where "
    "no adjacent narration tags a speaker, judge from the conversation as "
    "usual.")

ARMS = {"baseline": BASE_SYSTEM, "tagfirst": TAGFIRST_SYSTEM}
_want = [a.strip() for a in os.environ.get("EXPERIMENT_ARMS", "").split(",") if a.strip()]
if _want:
    ARMS = {a: ARMS[a] for a in _want}

client = OpenAI(base_url=BASE_URL, api_key="local")
params = LLMGenParams(max_tokens=12000, context_length=32768, temperature=0.0,
                      attribute_temperature=0.0, top_p=0.8,
                      reasoning_effort="none")

_env = os.environ.get("EXPERIMENT_ENV")
record = ExperimentRecord(
    "tag_priority", REPO, MODEL, BASE_URL, GOLD_PATH,
    {"temperature": 0.0, "attribute_temperature": 0.0, "max_tokens": 12000,
     "batch": BATCH, "width": 1},
    environment=json.loads(_env) if _env else None,
    notes="A precedence rule telling the model an adjacent speech tag outranks "
          "all other evidence. Derived from a labelling failure: a frontier "
          "judge with 12 segments got 19 of 19 wrong in one scene by taking "
          "names from surrounding prose rather than the tag beside the line.")
record.enable_checkpoint(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"tag_priority__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json.ckpt"))

windows = [list(range(s, min(s + BATCH, len(seg)))) for s in range(0, len(seg), BATCH)]
windows = [w for w in windows if any(norm(seg[i].get("text")) in want for i in w)]

summary = {}
for arm, system in ARMS.items():
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
        for offset, i in enumerate(send):
            key = norm(seg[i].get("text"))
            if key not in want:
                continue
            g = want[key]
            speaker = (out[offset] or {}).get("speaker") if offset < len(out) else None
            record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(),
                       speaker, same_speaker(g["expected_speaker"], speaker, GROUPS),
                       provenance=f"{arm}|tagged={tagged.get(g['id'])}")
        if n % 25 == 0:
            print(f"  {arm} {n}/{len(windows)} ...", flush=True)
    rows = [r for r in record.rows if r["arm"] == arm]
    hit = sum(1 for r in rows if r["correct"])
    unatt = sum(1 for r in rows if r["predicted"] is None)
    summary[arm] = (hit, len(rows), unatt, time.time() - started)
    lo, hi = clopper_pearson(hit, max(len(rows), 1))
    print(f"  {arm:9} {hit}/{len(rows)} = {hit/max(len(rows),1)*100:5.1f}%  "
          f"[{lo:.1f}-{hi:.1f}]  {unatt} unattributed  "
          f"{time.time()-started:.0f}s", flush=True)

for arm, (h, n, u, _) in summary.items():
    if n and u / n > MAX_UNATTRIBUTED:
        raise SystemExit(f"refusing to write: {arm} left {u}/{n} unattributed")

if len(summary) == 2:
    ans = {a: {r["id"]: r["correct"] for r in record.rows if r["arm"] == a}
           for a in ARMS}
    p, x, y, n = paired(ans["baseline"], ans["tagfirst"])
    h0, n0 = summary["baseline"][0], summary["baseline"][1]
    h1, n1 = summary["tagfirst"][0], summary["tagfirst"][1]
    print(f"\n  tagfirst vs baseline: {(h1/n1 - h0/n0)*100:+.1f} points  "
          f"+{y}/-{x}  p={p:.4g}")
    print(f"\n  {'rows':22} {'baseline':>9} {'tagfirst':>9} {'delta':>7}   n")
    for flag, label in ((True, "with an adjacent tag"), (False, "without one")):
        ids = [i for i in ans["baseline"] if tagged.get(i) is flag]
        if not ids:
            continue
        b = sum(1 for i in ids if ans["baseline"][i]) / len(ids) * 100
        t = sum(1 for i in ids if ans["tagfirst"][i]) / len(ids) * 100
        print(f"  {label:22} {b:8.1f}% {t:8.1f}% {t-b:+7.1f}   {len(ids)}")
    print("\n  A gain confined to tagged rows is the rule working. A loss on "
          "untagged\n  rows is it misfiring - the model reaching for a nearby "
          "name where no tag\n  licenses it, which is the failure the rule was "
          "written to prevent.")

out = record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"tag_priority__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json"),
    contract={"expected_arms": tuple(ARMS)})
print("wrote", out)
