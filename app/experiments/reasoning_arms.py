"""Is the model's reasoning accounted for anywhere in pass 2?

Three levels of the pipeline currently forbid it: the prompt says "no
explanations", the schema is {n, speaker} with no field for why, and every call
sets reasoning_effort=none. Meanwhile the gold set demands a `reasoning` field
from every judge - we require justification to trust a judgement, then instruct
the pipeline making the same judgement not to explain itself.

A partial probe from 2026-07-24 hints this matters: on the 78 gold lines both
arms covered, thinking-on scored 47.4% against thinking-off's 25.6%, 24 repairs
to 7 regressions, p=0.0033. That probe was never scored for accuracy at the time
- only speed and structural quality - and thinking was switched off on the
strength of a "run-to-run noise floor" that later proved to be GPU contention.

Arms, batched at 25 to mirror production, everything else identical:

  baseline   {n, speaker}, no explanations, reasoning_effort=none  (what ships)
  because    {n, speaker, because}, one clause per line, effort=none
  scaffold   {n, tag, addressed, previous_speaker, speaker} - the questions a
             human judge works through, asked explicitly and in order
  thinking   {n, speaker}, reasoning_effort unset so the model may think
  scaffold_thinking   both - the questions AND permission to reason

The last two arms cross the only two factors that matter, so the result can
distinguish four outcomes rather than two:

  thinking alone wins        - what it needed was room, not direction
  scaffold alone wins        - it needed direction, and thinking is wasted cost
  both together win          - they are complementary; pay for both
  scaffold_thinking is worse - the questions constrain a model that reasons
                               better unprompted, which is worth knowing before
                               anyone writes a prompt like this into production

`because` is the cheap version of the idea: a clause, not a monologue, and a
field that can be inspected, validated and used as a confidence signal. The
candidate-ID experiment removed expressiveness from the output and lost 13.6
points; this adds some back.
"""
import collections
import json, os, re, sys, time
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")
from openai import OpenAI
from experiments.manifest import ExperimentRecord
from three_pass_generate import build_roster, get_deterministic_named_entry

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
APP = REPO + "/app/"
M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
MODEL = os.environ.get("EXPERIMENT_MODEL", "qwen/qwen3-14b")
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
BASE_URL = "http://localhost:1234/v1"
BATCH = 25
# All five arms were measured on mushoku16 only. Every one of the four negatives
# is therefore a single-book result, and closed-6 has already reversed between
# books - so the arms have to be re-run on grimgar03 before any of them is
# treated as settled. BOOK reaches the output filename deliberately: without it
# a second book would overwrite the first book's artifact in place.
BOOK = os.environ.get("EXPERIMENT_BOOK", "mushoku16")
GOLD = os.environ.get("EXPERIMENT_GOLD", "fixtures/attribution_gold_random.json")
GOLD_PATH = APP + GOLD

gold = json.load(open(GOLD_PATH))
src = open(M + f"inputs/{BOOK}.txt", encoding="utf-8").read()
cp = json.load(open(M + INPUT_RUN + f"/{BOOK}/result.json.threepass_checkpoint.json"))
seg = cp["segmented"]
roster = build_roster([e for e in (cp.get("named") or []) if e], src)
AL = [{n.upper() for n in g} for g in gold.get("aliases", [])]

def same(a, b):
    a, b = (a or "").upper(), (b or "").upper()
    return a == b or any(a in g and b in g for g in AL)

def norm(t): return re.sub(r"\W+", "", t or "").lower()
# A gold line whose text repeats cannot be aligned to one position, so scoring
# it at each occurrence counts one judgement twice. This is the same defect
# already fixed in build_scoring_sheet, roster_warmup and two_by_two - the
# fourth harness to need it, which is why the manifest validator now refuses to
# write an artifact containing duplicate (arm, id) pairs.
_occurrences = collections.Counter(norm(e.get("text")) for e in seg)
want = {norm(g["line"]): g for g in gold["entries"]
        if _occurrences[norm(g["line"])] == 1}
print(f"scoring {len(want)} gold lines with unambiguous text", flush=True)
pos = {norm(e["text"]): i for i, e in enumerate(seg)}
client = OpenAI(base_url=BASE_URL, api_key="local")

BASE_SYS = ("You assign speaker names to already-segmented script entries. "
            "Output ONLY a valid JSON array.\n"
            'Return {"n": <same index>, "speaker": "NAME"} for every entry. '
            "NARRATOR entries take the speaker NARRATOR. Use UNKNOWN when the "
            "passage does not determine the speaker.")
# The questions a human judge actually works through, made explicit. Each field
# targets a failure mode measured on this corpus:
#   tag              - 6.8% of errors had a speech-verb tag nearby and missed it
#   addressed        - 13 errors named the person being spoken TO, not the speaker
#   previous_speaker - exchanges alternate; the model has no turn-taking state
# Answering them in order forces the same steps before the speaker is chosen,
# rather than hoping an unguided model performs them internally.
SCAFFOLD_SYS = (
    "You assign speaker names to already-segmented script entries. "
    "Output ONLY a valid JSON array.\n"
    "For each entry, answer these in order, then decide:\n"
    '  "n"                - echo the entry index unchanged\n'
    '  "tag"              - the dialogue tag attributing this line, verbatim, '
    "or null if there is none nearby\n"
    '  "addressed"        - who is being spoken TO, if a name inside the line '
    "is a form of address, else null\n"
    '  "previous_speaker" - who spoke the previous spoken line, or null\n'
    '  "speaker"          - who says THIS line\n'
    "A name inside the line is usually the person being addressed, not the "
    "speaker. Consecutive spoken lines usually alternate between two people. "
    "NARRATOR entries take the speaker NARRATOR. Use UNKNOWN when the passage "
    "does not determine it.")

BECAUSE_SYS = ("You assign speaker names to already-segmented script entries. "
               "Output ONLY a valid JSON array.\n"
               'Return {"n": <same index>, "speaker": "NAME", "because": '
               '"<one short clause>"} for every entry. State the evidence you '
               "used - a dialogue tag, who was addressed, whose turn it is. "
               "NARRATOR entries take the speaker NARRATOR. Use UNKNOWN when "
               "the passage does not determine the speaker.")

def ask(window, arm):
    payload = [{"n": i, "type": seg[j]["type"], "text": seg[j]["text"],
                "previous_context": seg[j-1]["text"] if j else None,
                "next_context": seg[j+1]["text"] if j+1 < len(seg) else None}
               for i, j in enumerate(window)]
    user = (f"ESTABLISHED ROSTER: {', '.join(roster) or '(none yet)'}\n\n"
            f"Assign a speaker to each entry:\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}")
    # Two independent factors: whether the model may think, and whether it is
    # asked the judge's questions. Crossing them is the point - scaffold may
    # help a model that cannot think, and may equally constrain one that can.
    thinks = arm in ("thinking", "scaffold_thinking")
    scaffolded = arm in ("scaffold", "scaffold_thinking")
    extra = {} if thinks else {"reasoning_effort": "none"}
    system = (SCAFFOLD_SYS if scaffolded
              else BECAUSE_SYS if arm == "because" else BASE_SYS)
    response = client.chat.completions.create(
        model=MODEL, temperature=0.0, max_tokens=8000,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        extra_body=extra)
    text = (response.choices[0].message.content or "")
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    thought = getattr(details, "reasoning_tokens", None) if details else None
    match = re.search(r"\[.*\]", text, re.S)
    try:
        return (json.loads(match.group(0)) if match else []), text, thought
    except json.JSONDecodeError:
        return [], text, thought

record = ExperimentRecord(
    "reasoning_arms", REPO, MODEL, BASE_URL, GOLD_PATH,
    {"temperature": 0.0, "max_tokens": 8000, "batch": BATCH},
    notes="baseline vs a 'because' field vs thinking-on, batched at 25.")

windows = [list(range(s, min(s + BATCH, len(seg)))) for s in range(0, len(seg), BATCH)]
windows = [w for w in windows if any(norm(seg[i].get("text")) in want for i in w)]
elapsed = {}
# Cheap arms first, so a stall in a thinking arm still leaves usable results.
for arm in ("baseline", "because", "scaffold", "thinking", "scaffold_thinking"):
    started = time.time()
    thought_total = 0
    for n, window in enumerate(windows, 1):
        send = [i for i in window if get_deterministic_named_entry(seg[i]) is None]
        if not send:
            continue
        answers, raw, thought = ask(send, arm)
        thought_total += thought or 0
        by_index = {a.get("n"): a for a in answers if isinstance(a, dict)}
        if answers and None in by_index:
            # A response that drops "n" silently misaligns every row: this
            # scored the scaffold arm at 7.7% when its answers were correct.
            print(f"  WARNING {arm}: {sum(1 for a in answers if a.get('n') is None)}"
                  f"/{len(answers)} responses omitted the index", flush=True)
        for position, entry_index in enumerate(send):
            key = norm(seg[entry_index].get("text"))
            if key not in want:
                continue
            item = by_index.get(position) or {}
            speaker = item.get("speaker")
            g = want[key]
            record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(),
                       speaker, same(speaker, g["expected_speaker"]),
                       provenance=arm,
                       raw=json.dumps({k: item.get(k) for k in
                                       ("because", "tag", "addressed",
                                        "previous_speaker")
                                       if item.get(k) is not None}
                                      | {"reasoning_tokens": thought},
                                      ensure_ascii=False))
        if n % 20 == 0:
            print(f"  {arm} {n}/{len(windows)} ...", flush=True)
    elapsed[arm] = round(time.time() - started, 1)
    rows = [r for r in record.rows if r["arm"] == arm]
    hit = sum(1 for r in rows if r["correct"])
    print(f"{arm:9} {hit}/{len(rows)} = {hit/max(len(rows),1)*100:.1f}%   "
          f"{elapsed[arm]:.0f}s   reasoning_tokens={thought_total}", flush=True)

record.meta["elapsed_by_arm_s"] = elapsed
print("\nfactorial view (accuracy, seconds):")
print(f"  {'':18} {'no scaffold':>14} {'scaffold':>12}")
for label, plain, scaf in (("thinking off", "baseline", "scaffold"),
                           ("thinking on ", "thinking", "scaffold_thinking")):
    cells = []
    for name in (plain, scaf):
        rows = [r for r in record.rows if r["arm"] == name]
        hit = sum(1 for r in rows if r["correct"])
        cells.append(f"{hit/max(len(rows),1)*100:5.1f}% {elapsed.get(name,0):5.0f}s")
    print(f"  {label:18} {cells[0]:>14} {cells[1]:>12}")
print("\nwrote", record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"reasoning_arms__{BOOK}__{MODEL.replace('/','__')}.json"),
    contract={"expected_arms": ("baseline", "because", "scaffold",
                               "thinking", "scaffold_thinking")}))
