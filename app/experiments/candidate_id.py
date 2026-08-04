"""Does returning an opaque candidate ID beat returning a name?

Proposed by external review. The attribution call currently asks the model to
do two jobs: decide who spoke, and serialise that decision as a name. The second
is where invented names (FUTURE_ME, 32 entries), misspellings (SYPHIL for
Sylphy, RENI for Renji) and alias splits (RUDI vs RUDEUS) come from - 33% of
measured errors are in that class.

Giving each roster member an opaque ID and asking for the ID makes those
*unrepresentable*. It does not follow that the underlying attribution errors
disappear: the model may pick a wrong valid ID instead, or abstain. That is
what this measures.

NOT_LISTED is always offered. 15% of gold lines have a true speaker absent from
the roster, and forcing a choice there would convert an honest abstention into
a confident error.

Both arms share model, lines, context, decoding and frozen inputs; only the
output contract differs.
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
from experiments.manifest import ExperimentRecord
from three_pass_generate import build_roster

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = REPO + "/app/"
M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
MODEL = os.environ.get("EXPERIMENT_MODEL", "qwen/qwen3-14b")
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
BASE_URL = "http://localhost:1234/v1"

BOOK = os.environ.get("EXPERIMENT_BOOK", "mushoku16")
GOLD_PATH = APP + os.environ.get(
    "EXPERIMENT_GOLD", "fixtures/attribution_gold_random.json")
gold = json.load(open(GOLD_PATH))
src = open(M + f"inputs/{BOOK}.txt", encoding="utf-8").read()
cp = json.load(open(M + INPUT_RUN + "/" + BOOK + "/result.json.threepass_checkpoint.json"))
seg = cp["segmented"]
roster = [r.upper() for r in build_roster([e for e in (cp.get("named") or []) if e], src)]
AL = [{"RUDEUS", "RUDI"}, {"SYLPHY", "SYLPHIETTE"}]

def same(a, b):
    a, b = (a or "").upper(), (b or "").upper()
    return a == b or any(a in g and b in g for g in AL)

def norm(t): return re.sub(r"\W+", "", t or "").lower()
pos = {norm(e["text"]): i for i, e in enumerate(seg)}
client = OpenAI(base_url=BASE_URL, api_key="local")

IDS = {f"C{n:02d}": name for n, name in enumerate(roster, 1)}
BY_NAME = {v: k for k, v in IDS.items()}

def passage(index):
    before = " ".join((seg[j].get("text") or "") for j in range(max(0, index-4), index))
    after = " ".join((seg[j].get("text") or "") for j in range(index+1, min(len(seg), index+4)))
    return before, after

NAME_SYS = ("You identify who speaks a line of dialogue in a novel. Answer with "
            "the speaker's name in CAPITALS and nothing else. If the passage "
            "does not determine it, answer UNKNOWN.")
ID_SYS = ("You identify who speaks a line of dialogue in a novel. Answer with "
          "exactly one code from the CAST list and nothing else - no name, no "
          "explanation. If the speaker is not in the CAST list, or the passage "
          "does not determine it, answer NOT_LISTED.")

def ask(index, line, as_id):
    before, after = passage(index)
    if as_id:
        cast = "\n".join(f"{code} = {name}" for code, name in IDS.items())
        user = (f"CAST:\n{cast}\nNOT_LISTED = anyone else, or undetermined\n\n"
                f"PASSAGE BEFORE:\n{before}\n\nLINE:\n{line}\n\n"
                f"PASSAGE AFTER:\n{after}\n\nWhich code speaks the LINE?")
    else:
        user = (f"The cast so far: {', '.join(roster)}\n\n"
                f"PASSAGE BEFORE:\n{before}\n\nLINE:\n{line}\n\n"
                f"PASSAGE AFTER:\n{after}\n\nWho speaks the LINE?")
    sys_prompt = ID_SYS if as_id else NAME_SYS
    r = client.chat.completions.create(
        model=MODEL, messages=[{"role": "system", "content": sys_prompt},
                               {"role": "user", "content": user}],
        temperature=0.0, max_tokens=24, extra_body={"reasoning_effort": "none"})
    return (r.choices[0].message.content or "").strip().upper().strip(".'\" "), user

record = ExperimentRecord(
    "candidate_id", REPO, MODEL, BASE_URL,
    GOLD_PATH,
    {"temperature": 0.0, "max_tokens": 24, "reasoning_effort": "none"},
    notes="Free-form name output vs opaque candidate-ID output, same model and "
          "lines. Tests whether removing the naming job removes the errors it "
          "causes, or merely relabels them.")

stats = {}
for arm, as_id in (("name", False), ("id", True)):
    invalid = notlisted = 0
    for n, g in enumerate(gold["entries"], 1):
        i = pos.get(norm(g["line"]))
        if i is None:
            continue
        raw, prompt = ask(i, g["line"], as_id)
        if as_id:
            code = raw.split()[0].strip(":,.") if raw else ""
            if code == "NOT_LISTED":
                answer = "UNKNOWN"; notlisted += 1
            elif code in IDS:
                answer = IDS[code]
            else:
                answer = ""; invalid += 1
        else:
            answer = raw
            if answer == "UNKNOWN":
                notlisted += 1
            elif answer and answer not in roster:
                invalid += 1          # a name outside the supplied cast
        ok = same(answer, g["expected_speaker"])
        record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(),
                   answer or raw, ok, candidates=list(IDS.values()),
                   provenance=arm, prompt=prompt, raw=raw)
        if n % 40 == 0:
            print(f"  {arm} {n}/147 ...", flush=True)
    rows = [r for r in record.rows if r["arm"] == arm]
    hit = sum(1 for r in rows if r["correct"])
    stats[arm] = (hit, len(rows), invalid, notlisted)
    print(f"{arm:6} {hit}/{len(rows)} = {hit/len(rows)*100:.1f}%   "
          f"off-cast/invalid {invalid}   abstained {notlisted}", flush=True)

print(f"\nbaseline, same model open arm: 48.3%")
print("wrote", record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"candidate_id__{MODEL.replace('/','__')}.json"),
    contract={"expected_arms": ("name", "id"),
              "expected_ids": {g["id"] for g in gold["entries"]}}))
