"""Conditional selection accuracy: can the 9B pick from a short list?

Roster recall is 85% and pipeline accuracy is 29.9%, so on 55% of lines the
right name was available and not chosen. This asks whether shrinking the choice
set fixes that, and measures the ceiling with an oracle set.

Run on an idle GPU. Temperature 0, so single runs are exact.
"""
import json, os, re, sys, random, collections
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")
from openai import OpenAI
from candidates import build_candidates
from experiments.manifest import ExperimentRecord


def _safe_name(model):
    """Model keys carry a publisher prefix ('microsoft/phi-4'), and a slash in
    a filename silently creates a directory instead of naming the artifact."""
    return model.replace("/", "__")
from three_pass_generate import build_roster

M = ("/home/fakemitch/pinokio/api/alexandria-audiobook2.git/"
     "ab_test_runtime/results/matrix_20260725-115148/")
MODEL = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
gold = json.load(open("/home/fakemitch/pinokio/api/alexandria-audiobook2.git/"
                      "app/fixtures/attribution_gold_random.json"))
src = open(M + "inputs/mushoku16.txt", encoding="utf-8").read()
cp = json.load(open(M + MODEL + "/mushoku16/result.json.threepass_checkpoint.json"))
seg, named = cp["segmented"], [e for e in (cp.get("named") or []) if e]
roster = [r.upper() for r in build_roster(named, src)]
AL = [{"RUDEUS", "RUDI"}, {"SYLPHY", "SYLPHIETTE"}]

def same(a, b):
    a, b = (a or "").upper(), (b or "").upper()
    return a == b or any(a in g and b in g for g in AL)

def norm(t): return re.sub(r"\W+", "", t or "").lower()
pos = {norm(e["text"]): i for i, e in enumerate(seg)}
BASE_URL = "http://localhost:1234/v1"
REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
GOLD_PATH = REPO + "/app/fixtures/attribution_gold_random.json"
DECODING = {"temperature": 0.0, "max_tokens": 24, "reasoning_effort": "none"}
client = OpenAI(base_url=BASE_URL, api_key="local")
record = ExperimentRecord(
    "closed_set", REPO, MODEL, BASE_URL, GOLD_PATH, DECODING,
    notes="Conditional selection accuracy: open roster vs scene candidates vs "
          "true speaker + 4 distractors. Answers whether candidate pruning "
          "can fix the 55-point available-but-not-chosen gap.")

SYSTEM = ("You identify who speaks a line of dialogue in a novel. Answer with "
          "the speaker's name in CAPITALS and nothing else. If the passage "
          "does not determine it, answer UNKNOWN.")

def ask(line, index, choices):
    """Returns (answer, prompt, raw) so the manifest can record all three."""
    before = " ".join((seg[j].get("text") or "")
                      for j in range(max(0, index - 4), index))
    after = " ".join((seg[j].get("text") or "")
                     for j in range(index + 1, min(len(seg), index + 4)))
    options = ("\nThe speaker is one of: " + ", ".join(choices + ["UNKNOWN"])
               if choices else "")
    user = (f"PASSAGE BEFORE:\n{before}\n\nLINE:\n{line}\n\n"
            f"PASSAGE AFTER:\n{after}\n{options}\n\nWho speaks the LINE?")
    r = client.chat.completions.create(
        model=MODEL, messages=[{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": user}],
        temperature=0.0, max_tokens=24,
        extra_body={"reasoning_effort": "none"})
    raw = (r.choices[0].message.content or "")
    return raw.strip().upper().strip(".'\" "), user, raw

arms = {}
for arm in ("open", "closed-6", "closed-oracle"):
    correct = available = cond_ok = n = 0
    for g in gold["entries"]:
        i = pos.get(norm(g["line"]))
        if i is None:
            continue
        n += 1
        truth = g["expected_speaker"]
        if arm == "open":
            choices = roster
        elif arm == "closed-6":
            choices = build_candidates(seg, named, i, roster)[:6]
        else:
            distractors = [x for x in roster if not same(x, truth)]
            choices = [truth] + random.Random(i).sample(
                distractors, min(4, len(distractors)))
            random.Random(i + 1).shuffle(choices)
        here = any(same(c, truth) for c in choices)
        available += here
        got, prompt, raw = ask(g["line"], i, choices)
        ok = same(got, truth)
        record.add(arm, g["id"], g["line"], truth.upper(), got, ok,
                   candidates=[c.upper() for c in choices],
                   provenance=("full_roster" if arm == "open" else
                               "tag+recent+scene" if arm == "closed-6"
                               else "oracle+4_distractors"),
                   prompt=prompt, raw=raw)
        correct += ok
        cond_ok += ok and here
        if n % 25 == 0:
            print(f"  {arm} {n}/147 ...", flush=True)
    arms[arm] = (correct, cond_ok, available, n)
    print(f"{arm:14} accuracy {correct}/{n} = {correct/n*100:.1f}%   "
          f"recall {available/n*100:.1f}%   "
          f"conditional {cond_ok}/{available} = {cond_ok/max(available,1)*100:.1f}%",
          flush=True)
print("\nbaseline (shipped batched pipeline): 44/147 = 29.9%")
# Declare what the run must contain so an artifact that silently drops an arm
# or half its lines is refused rather than validated on its own arithmetic.
contract = {"expected_arms": ("open", "closed-6", "closed-oracle"),
            "expected_ids": {g["id"] for g in gold["entries"]},
            "require_clean_tree": True}
out = record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"closed_set__{_safe_name(MODEL)}.json"), contract=contract)
print("wrote", out)
