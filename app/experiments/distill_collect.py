"""Collect 70B answers on hard rows, to train a 14B on what it cannot do.

The cost curve settled that the cascade needs a 70B specifically: a 32B scored
-2.2 on mushoku16's routed rows and a 27B +3.0/+3.7 on the new books, against
the 70B's +11.1 to +22.0. So "escalate to something bigger" is false and
"escalate to a 70B" is true, which makes the design a 70B-class commitment.

The obvious way out is to move the capability rather than rent it: have the
70B label the rows a 14B gets wrong, and fine-tune the 14B on those. This
script does the collection half.

WHY THE ROUTED ROWS AND NOT EVERYTHING. Training on rows the cheap model
already gets right teaches it what it knows. The router - two cheap passes
that disagree - isolates about 40% of rows where the 14B is unreliable and the
70B is not, and that disagreement needs no labels to compute. So the training
set builds itself on any segmented book.

WHY UNLABELLED BOOKS. grimgar06 and mushoku18 are segmented and have no gold,
which makes them free training data and, more importantly, keeps the four gold
books clean as an evaluation set. Training on a book and then reporting
accuracy on it would measure memorisation.

WHAT THIS DOES NOT DO. It does not fine-tune anything, and it cannot tell you
whether the distillation will work. It produces JSONL of (prompt, completion)
pairs and the disagreement statistics behind them. A separate training run and
a cross-book evaluation decide the question; this only makes them possible.

Output is one record per routed row:

    {"book", "segment_index", "roster", "context", "line", "teacher",
     "cheap_a", "cheap_b"}

`cheap_a` and `cheap_b` are kept so a later analysis can ask whether the
teacher agreed with either - a teacher that mostly reproduces one of the cheap
answers is not teaching much.
"""
import collections
import json, os, re, sys, time
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")
from openai import OpenAI
from generate_script import LLMGenParams
from three_pass_generate import (attribute_batch, build_roster,
                                 get_deterministic_named_entry)

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"

PHASE = os.environ.get("EXPERIMENT_PHASE", "cheap")     # cheap | teacher
BOOK = os.environ.get("EXPERIMENT_BOOK", "grimgar06")
MODEL = os.environ.get("EXPERIMENT_MODEL", "qwen/qwen3-14b")
TEACHER = os.environ.get("EXPERIMENT_TEACHER", "llama-3.3-70b")
BASE_URL = os.environ.get("EXPERIMENT_BASE_URL", "http://127.0.0.1:8090/v1")
OUT = os.environ.get("EXPERIMENT_OUT",
                     REPO + "/ab_test_runtime/distill")
STATE = os.path.join(OUT, f"routed__{BOOK}.json")
JSONL = os.path.join(OUT, f"train__{BOOK}.jsonl")
BATCH_A, BATCH_B = 25, 50
CONTEXT = 4

cp = json.load(open(M + INPUT_RUN + f"/{BOOK}/result.json.threepass_checkpoint.json"))
seg = cp["segmented"]
src = open(M + f"inputs/{BOOK}.txt", encoding="utf-8").read()
roster = [r.upper() for r in
          build_roster([e for e in (cp.get("named") or []) if e], src)]
client = OpenAI(base_url=BASE_URL, api_key="local")
params = LLMGenParams(max_tokens=12000, context_length=32768, temperature=0.0,
                      attribute_temperature=0.0, top_p=0.8,
                      reasoning_effort="none")
os.makedirs(OUT, exist_ok=True)

# Every non-deterministic spoken segment is a candidate. Unlike the scoring
# harnesses there is no gold to restrict to, which is the point.
targets = [i for i, e in enumerate(seg)
           if e.get("type") != "NARRATOR"
           and get_deterministic_named_entry(e) is None
           and (e.get("text") or "").strip()]
print(f"{BOOK}: {len(seg)} segments, {len(targets)} attributable lines, "
      f"roster {len(roster)}", flush=True)


def run(model, indices, batch, label):
    got, windows = {}, [list(range(s, min(s + batch, len(seg))))
                        for s in range(0, len(seg), batch)]
    wanted = set(indices)
    windows = [w for w in windows if any(i in wanted for i in w)]
    for n, win in enumerate(windows, 1):
        chunk = [i for i in win if get_deterministic_named_entry(seg[i]) is None]
        if not chunk:
            continue
        frozen = [{"type": seg[i]["type"], "text": seg[i]["text"]} for i in chunk]
        ctx = [{"previous_context": seg[i - 1] if i else None,
                "next_context": seg[i + 1] if i + 1 < len(seg) else None}
               for i in chunk]
        try:
            out = attribute_batch(client, model, frozen, params, roster,
                                  neighbor_contexts=ctx, source_text=src)
        except Exception as exc:
            print(f"  {label} window {n}: {type(exc).__name__}", flush=True)
            continue
        for off, i in enumerate(chunk):
            got[i] = ((out[off] or {}).get("speaker") or "").upper() \
                if off < len(out) else ""
        if n % 25 == 0:
            print(f"  {label} {n}/{len(windows)} ...", flush=True)
    return got


if PHASE == "cheap":
    started = time.time()
    a = run(MODEL, targets, BATCH_A, "b25")
    b = run(MODEL, targets, BATCH_B, "b50")
    routed = [i for i in targets if a.get(i) and b.get(i) and a[i] != b[i]]
    json.dump({"book": BOOK, "model": MODEL, "elapsed_s": round(time.time() - started, 1),
               "a": {str(i): a.get(i) for i in targets},
               "b": {str(i): b.get(i) for i in targets},
               "routed": [str(i) for i in routed]},
              open(STATE, "w", encoding="utf-8"), indent=1)
    print(f"\n  disagreement on {len(routed)}/{len(targets)} = "
          f"{len(routed)/max(len(targets),1)*100:.0f}% of lines")
    print(f"  wrote {STATE}")
    raise SystemExit(0)

state = json.load(open(STATE, encoding="utf-8"))
routed = [int(i) for i in state["routed"]]
print(f"  teaching {len(routed)} routed rows with {TEACHER}", flush=True)
taught = run(TEACHER, routed, BATCH_A, "teacher")


def context_of(index):
    lo, hi = max(0, index - CONTEXT), min(len(seg), index + CONTEXT + 1)
    return [{"type": seg[j].get("type"), "text": seg[j].get("text"),
             "target": j == index} for j in range(lo, hi)]


written = 0
agree_a = agree_b = 0
with open(JSONL, "w", encoding="utf-8") as fh:
    for i in routed:
        who = taught.get(i)
        if not who:
            continue
        # A teacher that just reproduces one of the cheap answers is teaching
        # the student something it already produced, so track how often that
        # happens rather than discovering it after a training run.
        agree_a += who == state["a"].get(str(i))
        agree_b += who == state["b"].get(str(i))
        fh.write(json.dumps({
            "book": BOOK, "segment_index": i, "roster": roster,
            "context": context_of(i), "line": seg[i].get("text"),
            "teacher": who,
            "cheap_a": state["a"].get(str(i)),
            "cheap_b": state["b"].get(str(i))}, ensure_ascii=False) + "\n")
        written += 1
print(f"\n  wrote {written} training rows to {JSONL}")
if written:
    print(f"  teacher matched the b25 answer {agree_a/written*100:.0f}% "
          f"and the b50 answer {agree_b/written*100:.0f}% of the time")
    print(f"  the remainder is where the teacher supplies something neither "
          f"cheap pass produced")
