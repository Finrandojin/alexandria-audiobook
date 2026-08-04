"""Conditional selection accuracy: can the 9B pick from a short list?

Roster recall is 85% and pipeline accuracy is 29.9%, so on 55% of lines the
right name was available and not chosen. This asks whether shrinking the choice
set fixes that, and measures the ceiling with an oracle set.

Run on an idle GPU. Temperature 0, so single runs are exact.
"""
import collections
import json, os, re, sys, random, time, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import openai
from openai import OpenAI

# A run against a Thunder instance goes through a forwarded port that can drop
# for seconds at a time; one such drop killed a mushoku16 run two arms in, after
# four minutes of work and with no artifact written. Availability errors only -
# APIStatusError covers 404/429/5xx, and BadRequestError is deliberately not
# caught, since a malformed request will fail identically on every attempt.
RETRYABLE = (openai.APIConnectionError, openai.APITimeoutError,
             openai.InternalServerError, openai.RateLimitError,
             openai.NotFoundError)
MAX_ATTEMPTS = 6
from candidates import build_candidates
from experiments.manifest import ExperimentRecord


def _safe_name(model):
    """Model keys carry a publisher prefix ('microsoft/phi-4'), and a slash in
    a filename silently creates a directory instead of naming the artifact."""
    return model.replace("/", "__")
from three_pass_generate import build_roster

M = (REPO + "/"
     "ab_test_runtime/results/matrix_20260725-115148/")
# The model under test. Only this varies between runs.
MODEL = os.environ.get("EXPERIMENT_MODEL",
                       "qwen3.5-9b-uncensored-hauhaucs-aggressive")
# The frozen inputs - segmentation and the roster derived from it - always come
# from the same source run whatever model is being tested. Deriving them from
# MODEL would compare two models on two different segmentations of the book,
# which measures pass 1 and pass 2 at once and settles neither.
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
# Everything in this investigation was measured on mushoku16. The brief's own
# handoff rule is that conclusions stay scoped to the tested book, so the same
# decomposition has to run on a second one before any ranking generalises.
BOOK = os.environ.get("EXPERIMENT_BOOK", "mushoku16")
GOLD = os.environ.get("EXPERIMENT_GOLD",
                      "fixtures/attribution_gold_random.json")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
GOLD_PATH = REPO + "/app/" + GOLD
gold = json.load(open(GOLD_PATH))
src = open(M + f"inputs/{BOOK}.txt", encoding="utf-8").read()
cp = json.load(open(M + INPUT_RUN + f"/{BOOK}/result.json.threepass_checkpoint.json"))
seg, named = cp["segmented"], [e for e in (cp.get("named") or []) if e]
roster = [r.upper() for r in build_roster(named, src)]
AL = [{n.upper() for n in g} for g in gold.get("aliases", [])]

def same(a, b):
    a, b = (a or "").upper(), (b or "").upper()
    return a == b or any(a in g and b in g for g in AL)

def norm(t): return re.sub(r"\W+", "", t or "").lower()
pos = {norm(e["text"]): i for i, e in enumerate(seg)}
# A gold line whose text repeats in the book cannot be aligned to one position.
# Fifth harness to need this; the manifest validator refuses duplicates, which
# is how the omission surfaced rather than silently double-counting.
_occ = collections.Counter(norm(e.get("text")) for e in seg)
SCOREABLE = [g for g in gold["entries"] if _occ[norm(g["line"])] == 1]
gold_ids = {g["id"] for g in SCOREABLE}
print(f"scoring {len(SCOREABLE)} of {len(gold['entries'])} gold lines "
      f"(unique text)", flush=True)
# Remote runs point this at a Thunder instance's forwarded port. The endpoint
# is recorded in the artifact, so a local and a rented-GPU run can never be
# confused for one another after the fact.
BASE_URL = os.environ.get("EXPERIMENT_BASE_URL", "http://localhost:1234/v1")
# The environment goes in the FILENAME, not just the manifest. Without it a
# cloud run and a local run of the same model and book write the same path, and
# the second silently destroys the first - which nearly cost the local grimgar03
# qwen3-14b artifact on 2026-07-27. Defaults to "local" so existing behaviour is
# only extended, never guessed at.
TAG = os.environ.get("EXPERIMENT_TAG",
                     "local" if "localhost" in BASE_URL or "127.0.0.1"
                     in BASE_URL else "remote")
DECODING = {"temperature": 0.0, "max_tokens": 24, "reasoning_effort": "none"}
client = OpenAI(base_url=BASE_URL, api_key="local")
# On a remote host the local `lms` CLI cannot see the server, so the caller
# supplies the environment it already verified over the control channel.
_env = os.environ.get("EXPERIMENT_ENV")
record = ExperimentRecord(
    "closed_set", REPO, MODEL, BASE_URL, GOLD_PATH, DECODING,
    environment=json.loads(_env) if _env else None,
    notes="Conditional selection accuracy: open roster vs scene candidates vs "
          "true speaker + 4 distractors. Answers whether candidate pruning "
          "can fix the 55-point available-but-not-chosen gap.")

record.enable_checkpoint(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"closed_set__{BOOK}__{_safe_name(MODEL)}__{TAG}.json.ckpt"))

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
    # One fixed policy for every attempt, decided before the first failure:
    # retry only availability failures - the endpoint being briefly absent,
    # rate-limited or erroring - and never a rejection of the request itself.
    # A remote endpoint drop is indistinguishable from a wrong model name at
    # the call site, so 404 is treated as availability: the model is verified
    # loaded before a run starts, and a 404 mid-run means the tunnel went away.
    # This is not a quality mechanism. Decoding is temperature 0 and identical
    # on every attempt, so a retry can only recover a lost request, never
    # resample a bad answer.
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
            delay = min(2 ** attempt, 30)
            print(f"    endpoint unavailable ({type(exc).__name__}), retry "
                  f"{attempt + 1}/{MAX_ATTEMPTS - 1} in {delay}s", flush=True)
            time.sleep(delay)
    raise RuntimeError(
        f"endpoint failed {MAX_ATTEMPTS} consecutive attempts against "
        f"{BASE_URL}; last error {type(last).__name__}: {last}") from last

arms = {}
for arm in ("open", "closed-6", "closed-oracle"):
    correct = available = cond_ok = n = 0
    for g in SCOREABLE:
        i = pos.get(norm(g["line"]))
        if i is None:
            continue
        n += 1
        truth = g["expected_speaker"]
        if record.done(arm, g["id"]):
            continue
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
        got, prompt, raw, retries = ask(g["line"], i, choices)
        ok = same(got, truth)
        record.add(arm, g["id"], g["line"], truth.upper(), got, ok,
                   candidates=[c.upper() for c in choices],
                   provenance=("full_roster" if arm == "open" else
                               "tag+recent+scene" if arm == "closed-6"
                               else "oracle+4_distractors"),
                   prompt=prompt, raw=raw, retries=retries)
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
contract = {"expected_arms": ("open", "closed-6", "closed-oracle"),
            "expected_ids": gold_ids,
            "require_clean_tree": True}
out = record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"closed_set__{BOOK}__{_safe_name(MODEL)}__{TAG}.json"), contract=contract)
print("wrote", out)
