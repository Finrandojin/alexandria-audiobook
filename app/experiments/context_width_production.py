"""Does the w4 context gain survive the PRODUCTION path?

The diagnostic sweep put w4 at +6.2 points over w1 (p=0.022) on grimgar03, and
production pass 2 supplies w1. The reviewer's answer to that was: gate it first,
because `because` looked like +10.8 in a diagnostic harness and reversed to -7.2
through `attribute_batch`. This is that gate.

Both arms run through `attribute_batch` with the shipping prompt, batched at 25,
identical in every respect except the width of the neighbour text. Production
passes `{"previous_context": seg[i-1], "next_context": seg[i+1]}`; the w4 arm
passes the same two keys with four segments merged into each. Because
`attribute_batch` spreads `neighbor_contexts[i]` straight into the batch JSON,
**the prompt contract is unchanged** - the model sees the same fields, with more
text in them. That isolation is what makes this a width test rather than a
prompt test, and it is the distinction that made `thinking` a fair experiment
and `because` an unfair one.

Both books, because the diagnostic result is grimgar03 only and every
single-book result in this investigation has needed a second book. Per the
review: if Grimgar reproduces and Mushoku does not materially regress, w4 is
ready for a guarded production switch.

Note on what this does NOT test: choosing width per line from the true
speaker's mention distance is an ORACLE policy using information unavailable in
production. The diagnostic stratification suggested it; bounding it needs a
separate oracle-adaptive arm, and building it needs a detector that sees only
roster names, tags and scene state.
"""
import collections
import json, os, re, sys, time
from dataclasses import replace
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")
import openai
from openai import OpenAI
from experiments.manifest import ExperimentRecord
from experiments.stats import paired
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
BASE_URL = os.environ.get("EXPERIMENT_BASE_URL", "http://localhost:1234/v1")
TAG = os.environ.get("EXPERIMENT_TAG",
                     "local" if "localhost" in BASE_URL or "127.0.0.1" in BASE_URL
                     else "remote")
BATCH = 25
WIDTHS = [int(w) for w in os.environ.get("EXPERIMENT_WIDTHS", "1,4").split(",")]

gold = json.load(open(GOLD_PATH))
src = open(M + f"inputs/{BOOK}.txt", encoding="utf-8").read()
cp = json.load(open(M + INPUT_RUN + f"/{BOOK}/result.json.threepass_checkpoint.json"))
seg = cp["segmented"]
roster = build_roster([e for e in (cp.get("named") or []) if e], src)
AL = [{n.upper() for n in g} for g in gold.get("aliases", [])]


def same(a, b):
    a, b = (a or "").upper(), (b or "").upper()
    return a == b or any(a in g and b in g for g in AL)


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


_occ = collections.Counter(norm(e.get("text")) for e in seg)
want = {norm(g["line"]): g for g in gold["entries"]
        if _occ[norm(g["line"])] == 1}
print(f"roster {len(roster)} | scoring {len(want)} unambiguous lines | widths {WIDTHS}",
      flush=True)

client = OpenAI(base_url=BASE_URL, api_key="local")
params = LLMGenParams(max_tokens=12000, context_length=32768, temperature=0.0,
                      attribute_temperature=0.0, top_p=0.8,
                      reasoning_effort="none")


def neighbours(index, width):
    """Production's two context keys, with `width` segments merged into each.

    Returning the same key names matters: attribute_batch spreads this dict into
    the batch JSON, so adding keys would change the output contract the model is
    being asked to satisfy, and the arm would no longer isolate width.
    """
    if width <= 1:
        return {"previous_context": seg[index - 1] if index else None,
                "next_context": (seg[index + 1] if index + 1 < len(seg) else None)}
    lo, hi = max(0, index - width), min(len(seg), index + 1 + width)
    prev_txt = " ".join((seg[j].get("text") or "") for j in range(lo, index))
    next_txt = " ".join((seg[j].get("text") or "") for j in range(index + 1, hi))
    return {
        "previous_context": ({"type": "CONTEXT", "text": prev_txt} if prev_txt else None),
        "next_context": ({"type": "CONTEXT", "text": next_txt} if next_txt else None),
    }


_env = os.environ.get("EXPERIMENT_ENV")
record = ExperimentRecord(
    "context_width_production", REPO, MODEL, BASE_URL, GOLD_PATH,
    {"temperature": 0.0, "attribute_temperature": 0.0, "max_tokens": 12000,
     "batch": BATCH, "widths": WIDTHS},
    environment=json.loads(_env) if _env else None,
    notes="Production-path gate for the diagnostic w4 result (+6.2, p=0.022). "
          "Same attribute_batch call, same shipping prompt, same batch size; "
          "only the neighbour-context width differs.")
record.enable_checkpoint(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"context_width_production__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json.ckpt"))

windows = [list(range(s, min(s + BATCH, len(seg)))) for s in range(0, len(seg), BATCH)]
windows = [w for w in windows if any(norm(seg[i].get("text")) in want for i in w)]

summary = {}
for width in WIDTHS:
    arm = f"w{width}"
    started, chars = time.time(), []
    for n, window in enumerate(windows, 1):
        send = [i for i in window if get_deterministic_named_entry(seg[i]) is None]
        if not send or not any(norm(seg[i].get("text")) in want for i in send):
            continue
        pending = [i for i in send if norm(seg[i].get("text")) in want
                   and not record.done(arm, want[norm(seg[i].get("text"))]["id"])]
        if not pending:
            continue
        frozen = [{"type": seg[i]["type"], "text": seg[i]["text"]} for i in send]
        contexts = [neighbours(i, width) for i in send]
        chars.append(sum(len((c.get("previous_context") or {}).get("text", "")) +
                         len((c.get("next_context") or {}).get("text", ""))
                         for c in contexts) / max(len(contexts), 1))
        try:
            out = attribute_batch(client, MODEL, frozen, params, roster,
                                  neighbor_contexts=contexts, source_text=src)
        except Exception as exc:
            # A batch that exhausts its retries is a PRODUCTION OUTCOME, not a
            # missing measurement: the pipeline emits no speaker for those
            # entries. The first version of this harness skipped them, which
            # silently removed from w4's denominator exactly the rows where the
            # wider context made it fail - inflating w4 from 63.5% to 69.9% and
            # leaving the two arms scoring different id sets. Record them as
            # failures so the comparison stays paired and honest.
            print(f"  {arm} window {n}: {type(exc).__name__}: {exc}", flush=True)
            for entry_index in send:
                key = norm(seg[entry_index].get("text"))
                if key not in want or record.done(arm, want[key]["id"]):
                    continue
                g = want[key]
                record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(),
                           None, False,
                           provenance=f"{arm}|batch_failed={type(exc).__name__}")
            continue
        for position, entry_index in enumerate(send):
            key = norm(seg[entry_index].get("text"))
            if key not in want:
                continue
            g = want[key]
            speaker = (out[position] or {}).get("speaker") if position < len(out) else None
            record.add(arm, g["id"], g["line"], g["expected_speaker"].upper(),
                       speaker, same(speaker, g["expected_speaker"]),
                       provenance=arm)
        if n % 20 == 0:
            print(f"  {arm} {n}/{len(windows)} ...", flush=True)
    rows = [r for r in record.rows if r["arm"] == arm]
    hit = sum(1 for r in rows if r["correct"])
    failed = sum(1 for r in rows if r["predicted"] is None)
    summary[arm] = (hit, len(rows), sum(chars) / max(len(chars), 1),
                    time.time() - started, failed)
    print(f"{arm:5} {hit}/{len(rows)} = {hit/max(len(rows),1)*100:5.1f}%   "
          f"mean neighbour chars {summary[arm][2]:.0f}   "
          f"{failed} unattributed   {summary[arm][3]:.0f}s", flush=True)

base = summary[f"w{WIDTHS[0]}"]
print(f"\n  arm   accuracy   vs w{WIDTHS[0]}   neighbour chars   unattributed   seconds")
for width in WIDTHS:
    h, n, c, t, f = summary[f"w{width}"]
    print(f"  w{width:<4} {h/n*100:7.1f}%  {(h-base[0])/n*100:+7.1f}   "
          f"{c:15.0f}   {f:12}   {t:7.0f}")

# Paired transitions, on the same rows, with unattributed rows counted as
# wrong. A width that buys accuracy by failing more batches is not a win.
w_lo, w_hi = f"w{WIDTHS[0]}", f"w{WIDTHS[-1]}"
_a = {r["id"]: r["correct"] for r in record.rows if r["arm"] == w_lo}
_b = {r["id"]: r["correct"] for r in record.rows if r["arm"] == w_hi}
_p, _x, _y, _n = paired(_a, _b)
print(f"\n  {w_hi} vs {w_lo}: rescues {_y}, breaks {_x}, exact McNemar "
      f"p={_p:.4g} over {_n} paired rows")
print("\n  A diagnostic gain that does not reproduce here is a harness artefact, "
      "which is\n  exactly what happened to `because` (+10.8 diagnostic, -7.2 production).")

# Scoring an exhausted batch as a failure is right for SPORADIC failures, where
# not attributing a line is a real production outcome. It is wrong when every
# batch fails for an environmental reason: on the A6000 a missing prompt file
# made all 800 rows fail, and the harness happily wrote an artifact reading
# 0.0% for both arms that passed validation - same ids, clean tree, arms
# present. A measurement that cannot distinguish "w4 does not help" from "the
# model was never called" is worse than no measurement.
MAX_UNATTRIBUTED = float(os.environ.get("EXPERIMENT_MAX_UNATTRIBUTED", "0.25"))
for width in WIDTHS:
    h, n, c, t, f = summary[f"w{width}"]
    if n and f / n > MAX_UNATTRIBUTED:
        raise SystemExit(
            f"refusing to write: w{width} left {f}/{n} rows unattributed "
            f"({f/n*100:.0f}% > {MAX_UNATTRIBUTED*100:.0f}%). That is an "
            f"environment failure, not a width result - check the endpoint, the "
            f"prompt files and the context length before believing any number "
            f"above.")

out = record.write(os.path.join(
    REPO, "ab_test_runtime", "experiments",
    f"context_width_production__{BOOK}__{MODEL.replace('/', '__')}__{TAG}.json"),
    contract={"expected_arms": tuple(f"w{w}" for w in WIDTHS),
              "require_clean_tree": True})
print("wrote", out)
