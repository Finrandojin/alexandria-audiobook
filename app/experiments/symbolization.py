"""Does hiding character NAMES make attribution better or worse?

SPC (EMNLP 2023 Findings) reports 94.6% in-domain accuracy on Chinese novels
with a fine-tuned RoBERTa-large, beating GPT-3.5-turbo zero-shot. A core piece
of its method is SYMBOLIZATION: every character name in the passage and the
candidate list is replaced by [C0], [C1] ... before the model sees it, so the
model cannot lean on name priors or on explicit "Tom said" patterns and has to
reason from who is present and who is addressing whom.

That is a prompt-level change, testable in this pipeline without adopting their
model, and it targets a defect already measured here: `offbyone_turns` found
57% of wrong predictions name a character who does not speak anywhere within
+-3 lines - the model reaching for someone not in the room.

TWO ARMS over the same rows, same model, same contexts:

    names      roster and context exactly as production sends them
    symbols    every roster name replaced by [C0].. throughout the roster, the
               target line and both neighbour contexts; the answer is mapped
               back for scoring

WHAT EACH OUTCOME WOULD MEAN.

  symbols better   the model is being distracted by names - leaning on
                   familiarity or on surface patterns rather than structure -
                   and symbolization is a cheap production change
  symbols worse    names carry real signal here (gender, honorifics, who a
                   name suggests), and stripping them costs more than the
                   distraction it removes. SPC's gain would then be about
                   their training regime rather than the representation
  no difference    the representation is not what limits this pipeline, and
                   the SPC result comes from fine-tuning rather than from
                   symbolization

A CONFOUND STATED UP FRONT. SPC symbolizes AND fine-tunes on symbolized data.
Applying symbolization to a model that never saw it in training tests only half
their method, and a loss here does not refute the technique - it would show it
needs the matching training, which is exactly what their fine-tuned encoder has
and this 14B does not.
"""
import argparse, collections, json, os, re, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = REPO + "/app/"
sys.path.insert(0, APP)
from openai import OpenAI
from experiments.scoring import alias_groups, same_speaker
from experiments.stats import clopper_pearson, paired
from generate_script import LLMGenParams
from three_pass_generate import attribute_batch, build_roster, get_deterministic_named_entry

M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
SPECIAL = {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE"}
BATCH = 25


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


def build_symbol_map(roster, groups):
    """One symbol per character, shared across that character's aliases.

    Aliases must map to the SAME symbol or symbolization would split one
    character into several candidates and make the task harder for a reason
    that has nothing to do with the technique.
    """
    canon, symbols = {}, {}
    n = 0
    for name in sorted(roster, key=lambda s: -len(s)):
        hit = None
        for other, sym in canon.items():
            if same_speaker(other, name, groups):
                hit = sym
                break
        if hit is None:
            hit = f"[C{n}]"
            n += 1
        canon[name] = hit
        symbols[name] = hit
    return symbols


def symbolize(text, symbols):
    """Replace every roster name occurrence, longest first so 'MR. TALL' wins
    over 'TALL'."""
    out = text or ""
    for name in sorted(symbols, key=len, reverse=True):
        if len(name) < 3:
            continue
        out = re.sub(re.escape(name), symbols[name], out, flags=re.I)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--book", default="grimgar03")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--model", default="qwen/qwen3-14b")
    ap.add_argument("--base_url", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--out", default=REPO + "/ab_test_runtime/experiments/symbolization.json")
    args = ap.parse_args()

    gold = json.load(open(APP + f"fixtures/attribution_gold_{args.book}.json"))
    src = open(M + f"inputs/{args.book}.txt", encoding="utf-8").read()
    cp = json.load(open(M + INPUT_RUN + f"/{args.book}/result.json.threepass_checkpoint.json"))
    seg = cp["segmented"]
    groups = alias_groups(gold)
    roster = [r.upper() for r in build_roster([e for e in (cp.get("named") or []) if e], src)]
    roster = sorted(set(roster) | {n.upper() for n in
                                   gold.get("roster_additions", {}).get("names", [])})
    occ = collections.Counter(norm(e.get("text")) for e in seg)
    want = {norm(g["line"]): g for g in gold["entries"]
            if occ[norm(g["line"])] == 1
            and g["expected_speaker"].upper() not in SPECIAL}

    items = []
    for i, e in enumerate(seg):
        key = norm(e.get("text"))
        if key not in want or get_deterministic_named_entry(e) is not None:
            continue
        items.append({"line": e["text"], "truth": want[key]["expected_speaker"].upper(),
                      "prev": (seg[i-1].get("text") or "") if i else "",
                      "next": (seg[i+1].get("text") or "") if i+1 < len(seg) else ""})
        if len(items) >= args.limit:
            break

    symbols = build_symbol_map(roster, groups)
    distinct = len(set(symbols.values()))
    print(f"{args.book}: {len(items)} lines, {len(roster)} roster names -> "
          f"{distinct} symbols\n")

    client = OpenAI(base_url=args.base_url, api_key="local")
    params = LLMGenParams(max_tokens=2000, context_length=32768, temperature=0.0,
                          attribute_temperature=0.0, top_p=0.8,
                          reasoning_effort="none")
    answers = {}
    for arm in ("names", "symbols"):
        started, hit, n = time.time(), 0, 0
        per_row = {}
        for s in range(0, len(items), BATCH):
            block = items[s:s + BATCH]
            if arm == "names":
                use_roster = roster
                frozen = [{"type": "SPOKEN", "text": b["line"]} for b in block]
                ctx = [{"previous_context": {"type": "NARRATOR", "text": b["prev"]},
                        "next_context": {"type": "NARRATOR", "text": b["next"]}}
                       for b in block]
            else:
                use_roster = sorted(set(symbols.values()))
                frozen = [{"type": "SPOKEN", "text": symbolize(b["line"], symbols)}
                          for b in block]
                ctx = [{"previous_context": {"type": "NARRATOR",
                                             "text": symbolize(b["prev"], symbols)},
                        "next_context": {"type": "NARRATOR",
                                         "text": symbolize(b["next"], symbols)}}
                       for b in block]
            try:
                out = attribute_batch(client, args.model, frozen, params, use_roster,
                                      neighbor_contexts=ctx)
            except Exception:
                n += len(block)
                continue
            for off, b in enumerate(block):
                sp = (out[off] or {}).get("speaker") if off < len(out) else None
                n += 1
                if arm == "names":
                    ok = same_speaker(b["truth"], sp, groups)
                else:
                    # map the model's symbol answer back to the true character
                    want_sym = symbols.get(b["truth"])
                    got = (sp or "").upper().replace(" ", "")
                    ok = bool(want_sym) and want_sym.replace(" ", "") == got
                per_row[s + off] = ok
                hit += ok
        answers[arm] = per_row
        lo, hi = clopper_pearson(hit, max(n, 1))
        print(f"  {arm:8} {hit}/{n} = {hit/max(n,1)*100:5.1f}%  [{lo:.1f}-{hi:.1f}]  "
              f"{time.time()-started:.0f}s", flush=True)

    if len(answers) == 2:
        p, x, y, n = paired(answers["names"], answers["symbols"])
        print(f"\n  symbols - names  +{y}/-{x} of {n}  p={p:.4g}")
        print("\n  SPC symbolizes AND fine-tunes on symbolized data. This applies")
        print("  only the representation to a model that never trained on it, so a")
        print("  loss here does not refute the technique - it would say it needs")
        print("  the matching training regime.")
    json.dump({"book": args.book, "answers": {k: {str(i): v for i, v in a.items()}
                                              for k, a in answers.items()}},
              open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
