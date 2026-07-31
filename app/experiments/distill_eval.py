"""Did distilling the 70B into the 14B actually move anything?

`distill_train` produces an adapter from 1,091 rows the 70B answered on two
books with no gold. This scores it on the four gold books, which share no rows
with the training data, so every number here is transfer rather than recall.

TWO ARMS, ONE LOADED MODEL:

    base     the 14B as it ships
    tuned    the same weights with the LoRA active

They run through the SAME model object, separated only by peft's
`disable_adapter()`. Loading the base and the tuned model separately would let
a different dtype, device map, or tokenizer revision creep into the comparison
and be read as the adapter's effect. Here the adapter is provably the only
difference.

THE PROMPT PATH IS PRODUCTION'S. Both arms call `attribute_batch`, so batching,
JSON repair, the text-freeze validator and the retry policy are identical to
what ships. That is also why this does not talk to llama.cpp: the adapter is
peft-format, so inference runs through transformers behind a shim that mimics
the sliver of the OpenAI client `call_llm_for_entries` touches. The shim exists
to keep the parsing and validation code in the comparison, not to reimplement
it.

WHAT WOULD MAKE THIS A NULL. Training was one entry per example and inference
sends 25, the mismatch `distill_train` records up front. If `tuned` collapses -
answering with one name everywhere, or breaking the JSON contract - the
batch-shape mismatch is the first suspect, and the per-arm unanswered and
distinct-speaker counts printed below are what distinguish "learned nothing"
from "cannot follow the batch format any more". A model that has stopped
producing parseable batches is not a model that failed to learn attribution.

The comparison that matters is not base vs tuned alone. A tuned 14B is only
interesting if it approaches what the 70B cascade buys, so the cascade's
measured gains on these same books are the standard to read it against.
"""
import argparse, collections, contextlib, json, os, re, sys, time

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
APP = REPO + "/app/"
sys.path.insert(0, APP)

from experiments.manifest import ExperimentRecord
from experiments.scoring import alias_groups, same_speaker
from experiments.stats import clopper_pearson, paired
from generate_script import LLMGenParams
from three_pass_generate import (attribute_batch, build_roster,
                                 get_deterministic_named_entry)

M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
BOOKS = ("grimgar03", "index18", "mushoku16", "owarimonogatari3")
SPECIAL = {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE"}
BATCH = 25


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


# --- the shim -------------------------------------------------------------
# call_llm_for_entries reads exactly: response.choices[0].message.content,
# .finish_reason, and (getattr-safe) .usage. Nothing else. Anything more
# elaborate would be inventing an interface the caller does not use.

class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content, finish_reason):
        self.message = _Msg(content)
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, content, finish_reason):
        self.choices = [_Choice(content, finish_reason)]
        self.usage = None


class LocalClient:
    """Mimics the sliver of the OpenAI client that the LLM path touches."""

    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        self.chat = self
        self.completions = self
        self.adapter_enabled = True

    def create(self, model=None, messages=None, temperature=0.0, top_p=1.0,
               presence_penalty=0.0, max_tokens=512, extra_body=None):
        import torch
        # Qwen3 emits <think> blocks by default. Production suppresses them
        # through extra_body (reasoning_effort="none"), which a local
        # tokenizer never sees - so without this the arms would run WITH
        # reasoning while the thing they are compared against ran without it,
        # and each call generated thousands of thinking tokens (~7 minutes per
        # batch, against ~8 seconds). Wrong configuration first, slow second.
        try:
            prompt = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            # Tokenizers without the flag never had the behaviour to disable.
            prompt = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        kw = dict(max_new_tokens=max_tokens,
                  pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        # temperature=0 must mean greedy, not a near-zero sample: every other
        # harness in this investigation ran deterministic and a sampled arm
        # would not be comparable to any of them.
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(**enc, **kw)
        gen = out[0][enc["input_ids"].shape[1]:]
        text = self.tok.decode(gen, skip_special_tokens=True)
        finish = "length" if len(gen) >= max_tokens else "stop"
        return _Response(text, finish)


def load_book(book):
    gold = json.load(open(APP + f"fixtures/attribution_gold_{book}.json"))
    src = open(M + f"inputs/{book}.txt", encoding="utf-8").read()
    cp = json.load(open(
        M + INPUT_RUN + f"/{book}/result.json.threepass_checkpoint.json"))
    seg = cp["segmented"]
    roster = [r.upper() for r in
              build_roster([e for e in (cp.get("named") or []) if e], src)]
    roster = sorted(set(roster) | {n.upper() for n in
                                   gold.get("roster_additions", {}).get("names", [])})
    occ = collections.Counter(norm(e.get("text")) for e in seg)
    want = {norm(g["line"]): g for g in gold["entries"]
            if occ[norm(g["line"])] == 1
            and g["expected_speaker"].upper() not in SPECIAL}
    return gold, src, seg, roster, want


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter", default=REPO + "/ab_test_runtime/distill/adapter")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--books", nargs="+", default=list(BOOKS))
    ap.add_argument("--tag", default=os.environ.get("EXPERIMENT_TAG", "distill"))
    ap.add_argument("--limit", type=int, default=0,
                    help="cap scored rows per book, for a smoke run")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    client = LocalClient(model, tok)
    params = LLMGenParams(max_tokens=12000, context_length=32768,
                          temperature=0.0, attribute_temperature=0.0,
                          top_p=0.8, reasoning_effort="none")

    # There is no LM Studio here, so the environment is stated rather than
    # queried: `parallel` is 1 because generate() is called serially, and
    # `context_length` is what the params actually enforce. The runtime marker
    # keeps this from being read later as an LM Studio run.
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    environment = {"loaded": True, "context_length": 32768, "parallel": 1,
                   "optimized": None, "runtime": "transformers+peft",
                   "gpu": gpu, "torch": torch.__version__,
                   "dtype": "bfloat16"}
    # The constructor hashes ONE fixture; this run spans four. Record the first
    # for the schema and every book's hash alongside it, so the artifact cannot
    # imply it was scored against a single gold file.
    record = ExperimentRecord(
        "distill_eval", REPO, args.model, f"peft:{args.adapter}",
        APP + f"fixtures/attribution_gold_{args.books[0]}.json",
        {"temperature": 0.0, "batch": BATCH, "adapter": args.adapter},
        environment=environment,
        notes="LoRA distilled from a 70B on 1,091 routed rows of grimgar06 and "
              "mushoku18, scored on four gold books it never saw. Arms share "
              "one loaded model and differ only by peft disable_adapter().")
    import hashlib
    record.meta["gold_files"] = {
        b: hashlib.sha256(open(APP + f"fixtures/attribution_gold_{b}.json",
                               "rb").read()).hexdigest()
        for b in args.books}

    totals = {"base": [0, 0], "tuned": [0, 0]}
    per_book, answers = {}, {"base": {}, "tuned": {}}
    for book in args.books:
        gold, src, seg, roster, want = load_book(book)
        groups = alias_groups(gold)
        windows = [list(range(s, min(s + BATCH, len(seg))))
                   for s in range(0, len(seg), BATCH)]
        windows = [w for w in windows
                   if any(norm(seg[i].get("text")) in want for i in w)]
        print(f"\n{book}: {len(want)} scoreable lines, roster {len(roster)}, "
              f"{len(windows)} windows", flush=True)
        for arm in ("base", "tuned"):
            started, scored = time.time(), 0
            for k, win in enumerate(windows, 1):
                if args.limit and scored >= args.limit:
                    break
                send = [i for i in win
                        if get_deterministic_named_entry(seg[i]) is None]
                if not send or not any(norm(seg[i].get("text")) in want
                                       for i in send):
                    continue
                frozen = [{"type": seg[i]["type"], "text": seg[i]["text"]}
                          for i in send]
                ctx = [{"previous_context": seg[i - 1] if i else None,
                        "next_context": seg[i + 1] if i + 1 < len(seg) else None}
                       for i in send]
                ctxmgr = (model.disable_adapter() if arm == "base"
                          else contextlib.nullcontext())
                try:
                    with ctxmgr:
                        out = attribute_batch(client, args.model, frozen, params,
                                              roster, neighbor_contexts=ctx,
                                              source_text=src)
                except Exception as exc:
                    print(f"  {arm} window {k}: {type(exc).__name__}", flush=True)
                    # A failed batch is a failure, not an absence. Dropping it
                    # would remove from the denominator exactly the rows this
                    # arm could not handle.
                    for i in send:
                        key = norm(seg[i].get("text"))
                        if key in want:
                            g = want[key]
                            record.add(arm, f"{book}:{g['id']}", g["line"],
                                       g["expected_speaker"].upper(), None, False,
                                       provenance=f"{arm}|{book}|batch_failed")
                            scored += 1
                    continue
                for off, i in enumerate(send):
                    key = norm(seg[i].get("text"))
                    if key not in want:
                        continue
                    g = want[key]
                    sp = (out[off] or {}).get("speaker") if off < len(out) else None
                    record.add(arm, f"{book}:{g['id']}", g["line"],
                               g["expected_speaker"].upper(), sp,
                               same_speaker(g["expected_speaker"], sp, groups),
                               provenance=f"{arm}|{book}")
                    scored += 1
                if k % 20 == 0:
                    print(f"  {arm} {k}/{len(windows)} ...", flush=True)
            rows = [r for r in record.rows
                    if r["arm"] == arm and r["id"].startswith(book + ":")]
            hit = sum(1 for r in rows if r["correct"])
            totals[arm][0] += hit
            totals[arm][1] += len(rows)
            per_book.setdefault(book, {})[arm] = (hit, len(rows))
            answers[arm].update({r["id"]: r["correct"] for r in rows})
            unanswered = sum(1 for r in rows if not r["predicted"])
            distinct = len({(r["predicted"] or "").upper() for r in rows})
            lo, hi = clopper_pearson(hit, max(len(rows), 1))
            print(f"  {arm:6} {hit}/{len(rows)} = {hit/max(len(rows),1)*100:5.1f}%"
                  f"  [{lo:.1f}-{hi:.1f}]  unanswered {unanswered}"
                  f"  distinct names {distinct}  {time.time()-started:.0f}s",
                  flush=True)

    print("\n  per book")
    for book, arms in per_book.items():
        b, t = arms.get("base", (0, 0)), arms.get("tuned", (0, 0))
        d = (t[0] / max(t[1], 1) - b[0] / max(b[1], 1)) * 100
        print(f"    {book:18} base {b[0]/max(b[1],1)*100:5.1f}%  "
              f"tuned {t[0]/max(t[1],1)*100:5.1f}%  {d:+6.1f}")
    p, x, y, n = paired(answers["base"], answers["tuned"])
    tb, tt = totals["base"], totals["tuned"]
    print(f"\n  pooled  base {tb[0]}/{tb[1]} = {tb[0]/max(tb[1],1)*100:.1f}%"
          f"   tuned {tt[0]}/{tt[1]} = {tt[0]/max(tt[1],1)*100:.1f}%")
    print(f"  paired  {(tt[0]/max(tt[1],1) - tb[0]/max(tb[1],1))*100:+.1f} points"
          f"  +{y}/-{x} of {n}  p={p:.4g}")
    print("\n  Read this against the cascade's measured gains on these same "
          "books.\n  A tuned 14B that does not approach them has not replaced "
          "the 70B,\n  whatever its sign.")
    out = record.write(os.path.join(
        REPO, "ab_test_runtime", "experiments",
        f"distill_eval__{args.tag}.json"),
        contract={"expected_arms": ("base", "tuned")})
    print("wrote", out)


if __name__ == "__main__":
    main()
