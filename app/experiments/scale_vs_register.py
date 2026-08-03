"""Is there an adapter strength that helps everywhere - or at least harms nowhere?

`pdnc_eval` found the adapter's effect swings hard by book: +10.0 on Pride and
Prejudice, +1.5 on The Awakening, -12.5 on The Sign of the Four, netting -0.3
across three public-domain English novels while giving +5.4 on the four light
novels it was tuned around. The gain is real where it lands and the damage is
real where it does not, and nothing so far predicts which a new book will get.

A LoRA does not have to be applied at full strength. llama.cpp will scale it
continuously, so this asks whether some fraction keeps most of the benefit
while blunting the harm - a shipping compromise rather than a coin flip per
book.

Two corpora, deliberately the extremes:

    grimgar03            light novel, adapter helps  (+4.2 in the shippable stack)
    TheSignOfTheFour     Doyle, adapter hurts        (-12.5)

Both scored at the same scales against their own gold, on one loaded server,
with the scale toggled through the API and read back each time.

READINGS, fixed before running:

  a scale that is positive on both     ship at that scale; the full-strength
                                       adapter was simply overshooting
  monotonic in opposite directions     no compromise exists - benefit on one
                                       register is bought with harm on the
                                       other, and the adapter is per-register
  both curves flat                     scale is not the lever and the
                                       difference between books is about
                                       something else entirely
"""
import argparse, collections, json, os, re, sys, time
import urllib.request

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
APP = REPO + "/app/"
sys.path.insert(0, APP)
from openai import OpenAI
from experiments.scoring import alias_groups, same_speaker
from experiments.stats import clopper_pearson
from generate_script import LLMGenParams
from three_pass_generate import attribute_batch, build_roster, get_deterministic_named_entry

M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
SPECIAL = {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE"}
BATCH = 25


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


def set_scale(base_url, scale):
    root = base_url.rsplit("/v1", 1)[0]
    body = json.dumps([{"id": 0, "scale": scale}]).encode()
    req = urllib.request.Request(root + "/lora-adapters", data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    urllib.request.urlopen(req, timeout=30).read()
    with urllib.request.urlopen(root + "/lora-adapters", timeout=30) as fh:
        got = float(json.loads(fh.read())[0].get("scale", -1))
    if abs(got - scale) > 1e-6:
        raise RuntimeError(f"scale did not take: {got} != {scale}")
    return got


def load_pdnc(name, limit):
    fx = json.load(open(APP + f"fixtures/attribution_gold_pdnc_{name.lower()}.json"))
    items = [{"line": e["line"], "truth": e["expected_speaker"],
              "prev": e["prev_context"], "next": e["next_context"]}
             for e in fx["entries"][:limit]]
    return items, fx["roster"], alias_groups({"aliases": fx["aliases"]})


def load_lightnovel(book, limit):
    gold = json.load(open(APP + f"fixtures/attribution_gold_{book}.json"))
    src = open(M + f"inputs/{book}.txt", encoding="utf-8").read()
    cp = json.load(open(M + INPUT_RUN + f"/{book}/result.json.threepass_checkpoint.json"))
    seg = cp["segmented"]
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
        items.append({"line": e["text"],
                      "truth": want[key]["expected_speaker"].upper(),
                      "prev": (seg[i-1].get("text") or "") if i else "",
                      "next": (seg[i+1].get("text") or "") if i+1 < len(seg) else ""})
        if len(items) >= limit:
            break
    return items, roster, alias_groups(gold)


def score(client, model, items, roster, groups, params):
    hit = n = 0
    for s in range(0, len(items), BATCH):
        block = items[s:s + BATCH]
        frozen = [{"type": "SPOKEN", "text": b["line"]} for b in block]
        ctx = [{"previous_context": {"type": "NARRATOR", "text": b["prev"]},
                "next_context": {"type": "NARRATOR", "text": b["next"]}}
               for b in block]
        try:
            out = attribute_batch(client, model, frozen, params, roster,
                                  neighbor_contexts=ctx)
        except Exception:
            n += len(block)
            continue
        for off, b in enumerate(block):
            sp = (out[off] or {}).get("speaker") if off < len(out) else None
            n += 1
            hit += same_speaker(b["truth"], sp, groups)
    return hit, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scales", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--model", default="qwen/qwen3-14b")
    ap.add_argument("--base_url", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--out", default=REPO + "/ab_test_runtime/experiments/scale_vs_register.json")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="local")
    params = LLMGenParams(max_tokens=2000, context_length=32768, temperature=0.0,
                          attribute_temperature=0.0, top_p=0.8,
                          reasoning_effort="none")
    corpora = {
        "grimgar03 (light novel)": load_lightnovel("grimgar03", args.limit),
        "SignOfFour (Doyle)": load_pdnc("TheSignOfTheFour", args.limit),
    }
    for name, (items, roster, _) in corpora.items():
        print(f"{name}: {len(items)} lines, {len(roster)} characters")

    results = collections.defaultdict(dict)
    print(f"\n  {'scale':>7}" + "".join(f"{n[:22]:>26}" for n in corpora))
    for scale in args.scales:
        set_scale(args.base_url, scale)
        row = []
        for name, (items, roster, groups) in corpora.items():
            started = time.time()
            hit, n = score(client, args.model, items, roster, groups, params)
            lo, hi = clopper_pearson(hit, max(n, 1))
            results[name][str(scale)] = {"correct": hit, "n": n,
                                         "acc": hit / max(n, 1)}
            row.append(f"{hit/max(n,1)*100:8.1f}% [{lo:.0f}-{hi:.0f}] {time.time()-started:4.0f}s")
        print(f"  {scale:7.2f}" + "".join(f"{c:>26}" for c in row), flush=True)

    print("\n  deltas against scale 0.0 (adapter off)")
    for name in corpora:
        base = results[name].get("0.0", {}).get("acc")
        if base is None:
            continue
        line = "  " + f"{name[:24]:26}"
        for scale in args.scales:
            a = results[name].get(str(scale), {}).get("acc")
            if a is not None:
                line += f"{(a-base)*100:+8.1f}"
        print(line)
    print(f"  {'':26}" + "".join(f"{s:8.2f}" for s in args.scales))
    print("\n  A scale positive on BOTH rows is a shipping compromise. Opposite")
    print("  monotonic trends mean no compromise exists and the adapter is")
    print("  per-register.")
    json.dump(results, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
