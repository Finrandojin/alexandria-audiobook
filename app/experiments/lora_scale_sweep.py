"""Is 1.0 the right adapter strength?

The adapter ships at scale 1.0 because that is the default, not because anyone
measured it. llama.cpp applies a LoRA at an arbitrary scale and lets it be
changed at runtime, so the whole sweep runs against ONE loaded model with no
retraining and no reloading - the cheapest experiment available.

Scale below 1.0 blends the adapter with the base. If the adapter is slightly
overfit to its 1,091 training rows, a partial application can beat a full one;
if it is underfit, above 1.0 may help. Both are cheap to check and neither has
been.

ONE SERVER, SCALE TOGGLED VIA POST /lora-adapters, read back after every change
- a scale that silently fails to take would make two arms identical and read as
a flat curve, which is exactly the shape a null result has.
"""
import argparse, collections, json, os, re, sys, time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = REPO + "/app/"
sys.path.insert(0, APP)
from openai import OpenAI
from experiments.scoring import alias_groups, same_speaker
from experiments.stats import clopper_pearson
from generate_script import LLMGenParams
from three_pass_generate import (attribute_batch, build_roster,
                                 get_deterministic_named_entry)

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
        raise RuntimeError(f"scale did not take: asked {scale}, got {got}")
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--book", default="grimgar03")
    ap.add_argument("--scales", nargs="+", type=float,
                    default=[0.0, 0.5, 0.75, 1.0, 1.25])
    ap.add_argument("--base_url", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--model", default="qwen/qwen3-14b")
    ap.add_argument("--out", default=REPO + "/ab_test_runtime/experiments/lora_scale_sweep.json")
    args = ap.parse_args()

    gold = json.load(open(APP + f"fixtures/attribution_gold_{args.book}.json"))
    src = open(M + f"inputs/{args.book}.txt", encoding="utf-8").read()
    cp = json.load(open(M + INPUT_RUN + f"/{args.book}/result.json.threepass_checkpoint.json"))
    seg = cp["segmented"]
    groups = alias_groups(gold)
    roster = [r.upper() for r in
              build_roster([e for e in (cp.get("named") or []) if e], src)]
    roster = sorted(set(roster) | {n.upper() for n in
                                   gold.get("roster_additions", {}).get("names", [])})
    occ = collections.Counter(norm(e.get("text")) for e in seg)
    want = {norm(g["line"]): g for g in gold["entries"]
            if occ[norm(g["line"])] == 1
            and g["expected_speaker"].upper() not in SPECIAL}

    client = OpenAI(base_url=args.base_url, api_key="local")
    params = LLMGenParams(max_tokens=2000, context_length=32768,
                          temperature=0.0, attribute_temperature=0.0,
                          top_p=0.8, reasoning_effort="none")
    windows = [list(range(s, min(s + BATCH, len(seg)))) for s in range(0, len(seg), BATCH)]
    windows = [w for w in windows if any(norm(seg[i].get("text")) in want for i in w)]
    print(f"{args.book}: {len(want)} scoreable lines, {len(windows)} windows", flush=True)

    results = {}
    for scale in args.scales:
        got = set_scale(args.base_url, scale)
        started, hit, total = time.time(), 0, 0
        for k, win in enumerate(windows, 1):
            send = [i for i in win if get_deterministic_named_entry(seg[i]) is None]
            rows = [i for i in send if norm(seg[i].get("text")) in want]
            if not rows:
                continue
            frozen = [{"type": seg[i]["type"], "text": seg[i]["text"]} for i in send]
            ctx = [{"previous_context": seg[i - 1] if i else None,
                    "next_context": seg[i + 1] if i + 1 < len(seg) else None}
                   for i in send]
            try:
                out = attribute_batch(client, args.model, frozen, params, roster,
                                      neighbor_contexts=ctx, source_text=src)
            except Exception:
                total += len(rows)
                continue
            for off, i in enumerate(send):
                key = norm(seg[i].get("text"))
                if key not in want:
                    continue
                g = want[key]
                sp = (out[off] or {}).get("speaker") if off < len(out) else None
                total += 1
                hit += same_speaker(g["expected_speaker"], sp, groups)
        lo, hi = clopper_pearson(hit, max(total, 1))
        results[str(scale)] = {"correct": hit, "n": total, "acc": hit / max(total, 1)}
        print(f"  scale {got:<5} {hit}/{total} = {hit/max(total,1)*100:5.1f}%  "
              f"[{lo:.1f}-{hi:.1f}]  {time.time()-started:.0f}s", flush=True)

    if results:
        best = max(results, key=lambda s: results[s]["acc"])
        one = results.get("1.0", {}).get("acc")
        print(f"\n  best scale {best} at {results[best]['acc']*100:.1f}%")
        if one is not None:
            print(f"  shipped scale 1.0 at {one*100:.1f}%  "
                  f"({(results[best]['acc']-one)*100:+.1f})")
        print("  A flat curve means the adapter is not sensitive to strength "
              "and 1.0 is\n  fine. A peak below 1.0 means it is slightly "
              "overfit to its 1,091 rows.")
        json.dump(results, open(args.out, "w"), indent=1)
        print("\nwrote", args.out)


if __name__ == "__main__":
    main()
