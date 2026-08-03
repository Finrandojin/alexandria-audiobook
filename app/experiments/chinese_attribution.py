"""The first non-English attribution number in this project.

Every accuracy in the ledger is English: four light novels in translation, plus
three public-domain English novels via PDNC. The adapter was trained on English
renderings of Japanese web novels. Whether any of it survives a change of
language has never been measured, and two attempts to find a corpus failed for
structural reasons - Aozora has no speaker labels, NaroU stores speaker to
utterance with no surrounding prose.

WP (Lu Yao, 平凡的世界) and JY (Jin Yong) do have the right shape:

    text          the passage, narration and dialogue interleaved
    target_idx    which line to attribute
    speaker_ids   the answer
    roleid2poses  where each character is mentioned in the context

CHARACTERS ARE ANONYMISED as [C0], [C1] ... rather than named. That makes this a
PURER attribution test than our own: no name knowledge, no aliases, no
romanisation variants - the model must work only from who is present and who is
speaking to whom. It also means the numbers are NOT comparable to the ledger's,
in the same way PDNC's are not, and for the same reason: a different task
surface.

WHAT TO EXPECT AND WHY IT IS STILL WORTH RUNNING. The adapter saw only English
light-novel prose, so the honest prior is that it does nothing here or hurts.
The base number matters independently: Qwen3 is a Chinese-developed model and
may be stronger on Chinese than on English, which would say something about
where the pipeline's difficulty actually comes from.
"""
import argparse, collections, json, os, re, sys, time
import urllib.request

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
APP = REPO + "/app/"
sys.path.insert(0, APP)
from openai import OpenAI
from experiments.stats import clopper_pearson
from generate_script import LLMGenParams

SP = ("/home/fakemitch/pinokio/cache/TMPDIR/claude-1000/"
      "-home-fakemitch-pinokio-api-alexandria-audiobook2-git/"
      "e5db5129-c65a-459a-82cf-736dd0a173e7/scratchpad/chinese")

SYSTEM = ("You identify who speaks a line of dialogue in a Chinese novel. "
          "Characters appear as [C0], [C1] and so on. Answer with the single "
          "character tag that speaks the marked line, nothing else.")


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


def build_prompt(inst):
    lines = inst["text"]
    t = inst["target_idx"]
    roles = sorted(inst.get("roleid2idx", {}) or {}, key=str)
    body = []
    for i, line in enumerate(lines):
        mark = "  <<< WHO SPEAKS THIS" if i == t else ""
        body.append(f"{i}: {line}{mark}")
    cands = ", ".join(f"[C{r}]" if not str(r).startswith("[") else str(r)
                      for r in roles) or "[C0]"
    return (f"Candidates: {cands}\n\nPassage:\n" + "\n".join(body) +
            "\n\nWhich candidate speaks the marked line? Answer with the tag only.")


def truth_of(inst):
    sid = inst.get("speaker_ids")
    if isinstance(sid, list):
        sid = sid[0] if sid else None
    return str(sid) if sid is not None else None


def normalise(ans):
    m = re.search(r"C\s*(\d+)", (ans or "").upper())
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default=SP + "/wp_test.json")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--scales", nargs="+", type=float, default=[0.0, 1.0])
    ap.add_argument("--model", default="qwen/qwen3-14b")
    ap.add_argument("--base_url", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--out", default=REPO + "/ab_test_runtime/experiments/chinese_attribution.json")
    args = ap.parse_args()

    data = json.load(open(args.dataset, encoding="utf-8"))
    items = (list(data.values()) if isinstance(data, dict) else data)[:args.limit]
    items = [i for i in items if truth_of(i) is not None]
    print(f"{os.path.basename(args.dataset)}: {len(items)} instances with a "
          f"labelled speaker\n")

    client = OpenAI(base_url=args.base_url, api_key="local")
    results = {}
    for scale in args.scales:
        set_scale(args.base_url, scale)
        started, hit, n, blank = time.time(), 0, 0, 0
        for inst in items:
            prompt = build_prompt(inst)
            try:
                r = client.chat.completions.create(
                    model=args.model, temperature=0.0, max_tokens=16,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": prompt}],
                    extra_body={"reasoning_effort": "none"})
                ans = normalise(r.choices[0].message.content)
            except Exception:
                ans = None
            n += 1
            if ans is None:
                blank += 1
            hit += (ans is not None and ans == truth_of(inst))
        lo, hi = clopper_pearson(hit, max(n, 1))
        results[str(scale)] = {"correct": hit, "n": n, "unparsed": blank}
        label = "adapter off" if scale == 0 else f"adapter {scale}"
        print(f"  {label:14} {hit}/{n} = {hit/max(n,1)*100:5.1f}%  "
              f"[{lo:.1f}-{hi:.1f}]  unparsed {blank}  {time.time()-started:.0f}s",
              flush=True)

    if "0.0" in results and "1.0" in results:
        b = results["0.0"]["correct"] / max(results["0.0"]["n"], 1)
        l = results["1.0"]["correct"] / max(results["1.0"]["n"], 1)
        print(f"\n  adapter effect on Chinese: {(l-b)*100:+.1f} points")
        print("  The adapter saw only English light-novel prose; anything other")
        print("  than 'no effect or worse' here would be surprising and would")
        print("  need explaining rather than celebrating.")
    print("\n  NOT comparable to the ledger: anonymised character tags remove")
    print("  name knowledge, aliases and romanisation entirely, so this measures")
    print("  a narrower and cleaner task than the pipeline performs.")
    json.dump({"dataset": os.path.basename(args.dataset), "results": results},
              open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
