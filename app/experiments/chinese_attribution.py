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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
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
    """Candidates are the TAGS from roleid2idx, not its keys.

    The dataset maps role id -> tag, e.g. {'2': '[C0]', '0': '[C1]'}. The first
    version iterated the KEYS and printed f"[C{key}]", which produced the right
    set of tags only by coincidence and silently disagreed with the mapping.
    """
    lines = inst["text"]
    t = inst["target_idx"]
    tags = sorted(set((inst.get("roleid2idx") or {}).values()), key=str)
    body = []
    for i, line in enumerate(lines):
        mark = "  <<< WHO SPEAKS THIS" if i == t else ""
        body.append(f"{i}: {line}{mark}")
    cands = ", ".join(str(tag) for tag in tags) or "[C0]"
    return (f"Candidates: {cands}\n\nPassage:\n" + "\n".join(body) +
            "\n\nWhich candidate speaks the marked line? Answer with the tag only.")


def truth_of(inst):
    """The gold TAG index, resolved through roleid2idx.

    THIS IS THE BUG THAT VOIDED THE FIRST RUN. `speaker_ids` holds a role id
    ('2'); the model answers with a tag ('[C0]') which normalises to '0'. The
    first version compared the raw role id against the tag index - two
    different namespaces - so nothing could ever match, and the run reported
    0/150 on every arm. Resolve through the mapping instead.
    """
    sid = inst.get("speaker_ids")
    if isinstance(sid, list):
        sid = sid[0] if sid else None
    if sid is None:
        return None
    tag = (inst.get("roleid2idx") or {}).get(str(sid))
    return normalise(tag) if tag is not None else None


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
    ap.add_argument("--no-scale", action="store_true", dest="no_scale",
                    help="skip adapter scaling; measure the served model as-is")
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
        # The adapter comparison needs llama.cpp's /lora-adapters endpoint.
        # --no-scale runs base-only against any OpenAI-compatible server, so
        # the prior question - can this model do Chinese attribution at all -
        # can be answered without standing up the serving stack.
        if not args.no_scale:
            set_scale(args.base_url, scale)
        started, hit, n, blank, errors = time.time(), 0, 0, 0, 0
        raw_samples = []
        for inst in items:
            prompt = build_prompt(inst)
            raw = None
            try:
                r = client.chat.completions.create(
                    model=args.model, temperature=0.0, max_tokens=16,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": prompt}],
                    extra_body={"reasoning_effort": "none"})
                raw = r.choices[0].message.content
                ans = normalise(raw)
            except Exception as exc:                       # noqa: BLE001
                # A transport failure and an unparseable answer are different
                # problems and were previously both tallied as "unparsed",
                # which is how a run against a dead server produced a clean
                # looking 0/150 artifact and got logged OK.
                errors += 1
                ans = None
                if errors <= 3:
                    print(f"    API error: {type(exc).__name__}: "
                          f"{str(exc)[:120]}", flush=True)
            n += 1
            if ans is None:
                blank += 1
                if raw is not None and len(raw_samples) < 5:
                    raw_samples.append(raw[:80])
            hit += (ans is not None and ans == truth_of(inst))
        # An arm that answered nothing measured nothing. Say so loudly rather
        # than emitting a plausible looking zero.
        if blank == n:
            print(f"    NOTHING PARSED on {n} rows. errors={errors}. "
                  f"Sample answers: {raw_samples!r}", flush=True)
        if errors > n * 0.5:
            raise RuntimeError(
                f"{errors}/{n} calls failed - the server is not answering; "
                f"refusing to report this as a result")
        lo, hi = clopper_pearson(hit, max(n, 1))
        results[str(scale)] = {"correct": hit, "n": n, "unparsed": blank,
                               "api_errors": errors,
                               "unparsed_samples": raw_samples}
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
