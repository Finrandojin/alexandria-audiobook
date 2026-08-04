"""Which segmentation path produces the misfiled lines?

Segmentation misfiles narration as speech on ~4% of judged rows, and that is
the largest uninstrumented error source left in the pipeline. This asks WHERE
it happens, because "segmentation is hard" is not a target and a code path is.

Pass 1 does not usually ask the model at all. `segment_chunk_adaptively` first
tries `analyze_outer_quote_regions`, and if the outer quote structure resolves
cleanly it returns those regions directly - 162 of index18's 165 chunks. The
resolution recorded per chunk says which route was taken:

    quote_presegmented               quotes balanced, structure trusted
    quote_presegmented_repaired      quotes did NOT balance; the code INFERRED
                                     a missing delimiter and proceeded
    quote_presegmented_continuation  a quote spans the chunk boundary, so the
                                     depth state is carried in as an assumption
    clean / near_miss / fail         the model actually segmented it

`repaired` and `continuation` are the two cases where quote state is a GUESS,
and a wrong guess flips the type of everything in the region - which is exactly
what misfiling is.

CHUNKS ARE RECOVERED EXACTLY, not by position. The checkpoint records one
resolution per chunk but not which chunk each segment came from, so an earlier
version of this mapped rows to chunks proportionally and measured 4.5x. Re-
running the deterministic chunker on the source recovers the real boundaries
and the effect is 7.0x: the approximation was diluting it.

WHY THE SHORTCUT'S OWN CHECK CANNOT CATCH THIS. The regions are accepted if
`validate_segment_quality(chunk, regions, quote_analysis=quote_analysis)`
passes - and `_quote_region_findings` then compares those regions against
`quote_analysis["regions"]`, the same object that produced them. A wrong repair
validates perfectly because the validator shares its assumption.
"""
import collections, glob, json, os, re, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO + "/app")
from generate_script import split_into_chunks

M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
RISKY = ("quote_presegmented_repaired", "quote_presegmented_continuation")


def main():
    tot, bad, skipped = collections.Counter(), collections.Counter(), []
    for path in sorted(glob.glob(REPO + "/ab_test_runtime/fixtures_draft/labelling_bundle__*.json")):
        book = os.path.basename(path).split("__")[1].replace(".json", "")
        src_path = M + f"inputs/{book}.txt"
        cp = M + INPUT_RUN + f"/{book}/result.json.threepass_checkpoint.json"
        if not (os.path.exists(src_path) and os.path.exists(cp)):
            continue
        src = open(src_path, encoding="utf-8").read()
        res = (json.load(open(cp)).get("resolutions") or [])
        chunks = split_into_chunks(src, max_size=3000)
        if len(chunks) != len(res):
            # A different chunk size was used for that run; a mismatched
            # mapping would silently attribute rows to the wrong path.
            skipped.append(f"{book} ({len(chunks)} chunks vs {len(res)} resolutions)")
            continue
        starts, off = [], 0
        for c in chunks:
            i = src.find(c, off)
            starts.append(i if i >= 0 else off)
            off = (i if i >= 0 else off) + len(c)
        for e in json.load(open(path))["entries"]:
            probe = (e.get("line") or "").strip()[:50]
            if len(probe) < 12:
                continue
            pos = src.find(probe)
            if pos < 0:
                continue
            k = max(0, sum(1 for s in starts if s <= pos) - 1)
            r = res[k] if k < len(res) else "?"
            tot[r] += 1
            if e.get("expected_speaker") == "NOT_DIALOGUE":
                bad[r] += 1

    if not tot:
        print("no book had a usable chunk mapping")
        return
    print(f"  {'resolution':38}{'judged':>8}{'misfiled':>10}{'rate':>8}")
    for r, n in tot.most_common():
        print(f"  {r:38}{n:8}{bad[r]:10}{bad[r]/n*100:7.1f}%")
    rn = sum(tot[r] for r in RISKY)
    rb = sum(bad[r] for r in RISKY)
    on = sum(tot.values()) - rn
    ob = sum(bad.values()) - rb
    if rn and on:
        print(f"\n  quote state GUESSED : {rb}/{rn} = {rb/rn*100:.1f}%")
        print(f"  quote state known   : {ob}/{on} = {ob/on*100:.1f}%")
        print(f"  ratio {(rb/rn)/max(ob/on,1e-9):.1f}x")
        print(f"\n  {rn/(rn+on)*100:.0f}% of judged rows carry "
              f"{rb/max(rb+ob,1)*100:.0f}% of the misfiling.")
    if skipped:
        print("\n  skipped (chunk/resolution mismatch): " + "; ".join(skipped))
    print("\n  INDICATED CHANGE: when quote_analysis['repairs'] is non-empty, or")
    print("  the chunk starts or ends inside a quote, the pre-segmentation")
    print("  shortcut should NOT be trusted - those chunks should fall through")
    print("  to the model. Today `repairs` only changes the label recorded, not")
    print("  the decision. The cost is speed on ~7% of chunks.")
    print("\n  The buckets are small (25 and 35 rows), so the RATES are soft even")
    print("  though the direction held across two independent chunk mappings.")


if __name__ == "__main__":
    main()
