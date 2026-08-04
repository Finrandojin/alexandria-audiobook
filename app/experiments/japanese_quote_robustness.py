"""Does the quote pre-segmenter survive real Japanese?

pass_quality._QUOTE_CHARS already lists the Japanese corner brackets - 「」 and
『』 - so the code anticipates Japanese input. As far as the ledger shows, that
path has never been run on any. It matters because the corpus is translated
Japanese web novels: the source language is one step upstream of everything
measured here.

Texts are from Aozora Bunko, the Japanese public-domain archive - Soseki,
Dazai, Akutagawa. No licence or consent problem, unlike scraping a fanfiction
archive, and unlike raw prose they exercise the exact punctuation conventions
the segmenter claims to handle.

WHAT THIS MEASURES. Not accuracy - there are no speaker labels here. It counts
how often the quote pre-segmenter has to GUESS, which `quote_repair_risk`
established is where misfiling concentrates: chunks whose quote state was
repaired carry roughly seven times the error rate of chunks where it was known.

FRESH AGAINST FRESH. An earlier version of this compared freshly computed
Japanese figures against the English repair counts stored in a checkpoint, and
those came from a run with different chunking - the stored resolutions do not
reproduce when recomputed. Both sides are now computed the same way in the same
run, which makes the gap larger rather than smaller.
"""
import collections, glob, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO + "/app")
from pass_quality import analyze_outer_quote_regions, validate_segment_quality
from generate_script import split_into_chunks

M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"


def survey(name, src, chunk_size=3000):
    counts = collections.Counter()
    codes = collections.Counter()
    for chunk in split_into_chunks(src, max_size=chunk_size):
        qa = analyze_outer_quote_regions(chunk)
        for r in qa["repairs"]:
            codes[r.get("code")] += 1
        regions = qa["regions"]
        if len(regions) > 1 and validate_segment_quality(
                chunk, regions, quote_analysis=qa)["passed"]:
            if qa["repairs"]:
                counts["repaired"] += 1
            elif qa.get("initial_depth") or qa.get("final_depth"):
                counts["continuation"] += 1
            else:
                counts["clean"] += 1
        else:
            counts["fellthrough"] += 1
    counts["chunks"] = sum(counts[k] for k in
                           ("clean", "repaired", "continuation", "fellthrough"))
    return counts, codes


def main():
    aozora = os.path.join(
        "/home/fakemitch/pinokio/cache/TMPDIR/claude-1000",
        "-home-fakemitch-pinokio-api-alexandria-audiobook2-git",
        "e5db5129-c65a-459a-82cf-736dd0a173e7/scratchpad/aozora")
    jobs = []
    for p in sorted(glob.glob(aozora + "/*.txt")):
        jobs.append((os.path.basename(p)[:-4] + " (ja)",
                     open(p, encoding="utf-8").read()))
    for book in ("index18", "grimgar03"):
        p = M + f"inputs/{book}.txt"
        if os.path.exists(p):
            jobs.append((book + " (en)", open(p, encoding="utf-8").read()))
    if not jobs:
        print("no texts found")
        return

    print(f"  {'text':22}{'chunks':>8}{'clean':>7}{'repaired':>10}"
          f"{'contin.':>9}{'fellthru':>10}")
    all_codes, rows_out = {}, {}
    for name, src in jobs:
        c, codes = survey(name, src)
        all_codes[name] = codes
        rows_out[name] = dict(c)
        print(f"  {name:22}{c['chunks']:8}{c['clean']:7}{c['repaired']:10}"
              f"{c['continuation']:9}{c['fellthrough']:10}")

    # Persist. This printed to a terminal and saved nothing, so the only copy
    # of the result was whatever scrollback happened to survive.
    out = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
        "ab_test_runtime", "experiments", "japanese_quote_robustness.json")
    json.dump({"per_text": rows_out,
               "repair_codes": {k: dict(v) for k, v in all_codes.items()},
               "reading": "Japanese repairs are all inferred_missing_close_quote: "
                          "the chunk ends inside an open bracket and the code "
                          "guesses where the quote closed. LATENT, NOT ACTIVE - "
                          "the pipeline processes English translations."},
              open(out, "w"), indent=1)

    print("\n  repair codes")
    for name, codes in all_codes.items():
        print(f"    {name:22}{dict(codes) if codes else 'none'}")

    print("\n  Every Japanese repair is inferred_missing_close_quote: the chunk")
    print("  ends while still inside an open bracket, so the code GUESSES where")
    print("  the quote closed, and that guess decides SPOKEN vs NARRATOR for the")
    print("  remainder. English quotes close inside their paragraph, so the same")
    print("  chunker never has to guess.")
    print("\n  The fix is upstream of the repair: split_into_chunks splits on size")
    print("  and paragraph structure with no awareness of quote depth. A chunker")
    print("  that refuses to cut inside an open quote removes the inference")
    print("  rather than improving it.")
    print("\n  LATENT, NOT ACTIVE: the pipeline processes English translations, so")
    print("  no shipped output is affected today.")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
