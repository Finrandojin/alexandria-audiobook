"""Would a quote-aware chunker remove the guessing entirely?

Two independent findings converge on the chunker:

  English   `quote_repair_risk` - chunks whose quote state was REPAIRED or
            carried across a boundary misfile narration as speech at 16.0% and
            14.3% against 2.3% elsewhere, and 7% of judged rows carry 41% of
            the errors.
  Japanese  `japanese_quote_robustness` - Kokoro produces 51
            inferred_missing_close_quote repairs where English produces none.

Both are the same defect: `split_into_chunk_records` packs paragraphs to a size
budget with no notion of quote depth, so a chunk can end with a bracket open.
`analyze_outer_quote_regions` then INFERS where the quote closed, and that
inference decides SPOKEN vs NARRATOR for the whole region.

It also cuts oversized paragraphs at `rfind(" ")`, which returns -1 in a
language without inter-word spaces - so Japanese falls through to a hard slice
at exactly max_size, mid-sentence.

This does not change production. It implements the alternative and counts
repairs both ways, which is the whole measurement: a repair is the code
admitting it had to guess, so fewer repairs is strictly less guessing.

WHAT IT CANNOT SHOW. Fewer repairs is not directly fewer misattributions -
that link comes from quote_repair_risk's 7x, measured on English with a
proportional-mapping caveat. A chunker that removed every repair would remove
the RISK, and the accuracy gain would still need measuring separately.
"""
import argparse, collections, glob, os, re, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO + "/app")
from pass_quality import analyze_outer_quote_regions, _QUOTE_CHARS
from generate_script import split_into_chunks

M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
AOZORA = ("/home/fakemitch/pinokio/cache/TMPDIR/claude-1000/"
          "-home-fakemitch-pinokio-api-alexandria-audiobook2-git/"
          "e5db5129-c65a-459a-82cf-736dd0a173e7/scratchpad/aozora")
OPENERS = {'"', '“', '「', '『'}
CLOSERS = {'"', '”', '」', '』'}
SENTENCE_END = "。！？!?."
# Priority-ordered split points, adapted from p0n1/epub_to_audiobook's
# split_long_sentence. Two things it gets right that rfind(" ") does not: CJK
# sentence-enders are tried BEFORE English ones, so a space-less language never
# falls through to a hard slice; and CLOSING BRACKETS are split points, so a cut
# lands after a quote closes rather than inside it.
PUNCT_PRIORITY = ['。', '！', '？', '. ', '! ', '? ', '；', ';', '，', ',',
                  '：', ':', '）', ')', ']', '】', '}', '」', '』']


def priority_cut(window):
    """Highest-priority punctuation position in `window`, or -1."""
    for mark in PUNCT_PRIORITY:
        idx = window.rfind(mark)
        if idx > 0:
            return idx + len(mark)
    return -1


def depth_after(text, start=0):
    """Quote depth at the end of `text`. Straight quotes toggle; paired
    brackets nest, which is why they cannot share one counter."""
    depth = start
    for ch in text:
        if ch in ('「', '『', '“'):
            depth += 1
        elif ch in ('」', '』', '”'):
            depth = max(0, depth - 1)
        elif ch == '"':
            depth = 0 if depth else 1
    return depth


def quote_aware_chunks(text, max_size=3000, slack=1.5):
    """See the note on `slack`: absorbing without a cap is not a fix."""
    """Pack paragraphs, but never end a chunk with a quote still open.

    When a chunk would close mid-quote it absorbs further paragraphs until the
    depth returns to zero - but only up to `slack` x max_size. Without that cap
    the first version removed 86% of repairs by making chunks enormous
    (owarimonogatari3 went from 222 chunks to 14, ~47k characters each), which
    is not a fix: it removes boundary problems by removing boundaries, and it
    would blow the context window. Past the cap it cuts anyway and accepts the
    repair, because a guess is better than an unusable chunk.

    Oversized paragraphs are cut at sentence-ending punctuation, with the space
    fallback kept only for languages that have spaces.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    chunks, current, depth = [], "", 0

    def cut_long(piece, depth_in):
        out, remaining, d = [], piece.strip(), depth_in
        while len(remaining) > max_size:
            window = remaining[:max_size + 1]
            cut = priority_cut(window)
            if cut <= 0:
                cut = window.rfind(" ")
                cut = cut + 1 if cut > 0 else max_size
            head = remaining[:cut]
            # do not emit a piece that ends inside a quote if avoidable
            probe = depth_after(head, d)
            if probe and len(remaining) > cut:
                nxt = remaining.find("」", cut)
                alt = remaining.find("”", cut)
                closer = min([x for x in (nxt, alt) if x >= 0], default=-1)
                if 0 <= closer < cut + max_size:
                    cut = closer + 1
                    head = remaining[:cut]
            d = depth_after(head, d)
            out.append(head.strip())
            remaining = remaining[cut:].strip()
        if remaining:
            out.append(remaining)
        return out, depth_after("".join(out), depth_in)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > max_size:
            if current:
                chunks.append(current)
                current, depth = "", 0
            pieces, depth = cut_long(para, depth)
            chunks.extend(pieces)
            continue
        would = (current + "\n\n" + para).strip() if current else para
        over_slack = len(would) > max_size * slack
        if len(would) > max_size and current and (depth == 0 or over_slack):
            chunks.append(current)
            current, depth = para, depth_after(para, 0)
        else:
            current = would
            depth = depth_after(current, 0)
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def repairs_for(chunks):
    codes = collections.Counter()
    for c in chunks:
        for r in analyze_outer_quote_regions(c)["repairs"]:
            codes[r.get("code")] += 1
    return codes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max_size", type=int, default=3000)
    ap.add_argument("--slack", type=float, default=1.5,
                    help="how far a chunk may exceed max_size to close a quote")
    args = ap.parse_args()

    jobs = []
    for p in sorted(glob.glob(AOZORA + "/*.txt")):
        jobs.append((os.path.basename(p)[:-4] + " (ja)", open(p, encoding="utf-8").read()))
    for book in ("index18", "grimgar03", "owarimonogatari3", "mushoku16"):
        p = M + f"inputs/{book}.txt"
        if os.path.exists(p):
            jobs.append((book + " (en)", open(p, encoding="utf-8").read()))

    print(f"  {'text':22}{'chunks now':>12}{'repairs now':>13}"
          f"{'chunks new':>12}{'repairs new':>13}{'max chunk':>12}")
    tot_a = tot_b = 0
    for name, src in jobs:
        now = split_into_chunks(src, max_size=args.max_size)
        new = quote_aware_chunks(src, max_size=args.max_size, slack=args.slack)
        ra, rb = sum(repairs_for(now).values()), sum(repairs_for(new).values())
        tot_a += ra
        tot_b += rb
        biggest = max((len(c) for c in new), default=0)
        print(f"  {name:22}{len(now):12}{ra:13}{len(new):12}{rb:13}{biggest:12}")
    print(f"\n  total repairs {tot_a} -> {tot_b}"
          f"  ({(tot_a-tot_b)/max(tot_a,1)*100:.0f}% removed)")
    print("\n  A repair is the code admitting it had to guess where a quote")
    print("  closed, and that guess sets SPOKEN vs NARRATOR for a whole region.")
    print("  Fewer repairs is strictly less guessing - it is NOT directly fewer")
    print("  misattributions; that link is quote_repair_risk's 7x, measured")
    print("  separately and on English only.")


if __name__ == "__main__":
    main()
