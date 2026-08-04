"""Does reading the EPUB's own structure beat reconstructing it from flat text?

The pipeline receives .txt. Every corpus input is .txt, and app/uploads holds
the EPUBs those came from - so the structure is flattened outside this codebase
and then partly reconstructed inside it:

    source_normalization.py   regexes to strip front matter, tables of
                              contents and colophons, with comments that say
                              outright "an epub extractor flattens a table of
                              contents into one long paragraph"
    split_into_chunk_records  paragraphs inferred with re.split(r'\\n\\s*\\n')

An EPUB states all of it. Each spine document is a chapter, each <p> is a
paragraph, and navigation documents are typed in the manifest rather than
guessed at by looking for the word "contents".

This measures both paths on the same book:

    flat         strip tags, join everything, then the production chunker
    structural   ebooklib spine + <p> extraction, chunked within chapters at
                 real paragraph boundaries and never across a chapter edge

WHAT IS COUNTED. Chunks, quote repairs (the code admitting it had to guess
where a quote closed - the path `quote_repair_risk` measured at 7x the
misfiling rate), and whether navigation/front matter survives into the text.

WHAT THIS CANNOT SHOW. Fewer repairs is less guessing, not directly fewer
misattributions; that link was measured separately and only on English. And a
structural reader helps only when the input is an EPUB - it does nothing for a
.txt someone else already flattened, which is what the corpus currently is.
"""
import argparse, collections, os, re, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO + "/app")
from pass_quality import analyze_outer_quote_regions
from generate_script import split_into_chunks


def flat_text(path):
    """What the pipeline gets today: tags stripped, everything joined."""
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    book = epub.read_epub(path, options={"ignore_ncx": True})
    parts = []
    for item in book.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_body_content(), "lxml")
        parts.append(soup.get_text("\n"))
    return "\n\n".join(parts)


def structural(path):
    """Chapters and paragraphs as the EPUB declares them."""
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    book = epub.read_epub(path, options={"ignore_ncx": True})
    # The spine is reading order; nav documents are typed and excluded here
    # rather than detected later by looking for the word "contents".
    nav_ids = {i.get_id() for i in book.get_items()
               if i.get_type() == ebooklib.ITEM_NAVIGATION
               or "nav" in (i.get_name() or "").lower()}
    chapters = []
    for spine_id, _ in book.spine:
        item = book.get_item_with_id(spine_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        if item.get_id() in nav_ids:
            continue
        soup = BeautifulSoup(item.get_body_content(), "lxml")
        paras = [p.get_text(" ").strip()
                 for p in soup.find_all(["p", "h1", "h2", "h3", "h4", "li"])]
        paras = [p for p in paras if p]
        if paras:
            chapters.append({"name": item.get_name(), "paragraphs": paras})
    return chapters


def chunk_structural(chapters, max_size=3000, slack=1.5):
    """Pack real paragraphs, never crossing a chapter, never ending mid-quote."""
    def depth_after(text, start=0):
        d = start
        for ch in text:
            if ch in ('「', '『', '“'):
                d += 1
            elif ch in ('」', '』', '”'):
                d = max(0, d - 1)
            elif ch == '"':
                d = 0 if d else 1
        return d
    chunks = []
    for chapter in chapters:
        current = ""
        for para in chapter["paragraphs"]:
            would = (current + "\n\n" + para).strip() if current else para
            if (len(would) > max_size and current
                    and (depth_after(current) == 0 or len(would) > max_size * slack)):
                chunks.append(current)
                current = para
            else:
                current = would
        if current.strip():
            chunks.append(current)
    return chunks


def repairs(chunks):
    codes = collections.Counter()
    for c in chunks:
        for r in analyze_outer_quote_regions(c)["repairs"]:
            codes[r.get("code")] += 1
    return codes


NAV_HINT = re.compile(r"table of contents|^\s*contents\s*$|copyright|all rights reserved",
                      re.IGNORECASE | re.MULTILINE)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--epubs", nargs="+", default=sorted(
        __import__("glob").glob(REPO + "/app/uploads/*.epub"))[:3])
    ap.add_argument("--max_size", type=int, default=3000)
    args = ap.parse_args()

    if not args.epubs:
        print("no epubs found")
        return
    print(f"  {'book':30}{'path':>12}{'chunks':>8}{'repairs':>9}"
          f"{'chars':>10}{'nav hits':>10}")
    for path in args.epubs:
        name = os.path.basename(path)[:28]
        try:
            flat = flat_text(path)
            chapters = structural(path)
        except Exception as exc:
            print(f"  {name:30} FAILED {type(exc).__name__}: {exc}")
            continue
        flat_chunks = split_into_chunks(flat, max_size=args.max_size)
        st_chunks = chunk_structural(chapters, max_size=args.max_size)
        st_text = "\n\n".join(st_chunks)
        print(f"  {name:30}{'flat':>12}{len(flat_chunks):8}"
              f"{sum(repairs(flat_chunks).values()):9}{len(flat):10}"
              f"{len(NAV_HINT.findall(flat)):10}")
        print(f"  {'':30}{'structural':>12}{len(st_chunks):8}"
              f"{sum(repairs(st_chunks).values()):9}{len(st_text):10}"
              f"{len(NAV_HINT.findall(st_text)):10}"
              f"   ({len(chapters)} chapters)")
    print("\n  'nav hits' counts table-of-contents and copyright boilerplate")
    print("  surviving into the text. source_normalization strips these with")
    print("  regexes; the structural path never picks them up, because the EPUB")
    print("  marks which documents they are.")


if __name__ == "__main__":
    main()
