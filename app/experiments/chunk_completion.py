"""Goal 3.1: what fraction of chunks complete, per model, on real books.

WHY THE CURRENT NUMBER CANNOT ANSWER IT. 3.1 records "mushoku16 9/9 clean,
grimgar03 and owarimonogatari3 failed chunk 1 outright" from a 2026-08-06 run,
and carries its own caveat: that run used **qwen2.5-14b**, which is not the
shipped model. The same mistake shaped goal 1.2, where a figure measured on a
weaker model made a solved problem look like the largest opportunity in the app
until it was re-measured. So this does not extend the old number - it asks the
question against every model the app has actually generated books with.

WHERE THE EVIDENCE IS. Each saved book has a sibling
`<name>.json.generation_quality.json` recording `model_name`, `total_chunks`,
`accepted_chunk_count` and a per-chunk list. That is the metric 3.1 asks for -
chunks completing without exhausting retries - already written down for every
book this app has generated, so no LLM inference is needed to read it.

WHAT IS COUNTED, AND WHAT IS NOT. A chunk is complete when it was accepted.
`accepted_chunk_count` against `total_chunks` is the completion rate. Books
whose status is not `complete` are reported separately rather than dropped: a
book that stopped early is the failure this goal is about, and averaging it
away with the finished ones would hide exactly what is being measured.

Per-model figures are reported with their book counts attached. A model that
generated one book is not evidence about that model, and a rate quoted without
its denominator invites reading it as though it were.
"""
import argparse
import collections
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if not os.path.isdir(os.path.join(REPO, "scripts")):
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "app"))

SUFFIX = ".json.generation_quality.json"


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scripts", default=os.path.join(REPO, "scripts"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "chunk_completion.json"))
    args = ap.parse_args()

    per_model = collections.defaultdict(
        lambda: {"books": 0, "total_chunks": 0, "accepted_chunks": 0,
                 "incomplete_books": [], "book_rates": []})
    books = []
    for path in sorted(glob.glob(os.path.join(args.scripts, "*" + SUFFIX))):
        name = os.path.basename(path)[:-len(SUFFIX)]
        data = load(path)
        model = data.get("model_name") or "<unrecorded>"
        total = data.get("total_chunks") or 0
        accepted = data.get("accepted_chunk_count") or 0
        status = data.get("status") or "<unrecorded>"
        if not total:
            continue
        rate = 100.0 * accepted / total
        entry = per_model[model]
        entry["books"] += 1
        entry["total_chunks"] += total
        entry["accepted_chunks"] += accepted
        entry["book_rates"].append(round(rate, 1))
        if status != "complete" or accepted < total:
            entry["incomplete_books"].append(
                {"book": name, "status": status,
                 "accepted": accepted, "total": total})
        books.append({"book": name, "model": model, "status": status,
                      "total_chunks": total, "accepted_chunks": accepted,
                      "completion_pct": round(rate, 1)})

    summary = {}
    for model, entry in per_model.items():
        summary[model] = {
            "books": entry["books"],
            "chunks": entry["total_chunks"],
            "completion_pct": round(
                100.0 * entry["accepted_chunks"] / entry["total_chunks"], 2),
            "books_not_fully_complete": len(entry["incomplete_books"]),
            "worst_book_pct": min(entry["book_rates"]),
            "incomplete": entry["incomplete_books"][:10],
        }

    chunks = sum(e["total_chunks"] for e in per_model.values())
    accepted = sum(e["accepted_chunks"] for e in per_model.values())
    result = {
        "scope": "every saved book with a generation_quality record; "
                 "completion = accepted_chunk_count / total_chunks",
        "target_pct": 99.0,
        "books": len(books), "chunks": chunks,
        "overall_completion_pct": round(100.0 * accepted / chunks, 2)
        if chunks else None,
        "per_model": summary, "per_book": books,
    }

    from utils import atomic_json_write
    atomic_json_write(result, args.out)

    print("=== goal 3.1: chunk completion, target 99% ===")
    print(f"  {len(books)} books, {chunks} chunks, overall "
          f"{result['overall_completion_pct']}%\n")
    for model, s in sorted(summary.items(),
                           key=lambda kv: -kv[1]["chunks"]):
        flag = "" if s["completion_pct"] >= 99.0 else "   BELOW TARGET"
        print(f"  {model[:44]:46} {s['completion_pct']:6.2f}%  "
              f"books={s['books']:<3} chunks={s['chunks']:<6} "
              f"worst book {s['worst_book_pct']}%{flag}")
    incomplete = [b for b in books if b["completion_pct"] < 100.0]
    print(f"\n  books not at 100%: {len(incomplete)}")
    for b in sorted(incomplete, key=lambda x: x["completion_pct"])[:8]:
        print(f"    {b['book'][:38]:40} {b['completion_pct']:6.1f}%  "
              f"({b['accepted_chunks']}/{b['total_chunks']})  {b['model'][:22]}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
