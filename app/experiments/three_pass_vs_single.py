"""Is three-pass generation better than the single pass that actually ships?

THE MISSING MEASUREMENT. `three_pass_generate.py` exists, has six settings in
config_settings.py, ships three prompt files, and is invoked by nothing:
`/api/generate_script` runs `generate_script.py`. Its own docstring calls it "a
side-by-side alternative... the single-pass path is untouched".

So the repository carries a second generation architecture that has never been
scored against the one in production. That is the gap this closes, and the
answer decides whether the module is wired up or deleted - both are fine
outcomes, and carrying an unmeasured alternative indefinitely is not.

WHAT IS COMPARED. Both paths run over the same source text, at the same
temperature, against the same attribution gold. The metric is speaker accuracy
on gold-labelled lines, which is what the four gold sets exist for.

PAIRED ON LINE ID, NOT POOLED. The two paths segment independently, so they do
not produce the same entries - three-pass may split a paragraph the single
pass keeps whole. Only lines both paths produced AND gold covers are scored.
Comparing different line sets is the asymmetry that has bitten this repo
repeatedly, and here it would be easy to miss because both numbers would look
reasonable.

REPORTED PER BOOK. Book identity dominates method in this project: across ~470
scored arms the median book differs by 19 points before any method is chosen,
and mushoku16 and grimgar03 differ by 24 on the same method. A pooled figure
would mostly measure which books were included.
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

DEFAULT_INPUTS = os.path.join(
    REPO, "ab_test_runtime", "results", "collect_all_20260722-155801", "inputs")


def load_gold(book):
    path = os.path.join(APP, "fixtures", f"attribution_gold_{book}.json")
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    rows = raw if isinstance(raw, list) else (raw.get("rows")
                                              or raw.get("entries") or [])
    return {r["id"]: r for r in rows if r.get("id")}


def normalise(name):
    """Compare speakers the way the other scorers do: case and spacing only."""
    return " ".join(str(name or "").split()).upper()


def index_entries(path):
    """-> {entry_index: speaker} from a generated script."""
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return {}
    entries = doc if isinstance(doc, list) else (doc.get("entries") or [])
    out = {}
    for i, e in enumerate(entries, 1):
        if isinstance(e, dict) and e.get("speaker"):
            out[i] = e["speaker"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--books", nargs="+",
                    default=["grimgar03", "index18", "mushoku16",
                             "owarimonogatari3"])
    ap.add_argument("--inputs", default=DEFAULT_INPUTS)
    ap.add_argument("--work", default=os.path.join(
        REPO, "ab_test_runtime", "three_pass_vs_single"))
    ap.add_argument("--timeout", type=int, default=14400)
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "three_pass_vs_single.json"))
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    logs = os.path.join(REPO, "ab_test_runtime", "logs")
    os.makedirs(logs, exist_ok=True)
    py = sys.executable

    results, failures = [], []
    for book in args.books:
        src = os.path.join(args.inputs, f"{book}.txt")
        if not os.path.exists(src):
            failures.append({"book": book, "error": f"no source at {src}"})
            print(f"  {book}: SKIPPED, no source text")
            continue
        gold = load_gold(book)
        if not gold:
            failures.append({"book": book, "error": "no gold"})
            continue

        produced = {}
        for arm, script in (("single", "generate_script.py"),
                            ("three_pass", "three_pass_generate.py")):
            out_path = os.path.join(args.work, f"{book}__{arm}.json")
            log = os.path.join(logs, f"tpvs_{book}_{arm}.log")
            t0 = time.time()
            cmd = [py, "-u", os.path.join(APP, script), src,
                   "--output", out_path]
            try:
                with open(log, "w", encoding="utf-8") as fh:
                    rc = subprocess.run(cmd, stdout=fh,
                                        stderr=subprocess.STDOUT, cwd=APP,
                                        timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                rc = -1
            mins = (time.time() - t0) / 60
            if rc != 0 or not os.path.exists(out_path):
                failures.append({"book": book, "arm": arm, "rc": rc})
                print(f"  {book:18} {arm:11} FAILED rc={rc} ({mins:.0f}m)")
                break
            produced[arm] = (index_entries(out_path), mins)
            print(f"  {book:18} {arm:11} ok, {len(produced[arm][0])} entries "
                  f"({mins:.0f}m)")

        # A book missing an arm is dropped whole. Scoring one arm against gold
        # while the other failed would publish a comparison that is not one.
        if len(produced) != 2:
            print(f"  {book}: dropped, both arms required")
            continue

        single, three = produced["single"][0], produced["three_pass"][0]
        # Gold is keyed by entry_index within the book, so only indices BOTH
        # paths produced and gold covers can be compared.
        common = [g for g in gold.values()
                  if int(g.get("entry_index") or 0) in single
                  and int(g.get("entry_index") or 0) in three]
        row = {"book": book, "gold_lines": len(gold),
               "comparable": len(common),
               "single_entries": len(single), "three_entries": len(three),
               "single_minutes": round(produced["single"][1], 1),
               "three_minutes": round(produced["three_pass"][1], 1)}
        for arm, mapping in (("single", single), ("three_pass", three)):
            correct = sum(
                1 for g in common
                if normalise(mapping.get(int(g["entry_index"])))
                == normalise(g["expected_speaker"]))
            row[arm] = {"correct": correct,
                        "accuracy": correct / len(common) if common else None}
        results.append(row)
        d = ((row["three_pass"]["accuracy"] or 0)
             - (row["single"]["accuracy"] or 0)) * 100
        print(f"  {book:18} comparable {len(common):4}  single "
              f"{row['single']['accuracy']*100:5.1f}%  three "
              f"{row['three_pass']['accuracy']*100:5.1f}%  {d:+5.1f}")

    if not results:
        print("\nno book produced both arms; nothing to compare")
    else:
        print(f"\n  {'book':20}{'n':>6}{'single':>9}{'three':>9}{'delta':>8}")
        for r in results:
            print(f"  {r['book']:20}{r['comparable']:6}"
                  f"{r['single']['accuracy']*100:8.1f}%"
                  f"{r['three_pass']['accuracy']*100:8.1f}%"
                  f"{(r['three_pass']['accuracy']-r['single']['accuracy'])*100:+8.1f}")
        wins = sum(1 for r in results
                   if r["three_pass"]["accuracy"] > r["single"]["accuracy"])
        print(f"\n  three-pass ahead on {wins} of {len(results)} books")
        print("  Per book, deliberately: book identity dominates method here, "
              "so a\n  pooled figure would mostly report which books were "
              "included.")
        cost = sum(r["three_minutes"] for r in results) / max(
            sum(r["single_minutes"] for r in results), 1e-9)
        print(f"  three-pass costs {cost:.1f}x the single pass in wall time.")

    doc = {"books": args.books, "results": results, "failures": failures}
    try:
        from experiments.provenance import provenance
        doc["provenance"] = provenance(__file__, args)
    except Exception as exc:                            # noqa: BLE001
        doc["provenance"] = {"error": str(exc)[:120]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
