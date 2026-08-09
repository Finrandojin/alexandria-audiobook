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
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

# The repo's one alias-aware speaker comparison, same as the legacy audit uses.
# Writing a local name-match here is how two scorers drift apart.
from experiments.scoring import alias_groups, same_speaker  # noqa: E402

DEFAULT_INPUTS = os.path.join(
    REPO, "ab_test_runtime", "results", "collect_all_20260722-155801", "inputs")


def load_gold(book):
    path = os.path.join(APP, "fixtures", f"attribution_gold_{book}.json")
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    rows = raw if isinstance(raw, list) else (raw.get("rows")
                                              or raw.get("entries") or [])
    # Aliases are load-bearing: BRITNEY/BRI-CHAN and ELIZARD/QUEEN ELIZARD are
    # one character each, and scoring without them reports name-form
    # differences as attribution errors.
    return ({r["id"]: r for r in rows if r.get("id")},
            alias_groups(raw) if isinstance(raw, dict) else ())


def norm_text(s):
    """Match key for a line: letters and digits only, casefolded."""
    return re.sub(r"[^0-9a-z]+", "", str(s or "").lower())


def index_entries(path):
    """-> {normalised line text: speaker} from a generated script.

    KEYED ON TEXT, NOT POSITION. This returned {entry_index: speaker} and the
    caller looked gold up by its own entry_index. Those indices come from
    different segmentations - the gold's are from source_run
    matrix_20260725-115148, and every fresh run re-segments the book - so index
    N in a new run is simply a different line. On 2026-08-06 that produced
    "single 0.0%, three_pass 0.7%" on 136 comparable lines for a book whose
    real accuracy is around 73%.

    The docstring above already named this hazard for the two arms against each
    other and guarded it. The same hazard against GOLD was not guarded. It only
    became visible because 0.0% is obviously broken; had it landed at 60 and 62
    it would have been believed.

    Lines whose normalised text is not unique within a script are dropped by the
    caller, because an ambiguous match is not a match. Same approach as
    collect_results.py's pipeline-repeat scoring.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return {}, collections.Counter()
    entries = doc if isinstance(doc, list) else (doc.get("entries") or [])
    out, occurrences = {}, collections.Counter()
    for e in entries:
        if not isinstance(e, dict):
            continue
        key = norm_text(e.get("text") or e.get("line"))
        if not key:
            continue
        occurrences[key] += 1
        if e.get("speaker"):
            out.setdefault(key, e["speaker"])
    return out, occurrences




def _is_reusable(out_path, expected_model=None):
    """True only when a previous run of this arm demonstrably finished.

    A leftover file proves nothing - the failed runs in this project all left
    output behind, which is the whole reason `accepted_chunk_count` exists. So
    reuse requires the sibling generation_quality record to say `complete`
    with every chunk accepted, and, when a model is named, to say it was that
    model. Anything else re-runs.
    """
    if not os.path.exists(out_path):
        return False
    quality = out_path + ".generation_quality.json"
    if not os.path.exists(quality):
        return False
    try:
        with open(quality, encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return False
    total = record.get("total_chunks")
    if record.get("status") != "complete" or not total:
        return False
    if record.get("accepted_chunk_count") != total:
        return False
    if expected_model and record.get("model_name") != expected_model:
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--books", nargs="+",
                    default=["grimgar03", "index18", "mushoku16",
                             "owarimonogatari3"])
    ap.add_argument("--inputs", default=DEFAULT_INPUTS)
    ap.add_argument("--work", default=os.path.join(
        REPO, "ab_test_runtime", "three_pass_vs_single"))
    ap.add_argument("--timeout", type=int, default=14400)
    ap.add_argument("--reuse-complete", action="store_true",
                    help="skip an arm whose output is already on disk AND "
                         "whose generation_quality record shows it completed "
                         "every chunk. Existence alone is not enough: a "
                         "partial or failed run leaves a file behind too.")
    ap.add_argument("--model", default=None,
                    help="with --reuse-complete, only reuse output recorded "
                         "as produced by this model. Without it, a reused arm "
                         "could silently come from a different model than the "
                         "arm it is compared against.")
    ap.add_argument("--pass2-on-exhaustion", choices=["fail", "fallback"],
                    default=None,
                    help="forwarded to three_pass_generate. Its own default "
                         "is 'fail' (abort the book, surfacing the failure "
                         "rate); 'fallback' labels unresolved spans UNKNOWN, "
                         "which is production behaviour and what an accuracy "
                         "comparison should measure.")
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
        gold, aliases = load_gold(book)
        if not gold:
            failures.append({"book": book, "error": "no gold"})
            continue

        produced = {}
        for arm, script in (("single", "generate_script.py"),
                            ("three_pass", "three_pass_generate.py")):
            out_path = os.path.join(args.work, f"{book}__{arm}.json")
            log = os.path.join(logs, f"tpvs_{book}_{arm}.log")
            t0 = time.time()
            if args.reuse_complete and _is_reusable(out_path, args.model):
                produced[arm] = (index_entries(out_path), 0.0)
                print(f"  {book:18} {arm:11} reused, "
                      f"{len(produced[arm][0][0])} entries (0m)")
                continue
            cmd = [py, "-u", os.path.join(APP, script), src,
                   "--output", out_path]
            # Only the three-pass arm has this switch, and only it needs one:
            # three_pass_generate defaults to on_exhaustion='fail', which
            # aborts the whole book when a single batch cannot be attributed.
            # That is the right default for surfacing a failure rate, and the
            # wrong one for an ACCURACY comparison - owarimonogatari3 lost a
            # 4454-entry book to one unattributable line, so 5.3 got one
            # scored book instead of two. 'fallback' is what production runs.
            if arm == "three_pass" and args.pass2_on_exhaustion:
                cmd += ["--pass2-on-exhaustion", args.pass2_on_exhaustion]
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
            print(f"  {book:18} {arm:11} ok, {len(produced[arm][0][0])} entries "
                  f"({mins:.0f}m)")

        # A book missing an arm is dropped whole. Scoring one arm against gold
        # while the other failed would publish a comparison that is not one.
        if len(produced) != 2:
            print(f"  {book}: dropped, both arms required")
            continue

        (single, single_occ) = produced["single"][0]
        (three, three_occ) = produced["three_pass"][0]
        # A gold line is comparable only when BOTH paths produced it and it is
        # unambiguous in both. Position cannot be used: all three segmentations
        # differ. See index_entries.
        common = []
        for g in gold.values():
            key = norm_text(g.get("line") or g.get("text"))
            if (key and single_occ.get(key) == 1 and three_occ.get(key) == 1
                    and key in single and key in three):
                common.append((key, g))
        row = {"book": book, "gold_lines": len(gold),
               "comparable": len(common),
               "single_entries": len(single), "three_entries": len(three),
               "single_minutes": round(produced["single"][1], 1),
               "three_minutes": round(produced["three_pass"][1], 1)}
        for arm, mapping in (("single", single), ("three_pass", three)):
            correct = sum(
                1 for key, g in common
                if same_speaker(mapping.get(key), g["expected_speaker"], aliases))
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
        for f in failures[:4]:
            print(f"    {f}")
        if any("Connection" in str(f) for f in failures) or failures:
            print("\n  If every arm failed, check the LLM server first:")
            print("    ./start_llama_server.sh")
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
        # A reused arm records 0 minutes because it was not run, not because
        # it was instant. Dividing by that produced "68900000000.0x" on the
        # first fallback run - a fabricated number printed with the same
        # authority as a measured one. Timing is only reported when every arm
        # in the comparison was actually timed.
        single_total = sum(r["single_minutes"] for r in results)
        three_total = sum(r["three_minutes"] for r in results)
        if single_total > 0 and three_total > 0:
            print(f"  three-pass costs {three_total / single_total:.1f}x "
                  "the single pass in wall time.")
        else:
            print("  wall-time comparison unavailable: an arm was reused "
                  "rather than run, so it has no measured duration.")

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

    # WRITING AN ARTIFACT IS NOT SUCCEEDING. On 2026-08-06 this exited 0 having
    # compared nothing - no LLM server was running, all four books failed, and
    # the artifact faithfully recorded that. The chain read rc=0 and logged OK.
    # A run that compared no books must be a failure to its caller.
    if not results:
        sys.exit(3)


if __name__ == "__main__":
    main()
