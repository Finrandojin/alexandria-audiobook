"""Index every experiment artifact into one table.

There are now artifacts from two machines, three inference stacks and two books,
several of which differ only by a filename suffix - and a cloud run has already
overwritten a local one once, before EXPERIMENT_TAG existed. Reading them one at
a time invites exactly that confusion, so this flattens all of them into one
CSV and one markdown summary, regenerable at any time.

Deliberately reports provenance next to every number: validation status, dirty
tree, endpoint and harness fingerprint. A result whose provenance is weak should
be visible as such in the same row, not discoverable by opening the file.
"""
import argparse, collections, csv, glob, io, json, os, re, sys, time

REPO = os.path.dirname(os.path.abspath(__file__))
E = os.path.join(REPO, "ab_test_runtime", "experiments")
AUDIT = os.path.join(REPO, "ab_test_runtime", "audit")
parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--check", action="store_true")
args = parser.parse_args()

def _load_audit(name):
    path = os.path.join(AUDIT, name)
    try:
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"evidence audit is unreadable: {path}: {exc}") from exc
    return {row["artifact"]: row["classification"]
            for row in doc.get("artifacts", [])}

structural_status = _load_audit("artifact_structural_audit.json")
legacy_status = _load_audit("legacy_attribution_audit.json")

def get_evidence_status(name):
    return legacy_status.get(name, structural_status.get(name, "not_audited"))

rows, files = [], sorted(glob.glob(os.path.join(E, "*.json")))

for path in files:
    name = os.path.basename(path)
    if name.endswith(".ckpt"):
        continue
    try:
        d = json.load(open(path))
    except (ValueError, OSError) as exc:
        rows.append({"artifact": name, "evidence_status": get_evidence_status(name),
                     "note": f"UNREADABLE: {exc}"})
        continue
    # The same lesson as the "rows" check below, one level up: a manifest can
    # legitimately be a LIST (validation_manifest, chapter_manifest and
    # fix_verification all are), and calling .get on it raised AttributeError
    # and killed the whole index rather than skipping one file. Guard the
    # top-level shape too, not just the field.
    if not isinstance(d, dict):
        rows.append({"artifact": name, "evidence_status": get_evidence_status(name),
                     "note": f"SKIPPED: not a result object ({type(d).__name__})"})
        continue
    if isinstance(d.get("provenance"), dict) and not isinstance(d.get("meta"), dict):
        rows.append({"artifact": name, "evidence_status": get_evidence_status(name),
                     "note": "NOT INDEXED: TTS provenance artifact; read its "
                             "per-book/category summary directly"})
        continue
    m, rr = d.get("meta") or {}, d.get("rows") or []
    # Analysis artifacts are plain JSON and may legitimately use "rows" for a
    # count rather than a list of scored lines - segmentation_classifier.json
    # does. Iterating that raised TypeError and killed the whole index rather
    # than skipping one file, so the shape is checked, not assumed.
    # THE SILENT ONE. Every other skip path above records a reason; this one
    # was a bare `continue`, so an artifact with no `rows` list vanished from
    # the index with no trace at all - not a row, not a note, nothing. On
    # 2026-08-04 `pitch_stability.json` and `pitch_separation.json` were
    # regenerated into the index and simply were not there, and the index still
    # claimed to cover "every experiment artifact".
    #
    # An artifact this table cannot represent must SAY SO. Silence reads as
    # "no such result", which is the most expensive kind of wrong.
    if not isinstance(rr, list) or not rr:
        rows.append({"artifact": name, "evidence_status": get_evidence_status(name),
                     "note": "NOT INDEXED: no 'rows' list - this table only "
                             "represents per-arm attribution results"})
        continue
    if not all(isinstance(r, dict) and "arm" in r for r in rr):
        rows.append({"artifact": name, "evidence_status": get_evidence_status(name),
                     "note": "SKIPPED: 'rows' is not a list of scored arms"})
        continue
    env = m.get("lmstudio") or {}
    git = m.get("git") or {}
    # Aggregate by (BOOK, arm), not arm alone. gold_path names only the FIRST
    # fixture, so a multi-book artifact was reported as one book: on
    # 2026-08-06 distill_eval__pdnc_only covered four books and
    # lora_serving_eval covered two, and both were labelled grimgar03. The
    # numbers were right and the scope was understated, which is the kind of
    # error that survives a review because nothing looks wrong.
    #
    # Row ids carry the book where the harness wrote one ("grimgar03:
    # grimgar03-00001"). Where they do not, everything falls into one bucket
    # and the metadata book is used, exactly as before.
    def _row_book(r):
        rid = str(r.get("id") or "")
        return rid.split(":")[0] if ":" in rid else None

    by = collections.defaultdict(lambda: [0, 0])
    for r in rr:
        b = by[(_row_book(r), r["arm"])]
        b[0] += 1
        b[1] += bool(r.get("correct"))
    # Derive book and environment from METADATA, not the filename. Filename
    # parsing mislabelled every pre-EXPERIMENT_TAG artifact - "closed_set__qwen__
    # qwen3-14b.json" split to book="qwen" - and an index that mislabels books is
    # worse than no index. gold_path names the fixture, which names the book.
    gold = str(m.get("gold_path", ""))
    # attribution_gold_random.json is mushoku16's original fixture, named
    # before there was more than one book. Everything since is
    # attribution_gold_<book>.json, so match the book names directly and fall
    # back to the filename only when nothing is recognised - a lookup that
    # quietly returns a filename groups nothing and is how index18 and
    # owarimonogatari3 rows ended up unjoinable to the rest.
    KNOWN = ("grimgar03", "mushoku16", "index18", "owarimonogatari3")
    book = next((b for b in KNOWN if b in gold), None)
    if book is None:
        book = "mushoku16" if "attribution_gold_random" in gold else (
            os.path.basename(gold) or "?")
    endpoint = str(m.get("endpoint", ""))
    # WHICH MACHINE, then which stack. Endpoint alone is not enough: a run
    # executed ON the rented instance uses 127.0.0.1 loopback, which an
    # endpoint-only rule labels "local" - and that silently merged the A6000's
    # qwen3-32b arms with genuinely local runs. Host is recorded in two places;
    # prefer the environment the caller verified, fall back to the machine that
    # ran the harness.
    host = str(env.get("host") or m.get("host") or "")
    if "thunder" in host.lower() or "thundercompute" in endpoint:
        machine = "cloud-a6000"
    else:
        machine = "local"
    backend = str(env.get("backend") or "")
    if backend:
        stack = backend.split()[0].replace("llama.cpp-", "llamacpp-")
    else:
        stack = "lmstudio"
    env_tag = f"{machine}-{stack}"
    for (row_book, arm), (n, ok) in sorted(
            by.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        rows.append({
            "artifact": name,
            "evidence_status": get_evidence_status(name),
            "experiment": m.get("experiment", ""),
            # The book the ROWS say, when they say one. gold_path names only
            # the first fixture and understated multi-book runs.
            "book": row_book or book,
            "model": m.get("model", ""),
            "env_tag": env_tag,
            "host": m.get("host", ""),
            "endpoint": m.get("endpoint", ""),
            "backend": env.get("backend", "lmstudio"),
            "ctx": env.get("context_length", ""),
            "parallel": env.get("parallel", ""),
            "kv": env.get("kv_cache", "f16?"),
            "arm": arm,
            "n": n,
            "correct": ok,
            "accuracy_pct": round(ok / n * 100, 1) if n else "",
            "validation": ("ok" if m.get("validation") == "ok"
                           else str(m.get("validation"))[:40]),
            "dirty": git.get("dirty"),
            "commit": (git.get("commit") or "")[:8],
            "elapsed_s": m.get("elapsed_s", ""),
            "finished": time.strftime("%m-%d %H:%M",
                                      time.gmtime(m["finished"]))
                        if m.get("finished") else "",
        })

# Pipeline repeats are three_pass_generate outputs, not ExperimentRecord
# artifacts, so the scan above cannot see them - and the determinism finding
# they produced (SD 0.00 across eight runs) lived only in a log file. Score them
# here so the consolidated index actually holds all the data.
_reps = os.path.join(REPO, "ab_test_runtime", "pipeline_repeats")
if os.path.isdir(_reps):
    import re as _re
    _gold = json.load(open(os.path.join(
        REPO, "app", "fixtures", "attribution_gold_grimgar03_provisional.json")))
    _AL = [{n.upper() for n in g} for g in _gold.get("aliases", [])]
    def _same(a, b):
        a, b = (a or "").upper(), (b or "").upper()
        return a == b or any(a in g and b in g for g in _AL)
    def _norm(t):
        return _re.sub(r"\W+", "", t or "").lower()
    for f in sorted(glob.glob(os.path.join(_reps, "run*.threepass_checkpoint.json"))):
        try:
            d = json.load(open(f))
        except (ValueError, OSError):
            continue
        seg = d.get("segmented") or []
        occ = collections.Counter(_norm(e.get("text")) for e in seg)
        idx = {}
        for e in (x for x in (d.get("named") or []) if x):
            idx.setdefault(_norm(e.get("text")), e.get("speaker"))
        n = ok = 0
        for g in _gold["entries"]:
            k = _norm(g["line"])
            if occ.get(k) == 1 and k in idx:
                n += 1
                ok += _same(idx[k], g["expected_speaker"])
        if not n:
            continue
        rows.append({
            "artifact": os.path.basename(f), "experiment": "pipeline_repeat",
            "evidence_status": "not_audited",
            "book": "grimgar03", "model": "qwen/qwen3-14b",
            "env_tag": "local-lmstudio", "host": "mitch-linux", "endpoint": "",
            "backend": "lmstudio", "ctx": 16384, "parallel": 1, "kv": "f16",
            "arm": os.path.basename(f).split(".")[0], "n": n, "correct": ok,
            "accuracy_pct": round(ok / n * 100, 1),
            "validation": "n/a (pipeline output, not an ExperimentRecord)",
            "dirty": "", "commit": "", "elapsed_s": "",
            "finished": time.strftime("%m-%d %H:%M",
                                      time.gmtime(os.path.getmtime(f)))})

out_csv = os.path.join(REPO, "results_index.csv")
# `note` is last but it is NOT optional: three code paths write it and it was
# missing from this list, so extrasaction="ignore" silently threw every skip
# reason away. Un-indexed artifacts appeared as a filename followed by nineteen
# empty fields, which looks like a broken row rather than a deliberate note.
cols = ["artifact", "evidence_status", "experiment", "book", "model", "env_tag", "host", "backend",
        "ctx", "parallel", "kv", "arm", "n", "correct", "accuracy_pct",
        "validation", "dirty", "commit", "elapsed_s", "finished", "endpoint",
        "note"]
csv_buffer = io.StringIO(newline="")
with csv_buffer as fh:
    # csv.excel defaults to CRLF. The repository is LF-normalized, so adding a
    # new result made `git diff --check` report every new CSV row as trailing
    # whitespace even though the fields were valid.
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore",
                       lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    csv_text = fh.getvalue()

md = [f"# Results index\n",
      f"Generated {time.strftime('%Y-%m-%d %H:%M')} from "
      f"`ab_test_runtime/experiments/` — {len(files)} artifacts, "
      f"{len([r for r in rows if r.get('arm')])} arms.\n",
      "Regenerate with `python3 collect_results.py`. Machine-readable copy in "
      "`results_index.csv`.\n",
      "`evidence_status` comes from the committed audit snapshots. "
      "`supported_structure` validates provenance shape only; "
      "`supported_measurement` is the strongest attribution classification. "
      "`not_audited` is explicit and must not be treated as support.\n",
      "`dirty=True` means tracked files were modified when the artifact was "
      "written: the numbers are inspectable but the run is not reproducible "
      "from its recorded commit.\n",
      # `valid=ok` only means the artifact is internally consistent - its rows
      # agree with its summary. It cannot know that an arm's INPUTS were built
      # from labels the gold standard later replaced, which is exactly the
      # closed-oracle case. Without this note the index shows closed-oracle at
      # 83.0% next to real arms, with an ok beside it.
      "**`closed-oracle` arms are invalidated.** Their candidate sets were "
      "built from the pre-gold labels, so the arm was shown shortlists derived "
      "from answers that have since changed. `valid=ok` on those rows means "
      "internally consistent, NOT trustworthy — do not read them as results.\n",
      "`closed_set.json` and `two_by_two.json` predate the environment "
      "contract and captured no `context_length` or `parallel`, which cannot "
      "be reconstructed. Their rows and summaries were recomputed and are "
      "internally consistent, but the runs are not comparable to artifacts "
      "that record an environment: inspectable, not citable.\n"]

for exp in sorted({r.get("experiment", "") for r in rows if r.get("experiment")}):
    sub = [r for r in rows if r.get("experiment") == exp]
    md.append(f"\n## {exp}\n")
    md.append("| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |")
    md.append("|---|---|---|---|---:|---|---:|---:|---|---|---|---:|")
    for r in sorted(sub, key=lambda x: (x["book"], str(x["model"]), x["arm"])):
        md.append(f"| {r['book']} | {str(r['model']).split('/')[-1][:26]} | "
                  f"{r['env_tag']} | {str(r['backend'])[:18]} | {r['ctx']} | "
                  f"{r['arm']} | {r['n']} | {r['accuracy_pct']}% | "
                  f"{r['evidence_status']} | "
                  f"{r['validation']} | {r['dirty']} | {r['elapsed_s']}s |")

# Artifacts this table cannot represent, listed rather than omitted. Most are
# TTS and audio experiments whose results are per-segment WER and failure
# counts, not per-arm accuracy - a different shape, not a broken one. Naming
# them keeps "not in the index" from being read as "no such result".
skipped = [r for r in rows if r.get("note")]
if skipped:
    md.append("\n## Not indexed\n")
    md.append("These artifacts exist and hold real results; this table only "
              "represents per-arm attribution accuracy, so they cannot be "
              "rendered as rows. Read them directly.\n")
    md.append("| artifact | why |")
    md.append("|---|---|")
    for r in sorted(skipped, key=lambda x: x["artifact"]):
        md.append(f"| `{r['artifact']}` | {r['note']} |")

md_text = "\n".join(md) + "\n"
md_path = os.path.join(REPO, "RESULTS_INDEX.md")
if args.check:
    try:
        with open(out_csv, encoding="utf-8", newline="") as handle:
            old_csv = handle.read()
        with open(md_path, encoding="utf-8") as handle:
            old_md = handle.read()
    except OSError as exc:
        raise SystemExit(f"results index is unreadable: {exc}") from exc
    normalize = lambda text: re.sub(r"Generated .*? from ", "Generated TIME from ", text, count=1)
    if old_csv != csv_text or normalize(old_md) != normalize(md_text):
        raise SystemExit("results index is stale; regenerate with python3 collect_results.py")
    print("results index is current")
    sys.exit(0)
with open(out_csv, "w", encoding="utf-8", newline="") as handle:
    handle.write(csv_text)
with open(md_path, "w", encoding="utf-8") as handle:
    handle.write(md_text)
print(f"{len(files)} artifacts -> "
      f"{len([r for r in rows if r.get('arm')])} arm rows, "
      f"{len(skipped)} not indexed")
print("wrote RESULTS_INDEX.md and results_index.csv")
