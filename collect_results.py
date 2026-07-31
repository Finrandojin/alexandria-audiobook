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
import collections, csv, glob, json, os, time

REPO = os.path.dirname(os.path.abspath(__file__))
E = os.path.join(REPO, "ab_test_runtime", "experiments")
rows, files = [], sorted(glob.glob(os.path.join(E, "*.json")))

for path in files:
    name = os.path.basename(path)
    if name.endswith(".ckpt"):
        continue
    try:
        d = json.load(open(path))
    except (ValueError, OSError) as exc:
        rows.append({"artifact": name, "note": f"UNREADABLE: {exc}"})
        continue
    m, rr = d.get("meta") or {}, d.get("rows") or []
    # Analysis artifacts are plain JSON and may legitimately use "rows" for a
    # count rather than a list of scored lines - segmentation_classifier.json
    # does. Iterating that raised TypeError and killed the whole index rather
    # than skipping one file, so the shape is checked, not assumed.
    if not isinstance(rr, list) or not rr:
        continue
    if not all(isinstance(r, dict) and "arm" in r for r in rr):
        rows.append({"artifact": name,
                     "note": "SKIPPED: 'rows' is not a list of scored arms"})
        continue
    env = m.get("lmstudio") or {}
    git = m.get("git") or {}
    by = collections.defaultdict(lambda: [0, 0])
    for r in rr:
        b = by[r["arm"]]
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
    for arm, (n, ok) in by.items():
        rows.append({
            "artifact": name,
            "experiment": m.get("experiment", ""),
            "book": book,
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
                                      time.localtime(m["finished"]))
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
            "book": "grimgar03", "model": "qwen/qwen3-14b",
            "env_tag": "local-lmstudio", "host": "mitch-linux", "endpoint": "",
            "backend": "lmstudio", "ctx": 16384, "parallel": 1, "kv": "f16",
            "arm": os.path.basename(f).split(".")[0], "n": n, "correct": ok,
            "accuracy_pct": round(ok / n * 100, 1),
            "validation": "n/a (pipeline output, not an ExperimentRecord)",
            "dirty": "", "commit": "", "elapsed_s": "",
            "finished": time.strftime("%m-%d %H:%M",
                                      time.localtime(os.path.getmtime(f)))})

out_csv = os.path.join(REPO, "results_index.csv")
cols = ["artifact", "experiment", "book", "model", "env_tag", "host", "backend",
        "ctx", "parallel", "kv", "arm", "n", "correct", "accuracy_pct",
        "validation", "dirty", "commit", "elapsed_s", "finished", "endpoint"]
with open(out_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)

md = [f"# Results index\n",
      f"Generated {time.strftime('%Y-%m-%d %H:%M')} from "
      f"`ab_test_runtime/experiments/` — {len(files)} artifacts, {len(rows)} arms.\n",
      "Regenerate with `python3 collect_results.py`. Machine-readable copy in "
      "`results_index.csv`.\n",
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
      "Arms with `valid=None` had no validation recorded at write time "
      "(`closed_set.json`, `two_by_two.json`, both pre-contract). They are "
      "unverified rather than known-bad.\n"]

for exp in sorted({r.get("experiment", "") for r in rows if r.get("experiment")}):
    sub = [r for r in rows if r.get("experiment") == exp]
    md.append(f"\n## {exp}\n")
    md.append("| book | model | env | backend | ctx | arm | n | acc | valid | dirty | elapsed |")
    md.append("|---|---|---|---|---:|---|---:|---:|---|---|---:|")
    for r in sorted(sub, key=lambda x: (x["book"], str(x["model"]), x["arm"])):
        md.append(f"| {r['book']} | {str(r['model']).split('/')[-1][:26]} | "
                  f"{r['env_tag']} | {str(r['backend'])[:18]} | {r['ctx']} | "
                  f"{r['arm']} | {r['n']} | {r['accuracy_pct']}% | "
                  f"{r['validation']} | {r['dirty']} | {r['elapsed_s']}s |")

open(os.path.join(REPO, "RESULTS_INDEX.md"), "w").write("\n".join(md) + "\n")
print(f"{len(files)} artifacts -> {len(rows)} arm rows")
print("wrote RESULTS_INDEX.md and results_index.csv")
