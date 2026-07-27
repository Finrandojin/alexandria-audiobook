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
    if not rr:
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
    if "grimgar03" in gold:
        book = "grimgar03"
    elif "attribution_gold_random" in gold:
        book = "mushoku16"
    else:
        book = os.path.basename(gold) or "?"
    endpoint = str(m.get("endpoint", ""))
    if "thundercompute" in endpoint:
        env_tag = "cloud-a6000"
    elif env.get("backend"):
        env_tag = "local-" + str(env["backend"]).split()[0].replace("llama.cpp-", "")
    elif "localhost" in endpoint or "127.0.0.1" in endpoint:
        env_tag = "local-lmstudio"
    else:
        env_tag = "?"
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
      "from its recorded commit.\n"]

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
