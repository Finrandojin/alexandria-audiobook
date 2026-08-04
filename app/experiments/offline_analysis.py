"""Three analyses that consume existing artifacts and need no GPU.

The review's recommended order puts these ahead of any new sweep, on the
grounds that the next useful information comes from decomposing what we already
measured rather than adding another row to an unresolved model table. All three
run against committed artifacts, so they can execute while the GPU is busy and
cannot disturb anything.

  1. ORACLE ERROR STRATA (review priority 6)
     The 24-34% failure rate when the true speaker is among five candidates is
     the least explained number in the ledger. Recall does not explain it, nor
     roster size, nor model size. This buckets those errors by a mechanical
     proxy for cause and separates errors where strong models AGREE - which are
     the ones most likely to be bad gold rather than model failure.

  2. FIXTURE REPRESENTATIVENESS (review priority 7)
     The fixture was random at construction, but ambiguity and unique-text
     filters change the evaluated population afterwards. If the scored rows are
     enriched for hard lines, the ~50-70% range describes a filtered subset and
     not representative dialogue - which would make the "plateau" an artefact of
     what we chose to measure.

  3. SELECTIVE-THINKING ROUTING FEATURES (review priority 1)
     `thinking` is the only intervention with a significant production result
     and its blocker is cost, not effect. Routing needs only correlation with
     difficulty, a far lower bar than the acceptance-criterion confidence signal
     that failed at 17% coverage. This labels every paired row RESCUE / HARM /
     NEUTRAL and reports which cheap features separate them.

Nothing here decides anything. Each section reports what the existing data can
and cannot support, and names the experiment that would settle it.
"""
import collections
import json, os, re, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
E = REPO + "/ab_test_runtime/experiments/"
M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
SPEECH_VERB = (r"\b(said|asked|replied|answered|shouted|whispered|muttered|"
               r"called|cried|yelled|groaned|sighed|laughed|nodded|exclaimed|"
               r"bellowed|agreed|told|added|continued|began|offered)\b")


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


def load(path):
    with open(E + path) as fh:
        return json.load(fh)


def rows_by_arm(doc):
    out = collections.defaultdict(list)
    for r in doc["rows"]:
        out[r["arm"]].append(r)
    return out


# --------------------------------------------------------------- 1. oracle
def oracle_strata():
    print("=" * 72)
    print("1. ORACLE ERROR STRATA - why does a five-way choice fail?")
    print("=" * 72)
    files = {f: load(f) for f in os.listdir(E)
             if f.startswith("closed_set__") and f.endswith(".json")}
    per_model = {}
    for name, doc in files.items():
        arms = rows_by_arm(doc)
        if "closed-oracle" not in arms:
            continue
        book = ("grimgar03" if "grimgar03" in str(doc["meta"].get("gold_path"))
                else "mushoku16")
        per_model[(doc["meta"]["model"], book, name)] = arms["closed-oracle"]

    # Errors shared by every model on a book are the interesting ones: a line
    # that several independent models get wrong with the answer in front of them
    # is a candidate bad-gold row, not a capability ceiling.
    for book in ("mushoku16", "grimgar03"):
        sets = {m: {r["id"] for r in rs if not r["correct"]}
                for (m, b, _), rs in per_model.items() if b == book}
        if len(sets) < 2:
            continue
        allids = set.intersection(*[{r["id"] for r in rs}
                                    for (m, b, _), rs in per_model.items()
                                    if b == book])
        shared = set.intersection(*sets.values())
        anyerr = set.union(*sets.values())
        print(f"\n  {book}: {len(sets)} model runs, {len(allids)} shared rows")
        print(f"    every model wrong on : {len(shared)} rows "
              f"({len(shared)/len(allids)*100:.1f}%)")
        print(f"    at least one wrong   : {len(anyerr)} rows "
              f"({len(anyerr)/len(allids)*100:.1f}%)")
        print(f"    -> {len(shared)} rows are where the gold, the context, or the "
              f"task is the suspect; the rest are model variance")

        # what did they answer instead, on the unanimous failures?
        any_run = next(rs for (m, b, _), rs in per_model.items() if b == book)
        by_id = {r["id"]: r for r in any_run}
        pat = collections.Counter()
        for i in list(shared)[:400]:
            r = by_id.get(i)
            if not r:
                continue
            pred, exp = (r["predicted"] or ""), r["expected"]
            cands = [c.upper() for c in (r["candidates"] or [])]
            if pred == "UNKNOWN":
                pat["answered UNKNOWN"] += 1
            elif pred not in cands:
                pat["answered off-list"] += 1
            elif pred in cands:
                pat["picked another candidate"] += 1
        for k, v in pat.most_common():
            print(f"      {k:28} {v}")
        print("    Mechanical strata only. Distinguishing addressee inversion "
              "from\n    turn-taking failure needs the blind adjudication the "
              "review asks for.")


# ------------------------------------------------- 2. representativeness
def representativeness():
    print("\n" + "=" * 72)
    print("2. FIXTURE REPRESENTATIVENESS - is the scored subset harder?")
    print("=" * 72)
    for book, goldfile in (("grimgar03", "attribution_gold_grimgar03_provisional.json"),
                           ("mushoku16", "attribution_gold_random.json")):
        gold = json.load(open(REPO + "/app/fixtures/" + goldfile))
        cp = json.load(open(M + INPUT_RUN + f"/{book}/result.json.threepass_checkpoint.json"))
        seg = cp["segmented"]
        spoken = [e for e in seg if e.get("type") != "NARRATOR"]
        occ = collections.Counter(norm(e.get("text")) for e in seg)
        scored = [g for g in gold["entries"] if occ[norm(g["line"])] == 1]
        dropped = [g for g in gold["entries"] if occ[norm(g["line"])] != 1]

        def stats(texts):
            lens = [len(t) for t in texts if t]
            tagged = sum(1 for t in texts if re.search(SPEECH_VERB, (t or "").lower()))
            return (statistics.median(lens) if lens else 0,
                    tagged / len(texts) * 100 if texts else 0)

        all_med, all_tag = stats([e.get("text") for e in spoken])
        sc_med, sc_tag = stats([g["line"] for g in scored])
        print(f"\n  {book}")
        print(f"    all spoken lines : {len(spoken):5}  median {all_med:4} chars  "
              f"speech tag nearby-in-line {all_tag:5.1f}%")
        print(f"    scored gold rows : {len(scored):5}  median {sc_med:4} chars  "
              f"speech tag nearby-in-line {sc_tag:5.1f}%")
        print(f"    dropped by the unique-text filter: {len(dropped)} "
              f"({len(dropped)/max(len(gold['entries']),1)*100:.1f}% of the fixture)")
        if dropped:
            dl = [len(g["line"]) for g in dropped]
            print(f"      dropped rows median {statistics.median(dl):.0f} chars - "
                  f"repeated text is SHORT text, so the filter removes short lines")
        amb = sum(1 for g in gold["entries"]
                  if str(g.get("expected_speaker", "")).upper() in ("AMBIGUOUS", "UNKNOWN"))
        print(f"    fixture rows marked AMBIGUOUS/UNKNOWN: {amb}")


# ----------------------------------------------------- 3. routing features
def routing_features():
    print("\n" + "=" * 72)
    print("3. SELECTIVE-THINKING ROUTING - what separates RESCUE from HARM?")
    print("=" * 72)
    pairs = [("grimgar03", "because_production__grimgar03__qwen__qwen3-14b__local.json"),
             ("mushoku16", "because_production__mushoku16__qwen__qwen3-14b__local.json")]
    for book, f in pairs:
        try:
            doc = load(f)
        except FileNotFoundError:
            continue
        arms = {a: {r["id"]: r for r in rs} for a, rs in rows_by_arm(doc).items()}
        if "baseline" not in arms or "thinking" not in arms:
            continue
        cp = json.load(open(M + INPUT_RUN + f"/{book}/result.json.threepass_checkpoint.json"))
        seg = cp["segmented"]
        pos = {norm(e["text"]): i for i, e in enumerate(seg)}
        b, t = arms["baseline"], arms["thinking"]
        labels = collections.Counter()
        feats = collections.defaultdict(lambda: collections.defaultdict(list))
        for i in set(b) & set(t):
            lab = ("RESCUE" if t[i]["correct"] and not b[i]["correct"] else
                   "HARM" if b[i]["correct"] and not t[i]["correct"] else "NEUTRAL")
            labels[lab] += 1
            line = b[i]["line"]
            j = pos.get(norm(line))
            near = ""
            if j is not None:
                near = " ".join((seg[k].get("text") or "")
                                for k in range(max(0, j - 1), min(len(seg), j + 2)))
            feats[lab]["len"].append(len(line))
            feats[lab]["tag"].append(1 if re.search(SPEECH_VERB, near.lower()) else 0)
            feats[lab]["quoted"].append(1 if line.strip().startswith(('"', '“')) else 0)
        n = sum(labels.values())
        print(f"\n  {book}: {n} paired rows  "
              f"RESCUE {labels['RESCUE']}  HARM {labels['HARM']}  "
              f"NEUTRAL {labels['NEUTRAL']}")
        print(f"    {'feature':22} {'RESCUE':>8} {'HARM':>8} {'NEUTRAL':>8}")
        for feat, fmt in (("len", "{:8.0f}"), ("tag", "{:8.2f}"), ("quoted", "{:8.2f}")):
            vals = []
            for lab in ("RESCUE", "HARM", "NEUTRAL"):
                v = feats[lab][feat]
                vals.append(statistics.mean(v) if v else 0)
            print(f"    {feat:22} " + " ".join(fmt.format(v) for v in vals))
        rescue_rate = labels["RESCUE"] / n * 100
        print(f"    A router must beat RANDOM routing, which at C% coverage "
              f"captures C% of\n    the {labels['RESCUE']} rescues. Any feature "
              f"above is only useful if it\n    concentrates them.")


if __name__ == "__main__":
    oracle_strata()
    representativeness()
    routing_features()
    print("\n" + "=" * 72)
    print("None of this decides anything. Each section names the experiment that")
    print("would - blind adjudication for the oracle strata, a representative")
    print("fixture for the plateau, and held-out scene-level routing evaluation.")
