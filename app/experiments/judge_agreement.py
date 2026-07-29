"""Compare two independent judge passes and emit the rows they disagree on.

Three of the four books have only ever been read by one judge. That matters
more than it sounds: the single largest labelling error found so far - 19
consecutive wrong rows in grimgar03's keep-assault scene, nine of them assigned
to a character who speaks none of them - was caught ONLY because that book
happened to have an earlier independent gold to contradict. index18 and
owarimonogatari3 have no such check, so an error of the same size and shape
could be sitting in either one undetected.

An automated substitute was tried and does not work. Looking for rows whose
label contradicts a name in the adjacent speech tag flags 14-47% of tagged
rows, but most flags are legitimate, because the tag names the ADDRESSEE at
least as often as the speaker:

    "Shove off, Moguzo! You just stay quiet!"    labelled RANTA, tag says MOGUZO
    "Roxy. Lately, have there been any changes"  labelled RUDEUS, tag says ROXY

Both correct. Speaker-mentions and addressee-mentions are not separable by
regex, so those percentages are not error rates and should not be quoted as
such. A second judge is the only instrument that works.

WHAT THIS REPORTS

  agreement       on rows where both judges named someone, after alias
                  expansion - the headline number, and an estimate of the
                  fixture's own accuracy ceiling
  category splits where one judge said NOT_DIALOGUE / UNNAMED / UNKNOWN and
                  the other named a speaker. These are not the same kind of
                  disagreement as two different names and are counted apart:
                  one is a judgement about the task, the other about the text.
  the queue       every disagreeing row, with both answers and both stated
                  reasons, ready for adjudication

Agreement is computed with `scoring.same_speaker` and the fixture's alias
groups, because the raw number is badly misleading otherwise: on index18's
top-up rows the raw figure was 4/14 and the alias-expanded figure 13/14, and
ten of those "disagreements" were KAMIJOU versus TOUMA KAMIJOU.
"""
import collections
import glob, json, os, re, sys
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")
from experiments.scoring import alias_groups, same_speaker
from experiments.stats import clopper_pearson

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
J = REPO + "/ab_test_runtime/judgements/"
BOOKS = {"grimgar03": "attribution_gold_grimgar03_provisional.json",
         "index18": "attribution_gold_index18_provisional.json",
         "mushoku16": "attribution_gold_mushoku16_provisional.json",
         "owarimonogatari3": "attribution_gold_owarimonogatari3_provisional.json"}
SPECIAL = {"NOT_DIALOGUE", "UNNAMED", "UNKNOWN"}
# Which folder holds which pass. `_wide` is the gemini 12-segment pass;
# `_openai` is the independent second read of the same prompts.
A_SUFFIX = os.environ.get("EXPERIMENT_A", "_wide")
B_SUFFIX = os.environ.get("EXPERIMENT_B", "_openai")
OUT = os.environ.get("EXPERIMENT_OUT",
                     REPO + "/ab_test_runtime/judgements/disagreements.json")


def parse(path):
    raw = json.loads(open(path, encoding="utf-8").read().strip())
    if isinstance(raw, dict):
        raw = next((v for v in raw.values() if isinstance(v, list)), [])
    return [{**x, "speaker": x.get("speaker") or x.get("ANSWER")}
            for x in raw if isinstance(x, dict) and "id" in x]


def load(book, suffix):
    out = {}
    for path in sorted(glob.glob(J + book + suffix + "/reply_*.json")):
        for item in parse(path):
            speaker = (item.get("speaker") or "").strip().upper()
            if speaker:
                out[item["id"]] = {"speaker": speaker,
                                   "why": (item.get("reasoning") or "").strip()}
    return out


def compare(book, goldfile):
    a, b = load(book, A_SUFFIX), load(book, B_SUFFIX)
    if not b:
        return None
    groups = alias_groups(json.load(open(REPO + "/app/fixtures/" + goldfile)))
    shared = sorted(set(a) & set(b))
    named = [i for i in shared
             if a[i]["speaker"] not in SPECIAL and b[i]["speaker"] not in SPECIAL]
    agree = [i for i in named if same_speaker(a[i]["speaker"], b[i]["speaker"], groups)]
    category = [i for i in shared
                if (a[i]["speaker"] in SPECIAL) != (b[i]["speaker"] in SPECIAL)]
    both_special = [i for i in shared
                    if a[i]["speaker"] in SPECIAL and b[i]["speaker"] in SPECIAL
                    and a[i]["speaker"] != b[i]["speaker"]]
    rows = []
    for i in shared:
        if i in agree:
            continue
        if a[i]["speaker"] == b[i]["speaker"]:
            continue
        rows.append({"id": i, "book": book,
                     "a": a[i]["speaker"], "a_why": a[i]["why"],
                     "b": b[i]["speaker"], "b_why": b[i]["why"],
                     "kind": ("category" if i in category else
                              "both-special" if i in both_special else "name")})
    lo, hi = clopper_pearson(len(agree), max(len(named), 1))
    return {"book": book, "shared": len(shared), "named": len(named),
            "agree": len(agree), "lo": lo, "hi": hi,
            "category": len(category), "both_special": len(both_special),
            "rows": rows}


if __name__ == "__main__":
    print(f"comparing {A_SUFFIX.lstrip('_')} against {B_SUFFIX.lstrip('_')}\n")
    print(f"  {'book':18}{'shared':>8}{'both named':>12}{'agree':>16}"
          f"{'category':>10}{'queue':>7}")
    everything = []
    for book, goldfile in BOOKS.items():
        r = compare(book, goldfile)
        if r is None:
            print(f"  {book:18}      -- no {B_SUFFIX.lstrip('_')} pass yet --")
            continue
        everything += r["rows"]
        print(f"  {book:18}{r['shared']:8}{r['named']:12}"
              f"{r['agree']:7} {r['agree']/max(r['named'],1)*100:5.1f}%"
              f" [{r['lo']:.0f}-{r['hi']:.0f}]{r['category']:8}{len(r['rows']):7}")
    if everything:
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump({"pass_a": A_SUFFIX.lstrip("_"),
                       "pass_b": B_SUFFIX.lstrip("_"),
                       "rows": everything}, fh, indent=1, ensure_ascii=False)
        kinds = collections.Counter(r["kind"] for r in everything)
        print(f"\n  {len(everything)} rows to adjudicate  {dict(kinds)}")
        print(f"  wrote {OUT}")
        print("\n  'category' rows are one judge calling a line NOT_DIALOGUE or "
              "UNNAMED while\n  the other names a speaker. Those are arguments "
              "about the task, not about\n  who spoke, and they are usually "
              "settled by the convention rather than by\n  rereading the "
              "passage.")
    else:
        print("\n  nothing to compare yet")
