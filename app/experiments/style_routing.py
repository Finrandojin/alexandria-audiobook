"""Should method selection be per BOOK or per SECTION?

Four interventions now split the same way, and the split is large:

                        grimgar03   mushoku16
    w4 context            +10.5       -5.0
    batch size peak         50          25
    tag-priority           +6.5       -5.8

The natural next step is to route per section rather than per book, since a
battle scene and an interior monologue want different treatment even inside one
novel. Both books do contain both regimes - grimgar03 has tag-poor windows and
mushoku16 has tag-rich ones - so there is room for it in principle.

TWO CANDIDATE FEATURES WERE TESTED AND NEITHER SUPPORTS IT.

Local tag density (speech verbs in adjacent narration, +-12 segments):

                    grimgar03   mushoku16
    tag-poor <25%     +15.4       -6.4
    mixed 25-60%       +4.1       -4.5
    tag-rich >60%      +9.0       (n=1)

Uniformly positive in one book and uniformly negative in the other, at every
band. Grimgar's tag-poor windows - the ones that most resemble mushoku - still
gain 15 points.

Local first-person density (pronouns per 1000 narration words):

                       grimgar03   mushoku16
    3rd-person <20        +6.7      -19.4
    mixed 20-45           +4.8      -11.1
    1st-person >45         0.0       +1.1

This one is stranger. The two books behave OPPOSITELY in third-person windows,
which is the reverse of what "third-person prose carries tags" would predict.
The one consistent row is the last: in first-person passages tag-priority does
nothing in either book.

WHAT THAT LEAVES. Per-section routing on either feature would pick the wrong
method about as often as the right one, so the shipping decision is per book.
The one section-level rule the data does support is negative and narrow: turn
tag-priority OFF in first-person passages, where it buys nothing anywhere.

A caution on mushoku16's third-person windows, the -19.4 cell: in a book that
is 62 first-person pronouns per 1000 words, a window with fewer than 20 is
unusual, and the likeliest candidates are the letter and diary passages -
written text with names but no speech tags, which is exactly where a
tag-priority rule should misfire. That is a hypothesis, not a measurement; it
needs the epistolary sections marked before it can be tested.
"""
import collections, json, re, statistics, os, sys
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
E = REPO + "/ab_test_runtime/experiments/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
VERB = (r"\b(said|asked|replied|answered|shouted|whispered|muttered|called|"
        r"cried|yelled|groaned|sighed|laughed|nodded|exclaimed|bellowed|agreed|"
        r"told|added|continued|began|offered|roared|declared|ordered|screamed|"
        r"moaned|snorted|murmured)\b")
FP = re.compile(r"\b(i|me|my|myself|mine)\b", re.I)


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


def window(seg, index, half=12):
    return seg[max(0, index - half):min(len(seg), index + half + 1)]


def tag_density(seg, index):
    w = window(seg, index)
    spoken = [e for e in w if e.get("type") != "NARRATOR"]
    if not spoken:
        return None
    tagged = 0
    for k, e in enumerate(w):
        if e.get("type") == "NARRATOR":
            continue
        near = " ".join((w[j].get("text") or "") for j in (k - 1, k + 1)
                        if 0 <= j < len(w) and w[j].get("type") == "NARRATOR")
        if near and re.search(VERB, near.lower()):
            tagged += 1
    return tagged / len(spoken) * 100


def person_density(seg, index):
    nar = " ".join((e.get("text") or "") for e in window(seg, index)
                   if e.get("type") == "NARRATOR")
    words = len(nar.split())
    return len(FP.findall(nar)) / words * 1000 if words >= 20 else None


def stratify(book, artifact, feature, bands):
    seg = json.load(open(M + INPUT_RUN + f"/{book}/result.json.threepass_checkpoint.json"))["segmented"]
    pos = {norm(e.get("text")): i for i, e in enumerate(seg)}
    doc = json.load(open(E + artifact))
    arms = collections.defaultdict(dict)
    for r in doc["rows"]:
        arms[r["arm"]][r["id"]] = (r["correct"], r["line"])
    a, b = sorted(arms)
    out = collections.defaultdict(lambda: [0, 0, 0])
    for i, (ok_a, line) in arms[a].items():
        if i not in arms[b]:
            continue
        j = pos.get(norm(line))
        if j is None:
            continue
        v = feature(seg, j)
        if v is None:
            continue
        label = next(name for name, hi in bands if v < hi)
        cell = out[label]
        cell[0] += 1
        cell[1] += ok_a
        cell[2] += arms[b][i][0]
    return a, b, out


if __name__ == "__main__":
    TAG_BANDS = [("tag-poor <25%", 25), ("mixed 25-60%", 60), ("tag-rich >60%", 1e9)]
    FP_BANDS = [("3rd-person <20", 20), ("mixed 20-45", 45), ("1st-person >45", 1e9)]
    for title, feature, bands in (("LOCAL TAG DENSITY", tag_density, TAG_BANDS),
                                  ("LOCAL FIRST-PERSON DENSITY", person_density, FP_BANDS)):
        print("=" * 66)
        print(title)
        print("=" * 66)
        for book in ("grimgar03", "mushoku16"):
            art = f"tag_priority__{book}__qwen__qwen3-14b__local-llamacpp.json"
            try:
                a, b, out = stratify(book, art, feature, bands)
            except FileNotFoundError:
                print(f"  {book}: no artifact")
                continue
            print(f"\n  {book}   {a} -> {b}")
            print(f"    {'band':18}{'n':>5}{'from':>9}{'to':>9}{'delta':>8}")
            for name, _ in bands:
                n, x, y = out[name]
                if not n:
                    continue
                print(f"    {name:18}{n:5}{x/n*100:8.1f}%{y/n*100:8.1f}%{(y-x)/n*100:+8.1f}")
        print()
    print("Neither feature flips the sign inside a book, so routing per section on")
    print("either would pick wrong about as often as right. The shipping decision")
    print("is per book. The one section-level rule supported is negative: switch")
    print("tag-priority off in first-person passages, where it buys nothing.")
