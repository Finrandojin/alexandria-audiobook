"""Is a per-line context width worth building, before anyone builds it?

The width sweep found w4 best on average and flat past it, and the per-row
stratification suggested picking width from how far the true speaker's mention
sits from the line. The reviewer's objection to that is correct and fatal as
stated: `name_at` is computed from the GOLD answer. A production router does not
know who the speaker is - that is the entire task - so a policy keyed on
`name_at` is not a policy, it is a leak.

This bounds the idea instead of shipping it. Four numbers, from the committed
sweep artifact, no GPU:

  fixed-best        the best single width, applied to every row - the incumbent
  observable        a router using only features a production run can see,
                    scored under leave-one-out so it never picks a width using
                    the row it is being tested on
  oracle-name_at    the leaky policy, kept as a DIAGNOSTIC ceiling for
                    "how good could a perfect difficulty detector be"
  any-width         a row counts correct if ANY width got it right - the
                    absolute ceiling of every possible width policy, perfect
                    detector included

Decision rule, fixed before running:

  observable beats fixed-best by a significant margin   build the router
  observable ties fixed-best, oracle-name_at is high    the ceiling is real but
                                                        needs a detector we do
                                                        not have; the next
                                                        experiment is the
                                                        detector, not the router
  any-width barely beats fixed-best                     there is nothing to
                                                        route to; close the
                                                        line of enquiry

The last case is the one that would kill it outright, and it is checkable
first: if the union over all four widths is only a point or two above the best
fixed width, then the widths are failing on the SAME rows and no policy over
them - however clairvoyant - can do much.
"""
import collections
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.stats import clopper_pearson, exact_mcnemar

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
E = REPO + "/ab_test_runtime/experiments/"
M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
ARTIFACT = os.environ.get(
    "EXPERIMENT_ARTIFACT",
    "context_width__grimgar03__qwen__qwen3-14b__local-llamacpp.json")
BOOK = os.environ.get("EXPERIMENT_BOOK", "grimgar03")
SPEECH_VERB = (r"\b(SAID|ASKED|REPLIED|ANSWERED|SHOUTED|WHISPERED|MUTTERED|"
               r"CALLED|CRIED|YELLED|GROANED|SIGHED|LAUGHED|NODDED|EXCLAIMED|"
               r"BELLOWED|AGREED|TOLD|ADDED|CONTINUED|BEGAN|OFFERED)\b")


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


doc = json.load(open(E + ARTIFACT))
cp = json.load(open(M + INPUT_RUN + f"/{BOOK}/result.json.threepass_checkpoint.json"))
seg = cp["segmented"]
pos = {norm(e.get("text")): i for i, e in enumerate(seg)}

# {id: {width: correct}} plus the per-row features recorded during the sweep.
grid = collections.defaultdict(dict)
meta_row, widths = {}, set()
for row in doc["rows"]:
    w = int(row["arm"].lstrip("w"))
    widths.add(w)
    grid[row["id"]][w] = bool(row["correct"])
    prov = row.get("candidate_provenance") or ""
    name_at = re.search(r"name_at=(\w+)", prov)
    tag_at = re.search(r"tag_at=(\w+)", prov)

    def _num(m):
        return None if not m or m.group(1) == "None" else int(m.group(1))

    meta_row[row["id"]] = {"line": row["line"], "name_at": _num(name_at),
                           "tag_at": _num(tag_at)}
WIDTHS = sorted(widths)
IDS = [i for i in grid if len(grid[i]) == len(WIDTHS)]
print(f"{ARTIFACT}\n{len(IDS)} rows scored at every width {WIDTHS}\n")


# ------------------------------------------------------------------ features
def observable_features(row_id):
    """Everything a production router could compute WITHOUT the gold answer.

    `tag_at` is the distance to the nearest explicit speech verb. Unlike
    `name_at` it needs no knowledge of who the speaker is - a regex over
    neighbouring segments finds it - so a policy keyed on it is shippable.
    """
    m = meta_row[row_id]
    i = pos.get(norm(m["line"]))
    prev_spoken = next_spoken = None
    if i is not None:
        prev_spoken = i > 0 and seg[i - 1].get("type") != "NARRATOR"
        next_spoken = (i + 1 < len(seg)
                       and seg[i + 1].get("type") != "NARRATOR")
    tag = m["tag_at"]
    return (
        "short" if len(m["line"]) < 40 else "long",
        "tag-near" if tag is not None and tag <= 1 else
        "tag-far" if tag is not None else "tag-none",
        # Whether the line abuts other speech is the turn-taking situation, and
        # it is observable: it reads segment TYPES, not speaker identities.
        "abuts" if (prev_spoken or next_spoken) else "anchored",
    )


def accuracy(chooser, ids):
    k = sum(1 for i in ids if grid[i][chooser(i)])
    return k, len(ids)


def show(label, k, n, note=""):
    lo, hi = clopper_pearson(k, n)
    print(f"  {label:22} {k:4}/{n} = {k/n*100:5.1f}%   [{lo:4.1f}-{hi:4.1f}]  {note}")


# ------------------------------------------------------------- the incumbent
fixed = {w: sum(1 for i in IDS if grid[i][w]) for w in WIDTHS}
best_w = max(fixed, key=lambda w: fixed[w])
print("fixed widths")
for w in WIDTHS:
    show(f"w{w}", fixed[w], len(IDS), "<- best fixed" if w == best_w else "")

# ---------------------------------------------------------------- the ceiling
union = sum(1 for i in IDS if any(grid[i].values()))
inter = sum(1 for i in IDS if all(grid[i].values()))
print("\nceiling and floor over the width family")
show("any-width (union)", union, len(IDS), "perfect clairvoyant router")
show("all-widths (agree)", inter, len(IDS), "width-insensitive rows")
headroom = (union - fixed[best_w]) / len(IDS) * 100
print(f"\n  routable headroom above the best fixed width: {headroom:+.1f} points")

# A union over four arms is large BY CONSTRUCTION. Four arms that each answer
# 60% of rows correctly at random, with no per-row structure at all, still cover
# most of the fixture between them - so a big union is not evidence that width
# is exploitable. The null below destroys any per-row structure while keeping
# each arm's marginal accuracy exactly, and asks what union that produces.
# Reporting the union without this is the same error as calling a difference
# "equivalent" because it was not significant.
import random

rng = random.Random(20260728)
null = []
for _ in range(2000):
    shuffled = {w: rng.sample(range(len(IDS)), fixed[w]) for w in WIDTHS}
    covered = set()
    for w in WIDTHS:
        covered |= set(shuffled[w])
    null.append(len(covered))
null.sort()
lo_null, hi_null = null[int(0.025 * len(null))], null[int(0.975 * len(null))]
print(f"  null union from unstructured arms at the same marginals: "
      f"{sum(null)/len(null)/len(IDS)*100:.1f}% "
      f"[{lo_null/len(IDS)*100:.1f}-{hi_null/len(IDS)*100:.1f}]")
if union < lo_null:
    print("  -> the observed union sits BELOW the null: the widths fail together")
    print("     far more than unrelated arms would. Rows the family gets wrong are")
    print("     mostly hard for every width, and the union ceiling overstates what")
    print("     any router could reach.")
elif union <= hi_null:
    print("  -> the observed union is INSIDE the null: the arms disagree no more")
    print("     informatively than four unrelated classifiers, so the headroom is")
    print("     an artefact of counting unions, not a routable signal.")
else:
    print("  -> the observed union exceeds the null, which needs explaining before")
    print("     it is believed - check the arms are truly paired on the same rows.")
print("     Either way the union is a CEILING, not a forecast: whether any of it")
print("     is reachable is settled by the observable router below, not here.")
if headroom < 3:
    print("  -> the widths fail on the SAME rows. No policy over this family can")
    print("     recover much, and the detector question is moot.")

# ------------------------------------------------------- observable, honest
strata = collections.defaultdict(list)
for i in IDS:
    strata[observable_features(i)].append(i)


def loo_choice(row_id):
    """Best width for this row's stratum, computed from the OTHER rows in it.

    Leave-one-out is what separates a router estimate from a restatement of the
    answer key. Without it every stratum picks the width that happens to win on
    its own members and the reported accuracy is fitted, not predicted.
    """
    peers = [j for j in strata[observable_features(row_id)] if j != row_id]
    if not peers:
        return best_w
    tally = {w: sum(1 for j in peers if grid[j][w]) for w in WIDTHS}
    top = max(tally.values())
    # Ties go to the narrower window: cheaper, and it keeps the policy from
    # drifting wide on no evidence.
    return min(w for w in WIDTHS if tally[w] == top)


print(f"\nobservable router  ({len(strata)} strata from length x speech-tag x abuts)")
k, n = accuracy(loo_choice, IDS)
show("observable (LOO)", k, n)
b = sum(1 for i in IDS if grid[i][loo_choice(i)] and not grid[i][best_w])
c = sum(1 for i in IDS if grid[i][best_w] and not grid[i][loo_choice(i)])
p, _, _ = exact_mcnemar(b, c)
print(f"  vs fixed w{best_w}: +{b} / -{c} discordant, exact McNemar p={p:.3f}")

print(f"\n  {'stratum':38} {'n':>4}  " + "  ".join(f"w{w:<4}" for w in WIDTHS) + "  picks")
for key in sorted(strata, key=lambda s: -len(strata[s])):
    rows = strata[key]
    tally = {w: sum(1 for j in rows if grid[j][w]) / len(rows) * 100 for w in WIDTHS}
    pick = max(tally, key=lambda w: tally[w])
    print(f"  {' / '.join(key):38} {len(rows):4}  "
          + "  ".join(f"{tally[w]:4.0f}%" for w in WIDTHS) + f"   w{pick}")

# --------------------------------------------------------- the leaky ceiling
def oracle_choice(row_id):
    d = meta_row[row_id]["name_at"]
    if d is None:
        return max(WIDTHS)
    return min([w for w in WIDTHS if w >= d] or [max(WIDTHS)])


k, n = accuracy(oracle_choice, IDS)
print()
show("oracle name_at", k, n, "NOT SHIPPABLE - uses the gold speaker")
print("  Reported only to price a perfect difficulty detector. If this sits far")
print("  below any-width, then even perfect knowledge of where the speaker is")
print("  mentioned is not the routing signal, and the stratification that")
print("  suggested this policy was describing something else.")
