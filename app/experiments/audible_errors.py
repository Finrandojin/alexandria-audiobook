"""How many attribution errors does a LISTENER actually hear?

Every accuracy number in this ledger counts a wrong SPEAKER NAME. A listener
hears a wrong VOICE, and those are not the same thing. If two characters are
cast to the same voice - which happens whenever the cast is larger than the
voice pool - confusing them changes nothing audible. The +5.4 the adapter buys
is an upper bound on what reaches the ear.

`listener_impact` approximated this by weighting leads more heavily. This does
it properly: it assigns a cast the way the app does, then asks whether each
error crosses a VOICE boundary.

HOW THE CAST IS BUILT. `voice_config.json` shows what the app stores per
character - type, voice, adapter, seed, style - and two characters are audibly
identical when those match. There is no cast on disk for the gold books, so
one is constructed the way a user would end up with: the narrator gets its own
voice, the most frequent speakers get distinct voices until the pool runs out,
and the remaining long tail shares. Pool size is swept because it is the thing
a user actually controls.

WHAT THIS CAN SHOW.

  the audible gain tracks the measured gain    the accuracy number is a fair
                                               proxy and this whole ledger
                                               means what it says
  the audible gain is much smaller             a large share of errors are
                                               between characters sharing a
                                               voice, and the headline
                                               overstates what is heard
  the audible gain is larger                   errors concentrate on
                                               distinctly-voiced characters,
                                               and the headline understates it

THE HONEST LIMIT. This still is not audio. It assumes a listener notices any
voice change and never notices a same-voice swap, which is the right first
approximation and not the truth: prosody and style differ even within one
voice. Generating and listening is the only thing that settles it, and that
needs a person's ears rather than a metric.
"""
import argparse, collections, glob, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = REPO + "/app/"
sys.path.insert(0, APP)

from experiments.scoring import alias_groups, same_speaker

LEDGER = REPO + "/ab_test_runtime/experiments"
BOOKS = ("grimgar03", "index18", "mushoku16", "owarimonogatari3")
SPECIAL = {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE"}
POOLS = (4, 6, 8, 12, 20, 999)


def groups_for(book):
    path = APP + f"fixtures/attribution_gold_{book}.json"
    return alias_groups(json.load(open(path))) if os.path.exists(path) else []


def voice_of(name, cast, groups):
    """The voice a predicted name maps to, alias-aware.

    Exact matching would send an alias of a cast member to the overflow voice
    and count an inaudible swap as heard - and it would also disagree with
    every other accuracy in this ledger, which is alias-aware.
    """
    up = (name or "").upper()
    if up in cast:
        return cast[up]
    for member, voice in cast.items():
        if same_speaker(member, up, groups):
            return voice
    return "voice_overflow"


def cast_for(book, pool):
    """character -> voice id, the way a user's cast ends up looking.

    The narrator always holds its own voice; the most-heard characters take the
    remaining slots; everyone past that shares a single overflow voice, which is
    what happens when a 21-character book meets an 8-voice pool.
    """
    path = APP + f"fixtures/attribution_gold_{book}.json"
    if not os.path.exists(path):
        return None
    entries = json.load(open(path))["entries"]
    counts = collections.Counter(e["expected_speaker"].upper() for e in entries
                                 if e["expected_speaker"].upper() not in SPECIAL)
    cast = {"NARRATOR": "voice_narrator"}
    ranked = [s for s, _ in counts.most_common() if s != "NARRATOR"]
    for i, name in enumerate(ranked):
        cast[name] = f"voice_{i}" if i < pool - 1 else "voice_overflow"
    return cast


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=LEDGER + "/audible_errors.json")
    args = ap.parse_args()

    base, lora, expected = {}, {}, {}
    for path in sorted(glob.glob(LEDGER + "/lora_serving_eval__*.json")):
        doc = json.load(open(path))
        for row in doc["rows"]:
            target = base if row["arm"] == "base" else lora
            target[row["id"]] = (row.get("predicted") or "").upper()
            expected[row["id"]] = (row.get("expected") or "").upper()
    shared = sorted(set(base) & set(lora))
    if not shared:
        print("no shippable-stack artifact with both arms yet")
        return

    print(f"{len(shared)} rows, shippable stack (Q4 base + f16 LoRA)\n")
    print("  A wrong name is only heard when it crosses a VOICE boundary.\n")
    print(f"  {'pool':>6}{'base heard':>13}{'lora heard':>13}{'audible gain':>15}"
          f"{'name gain':>12}{'inaudible':>12}")

    all_groups = {b: groups_for(b) for b in BOOKS}
    results = {}
    for pool in POOLS:
        casts = {b: cast_for(b, pool) for b in BOOKS}
        stats = collections.Counter()
        for gid in shared:
            book = gid.split(":", 1)[0]
            cast = casts.get(book)
            if not cast:
                continue
            groups = all_groups[book]
            true_voice = voice_of(expected[gid], cast, groups)
            for arm, preds in (("base", base), ("lora", lora)):
                pred = preds[gid]
                # An unanswered row reaches TTS as an unresolved speaker, which
                # is audible however the cast is arranged.
                if not pred:
                    stats[arm + "_heard"] += 1
                    continue
                if voice_of(pred, cast, groups) != true_voice:
                    stats[arm + "_heard"] += 1
            # An error both arms make that never crosses a voice boundary is
            # invisible to the listener no matter which arm ships.
            if not same_speaker(expected[gid], base[gid], groups) and \
                    voice_of(base[gid], cast, groups) == true_voice:
                stats["inaudible_errors"] += 1
        n = len(shared)
        bh, lh = stats["base_heard"] / n, stats["lora_heard"] / n
        name_gain = (
            sum(1 for g in shared
                if same_speaker(expected[g], lora[g], all_groups[g.split(":", 1)[0]]))
            - sum(1 for g in shared
                  if same_speaker(expected[g], base[g], all_groups[g.split(":", 1)[0]]))) / n
        label = "all" if pool == 999 else str(pool)
        results[label] = {"base_heard": bh, "lora_heard": lh,
                          "audible_gain": bh - lh,
                          "inaudible_errors": stats["inaudible_errors"] / n}
        print(f"  {label:>6}{bh*100:12.1f}%{lh*100:12.1f}%"
              f"{(bh-lh)*100:+14.1f}{name_gain*100:+11.1f}"
              f"{stats['inaudible_errors']/n*100:11.1f}%")

    print("\n  'pool' is how many distinct voices the cast uses; 'all' gives "
          "every\n  character its own voice, which is the ceiling on how much "
          "can be heard.")
    small = results.get("6", {}).get("audible_gain")
    full = results.get("all", {}).get("audible_gain")
    if small is not None and full is not None:
        print(f"\n  With a 6-voice cast the adapter removes "
              f"{small*100:.1f} points of heard error;\n  with a voice per "
              f"character, {full*100:.1f}.")
        if small < full * 0.75:
            print("  A smaller pool hides a real part of the gain: some of what "
                  "the adapter\n  fixes is confusion between characters a "
                  "listener could not tell apart\n  anyway.")
        else:
            print("  The gain survives a small pool, so the adapter is fixing "
                  "confusions\n  between characters that genuinely sound "
                  "different.")
    print("\n  NOT AUDIO. This assumes any voice change is heard and any "
          "same-voice swap\n  is not. Generating a chapter and listening is "
          "what would settle it, and\n  that needs ears rather than a metric.")

    json.dump(results, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
