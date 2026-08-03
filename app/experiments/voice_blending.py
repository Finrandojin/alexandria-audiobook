"""Would blended voices remove the overflow voice, and does that matter?

`audible_errors.py` established that an attribution error is only heard when it
crosses a VOICE boundary, and it models a cast the way a user's actually ends
up: the narrator gets a voice, the most frequent speakers take the remaining
slots, and everyone past that shares one overflow voice. That overflow is not a
modelling convenience - it is what a listener hears when a book has more
characters than the pool has voices.

kokoro-tts widens its pool by blending: a weighted interpolation of two voices,
"af_sarah:60,am_adam:40". From B base voices, the ordered weighted pairs give
far more than B usable identities. If that transfers to our stack, the overflow
voice can be replaced by distinct blends.

TWO QUESTIONS, AND ONLY THE FIRST IS ANSWERED HERE.

  1. How much is the overflow worth?     Pure arithmetic over the gold books.
                                         No TTS, no model, runs today. If the
                                         answer is small, question 2 is moot
                                         and nothing further is worth building.

  2. Do blends sound distinct?           Needs generated audio and a listener,
                                         and is NOT settled by this script.

WHAT THIS DOES NOT SHOW. That blending works on Qwen3-TTS at all. kokoro
interpolates style vectors, which is architecturally trivial; our path is LoRA
adapters plus reference-audio cloning, where linear interpolation is plausible
(LoRA merging is established) but unverified. This script quantifies the PRIZE.
It is silent on whether the prize is reachable, and a large number here is a
reason to run the audio test, not a result about blending.

Idea credit: nazdridoy/kokoro-tts (MIT). No code taken; see
THIRD_PARTY_NOTICES.md.
"""
import argparse, collections, glob, json, os, sys

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
APP = REPO + "/app/"
sys.path.insert(0, APP)

LEDGER = REPO + "/ab_test_runtime/experiments"
SPECIAL = {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE"}
POOLS = (4, 6, 8, 12)


def blend_capacity(base_voices, max_components=2, weight_steps=(50, 65, 80)):
    """How many distinct identities can `base_voices` produce by blending?

    An unordered pair {A,B} at weight 50/50 is one identity, but at 65/35 it is
    two - 65% A and 65% B are different voices. So the count is ordered pairs
    for asymmetric weights, unordered for the even split.

    Returns the total including the unblended base voices themselves.
    """
    if base_voices < 1:
        return 0
    total = base_voices
    if max_components < 2 or base_voices < 2:
        return total
    pairs = base_voices * (base_voices - 1) // 2
    for w in weight_steps:
        # A 50/50 blend of A and B is the same voice as 50/50 of B and A.
        total += pairs if w == 50 else pairs * 2
    return total


def parse_blend_spec(spec):
    """'af_sarah:60,am_adam:40' -> [('af_sarah', 0.6), ('am_adam', 0.4)].

    Weights are normalised to sum to 1.0, so ':60,:40' and ':3,:2' differ.
    A component with no weight defaults to an equal share of what is left.
    """
    parts = [p.strip() for p in (spec or "").split(",") if p.strip()]
    if not parts:
        raise ValueError("empty blend spec")
    names, weights = [], []
    for part in parts:
        name, sep, raw = part.rpartition(":")
        if not sep:
            name, raw = part, ""
        name = name.strip()
        if not name:
            raise ValueError(f"blend component missing a voice name: {part!r}")
        names.append(name)
        weights.append(float(raw) if raw.strip() else None)
    if len(set(names)) != len(names):
        raise ValueError(f"blend repeats a voice: {spec!r}")
    known = [w for w in weights if w is not None]
    if any(w < 0 for w in known):
        raise ValueError(f"negative weight in blend: {spec!r}")
    stated = sum(known)
    missing = [i for i, w in enumerate(weights) if w is None]
    if missing:
        share = max(0.0, 100.0 - stated) / len(missing)
        for i in missing:
            weights[i] = share
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"blend weights sum to zero: {spec!r}")
    return [(n, w / total) for n, w in zip(names, weights)]


def overflow_share(book):
    """Fraction of a book's spoken lines that fall past each pool size.

    The narrator holds a slot of its own, matching `audible_errors.cast_for`,
    so a pool of N leaves N-1 slots for characters.
    """
    path = APP + f"fixtures/attribution_gold_{book}.json"
    if not os.path.exists(path):
        return None
    entries = json.load(open(path))["entries"]
    counts = collections.Counter(
        e["expected_speaker"].upper() for e in entries
        if e["expected_speaker"].upper() not in SPECIAL)
    total = sum(counts.values())
    if not total:
        return None
    ranked = [n for _, n in counts.most_common()]
    out = {"characters": len(counts), "lines": total}
    for pool in POOLS:
        out[pool] = sum(ranked[pool - 1:]) / total
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=LEDGER + "/voice_blending.json")
    args = ap.parse_args()

    books = sorted({os.path.basename(p)[len("attribution_gold_"):-len(".json")]
                    for p in glob.glob(APP + "fixtures/attribution_gold_*.json")
                    if "_provisional" not in p})
    rows = {}
    print("Share of spoken lines collapsed onto the shared overflow voice\n")
    print(f"  {'book':28}{'chars':>7}{'lines':>7}" +
          "".join(f"{'pool ' + str(p):>10}" for p in POOLS))
    for book in books:
        r = overflow_share(book)
        if not r:
            continue
        rows[book] = r
        print(f"  {book[:26]:28}{r['characters']:7}{r['lines']:7}" +
              "".join(f"{r[p] * 100:9.1f}%" for p in POOLS))

    if not rows:
        print("no gold fixtures found")
        return

    print(f"\n  {'pooled across books':28}{'':7}{'':7}" +
          "".join(f"{sum(r[p] * r['lines'] for r in rows.values()) / sum(r['lines'] for r in rows.values()) * 100:9.1f}%"
                  for p in POOLS))

    print("\n  What blending would buy: identities reachable from B base voices")
    print(f"  {'base voices':>13}{'blended identities':>21}")
    for b in (4, 6, 8, 12):
        print(f"  {b:13}{blend_capacity(b):21}")

    worst = max(rows.items(), key=lambda kv: kv[1][8])
    pooled8 = (sum(r[8] * r["lines"] for r in rows.values())
               / sum(r["lines"] for r in rows.values()))
    print(f"\n  At a pool of 8, {pooled8 * 100:.1f}% of spoken lines share one "
          f"voice across the\n  corpus, and {worst[1][8] * 100:.1f}% on the "
          f"worst book ({worst[0]}). Blending an 8-voice\n  pool reaches "
          f"{blend_capacity(8)} identities, which covers every character in "
          f"every book\n  measured here.")
    print("\n  THIS IS THE PRIZE, NOT THE RESULT. Nothing here shows a blended "
          "voice is\n  intelligible, distinct, or stable on Qwen3-TTS. That "
          "needs generated audio\n  and a listener. A large number above is a "
          "reason to run that test.")

    json.dump({"per_book": {b: {str(k): v for k, v in r.items()}
                            for b, r in rows.items()},
               "pooled_overflow_at_8": pooled8,
               "blend_capacity": {str(b): blend_capacity(b)
                                  for b in (4, 6, 8, 12)},
               "caveat": "arithmetic over gold casts only; no audio was "
                         "generated and blending is unverified on Qwen3-TTS"},
              open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
