"""Cast for contrast between characters who share scenes, not globally.

Professional narrators pick voices so that characters WHO APPEAR TOGETHER are
distinguishable, and they work with 4-8 distinct character voices plus a
narrator because more than that stops a listener tracking who is speaking.
Sources in THIRD_PARTY_NOTICES.md.

The current allocator (`routers/voices.get_voice_allocation`) spreads voices by
a GLOBAL reuse count: every character that reuses an adapter pays the same
penalty whether or not they ever share a page with the other user of it. That
optimises the wrong thing in both directions - it burns distinct voices on
characters who never meet, and it does nothing to guarantee separation between
two characters arguing in the same scene.

WHAT THIS MEASURES

    conflicts       pairs of characters who co-occur AND share a voice. These
                    are the audible failures; a listener hears one voice
                    answering itself.
    pitch gap       the smallest mean_f0 separation between any two
                    co-occurring characters. Narrators cite pitch as the first
                    contrast cue, and `lora_models/manifest.json` already
                    stores mean_f0 per adapter, so this needs no new data.
    voices used     fewer is better once conflicts are zero, because the pool
                    is crowded - 61% of the 75 adapters sit between 100-150 Hz.

SCENES ARE APPROXIMATED by a sliding window of consecutive lines rather than
detected. A real scene break is not marked in the script, and a window is both
honest about that and conservative: it can only ever claim two characters are
adjacent when they speak near each other, which is exactly the condition that
matters for confusion.

WHAT THIS IS NOT. It does not judge whether a voice SUITS a character - that is
`_infer_character_traits`' job and needs the persona description. This decides
which characters may share, and how far apart the ones that cannot should be.
A cast that is perfectly separated and completely miscast would score well
here, so the two have to be read together.
"""
import argparse, collections, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

DEFAULT_WINDOW = 20
NARRATOR = "NARRATOR"


def canonicalise(speakers, aliases):
    """Fold alias spellings before anything else.

    'ANASTASIA' and 'Anastasia' are one character. Left unfolded they co-occur
    with themselves and share a voice, and get counted as a conflict - which
    inflated the first run's count and would have credited the new allocator
    with fixing a problem that does not exist.
    """
    folded = {k.lower(): v for k, v in (aliases or {}).items()}
    return [folded.get((s or "").lower(), s).upper() if s else s
            for s in speakers]


def build_cooccurrence(speakers, window=DEFAULT_WINDOW):
    """character -> set of characters who speak within `window` lines of them.

    The narrator is excluded: it speaks constantly, would be adjacent to
    everyone, and already holds its own voice by convention.
    """
    adj = collections.defaultdict(set)
    seq = [s for s in speakers if s]
    for i in range(len(seq)):
        near = {s for s in seq[i:i + window]
                if s and s.upper() != NARRATOR}
        for a in near:
            adj[a] |= (near - {a})
    return adj


def conflicts(assignment, adj):
    """Co-occurring pairs that share a voice - the audible failures."""
    out = []
    for a, neighbours in adj.items():
        for b in neighbours:
            if a < b and assignment.get(a) and assignment.get(a) == assignment.get(b):
                out.append((a, b))
    return sorted(set(out))


def colour_graph(adj, line_counts):
    """Greedy colouring, busiest characters first.

    Order matters: colouring the most-heard characters first gives them the
    lowest-numbered slots, which is where the best-separated voices get
    assigned below. A character with four hundred lines deserves a clearer
    voice than one with two.
    """
    colour = {}
    for name in sorted(adj, key=lambda n: -line_counts.get(n, 0)):
        used = {colour[n] for n in adj[name] if n in colour}
        slot = 0
        while slot in used:
            slot += 1
        colour[name] = slot
    return colour


def spread_adapters(adapters, k):
    """Pick k adapters maximally separated in pitch, greedily.

    max-min rather than even spacing: the pool is not uniform, so asking for
    evenly spaced pitches would demand voices that do not exist. This takes the
    widest separation the pool can actually supply.
    """
    pool = [a for a in adapters if a.get("mean_f0")]
    if not pool or k <= 0:
        return []
    chosen = [max(pool, key=lambda a: a["mean_f0"])]
    while len(chosen) < k and len(chosen) < len(pool):
        nxt = max(pool, key=lambda a: min(abs(a["mean_f0"] - c["mean_f0"])
                                          for c in chosen))
        if nxt in chosen:
            break
        chosen.append(nxt)
    return sorted(chosen, key=lambda a: a["mean_f0"])


def min_gap(assignment, adj, pitch):
    """Smallest pitch separation between any two co-occurring characters."""
    gaps = []
    for a, neighbours in adj.items():
        for b in neighbours:
            if a < b and assignment.get(a) and assignment.get(b):
                pa, pb = pitch.get(assignment[a]), pitch.get(assignment[b])
                if pa and pb:
                    gaps.append(abs(pa - pb))
    return min(gaps) if gaps else None


def load_adapters(path):
    raw = json.load(open(path, encoding="utf-8"))
    items = raw if isinstance(raw, list) else list(raw.values())
    out = []
    for i in items:
        if not isinstance(i, dict):
            continue
        f0 = (i.get("voice_features") or {}).get("mean_f0")
        if i.get("id") and f0:
            out.append({"adapter_id": i["id"], "mean_f0": float(f0)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--script", default=os.path.join(REPO, "chunks.json"))
    ap.add_argument("--voice-config", default=os.path.join(REPO, "voice_config.json"))
    ap.add_argument("--manifest", default=os.path.join(REPO, "lora_models", "manifest.json"))
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    ap.add_argument("--aliases", default=os.path.join(
        REPO, "character_aliases.json"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "scene_aware_casting.json"))
    args = ap.parse_args()

    doc = json.load(open(args.script, encoding="utf-8"))
    entries = doc if isinstance(doc, list) else (doc.get("entries") or [])
    # args.aliases, not a rebuilt default path. The CLI accepted --aliases and
    # then ignored it, so a caller supplying a book-specific map silently got
    # the repository default and a conflict count computed from the wrong
    # canonicalisation.
    aliases = {}
    if os.path.exists(args.aliases):
        aliases = json.load(open(args.aliases, encoding="utf-8"))
    speakers = canonicalise([e.get("speaker") for e in entries
                             if isinstance(e, dict) and e.get("text")], aliases)
    line_counts = collections.Counter(s for s in speakers if s)
    adj = build_cooccurrence(speakers, args.window)
    characters = [c for c in line_counts if c and c.upper() != NARRATOR]

    adapters = load_adapters(args.manifest)
    pitch = {a["adapter_id"]: a["mean_f0"] for a in adapters}
    print(f"{len(characters)} characters, {len(adapters)} adapters with pitch, "
          f"window {args.window} lines\n")

    # --- current cast -------------------------------------------------------
    raw_vc = json.load(open(args.voice_config, encoding="utf-8"))
    vc = (raw_vc.get("characters")
          if isinstance(raw_vc.get("characters"), dict) else raw_vc)
    # voice_config is keyed by the raw spelling, so look the canonical name up
    # through every spelling that maps to it.
    by_canon = collections.defaultdict(list)
    folded = {k.lower(): v for k, v in aliases.items()}
    for key, entry in vc.items():
        if isinstance(entry, dict) and entry.get("adapter_id"):
            by_canon[folded.get(key.lower(), key).upper()].append(
                entry["adapter_id"])
    current = {c: by_canon[c][0] for c in characters if by_canon.get(c)}

    cur_conflicts = conflicts(current, adj)
    cur_gap = min_gap(current, adj, pitch)
    print(f"  CURRENT cast: {len(set(current.values()))} distinct adapters "
          f"over {len(current)} characters")
    print(f"    co-occurring pairs sharing a voice: {len(cur_conflicts)}")
    print(f"    smallest pitch gap between co-occurring characters: "
          f"{f'{cur_gap:.0f} Hz' if cur_gap is not None else 'n/a'}")
    for a, b in cur_conflicts[:6]:
        print(f"      {a} / {b}  -> {current[a]}")

    # --- scene-aware --------------------------------------------------------
    colour = colour_graph({c: adj[c] for c in characters}, line_counts)
    k = max(colour.values()) + 1 if colour else 0
    chosen = spread_adapters(adapters, k)
    slot_to_adapter = {slot: chosen[slot]["adapter_id"]
                       for slot in range(min(k, len(chosen)))}
    proposed = {c: slot_to_adapter.get(colour[c]) for c in characters
                if colour.get(c) in slot_to_adapter}

    new_conflicts = conflicts(proposed, adj)
    new_gap = min_gap(proposed, adj, pitch)
    print(f"\n  SCENE-AWARE cast: {k} distinct adapters "
          f"over {len(proposed)} characters")
    print(f"    co-occurring pairs sharing a voice: {len(new_conflicts)}")
    print(f"    smallest pitch gap between co-occurring characters: "
          f"{f'{new_gap:.0f} Hz' if new_gap is not None else 'n/a'}")

    print(f"\n  {'slot':>5}{'pitch':>9}  adapter")
    for slot in sorted(slot_to_adapter):
        n = sum(1 for c in proposed if colour[c] == slot)
        print(f"  {slot:5}{chosen[slot]['mean_f0']:8.0f} Hz  "
              f"{slot_to_adapter[slot]}  ({n} characters)")

    print("\n  Conflicts are the audible failure - one voice answering itself\n"
          "  in a scene. Pitch gap is the first contrast cue narrators name.\n"
          "  This does NOT judge whether a voice SUITS a character; a perfectly\n"
          "  separated but miscast cast would still score well here.")

    from experiments.provenance import input_sha256, provenance
    from utils import atomic_json_write
    result = {
        "status": "complete",
        "provenance": provenance(
            __file__, args,
            input_sha256=input_sha256((args.script, args.voice_config,
                                       args.manifest, args.aliases))),
        "window": args.window,
        "characters": len(characters),
        "current": {"adapters": len(set(current.values())),
                    "conflicts": len(cur_conflicts),
                    "conflict_pairs": cur_conflicts,
                    "min_pitch_gap": cur_gap},
        "scene_aware": {"adapters": k,
                        "conflicts": len(new_conflicts),
                        "min_pitch_gap": new_gap,
                        "assignment": proposed},
    }
    atomic_json_write(result, args.out)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
