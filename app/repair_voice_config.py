"""Find and merge characters cast in two voices under two spellings.

`voice_config.json` is keyed by the raw speaker string the annotator produced.
When a character appears as both 'Anastasia' and 'ANASTASIA', each spelling gets
its OWN entry, and `tts.py`'s `voice_config.get(speaker)` resolves both exactly
- so canonicalising at lookup time would never fire. The duplicate has to be
removed from the data.

Measured on the live book, 2026-08-03: EIGHT characters split this way, 238
lines belonging to them, at least 32 spoken in the wrong voice. Not a subtle
drift - 'Anastasia' is lora/Ryan across 68 lines while 'ANASTASIA' is
custom/Aiden across 2. Man 1 and Man 2 split almost evenly, so those characters
audibly change voice mid-scene.

WHICH ENTRY WINS. The more deliberately configured one: a `lora`, `clone` or
`design` voice was made for that character on purpose, while the losing entries
here are all the same `custom` fallback voice at seed -1, i.e. auto-created.
Line count breaks ties only. Ranking by lines FIRST was tried and is wrong - it
gave PUCK the auto-created custom voice over a character LoRA on a 1-vs-0 count.

Where the two rules disagree the choice is genuinely arguable, so those merges
are flagged as disputed and listed at the end rather than buried in a sort key.

REPORT ONLY BY DEFAULT. `--apply` writes, and always backs up first. This edits
a file the user has hand-tuned through the UI; it must never be a silent fix.

WHAT THIS DOES NOT DO. It does not stop the duplicates being recreated.
`generate_personas.py` writes `voice_config[speaker]` under whatever spelling
the annotator emitted, so a later persona run can reintroduce the split. Fixing
that means canonicalising at write time and is a separate change.
"""
import argparse, collections, json, os, shutil, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "app"))

DEFAULT_CONFIG = os.path.join(REPO, "voice_config.json")
DEFAULT_ALIASES = os.path.join(REPO, "character_aliases.json")
DEFAULT_SCRIPT = os.path.join(REPO, "chunks.json")

# A deliberately configured voice beats one that is just the default shape.
# `clone` belongs here for the same reason as `lora`: someone supplied
# reference audio for that character. Omitting it ranked a clone below the
# auto-created custom entry it was competing with.
TYPE_RANK = {"lora": 3, "builtin_lora": 3, "clone": 3, "design": 2, "custom": 1}


def canonical(name, aliases):
    """Speaker string -> canonical identity, case-insensitively.

    find_nicknames writes {"ALIAS": "CANONICAL"} with inconsistent casing, so
    the map is folded before use; matching it case-sensitively is what let
    'SUBARU' resolve while 'Subaru' did not.
    """
    if not name:
        return ""
    folded = {k.lower(): v for k, v in (aliases or {}).items()}
    return folded.get(name.lower(), name).upper()


def entry_of(config, key):
    value = config.get(key)
    return value if isinstance(value, dict) else {}


def voice_signature(entry):
    """What makes two entries audibly identical, per audible_errors.py."""
    return (entry.get("type"), entry.get("voice"), entry.get("adapter"),
            entry.get("seed"), entry.get("style"))


def find_splits(config, aliases, line_counts):
    """-> [{canonical, keys, winner, reason, signatures}] for split characters.

    Only characters whose spellings disagree on the VOICE are reported. Two
    spellings sharing one voice are harmless duplication, not a defect, and
    flagging them would bury the eight that matter.
    """
    groups = collections.defaultdict(list)
    for key in config:
        if isinstance(config.get(key), dict):
            groups[canonical(key, aliases)].append(key)

    splits = []
    for canon, keys in sorted(groups.items()):
        if len(keys) < 2:
            continue
        sigs = {k: voice_signature(entry_of(config, k)) for k in keys}
        if len(set(sigs.values())) < 2:
            continue
        # Voice TYPE outranks line count. A `lora` or `design` entry was made
        # for that character on purpose; the `custom` entries here are all the
        # same fallback voice at seed -1, i.e. auto-created. Ranking by lines
        # first picked custom/Aiden for PUCK over a character LoRA on a 1-vs-0
        # count, which is exactly backwards.
        ranked = sorted(
            keys,
            key=lambda k: (TYPE_RANK.get(entry_of(config, k).get("type"), 0),
                           line_counts.get(k, 0)),
            reverse=True)
        winner = ranked[0]
        by_lines = max(keys, key=lambda k: line_counts.get(k, 0))
        # Where the two rules disagree the choice is genuinely arguable, so it
        # is surfaced rather than buried in a sort key.
        disputed = (by_lines != winner
                    and line_counts.get(by_lines, 0) > line_counts.get(winner, 0))
        reason = "richer voice type"
        if disputed:
            reason += (f"; NOTE {by_lines!r} has more lines "
                       f"({line_counts.get(by_lines, 0)} vs "
                       f"{line_counts.get(winner, 0)})")
        splits.append({"canonical": canon, "keys": ranked, "winner": winner,
                       "reason": reason, "disputed": disputed,
                       "lines": {k: line_counts.get(k, 0) for k in ranked},
                       "signatures": {k: sigs[k] for k in ranked}})
    return splits


def apply_merges(config, splits):
    """Point every spelling at the winner's settings. Returns a NEW dict.

    Keys are kept rather than deleted: the script still refers to them by their
    original spelling, and removing them would send those lines to a fallback
    voice - trading a wrong voice for no voice.
    """
    merged = dict(config)
    for split in splits:
        winning = dict(entry_of(config, split["winner"]))
        for key in split["keys"]:
            if key != split["winner"]:
                merged[key] = dict(winning)
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--aliases", default=DEFAULT_ALIASES)
    ap.add_argument("--script", default=DEFAULT_SCRIPT,
                    help="chunks.json or annotated_script.json, for line counts")
    ap.add_argument("--apply", action="store_true",
                    help="write the merge (backs up first); default is report only")
    args = ap.parse_args()

    raw = json.load(open(args.config, encoding="utf-8"))
    config = raw.get("characters") if isinstance(raw.get("characters"), dict) else raw
    aliases = (json.load(open(args.aliases, encoding="utf-8"))
               if os.path.exists(args.aliases) else {})

    line_counts = collections.Counter()
    if os.path.exists(args.script):
        doc = json.load(open(args.script, encoding="utf-8"))
        entries = doc if isinstance(doc, list) else (doc.get("entries") or [])
        for e in entries:
            if isinstance(e, dict) and e.get("speaker"):
                line_counts[e["speaker"]] += 1

    splits = find_splits(config, aliases, line_counts)
    if not splits:
        print("No character is cast in two voices. Nothing to repair.")
        return

    affected = sum(sum(s["lines"].values()) for s in splits)
    wrong = sum(sum(n for k, n in s["lines"].items() if k != s["winner"])
                for s in splits)
    print(f"{len(splits)} characters cast in more than one voice\n")
    for s in splits:
        print(f"  {s['canonical']}  (keeping {s['winner']!r} - {s['reason']})")
        for k in s["keys"]:
            mark = "KEEP" if k == s["winner"] else "->  "
            print(f"    {mark} {k!r:24} {s['lines'][k]:4} lines  "
                  f"{s['signatures'][k]}")
    disputed = [s for s in splits if s.get("disputed")]
    print(f"\n  {affected} lines belong to these characters; "
          f"{wrong} are spoken in the losing voice.")
    if disputed:
        print(f"  {len(disputed)} choice(s) arguable - the line-majority "
              f"spelling lost to a richer\n  voice type. Review these before "
              f"applying: "
              f"{', '.join(s['canonical'] for s in disputed)}")

    if not args.apply:
        print("\n  Report only. Re-run with --apply to merge (a backup is "
              "written first).")
        return

    backup = f"{args.config}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(args.config, backup)
    merged = apply_merges(config, splits)
    if isinstance(raw.get("characters"), dict):
        raw["characters"] = merged
        out = raw
    else:
        out = merged
    with open(args.config, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\n  backed up to {backup}")
    print(f"  merged {wrong} lines onto their character's main voice")


if __name__ == "__main__":
    main()
