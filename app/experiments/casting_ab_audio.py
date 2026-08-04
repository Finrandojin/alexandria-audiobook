"""Generate one real scene under two casts, so the difference can be HEARD.

Every casting result so far ends in the same place: a metric that defers to a
listener. Conflicts, pitch gaps, 4-8 voices, "is 17 Hz enough" - none of it is
settled by a number, and `audible_errors.py` said as much months ago and was
never followed up with audio.

This renders one scene twice, identical text and order, differing only in the
cast:

    current       whatever voice_config says today
    scene_aware   the assignment from scene_aware_casting

The scene is chosen because it CONTAINS a measured conflict - FELT and
REINHARD both on warm_baritone_50s_m_gothic, speaking within twenty lines. If
the conflict is inaudible, that is worth knowing and cheap to learn; if it is
obvious, the metric is validated and the allocator is worth wiring in.

WHY THIS IS THE RIGHT NEXT STEP rather than another metric: the last several
findings all bottom out in perception, and no amount of further measurement
resolves a perceptual question. A single scene a person can play settles more
than another table.

The narrator keeps its own voice in both arms, as it does in production and as
narrators do in practice, so the only thing that changes between the two files
is which voice each CHARACTER gets.
"""
import argparse, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)


def find_scene(chunks, want, size=14):
    """First window containing at least two lines from each named character.

    `range(len(chunks) - size + 1)`, not `- size`: with exactly `size` chunks
    the old form performed zero iterations, and a valid scene starting at the
    last possible index was never examined.
    """
    want = [w.upper() for w in want]
    if size <= 0 or not chunks:
        return None, []
    for i in range(len(chunks) - size + 1):
        window = chunks[i:i + size]
        speakers = [str(c.get("speaker") or "").upper() for c in window]
        if all(speakers.count(w) >= 2 for w in want):
            return i, window
    return None, []


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--script", default=os.path.join(REPO, "chunks.json"))
    ap.add_argument("--voice-config", default=os.path.join(REPO, "voice_config.json"))
    ap.add_argument("--config", default=os.path.join(APP, "config.json"))
    ap.add_argument("--casting", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "scene_aware_casting.json"))
    ap.add_argument("--characters", nargs="+", default=["FELT", "REINHARD"])
    ap.add_argument("--size", type=int, default=14)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "ab_test_runtime", "casting_ab"))
    ap.add_argument("--seed", type=int, default=1234,
                    help="fixed generation seed. The first run was unseeded, so "
                         "the two arms differed by random draw as well as by "
                         "cast - and NARRATOR, identical in both arms, was "
                         "redrawn per line. A listener could not tell which "
                         "difference they were hearing.")
    args = ap.parse_args()

    chunks = [c for c in json.load(open(args.script, encoding="utf-8"))
              if c.get("text")]
    index, scene = find_scene(chunks, args.characters, args.size)
    if not scene:
        sys.exit(f"no scene found containing {args.characters}")

    raw_vc = json.load(open(args.voice_config, encoding="utf-8"))
    vc = (raw_vc.get("characters")
          if isinstance(raw_vc.get("characters"), dict) else raw_vc)
    proposed = json.load(open(args.casting, encoding="utf-8"))["scene_aware"]["assignment"]
    aliases = json.load(open(os.path.join(REPO, "character_aliases.json"),
                             encoding="utf-8"))
    folded = {k.lower(): v for k, v in aliases.items()}
    canon = lambda s: folded.get((s or "").lower(), s or "").upper()

    manifest = json.load(open(os.path.join(REPO, "lora_models", "manifest.json"),
                              encoding="utf-8"))
    items = manifest if isinstance(manifest, list) else list(manifest.values())
    pitch = {i["id"]: (i.get("voice_features") or {}).get("mean_f0")
             for i in items if isinstance(i, dict) and i.get("id")}

    print(f"scene at chunk {index}, {len(scene)} lines\n")
    print(f"  {'speaker':14}{'current':>34}{'scene-aware':>34}")
    for name in sorted({canon(c.get('speaker')) for c in scene}):
        if name == "NARRATOR":
            continue
        cur = (vc.get(name) or {}).get("adapter_id") or "(none)"
        new = proposed.get(name) or "(unchanged)"
        print(f"  {name[:12]:14}{cur[:32]:>34}{new[:32]:>34}")
        if cur in pitch and new in pitch and pitch[cur] and pitch[new]:
            print(f"  {'':14}{pitch[cur]:>31.0f} Hz{pitch[new]:>31.0f} Hz")

    os.makedirs(args.out_dir, exist_ok=True)
    from tts import TTSEngine, voice_category
    from experiments.generation import render
    engine = TTSEngine(json.load(open(args.config, encoding="utf-8")))

    import soundfile as sf
    import numpy as np
    # Render EVERY line for both arms first and only publish a comparison if
    # both arms produced the identical, complete line set. Previously each arm
    # caught its own failures and continued, so line 4 failing in one arm alone
    # yielded a 14-line file against a 13-line file - both looking like a normal
    # A/B while the CONTENT differed, not just the cast. An asymmetric pair is
    # worse than no pair: it invites a listener to attribute a difference to
    # casting that is actually a missing sentence.
    results, rendered, failures = {}, {}, {}
    for arm in ("current", "scene_aware"):
        pieces, rate, ok = [], None, []
        for n, chunk in enumerate(scene):
            speaker = canon(chunk.get("speaker"))
            entry = dict(vc.get(speaker) or vc.get(chunk.get("speaker")) or {})
            entry["seed"] = str(args.seed)
            # The narrator is untouched in both arms; only characters move.
            if arm == "scene_aware" and speaker != "NARRATOR" and proposed.get(speaker):
                entry["adapter_id"] = proposed[speaker]
                entry["adapter_path"] = os.path.join("lora_models", proposed[speaker])
                entry["type"] = "lora"
            wav = os.path.join(args.out_dir, f"{arm}_{n:02d}.wav")
            try:
                render(engine, chunk["text"], chunk.get("instruct", ""),
                       speaker, vc, entry, wav)
            except Exception as exc:                       # noqa: BLE001
                failures.setdefault(arm, []).append({"line": n,
                                                     "error": str(exc)[:120]})
                print(f"  [{arm} {n}] FAILED: {str(exc)[:70]}")
                continue
            if os.path.exists(wav):
                audio, rate = sf.read(wav)
                pieces.append(audio)
                pieces.append(np.zeros(int(rate * 0.35)))
                ok.append(n)
            else:
                failures.setdefault(arm, []).append({"line": n,
                                                     "error": "no file written"})
        rendered[arm] = {"lines": ok, "pieces": pieces, "rate": rate}

    complete = {arm: set(v["lines"]) for arm, v in rendered.items()}
    expected = set(range(len(scene)))
    asymmetric = [arm for arm, got in complete.items() if got != expected]
    if asymmetric:
        print(f"\n  REFUSING TO PUBLISH a comparison pair: "
              f"{', '.join(asymmetric)} did not render every line.")
        for arm, fails in failures.items():
            print(f"    {arm}: {len(fails)} failed -> {fails[:3]}")
        json.dump({"scene_index": index, "lines": len(scene),
                   "characters": args.characters, "published": False,
                   "rendered": {a: v["lines"] for a, v in rendered.items()},
                   "failures": failures},
                  open(os.path.join(args.out_dir, "manifest.json"), "w"), indent=1)
        sys.exit(3)

    for arm, v in rendered.items():
        joined = np.concatenate(v["pieces"])
        out = os.path.join(args.out_dir, f"scene_{arm}.wav")
        sf.write(out, joined, v["rate"])
        results[arm] = {"path": out, "seconds": round(len(joined) / v["rate"], 1),
                        "lines": v["lines"]}
        print(f"\n  wrote {out}  ({results[arm]['seconds']}s)")

    json.dump({"scene_index": index, "lines": len(scene),
               "characters": args.characters, "published": True,
               "line_ids": sorted(expected), "arms": results},
              open(os.path.join(args.out_dir, "manifest.json"), "w"), indent=1)
    print("\n  Play both. The question is whether FELT and REINHARD are\n"
          "  distinguishable in the current arm - the metric says they share a\n"
          "  voice exactly, 0 Hz apart. If that is inaudible, the conflict\n"
          "  count is measuring something that does not matter.")


if __name__ == "__main__":
    main()
