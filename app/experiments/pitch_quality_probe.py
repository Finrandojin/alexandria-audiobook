"""Pitch range (goal 2.5) and voice quality (goal 2.6) at a usable sample size.

WHY THIS EXISTS SEPARATELY FROM voice_compare_view. That module owns the
measurements - `pitch_stats` and `voice_quality` - but its CLI renders an HTML
comparison view, one figure per line with the audio embedded. Asking it for a
hundred lines produces a 209 MB page and no numbers. GOALS.md cites the
functions as the probe for 2.5 and 2.6, which is accurate, and running the
script is not the same thing as calling them. This is the missing CLI.

WHAT IT REPORTS, AND WHAT IT REFUSES TO. Both goals are about PRESERVATION: how
close the generated take sits to the same human speaking the same line, not
whether the number is good in isolation. A jitter of 0.02 means nothing alone;
0.02 against a human's 0.04 on the identical sentence means the synthetic voice
is half as unsteady as the person it imitates, which is the tell that a voice
is too clean. So every measure is reported as a paired (human, generated)
median plus the ratio between them, and no measure is reported for a clip where
either side failed to voice - dropping the pair, never substituting a zero,
because a zero here would read as perfect steadiness.

Paired clips come from the `*_generate.json` manifests already on disk, so this
generates no audio and needs no GPU beyond what librosa and praat use.
"""
import argparse
import json
import os
import statistics
import sys

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(APP)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voice_compare_view import pitch_stats, voice_quality  # noqa: E402

# f0_spread is the 10-90 percentile band: goal 2.5's "pitch range, not just
# pitch shape". The quality three are goal 2.6.
MEASURES = ("f0_median", "f0_spread", "jitter_local", "shimmer_local", "hnr_db")

LANGUAGES = {"en": "ljspeech_generate.json",
             "ja": "kokoro_generate.json",
             "zh": "aishell3_generate.json"}


def resolve(path):
    """Manifest paths are stored relative to the repo root, not to app/."""
    if not path or os.path.isabs(path):
        return path
    return os.path.join(REPO, path)


def measure(path):
    """All five measures for one clip, or {} if it could not be measured."""
    path = resolve(path)
    if not path or not os.path.exists(path):
        return {}
    out = {}
    out.update(pitch_stats(path) or {})
    out.update(voice_quality(path) or {})
    return out


def run_language(manifest_path, limit):
    with open(manifest_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("results") or payload.get("rows") or payload
    pairs, dropped = [], 0
    for row in rows:
        if len(pairs) >= limit:
            break
        human, gen = row.get("human_wav"), row.get("lora_wav")
        h, g = measure(human), measure(gen)
        # A pair survives only if BOTH sides yielded every measure. Partial
        # pairs would silently change which clips each measure is computed
        # over, making the five numbers incomparable to each other.
        if any(h.get(m) is None or m not in h for m in MEASURES) or \
           any(g.get(m) is None or m not in g for m in MEASURES):
            dropped += 1
            continue
        pairs.append({"id": row.get("id"), "human": h, "generated": g})
        print(f"  {row.get('id')}  ok ({len(pairs)})", flush=True)
    summary = {}
    for m in MEASURES:
        if not pairs:
            break
        hv = [p["human"][m] for p in pairs]
        gv = [p["generated"][m] for p in pairs]
        hm, gm = statistics.median(hv), statistics.median(gv)
        summary[m] = {
            "human_median": round(hm, 4),
            "generated_median": round(gm, 4),
            # HNR is in dB - a log scale - so a ratio of two dB figures is not
            # a meaningful "fraction of human". Report the difference there.
            ("difference_db" if m == "hnr_db" else "ratio"):
                round(gm - hm, 4) if m == "hnr_db"
                else (round(gm / hm, 4) if hm else None),
        }
    # Dropping every pair is not a measurement of zero, it is a broken run -
    # a missing file or an unreadable manifest looks exactly like "no clip
    # voiced cleanly" unless it is called out. Caught a repo-relative path bug.
    if pairs == [] and dropped:
        raise SystemExit(
            f"every one of {dropped} pairs was dropped for "
            f"{os.path.basename(manifest_path)} - check the wav paths resolve")
    return {"n_pairs": len(pairs), "dropped": dropped,
            "summary": summary, "pairs": pairs}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lines", type=int, default=100,
                    help="paired clips per language")
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "pitch_quality_probe.json"))
    args = ap.parse_args()

    result = {
        "scope": "paired human-vs-generated preservation, per language; "
                 "medians over clips where both sides measured cleanly",
        "lines_requested": args.lines,
        "languages": {},
    }
    for lang, name in LANGUAGES.items():
        path = os.path.join(REPO, "ab_test_runtime", "experiments", name)
        if not os.path.exists(path):
            print(f"SKIP {lang}: no manifest at {path}", flush=True)
            continue
        print(f"=== {lang} ===", flush=True)
        result["languages"][lang] = run_language(path, args.lines)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    from utils import atomic_json_write
    atomic_json_write(result, args.out)

    print("\n=== SUMMARY (generated vs human, same line) ===")
    for lang, data in result["languages"].items():
        print(f"\n{lang}  n={data['n_pairs']} (dropped {data['dropped']})")
        for m, s in data["summary"].items():
            rel = s.get("ratio", s.get("difference_db"))
            unit = " dB" if m == "hnr_db" else "x"
            print(f"  {m:15} human {s['human_median']:>9} "
                  f"gen {s['generated_median']:>9}   {rel}{unit}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
