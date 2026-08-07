"""Does a voice stay the same voice across a whole book?

THE GAP THIS FILLS. Every voice measurement in this project is a SINGLE LINE.
The longest generation artifact holds 150. A real audiobook is five to twenty
thousand lines, and the product being sold is ten hours of one consistent
voice - a property no measurement here has ever addressed.

So the question is simply whether line 4000 still sounds like line 1. Nothing
in the metric set could see a slow drift: ECAPA, MCD and f0 correlation are all
computed per line against that line's own human reference, so a voice that
wanders steadily would score the same at the start and the end while sounding
obviously wrong to a listener who sat through it.

WHAT IS MEASURED. Consecutive lines from real book text, generated in one
continuous run, each compared back to an ANCHOR built from the first lines of
that same run. Position is the independent variable:

    flat        no drift
    sloped      the voice is moving; the slope IS the defect
    stepped     something changed at a point - worth finding what

Also tracked per line: pitch median, vocal tract length, jitter and HNR, so a
drift can be attributed rather than just detected.

WHY AN ANCHOR OF THE FIRST LINES, not the reference clip. The question is
self-consistency over a run, not fidelity to the narrator - that is goal 2.1
and already measured. A voice can be a poor imitation and perfectly stable, or
a good imitation that falls apart by chapter three, and those need separate
numbers.

A SLOW RUN ON PURPOSE. Generation is real TTS at roughly 0.8x real time, so a
400-line run is about an hour. There is no shortcut: the failure mode only
appears at length, which is exactly why it has never been looked at.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

SIBLING_PY = os.environ.get(
    "ALEXANDRIA_SIBLING_PYTHON",
    os.path.join(os.path.dirname(REPO), "alexandria-audiobook.git",
                 "app", "env", "bin", "python"))


def ecapa(pairs):
    if not pairs:
        return None, "no pairs"
    if not os.path.exists(SIBLING_PY):
        return None, "sibling interpreter missing"
    script = os.path.join(APP, "experiments", "_ecapa_batch.py")
    try:
        out = subprocess.run(
            [SIBLING_PY, script],
            input=json.dumps([[os.path.abspath(a), os.path.abspath(b)]
                              for a, b in pairs]),
            capture_output=True, text=True, timeout=7200, cwd=APP)
    except subprocess.SubprocessError as exc:
        return None, str(exc)[:140]
    if out.returncode != 0:
        return None, f"rc={out.returncode} {out.stderr[-200:]}"
    try:
        return json.loads(out.stdout.strip().splitlines()[-1]), None
    except Exception as exc:                                # noqa: BLE001
        return None, f"unparsable: {exc}"


def book_lines(path, count, min_chars=40, max_chars=220):
    """Consecutive sentences from real prose. Not random text: pacing and
    punctuation vary through a book and that variation is part of what a long
    run has to survive."""
    import re
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    text = re.sub(r"\s+", " ", text)
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        sent = sent.strip()
        if min_chars <= len(sent) <= max_chars and sent.count("�") == 0:
            out.append(sent)
        if len(out) >= count:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter", required=True,
                    help="adapter directory under lora_models/")
    ap.add_argument("--source", default=os.path.join(
        REPO, "ab_test_runtime", "results", "collect_all_20260722-155801",
        "inputs", "grimgar03.txt"))
    ap.add_argument("--lines", type=int, default=400)
    ap.add_argument("--anchor-lines", type=int, default=8,
                    help="how many opening lines form the reference anchor")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--work", default=os.path.join(
        REPO, "ab_test_runtime", "voice_drift"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    name = os.path.basename(args.adapter.rstrip("/"))
    out_path = args.out or os.path.join(
        REPO, "ab_test_runtime", "experiments", f"voice_drift__{name}.json")
    wdir = os.path.join(args.work, name)
    os.makedirs(wdir, exist_ok=True)

    lines = book_lines(args.source, args.lines)
    if len(lines) < args.anchor_lines * 3:
        sys.exit(f"only {len(lines)} usable lines in {args.source}")
    print(f"{name}: {len(lines)} lines from {os.path.basename(args.source)}\n")

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vcv", os.path.join(APP, "experiments", "voice_compare_view.py"))
    vcv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vcv)

    from tts import TTSEngine
    from experiments.generation import render, GenerationFailed
    import soundfile as sf
    engine = TTSEngine(json.load(open(os.path.join(APP, "config.json"),
                                      encoding="utf-8")))
    entry = {"type": "lora",
             "adapter_path": os.path.relpath(
                 os.path.join(REPO, "lora_models", name), REPO),
             "seed": str(args.seed)}

    rows, failures = [], 0
    for i, text in enumerate(lines):
        wav = os.path.join(wdir, f"line_{i:05d}.wav")
        if not os.path.exists(wav):
            try:
                render(engine, text, "", "SPEAKER", {"SPEAKER": entry},
                       entry, wav)
            except GenerationFailed as exc:
                failures += 1
                rows.append({"i": i, "error": str(exc)[:90]})
                continue
        info = sf.info(wav)
        rec = {"i": i, "wav": os.path.relpath(wav, REPO),
               "seconds": round(info.frames / info.samplerate, 3),
               "chars": len(text)}
        p = vcv.pitch_stats(wav)
        q = vcv.voice_quality(wav)
        rec.update({"f0_median": p.get("f0_median"),
                    "f0_spread": p.get("f0_spread"),
                    "vtl": vcv.vocal_tract_length(wav),
                    "jitter": q.get("jitter_local"), "hnr": q.get("hnr_db")})
        rows.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(lines)}  {failures} failed", flush=True)

    ok = [r for r in rows if not r.get("error")]
    if len(ok) < args.anchor_lines * 3:
        json.dump({"adapter": name, "error": "too few lines generated",
                   "rows": rows}, open(out_path, "w"), indent=1)
        sys.exit(3)

    # Anchor: the opening lines joined, so the comparison target carries enough
    # audio to be stable - the same lesson the human_vs_human anchor learned.
    import numpy as np
    chunks, rate = [], None
    for r in ok[:args.anchor_lines]:
        y, sr = sf.read(os.path.join(REPO, r["wav"]), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        rate = rate or sr
        chunks.append(y)
    anchor = os.path.join(wdir, "_anchor.wav")
    sf.write(anchor, np.concatenate(chunks), rate)
    print(f"\n  anchor: {len(chunks)} opening lines, "
          f"{sum(len(c) for c in chunks)/rate:.1f}s")

    later = ok[args.anchor_lines:]
    cos, err = ecapa([(anchor, os.path.join(REPO, r["wav"])) for r in later])
    if err:
        print(f"  ECAPA failed: {err}")
    else:
        for r, c in zip(later, cos):
            r["ecapa_vs_anchor"] = c

    doc = {"adapter": name, "source": os.path.relpath(args.source, REPO),
           "lines_requested": args.lines, "generated": len(ok),
           "failed": failures, "anchor_lines": args.anchor_lines,
           "seed": args.seed, "ecapa_error": err, "rows": rows}

    scored = [r for r in later if r.get("ecapa_vs_anchor") is not None]
    if scored:
        # Slope over position is the whole point. Reported as the change across
        # the run rather than a per-line coefficient, because "0.04 lower by
        # the end" is a sentence anyone can act on.
        xs = np.array([r["i"] for r in scored], dtype=float)
        ys = np.array([r["ecapa_vs_anchor"] for r in scored], dtype=float)
        slope, intercept = np.polyfit(xs, ys, 1)
        span = slope * (xs.max() - xs.min())
        third = max(len(scored) // 3, 1)
        doc["drift"] = {
            "slope_per_line": float(slope),
            "change_across_run": float(span),
            "first_third_median": round(statistics.median(
                [r["ecapa_vs_anchor"] for r in scored[:third]]), 4),
            "last_third_median": round(statistics.median(
                [r["ecapa_vs_anchor"] for r in scored[-third:]]), 4),
        }
        d = doc["drift"]
        print(f"\n  similarity to opening: first third "
              f"{d['first_third_median']:.3f} -> last third "
              f"{d['last_third_median']:.3f}")
        print(f"  fitted change across the run: {d['change_across_run']:+.4f}")
        for key in ("f0_median", "vtl", "hnr"):
            v = [r[key] for r in ok if r.get(key) is not None]
            if len(v) > 6:
                a = statistics.median(v[:len(v) // 3])
                b = statistics.median(v[-len(v) // 3:])
                doc["drift"][f"{key}_first_third"] = round(a, 3)
                doc["drift"][f"{key}_last_third"] = round(b, 3)
                print(f"  {key:10} {a:8.2f} -> {b:8.2f}  ({(b-a)/a*100:+.1f}%)")

    try:
        from experiments.provenance import provenance
        doc["provenance"] = provenance(__file__, args)
    except Exception as exc:                                # noqa: BLE001
        doc["provenance"] = {"error": str(exc)[:120]}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
    print(f"\nwrote {out_path}")
    if not scored:
        sys.exit(3)


if __name__ == "__main__":
    main()
