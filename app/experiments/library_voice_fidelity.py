"""How close is each shipped LoRA voice to the narrator it was trained on?

WHAT THIS IS FOR. The library holds 75 trained adapters. Their quality has
never been measured against the humans they imitate - `clone_vs_lora` and
`ljspeech_score` used purpose-built eval sets, not the voices that actually
ship. Every source dataset carries the narrator's own audio and transcripts, so
the comparison is available and has simply never been run.

CONTAMINATION, STATED FIRST BECAUSE IT BOUNDS EVERYTHING BELOW. Each dataset zip
splits 180 train / 20 val with zero overlap. `train_lora.py` loads the ROOT
`metadata.jsonl`, which is all 200, so **the adapters were trained on their own
validation set** - 67 of 75 record num_samples=200. The split exists and the
trainer ignores it.

So these scores are an UPPER BOUND: the model has heard every line it is being
asked to reproduce. What they can still do is RANK, because every adapter is
contaminated identically. A voice that scores badly here scores badly on
material it memorised, which is a strong signal. A voice that scores well has
proven only that it can repeat its own training data.

An honest held-out number needs clips from the source audiobooks that never
entered any dataset. That is a preparer run, not this script.

METRICS. Speaker embedding (ECAPA, via the sibling interpreter that has
speechbrain), plus the preservation set: vocal tract length from formant
dispersion, f0 median and spread, jitter, shimmer, HNR, and duration ratio.
Deliberately NOT a gender or age label - this asks only whether the adapter
carried the narrator's own properties across.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

DEFAULT_ZIPS = os.environ.get(
    "ALEXANDRIA_VOICE_ZIPS",
    os.path.join(os.path.expanduser("~"), "Desktop", "zips2",
                 "_deduped_labeled"))


def adapter_sources(models_dir):
    """-> [(adapter_name, dataset_name)] from each adapter's training_meta."""
    out = []
    for name in sorted(os.listdir(models_dir)):
        meta_path = os.path.join(models_dir, name, "training_meta.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            meta = json.load(open(meta_path, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        ref = str(meta.get("ref_sample_audio") or "")
        dataset = os.path.basename(os.path.dirname(ref)) if ref else ""
        if dataset:
            out.append((name, dataset, meta))
    return out


def find_zip(dataset, zip_dir, _cache={}):
    """Match a dataset name to its source zip.

    The zip names carry punctuation the dataset names have had stripped
    (colons, brackets), so this compares on alphanumerics only rather than
    trying to reverse the slugging.
    """
    if "index" not in _cache:
        idx = {}
        for f in os.listdir(zip_dir):
            if f.endswith(".zip"):
                key = "".join(c for c in f.lower() if c.isalnum())
                idx[key] = os.path.join(zip_dir, f)
        _cache["index"] = idx
    key = "".join(c for c in dataset.lower() if c.isalnum())
    for zkey, path in _cache["index"].items():
        if zkey.startswith(key):
            return path
    return None


def extract_val(zip_path, out_dir, limit):
    """Pull val clips + text. Returns [(wav_path, text)]."""
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        meta_name = "val/metadata.jsonl"
        if meta_name not in names:
            return []
        entries = [json.loads(l) for l in
                   z.read(meta_name).decode("utf-8").splitlines() if l.strip()]
        for e in entries[:limit]:
            rel = e.get("audio_filepath")
            if not rel or rel not in names:
                continue
            dest = os.path.join(out_dir, os.path.basename(rel))
            with open(dest, "wb") as fh:
                fh.write(z.read(rel))
            rows.append((dest, e.get("text") or ""))
    return rows


def ecapa_pairs(pairs, python_bin):
    """Speaker-embedding cosine, run under the interpreter that has
    speechbrain. Never falls back to an acoustic distance: a silent
    substitution of a different metric is the failure mode this whole
    investigation keeps rediscovering."""
    if not pairs or not python_bin or not os.path.exists(python_bin):
        return [None] * len(pairs), "no speechbrain interpreter"
    script = os.path.join(APP, "experiments", "_ecapa_batch.py")
    if not os.path.exists(script):
        return [None] * len(pairs), "_ecapa_batch.py missing"
    # Contract is STDIN and cwd=APP, per its own docstring. The first version
    # here passed a temp-file path as argv and got an unparsable-output error
    # that looked like a speechbrain problem.
    try:
        out = subprocess.run([python_bin, script],
                             input=json.dumps([[a, b] for a, b in pairs]),
                             capture_output=True, text=True, timeout=3600,
                             cwd=APP)
        if out.returncode != 0:
            return [None] * len(pairs), f"rc={out.returncode} {out.stderr[-160:]}"
        return json.loads(out.stdout.strip().splitlines()[-1]), None
    except Exception as exc:                                # noqa: BLE001
        return [None] * len(pairs), str(exc)[:140]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=os.path.join(REPO, "lora_models"))
    ap.add_argument("--zips", default=DEFAULT_ZIPS)
    ap.add_argument("--lines", type=int, default=4,
                    help="val clips per adapter")
    ap.add_argument("--limit-adapters", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--work", default=os.path.join(
        REPO, "ab_test_runtime", "library_fidelity"))
    ap.add_argument("--ecapa-python", default=os.environ.get(
        "ALEXANDRIA_SIBLING_PYTHON",
        os.path.join(os.path.dirname(REPO), "alexandria-audiobook.git",
                     "app", "env", "bin", "python")))
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "library_voice_fidelity.json"))
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(APP, "experiments"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vcv", os.path.join(APP, "experiments", "voice_compare_view.py"))
    vcv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vcv)

    from tts import TTSEngine
    from experiments.generation import render, GenerationFailed
    config = json.load(open(os.path.join(APP, "config.json"), encoding="utf-8"))
    engine = TTSEngine(config)

    pairs = adapter_sources(args.models)
    if args.limit_adapters:
        pairs = pairs[:args.limit_adapters]
    os.makedirs(args.work, exist_ok=True)
    print(f"{len(pairs)} adapters, {args.lines} val clips each\n")

    results, ecapa_jobs, ecapa_index = [], [], []
    for name, dataset, meta in pairs:
        zpath = find_zip(dataset, args.zips)
        if not zpath:
            results.append({"adapter": name, "dataset": dataset,
                            "error": "source zip not found"})
            print(f"  {name[:34]:36} NO ZIP")
            continue
        wdir = os.path.join(args.work, name)
        os.makedirs(wdir, exist_ok=True)
        clips = extract_val(zpath, wdir, args.lines)
        if not clips:
            results.append({"adapter": name, "dataset": dataset,
                            "error": "no val clips"})
            print(f"  {name[:34]:36} NO VAL")
            continue

        entry = {"type": "lora",
                 "adapter_path": os.path.relpath(
                     os.path.join(args.models, name), REPO),
                 "seed": str(args.seed)}
        rows = []
        for i, (human_wav, text) in enumerate(clips):
            gen = os.path.join(wdir, f"gen_{i}.wav")
            try:
                render(engine, text, "", "SPEAKER", {"SPEAKER": entry}, entry, gen)
            except GenerationFailed as exc:
                rows.append({"error": str(exc)[:100]})
                continue
            hq, aq = vcv.voice_quality(human_wav), vcv.voice_quality(gen)
            hp, apn = vcv.pitch_stats(human_wav), vcv.pitch_stats(gen)
            hv, av = vcv.vocal_tract_length(human_wav), vcv.vocal_tract_length(gen)
            import soundfile as sf
            hs = sf.info(human_wav).frames / sf.info(human_wav).samplerate
            gs = sf.info(gen).frames / sf.info(gen).samplerate

            def ratio(a, b):
                return round(a / b, 4) if (a and b) else None
            rows.append({
                "human_wav": os.path.relpath(human_wav, REPO),
                "gen_wav": os.path.relpath(gen, REPO),
                "dur_ratio": round(gs / hs, 3) if hs else None,
                "vtl_ratio": ratio(av, hv),
                "f0_median_ratio": ratio(apn.get("f0_median"), hp.get("f0_median")),
                "f0_spread_ratio": ratio(apn.get("f0_spread"), hp.get("f0_spread")),
                "jitter_ratio": ratio(aq.get("jitter_local"), hq.get("jitter_local")),
                "shimmer_ratio": ratio(aq.get("shimmer_local"), hq.get("shimmer_local")),
                "hnr_ratio": ratio(aq.get("hnr_db"), hq.get("hnr_db")),
            })
            ecapa_jobs.append([human_wav, gen])
            ecapa_index.append((len(results), len(rows) - 1))

        rec = {"adapter": name, "dataset": dataset,
               "num_samples_trained": meta.get("num_samples"),
               "final_loss": meta.get("final_loss"), "rows": rows}
        results.append(rec)
        ok = [r for r in rows if not r.get("error")]
        if ok:
            def med(k):
                v = [r[k] for r in ok if r.get(k) is not None]
                return round(statistics.median(v), 3) if v else None
            rec.update({"dur_ratio": med("dur_ratio"),
                        "vtl_ratio": med("vtl_ratio"),
                        "f0_median_ratio": med("f0_median_ratio"),
                        "f0_spread_ratio": med("f0_spread_ratio")})
            print(f"  {name[:34]:36} dur {rec['dur_ratio']}  vtl {rec['vtl_ratio']}"
                  f"  f0 {rec['f0_median_ratio']}  spread {rec['f0_spread_ratio']}")

    cos, err = ecapa_pairs(ecapa_jobs, args.ecapa_python)
    if err:
        print(f"\n  ECAPA unavailable: {err}")
    else:
        for (ri, rj), value in zip(ecapa_index, cos):
            results[ri]["rows"][rj]["ecapa"] = value
        for rec in results:
            vals = [r.get("ecapa") for r in rec.get("rows", [])
                    if r.get("ecapa") is not None]
            if vals:
                rec["ecapa"] = round(statistics.median(vals), 4)

    scored = [r for r in results if r.get("ecapa") is not None]
    scored.sort(key=lambda r: r["ecapa"])
    if scored:
        print(f"\n  {'WORST 10 by speaker similarity':46}{'ecapa':>8}{'vtl':>7}{'f0':>7}")
        for r in scored[:10]:
            print(f"  {r['adapter'][:44]:46}{r['ecapa']:8.3f}"
                  f"{r.get('vtl_ratio') or float('nan'):7.2f}"
                  f"{r.get('f0_median_ratio') or float('nan'):7.2f}")

    doc = {"contamination": "adapters trained on their own val split; scores "
                            "are an upper bound, useful for ranking only",
           "lines_per_adapter": args.lines, "seed": args.seed,
           "ecapa_error": err, "results": results}
    try:
        from experiments.provenance import provenance
        doc["provenance"] = provenance(__file__, args)
    except Exception as exc:                                # noqa: BLE001
        doc["provenance"] = {"error": str(exc)[:120]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
    print(f"\nwrote {args.out}")
    if not scored:
        sys.exit(3)


if __name__ == "__main__":
    main()
