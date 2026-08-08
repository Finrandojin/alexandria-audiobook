"""Which ASR backend should the preparer transcribe audiobooks with?

THE DECISION THIS INFORMS. `alexandria_preparer_rocm_compatible.py` slices an
audiobook into training clips for Voice Lab. Two things matter and they are not
the same:

    WER               did it hear the right words
    ALIGNMENT ERROR   did it put the boundary in the right place

Only the second decides whether a backend can replace the align step. A clip cut
30ms early clips a consonant off every sample in the dataset; a clip cut 300ms
late carries a word from the next line into the training data. WER says nothing
about either, which is why a benchmark that reports WER alone would pick the
wrong winner.

WHY THE ALIGNMENT TEST IS SYNTHETIC. LJSpeech clips are single utterances with
no internal boundaries, so they cannot score alignment on their own. This
concatenates known clips into one file with **exact** boundary times, then asks
each backend to recover them. Ground truth is arithmetic, not annotation - the
one case in this repo where the answer key cannot itself be wrong.

WHAT IS BEING COMPARED. Backends the preparer can actually select, plus
SenseVoice as a candidate (github.com/QwenAudio/SenseVoice) whose claim is 5x
faster than Whisper-Small at better Chinese accuracy. Claims are the vendor's;
this measures them.

READ THE SPEED NUMBER CAREFULLY. Backends here run on different stacks - a GGML
binary, a torch pipeline - so a speed difference is a difference between
*implementations on this machine*, not between model architectures. Reported
because the preparer's throughput is what a user waits on, not as a statement
about the models.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)


def is_cjk(text):
    return any("぀" <= c <= "ヿ" or "一" <= c <= "鿿"
               for c in str(text or ""))


def normalise(text, char_level=False):
    """Casefolded, punctuation-free, single-spaced. Digits kept as written -
    expanding them is a normalisation choice that would flatter whichever
    backend happens to share this project's convention.

    CHARACTER LEVEL FOR CJK. Japanese and Chinese are not space-delimited, so
    word-level WER on them measures the tokeniser, not the transcription: a
    backend that segments differently scores as if every word were wrong. CJK
    is therefore scored as CER, which is what the ASR literature reports for
    these languages anyway.
    """
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    text = re.sub(r"[^\w\s']", " ", text)
    if char_level:
        return " ".join("".join(text.split()))
    return " ".join(text.split())


def word_error_rate(reference, hypothesis, char_level=None):
    """Levenshtein over tokens / reference length. Implemented rather than
    imported: it is twenty lines, and a dependency for twenty lines is a
    dependency to install on every machine that ever reruns this.

    char_level=None auto-detects CJK from the REFERENCE, never the hypothesis -
    a backend that fails and returns empty text must not change how it is
    scored."""
    if char_level is None:
        char_level = is_cjk(reference)
    ref = normalise(reference, char_level).split()
    hyp = normalise(hypothesis, char_level).split()
    if not ref:
        return None
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


# ── backends ────────────────────────────────────────────────────────────
# Each returns (text, segments) where segments is [(start, end, text)] or []
# when the backend gives no timestamps. An empty list is a legitimate answer
# and must not be silently treated as "aligned at zero".

def run_whisper_cpp(wav, model, binary, language="en", max_len=0):
    # NO `-nt`. The first version of this passed it, which is
    # --no-timestamps, and then reported that whisper.cpp merged ten clips into
    # three segments with a six-second median boundary error. That was the
    # flag, not the backend. Giving each backend its best shot at timestamps is
    # the whole point, and this repo has a documented case of a diagnostic
    # harness reversing sign once the arm was configured fairly.
    cmd = [binary, "-m", model, "-f", wav, "-oj", "-of", wav + ".wcpp",
           "-l", language, "-np"]
    if max_len:
        cmd += ["-ml", str(max_len), "-sow"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        raise RuntimeError(f"whisper.cpp rc={out.returncode}: {out.stderr[-200:]}")
    with open(wav + ".wcpp.json", encoding="utf-8") as fh:
        doc = json.load(fh)
    segs = [(s["offsets"]["from"] / 1000.0, s["offsets"]["to"] / 1000.0,
             s["text"]) for s in doc.get("transcription", [])]
    return " ".join(s[2] for s in segs), segs


def run_whisper_cpp_hybrid(wav, seg_model, txt_model, binary, language="en"):
    """Boundaries from one checkpoint, words from another. Goal 5.4.

    WHY THIS EXISTS. On Chinese, measured 2026-08-06, neither checkpoint clears
    the target alone and they fail on opposite axes:

        base      CER 44.3%  align  84 ms  100% within 300 ms
        large-v3  CER 14.1%  align 826 ms   20% within 300 ms

    base is the better SEGMENTER by a factor of ten and the worse TRANSCRIBER
    by a factor of three. Since the preparer needs both - clip boundaries to
    cut on, and text to train against - and since the two jobs are separable,
    nothing requires one model to do both.

    So base decides where the clips are, and large-v3 is asked what was said
    inside each one. Each model is used only for the thing it measurably wins.

    WHAT WOULD FALSIFY THE IDEA. large-v3's transcription advantage might come
    partly from choosing its own segment boundaries - a model given a window it
    did not pick could transcribe worse than it does alone. If the hybrid's CER
    lands near base's 44.3% rather than large-v3's 14.1%, that is the answer,
    and the hybrid is not merely unhelpful but wrong about why large-v3 wins.
    """
    import soundfile as sf

    _text, segments = run_whisper_cpp(wav, seg_model, binary, language=language)
    if not segments:
        return "", []
    audio, rate = sf.read(wav)
    pieces = []
    with tempfile.TemporaryDirectory() as work:
        for index, (start, end, _seg_text) in enumerate(segments):
            first = max(0, int(start * rate))
            last = min(len(audio), int(end * rate))
            if last <= first:
                pieces.append("")
                continue
            chunk = os.path.join(work, f"seg{index:04d}.wav")
            sf.write(chunk, audio[first:last], rate)
            spoken, _ = run_whisper_cpp(chunk, txt_model, binary,
                                        language=language)
            pieces.append(spoken.strip())
    # Boundaries stay base's; only the words are replaced. Returning
    # large-v3's own timestamps here would silently reintroduce the 826 ms
    # error this function exists to avoid.
    rebuilt = [(s[0], s[1], t) for s, t in zip(segments, pieces)]
    return " ".join(p for p in pieces if p), rebuilt


def run_transformers_whisper(wav, model_id=None, language="en", _cache={}):
    # base.en cannot decode Japanese or Chinese at all, so the multilingual
    # checkpoint is used whenever the language is not English. Scoring a
    # non-English clip with an English-only model would measure the choice of
    # checkpoint, not the backend.
    import torch
    model_id = model_id or ("openai/whisper-base.en" if language == "en"
                            else "openai/whisper-base")
    key = "pipe:" + model_id
    if key not in _cache:
        from transformers import pipeline
        dev = 0 if torch.cuda.is_available() else -1
        _cache[key] = pipeline("automatic-speech-recognition", model=model_id,
                               device=dev, return_timestamps=True)
    kw = {} if language == "en" else {"generate_kwargs": {"language": language}}
    r = _cache[key](wav, **kw)
    segs = [(c["timestamp"][0], c["timestamp"][1], c["text"])
            for c in (r.get("chunks") or [])
            if c.get("timestamp") and c["timestamp"][0] is not None
            and c["timestamp"][1] is not None]
    return r.get("text", ""), segs


def run_sensevoice(wav, model_dir="iic/SenseVoiceSmall", language="auto", _cache={}):
    if "m" not in _cache:
        from funasr import AutoModel
        _cache["m"] = AutoModel(model=model_dir, vad_model="fsmn-vad",
                                device="cuda", disable_update=True)
    r = _cache["m"].generate(input=wav, language=language, use_itn=True,
                             sentence_timestamp=True)
    if not r:
        return "", []
    doc = r[0]
    # SenseVoice tags emotion/event inline as <|HAPPY|> <|Speech|> etc. Those
    # are a feature, not transcript, and must not be scored as words.
    text = re.sub(r"<\|[^|]*\|>", " ", str(doc.get("text") or ""))
    segs = []
    for s in (doc.get("sentence_info") or []):
        if s.get("start") is not None and s.get("end") is not None:
            segs.append((s["start"] / 1000.0, s["end"] / 1000.0,
                         re.sub(r"<\|[^|]*\|>", " ", str(s.get("text") or ""))))
    return text, segs


def build_alignment_probe(rows, out_wav, gap=0.5):
    """Concatenate clips with silence between, returning exact boundary times.

    Ground truth here is arithmetic: we place the clips, so we know to the
    sample where each one starts. That makes this the only answer key in this
    repo that cannot itself be wrong - everything else is judged.

    The gap matters. With clips butted together, a backend that merges two
    utterances into one segment scores the same as one that finds the seam.
    Half a second of silence is long enough that missing the boundary is a real
    failure rather than a tie-break.
    """
    import numpy as np
    import soundfile as sf
    pieces, truth, cursor, rate = [], [], 0.0, None
    for row in rows:
        wav = os.path.join(REPO, row["human_wav"])
        if not os.path.exists(wav):
            continue
        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate is None:
            rate = sr
        elif sr != rate:
            continue                      # never resample silently
        dur = len(audio) / float(sr)
        truth.append({"id": row["id"], "start": round(cursor, 4),
                      "end": round(cursor + dur, 4), "text": row["text"]})
        pieces.append(audio)
        pieces.append(np.zeros(int(gap * sr), dtype="float32"))
        cursor += dur + gap
    if not pieces:
        return None, []
    sf.write(out_wav, np.concatenate(pieces), rate)
    return out_wav, truth


def score_alignment(truth, segments, tolerance=0.30):
    """Per true clip, the error of the nearest predicted segment start.

    Reports the distribution, not a single number, because the failure that
    hurts the preparer is the TAIL: a median of 40ms with a worst case of two
    seconds still ruins whichever training clips landed in the tail.
    """
    if not segments:
        return {"scored": 0, "note": "backend returned no timestamps"}
    errors = []
    for t in truth:
        nearest = min(segments, key=lambda s: abs(s[0] - t["start"]))
        errors.append(abs(nearest[0] - t["start"]))
    errors.sort()
    within = sum(1 for e in errors if e <= tolerance)
    return {
        "scored": len(errors),
        "predicted_segments": len(segments),
        "expected_segments": len(truth),
        "median_error_s": round(statistics.median(errors), 3),
        "p90_error_s": round(errors[int(len(errors) * 0.9) - 1], 3),
        "worst_error_s": round(errors[-1], 3),
        "within_tolerance_pct": round(within / len(errors) * 100, 1),
        "tolerance_s": tolerance,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", default=os.path.join(
        REPO, "ab_test_runtime", "ljspeech_eval", "build.json"))
    ap.add_argument("--backends", nargs="+",
                    default=["whisper_cpp", "transformers_whisper"])
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--align-clips", type=int, default=12,
                    help="clips concatenated into the alignment probe")
    ap.add_argument("--whisper-cpp-bin", default=os.path.join(
        REPO, "whisper.cpp", "build", "bin", "whisper-cli"))
    ap.add_argument("--whisper-cpp-model", default=os.path.join(
        REPO, "whisper.cpp", "models", "ggml-base.en.bin"))
    ap.add_argument("--lang", default="en",
                    help="language code passed to each backend (en/ja/zh)")
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "asr_backends.json"))
    args = ap.parse_args()

    build = json.load(open(args.build, encoding="utf-8"))
    rows = build["test"][:args.limit]
    if not rows:
        sys.exit("no test rows")

    # English-only checkpoints cannot decode ja/zh at all; using one would
    # measure the checkpoint choice rather than the backend.
    wcpp_model = args.whisper_cpp_model
    if args.lang != "en" and wcpp_model.endswith("base.en.bin"):
        wcpp_model = wcpp_model.replace("base.en.bin", "base.bin")

    def backend(name):
        if name == "whisper_cpp":
            return lambda w: run_whisper_cpp(
                w, wcpp_model, args.whisper_cpp_bin, language=args.lang)
        if name == "transformers_whisper":
            return lambda w: run_transformers_whisper(w, language=args.lang)
        if name == "whisper_cpp_hybrid":
            # Boundaries from base, words from large-v3. Both paths resolve
            # from the same models directory the single-model arm uses.
            models = os.path.dirname(wcpp_model)
            seg = os.path.join(models, "ggml-base.bin")
            txt = os.path.join(models, "ggml-large-v3.bin")
            return lambda w: run_whisper_cpp_hybrid(
                w, seg, txt, args.whisper_cpp_bin, language=args.lang)
        if name == "sensevoice":
            sv = {"en": "en", "ja": "ja", "zh": "zh"}.get(args.lang, "auto")
            return lambda w: run_sensevoice(w, language=sv)
        raise SystemExit(f"unknown backend {name}")

    import soundfile as sf
    results = {}
    for name in args.backends:
        fn = backend(name)
        wers, secs, audio_secs, failures, no_ts = [], [], [], [], 0
        for row in rows:
            wav = os.path.join(REPO, row["human_wav"])
            if not os.path.exists(wav):
                failures.append({"id": row["id"], "error": "missing wav"})
                continue
            try:
                t0 = time.time()
                text, segs = fn(wav)
                secs.append(time.time() - t0)
            except Exception as exc:                        # noqa: BLE001
                failures.append({"id": row["id"], "error": str(exc)[:160]})
                continue
            if not segs:
                no_ts += 1
            info = sf.info(wav)
            audio_secs.append(info.frames / float(info.samplerate))
            w = word_error_rate(row["text"], text)
            if w is not None:
                wers.append(w)
        rec = {"n": len(wers), "failures": failures[:6],
               "failed": len(failures), "clips_without_timestamps": no_ts}
        if wers:
            rec.update({
                "wer_mean": round(statistics.mean(wers), 4),
                "wer_median": round(statistics.median(wers), 4),
                "rtf": round(sum(secs) / max(sum(audio_secs), 1e-9), 3),
                "seconds_total": round(sum(secs), 1)})
        results[name] = rec
        if wers:
            print(f"  {name:22} WER {rec['wer_mean']*100:5.1f}%  "
                  f"RTF {rec['rtf']:.3f}  n={rec['n']}  failed={rec['failed']}")
        else:
            print(f"  {name:22} PRODUCED NOTHING  failed={rec['failed']}"
                  f"  first={failures[0]['error'][:70] if failures else ''}")

    # ── alignment probe ─────────────────────────────────────────────────
    # The deciding axis. A backend that hears every word but cannot place a
    # boundary cannot replace the align step, and the preparer needs both.
    probe_dir = os.path.join(REPO, "ab_test_runtime", "asr_bench")
    os.makedirs(probe_dir, exist_ok=True)
    probe_wav, truth = build_alignment_probe(
        build["test"][:args.align_clips], os.path.join(probe_dir, "probe.wav"))
    alignment = {}
    if probe_wav:
        total = truth[-1]["end"] if truth else 0
        print(f"\n  alignment probe: {len(truth)} clips, {total:.1f}s, "
              f"exact boundaries known")
        for name in args.backends:
            try:
                _, segs = backend(name)(probe_wav)
                alignment[name] = score_alignment(truth, segs)
            except Exception as exc:                        # noqa: BLE001
                alignment[name] = {"error": str(exc)[:160]}
            a = alignment[name]
            if a.get("scored"):
                print(f"  {name:22} median {a['median_error_s']*1000:6.0f}ms  "
                      f"p90 {a['p90_error_s']*1000:6.0f}ms  "
                      f"worst {a['worst_error_s']*1000:6.0f}ms  "
                      f"within {a['within_tolerance_pct']:.0f}%  "
                      f"segs {a['predicted_segments']}/{a['expected_segments']}")
            else:
                print(f"  {name:22} {a.get('note') or a.get('error','')[:70]}")
    else:
        print("\n  alignment probe could not be built")

    doc = {"build": os.path.relpath(args.build, REPO), "limit": args.limit,
           "language": args.lang, "whisper_cpp_model": os.path.basename(wcpp_model),
           "backends": args.backends, "results": results,
           "alignment": alignment,
           "alignment_truth_clips": len(truth) if probe_wav else 0}
    try:
        from experiments.provenance import provenance
        doc["provenance"] = provenance(__file__, args)
    except Exception as exc:                                # noqa: BLE001
        doc["provenance"] = {"error": str(exc)[:120]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
    print(f"\nwrote {args.out}")

    # A run where every backend failed is a failed run, not a published zero.
    if not any(r.get("n") for r in results.values()):
        sys.exit(3)


if __name__ == "__main__":
    main()
