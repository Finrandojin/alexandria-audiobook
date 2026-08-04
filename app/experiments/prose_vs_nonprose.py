"""Is Qwen3-TTS worse on non-prose, independently of length and of symbols?

The bullet-list defect is fixed: `normalize_for_speech` turns U+2022 into a
sentence break, and the segment that produced 24.2s of non-speech now scores
52 of 52 words on three consecutive runs. But that fix addresses SYMBOLS, and
the failing segment had two things wrong with it at once - it was full of
bullets AND it had no sentence structure. Fixing one does not clear the other.

Evidence the structural half survives: in the 60-segment baseline the ISBN
segment failed with digits dropped, and it contains no symbol the normaliser
touches.

THE CONFOUND, and why this is not just "generate front matter and count". Front
matter is SHORTER than prose and full of numbers, so a naive comparison would
report a length effect or a digit effect as a structure effect. Segments are
therefore matched into pairs of similar length, one from each class, and only
matched pairs are compared. The baseline already showed length running the
other way - longer segments scored BETTER - so an unmatched comparison would
have been actively misleading.

CLASSIFICATION is by explicit inspectable signals rather than position in the
book, because "first N chunks" is not a property a generator can act on:

    non-prose   short fragments ending in periods with no clause structure,
                high digit or capital density, publisher/rights vocabulary
    prose       ordinary narration and dialogue

Both classes are generated WITH the normaliser active, so a difference that
survives is structure, not symbols.

CORRECTION, 2026-08-03, after the run. The result below is real but MUCH
NARROWER than "non-prose fails". Auditing the 25 segments the sampler actually
chose:

    18 of 25 are near-duplicates of ONE line - "Identifiers LCCN 2016031562
       ISBN 9780316315302..." - repeated across Re:Zero volumes
     2 of 25 are not book text at all, but LLM quality-rejection messages that
       leaked into scripts/*.json
    ~5 genuinely distinct texts underlie all 25 "samples"

So the measured 56-point gap says IDENTIFIER AND ISBN STRINGS FAIL BADLY. It
does not establish anything about non-prose or structure in general, because
the class was five texts replicated rather than 25 independent draws, and the
effective n is nearer 5 than 25. The headline was stated wider than the data.

Two fixes are needed before this claim can be widened: deduplicate near
identical texts in the sampler, and exclude entries that are machine output
rather than book text. Neither is done here; this docstring is the correction.

READINGS, fixed before running:

  non-prose much worse    structure is a real, separate failure mode and the
                          symbol fix was half a fix. Front matter needs its own
                          handling - most simply, not being sent as one chunk.
  no difference           the bullet was the whole story and the fix is
                          complete. Front matter is safe as it stands.
"""
import argparse, collections, json, os, re, statistics, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

PUBLISHER_WORDS = {"copyright", "isbn", "published", "publisher", "rights",
                   "reserved", "edition", "translation", "printing", "llc",
                   "inc", "press", "corporation", "trademark", "scanning",
                   "uploading", "permission", "ebook", "cover", "art"}


def nonprose_score(text):
    """0-1. Higher means less like a sentence a narrator reads.

    Four independent signals, so no single quirk decides it. Kept explicit
    rather than learned because the point is to inspect what was measured.
    """
    words = re.findall(r"[^\W\d_]+", text)
    if not words:
        return 1.0
    fragments = [f.strip() for f in re.split(r"[.!?]", text) if f.strip()]
    mean_fragment = (statistics.mean(len(f.split()) for f in fragments)
                     if fragments else 0)
    digits = sum(c.isdigit() for c in text) / max(len(text), 1)
    caps = sum(1 for w in words if len(w) > 1 and w.isupper()) / len(words)
    publisher = sum(1 for w in words
                    if w.lower() in PUBLISHER_WORDS) / len(words)
    return min(1.0, (
        (1.0 if mean_fragment < 6 else 0.0) * 0.35 +
        min(digits * 12, 1.0) * 0.2 +
        min(caps * 4, 1.0) * 0.2 +
        min(publisher * 8, 1.0) * 0.25))


def classify(text, high=0.45, low=0.12):
    score = nonprose_score(text)
    if score >= high:
        return "nonprose"
    if score <= low:
        return "prose"
    return None          # ambiguous: excluded rather than guessed


def match_pairs(chunks, tolerance=0.25, limit=15):
    """Pair each non-prose segment with a prose one of similar length.

    Without this the comparison measures length, which the baseline already
    showed runs the OTHER way - longer scored better - so an unmatched result
    would understate any structural effect or invent one.
    """
    tagged = [(classify(c["text"]), c) for c in chunks]
    nonprose = [c for t, c in tagged if t == "nonprose"]
    prose = sorted([c for t, c in tagged if t == "prose"],
                   key=lambda c: len(c["text"]))
    used, pairs = set(), []
    for np_chunk in sorted(nonprose, key=lambda c: -len(c["text"])):
        target = len(np_chunk["text"])
        best = None
        for p in prose:
            if p["uid"] in used:
                continue
            if abs(len(p["text"]) - target) <= target * tolerance:
                if best is None or abs(len(p["text"]) - target) < abs(len(best["text"]) - target):
                    best = p
        if best:
            used.add(best["uid"])
            pairs.append((np_chunk, best))
        if len(pairs) >= limit:
            break
    return pairs


def load_chunks(args):
    """Segments to draw from, optionally pooled across the whole library.

    The live book holds only FIVE non-prose segments, which cannot separate
    anything. Pooling `scripts/*.json` raises that enough to measure. Pooled
    segments carry no voice assignment of their own, so everything is generated
    with one fixed voice - which also removes voice as a confound, since front
    matter is narrator-only while prose is spread across a cast.
    """
    if not args.pool_library:
        return [c for c in json.load(open(args.script, encoding="utf-8"))
                if c.get("text") and c.get("uid") and len(c["text"]) >= 40]
    import glob, hashlib
    seen, out = set(), []
    for path in sorted(glob.glob(os.path.join(REPO, "scripts", "*.json"))):
        if "voice_config" in path or "generation_quality" in path:
            continue
        try:
            doc = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        entries = doc if isinstance(doc, list) else (
            doc.get("entries") or doc.get("chunks") or [])
        for e in entries:
            if not isinstance(e, dict):
                continue
            text = (e.get("text") or "").strip()
            if len(text) < 40 or text in seen:
                continue
            seen.add(text)
            out.append({"text": text, "instruct": e.get("instruct", ""),
                        "speaker": args.voice,
                        "uid": hashlib.md5(text.encode()).hexdigest()[:12]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--script", default=os.path.join(REPO, "chunks.json"))
    ap.add_argument("--voice-config", default=os.path.join(REPO, "voice_config.json"))
    ap.add_argument("--config", default=os.path.join(APP, "config.json"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "ab_test_runtime", "prose_audio"))
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--pool-library", action="store_true",
                    help="draw from scripts/*.json instead of the live book")
    ap.add_argument("--voice", default="NARRATOR",
                    help="single voice used for every pooled segment")
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "prose_vs_nonprose.json"))
    args = ap.parse_args()

    chunks = load_chunks(args)
    pairs = match_pairs(chunks, limit=args.limit)
    if not pairs:
        print("no matched pairs found")
        return
    lengths = [(len(a["text"]), len(b["text"])) for a, b in pairs]
    print(f"{len(pairs)} matched pairs; mean length "
          f"nonprose {statistics.mean(x for x, _ in lengths):.0f} vs "
          f"prose {statistics.mean(y for _, y in lengths):.0f} chars\n")

    raw_vc = json.load(open(args.voice_config, encoding="utf-8"))
    voice_config = (raw_vc.get("characters")
                    if isinstance(raw_vc.get("characters"), dict) else raw_vc)
    os.makedirs(args.out_dir, exist_ok=True)

    from tts import TTSEngine, voice_category
    from experiments.tts_output_validation import transcribe, validate
    engine = TTSEngine(json.load(open(args.config, encoding="utf-8")))

    rows = []
    for i, pair in enumerate(pairs, 1):
        for label, chunk in (("nonprose", pair[0]), ("prose", pair[1])):
            speaker = chunk.get("speaker")
            voice_data = voice_config.get(speaker) or {}
            wav = os.path.join(args.out_dir, f"{label}_{chunk['uid']}.wav")
            try:
                if voice_category(voice_data) == "lora":
                    engine.generate_lora_voice(chunk["text"],
                                               chunk.get("instruct", ""),
                                               voice_data, wav)
                elif voice_category(voice_data) == "clone":
                    engine.generate_clone_voice(chunk["text"], speaker,
                                                voice_config, wav)
                else:
                    engine.generate_custom_voice(chunk["text"],
                                                 chunk.get("instruct", ""),
                                                 speaker, voice_config, wav)
                r = validate(chunk["text"], transcribe(wav))
            except Exception as exc:                  # noqa: BLE001
                print(f"  [{i}] {label} FAILED: {str(exc)[:80]}")
                continue
            r.update({"class": label, "chars": len(chunk["text"]),
                      "uid": chunk["uid"], "wav": wav})
            r.pop("detail", None)
            rows.append(r)
            print(f"  [{i}/{len(pairs)}] {label:9} {len(chunk['text']):4}ch  "
                  f"{r['errors']:3}/{r['threshold']:<3} err  "
                  f"{'FAIL' if r['failed'] else 'ok'}"
                  f"{'  NON-SPEECH' if r['non_speech'] else ''}")

    print()
    summary = {}
    for label in ("nonprose", "prose"):
        sel = [r for r in rows if r["class"] == label]
        if not sel:
            continue
        wer = sum(r["errors"] for r in sel) / max(sum(r["words"] for r in sel), 1)
        summary[label] = {"n": len(sel), "wer": wer,
                          "failed": sum(r["failed"] for r in sel),
                          "non_speech": sum(r["non_speech"] for r in sel),
                          "mean_chars": statistics.mean(r["chars"] for r in sel)}
        print(f"  {label:9} n={len(sel):3}  WER {wer*100:6.2f}%  "
              f"failed {summary[label]['failed']:2}  "
              f"non-speech {summary[label]['non_speech']:2}  "
              f"mean {summary[label]['mean_chars']:.0f} chars")

    if len(summary) == 2:
        d = summary["nonprose"]["wer"] - summary["prose"]["wer"]
        print(f"\n  non-prose WER is {d*100:+.2f} points vs prose at matched "
              f"length.")
        print("  Both classes were generated WITH the symbol normaliser "
              "active, so a\n  difference here is STRUCTURE, not symbols.")
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
