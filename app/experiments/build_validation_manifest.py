"""Generate a stratified audio sample so the output gate has something to check.

`tts_output_validation.py` needs {text, wav} pairs and nothing in this project
has ever generated audio for inspection - chunks.json is 2606 chunks all
`pending`. This produces the sample, deliberately shaped around the two open
questions rather than drawn at random:

  the 200-character cap   tts.py's external clone path passes
                          max_chunk_chars=200; generate_lora_voice passes no
                          cap, and the local Qwen3TTSModel.generate_voice_clone
                          does no internal splitting. Over-sampling the
                          201-500 band against a <=200 control is what tells
                          the two apart. If word-error rate and truncation
                          climb with length, the local path needs the cap.

  verbalization           digits and roman numerals reach ~1.2% of the live
                          library. Whether Qwen3-TTS reads "XIV" correctly is
                          unknown, and every such line is included so the
                          answer comes free with this run.

WHAT THIS SAMPLE CANNOT SETTLE. The live book tops out at 622 characters. The
extreme tail seen elsewhere in the corpus - grimgar03 reaches 2742 - is absent,
so a clean result here is evidence about 200-600 and says nothing about 2700.

Voices are resolved exactly as production does, through `tts.voice_category`,
so a defect found here is a defect a listener would hear.
"""
import argparse, collections, json, os, random, re, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

DIGIT_RE = re.compile(r"\d")
ROMAN_RE = re.compile(
    r"\b(?=[MDCLXVI]{2,})M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})\b")


def stratified_sample(chunks, per_band, seed=1234):
    """Pick chunks across length bands, plus every verbalization case.

    Sampling is seeded so a rerun compares like with like; an unseeded sample
    would make two runs differ for reasons unrelated to what changed.
    """
    rng = random.Random(seed)
    bands = [("<=200", 0, 200), ("201-500", 201, 500), (">500", 501, 10 ** 9)]
    picked, seen = [], set()
    for label, lo, hi in bands:
        pool = [c for c in chunks if lo <= len(c["text"]) <= hi]
        for c in rng.sample(pool, min(per_band, len(pool))):
            if c["uid"] not in seen:
                seen.add(c["uid"])
                picked.append((label, c))
    for c in chunks:
        if c["uid"] in seen:
            continue
        if DIGIT_RE.search(c["text"]) or ROMAN_RE.search(c["text"]):
            seen.add(c["uid"])
            picked.append(("verbalization", c))
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--script", default=os.path.join(REPO, "chunks.json"))
    ap.add_argument("--voice-config", default=os.path.join(REPO, "voice_config.json"))
    ap.add_argument("--config", default=os.path.join(APP, "config.json"))
    ap.add_argument("--out-dir", default=os.path.join(
        REPO, "ab_test_runtime", "validation_audio"))
    ap.add_argument("--per-band", type=int, default=20)
    ap.add_argument("--manifest", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "validation_manifest.json"))
    args = ap.parse_args()

    chunks = [c for c in json.load(open(args.script, encoding="utf-8"))
              if c.get("text") and c.get("uid")]
    raw_vc = json.load(open(args.voice_config, encoding="utf-8"))
    voice_config = (raw_vc.get("characters")
                    if isinstance(raw_vc.get("characters"), dict) else raw_vc)
    config = json.load(open(args.config, encoding="utf-8"))

    picked = stratified_sample(chunks, args.per_band)
    counts = collections.Counter(label for label, _ in picked)
    print(f"{len(picked)} segments selected: " +
          ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    os.makedirs(args.out_dir, exist_ok=True)
    from tts import TTSEngine, voice_category
    engine = TTSEngine(config)

    manifest, failures = [], []
    for i, (label, chunk) in enumerate(picked, 1):
        speaker = chunk.get("speaker")
        voice_data = voice_config.get(speaker) or {}
        category = voice_category(voice_data)
        wav = os.path.join(args.out_dir, f"{chunk['uid']}.wav")
        instruct = chunk.get("instruct") or ""
        try:
            if category == "lora":
                engine.generate_lora_voice(chunk["text"], instruct, voice_data, wav)
            elif category == "clone":
                engine.generate_clone_voice(chunk["text"], speaker, voice_config, wav)
            else:
                engine.generate_custom_voice(chunk["text"], instruct, speaker,
                                             voice_config, wav)
        except Exception as exc:                      # noqa: BLE001
            # A generation failure is itself a finding; recording it and
            # continuing beats losing the whole run to one bad segment.
            failures.append({"uid": chunk["uid"], "error": str(exc)[:200]})
            print(f"  [{i}/{len(picked)}] FAILED {speaker}: {str(exc)[:90]}")
            continue
        if not os.path.exists(wav):
            failures.append({"uid": chunk["uid"], "error": "no file written"})
            continue
        manifest.append({"text": chunk["text"], "wav": wav, "band": label,
                         "speaker": speaker, "category": category,
                         "chars": len(chunk["text"])})
        print(f"  [{i}/{len(picked)}] ok {label:14} {len(chunk['text']):4} chars"
              f"  {speaker}")

    json.dump(manifest, open(args.manifest, "w"), indent=1)
    print(f"\n{len(manifest)} generated, {len(failures)} failed")
    if failures:
        print("  failures:", json.dumps(failures[:5], indent=1)[:400])
    print("wrote", args.manifest)


if __name__ == "__main__":
    main()
