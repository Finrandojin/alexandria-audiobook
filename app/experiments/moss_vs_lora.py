"""Zero-shot cloning against a trained LoRA, on the same speaker and lines.

THE QUESTION THIS ANSWERS. MOSS-TTSD clones a voice from one short reference
clip with no training. If that lands near what a trained LoRA achieves, the
entire adapter pipeline becomes optional: goal 2.7's 60 contaminated adapters,
the Voice Lab chain, the reference-clip selection work - all of it exists to
produce something a zero-shot model would give for free. If it lands far below,
it is a non-starter for voice fidelity regardless of how good its dialogue
handling is, and the honest answer is worth one GPU run either way.

WHY THIS COMPARISON IS FAIR, AND WHERE IT IS NOT.

Fair: both sides are scored by the same ECAPA speaker embedding against the
SAME held-out human recordings, using the val split the LoRA never trained on.
The reference clip handed to MOSS comes from the train split, so neither system
sees the evaluation audio.

Not fair to MOSS, deliberately: it is being judged on single-speaker identity
match, which is not what it is built for. Its claims are about dialogue flow,
turn-taking and 60-minute consistency. A weak score here means "do not replace
the LoRA pipeline with it", NOT "the model is bad". That distinction has to
survive into whatever gets written down.

Not fair to the LoRA: it was trained on 180 clips of this speaker. MOSS gets
one clip. That is the point - the comparison is convenience against fidelity,
and the number decides whether the convenience is affordable.

WHAT COUNTS AS A RESULT. The LoRA scores for these speakers are 0.53-0.69
held-out. Zero-shot near 0.60 would make the pipeline optional. Near 0.35 would
close the question. Anything between is a real trade-off to think about rather
than a verdict.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(APP)
sys.path.insert(0, APP)

SIBLING_PY = os.environ.get(
    "ALEXANDRIA_SIBLING_PYTHON",
    os.path.join(os.path.dirname(REPO), "alexandria-audiobook.git",
                 "app", "env", "bin", "python"))


def load_split(data_dir, split):
    """-> (rows, base). `audio_filepath` is relative to data_dir, not to the
    split directory: a val row reads "val/sample_0217.wav", so joining it to
    data/val would produce data/val/val/... and find nothing."""
    path = os.path.join(data_dir, split, "metadata.jsonl")
    if not os.path.exists(path):
        path = os.path.join(data_dir, "metadata.jsonl")
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows, data_dir


def ecapa_pairs(pairs):
    """Cosine similarity per (a, b) wav pair via the sibling interpreter."""
    if not pairs or not os.path.exists(SIBLING_PY):
        return None
    script = os.path.join(APP, "experiments", "_ecapa_batch.py")
    out = subprocess.run(
        [SIBLING_PY, script],
        input=json.dumps([[os.path.abspath(a), os.path.abspath(b)]
                          for a, b in pairs]),
        capture_output=True, text=True, timeout=1800, cwd=APP)
    if out.returncode != 0:
        return None
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def synthesise(model, processor, reference_wav, reference_text, text, out_path):
    import torch
    import soundfile as sf
    wav, _sr = sf.read(reference_wav, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav).transpose(0, 1)
    codes = processor.encode_audios_from_wav(
        [wav], sampling_rate=int(processor.model_config.sampling_rate))
    # One speaker only: the tag exists so the model knows whose turn it is, and
    # a single-speaker script is the narrowest case its format allows.
    full = f"{reference_text} [S1] {text}"
    conversations = [[
        processor.build_user_message(text=full, reference=codes),
        processor.build_assistant_message(audio_codes_list=[codes[0]]),
    ]]
    batch = processor(conversations, mode="continuation")
    outputs = model.generate(
        input_ids=batch["input_ids"].to("cuda"),
        attention_mask=batch["attention_mask"].to("cuda"),
        max_new_tokens=2000)
    audio = processor.decode(outputs)[0].audio_codes_list[0]
    sf.write(out_path, audio.cpu().numpy(),
             int(processor.model_config.sampling_rate))
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--lines", type=int, default=10,
                    help="held-out lines per speaker")
    ap.add_argument("--work", default=os.path.join(
        REPO, "ab_test_runtime", "moss_vs_lora"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "moss_vs_lora.json"))
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoProcessor

    os.makedirs(args.work, exist_ok=True)
    print("  loading MOSS-TTSD (8B, bfloat16)...", flush=True)
    processor = AutoProcessor.from_pretrained(
        "OpenMOSS-Team/MOSS-TTSD-v1.0", trust_remote_code=True,
        codec_path="OpenMOSS-Team/MOSS-Audio-Tokenizer")
    model = AutoModel.from_pretrained(
        "OpenMOSS-Team/MOSS-TTSD-v1.0", trust_remote_code=True,
        torch_dtype=torch.bfloat16).to("cuda")
    print("  loaded", flush=True)

    lora_scores = {r["adapter"]: r.get("ecapa") for r in json.load(open(
        os.path.join(REPO, "ab_test_runtime", "experiments",
                     "library_voice_fidelity_n10.json"),
        encoding="utf-8"))["results"]}

    results = []
    for adapter in args.adapters:
        data_dir = os.path.join(REPO, "ab_test_runtime", "retrain_honest",
                                adapter, "data")
        if not os.path.isdir(data_dir):
            print(f"  SKIP {adapter}: no dataset"); continue
        train_rows, train_base = load_split(data_dir, "train")
        val_rows, val_base = load_split(data_dir, "val")
        if not train_rows or not val_rows:
            print(f"  SKIP {adapter}: missing split"); continue

        # The reference comes from TRAIN so the evaluation audio stays unseen.
        ref = train_rows[0]
        ref_wav = os.path.join(train_base, ref["audio_filepath"])
        ref_text = ref.get("text") or ""

        pairs, made = [], 0
        for row in val_rows[:args.lines]:
            human = os.path.join(val_base, row["audio_filepath"])
            text = row.get("text") or ""
            if not (os.path.exists(human) and text.strip()):
                continue
            out_wav = os.path.join(args.work, f"{adapter}__{made:03d}.wav")
            try:
                synthesise(model, processor, ref_wav, ref_text, text, out_wav)
            except Exception as exc:                        # noqa: BLE001
                print(f"    {adapter} line {made}: FAILED {str(exc)[:70]}")
                continue
            pairs.append((human, out_wav))
            made += 1
            print(f"    {adapter}: {made}/{args.lines}", flush=True)

        sims = ecapa_pairs(pairs) or []
        clean = [s for s in sims if s is not None]
        zero_shot = round(statistics.median(clean), 4) if clean else None
        results.append({
            "adapter": adapter, "lines_scored": len(clean),
            "zero_shot_ecapa": zero_shot,
            "lora_ecapa_shipped": lora_scores.get(adapter),
            "reference_clip": os.path.basename(ref_wav),
        })
        print(f"  {adapter}: zero-shot {zero_shot}  "
              f"(shipped LoRA {lora_scores.get(adapter)})", flush=True)

    report = {
        "scope": "single-speaker identity match only. MOSS-TTSD is built for "
                 "dialogue flow, turn-taking and long-form consistency; a weak "
                 "score here means do not replace the LoRA pipeline with it, "
                 "not that the model is weak.",
        "lines_per_speaker": args.lines,
        "reference_from": "train split (evaluation audio never seen by either)",
        "results": results,
    }
    from utils import atomic_json_write
    atomic_json_write(report, args.out)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
