"""Test fresh-process determinism and whether extreme instructions reach TTS.

This is deliberately a control experiment, not a production feature.  Each
render runs in a new Python process, so equality cannot be attributed to a
model or RNG state retained by an earlier arm.  The instruction controls use
the same text, adapter and seed; only the instruction changes.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import wave

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
DEFAULT_OUT_DIR = os.path.join(REPO, "ab_test_runtime", "assumption_controls")
DEFAULT_OUT = os.path.join(
    REPO, "ab_test_runtime", "experiments", "seed_instruction_controls.json")

DEFAULT_SPEAKERS = ("NARRATOR", "EMILIA", "NATSUKI SUBARU")
TEXT = "The lantern trembled in her hand as the footsteps approached the door."
INSTRUCTIONS = {
    "neutral": "",
    "very_slow": "Speak extremely slowly, with long deliberate pauses between phrases.",
    "very_fast": "Speak extremely quickly, with urgent rapid delivery and no long pauses.",
}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def wav_info(path):
    with wave.open(path, "rb") as fh:
        frames, rate = fh.getnframes(), fh.getframerate()
    return {"sha256": file_sha256(path), "frames": frames, "rate": rate,
            "duration_s": frames / rate if rate else 0.0}


def render_worker(args):
    sys.path.insert(0, APP)
    raw = json.load(open(args.voice_config, encoding="utf-8"))
    voices = raw.get("characters", raw)
    if args.speaker not in voices:
        raise SystemExit(f"speaker not found in voice config: {args.speaker}")
    entry = dict(voices[args.speaker])
    if entry.get("type") != "lora" or not entry.get("adapter_path"):
        raise SystemExit(f"speaker has no configured LoRA adapter: {args.speaker}")
    entry["seed"] = str(args.seed)
    from tts import TTSEngine
    from experiments.generation import render
    engine = TTSEngine(json.load(open(args.config, encoding="utf-8")))
    render(engine, args.text, args.instruct, args.speaker, voices, entry, args.wav)
    print(json.dumps(wav_info(args.wav), sort_keys=True))


def run_child(args, speaker, seed, instruct, wav):
    cmd = [sys.executable, os.path.abspath(__file__), "--worker",
           "--speaker", speaker, "--seed", str(seed), "--text", args.text,
           "--instruct", instruct, "--wav", wav,
           "--voice-config", args.voice_config, "--config", args.config]
    proc = subprocess.run(cmd, cwd=APP, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"worker failed for {speaker}, seed={seed}: "
                           f"{proc.stderr[-1000:]}")
    return wav_info(wav)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--speakers", nargs="+", default=list(DEFAULT_SPEAKERS))
    ap.add_argument("--speaker", default="")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--other-seed", type=int, default=5678)
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--instruct", default="")
    ap.add_argument("--wav", default="")
    ap.add_argument("--voice-config", default=os.path.join(REPO, "voice_config.json"))
    ap.add_argument("--config", default=os.path.join(APP, "config.json"))
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    if args.worker:
        if not args.wav:
            ap.error("--worker requires --wav")
        render_worker(args)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for speaker in args.speakers:
        renders = {}
        arms = (("same_seed_a", args.seed, ""),
                ("same_seed_b", args.seed, ""),
                ("different_seed", args.other_seed, ""))
        for arm, seed, instruct in arms:
            wav = os.path.join(args.out_dir, f"{speaker}_{arm}.wav")
            renders[arm] = run_child(args, speaker, seed, instruct, wav)
            renders[arm].update({"seed": seed, "instruction": instruct,
                                 "file": os.path.relpath(wav, REPO)})
        deterministic = renders["same_seed_a"]["sha256"] == \
            renders["same_seed_b"]["sha256"]
        seed_varies = renders["same_seed_a"]["sha256"] != \
            renders["different_seed"]["sha256"]

        instruction = {}
        for arm, instruct in INSTRUCTIONS.items():
            wav = os.path.join(args.out_dir, f"{speaker}_instruction_{arm}.wav")
            instruction[arm] = run_child(args, speaker, args.seed, instruct, wav)
            instruction[arm].update({"seed": args.seed, "instruction": instruct,
                                     "file": os.path.relpath(wav, REPO)})
        duration_order = instruction["very_slow"]["duration_s"] > \
            instruction["very_fast"]["duration_s"]
        rows.append({"speaker": speaker, "deterministic": deterministic,
                     "different_seed_varies": seed_varies,
                     "duration_order_control_passes": duration_order,
                     "renders": renders, "instruction_controls": instruction})
        print(speaker, "deterministic=", deterministic,
              "different_seed_varies=", seed_varies,
              "slow_gt_fast=", duration_order)

    from experiments.provenance import provenance
    result = {"provenance": provenance(__file__, args), "text": args.text,
              "rows": rows,
              "interpretation": {
                  "determinism": "SHA-256 equality across fresh processes",
                  "instruction_positive_control": "Slow duration greater than fast is only a plumbing positive control; delivery quality still requires blinded listening.",
              }}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
