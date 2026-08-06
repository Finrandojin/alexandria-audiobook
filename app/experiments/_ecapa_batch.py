"""Score speaker-embedding pairs in the interpreter that has speechbrain.

Run as a subprocess by `ljspeech_score.py`, never imported. `app/env` has no
speechbrain, and the existing `voice_data_saturation.embedder()` responds to
that by silently returning None and falling back to acoustic-feature distance -
which is precisely the substitution of a weaker metric for a stronger one that
the 2026-08-04 test plan forbids.

Reads a JSON list of [a, b] paths on stdin, writes a JSON list of cosine
similarities on the last stdout line. Exits non-zero if speechbrain is absent,
so the caller reports "not measured" instead of quietly measuring something
else.
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def main():
    pairs = json.loads(sys.stdin.read())
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception as exc:                            # noqa: BLE001
        print(f"speechbrain unavailable: {exc}", file=sys.stderr)
        return 2

    import numpy as np
    import soundfile as sf
    import torch
    # CPU deliberately. The sibling environment runs an older ROCm build whose
    # device-string parsing core-dumped here ("Could not parse CUDA device
    # string 'cuda'"), and this is a scoring pass over a few hundred short
    # clips - fast enough on CPU, and it leaves the card free for generation
    # instead of contending with it.
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(REPO, "ab_test_runtime", "ecapa"),
        run_opts={"device": "cpu"})

    cache = {}

    def embed(path):
        """Load with soundfile, not torchaudio.

        `torchaudio.load` CORE DUMPS in this interpreter - torch, torchaudio
        and speechbrain all import cleanly, but the loader's backend dispatch
        segfaults. soundfile is present here and does the same job, so the
        broken path is routed around rather than debugged in an environment
        this project does not own.
        """
        if path in cache:
            return cache[path]
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)                       # to mono
        if rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
        wav = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
        vec = enc.encode_batch(wav).squeeze().detach().cpu().numpy()
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        cache[path] = vec
        return vec

    out, failed = [], 0
    for a, b in pairs:
        try:
            out.append(float(np.dot(embed(a), embed(b))))
        except Exception as exc:                        # noqa: BLE001
            print(f"pair failed {a} {b}: {exc}", file=sys.stderr)
            out.append(None)
            failed += 1
    # A run where EVERY pair failed - a wrong working directory, a missing
    # tree - previously returned a list of nulls and exit 0, so the caller
    # reported "mean = None" as though the metric had simply been quiet. That
    # is the silent-degradation failure this file exists to prevent, so it is
    # an error here rather than a shrug.
    if pairs and failed == len(pairs):
        print(f"every one of {failed} pairs failed - check the paths",
              file=sys.stderr)
        return 3
    if failed:
        print(f"{failed} of {len(pairs)} pairs failed", file=sys.stderr)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
