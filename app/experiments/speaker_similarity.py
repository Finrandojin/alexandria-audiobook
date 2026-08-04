"""Shared, explicit speaker-similarity scoring for TTS experiments."""
import os

import numpy as np


class MetricUnavailable(RuntimeError):
    """The requested metric cannot run in this interpreter."""


def get_ecapa_encoder(cache_dir):
    """Load ECAPA or fail loudly; never substitute a different metric."""
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception as exc:                            # noqa: BLE001
        raise MetricUnavailable(
            "ECAPA scoring requires speechbrain; use the configured sibling "
            "interpreter for the scoring phase") from exc
    try:
        return EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.abspath(cache_dir))
    except Exception as exc:                            # noqa: BLE001
        raise MetricUnavailable(f"ECAPA model could not load: {exc}") from exc


def get_ecapa_embedding(encoder, path):
    import math
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    waveform, rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if not len(waveform) or not rate:
        raise ValueError(f"speaker-similarity input has no audio: {path}")
    if rate != 16000:
        divisor = math.gcd(int(rate), 16000)
        waveform = resample_poly(waveform, 16000 // divisor,
                                 int(rate) // divisor).astype("float32")
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    with torch.no_grad():
        value = encoder.encode_batch(waveform).squeeze().detach().cpu().numpy()
    return value / (np.linalg.norm(value) + 1e-9)


def score_ecapa_pair(encoder, first, second):
    """Return cosine similarity between two audio files."""
    return float(np.dot(get_ecapa_embedding(encoder, first),
                        get_ecapa_embedding(encoder, second)))
