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
    import torch
    import torchaudio
    waveform, rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if rate != 16000:
        waveform = torchaudio.functional.resample(waveform, rate, 16000)
    with torch.no_grad():
        value = encoder.encode_batch(waveform).squeeze().detach().cpu().numpy()
    return value / (np.linalg.norm(value) + 1e-9)


def score_ecapa_pair(encoder, first, second):
    """Return cosine similarity between two audio files."""
    return float(np.dot(get_ecapa_embedding(encoder, first),
                        get_ecapa_embedding(encoder, second)))
