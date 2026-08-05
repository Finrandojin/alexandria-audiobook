import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.speaker_similarity import (MetricUnavailable,
                                              get_ecapa_embedding,
                                              get_ecapa_encoder)


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    @property
    def shape(self):
        return self.value.shape

    def unsqueeze(self, axis):
        return _Tensor(np.expand_dims(self.value, axis))

    def squeeze(self):
        return _Tensor(np.squeeze(self.value))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


_FAKE_TORCH = types.ModuleType("torch")
_FAKE_TORCH.from_numpy = lambda value: _Tensor(value)
_FAKE_TORCH.tensor = lambda value: _Tensor(value)
_FAKE_TORCH.no_grad = lambda: _NoGrad()


class TestSpeakerSimilarity(unittest.TestCase):
    def test_missing_speechbrain_fails_instead_of_falling_back(self):
        with patch.dict(sys.modules, {"speechbrain": None,
                                      "speechbrain.inference": None,
                                      "speechbrain.inference.speaker": None}):
            with self.assertRaises(MetricUnavailable) as cm:
                get_ecapa_encoder("unused")
        self.assertIn("requires speechbrain", str(cm.exception))

    def test_embedding_decodes_and_resamples_without_torchaudio(self):
        class Encoder:
            def __init__(self):
                self.received = None

            def encode_batch(self, waveform):
                self.received = waveform
                return _FAKE_TORCH.tensor([[[3.0, 4.0]]])

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "stereo.wav")
            audio = np.zeros((2400, 2), dtype="float32")
            audio[:, 0] = 0.25
            audio[:, 1] = -0.125
            sf.write(path, audio, 24000)
            encoder = Encoder()
            with patch.dict(sys.modules, {"torch": _FAKE_TORCH}):
                embedding = get_ecapa_embedding(encoder, path)
        self.assertEqual((1, 1600), tuple(encoder.received.shape))
        np.testing.assert_allclose([0.6, 0.8], embedding, atol=1e-6)

    def test_empty_audio_fails_loudly(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "empty.wav")
            sf.write(path, np.array([], dtype="float32"), 16000)
            with patch.dict(sys.modules, {"torch": _FAKE_TORCH}):
                with self.assertRaisesRegex(ValueError, "has no audio"):
                    get_ecapa_embedding(object(), path)


if __name__ == "__main__":
    unittest.main()
