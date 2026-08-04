import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.speaker_similarity import MetricUnavailable, get_ecapa_encoder


class TestSpeakerSimilarity(unittest.TestCase):
    def test_missing_speechbrain_fails_instead_of_falling_back(self):
        with patch.dict(sys.modules, {"speechbrain": None,
                                      "speechbrain.inference": None,
                                      "speechbrain.inference.speaker": None}):
            with self.assertRaises(MetricUnavailable) as cm:
                get_ecapa_encoder("unused")
        self.assertIn("requires speechbrain", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
