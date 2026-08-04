import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.nonprose_replication import (feature_gap, summarize,
                                               surface_features)


class TestNonproseReplication(unittest.TestCase):
    def test_surface_features_are_explicit_and_bounded(self):
        features = surface_features("ISBN 12 / Title!")
        self.assertEqual(16, features["chars"])
        self.assertEqual(2, features["words"])
        for key in ("digit_fraction", "uppercase_word_fraction",
                    "punctuation_fraction"):
            self.assertGreaterEqual(features[key], 0)
            self.assertLessEqual(features[key], 1)

    def test_feature_gap_does_not_claim_exact_matching(self):
        gap = feature_gap("ISBN 12.", "A sentence.")
        self.assertGreater(gap["digit_fraction"], 0)
        self.assertGreater(gap["uppercase_word_fraction"], 0)

    def test_summary_keeps_adapter_seed_and_class_separate(self):
        base = {"adapter": "a", "seed": 1, "words": 10,
                "errors": 2, "failed": True, "substitutions": 1,
                "deletions": 0, "insertions": 1}
        rows = [{**base, "class": "nonprose"},
                {**base, "class": "prose", "errors": 0,
                 "failed": False, "substitutions": 0, "insertions": 0}]
        summary = summarize(rows)
        self.assertEqual(2, len(summary))
        by_class = {r["class"]: r for r in summary}
        self.assertEqual(1, by_class["nonprose"]["insertions"])
        self.assertEqual(0, by_class["prose"]["insertions"])


if __name__ == "__main__":
    unittest.main()
