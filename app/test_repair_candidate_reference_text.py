import unittest

from experiments.repair_candidate_reference_text import get_repaired_metadata


class RepairCandidateReferenceTextTests(unittest.TestCase):
    def test_returns_copy_with_matching_text(self):
        original = {"ref_sample_text": "wrong", "num_samples": 180}
        repaired = get_repaired_metadata(original, "matching transcript")
        self.assertEqual("wrong", original["ref_sample_text"])
        self.assertEqual("matching transcript", repaired["ref_sample_text"])
        self.assertEqual(180, repaired["num_samples"])


if __name__ == "__main__":
    unittest.main()
