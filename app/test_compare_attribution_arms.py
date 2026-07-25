import unittest

from compare_attribution_arms import find_disagreements, sample_disagreements


class FindDisagreementsTest(unittest.TestCase):

    def test_matching_arms_have_no_disagreements(self):
        arm_a = [{"speaker": "ARARAGI", "text": "Hi"},
                 {"speaker": "HACHIKUJI", "text": "Bye"}]
        self.assertEqual(find_disagreements(arm_a, list(arm_a)), [])

    def test_differing_speaker_is_reported(self):
        arm_a = [{"speaker": "ARARAGI", "text": "Hi"}]
        arm_b = [{"speaker": "HANEKAWA", "text": "Hi"}]
        found = find_disagreements(arm_a, arm_b)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["index"], 0)
        self.assertEqual(found[0]["arm_a"], "ARARAGI")
        self.assertEqual(found[0]["arm_b"], "HANEKAWA")
        self.assertEqual(found[0]["text"], "Hi")

    def test_empty_arm_yields_no_disagreements(self):
        # Previously raised on any length mismatch, which made the tool useless
        # against real arms: segmentation is not deterministic.
        self.assertEqual(find_disagreements([{"speaker": "A", "text": "x"}], []), [])

    def test_null_entries_are_skipped(self):
        arm_a = [None, {"speaker": "ARARAGI", "text": "Hi"}]
        arm_b = [None, {"speaker": "HANEKAWA", "text": "Hi"}]
        found = find_disagreements(arm_a, arm_b)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["index"], 1)


class SampleDisagreementsTest(unittest.TestCase):

    def test_sample_is_deterministic_for_a_seed(self):
        rows = [{"index": i, "arm_a": "A", "arm_b": "B", "text": str(i)}
                for i in range(200)]
        first = sample_disagreements(rows, size=50, seed=7)
        second = sample_disagreements(rows, size=50, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 50)

    def test_sample_smaller_than_size_returns_all(self):
        rows = [{"index": 0, "arm_a": "A", "arm_b": "B", "text": "x"}]
        self.assertEqual(len(sample_disagreements(rows, size=50, seed=7)), 1)


class TextAlignmentTest(unittest.TestCase):
    """Segmentation is not deterministic: two identical runs of the same book
    produced 1,995 and 2,036 entries. Index-based comparison raised outright,
    so arms could not be compared at all. Alignment has to be on text."""

    def test_arms_of_different_length_align(self):
        arm_a = [{"speaker": "ARARAGI", "text": "Hello there."},
                 {"speaker": "HANEKAWA", "text": "Good morning."},
                 {"speaker": "ARARAGI", "text": "Goodbye."}]
        # arm_b split the middle line into two entries.
        arm_b = [{"speaker": "ARARAGI", "text": "Hello there."},
                 {"speaker": "HANEKAWA", "text": "Good"},
                 {"speaker": "HANEKAWA", "text": "morning."},
                 {"speaker": "SENJOGAHARA", "text": "Goodbye."}]
        rows = find_disagreements(arm_a, arm_b)
        texts = [r["text"] for r in rows]
        self.assertIn("Goodbye.", texts)
        self.assertNotIn("Hello there.", texts)

    def test_identical_arms_of_equal_length_have_no_disagreements(self):
        arm = [{"speaker": "ARARAGI", "text": "Hi"},
               {"speaker": "HACHIKUJI", "text": "Bye"}]
        self.assertEqual(find_disagreements(arm, list(arm)), [])

    def test_whitespace_differences_do_not_count_as_disagreement(self):
        arm_a = [{"speaker": "ARARAGI", "text": "Hello  there."}]
        arm_b = [{"speaker": "ARARAGI", "text": "Hello there."}]
        self.assertEqual(find_disagreements(arm_a, arm_b), [])

    def test_alignment_reports_coverage(self):
        from compare_attribution_arms import align_arms
        arm_a = [{"speaker": "A", "text": "one"}, {"speaker": "B", "text": "two"}]
        arm_b = [{"speaker": "A", "text": "one"}, {"speaker": "C", "text": "three"}]
        pairs, coverage = align_arms(arm_a, arm_b)
        self.assertEqual(len(pairs), 1)
        self.assertLess(coverage, 1.0)
        self.assertGreater(coverage, 0.0)


if __name__ == "__main__":
    unittest.main()
