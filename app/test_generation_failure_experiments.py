import unittest

from experiments.adaptive_split_floor import get_floor_splitter
from experiments.targeted_missing_repair import get_missing_passage


class GenerationFailureExperimentTests(unittest.TestCase):
    def test_floor_splitter_binds_each_arm_independently(self):
        calls = []

        def splitter(chunk, minimum_chars):
            calls.append((chunk, minimum_chars))
            return []

        get_floor_splitter(splitter, 800)("old")
        get_floor_splitter(splitter, 400)("new")

        self.assertEqual([("old", 800), ("new", 400)], calls)

    def test_missing_passage_preserves_exact_omitted_sentence(self):
        source = "First sentence stays. Second sentence was omitted. Third sentence stays."
        entries = [{"speaker": "NARRATOR",
                    "text": "First sentence stays. Third sentence stays.",
                    "instruct": ""}]

        self.assertEqual("Second sentence was omitted.",
                         get_missing_passage(source, entries))

    def test_missing_passage_is_empty_for_complete_conversion(self):
        source = "First sentence. Second sentence."
        entries = [{"speaker": "NARRATOR", "text": source, "instruct": ""}]

        self.assertEqual("", get_missing_passage(source, entries))


if __name__ == "__main__":
    unittest.main()
