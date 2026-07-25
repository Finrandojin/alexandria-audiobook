import json
import tempfile
import unittest

from attribution_accuracy import (load_gold, normalize_speaker, score_run,
                                  summarize)


class GoldSetTest(unittest.TestCase):
    """Every pipeline gate checks form, none checks whether the speaker is
    right. This is the only test that can catch a correctness regression."""

    def test_gold_set_loads_and_is_well_formed(self):
        gold = load_gold()
        self.assertGreater(len(gold["entries"]), 40)
        for item in gold["entries"]:
            self.assertTrue(item["expected_speaker"])
            self.assertEqual(item["expected_speaker"],
                             item["expected_speaker"].upper())
            self.assertIsInstance(item["entry_index"], int)
            self.assertTrue(item["line"])

    def test_disputed_entries_are_withheld_by_default(self):
        gold = load_gold()
        disputed = [e for e in gold["entries"] if e.get("disputed")]
        self.assertTrue(disputed, "the row-1 dispute should be recorded")
        for item in disputed:
            self.assertTrue(item.get("dispute_note"),
                            "a disputed entry must say why")


class ScoringTest(unittest.TestCase):

    GOLD = {"entries": [
        {"id": "a", "entry_index": 0, "line": "Are you okay, Rudi?",
         "expected_speaker": "SYLPHY"},
        {"id": "b", "entry_index": 1, "line": "Sorry.",
         "expected_speaker": "RUDI"},
    ]}

    def test_correct_and_incorrect_are_counted(self):
        named = [{"speaker": "SYLPHY", "text": "Are you okay, Rudi?"},
                 {"speaker": "ROXY", "text": "Sorry."}]
        stats = summarize(score_run(named, self.GOLD))
        self.assertEqual(stats["aligned"], 2)
        self.assertEqual(stats["correct"], 1)
        self.assertAlmostEqual(stats["accuracy"], 0.5)

    def test_misaligned_run_is_reported_not_scored_wrong(self):
        # A run that segmented differently must not look like a wrong answer.
        named = [{"speaker": "SYLPHY", "text": "Something else entirely"},
                 {"speaker": "RUDI", "text": "Sorry."}]
        stats = summarize(score_run(named, self.GOLD))
        self.assertEqual(stats["aligned"], 1)
        self.assertEqual(stats["correct"], 1)

    def test_missing_entry_is_not_a_crash(self):
        stats = summarize(score_run([], self.GOLD))
        self.assertEqual(stats["aligned"], 0)
        self.assertEqual(stats["accuracy"], 0.0)

    def test_confusion_is_recorded(self):
        named = [{"speaker": "ROXY", "text": "Are you okay, Rudi?"},
                 {"speaker": "ROXY", "text": "Sorry."}]
        stats = summarize(score_run(named, self.GOLD))
        self.assertEqual(stats["confusion"][("SYLPHY", "ROXY")], 1)
        self.assertEqual(stats["missed"]["RUDI"], 1)

    def test_case_and_whitespace_do_not_matter(self):
        named = [{"speaker": " sylphy ", "text": "Are you okay, Rudi?"},
                 {"speaker": "RUDI", "text": "Sorry."}]
        stats = summarize(score_run(named, self.GOLD))
        self.assertEqual(stats["correct"], 2)

    def test_empty_speaker_is_wrong_not_correct(self):
        named = [{"speaker": "", "text": "Are you okay, Rudi?"},
                 {"speaker": None, "text": "Sorry."}]
        stats = summarize(score_run(named, self.GOLD))
        self.assertEqual(stats["correct"], 0)

    def test_normalize(self):
        self.assertEqual(normalize_speaker("  roxy  "), "ROXY")
        self.assertEqual(normalize_speaker(None), "")


if __name__ == "__main__":
    unittest.main()
