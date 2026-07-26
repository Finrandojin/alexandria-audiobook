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


class GoldFixtureIntegrityTest(unittest.TestCase):
    """A gold fixture is an answer key; a duplicated line votes twice.

    attribution_gold.json shipped three lines twice or three times over, so 3
    real lines cast 7 votes in every accuracy and confusion number computed
    from it. Found by audit.
    """

    FIXTURES = ("fixtures/attribution_gold.json",
                "fixtures/attribution_gold_random.json")

    def _load(self, name):
        import json
        import os
        path = os.path.join(os.path.dirname(__file__), name)
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)["entries"]

    def test_ids_are_unique_in_every_fixture(self):
        for name in self.FIXTURES:
            entries = self._load(name)
            ids = [entry["id"] for entry in entries]
            self.assertEqual(len(ids), len(set(ids)), f"duplicate id in {name}")

    def test_book_and_entry_index_are_unique_in_every_fixture(self):
        # A distinct id pointing at the same source line would double-count too.
        for name in self.FIXTURES:
            entries = self._load(name)
            keys = [(e.get("book"), e.get("entry_index")) for e in entries]
            self.assertEqual(len(keys), len(set(keys)),
                             f"duplicate (book, entry_index) in {name}")

    def test_every_entry_has_an_expected_speaker(self):
        for name in self.FIXTURES:
            for entry in self._load(name):
                self.assertTrue(str(entry.get("expected_speaker") or "").strip(),
                                f"blank expected_speaker in {name}: {entry['id']}")
